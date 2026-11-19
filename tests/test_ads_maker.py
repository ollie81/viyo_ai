"""Ads Maker for creators: sign-in required, a creator only reaches their own campaigns/files/jobs, admin
routes stay admin-only, and paid steps hold Studio credits and keep only the real cost.

The HTTP tests go through the real sign-in check (main.get_current_user_id_no_guest; locally it reads the
token without a signature, the same path as other route tests) and an in-memory database; the credit SQL is
tested against a real Postgres."""
import io
import uuid

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import ads_maker
import ads_studio as core
from conftest import ROOT, connect, make_user, pg_rpc
from test_ads_quality import _Q, _Res


def token(uid, anonymous=False):
    return {"Authorization": "Bearer " + jwt.encode({"sub": uid, "is_anonymous": anonymous}, "x", algorithm="HS256")}


class Store:
    def __init__(self):
        self.files = {}

    def from_(self, bucket):
        st = self

        class B:
            def upload(self, path, data, opts=None):
                st.files[f"{bucket}/{path}"] = data

            def create_signed_url(self, path, seconds):
                return {"signedURL": f"https://signed.test/{bucket}/{path}?exp={seconds}"} if f"{bucket}/{path}" in st.files else {}

            def list(self, folder, opts=None):
                pre = f"{bucket}/{folder}/"
                return [{"name": k[len(pre):]} for k in st.files if k.startswith(pre)]

            def remove(self, paths):
                for p in paths:
                    st.files.pop(f"{bucket}/{p}", None)
        return B()


class FakeDB:
    def __init__(self):
        self.db = {"ad_campaigns": [], "ad_assets": [], "ad_hook_candidates": [], "ad_generation_jobs": []}
        self.storage = Store()
        self.credits = {}
        self.holds = {}
        self.calls = []

    def table(self, name):
        q = _Q(self.db, name)
        orig_select = q.select

        def select(*a, **k):
            return orig_select()
        q.select = select
        return q

    def rpc(self, name, params):
        self.calls.append(name)
        db = self

        class Call:
            def execute(self_inner):
                u = params.get("p_user_id")
                if name == "studio_credit_balance":
                    return _Res({"available": db.credits.get(u, 0)})
                if name == "ads_credit_hold":
                    need = params["p_credits"]
                    if db.credits.get(u, 0) < need:
                        return _Res({"ok": False, "error": "insufficient_credits", "balance": {"available": db.credits.get(u, 0)}})
                    db.credits[u] -= need
                    hid = str(uuid.uuid4())
                    db.holds[hid] = {"user": u, "held": need, "used": None}
                    return _Res({"ok": True, "hold_id": hid})
                if name == "ads_credit_settle":
                    h = db.holds[params["p_hold_id"]]
                    if h["used"] is None:
                        h["used"] = min(params["p_used"], h["held"])
                        db.credits[h["user"]] += h["held"] - h["used"]
                    return _Res(h["used"])
                raise AssertionError(name)
        return Call()


ALICE, BOB = str(uuid.uuid4()), str(uuid.uuid4())


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(core, "supabase_admin", db)
    monkeypatch.setattr(core, "_require_configured", lambda: None)
    monkeypatch.setattr(core, "_load_campaign_assets",
                        lambda cid: [a for a in db.db["ad_assets"] if a["campaign_id"] == cid])
    monkeypatch.setenv("STUDIO_CREATORS_OPEN", "true")
    app = FastAPI()
    app.include_router(ads_maker.router)
    app.include_router(core.router)
    return {"db": db, "c": TestClient(app)}


def new_campaign(env, uid, **over):
    body = {"promote_target": "other_app", "target_name": "Ollie", "destination_link": "https://ourollie.space",
            "cta_text": "Try Ollie free", **over}
    r = env["c"].post("/api/v1/ads-maker/campaigns", json=body, headers=token(uid))
    assert r.status_code == 200, r.text
    return r.json()


def png(w=200, h=200):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (90, 80, 230)).save(b, "PNG")
    return b.getvalue()


# ---------------------------------------------------------------- ownership

def test_signed_in_creator_creates_their_own_campaign(env):
    c = new_campaign(env, ALICE)
    row = env["db"].db["ad_campaigns"][0]
    assert row["user_id"] == ALICE and row["created_via"] == "creator" and c["destination_link"] == "https://ourollie.space"


def test_user_id_in_the_body_is_ignored(env):
    r = env["c"].post("/api/v1/ads-maker/campaigns", json={"target_name": "X", "user_id": BOB}, headers=token(ALICE))
    assert r.status_code == 200 and env["db"].db["ad_campaigns"][0]["user_id"] == ALICE


def test_no_sign_in_or_guest_is_refused(env):
    assert env["c"].get("/api/v1/ads-maker/campaigns").status_code == 401
    assert env["c"].get("/api/v1/ads-maker/campaigns", headers=token(ALICE, anonymous=True)).status_code == 403


def test_creators_only_list_their_own(env):
    new_campaign(env, ALICE)
    new_campaign(env, BOB, target_name="Bob's Bakery")
    names = [c["target_name"] for c in env["c"].get("/api/v1/ads-maker/campaigns", headers=token(ALICE)).json()]
    assert names == ["Ollie"]


@pytest.mark.parametrize("method,path", [
    ("get", ""), ("put", ""), ("delete", ""), ("get", "/assets"), ("get", "/hooks"), ("post", "/hooks"),
    ("post", "/hooks/auto/select"), ("post", "/script"), ("put", "/script"), ("post", "/generate"), ("get", "/job"),
    ("get", "/estimate")])
def test_another_creators_campaign_is_not_found(env, method, path):
    cid = new_campaign(env, ALICE)["id"]
    body = {"scenes": []} if path == "/script" and method == "put" else {}
    r = getattr(env["c"], method)(f"/api/v1/ads-maker/campaigns/{cid}{path}", headers=token(BOB),
                                  **({"json": body} if method in ("put", "post") else {}))
    assert r.status_code == 404, (method, path, r.status_code)
    assert env["db"].db["ad_campaigns"][0]["target_name"] == "Ollie"


def test_malformed_ids_are_not_found(env):
    assert env["c"].get("/api/v1/ads-maker/campaigns/not-a-uuid", headers=token(ALICE)).status_code == 404


def test_creators_cannot_advertise_as_viyo(env):
    r = env["c"].post("/api/v1/ads-maker/campaigns", json={"promote_target": "viyo"}, headers=token(ALICE))
    assert r.status_code == 422


def test_invalid_website_rejected_on_create(env):
    r = env["c"].post("/api/v1/ads-maker/campaigns", json={"destination_link": "ourollie"}, headers=token(ALICE))
    assert r.status_code == 400


def test_admin_routes_still_need_the_admin_key(env, monkeypatch):
    import studio
    monkeypatch.setattr(studio, "ADMIN_API_KEY", "secret")
    for path in ("/api/v1/admin/ads-studio/campaigns", "/api/v1/admin/ads-studio/viyo-asset-library"):
        assert env["c"].get(path, headers=token(ALICE)).status_code == 403          # a creator's sign-in isn't enough
        assert env["c"].get(path, headers={"X-Admin-Key": "wrong"}).status_code == 403
    assert env["c"].get("/api/v1/admin/ads-studio/campaigns", headers={"X-Admin-Key": "secret"}).status_code == 200


# ---------------------------------------------------------------- files

def test_assets_are_private_and_signed(env):
    cid = new_campaign(env, ALICE)["id"]
    r = env["c"].post(f"/api/v1/ads-maker/campaigns/{cid}/assets?asset_type=logo",
                      files={"file": ("logo.png", png(), "image/png")}, headers=token(ALICE))
    assert r.status_code == 200, r.text
    stored = env["db"].db["ad_assets"][0]["url"]
    assert stored.startswith("private:creator-studio/ads/" + ALICE + "/" + cid + "/")
    assert r.json()["url"].startswith("https://signed.test/creator-studio/ads/" + ALICE)


def test_another_creator_cannot_list_or_delete_assets(env):
    a = new_campaign(env, ALICE)["id"]
    b = new_campaign(env, BOB, target_name="Bob")["id"]
    up = env["c"].post(f"/api/v1/ads-maker/campaigns/{a}/assets", files={"file": ("s.png", png(), "image/png")},
                       headers=token(ALICE)).json()
    assert env["c"].get(f"/api/v1/ads-maker/campaigns/{a}/assets", headers=token(BOB)).status_code == 404
    # Bob's own campaign id with Alice's asset id: still not found.
    assert env["c"].delete(f"/api/v1/ads-maker/campaigns/{b}/assets/{up['id']}", headers=token(BOB)).status_code == 404
    assert len(env["db"].db["ad_assets"]) == 1


def test_non_images_are_rejected(env):
    cid = new_campaign(env, ALICE)["id"]
    r = env["c"].post(f"/api/v1/ads-maker/campaigns/{cid}/assets", files={"file": ("x.png", b"<html>", "image/png")},
                      headers=token(ALICE))
    assert r.status_code == 400 and not env["db"].db["ad_assets"]


# ---------------------------------------------------------------- credits

def test_paid_step_needs_enough_credits(env, monkeypatch):
    ran = []
    monkeypatch.setattr(core, "generate_hooks", lambda cid: ran.append(cid))
    cid = new_campaign(env, ALICE)["id"]
    r = env["c"].post(f"/api/v1/ads-maker/campaigns/{cid}/hooks", headers=token(ALICE))
    assert r.status_code == 402 and r.json()["detail"]["error"] == "insufficient_credits" and not ran


def test_only_the_real_cost_is_kept(env, monkeypatch):
    async def hooks(cid):
        core._cap(1)
        core._spent(cid, "ad_hooks", 1)      # the provider call cost 1 cent
        return {"hooks": []}
    monkeypatch.setattr(core, "generate_hooks", hooks)
    monkeypatch.setattr(core, "_spent", lambda *a: core._creator_run.get().__setitem__(
        "spent_cents", core._creator_run.get()["spent_cents"] + a[2]))
    env["db"].credits[ALICE] = 100
    cid = new_campaign(env, ALICE)["id"]
    assert env["c"].post(f"/api/v1/ads-maker/campaigns/{cid}/hooks", headers=token(ALICE)).status_code == 200
    hold = next(iter(env["db"].holds.values()))
    assert hold["used"] == ads_maker._used_credits(1, hold["held"]) and env["db"].credits[ALICE] == 100 - hold["used"]


def test_a_failed_step_keeps_nothing(env, monkeypatch):
    async def boom(cid):
        raise core.HTTPException(status_code=502, detail="provider down")
    monkeypatch.setattr(core, "generate_hooks", boom)
    env["db"].credits[ALICE] = 100
    cid = new_campaign(env, ALICE)["id"]
    assert env["c"].post(f"/api/v1/ads-maker/campaigns/{cid}/hooks", headers=token(ALICE)).status_code == 502
    assert env["db"].credits[ALICE] == 100


def test_a_run_cannot_spend_more_than_it_held(env, monkeypatch):
    async def greedy(cid):
        for _ in range(1000):
            core._cap(7)                      # e.g. an image
            core._creator_run.get()["spent_cents"] += 7
    monkeypatch.setattr(core, "generate_script", greedy)
    env["db"].credits[ALICE] = 1000
    cid = new_campaign(env, ALICE)["id"]
    r = env["c"].post(f"/api/v1/ads-maker/campaigns/{cid}/script", headers=token(ALICE))
    assert r.status_code == 402 and env["db"].credits[ALICE] == 1000, "stopped, and a failed step keeps nothing"


def test_creator_runs_do_not_use_the_admin_daily_cap(env, monkeypatch):
    def cap(_):
        raise AssertionError("admin cap must not apply to a creator-paid run")
    monkeypatch.setattr(core, "_check_daily_cap", cap)
    token_ = core._creator_run.set({"spent_cents": 0, "budget_cents": 100})
    try:
        core._cap(7)
    finally:
        core._creator_run.reset(token_)
    with pytest.raises(AssertionError):
        core._cap(7)                           # outside a creator run the admin cap applies


def test_invalid_website_stops_generation_before_any_credits_are_held(env):
    cid = new_campaign(env, ALICE)["id"]
    env["db"].db["ad_campaigns"][0].update(destination_link="not a website", script=[{"seconds": 3}])
    env["db"].credits[ALICE] = 1000
    r = env["c"].post(f"/api/v1/ads-maker/campaigns/{cid}/generate", headers=token(ALICE))
    assert r.status_code == 400 and "ads_credit_hold" not in env["db"].calls


@pytest.mark.parametrize("outcome,keeps", [("ready", True), ("needs_correction", False), ("failed", False)])
def test_video_keeps_credits_only_when_the_ad_passes(env, monkeypatch, outcome, keeps):
    def run(cid, job_id):
        core._cap(7)
        core._creator_run.get()["spent_cents"] += 30
        env["db"].db["ad_campaigns"][0]["status"] = outcome
    monkeypatch.setattr(core, "_run_ad_generation", run)
    cid = new_campaign(env, ALICE)["id"]
    env["db"].db["ad_campaigns"][0]["script"] = [{"seconds": 3}]
    env["db"].credits[ALICE] = 1000
    r = env["c"].post(f"/api/v1/ads-maker/campaigns/{cid}/generate", headers=token(ALICE))
    assert r.status_code == 200, r.text
    hold = next(iter(env["db"].holds.values()))
    assert (hold["used"] > 0) == keeps
    assert env["db"].credits[ALICE] == 1000 - hold["used"]


def test_estimate_covers_what_the_pipeline_counts(env):
    c = {"id": "x", "duration_seconds": 30, "use_veo": False, "target_name": "Ollie"}
    est = ads_maker.estimate(c, assets=[])
    worst = (core._scene_count_for_duration(30) - 1 + core.MAX_IMAGE_REGENERATIONS) * core.GEMINI_IMAGE_COST_USD_CENTS
    assert ads_maker._budget_cents(est["video"]) >= worst


# ---------------------------------------------------------------- credit SQL on Postgres

@pytest.fixture
def credits_db(migrated, db):
    conn = connect()
    conn.execute((ROOT / "creator_studio_schema.sql").read_text())
    conn.execute((ROOT / "ads_studio_schema.sql").read_text())
    conn.execute((ROOT / "ads_studio_quality.sql").read_text())
    conn.execute((ROOT / "ads_studio_quality.sql").read_text())        # safe to run twice
    conn.close()
    db.execute("truncate public.studio_credit_ledger, public.studio_credit_hold_allocations, public.studio_credit_holds, "
               "public.studio_credit_lots, public.studio_credit_purchases cascade")
    return db


def test_credit_hold_and_settle_on_postgres(credits_db):
    rpc = pg_rpc(credits_db)
    u = make_user(credits_db)
    rpc("studio_add_credits", {"p_user_id": u, "p_credits": 100, "p_source": "grant", "p_key": f"g-{u}"})
    r = rpc("ads_credit_hold", {"p_user_id": u, "p_credits": 40})
    assert r["ok"]
    assert rpc("studio_credit_balance", {"p_user_id": u})["available"] == 60
    assert rpc("ads_credit_settle", {"p_hold_id": r["hold_id"], "p_used": 12}) == 12
    assert rpc("studio_credit_balance", {"p_user_id": u})["available"] == 88
    assert rpc("ads_credit_settle", {"p_hold_id": r["hold_id"], "p_used": 40}) == 0, "settling twice does nothing"
    no = rpc("ads_credit_hold", {"p_user_id": u, "p_credits": 500})
    assert no == {**no, "ok": False, "error": "insufficient_credits"}
    assert rpc("studio_credit_balance", {"p_user_id": u})["available"] == 88, "a refused hold takes nothing"
