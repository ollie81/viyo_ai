"""Creator Studio end to end: API + worker + render against the real SQL
(local Postgres), local files and a fake AI that costs nothing."""
import uuid

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

import creator_studio as cs
import studio_ai
import studio_worker as sw
from conftest import ROOT, connect, make_user
from studio_store import LocalFiles, PgStore


@pytest.fixture(scope="module")
def studio_schema(migrated):
    conn = connect()
    conn.execute((ROOT / "creator_studio_schema.sql").read_text())
    conn.execute("alter table public.profiles add column if not exists username text")
    conn.close()
    return migrated


@pytest.fixture
def env(studio_schema, db, tmp_path, monkeypatch):
    db.execute("truncate public.studio_usage, public.studio_jobs, public.studio_credit_ledger, "
               "public.studio_credit_hold_allocations, public.studio_credit_holds, public.studio_credit_lots, "
               "public.studio_credit_purchases, public.studio_renders, public.studio_assets, public.studio_scenes, "
               "public.studio_characters, public.studio_projects cascade")
    store, files, ai = PgStore(db), LocalFiles(str(tmp_path)), studio_ai.FakeStudioAI()
    monkeypatch.setattr(cs, "_store", store)
    monkeypatch.setattr(cs, "_files", files)
    monkeypatch.setattr(cs, "_job_log", cs.defaultdict(cs.deque))
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    app = FastAPI()
    app.include_router(cs.router)

    async def fake_creator(authorization: str = Header(None)):
        if not authorization:
            raise HTTPException(status_code=401, detail="Not signed in")
        if authorization == "Bearer guest":
            raise HTTPException(status_code=403, detail="Create an account to use Viyo Studio.")
        return authorization.removeprefix("Bearer ")

    app.dependency_overrides[cs.creator] = fake_creator
    gen = sw.StudioWorker(store, files, ai, "generation")
    ren = sw.StudioWorker(store, files, ai, "render")
    return TestClient(app), db, store, files, ai, gen, ren


def H(u):
    return {"Authorization": f"Bearer {u}"}


def grant(client, user, credits, kind="grant", ref=None):
    r = client.post("/api/v1/studio/admin/credits/grant", headers={"X-Admin-Key": "admin-secret"},
                    json={"user": str(user), "credits": credits, "kind": kind, "reference": ref or uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    return r.json()


def new_project(client, user, **over):
    body = {"title": "Night Shift", "logline": "A nurse discovers her patient is her missing brother.",
            "kind": "episode", "aspect_ratio": "9:16", "genre": "Drama", "target_seconds": 60, "accept_terms": True}
    body.update(over)
    r = client.post("/api/v1/studio/projects", headers=H(user), json=body)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def run(worker, times=50):
    n = 0
    while n < times and worker.run_once():
        n += 1
    return n


def start(client, user, pid, kind, targets=None, rid=None):
    return client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(user),
                       json={"kind": kind, "target_ids": targets or [], "request_id": rid or uuid.uuid4().hex})


def balance(client, user):
    return client.get("/api/v1/studio/me", headers=H(user)).json()["balance"]


def test_full_production_from_idea_to_downloaded_film(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = new_project(client, u)
    # Not free: nothing runs without credits.
    r = start(client, u, pid, "script")
    assert r.status_code == 402 and r.json()["detail"]["message"] == "Not enough credits."
    grant(client, u, 2000, kind="purchase")

    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "script"}).json()
    assert est["total_credits"] >= 1 and est["items"][0]["blocked"] is None
    assert start(client, u, pid, "script").status_code == 200
    assert run(gen) == 1
    proj = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()
    assert "INT. ROOFTOP" in proj["project"]["script"]
    assert proj["jobs"][0]["status"] == "succeeded" and proj["jobs"][0]["charged_credits"] >= 1

    assert start(client, u, pid, "breakdown").status_code == 200
    run(gen)
    proj = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()
    assert [c["name"] for c in proj["characters"]] == ["Maya", "Daniel"]
    assert all(c["voice"] in studio_ai.VOICE_NAMES for c in proj["characters"])
    assert len(proj["scenes"]) == 2

    assert start(client, u, pid, "portrait").json()["job_ids"]
    run(gen)
    assert start(client, u, pid, "image").status_code == 200
    run(gen)
    assert ("image", "9:16", 1) in ai.calls, "scene images use the approved character portrait"
    assert start(client, u, pid, "voice").status_code == 200
    run(gen)
    proj = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()
    assert all(s["has_image"] and s["has_audio"] and s["image_url"] for s in proj["scenes"])

    assert start(client, u, pid, "render").status_code == 200
    assert run(gen) == 0, "renders run in their own lane"
    assert run(ren) == 1
    proj = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()
    render = proj["renders"][0]
    assert (render["width"], render["height"], render["version"]) == (1080, 1920, 1)
    dl = client.post(f"/api/v1/studio/renders/{render['id']}/download", headers=H(u)).json()
    assert dl["file_name"] == "night-shift-v1.mp4" and "exp=" in dl["url"]

    b = balance(client, u)
    charged = sum(j["charged_credits"] for j in proj["jobs"])
    assert b["held"] == 0 and b["available"] == 2000 - charged
    usage = db.execute("select count(*), sum(credits_charged) from public.studio_usage where status='charged'").fetchone()
    assert usage[0] == len(proj["jobs"]) and usage[1] == charged, "every operation's cost is recorded"
    # Watch coins untouched
    assert db.execute("select points_balance from public.profiles where id=%s", (u,)).fetchone()[0] == 100
    assert db.execute("select count(*) from public.transactions").fetchone()[0] == 0


def test_landscape_render_and_a_new_render_keeps_the_old_version(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 1000)
    pid = new_project(client, u, aspect_ratio="16:9", kind="film")
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u),
                    json={"heading": "EXT. HILL - DAWN", "visual": "A lone rider on a hill", "duration_seconds": 3}).json()
    start(client, u, pid, "image", [s["id"]])
    run(gen)
    for _ in range(2):
        start(client, u, pid, "render")
        run(ren)
    renders = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["renders"]
    assert [r["version"] for r in renders] == [2, 1]
    assert all((r["width"], r["height"]) == (1920, 1080) for r in renders)
    paths = [row[0] for row in db.execute("select path from public.studio_renders order by version").fetchall()]
    assert all(files.exists(p) for p in paths), "older versions are never overwritten"


def test_regenerating_an_image_keeps_the_previous_file(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 500)
    pid = new_project(client, u)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "A rainy street"}).json()
    for _ in range(2):
        start(client, u, pid, "image", [s["id"]])
        run(gen)
    assets = db.execute("select path from public.studio_assets where kind='image'").fetchall()
    assert len(assets) == 2 and all(files.exists(a[0]) for a in assets)


def test_a_double_tap_creates_one_job_and_one_charge(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 500)
    pid = new_project(client, u)
    a = start(client, u, pid, "script", rid="same-tap-123").json()
    b = start(client, u, pid, "script", rid="same-tap-123").json()
    assert a["job_ids"] == b["job_ids"]
    assert db.execute("select count(*) from public.studio_jobs").fetchone()[0] == 1


def test_creators_cannot_see_or_touch_each_others_work(env):
    client, db, store, files, ai, gen, ren = env
    a, b = make_user(db), make_user(db)
    grant(client, a, 500)
    grant(client, b, 500)
    pid = new_project(client, a)
    scene = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(a), json={"visual": "x"}).json()
    char = client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(a), json={"name": "Maya"}).json()
    job = start(client, a, pid, "script").json()["job_ids"][0]
    checks = [
        client.get(f"/api/v1/studio/projects/{pid}", headers=H(b)),
        client.patch(f"/api/v1/studio/projects/{pid}", headers=H(b), json={"title": "stolen"}),
        client.patch(f"/api/v1/studio/scenes/{scene['id']}", headers=H(b), json={"visual": "y"}),
        client.delete(f"/api/v1/studio/scenes/{scene['id']}", headers=H(b)),
        client.patch(f"/api/v1/studio/characters/{char['id']}", headers=H(b), json={"name": "z"}),
        client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(b), json={"kind": "script"}),
        start(client, b, pid, "script"),
        client.post(f"/api/v1/studio/jobs/{job}/cancel", headers=H(b)),
        client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(b)),
    ]
    assert [c.status_code for c in checks] == [404] * len(checks)
    assert client.get("/api/v1/studio/projects", headers=H(b)).json()["projects"] == []
    assert balance(client, b)["available"] == 500
    assert client.get(f"/api/v1/studio/projects/{pid}", headers=H(a)).json()["project"]["title"] == "Night Shift"


def test_render_download_is_owner_only(env):
    client, db, store, files, ai, gen, ren = env
    a, b = make_user(db), make_user(db)
    grant(client, a, 500)
    pid = new_project(client, a)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(a), json={"visual": "x", "duration_seconds": 2}).json()
    start(client, a, pid, "image", [s["id"]])
    run(gen)
    start(client, a, pid, "render")
    run(ren)
    rid = client.get(f"/api/v1/studio/projects/{pid}", headers=H(a)).json()["renders"][0]["id"]
    assert client.post(f"/api/v1/studio/renders/{rid}/download", headers=H(b)).status_code == 404


def test_guests_and_signed_out_users_are_refused(env):
    client = env[0]
    assert client.get("/api/v1/studio/projects").status_code == 401
    assert client.get("/api/v1/studio/projects", headers=H("guest")).status_code == 403


def test_terms_must_be_accepted(env):
    client, db = env[0], env[1]
    u = make_user(db)
    r = client.post("/api/v1/studio/projects", headers=H(u), json={"title": "x"})
    assert r.status_code == 400


def test_provider_failure_returns_the_credits(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "x"}).json()
    ai.fail["image"] = studio_ai.ProviderError("boom")
    start(client, u, pid, "image", [s["id"]])
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed" and "credits were returned" in job["error"]
    assert balance(client, u) == {"available": 300, "purchased": 0, "promotional": 300, "held": 0}


def test_busy_provider_is_retried_and_charged_only_once_it_works(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "x"}).json()
    ai.fail["image"] = [studio_ai.ProviderBusy("429")]
    start(client, u, pid, "image", [s["id"]])
    assert run(gen) == 2
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded"
    assert balance(client, u)["available"] == 300 - job["charged_credits"]


def test_blocked_content_is_not_charged(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "x"}).json()
    ai.fail["image"] = studio_ai.PolicyBlocked("No image came back for this prompt. Try rewording the scene.")
    start(client, u, pid, "image", [s["id"]])
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed" and "rewording" in job["error"]
    assert balance(client, u)["available"] == 300


def test_cancel_before_start_refunds(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    jid = start(client, u, pid, "script").json()["job_ids"][0]
    assert balance(client, u)["held"] > 0
    assert client.post(f"/api/v1/studio/jobs/{jid}/cancel", headers=H(u)).status_code == 200
    assert balance(client, u) == {"available": 300, "purchased": 0, "promotional": 300, "held": 0}
    assert run(gen) == 0


def test_steps_that_are_not_ready_are_refused_before_charging(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u, logline="")
    r = start(client, u, pid, "script")
    assert r.status_code == 400 and "idea" in r.json()["detail"]["message"]
    assert start(client, u, pid, "breakdown").status_code == 400
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "x"}).json()
    r = start(client, u, pid, "render")
    assert r.status_code == 400 and "no picture" in r.json()["detail"]["message"]
    assert start(client, u, pid, "voice", [s["id"]]).status_code == 400
    assert balance(client, u)["available"] == 300


def test_editing_lines_clears_the_old_voice_track(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u),
                    json={"visual": "x", "dialogue": [{"character": "Maya", "text": "Hello there"}]}).json()
    start(client, u, pid, "voice", [s["id"]])
    run(gen)
    s2 = client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u),
                      json={"dialogue": [{"character": "Maya", "text": "Goodbye"}]}).json()
    assert s2["has_audio"] is False
    assert db.execute("select count(*) from public.studio_assets where kind='audio'").fetchone()[0] == 1


def test_scene_order_and_delete(env):
    client, db = env[0], env[1]
    u = make_user(db)
    pid = new_project(client, u)
    ids = [client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"heading": h}).json()["id"]
           for h in ("A", "B", "C")]
    assert client.post(f"/api/v1/studio/projects/{pid}/scenes/order", headers=H(u),
                       json={"scene_ids": [ids[2], ids[0], ids[1]]}).status_code == 200
    client.delete(f"/api/v1/studio/scenes/{ids[0]}", headers=H(u))
    scenes = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"]
    assert [(s["heading"], s["position"]) for s in scenes] == [("C", 0), ("B", 1)]


def test_admin_grant_needs_the_key_and_purchases_are_not_double_credited(env):
    client, db = env[0], env[1]
    u = make_user(db)
    bad = client.post("/api/v1/studio/admin/credits/grant", headers={"X-Admin-Key": "nope"},
                      json={"user": str(u), "credits": 100})
    assert bad.status_code == 403
    grant(client, u, 500, kind="purchase", ref="momo-tx-1")
    again = grant(client, u, 500, kind="purchase", ref="momo-tx-1")
    assert again["duplicate"] and balance(client, u)["available"] == 500


def test_archived_projects_disappear_but_keep_their_files(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "x"}).json()
    start(client, u, pid, "image", [s["id"]])
    run(gen)
    client.post(f"/api/v1/studio/projects/{pid}/archive", headers=H(u))
    assert client.get("/api/v1/studio/projects", headers=H(u)).json()["projects"] == []
    assert client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).status_code == 404
    path = db.execute("select path from public.studio_assets").fetchone()[0]
    assert files.exists(path)


def test_pricing_formula():
    import studio_pricing as p
    assert p.credits_for(0.75, margin=0.25) == 100
    assert p.credits_for(0.0001, margin=0.25) == 1
    assert p.credits_for(1.0, margin=0.0) == 100
    assert p.pack_price(500, 0.065, 0.50) == round((5.0 + 0.5) / 0.935, 2)
    assert p.image_cost().provider_cost_usd == pytest.approx(0.067)
    v = p.video_cost(8.0)
    assert v.provider_cost_usd == pytest.approx(8 * 5792 * 17.5 / 1e6)
    import datetime
    assert p.prices(datetime.date(2027, 1, 2))["text_input_per_m"] == 1.50
    assert p.prices(datetime.date(2026, 11, 5))["text_input_per_m"] == 0.75
