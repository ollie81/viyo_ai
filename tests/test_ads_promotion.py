"""Paid promotion: off until enabled; transparent quote; only your own public ad; the budget is taken once,
never exceeded, paced per day; impressions/clicks counted once per viewer per day (never the owner's);
unused coins refunded exactly once when it ends, is stopped or the ad stops being public."""
import datetime as dt
import uuid

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import ads_promotion as promo
import ads_studio as core
from test_ads_maker import ALICE, BOB, FakeDB, token

CAROL = str(uuid.uuid4())


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    for t in ("posts", "ad_promotions", "ad_promotion_events"):
        db.db[t] = []
    monkeypatch.setattr(core, "supabase_admin", db)
    monkeypatch.setenv("STUDIO_CREATORS_OPEN", "true")
    monkeypatch.setenv("PROMOTIONS_ENABLED", "true")
    wallet = {ALICE: 1000, BOB: 50, CAROL: 0}
    ledger = []

    def debit(admin, uid, amount, type_, desc):
        if wallet.get(uid, 0) < amount:
            raise HTTPException(status_code=402, detail={"error": "insufficient_coins"})
        wallet[uid] -= amount
        ledger.append(("debit", uid, amount, type_))

    def credit(admin, uid, amount, type_, desc, external_ref=None):
        if external_ref and any(e[-1] == external_ref for e in ledger if len(e) == 5):
            return False
        wallet[uid] = wallet.get(uid, 0) + amount
        ledger.append(("credit", uid, amount, type_, external_ref))
        return True
    monkeypatch.setattr(promo, "debit_coins", debit)
    monkeypatch.setattr(promo, "credit_coins", credit)
    clock = {"now": dt.datetime(2026, 11, 20, 9, 0, tzinfo=dt.timezone.utc)}
    monkeypatch.setattr(promo, "_now", lambda: clock["now"])
    app = FastAPI()
    app.include_router(promo.router)
    return {"db": db, "c": TestClient(app), "wallet": wallet, "ledger": ledger, "clock": clock}


def ad_post(env, uid=ALICE, **kw):
    p = {"id": str(uuid.uuid4()), "user_id": uid, "post_kind": "ad", "is_private": False, "is_archived": False,
         "caption": "Kivu", **kw}
    env["db"].db["posts"].append(p)
    return p


def start(env, post, uid=ALICE, budget=500, days=5, **kw):
    return env["c"].post("/api/v1/ads-maker/promotions", json={"post_id": post["id"], "budget_coins": budget,
                                                               "days": days, **kw}, headers=token(uid))


def view(env, promo_id, viewer, kind="impression"):
    return env["c"].post(f"/api/v1/promotions/{promo_id}/{kind}", headers=token(viewer)).json()


def test_quote_shows_the_full_price_first(env):
    q = env["c"].get("/api/v1/ads-maker/promotions/quote?budget=500&days=5", headers=token(ALICE)).json()
    assert q["coins_per_1000_impressions"] == promo.COINS_PER_1000 and q["max_impressions"] == 10000
    assert q["per_day_coins"] == 100 and any("not guaranteed" in t for t in q["terms"])
    assert env["c"].get("/api/v1/ads-maker/promotions/quote?budget=5&days=5", headers=token(ALICE)).status_code == 400
    assert env["c"].get("/api/v1/ads-maker/promotions/quote?budget=500&days=30", headers=token(ALICE)).status_code == 400


def test_off_until_enabled(env, monkeypatch):
    monkeypatch.setenv("PROMOTIONS_ENABLED", "false")
    assert start(env, ad_post(env)).status_code == 503
    assert env["wallet"][ALICE] == 1000
    assert env["c"].get("/api/v1/promotions/serve", headers=token(BOB)).json() == {"promotions": []}


def test_start_takes_the_budget_once(env):
    r = start(env, ad_post(env))
    assert r.status_code == 200, r.text
    assert env["wallet"][ALICE] == 500 and r.json()["status"] == "active" and r.json()["coins_spent"] == 0


@pytest.mark.parametrize("case", ["other_owner", "not_ad", "private", "archived"])
def test_only_your_own_public_ad(env, case):
    post = {"other_owner": lambda: ad_post(env, uid=BOB), "not_ad": lambda: ad_post(env, post_kind="post"),
            "private": lambda: ad_post(env, is_private=True), "archived": lambda: ad_post(env, is_archived=True)}[case]()
    assert start(env, post).status_code in (400, 404)
    assert env["wallet"][ALICE] == 1000 and not env["db"].db["ad_promotions"]


def test_not_enough_coins(env):
    assert start(env, ad_post(env, uid=BOB), uid=BOB, budget=100).status_code == 402
    assert not env["db"].db["ad_promotions"]


def test_one_promotion_per_ad(env):
    p = ad_post(env)
    start(env, p)
    assert start(env, p).status_code == 400 and env["wallet"][ALICE] == 500


def test_impressions_counted_once_per_viewer_per_day_and_never_the_owner(env):
    pid = start(env, ad_post(env)).json()["id"]
    assert view(env, pid, BOB) == {"counted": True}
    assert view(env, pid, BOB) == {"counted": False}
    assert view(env, pid, ALICE) == {"counted": False}
    assert view(env, pid, CAROL) == {"counted": True}
    row = env["db"].db["ad_promotions"][0]
    assert row["impressions"] == 2 and row["spent_millicoins"] == 2 * promo.COINS_PER_1000
    env["clock"]["now"] += dt.timedelta(days=1)
    assert view(env, pid, BOB) == {"counted": True}, "a new day counts again"


def test_clicks_are_counted_and_cost_nothing(env):
    pid = start(env, ad_post(env)).json()["id"]
    assert view(env, pid, BOB, "click") == {"counted": True}
    assert view(env, pid, BOB, "click") == {"counted": False}
    row = env["db"].db["ad_promotions"][0]
    assert row["clicks"] == 1 and row["spent_millicoins"] == 0


def test_spending_is_paced_per_day_and_never_exceeds_the_budget(env, monkeypatch):
    monkeypatch.setattr(promo, "COINS_PER_1000", 50000)       # 50 coins per impression, to reach limits quickly
    pid = start(env, ad_post(env), budget=200, days=2).json()["id"]   # 100 coins a day = 2 impressions a day
    viewers = [str(uuid.uuid4()) for _ in range(6)]
    counted = [view(env, pid, v)["counted"] for v in viewers[:3]]
    assert counted == [True, True, False], "today's share is used up"
    assert env["c"].get("/api/v1/promotions/serve", headers=token(viewers[5])).json()["promotions"] == []
    env["clock"]["now"] += dt.timedelta(days=1)
    assert [view(env, pid, v)["counted"] for v in viewers[3:6]] == [True, True, False]
    row = env["db"].db["ad_promotions"][0]
    assert row["spent_millicoins"] == 200 * 1000 and row["impressions"] == 4


def test_unused_coins_come_back_once_when_it_ends(env):
    pid = start(env, ad_post(env), budget=500, days=2).json()["id"]
    view(env, pid, BOB)
    view(env, pid, CAROL)                                   # 2 impressions = 0.1 coin -> 1 coin used (rounded up)
    env["clock"]["now"] += dt.timedelta(days=3)
    got = env["c"].get("/api/v1/ads-maker/promotions", headers=token(ALICE)).json()["promotions"][0]
    assert got["status"] == "ended" and got["ended_reason"] == "finished"
    assert got["coins_spent"] == 1 and got["coins_refunded"] == 499 and env["wallet"][ALICE] == 999
    env["c"].get("/api/v1/ads-maker/promotions", headers=token(ALICE))
    env["c"].post(f"/api/v1/ads-maker/promotions/{pid}/stop", headers=token(ALICE))
    assert env["wallet"][ALICE] == 999, "refunded exactly once"
    assert view(env, pid, str(uuid.uuid4())) == {"counted": False}


def test_stop_refunds_and_only_the_owner_can_stop(env):
    pid = start(env, ad_post(env)).json()["id"]
    assert env["c"].post(f"/api/v1/ads-maker/promotions/{pid}/stop", headers=token(BOB)).status_code == 404
    r = env["c"].post(f"/api/v1/ads-maker/promotions/{pid}/stop", headers=token(ALICE)).json()
    assert r["status"] == "ended" and r["ended_reason"] == "stopped" and env["wallet"][ALICE] == 1000


def test_taking_the_ad_down_ends_the_promotion_and_refunds(env):
    post = ad_post(env)
    pid = start(env, post).json()["id"]
    post["is_archived"] = True
    assert env["c"].get("/api/v1/promotions/serve", headers=token(BOB)).json()["promotions"] == []
    row = env["db"].db["ad_promotions"][0]
    assert row["status"] == "ended" and row["ended_reason"] == "ad_not_public" and env["wallet"][ALICE] == 1000
    assert view(env, pid, BOB) == {"counted": False}


def test_serving_skips_your_own_and_wrong_placements(env):
    start(env, ad_post(env), placements=["ads"])
    assert env["c"].get("/api/v1/promotions/serve?placement=for_you", headers=token(BOB)).json()["promotions"] == []
    got = env["c"].get("/api/v1/promotions/serve?placement=ads", headers=token(BOB)).json()["promotions"]
    assert len(got) == 1 and set(got[0]) == {"promotion_id", "post_id"}
    assert env["c"].get("/api/v1/promotions/serve?placement=ads", headers=token(ALICE)).json()["promotions"] == []
    assert env["c"].get("/api/v1/promotions/serve?placement=dramas", headers=token(BOB)).json()["promotions"] == []


def test_my_promotions_are_private(env):
    start(env, ad_post(env))
    assert env["c"].get("/api/v1/ads-maker/promotions", headers=token(BOB)).json()["promotions"] == []
    mine = env["c"].get("/api/v1/ads-maker/promotions", headers=token(ALICE)).json()
    assert len(mine["promotions"]) == 1 and "can't be split" in mine["note"]
