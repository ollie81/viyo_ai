"""Creator Hub: only the signed-in creator's own data, real counts and real earnings per production, and an
earning explanation generated from the constants the money code uses."""
import uuid

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import creator_hub
import episodes
import wallet
from test_ads_quality import FakeDB

A, B = str(uuid.uuid4()), str(uuid.uuid4())


def tok(uid, anonymous=False):
    return {"Authorization": "Bearer " + jwt.encode({"sub": uid, "is_anonymous": anonymous}, "x", algorithm="HS256")}


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
    e1, e2, e3, v1 = (str(uuid.uuid4()) for _ in range(4))
    db.db.update({
        "series": [{"id": s1, "user_id": A, "title": "Kigali Nights", "coin_price_per_episode": 30, "content_type": "short_drama"},
                   {"id": s2, "user_id": A, "title": "Free film", "coin_price_per_episode": 0, "content_type": "movie"},
                   {"id": str(uuid.uuid4()), "user_id": B, "title": "Bob's", "coin_price_per_episode": 30}],
        "posts": [
            {"id": e1, "user_id": A, "series_id": s1, "post_type": "video", "view_count": 1000, "like_count": 50, "comment_count": 5, "share_count": 2},
            {"id": e2, "user_id": A, "series_id": s1, "post_type": "video", "view_count": 400, "like_count": 10, "is_private": True},
            {"id": e3, "user_id": A, "series_id": s1, "post_type": "video", "view_count": 999, "is_archived": True},
            {"id": v1, "user_id": A, "post_type": "video", "caption": "Behind the scenes\nmore", "view_count": 300, "like_count": 30},
            {"id": str(uuid.uuid4()), "user_id": A, "post_type": "video", "post_kind": "ad", "view_count": 5000},
            {"id": str(uuid.uuid4()), "user_id": B, "post_type": "video", "view_count": 99999}],
        "creator_earnings": [
            {"creator_id": A, "source_post_id": e1, "coins": 200, "status": "available", "created_at": "2099-01-01T00:00:00Z"},
            {"creator_id": A, "source_post_id": e2, "coins": 100, "status": "paid", "created_at": "2020-01-01T00:00:00Z"},
            {"creator_id": B, "source_post_id": "x", "coins": 5000, "status": "available"}],
        "follows": [{"follower_id": B, "following_id": A}, {"follower_id": "z", "following_id": B}],
        "creator_payout_settings": [], "withdrawal_requests": [],
    })
    monkeypatch.setattr(creator_hub, "supabase_admin", db)
    monkeypatch.setattr(wallet, "supabase_admin", db)
    app = FastAPI()
    app.include_router(creator_hub.router)
    return {"c": TestClient(app), "s1": s1, "s2": s2, "v1": v1}


def test_hub_shows_only_my_real_numbers(env):
    r = env["c"].get("/api/v1/creator/hub", headers=tok(A))
    assert r.status_code == 200, r.text
    h = r.json()
    assert h["audience"]["followers"] == 1
    # archived episode left out; the ad post isn't a production here (it's in Ad insights)
    assert h["audience"]["views"] == 1000 + 400 + 300 + 5000, "every post of mine that is still up, ads included"
    series = next(p for p in h["productions"] if p["id"] == env["s1"])
    assert (series["views"], series["likes"], series["episodes"], series["public_episodes"]) == (1400, 60, 2, 1)
    assert series["earned_coins"] == 300 and series["earns"] is True
    free = next(p for p in h["productions"] if p["id"] == env["s2"])
    assert free["earns"] is False and free["earned_coins"] == 0
    video = next(p for p in h["productions"] if p["id"] == env["v1"])
    assert video["title"] == "Behind the scenes" and video["kind"] == "video"
    assert all(p["title"] != "Bob's" for p in h["productions"])
    assert not any(p.get("views") == 5000 for p in h["productions"]), "ads are not listed as productions"


def test_earnings_are_the_real_credited_coins(env):
    e = env["c"].get("/api/v1/creator/hub", headers=tok(A)).json()["earnings"]
    assert e["lifetime_coins"] == 300 and e["available_coins"] == 200 and e["paid_coins"] == 100
    assert e["available_usd"] == wallet.coins_to_usd(200) and e["last_30_days_coins"] == 200
    assert e["can_withdraw"] is False and e["payout_details_saved"] is False


def test_how_you_earn_comes_from_the_money_rules(env):
    h = env["c"].get("/api/v1/creator/hub", headers=tok(A)).json()["how_you_earn"]
    detail = h["earns"][0]["detail"]
    assert f"{round(episodes.CREATOR_SHARE * 100)}%" in detail and f"first {episodes.FREE_EPISODE_COUNT} episodes" in detail
    assert h["payout"]["min_usd"] == wallet.MIN_WITHDRAWAL_USD and h["payout"]["usd_per_coin"] == wallet.COIN_USD
    joined = " ".join(h["does_not_earn"]).lower()
    assert "free coins" in joined and "premium" in joined and "views" in joined


def test_sign_in_needed_and_guests_refused(env):
    assert env["c"].get("/api/v1/creator/hub").status_code == 401
    assert env["c"].get("/api/v1/creator/hub", headers=tok(A, anonymous=True)).status_code == 403


def test_a_new_creator_sees_zeros_not_errors(env):
    h = env["c"].get("/api/v1/creator/hub", headers=tok(str(uuid.uuid4()))).json()
    assert h["productions"] == [] and h["earnings"]["lifetime_coins"] == 0 and h["audience"]["views"] == 0
