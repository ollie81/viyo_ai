"""Creator analytics from real records (each person once, own viewing excluded, honest watch time, retention,
advice only with enough viewers) and fraud review (detectors, no duplicate flags, effects limited to trending,
admin review, the person sees and appeals), plus honest view counting."""
import datetime as dt
import uuid

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import activity
import creator_analytics as ca
import fraud
import interactions
from test_ads_quality import FakeDB

CR = str(uuid.uuid4())


def tok(uid, anon=False):
    return {"Authorization": "Bearer " + jwt.encode({"sub": uid, "is_anonymous": anon}, "x", algorithm="HS256")}


def ts(hours=1):
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)).isoformat()


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    db.db.update({
        "series": [{"id": "s1", "user_id": CR, "title": "Kigali Nights"}],
        "posts": [{"id": "e1", "user_id": CR, "series_id": "s1", "episode_number": 1, "duration_seconds": 100, "view_count": 999},
                  {"id": "e2", "user_id": CR, "series_id": "s1", "episode_number": 2, "duration_seconds": 100}],
        "viewer_events": [], "watch_progress": [], "follows": [], "creator_earnings": [], "fraud_flags": [],
        "transactions": []})
    monkeypatch.setattr(activity, "supabase_admin", db)
    monkeypatch.setenv("ADMIN_API_KEY", "k")
    app = FastAPI()
    app.include_router(ca.router)
    app.include_router(fraud.router)
    return {"c": TestClient(app), "db": db}


def ev(db, user, kind, post, value=None, hours=1):
    db.db["viewer_events"].append({"user_id": user, "kind": kind, "post_id": post, "creator_id": CR, "value": value,
                                   "created_at": ts(hours)})


def test_unique_viewers_completion_watch_time_and_retention(env):
    db = env["db"]
    for i in range(30):
        u = f"v{i}"
        ev(db, u, "view", "e1")
        ev(db, u, "view", "e1")                         # watching twice is still one viewer
        ev(db, u, "progress", "e1", 1.0 if i < 24 else 0.5)
        if i < 9:
            ev(db, u, "progress", "e2", 1.0)
    ev(db, CR, "view", "e1")                            # the creator's own viewing doesn't count
    a = env["c"].get("/api/v1/creator/analytics", headers=tok(CR)).json()
    p = a["productions"][0]
    assert a["totals"]["unique_viewers"] == 30 and p["unique_viewers"] == 30
    assert p["episodes"][0]["completion"] == 0.8 and p["episodes"][1]["retention_from_first"] == 0.3
    # watch time: 24 people x 100s + 6 x 50s on e1 + 9 x 100s on e2 = 3600s = 60 min (each person once per video)
    assert p["watch_minutes"] == 60.0
    assert p["views_counter"] == 999, "the raw counter is shown separately, not mixed into unique viewers"
    texts = " ".join(x["text"] for x in a["advice"])
    assert "stop after episode 1" in texts and "30%" in texts


def test_no_advice_without_enough_viewers(env):
    ev(env["db"], "v1", "view", "e1")
    a = env["c"].get("/api/v1/creator/analytics", headers=tok(CR)).json()
    assert [x["text"][:36] for x in a["advice"]] == ["Advice appears once at least 20 peop"]
    assert all("guarantee" not in x["text"].lower() for x in a["advice"])


def test_followers_and_earnings_by_day(env):
    env["db"].db["follows"] = [{"follower_id": "a", "following_id": CR, "created_at": ts(5)},
                               {"follower_id": "b", "following_id": CR, "created_at": ts(24 * 60)}]
    env["db"].db["creator_earnings"] = [{"creator_id": CR, "coins": 20, "created_at": ts(2)}]
    a = env["c"].get("/api/v1/creator/analytics", headers=tok(CR)).json()
    assert a["totals"]["followers"] == 2 and a["totals"]["new_followers"] == 1 and a["totals"]["earned_coins"] == 20


def test_analytics_need_a_real_account(env):
    assert env["c"].get("/api/v1/creator/analytics", headers=tok(CR, anon=True)).status_code == 403


# ---------------------------------------------------------------- fraud review

def test_detectors_flag_without_duplicates(env, monkeypatch):
    monkeypatch.setattr(fraud, "VIEW_BURST_POSTS_PER_DAY", 5)
    db = env["db"]
    for i in range(6):
        ev(db, "bot", "view", f"p{i}")
    for i in range(12):                                  # 12 accounts that only ever like this one post
        ev(db, f"liker{i}", "like", "e1")
    db.db["transactions"] = [{"user_id": "farmer", "amount": 2000, "type": "mission_reward", "created_at": ts(3)},
                             {"user_id": "buyer", "amount": 5000, "type": "coin_purchase", "created_at": ts(3)}]
    r = env["c"].post("/api/v1/admin/fraud/scan", headers={"X-Admin-Key": "k"}).json()
    got = {(f["subject_id"], f["reason"]) for f in r["flags_created"]}
    assert got == {("bot", "view_burst"), ("e1", "like_ring"), ("farmer", "free_coin_velocity")}
    again = env["c"].post("/api/v1/admin/fraud/scan", headers={"X-Admin-Key": "k"}).json()
    assert again["flags_created"] == [] and len(db.db["fraud_flags"]) == 3


def test_ordinary_heavy_users_are_not_flagged(env):
    for i in range(50):
        ev(env["db"], "fan", "view", f"p{i}")
    assert env["c"].post("/api/v1/admin/fraud/scan", headers={"X-Admin-Key": "k"}).json()["flags_created"] == []


def test_review_and_appeal(env):
    db = env["db"]
    db.db["fraud_flags"] = [{"id": "f1", "subject_type": "user", "subject_id": "u1", "reason": "view_burst", "status": "open"},
                            {"id": "f2", "subject_type": "post", "subject_id": "e1", "reason": "like_ring", "status": "open"}]
    assert env["c"].get("/api/v1/admin/fraud/flags").status_code == 403
    assert len(env["c"].get("/api/v1/admin/fraud/flags", headers={"X-Admin-Key": "k"}).json()) == 2
    mine = env["c"].get("/api/v1/account/reviews", headers=tok("u1")).json()
    assert [m["id"] for m in mine] == ["f1"] and "Nothing else changes" in mine[0]["effect"]
    creator = env["c"].get("/api/v1/account/reviews", headers=tok(CR)).json()
    assert [m["id"] for m in creator] == ["f2"], "a creator sees reviews about their posts"
    assert env["c"].post("/api/v1/account/reviews/f1/appeal", json={"text": "I binge-watch on weekends"},
                         headers=tok("someone_else")).status_code == 404
    assert env["c"].post("/api/v1/account/reviews/f1/appeal", json={"text": "I binge-watch on weekends"},
                         headers=tok("u1")).json() == {"appealed": True}
    r = env["c"].post("/api/v1/admin/fraud/flags/f1/resolve", json={"status": "cleared", "note": "real fan"},
                      headers={"X-Admin-Key": "k"}).json()
    assert r["status"] == "cleared" and db.db["fraud_flags"][0]["appeal"] == "I binge-watch on weekends"


# ---------------------------------------------------------------- honest view counts

def test_views_count_once_per_person_and_never_the_creator(env, monkeypatch):
    db = env["db"]
    db.db["posts"][0]["view_count"] = 0
    monkeypatch.setattr(interactions, "supabase_admin", db)
    app = FastAPI()
    app.include_router(interactions.router)
    c = TestClient(app)
    viewer = str(uuid.uuid4())
    assert c.post("/api/v1/posts/e1/view", headers=tok(viewer)).json()["view_count"] == 1
    assert c.post("/api/v1/posts/e1/view", headers=tok(viewer)).json()["view_count"] == 1
    assert c.post("/api/v1/posts/e1/view", headers=tok(CR)).json()["view_count"] == 1
    assert c.post("/api/v1/posts/e1/view", headers=tok(str(uuid.uuid4()))).json()["view_count"] == 2
    assert [e["kind"] for e in db.db["viewer_events"]].count("view") == 2
