"""Insights and the experiment lab: only the creator's own ads, real numbers only, organic and paid apart,
trends only from real dates/snapshots; experiments create unpaid drafts and only name a leader when the data
is clear."""
import datetime as dt
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ads_brands
import ads_insights
import ads_maker
from test_ads_brands import brand, env as _brand_env  # noqa: F401
from test_ads_maker import ALICE, BOB, token

TODAY = dt.datetime.now(dt.timezone.utc).date()


def _ts(days_ago=0):
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)).isoformat()


@pytest.fixture
def env(_brand_env):
    db = _brand_env["db"].db
    for t in ("posts", "likes", "comments", "follows", "ad_post_stats_daily", "ad_experiments", "ad_ideas"):
        db[t] = []
    app = FastAPI()
    app.include_router(ads_insights.router)
    app.include_router(ads_brands.router)
    app.include_router(ads_maker.router)
    _brand_env["c"] = TestClient(app)
    return _brand_env


def _post(env, uid=ALICE, **kw):
    p = {"id": str(uuid.uuid4()), "user_id": uid, "post_kind": "ad", "view_count": 0, "like_count": 0,
         "comment_count": 0, "share_count": 0, "is_boosted": False, "is_private": False, "created_at": _ts(3), **kw}
    env["db"].db["posts"].append(p)
    return p


def test_insights_show_only_my_ads_with_real_counts(env):
    mine = _post(env, view_count=1000, like_count=40, comment_count=6, share_count=4, caption="Kivu")
    _post(env, uid=BOB, view_count=99999)
    _post(env, post_kind="post", view_count=500)                    # an ordinary post isn't an ad
    _post(env, is_archived=True, view_count=700)                     # taken down
    r = env["c"].get("/api/v1/ads-maker/insights", headers=token(ALICE)).json()
    assert [p["post_id"] for p in r["posts"]] == [mine["id"]]
    p = r["posts"][0]
    assert (p["views"], p["likes"], p["comments"], p["shares"]) == (1000, 40, 6, 4)
    assert p["engagement_rate"] == 0.05
    assert r["totals"]["organic"]["views"] == 1000 and r["totals"]["paid"]["ads"] == 0


def test_organic_and_paid_are_kept_apart(env):
    _post(env, view_count=100, like_count=10)
    _post(env, view_count=900, like_count=9, is_boosted=True)
    r = env["c"].get("/api/v1/ads-maker/insights", headers=token(ALICE)).json()
    assert r["totals"]["organic"]["views"] == 100 and r["totals"]["paid"]["views"] == 900
    assert r["posts"][0]["paid"] is True
    assert any("paid" in n for n in r["notes"])


def test_trends_come_only_from_real_dates_and_snapshots(env):
    p = _post(env, view_count=300, share_count=5)
    env["db"].db["likes"] += [{"post_id": p["id"], "user_id": "x", "created_at": _ts(1)},
                              {"post_id": p["id"], "user_id": "y", "created_at": _ts(1)},
                              {"post_id": p["id"], "user_id": "z", "created_at": _ts(40)}]   # outside the window
    env["db"].db["comments"].append({"post_id": p["id"], "user_id": "x", "created_at": _ts(0)})
    yesterday = (TODAY - dt.timedelta(days=1)).isoformat()
    env["db"].db["ad_post_stats_daily"].append({"post_id": p["id"], "owner_id": ALICE, "day": yesterday,
                                                "views": 100, "shares": 2})
    r = env["c"].get("/api/v1/ads-maker/insights?days=7", headers=token(ALICE)).json()
    by = {t["day"]: t for t in r["trend"]}
    assert len(r["trend"]) == 7
    assert by[yesterday]["likes"] == 2 and by[TODAY.isoformat()]["comments"] == 1
    assert by[TODAY.isoformat()]["views"] == 200 and by[TODAY.isoformat()]["shares"] == 3, "today's snapshot minus yesterday's"
    assert by[yesterday]["views"] is None, "no earlier snapshot: unknown, not zero or guessed"
    # one snapshot per post per day
    env["c"].get("/api/v1/ads-maker/insights", headers=token(ALICE))
    assert len([s for s in env["db"].db["ad_post_stats_daily"] if s["day"] == TODAY.isoformat()]) == 1


def test_follower_growth(env):
    env["db"].db["follows"] += [{"follower_id": "a", "following_id": ALICE, "created_at": _ts(2)},
                                {"follower_id": "b", "following_id": ALICE, "created_at": _ts(60)},
                                {"follower_id": "c", "following_id": BOB, "created_at": _ts(1)}]
    f = env["c"].get("/api/v1/ads-maker/insights", headers=token(ALICE)).json()["followers"]
    assert f["total"] == 2 and f["gained"] == 1 and len(f["by_day"]) == 1


def test_no_ads_yet(env):
    r = env["c"].get("/api/v1/ads-maker/insights", headers=token(ALICE)).json()
    assert r["posts"] == [] and r["totals"]["organic"]["engagement_rate"] is None


# ---------------------------------------------------------------- comparison

def test_no_winner_without_enough_views():
    r = ads_insights.compare([{"label": "Funny", "views": 150, "engagements": 30},
                              {"label": "Emotional", "views": 500, "engagements": 10}])
    assert r["verdict"] == "not_enough_data" and "Funny (50 more views)" in r["message"]


def test_no_winner_when_the_difference_is_not_clear():
    r = ads_insights.compare([{"label": "Funny", "views": 300, "engagements": 21},
                              {"label": "Emotional", "views": 300, "engagements": 18}])
    assert r["verdict"] == "no_clear_difference"


def test_a_clear_leader_is_named_without_promising_sales():
    r = ads_insights.compare([{"label": "Funny", "views": 2000, "engagements": 200},
                              {"label": "Emotional", "views": 2000, "engagements": 100},
                              {"label": "Cinematic", "views": 2000, "engagements": 90}])
    assert r["verdict"] == "leader" and r["leader"] == "Funny"
    assert "not a prediction of sales" in r["message"]


def test_one_version_published_is_not_a_comparison():
    assert ads_insights.compare([{"label": "Funny", "views": 5000, "engagements": 500}])["verdict"] == "not_enough_data"


# ---------------------------------------------------------------- experiments

def test_experiment_creates_unpaid_drafts_in_each_tone(env):
    b = brand(env)
    r = env["c"].post(f"/api/v1/ads-maker/brands/{b['id']}/experiments",
                      json={"tones": ["funny", "emotional", "cinematic"]}, headers=token(ALICE))
    assert r.status_code == 200, r.text
    e = r.json()
    assert [v["label"] for v in e["variants"]] == ["Funny", "Emotional", "Cinematic"]
    camps = env["db"].db["ad_campaigns"]
    assert len(camps) == 3 and all(c["experiment_id"] == e["id"] and c["brand_id"] == b["id"] for c in camps)
    assert {c["ad_style"] for c in camps} == {"social", "lifestyle", "product_cinematic"}
    assert all("Tone for this version" in c["creative_brief"] for c in camps)
    assert all(c.get("status", "draft") == "draft" for c in camps) and not env["db"].calls, "nothing charged"
    assert e["comparison"]["verdict"] == "not_enough_data"


@pytest.mark.parametrize("tones", [["funny"], ["funny", "funny"], ["funny", "sad"], list(ads_insights.TONES) + ["x"]])
def test_bad_experiments_are_refused(env, tones):
    b = brand(env)
    r = env["c"].post(f"/api/v1/ads-maker/brands/{b['id']}/experiments", json={"tones": tones}, headers=token(ALICE))
    assert r.status_code == 400 and not env["db"].db["ad_campaigns"]


def test_experiments_are_private(env):
    b = brand(env)
    e = env["c"].post(f"/api/v1/ads-maker/brands/{b['id']}/experiments", json={"tones": ["funny", "demo"]},
                      headers=token(ALICE)).json()
    assert env["c"].get(f"/api/v1/ads-maker/experiments/{e['id']}", headers=token(BOB)).status_code == 404
    assert env["c"].get("/api/v1/ads-maker/experiments", headers=token(BOB)).json() == []
    assert env["c"].post(f"/api/v1/ads-maker/brands/{b['id']}/experiments", json={"tones": ["funny", "demo"]},
                         headers=token(BOB)).status_code == 404


def test_experiment_compares_only_public_published_versions(env):
    b = brand(env)
    e = env["c"].post(f"/api/v1/ads-maker/brands/{b['id']}/experiments", json={"tones": ["funny", "emotional"]},
                      headers=token(ALICE)).json()
    camps = {c["variant_tone"]: c for c in env["db"].db["ad_campaigns"]}
    p1 = _post(env, view_count=2000, like_count=200)
    p2 = _post(env, view_count=2000, like_count=80)
    camps["funny"]["published_post_id"], camps["emotional"]["published_post_id"] = p1["id"], p2["id"]
    got = env["c"].get(f"/api/v1/ads-maker/experiments/{e['id']}", headers=token(ALICE)).json()
    assert got["comparison"]["verdict"] == "leader" and got["comparison"]["leader"] == "Funny"
    p2["is_private"] = True
    got = env["c"].get(f"/api/v1/ads-maker/experiments/{e['id']}", headers=token(ALICE)).json()
    assert got["comparison"]["verdict"] == "not_enough_data", "a private version isn't seen by anyone, so isn't compared"


def test_experiment_from_an_idea_uses_its_concept(env):
    b = brand(env)
    env["db"].db["ad_ideas"].append({"id": str(uuid.uuid4()), "owner_id": ALICE, "brand_id": b["id"],
                                     "title": "Sunset sips", "hook": "Hot day.", "concept": "Friends share cans."})
    iid = env["db"].db["ad_ideas"][0]["id"]
    env["c"].post(f"/api/v1/ads-maker/brands/{b['id']}/experiments", json={"tones": ["funny", "emotional"], "idea_id": iid},
                  headers=token(ALICE))
    assert all("Sunset sips" in c["creative_brief"] for c in env["db"].db["ad_campaigns"])
