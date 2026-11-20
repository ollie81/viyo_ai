"""Trending counts distinct people (not raw counters, not the creator, not accounts under review), dampens fame,
caps per creator; fast-growing, new releases, rising creators and genres; search handles partial words, typos,
story descriptions, creators, and leaves out private and blocked content."""
import datetime as dt
import uuid

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import activity
import recommend as rec
import search
import trending
from test_ads_quality import FakeDB


def tok(uid):
    return {"Authorization": "Bearer " + jwt.encode({"sub": uid}, "x", algorithm="HS256")}


def ts(hours):
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)).isoformat()


BIG, SMALL, OTHER = "big", "small", "other"
VIEWER = str(uuid.uuid4())


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    posts, series = [], []

    def add_series(sid, owner, title, genre, ctype="short_drama", days=3, desc=""):
        series.append({"id": sid, "user_id": owner, "title": title, "genre": genre, "content_type": ctype, "description": desc})
        posts.append({"id": f"{sid}-e1", "user_id": owner, "series_id": sid, "episode_number": 1, "media_url": "m",
                      "is_private": False, "is_archived": False, "created_at": ts(days * 24)})
    add_series("s_love", SMALL, "Kigali Nights", "Romance", desc="A girl falls in love with a billionaire")
    add_series("s_star", BIG, "Star Power", "Thriller")
    add_series("s_star2", BIG, "Star Power 2", "Thriller")
    add_series("s_star3", BIG, "Star Power 3", "Thriller")
    add_series("s_ghost", OTHER, "The Haunted House", "Horror", ctype="movie", days=40)
    posts.append({"id": "secret", "user_id": OTHER, "post_type": "video", "caption": "Kigali secret", "media_url": "m",
                  "is_private": True, "is_archived": False, "created_at": ts(5)})
    ev = []

    def watch(post, users, hours=10, kind="complete"):
        for u in users:
            ev.append({"user_id": u, "kind": kind, "post_id": post, "created_at": ts(hours)})
    fans = [f"u{i}" for i in range(12)]
    watch("s_love-e1", fans[:6])                            # 6 people finish the small creator's drama
    watch("s_star-e1", fans[:8], kind="view")               # the big creator: more viewers, fewer finish
    for p in ("s_star2-e1", "s_star3-e1"):
        watch(p, fans[:6], kind="view")
    for _ in range(40):                                     # one account replaying 40 times = one person
        ev.append({"user_id": "bot", "kind": "view", "post_id": "s_ghost-e1", "created_at": ts(5)})
    watch("s_ghost-e1", [OTHER] * 5)                        # the creator watching their own film doesn't count
    watch("s_love-e1", ["old1"], hours=24 * 8, kind="view")  # last week: growth comparison
    db.db.update({"series": series, "posts": posts, "viewer_events": ev, "fraud_flags": [], "blocks": [],
                  "follows": [], "watch_progress": [], "viewer_prefs": [],
                  "profiles": [{"id": SMALL, "username": "aline_films", "display_name": "Aline"},
                               {"id": BIG, "username": "starstudio", "display_name": "Star Studio"},
                               {"id": OTHER, "username": "ghostmaker", "display_name": "Ghost Maker"}]})
    monkeypatch.setattr(activity, "supabase_admin", db)
    monkeypatch.setattr(trending, "_cache", {})
    monkeypatch.setattr(search, "_cache", {})
    app = FastAPI()
    app.include_router(trending.router)
    app.include_router(search.router)
    return {"c": TestClient(app), "db": db}


def test_trending_counts_people_and_quality_and_dampens_fame(env):
    t = env["c"].get("/api/v1/discover/trending", headers=tok(VIEWER)).json()
    ids = [x["series_id"] for x in t["trending"]]
    assert ids[0] == "s_love", ids                          # fewer viewers, but they finish: and a small creator
    assert "s_ghost" not in ids, "one replaying account and the creator's own views aren't popularity"
    assert sum(1 for i in ids if i.startswith("s_star")) <= 2, "at most two per creator"


def test_rising_creators_new_releases_genres_and_regions(env):
    t = env["c"].get("/api/v1/discover/trending", headers=tok(VIEWER)).json()
    assert [x["series_id"] for x in t["rising_creators"]] == ["s_love"]
    assert "s_ghost" not in [x["series_id"] for x in t["new_releases"]], "40 days old isn't new"
    assert t["genres"][0]["genre"] in ("Romance", "Thriller")
    assert t["regions"] == [] and "region" in t["note"].lower()


def test_fast_growing(env):
    t = env["c"].get("/api/v1/discover/trending", headers=tok(VIEWER)).json()
    assert "s_love" in [x["series_id"] for x in t["fast_growing"]]


def test_accounts_under_review_dont_count(env):
    env["db"].db["fraud_flags"] = [{"subject_type": "user", "subject_id": f"u{i}", "status": "open"} for i in range(6)]
    t = env["c"].get("/api/v1/discover/trending", headers=tok(VIEWER)).json()
    assert "s_love" not in [x["series_id"] for x in t["trending"]]


def q(env, text):
    return env["c"].get("/api/v1/search", params={"q": text}, headers=tok(VIEWER)).json()


def test_search_partial_words_and_typos(env):
    assert q(env, "kiga")["results"][0]["series_id"] == "s_love"
    assert q(env, "haunted hose")["results"][0]["series_id"] == "s_ghost"
    assert q(env, "romence")["results"][0]["series_id"] == "s_love"


def test_search_by_describing_a_story(env):
    r = q(env, "a scary story about a haunted house")
    assert r["results"][0]["series_id"] == "s_ghost" and "Horror" in r["understood_genres"]
    r = q(env, "something where she falls in love with a rich man")
    assert r["results"][0]["series_id"] == "s_love" and r["results"][0]["kind"] == "drama"


def test_search_finds_creators_and_kinds(env):
    r = q(env, "aline")
    assert r["creators"][0]["username"] == "aline_films"
    assert q(env, "haunted")["results"][0]["kind"] == "movie"


def test_search_leaves_out_private_and_blocked(env):
    assert "secret" not in [x["post_id"] for x in q(env, "kigali secret")["results"]]
    env["db"].db["blocks"] = [{"blocker_id": VIEWER, "blocked_id": SMALL}]
    assert all(x["creator_id"] != SMALL for x in q(env, "kigali")["results"])


def test_search_is_recorded_for_the_viewer_and_short_queries_are_ignored(env):
    q(env, "haunted")
    assert any(e["kind"] == "search" and e["user_id"] == VIEWER for e in env["db"].db["viewer_events"])
    assert q(env, "a")["results"] == []
