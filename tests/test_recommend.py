"""For You: quality over raw views, personal affinity, continue watching, freshness, new creators, diversity,
not-interested, no repeats of finished titles, privacy switch, cold start, and the HTTP endpoints."""
import datetime as dt
import uuid

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import activity
import recommend as rec
from recommend import Item, Stats, Viewer
from test_ads_quality import FakeDB

NOW = dt.datetime(2026, 11, 20, 12, 0, tzinfo=dt.timezone.utc)


def item(key, creator="c1", kind="video", genre="", days=1, title="", ctype=""):
    return Item(key=key, kind=kind, post_id=key[2:], creator_id=creator, title=title, genre=genre,
                content_type=ctype, published=NOW - dt.timedelta(days=days))


def ranked(items, stats=None, v=None, cv=None, limit=20):
    return [r["post_id"] for r in rec.rank(items, stats or {}, v or Viewer(user_id="me"), NOW, cv or {}, limit=limit, seed=1)]


def test_quality_beats_raw_popularity():
    # Many viewers who quit vs. fewer viewers who finish and love it.
    crowd = Stats(viewers=900, completers=90, likers=20)
    gem = Stats(viewers=60, completers=50, likers=25)
    assert rec.quality(gem) > rec.quality(crowd)
    out = ranked([item("p:crowd", "a"), item("p:gem", "b")], {"p:crowd": crowd, "p:gem": gem}, cv={"a": 5000, "b": 5000})
    assert out[0] == "gem"


def test_a_tiny_sample_cant_look_perfect():
    assert rec.quality(Stats(viewers=2, completers=2, likers=2)) < rec.quality(Stats(viewers=80, completers=70, likers=30))


def test_personal_genre_and_followed_creators_rise():
    v = Viewer(user_id="me", genre_w={"romance": 1.0}, follows={"fav"})
    out = ranked([item("p:a", "x", genre="Horror"), item("p:b", "y", genre="Romance"), item("p:c", "fav")], v=v,
                 cv={"x": 999, "y": 999, "fav": 999})
    assert set(out[:2]) == {"b", "c"}


def test_continue_watching_comes_first():
    v = Viewer(user_id="me", progress_series={"s1": 2})
    out = rec.rank([item("p:a", "x"), item("s:s1", "y", kind="series")], {}, v, NOW, {"x": 999, "y": 999}, seed=1)
    assert out[0]["series_id"] == "s1" and out[0]["reason"] == "Continue watching"


def test_finished_hidden_blocked_and_own_items_never_show():
    v = Viewer(user_id="me", completed={"done"}, hidden_keys={"p:nope"}, hidden_creators={"meh"}, blocked={"bad"})
    out = ranked([item("p:done"), item("p:nope"), item("p:x", "meh"), item("p:y", "bad"), item("p:mine", "me"),
                  item("p:ok", "z")], v=v)
    assert out == ["ok"]


def test_shown_again_and_again_without_opening_sinks():
    v = Viewer(user_id="me")
    v.seen_recent["p:stale"] = 3
    out = ranked([item("p:stale", "a"), item("p:fresh2", "b")], v=v, cv={"a": 999, "b": 999})
    assert out[0] == "fresh2"


def test_newer_titles_rank_above_older_equals():
    out = ranked([item("p:old", "a", days=40), item("p:new", "b", days=1)], cv={"a": 999, "b": 999})
    assert out[0] == "new"


def test_good_new_creators_get_a_lift_bad_ones_dont():
    good = Stats(viewers=10, completers=9, likers=4)
    out = rec.rank([item("p:big", "big"), item("p:new", "small")], {"p:big": good, "p:new": good},
                   Viewer(user_id="me"), NOW, {"big": 5000, "small": 12}, seed=1)
    assert out[0]["post_id"] == "new" and out[0]["reason"] == "New creator"
    weak = Stats(viewers=40, completers=2)
    assert rec.score_item(item("p:w", "small"), weak, Viewer(), NOW, 12)[1] != "New creator"


def test_no_creator_floods_the_feed():
    items = [item(f"p:a{i}", "same") for i in range(6)] + [item(f"p:b{i}", f"other{i}") for i in range(6)]
    stats = {f"p:a{i}": Stats(viewers=100, completers=95, likers=60) for i in range(6)}
    out = rec.rank(items, stats, Viewer(user_id="me"), NOW, {"same": 9999}, limit=8, seed=1)
    creators = [it["post_id"][0] for it in out]
    assert creators.count("a") <= 2, creators


def test_some_places_go_to_little_seen_titles():
    items = [item(f"p:pop{i}", f"c{i}") for i in range(8)] + [item("p:unseen", "nobody", days=2)]
    stats = {f"p:pop{i}": Stats(viewers=500, completers=450, likers=200) for i in range(8)}
    out = ranked(items, stats, cv={f"c{i}": 9999 for i in range(8)}, limit=6)
    assert "unseen" in out[:5]


def test_ads_are_capped_and_never_first():
    items = [item(f"p:ad{i}", f"adv{i}", kind="ad") for i in range(4)] + [item(f"p:v{i}", f"c{i}") for i in range(10)]
    stats = {f"p:ad{i}": Stats(viewers=300, completers=290, likers=200) for i in range(4)}
    out = rec.rank(items, stats, Viewer(user_id="me"), NOW, {}, limit=14, seed=1)
    ad_at = [i for i, r in enumerate(out) if r["kind"] == "ad"]
    assert ad_at and ad_at[0] >= 3 and all(b - a >= 8 for a, b in zip(ad_at, ad_at[1:]))


def test_personalization_off_ignores_history():
    v = Viewer(user_id="me", personalized=False, genre_w={"romance": 1.0}, follows={"fav"})
    assert rec.affinity(item("p:x", "fav", genre="Romance"), v)[0] == 0.0


def test_new_accounts_get_a_sensible_mixed_feed():
    items = [item(f"p:{g}{i}", f"{g}{i}", genre=g) for g in ("Romance", "Horror", "Comedy") for i in range(3)]
    stats = {it.key: Stats(viewers=50, completers=30 + i, likers=10) for i, it in enumerate(items)}
    out = rec.rank(items, stats, Viewer(user_id="new"), NOW, {}, limit=6, seed=1)
    assert len(out) == 6 and Viewer(user_id="new").is_new


# ---------------------------------------------------------------- end to end over the API

A, B, C = (str(uuid.uuid4()) for _ in range(3))


def tok(uid):
    return {"Authorization": "Bearer " + jwt.encode({"sub": uid}, "x", algorithm="HS256")}


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    ts = lambda d: (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=d)).isoformat()
    s1 = str(uuid.uuid4())
    db.db.update({
        "series": [{"id": s1, "user_id": B, "title": "Kigali Nights", "genre": "Romance", "content_type": "short_drama"}],
        "posts": [
            {"id": "e1", "user_id": B, "series_id": s1, "episode_number": 1, "media_url": "m", "is_private": False, "is_archived": False, "created_at": ts(5)},
            {"id": "e2", "user_id": B, "series_id": s1, "episode_number": 2, "media_url": "m", "is_private": False, "is_archived": False, "created_at": ts(4)},
            {"id": "v1", "user_id": C, "post_type": "video", "post_kind": "post", "caption": "Funny cooking", "media_url": "m", "is_private": False, "is_archived": False, "created_at": ts(1)},
            {"id": "mine", "user_id": A, "post_type": "video", "media_url": "m", "is_private": False, "is_archived": False, "created_at": ts(1)},
            {"id": "secret", "user_id": C, "post_type": "video", "media_url": "m", "is_private": True, "is_archived": False, "created_at": ts(1)},
        ],
        "viewer_events": [], "viewer_prefs": [], "watch_progress": [], "follows": [], "blocks": [], "fraud_flags": []})
    monkeypatch.setattr(activity, "supabase_admin", db)
    monkeypatch.setattr(rec, "_cache", {})
    monkeypatch.setattr(activity, "_recent", __import__("collections").defaultdict(__import__("collections").deque))
    app = FastAPI()
    app.include_router(activity.router)
    app.include_router(rec.router)
    return {"c": TestClient(app), "db": db, "s1": s1}


def feed(env, uid=A):
    return env["c"].get("/api/v1/feed/for-you?limit=10", headers=tok(uid)).json()["items"]


def test_feed_endpoint_lists_public_items_not_own_or_private(env):
    ids = [i["post_id"] for i in feed(env)]
    assert set(ids) == {"e1", "v1"}, ids


def test_series_continue_with_the_next_episode(env):
    env["c"].post("/api/v1/events", json={"events": [{"kind": "progress", "post_id": "e1", "value": 1.0}]}, headers=tok(A))
    items = feed(env)
    s = next(i for i in items if i["series_id"] == env["s1"])
    assert s["post_id"] == "e2" and s["reason"] == "Continue watching" and items[0] is s


def test_not_interested_removes_it(env):
    env["c"].post("/api/v1/events", json={"events": [{"kind": "not_interested", "post_id": "v1"}]}, headers=tok(A))
    assert "v1" not in [i["post_id"] for i in feed(env)]


def test_events_are_checked_server_side(env):
    r = env["c"].post("/api/v1/events", json={"events": [
        {"kind": "like", "post_id": "v1"},             # server-only kind: dropped
        {"kind": "progress", "post_id": "nope"},       # unknown post: dropped
        {"kind": "progress", "post_id": "secret"},     # someone else's private post: dropped
        {"kind": "progress", "post_id": "v1", "value": 7},   # clamped to 1
        {"kind": "search", "query": "love story"}]}, headers=tok(A)).json()
    assert r == {"stored": 2, "dropped": 3}
    ev = env["db"].db["viewer_events"]
    assert {e["kind"] for e in ev} == {"progress", "search"} and ev[0]["creator_id"] == C and ev[0]["value"] == 1.0


def test_privacy_switch_and_clearing_history(env):
    env["c"].post("/api/v1/events", json={"events": [{"kind": "search", "query": "cooking"}]}, headers=tok(A))
    assert env["c"].put("/api/v1/viewer/prefs", json={"personalized": False}, headers=tok(A)).json() == {"personalized": False}
    assert env["c"].get("/api/v1/viewer/prefs", headers=tok(A)).json()["personalized"] is False
    assert env["c"].delete("/api/v1/viewer/history", headers=tok(A)).json() == {"cleared": True}
    assert not [e for e in env["db"].db["viewer_events"] if e["user_id"] == A]


def test_hourly_event_limit(env, monkeypatch):
    monkeypatch.setattr(activity, "MAX_PER_HOUR", 3)
    r = env["c"].post("/api/v1/events", json={"events": [{"kind": "impression", "post_id": "v1"}] * 5}, headers=tok(A)).json()
    assert r["stored"] == 3


def test_sign_in_needed(env):
    assert env["c"].get("/api/v1/feed/for-you").status_code == 401
