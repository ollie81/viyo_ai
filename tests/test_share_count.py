"""Share counts: one per person per post per window, private/archived posts only by their owner."""
import uuid

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import interactions
from test_ads_quality import FakeDB

A, B = str(uuid.uuid4()), str(uuid.uuid4())


def tok(uid):
    return {"Authorization": "Bearer " + jwt.encode({"sub": uid}, "x", algorithm="HS256")}


@pytest.fixture
def c(monkeypatch):
    db = FakeDB()
    db.db["posts"] = [{"id": "p1", "user_id": A, "share_count": 0, "is_private": False, "is_archived": False},
                      {"id": "p2", "user_id": A, "share_count": 0, "is_private": True, "is_archived": False}]
    monkeypatch.setattr(interactions, "supabase_admin", db)
    monkeypatch.setattr(interactions, "_recent_shares", {})
    app = FastAPI()
    app.include_router(interactions.router)
    return TestClient(app), db


def test_a_share_is_counted_once_per_person(c):
    client, db = c
    assert client.post("/api/v1/posts/p1/share", headers=tok(B)).json()["share_count"] == 1
    assert client.post("/api/v1/posts/p1/share", headers=tok(B)).json()["share_count"] == 1
    assert client.post("/api/v1/posts/p1/share", headers=tok(A)).json()["share_count"] == 2
    assert db.db["posts"][0]["share_count"] == 2


def test_private_post_cannot_be_shared_by_others(c):
    client, db = c
    assert client.post("/api/v1/posts/p2/share", headers=tok(B)).status_code == 404
    assert client.post("/api/v1/posts/p2/share", headers=tok(A)).status_code == 200


def test_missing_post_and_no_sign_in(c):
    client, _ = c
    assert client.post("/api/v1/posts/nope/share", headers=tok(B)).status_code == 404
    assert client.post("/api/v1/posts/p1/share").status_code == 401
