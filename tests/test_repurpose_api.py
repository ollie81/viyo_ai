"""
HTTP contract tests for the repurpose endpoints, against the real
database functions. Covers the new private-upload flow and that the
responses app versions released before it rely on are unchanged.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import repurpose as r
from conftest import connect, make_user, pg_rpc
from repurpose_jobs import JobConfig, JobStore
from repurpose_storage import SOURCE_BUCKET, LocalStorage


@pytest.fixture
def api(db, tmp_path, monkeypatch):
    store = JobStore(pg_rpc(connect()), JobConfig(cost=40, daily_limit=5))
    storage = LocalStorage(str(tmp_path))
    monkeypatch.setattr(r, "job_store", store)
    monkeypatch.setattr(r, "storage", storage)
    monkeypatch.setattr(r, "job_config", store.config)
    monkeypatch.setattr(r, "SUPABASE_URL", "https://proj.supabase.co")

    app = FastAPI()
    app.include_router(r.router)
    state = {"user": None}
    app.dependency_overrides[r._get_current_user_id] = lambda: state["user"]
    client = TestClient(app)

    def as_user(uid):
        state["user"] = str(uid)
        return client

    return {"db": db, "store": store, "storage": storage, "as_user": as_user, "tmp": tmp_path}


def upload(api, uid, size=1000):
    resp = api["as_user"](uid).post("/api/v1/repurpose/uploads",
                                    json={"content_type": "video/mp4", "size_bytes": size})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    src = api["tmp"] / "src.mp4"
    src.write_bytes(b"\0" * size)
    api["storage"].put_source(SOURCE_BUCKET, body["source_path"], str(src))
    return body


def test_upload_issues_a_private_path_owned_by_the_caller(api):
    uid = make_user(api["db"])
    body = upload(api, uid)
    assert body["source_path"].startswith(f"{uid}/") and body["source_path"].endswith(".mp4")
    assert body["max_bytes"] == r.MAX_INPUT_SIZE_BYTES
    row = api["db"].execute("select user_id, bucket from repurpose_uploads where source_path=%s",
                            (body["source_path"],)).fetchone()
    assert str(row[0]) == str(uid) and row[1] == SOURCE_BUCKET


@pytest.mark.parametrize("payload,status", [
    ({"content_type": "image/png", "size_bytes": 10}, 400),
    ({"content_type": "video/mp4", "size_bytes": 10 * 1024 ** 3}, 400),
    ({"content_type": "video/mp4", "size_bytes": 0}, 422),
])
def test_upload_rejects_wrong_type_and_size(api, payload, status):
    uid = make_user(api["db"])
    assert api["as_user"](uid).post("/api/v1/repurpose/uploads", json=payload).status_code == status


def test_create_charges_once_per_client_request_id(api):
    uid = make_user(api["db"], 100)
    path = upload(api, uid)["source_path"]
    client = api["as_user"](uid)
    body = {"source_path": path, "client_request_id": "tap-123456"}
    first = client.post("/api/v1/repurpose", json=body)
    second = client.post("/api/v1/repurpose", json=body)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"job_id": first.json()["job_id"], "status": "processing"}
    bal = api["db"].execute("select points_balance from profiles where id=%s", (uid,)).fetchone()[0]
    assert bal == 60


def test_create_refuses_an_upload_that_has_not_arrived_without_charging(api):
    uid = make_user(api["db"], 100)
    resp = api["as_user"](uid).post("/api/v1/repurpose/uploads",
                                    json={"content_type": "video/mp4", "size_bytes": 10})
    path = resp.json()["source_path"]
    resp = api["as_user"](uid).post("/api/v1/repurpose", json={"source_path": path})
    assert resp.status_code == 400
    assert api["db"].execute("select points_balance from profiles where id=%s", (uid,)).fetchone()[0] == 100


def test_cannot_use_someone_elses_upload(api):
    owner, thief = make_user(api["db"]), make_user(api["db"])
    path = upload(api, owner)["source_path"]
    resp = api["as_user"](thief).post("/api/v1/repurpose", json={"source_path": path})
    assert resp.status_code == 400


def test_insufficient_coins_keeps_the_402_shape_the_app_parses(api):
    uid = make_user(api["db"], 10)
    path = upload(api, uid)["source_path"]
    resp = api["as_user"](uid).post("/api/v1/repurpose", json={"source_path": path})
    assert resp.status_code == 402
    assert resp.json()["detail"] == {"error": "insufficient_coins", "feature": "repurpose",
                                     "balance": 10, "needed": 40}


def test_legacy_public_url_requests_still_work(api):
    uid = make_user(api["db"], 100)
    resp = api["as_user"](uid).post("/api/v1/repurpose", json={
        "video_url": "https://proj.supabase.co/storage/v1/object/public/posts-media/a.mp4"})
    assert resp.status_code == 200
    bad = api["as_user"](uid).post("/api/v1/repurpose", json={"video_url": "https://evil.example/a.mp4"})
    assert bad.status_code == 400


def test_daily_limit_message_is_unchanged(api):
    uid = make_user(api["db"], 10_000)
    client = api["as_user"](uid)
    url = "https://proj.supabase.co/storage/v1/object/public/posts-media/a.mp4"
    for _ in range(5):
        assert client.post("/api/v1/repurpose", json={"video_url": url}).status_code == 200
    resp = client.post("/api/v1/repurpose", json={"video_url": url})
    assert resp.status_code == 429 and "per day" in resp.json()["detail"]


def test_status_maps_to_the_values_the_app_polls_for(api):
    uid = make_user(api["db"], 100)
    path = upload(api, uid)["source_path"]
    client = api["as_user"](uid)
    job_id = client.post("/api/v1/repurpose", json={"source_path": path}).json()["job_id"]

    queued = client.get(f"/api/v1/repurpose/{job_id}").json()
    assert queued["status"] == "processing" and queued["result"] is None
    assert queued["message"] == "Waiting for a processing slot"

    store = api["store"]
    job = store.claim("w")
    store.heartbeat(job_id, job["lease_token"], "transcribing", 0.3, "Transcribing")
    running = client.get(f"/api/v1/repurpose/{job_id}").json()
    assert (running["status"], running["stage"], running["progress"]) == ("processing", "transcribing", pytest.approx(0.3))

    from repurpose_jobs import PermanentJobError
    store.fail(job_id, job["lease_token"], PermanentJobError("no_speech", "No speech found."), 1)
    failed = client.get(f"/api/v1/repurpose/{job_id}").json()
    assert failed["status"] == "failed"
    assert failed["error"] == "No speech found. Your 40 coins were refunded."
    assert failed["coins_refunded"] == 40


def test_done_status_returns_the_result_in_the_original_shape(api):
    uid = make_user(api["db"], 100)
    client = api["as_user"](uid)
    path = upload(api, uid)["source_path"]
    job_id = client.post("/api/v1/repurpose", json={"source_path": path}).json()["job_id"]
    job = api["store"].claim("w")
    result = {"status": "success", "transcript": "hi", "feedback": None, "selection": {"target": 1},
              "clips": [{"processed_video_url": "https://x/c.mp4", "thumbnail_url": None,
                         "quote_card_url": None,
                         "highlight": {"start_time": 1.0, "end_time": 20.0, "reason": "r",
                                       "suggested_title": "t", "score": 77}}]}
    api["store"].complete(job_id, job["lease_token"], result)
    body = client.get(f"/api/v1/repurpose/{job_id}").json()
    assert body["status"] == "done"
    assert body["result"]["clips"][0]["processed_video_url"] == "https://x/c.mp4"
    assert body["result"]["clips"][0]["highlight"]["score"] == 77


def test_other_users_cannot_read_a_job(api):
    owner, other = make_user(api["db"], 100), make_user(api["db"], 100)
    path = upload(api, owner)["source_path"]
    job_id = api["as_user"](owner).post("/api/v1/repurpose", json={"source_path": path}).json()["job_id"]
    assert api["as_user"](other).get(f"/api/v1/repurpose/{job_id}").status_code == 403
    assert api["as_user"](other).get("/api/v1/repurpose/not-a-uuid").status_code == 404
