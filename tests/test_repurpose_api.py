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
    store = JobStore(pg_rpc(connect()), JobConfig(cost=40, daily_limit=5, free_trial_daily_cap=0))
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
    assert first.json() == second.json() == {"job_id": first.json()["job_id"], "status": "processing",
                                             "free_trial": False}
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


def test_failed_job_shows_redacted_error_detail_to_its_owner(api):
    from repurpose_jobs import PermanentJobError
    uid = make_user(api["db"], 100)
    client = api["as_user"](uid)
    path = upload(api, uid)["source_path"]
    job_id = client.post("/api/v1/repurpose", json={"source_path": path}).json()["job_id"]
    job = api["store"].claim("w")
    api["store"].fail(job_id, job["lease_token"], PermanentJobError(
        "selection_unavailable", "Clip selection failed.",
        "LLM error: Error code: 401 - Incorrect API key provided: sk-proj-abc123SECRET456"), 1)
    body = client.get(f"/api/v1/repurpose/{job_id}").json()
    assert "401" in body["error_detail"]
    assert "SECRET" not in body["error_detail"] and "[redacted]" in body["error_detail"]


# ---------------------------------------------------------------- Bunny route

class FakeBunny:
    """Stands in for bunny_stream: one video whose state the test controls."""
    class BunnyError(Exception):
        pass

    class BunnyVideoNotFound(BunnyError):
        pass

    def __init__(self):
        self.videos = {}
        self.deleted = []

    def create_upload(self, title):
        import types
        vid = f"vid{len(self.videos) + 1:04d}"
        self.videos[vid] = {"status": 4, "storageSize": 5_000_000_000, "length": 600}
        data = dict(video_id=vid, library_id="1", upload_endpoint="https://tus", authorization_signature="s",
                    authorization_expire=1, playback_url="p", thumbnail_url="t")
        return types.SimpleNamespace(video_id=vid, model_dump=lambda: data)

    def get_video_info(self, vid):
        if vid not in self.videos:
            raise self.BunnyVideoNotFound(vid)
        return self.videos[vid]

    def is_finished(self, info):
        return info["status"] in (4, 8)

    def is_failed(self, info):
        return info["status"] in (5, 6)

    def delete_video(self, vid):
        self.deleted.append(vid)
        return True


@pytest.fixture
def bunny(monkeypatch):
    import sys
    fake = FakeBunny()
    monkeypatch.setitem(sys.modules, "bunny_stream", fake)
    return fake


def test_bunny_upload_is_registered_to_the_caller_and_start_uses_it(api, bunny):
    uid = make_user(api["db"], 100)
    client = api["as_user"](uid)
    t = client.post("/api/v1/repurpose/bunny-uploads", json={"size_bytes": 80_000_000})
    assert t.status_code == 200, t.text
    ticket = t.json()
    assert ticket["source_path"] == f"{uid}/{ticket['video_id']}"
    row = api["db"].execute("select user_id, bucket from repurpose_uploads where source_path=%s",
                            (ticket["source_path"],)).fetchone()
    assert str(row[0]) == str(uid) and row[1] == "bunny"
    started = client.post("/api/v1/repurpose", json={"source_path": ticket["source_path"], "source": "bunny",
                                                     "client_request_id": "bunny-tap-1"})
    assert started.status_code == 200, started.text
    job = api["db"].execute("select source_bucket, source_kind from repurpose_jobs where id=%s",
                            (started.json()["job_id"],)).fetchone()
    assert tuple(job) == ("bunny", "private_upload")


def test_bunny_start_refuses_someone_elses_video_and_unfinished_ones(api, bunny):
    a, b = make_user(api["db"], 100), make_user(api["db"], 100)
    ticket = api["as_user"](a).post("/api/v1/repurpose/bunny-uploads", json={"size_bytes": 1000}).json()
    # another user can't start a job from it
    other = api["as_user"](b).post("/api/v1/repurpose", json={"source_path": ticket["source_path"],
                                                              "source": "bunny", "client_request_id": "x-tap-0001"})
    assert other.status_code == 400
    # still processing -> asked to wait, nothing charged
    bunny.videos[ticket["video_id"]]["status"] = 2
    wait = api["as_user"](a).post("/api/v1/repurpose", json={"source_path": ticket["source_path"],
                                                             "source": "bunny", "client_request_id": "x-tap-0002"})
    assert wait.status_code == 400 and "still being prepared" in wait.json()["detail"]
    bal = api["db"].execute("select points_balance from profiles where id=%s", (a,)).fetchone()[0]
    assert bal == 100


def test_bunny_start_enforces_size_and_length_limits(api, bunny):
    uid = make_user(api["db"], 100)
    client = api["as_user"](uid)
    ticket = client.post("/api/v1/repurpose/bunny-uploads", json={"size_bytes": 1000}).json()
    bunny.videos[ticket["video_id"]]["length"] = 99999
    resp = client.post("/api/v1/repurpose", json={"source_path": ticket["source_path"], "source": "bunny",
                                                  "client_request_id": "x-tap-0003"})
    assert resp.status_code == 400 and "too long" in resp.json()["detail"]


# ---------------------------------------------------------------- free first video

def _free(api, cap=2):
    import dataclasses
    api["store"].config = dataclasses.replace(api["store"].config, free_trial_daily_cap=cap)
    return api


def _bal(api, uid):
    return api["db"].execute("select points_balance from profiles where id=%s", (uid,)).fetchone()[0]


def test_first_video_is_free_once_then_paid(api):
    _free(api)
    uid = make_user(api["db"], 0)
    client = api["as_user"](uid)
    assert client.get("/api/v1/repurpose/free-trial").json()["eligible"] is True
    first = client.post("/api/v1/repurpose", json={"source_path": upload(api, uid)["source_path"],
                                                   "client_request_id": "free-tap-1"})
    assert first.status_code == 200 and first.json()["free_trial"] is True
    again = client.post("/api/v1/repurpose", json={"source_path": upload(api, uid)["source_path"],
                                                   "client_request_id": "free-tap-1"})
    assert again.json()["job_id"] == first.json()["job_id"], "same tap, same job"
    assert _bal(api, uid) == 0
    status = client.get("/api/v1/repurpose/free-trial").json()
    assert status["eligible"] is False and status["used"] is True
    second = client.post("/api/v1/repurpose", json={"source_path": upload(api, uid)["source_path"]})
    assert second.status_code == 402, "second video needs coins"


def test_failed_free_video_gives_the_try_back(api):
    _free(api)
    uid = make_user(api["db"], 0)
    client = api["as_user"](uid)
    job = client.post("/api/v1/repurpose", json={"source_path": upload(api, uid)["source_path"]}).json()
    api["db"].execute("update repurpose_jobs set status='failed' where id=%s", (job["job_id"],))
    assert client.get("/api/v1/repurpose/free-trial").json()["eligible"] is True
    assert api["db"].execute("select repurpose_refund_job(%s)", (job["job_id"],)).fetchone()[0] == 0
    assert _bal(api, uid) == 0, "nothing to refund on a free run"


def test_free_videos_stop_at_the_daily_cap(api):
    _free(api, cap=2)
    users = [make_user(api["db"], 0) for _ in range(3)]
    codes = [api["as_user"](u).post("/api/v1/repurpose",
                                    json={"source_path": upload(api, u)["source_path"]}).status_code
             for u in users]
    assert codes == [200, 200, 402]
    assert api["as_user"](users[2]).get("/api/v1/repurpose/free-trial").json()["sold_out"] is True


def test_free_trial_switched_off_means_normal_charging(api):
    uid = make_user(api["db"], 100)
    client = api["as_user"](uid)
    assert client.get("/api/v1/repurpose/free-trial").json()["eligible"] is False
    resp = client.post("/api/v1/repurpose", json={"source_path": upload(api, uid)["source_path"]})
    assert resp.json()["free_trial"] is False and _bal(api, uid) == 60


def test_clip_length_choice_is_saved_on_the_job(api):
    uid = make_user(api["db"], 200)
    client = api["as_user"](uid)
    for choice, saved in [("short", "short"), ("auto", None), (None, None)]:
        body = {"source_path": upload(api, uid)["source_path"]}
        if choice:
            body["clip_length"] = choice
        job = client.post("/api/v1/repurpose", json=body).json()["job_id"]
        row = api["db"].execute("select clip_length from repurpose_jobs where id=%s", (job,)).fetchone()
        assert row[0] == saved, choice
    assert client.post("/api/v1/repurpose", json={"source_path": "x", "clip_length": "huge"}).status_code == 422
