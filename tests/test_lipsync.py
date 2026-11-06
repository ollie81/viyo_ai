"""Lip sync with sync.so: the saved voice plus matching lips. A fake HTTP
layer and a fake lip-sync provider only; nothing calls sync.so."""
import types

import pytest

import studio_ai
import studio_lipsync as ls
import studio_render as render
import studio_worker as sw
from conftest import make_user
from test_creator_studio import H, balance, env, grant, run, start, studio_schema  # noqa: F401

DAN = {"name": "Daniel", "description": "Man, 40, serious", "gender": "male", "age_category": "middle_aged",
       "age_years": 40, "age_confidence": "high"}
TALK = {"heading": "INT. KITCHEN - NIGHT", "visual": "Daniel at the table", "duration_seconds": 4,
        "dialogue": [{"character": "Daniel", "text": "We need to talk."}]}


# ---------------------------------------------------------------- the sync.so client

class FakeHTTP:
    def __init__(self, statuses, post_status=201):
        self.statuses, self.post_status = list(statuses), post_status
        self.posts, self.gets = [], []
        clip, _ = studio_ai.FakeStudioAI().video("x", None, "9:16", 3)
        self.clip = clip

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json, headers))
        return types.SimpleNamespace(status_code=self.post_status, json=lambda: {"id": "gen_1"}, text="nope")

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        if url.startswith(ls.SYNC_BASE):
            body = self.statuses.pop(0)
            return types.SimpleNamespace(status_code=200, json=lambda: body, text="")
        return types.SimpleNamespace(status_code=200, content=self.clip)


def test_request_matches_sync_so_docs_and_polls_to_the_result():
    http = FakeHTTP([{"status": "PENDING"}, {"status": "PROCESSING"},
                     {"status": "COMPLETED", "outputUrl": "https://cdn.sync.so/out.mp4", "model": "lipsync-2"}])
    client = ls.SyncLipSync(api_key="k", http=http, poll_seconds=0)
    started = []
    data, cost, gid = client.run("video", "https://files/v.mp4", "https://files/a.m4a", on_started=started.append)
    url, body, headers = http.posts[0]
    assert url == "https://api.sync.so/v2/generate" and headers["x-api-key"] == "k"
    assert body == {"model": "lipsync-2", "input": [{"type": "video", "url": "https://files/v.mp4"},
                                                    {"type": "audio", "url": "https://files/a.m4a"}]}
    assert started == ["gen_1"] and gid == "gen_1" and data == http.clip
    assert http.gets[:3] == ["https://api.sync.so/v2/generate/gen_1"] * 3 and http.gets[3] == "https://cdn.sync.so/out.mp4"
    # Billed per frame: 3 s at 24 fps x $0.002.
    assert cost.provider == "sync" and cost.units["frames"] == 72 and cost.provider_cost_usd == pytest.approx(0.144)


def test_resuming_never_starts_a_second_paid_job():
    http = FakeHTTP([{"status": "COMPLETED", "outputUrl": "https://cdn.sync.so/o.mp4"}])
    ls.SyncLipSync(api_key="k", http=http, poll_seconds=0).run("video", "v", "a", resume_id="gen_9")
    assert http.posts == [] and http.gets[0].endswith("/v2/generate/gen_9")


@pytest.mark.parametrize("status,exc", [(402, studio_ai.ProviderError), (429, studio_ai.ProviderBusy),
                                        (503, studio_ai.ProviderBusy), (422, studio_ai.ProviderError)])
def test_start_errors_are_classified(status, exc):
    with pytest.raises(exc):
        ls.SyncLipSync(api_key="k", http=FakeHTTP([], post_status=status)).start("video", "v", "a")


def test_rejected_and_failed_jobs():
    with pytest.raises(studio_ai.PolicyBlocked):
        ls.SyncLipSync(api_key="k", http=FakeHTTP([{"status": "REJECTED", "error": "no face"}]), poll_seconds=0).wait("g")
    with pytest.raises(studio_ai.ProviderError):
        ls.SyncLipSync(api_key="k", http=FakeHTTP([{"status": "FAILED"}]), poll_seconds=0).wait("g")


def test_prices_from_sync_so_pricing_page():
    assert ls.lipsync_cost(10, 25, "lipsync-2").provider_cost_usd == pytest.approx(0.50)
    assert ls.lipsync_cost(10, 25, "lipsync-2-pro").provider_cost_usd == pytest.approx(0.8325)
    assert ls.lipsync_cost(10, 25, "sync-3").provider_cost_usd == pytest.approx(1.3325)
    assert ls.estimate_lipsync(10).units["fps"] == 30, "holds cover a 30 fps clip"


# ---------------------------------------------------------------- in Studio

def setup(client, ai, gen, u, pid_style="video"):
    grant(client, u, 5000)
    pid = client.post("/api/v1/studio/projects", headers=H(u),
                      json={"title": "T", "accept_terms": True, "production_style": pid_style}).json()["id"]
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": "INT. KITCHEN\n\nDANIEL\nWe need to talk.\n"})
    ai.breakdown_result = {"characters": [DAN], "scenes": [TALK]}
    start(client, u, pid, "breakdown")
    run(gen)
    return pid


def test_lip_sync_stays_off_until_switched_on(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.delenv("STUDIO_LIPSYNC_ENABLED", raising=False)
    u = make_user(db)
    pid = setup(client, ai, gen, u)
    s = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]
    assert client.get("/api/v1/studio/me", headers=H(u)).json()["lipsync_available"] is False
    r = client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u), json={"speech": "lipsync"})
    assert r.status_code == 400 and "isn't switched on" in r.json()["detail"]


def test_saved_voice_plus_lip_sync_end_to_end(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setenv("STUDIO_LIPSYNC_ENABLED", "true")
    monkeypatch.setenv("SYNC_API_KEY", "test-only")
    fake = ls.FakeLipSync()
    gen = sw.StudioWorker(store, files, ai, "generation", lipsync=fake)
    u = make_user(db)
    pid = setup(client, ai, gen, u)
    s = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]
    assert client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u), json={"speech": "lipsync"}).status_code == 200
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "lipsync", "target_ids": [s["id"]]}).json()
    assert est["items"][0]["blocked"] == "Record this scene's voices first."
    start(client, u, pid, "image")
    run(gen)
    start(client, u, pid, "voice", [s["id"]])
    run(gen)
    start(client, u, pid, "video", [s["id"]])
    run(gen)
    assert "No dialogue and no speech" in ai.last_video_prompt, "the clip is silent; the saved voice speaks"
    r = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "render"}).json()
    assert r["items"][0]["blocked"] == "Scene 1's lip sync isn't made yet."
    before = balance(client, u)["available"]
    assert start(client, u, pid, "lipsync", [s["id"]]).status_code == 200
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded" and job["kind"] == "lipsync"
    assert 0 < before - balance(client, u)["available"] <= est["items"][0]["credits"]
    media_type, media_url, audio_url, resume = fake.calls[0]
    assert media_type == "video" and "video" in media_url and "audio" in audio_url and resume is None
    scene = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]
    assert scene["has_lipsync"] and scene["speech"] == "lipsync"
    assert db.execute("select provider_job_id from public.studio_jobs where kind='lipsync'").fetchone()[0] == "fake-1"

    seen = {}
    real = render.render_film
    monkeypatch.setattr(render, "render_film", lambda scenes, *a, **kw: seen.update(scenes=scenes) or real(scenes, *a, **kw))
    start(client, u, pid, "render")
    run(ren)
    (si,) = seen["scenes"]
    assert si.video and si.audio is None and si.video_audio is True, "the lip-synced clip carries the saved voice"

    # New words: the old lip sync no longer matches and must be redone.
    start(client, u, pid, "voice", [s["id"]])
    run(gen)
    assert not client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]["has_lipsync"]
    assert db.execute("select count(*) from public.studio_assets where kind='video'").fetchone()[0] == 2, \
        "the old lip-synced file is kept in the history"


def test_a_failed_lip_sync_refunds(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setenv("STUDIO_LIPSYNC_ENABLED", "true")
    monkeypatch.setenv("SYNC_API_KEY", "test-only")
    gen = sw.StudioWorker(store, files, ai, "generation", lipsync=ls.FakeLipSync(fail=studio_ai.ProviderError("boom")))
    u = make_user(db)
    pid = setup(client, ai, gen, u, pid_style="pictures")
    s = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]
    client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u), json={"speech": "lipsync"})
    start(client, u, pid, "image")
    run(gen)
    start(client, u, pid, "voice", [s["id"]])
    run(gen)
    before = balance(client, u)
    start(client, u, pid, "lipsync", [s["id"]])
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed" and balance(client, u) == before


def test_a_restarted_worker_resumes_the_same_sync_job(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setenv("STUDIO_LIPSYNC_ENABLED", "true")
    monkeypatch.setenv("SYNC_API_KEY", "test-only")
    fake = ls.FakeLipSync()
    gen = sw.StudioWorker(store, files, ai, "generation", lipsync=fake, stale_seconds=0)
    u = make_user(db)
    pid = setup(client, ai, gen, u, pid_style="pictures")
    s = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]
    client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u), json={"speech": "lipsync"})
    start(client, u, pid, "image")
    run(gen)
    start(client, u, pid, "voice", [s["id"]])
    run(gen)
    start(client, u, pid, "lipsync", [s["id"]])
    # A worker took the job, told sync.so to start (gen_77), then crashed.
    job = store.rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 0})
    store.rpc("studio_heartbeat", {"p_job_id": job["id"], "p_lease": job["lease"], "p_provider_job_id": "gen_77"})
    db.execute("update public.studio_jobs set heartbeat_at = now() - interval '1 hour' where id = %s", (job["id"],))
    run(gen)
    assert fake.resumed == ["gen_77"] and [c[3] for c in fake.calls] == ["gen_77"], "picked up, not paid twice"
    assert client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]["status"] == "succeeded"
