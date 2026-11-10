"""Backup video models (fal.ai) when Google refuses a scene. Fake network only."""
import json

import httpx
import pytest

import studio_ai
import studio_fal
from conftest import make_user
from test_creator_studio import H, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import TALK, detail, project, split

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64


def test_off_unless_the_key_and_the_switch_are_set(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setenv("STUDIO_VIDEO_BACKUP", "kling,seedance")
    assert studio_fal.models() == []
    monkeypatch.setenv("FAL_KEY", "k")
    monkeypatch.delenv("STUDIO_VIDEO_BACKUP")
    assert studio_fal.models() == []
    monkeypatch.setenv("STUDIO_VIDEO_BACKUP", "kling, nope ,seedance")
    assert studio_fal.models() == ["kling", "seedance"]


def test_kling_only_voices_english_and_chinese():
    assert studio_fal.usable("kling", True, "English")
    assert not studio_fal.usable("kling", True, "Kinyarwanda")
    assert studio_fal.usable("kling", False, "Kinyarwanda"), "no speech: the language doesn't matter"
    assert studio_fal.usable("seedance", True, "Kinyarwanda")


def test_generate_submits_waits_and_downloads(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "secret")
    monkeypatch.setattr(studio_fal, "POLL_SECONDS", 0)
    calls = []

    def handler(req: httpx.Request):
        calls.append((req.method, str(req.url)))
        assert req.headers.get("authorization") == "Key secret" or "cdn" in str(req.url)
        if req.method == "POST":
            body = json.loads(req.content)
            assert body["start_image_url"].startswith("data:image/png;base64,") and body["duration"] == "8"
            return httpx.Response(200, json={"status_url": "https://queue.fal.run/s", "response_url": "https://queue.fal.run/r"})
        if str(req.url).endswith("/s"):
            n = sum(1 for c in calls if c[1].endswith("/s"))
            return httpx.Response(200, json={"status": "COMPLETED" if n > 1 else "IN_PROGRESS"})
        if str(req.url).endswith("/r"):
            return httpx.Response(200, json={"video": {"url": "https://cdn.example/v.mp4"}})
        return httpx.Response(200, content=b"\x00" * 4096)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    data, cost = studio_fal.generate("kling", "Maya runs", PNG, "9:16", 8, True, client=client)
    assert len(data) == 4096
    assert cost.provider_cost_usd == pytest.approx(8 * 0.168) and cost.provider == "fal"


def test_a_refusal_from_the_backup_is_reported(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(422, json={"detail": "content policy violation"})))
    with pytest.raises(studio_fal.BackupFailed) as e:
        studio_fal.generate("seedance", "x", PNG, "9:16", 6, True, client=client)
    assert e.value.blocked


def _talking_scene_with_picture(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 5000)
    p = project(client, u)
    split(client, ai, gen, u, p["id"], [TALK])
    s = detail(client, u, p["id"])["scenes"][0]
    start(client, u, p["id"], "image", [s["id"]])
    run(gen)
    return client, ai, gen, u, p["id"], s["id"]


def test_google_refuses_and_the_backup_makes_the_scene(env, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    monkeypatch.setenv("STUDIO_VIDEO_BACKUP", "kling")
    client, ai, gen, u, pid, sid = _talking_scene_with_picture(env)
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "video", "target_ids": [sid]}).json()
    tried = []
    monkeypatch.setattr(studio_fal, "generate", lambda model, prompt, image, ar, secs, audio:
                        tried.append((model, audio, "You came back." in prompt)) or
                        (b"\x00" * 4096, studio_fal.cost(model, secs, audio)))
    ai.fail["video"] = studio_ai.PolicyBlocked("filtered")
    assert start(client, u, pid, "video", [sid]).status_code == 200
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded", job
    assert tried == [("kling", True, True)], "talking scene: with sound and the lines"
    assert job["charged_credits"] <= est["total_credits"], "never more than the price shown"
    assert detail(client, u, pid)["scenes"][0]["has_video"]


def test_without_the_backup_the_refusal_stands(env, monkeypatch):
    monkeypatch.delenv("STUDIO_VIDEO_BACKUP", raising=False)
    client, ai, gen, u, pid, sid = _talking_scene_with_picture(env)
    monkeypatch.setattr(studio_fal, "generate", lambda *a, **k: pytest.fail("backup is off"))
    ai.fail["video"] = studio_ai.PolicyBlocked("filtered")
    start(client, u, pid, "video", [sid])
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed" and job["charged_credits"] == 0


def test_the_price_covers_the_backup_only_when_it_is_on(env, monkeypatch):
    client, ai, gen, u, pid, sid = _talking_scene_with_picture(env)
    monkeypatch.delenv("STUDIO_VIDEO_BACKUP", raising=False)
    off = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "video", "target_ids": [sid]}).json()
    monkeypatch.setenv("FAL_KEY", "k")
    monkeypatch.setenv("STUDIO_VIDEO_BACKUP", "kling")
    on = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "video", "target_ids": [sid]}).json()
    assert on["total_credits"] > off["total_credits"]
