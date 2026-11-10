import base64
import os
import subprocess

import pytest

import studio_music
import studio_pricing
from conftest import make_user
from test_creator_studio import H, env, grant, studio_schema  # noqa: F401
from test_creator_studio import new_project, run, start

needs_db = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_music.sql")


def test_music_prompt_is_instrumental_and_uses_project_context():
    prompt = studio_music.music_prompt({
        "genre": "mystery",
        "logline": "A detective discovers a hidden family secret.",
        "visual_style": "moody, cinematic",
    })
    assert "instrumental-only" in prompt
    assert "no vocals" in prompt.lower()
    assert "mystery" in prompt
    assert "hidden family secret" in prompt
    assert "Do not imitate" in prompt


def test_audio_parser_reads_documented_output_audio_shape():
    raw = b"ID3" + (b"x" * 2048)
    payload = {"output_audio": {"data": base64.b64encode(raw).decode("ascii")}}
    assert studio_music._audio_from_response(payload) == raw


def test_audio_parser_reads_rest_steps_shape():
    raw = b"ID3" + (b"y" * 2048)
    payload = {"steps": [{"type": "model_output", "content": [
        {"type": "audio", "data": base64.b64encode(raw).decode("ascii")}
    ]}]}
    assert studio_music._audio_from_response(payload) == raw


def test_audio_parser_rejects_missing_audio():
    with pytest.raises(studio_music.MusicGenerationError):
        studio_music._audio_from_response({"steps": []})


def test_render_estimate_includes_music_only_when_no_creator_track():
    without_music = studio_pricing.render_cost(90, music_generated=False)
    with_music = studio_pricing.render_cost(90, music_generated=True)
    assert with_music.total_usd - without_music.total_usd == pytest.approx(0.08)
    assert with_music.usage()["units"]["generated_music"] is True


def test_prompt_asks_for_a_loop_friendly_ending_only_for_long_films():
    assert "natural ending" in studio_music.music_prompt({}, 60)
    assert "repeat" in studio_music.music_prompt({}, 300)


def test_request_sends_the_api_revision_header(monkeypatch):
    seen = {}

    class R:
        status_code = 200

        def json(self):
            return {"output_audio": {"data": base64.b64encode(b"ID3" + b"z" * 2048).decode()}}

    def fake_post(url, headers, json, timeout):
        seen.update(headers=headers, body=json)
        return R()

    monkeypatch.setattr(studio_music.httpx, "post", fake_post)
    assert studio_music.generate({"genre": "thriller"}, 40, api_key="k").startswith(b"ID3")
    assert seen["headers"]["Api-Revision"] == "2026-05-20"
    assert seen["body"]["model"] == "lyria-3.5"


def test_no_key_means_no_music_not_a_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(studio_music.httpx, "post", lambda *a, **k: pytest.fail("no call without a key"))
    with pytest.raises(studio_music.MusicGenerationError):
        studio_music.generate({})


def _tone_bytes(tmp_path, seconds=6):
    out = tmp_path / "score.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=f=220:d={seconds}", str(out)], check=True)
    return out.read_bytes()


def _film(env, user, **project):
    client, db, store, files, ai, gen, ren = env
    pid = new_project(client, user, **project)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(user), json={"visual": "x", "duration_seconds": 2}).json()
    start(client, user, pid, "image", [s["id"]])
    run(gen)
    return pid


def _render(env, user, pid):
    client = env[0]
    r = start(client, user, pid, "render")
    assert r.status_code == 200, r.text
    run(env[6])
    return client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(user)).json()["jobs"][0]


@needs_db
def test_render_makes_an_original_score_once_and_charges_for_it(env, tmp_path):
    client, db = env[0], env[1]
    calls = []
    env[6].make_music = lambda project, secs: calls.append(secs) or _tone_bytes(tmp_path)
    u = make_user(db)
    grant(client, u, 500)
    pid = _film(env, u)
    job = _render(env, u, pid)
    assert job["status"] == "succeeded", job
    assert len(calls) == 1 and calls[0] == pytest.approx(2, abs=0.5)
    p = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["project"]
    assert p["has_music"] and p["music_name"] == "Viyo Original Score"
    usage = db.execute("select provider_cost_usd from public.studio_usage where status='charged' "
                       "order by id desc limit 1").fetchone()[0]
    assert float(usage) == pytest.approx(0.08)
    # A second render reuses the saved score: no new song, no new charge.
    assert _render(env, u, pid)["status"] == "succeeded" and len(calls) == 1


@needs_db
def test_a_music_failure_still_makes_the_film_without_charging_for_music(env):
    client, db = env[0], env[1]

    def broken(project, secs):
        raise studio_music.MusicGenerationError("The music service is busy.", retry=True)

    env[6].make_music = broken
    u = make_user(db)
    grant(client, u, 500)
    pid = _film(env, u)
    assert _render(env, u, pid)["status"] == "succeeded"
    p = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["project"]
    assert not p["has_music"]
    usage = db.execute("select provider_cost_usd from public.studio_usage where status='charged' "
                       "order by id desc limit 1").fetchone()[0]
    assert float(usage) == 0


@needs_db
def test_creators_can_turn_automatic_music_off(env):
    client, db = env[0], env[1]
    env[6].make_music = lambda project, secs: pytest.fail("music was turned off")
    u = make_user(db)
    grant(client, u, 500)
    pid = _film(env, u)
    r = client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"auto_music": False})
    assert r.status_code == 200 and r.json()["auto_music"] is False
    assert _render(env, u, pid)["status"] == "succeeded"
