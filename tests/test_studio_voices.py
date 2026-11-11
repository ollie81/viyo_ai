"""Studio stage 1: permanent character voices. Fake AI only; nothing here
calls a real provider."""
import base64
import types
import uuid

import pytest

import studio_ai
import studio_pricing as pricing
import studio_worker as sw
from conftest import make_user
from test_creator_studio import H, balance, env, grant, new_project, run, start, studio_schema  # noqa: F401


def detail(client, u, pid):
    return client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()


def cast_of(client, u, pid):
    return {c["name"]: c for c in detail(client, u, pid)["characters"]}


def voice_calls(ai):
    return [(x["text"], x["voice"], x["style"]) for x in getattr(ai, "voice_log", [])]


def made(client, ai, gen, u, credits=5000, **project):
    grant(client, u, credits, kind="purchase")
    pid = new_project(client, u, **project)
    start(client, u, pid, "script")
    run(gen)
    start(client, u, pid, "breakdown")
    run(gen)
    return pid


def voice_all(client, gen, u, pid, targets=None):
    r = start(client, u, pid, "voice", targets)
    assert r.status_code == 200, r.text
    run(gen)


# ---------------------------------------------------------------- provider

class _Interactions:
    def __init__(self, data, mime=""):
        self.sent, self.data, self.mime = {}, data, mime

    def create(self, **kw):
        self.sent.update(kw)
        return types.SimpleNamespace(output_audio=types.SimpleNamespace(data=self.data, mime_type=self.mime))


def _google(data, mime=""):
    g = object.__new__(studio_ai.GoogleStudioAI)
    g.client = types.SimpleNamespace(interactions=_Interactions(data, mime))
    return g


def test_tts_request_uses_the_documented_3_8_shape_and_keeps_the_wav():
    wav = studio_ai._pcm_to_wav(b"\x00\x00" * 24000 * 2, 24000)        # 2 s, as Gemini 3.8 returns it
    g = _google(base64.b64encode(wav).decode())
    out, secs, cost = g.voice("Muraho!", "Kore", "warm, whispering")
    sent = g.client.interactions.sent
    assert sent["model"] == pricing.TTS_MODEL == "gemini-3.8-flash-tts"
    assert sent["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
    assert sent["response_format"] == {"type": "audio"}
    assert sent["input"] == [{"type": "user_input", "content": [
        {"type": "text", "text": "Muraho!", "annotations": [{"type": "speech_metadata", "style": "warm, whispering"}]}]}]
    assert out == wav, "a WAV is never wrapped in a second header"
    assert secs == pytest.approx(2.0) and cost.units["audio_seconds"] == 2.0


def test_tts_without_style_sends_only_the_transcript():
    g = _google(studio_ai._pcm_to_wav(b"\x00\x00" * 2400, 24000))
    g.voice("Hello.", "Puck")
    assert g.client.interactions.sent["input"][0]["content"] == [{"type": "text", "text": "Hello."}]


def test_raw_pcm_from_older_models_is_wrapped_once():
    wav, secs = studio_ai.as_wav(b"\x00\x00" * 12000, "audio/L16;rate=24000")
    assert wav[:4] == b"RIFF" and secs == pytest.approx(0.5)


def test_no_audio_is_a_clear_refundable_error():
    with pytest.raises(studio_ai.PolicyBlocked):
        _google(None).voice("Hi", "Kore")


# ---------------------------------------------------------------- one character, one voice

def test_a_character_keeps_the_same_voice_in_every_scene_and_recording(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = made(client, ai, gen, u)
    cast = cast_of(client, u, pid)
    maya, daniel = cast["Maya"]["voice"], cast["Daniel"]["voice"]
    assert maya != daniel and {maya, daniel} <= studio_ai.VOICE_NAMES
    assert cast["Maya"]["voice_provider"] == "google" and cast["Maya"]["voice_language"] == "English"
    voice_all(client, gen, u, pid)
    assert {(t, v) for t, v, _ in voice_calls(ai)} == {("They said I'd never come back.", maya), ("Maya! Wait!", daniel)}
    scenes = detail(client, u, pid)["scenes"]
    rows = dict(db.execute("select position, audio_voices from public.studio_scenes where project_id=%s", (pid,)).fetchall())
    voices = {k: {c: v for c, v in r.items() if not c.startswith("_")} for k, r in rows.items()}   # "_timings" is line timing
    assert voices[0] == {cast["Maya"]["id"]: maya} and voices[1] == {cast["Daniel"]["id"]: daniel}
    assert all("_timings" in r for r in rows.values()), "each recording keeps when its lines are heard"
    assert all(s["voice_outdated"] == [] for s in scenes)
    assert cast_of(client, u, pid)["Maya"]["voice_locked"] is True

    # Regenerating a scene's dialogue uses exactly the same voice.
    voice_all(client, gen, u, pid, [scenes[0]["id"]])
    voice_all(client, gen, u, pid, [scenes[0]["id"]])
    assert [v for t, v, _ in voice_calls(ai) if t.startswith("They said")] == [maya, maya, maya]


def test_a_busy_voice_provider_is_retried_with_the_same_voice_and_charged_once(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(u), json={"name": "Maya", "voice": "Kore"})
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u),
                    json={"visual": "x", "dialogue": [{"character": "Maya", "text": "Hello"}]}).json()
    ai.fail["voice"] = [studio_ai.ProviderBusy("429")]
    start(client, u, pid, "voice", [s["id"]])
    assert run(gen) == 2                                                # busy, then retried
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded"
    assert [v for _t, v, _s in voice_calls(ai)] == ["Kore", "Kore"], "the failed try and the retry: same voice"
    assert balance(client, u)["available"] == 300 - job["charged_credits"] and balance(client, u)["held"] == 0
    assert db.execute("select count(*) from public.studio_jobs").fetchone()[0] == 1, "one job, one charge"


def test_delivery_note_and_line_emotion_reach_the_voice_model(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(u),
                json={"name": "Maya", "voice": "Kore", "voice_style": "calm, low"})
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "x", "dialogue": [
        {"character": "Maya", "text": "Get out.", "emotion": "angry"},
        {"character": "Maya", "text": "Please."},
        {"character": "NARRATOR", "text": "She never came back.", "emotion": "sad"}]}).json()
    assert s["dialogue"][0]["emotion"] == "angry"
    voice_all(client, gen, u, pid, [s["id"]])
    assert voice_calls(ai) == [("Get out.", "Kore", "calm, low, angry"), ("Please.", "Kore", "calm, low"),
                               ("She never came back.", sw.NARRATOR_VOICE, "sad")]


def test_breakdown_emotions_are_kept_on_the_lines(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 2000)
    pid = new_project(client, u)
    start(client, u, pid, "script")
    run(gen)
    ai.breakdown_result = {"characters": [{"name": "Maya", "description": "d", "gender": "female"}],
                           "scenes": [{"heading": "INT. ROOM - DAY", "visual": "v", "duration_seconds": 3,
                                       "dialogue": [{"character": "Maya", "text": "No.", "emotion": "whispering"}]}]}
    start(client, u, pid, "breakdown")
    run(gen)
    assert detail(client, u, pid)["scenes"][0]["dialogue"] == [{"character": "Maya", "text": "No.", "emotion": "whispering"}]
    assert "emotion" in studio_ai.BREAKDOWN_SCHEMA["properties"]["scenes"]["items"]["properties"]["dialogue"]["items"]["properties"]


# ---------------------------------------------------------------- deliberate changes

def test_changing_a_recorded_voice_needs_confirmation_and_keeps_old_recordings(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = made(client, ai, gen, u)
    voice_all(client, gen, u, pid)
    maya = cast_of(client, u, pid)["Maya"]
    new_voice = next(v for v in studio_ai.VOICE_NAMES if v not in (maya["voice"], cast_of(client, u, pid)["Daniel"]["voice"]))
    audio_before = db.execute("select audio_path from public.studio_scenes where project_id=%s order by position", (pid,)).fetchall()

    r = client.patch(f"/api/v1/studio/characters/{maya['id']}", headers=H(u), json={"voice": new_voice})
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "confirm_voice_change" and d["scenes"] == 1 and "keep the old voice" in d["message"]
    assert cast_of(client, u, pid)["Maya"]["voice"] == maya["voice"], "nothing changed without confirmation"

    r = client.patch(f"/api/v1/studio/characters/{maya['id']}", headers=H(u),
                     json={"voice": new_voice, "confirm_voice_change": True})
    assert r.status_code == 200
    out = r.json()
    scene0 = detail(client, u, pid)["scenes"][0]
    assert out["voice"] == new_voice and out["rerecord_scene_ids"] == [scene0["id"]] and out["voice_changed_at"]
    assert db.execute("select audio_path from public.studio_scenes where project_id=%s order by position", (pid,)).fetchall() == audio_before
    assert scene0["voice_outdated"] == ["Maya"]
    hist = db.execute("select old_voice, new_voice, scenes_affected from public.studio_voice_changes").fetchall()
    assert hist == [(maya["voice"], new_voice, 1)]

    voice_all(client, gen, u, pid, out["rerecord_scene_ids"])
    assert voice_calls(ai)[-1][1] == new_voice
    assert detail(client, u, pid)["scenes"][0]["voice_outdated"] == []


def test_changing_a_voice_before_any_recording_needs_no_confirmation(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = made(client, ai, gen, u)
    maya = cast_of(client, u, pid)["Maya"]
    r = client.patch(f"/api/v1/studio/characters/{maya['id']}", headers=H(u), json={"voice": "Sulafat", "voice_style": "gentle"})
    assert r.status_code == 200 and r.json()["voice"] == "Sulafat" and r.json()["voice_style"] == "gentle"
    assert r.json()["rerecord_scene_ids"] == []
    # Saving the same voice again is not a change.
    r = client.patch(f"/api/v1/studio/characters/{maya['id']}", headers=H(u), json={"voice": "Sulafat", "name": "Maya"})
    assert "rerecord_scene_ids" not in r.json()
    assert db.execute("select count(*) from public.studio_voice_changes").fetchone()[0] == 1


def test_unknown_voices_are_refused_when_chosen(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = made(client, ai, gen, u)
    maya = cast_of(client, u, pid)["Maya"]
    assert client.patch(f"/api/v1/studio/characters/{maya['id']}", headers=H(u), json={"voice": "Nope"}).status_code == 400


def test_editing_only_an_emotion_needs_a_new_recording_but_identical_lines_do_not(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = made(client, ai, gen, u)
    voice_all(client, gen, u, pid)
    s = detail(client, u, pid)["scenes"][0]
    same = client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u), json={"dialogue": s["dialogue"]}).json()
    assert same["has_audio"] is True
    lines = [dict(s["dialogue"][0], emotion="shouting")]
    assert client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u), json={"dialogue": lines}).json()["has_audio"] is False


# ---------------------------------------------------------------- across episodes

def test_the_next_episode_keeps_every_voice_look_and_identity(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = made(client, ai, gen, u)
    start(client, u, pid, "portrait")
    run(gen)
    client.patch(f"/api/v1/studio/characters/{cast_of(client, u, pid)['Maya']['id']}", headers=H(u),
                 json={"voice_style": "tired"})
    ep1 = cast_of(client, u, pid)
    r = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u))
    assert r.status_code == 200
    ep2 = r.json()
    assert ep2["title"] == "Night Shift · Episode 2" and ep2["series_id"] == pid
    assert detail(client, u, pid)["project"]["series_id"] == pid
    cast2 = cast_of(client, u, ep2["id"])
    for name in ("Maya", "Daniel"):
        assert cast2[name]["voice"] == ep1[name]["voice"]
        assert cast2[name]["identity_id"] == ep1[name]["id"] and cast2[name]["id"] != ep1[name]["id"]
        assert cast2[name]["portrait_url"], "same face reference"
    assert cast2["Maya"]["voice_style"] == "tired"

    # Episode 3, made from episode 2, still points at the same people.
    ep3 = client.post(f"/api/v1/studio/projects/{ep2['id']}/next-episode", headers=H(u)).json()
    assert ep3["title"] == "Night Shift · Episode 3" and ep3["series_id"] == pid
    assert cast_of(client, u, ep3["id"])["Maya"]["identity_id"] == ep1["Maya"]["id"]

    # Dialogue in episode 2 uses the episode 1 voices.
    client.patch(f"/api/v1/studio/projects/{ep2['id']}", headers=H(u), json={"script": "INT. ROOFTOP - NIGHT\n\nMAYA\nI'm back.\n"})
    ai.breakdown_result = {"characters": [{"name": "Maya", "description": "x", "gender": "female"},
                                          {"name": "Ines", "description": "Woman, 40s", "gender": "female"}],
                           "scenes": [{"heading": "INT. ROOFTOP - NIGHT", "visual": "v", "duration_seconds": 3,
                                       "dialogue": [{"character": "Maya", "text": "I'm back."},
                                                    {"character": "Ines", "text": "I know."}]}]}
    start(client, u, ep2["id"], "breakdown")
    run(gen)
    voice_all(client, gen, u, ep2["id"])
    cast2 = cast_of(client, u, ep2["id"])
    assert ("I'm back.", ep1["Maya"]["voice"], "tired") in voice_calls(ai)
    assert cast2["Ines"]["voice"] not in {ep1["Maya"]["voice"], ep1["Daniel"]["voice"]}, "a new person gets a new voice"


def test_a_character_coming_back_later_in_the_series_gets_their_old_voice(env):
    """Not copied into this episode, but in the series before: the breakdown
    brings back the same voice, delivery note, look and identity."""
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = made(client, ai, gen, u)
    ep1 = cast_of(client, u, pid)
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u)).json()
    db.execute("delete from public.studio_characters where project_id=%s and name='Daniel'", (ep2["id"],))
    client.patch(f"/api/v1/studio/projects/{ep2['id']}", headers=H(u), json={"script": "EXT. STREET\n\nDANIEL\nHey.\n"})
    ai.breakdown_result = {"characters": [{"name": "Daniel", "description": "a totally different description", "gender": "male"}],
                           "scenes": [{"heading": "EXT. STREET", "visual": "v", "duration_seconds": 3,
                                       "dialogue": [{"character": "Daniel", "text": "Hey."}]}]}
    start(client, u, ep2["id"], "breakdown")
    run(gen)
    back = cast_of(client, u, ep2["id"])["Daniel"]
    assert back["voice"] == ep1["Daniel"]["voice"] and back["identity_id"] == ep1["Daniel"]["id"]
    assert back["description"] == ep1["Daniel"]["description"], "established look not replaced"


def test_other_creators_series_are_never_used(env):
    client, db, store, files, ai, gen, ren = env
    a, b = make_user(db), make_user(db)
    pid = made(client, ai, gen, a)
    assert client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(b)).status_code == 404


# ---------------------------------------------------------------- voice list and samples

def test_voice_list_with_samples_and_admin_sample_making(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    db.execute("truncate public.studio_voice_samples")
    u = make_user(db)
    v = client.get("/api/v1/studio/voices", headers=H(u)).json()
    assert v["provider"] == "google" and len(v["voices"]) == 30 and all(x["sample_url"] is None for x in v["voices"])

    built = []
    monkeypatch.setattr(studio_ai, "GoogleStudioAI", lambda: built.append(1) or ai)
    admin = {"X-Admin-Key": "admin-secret"}
    dry = client.post("/api/v1/studio/admin/voices/samples", headers=admin, json={"voices": ["Kore", "Puck"]}).json()
    assert dry["would_make"] == ["Kore", "Puck"] and dry["made"] == [] and dry["estimated_cost_usd"] < 0.01
    assert not built, "no model call without confirm_spend"
    assert client.post("/api/v1/studio/admin/voices/samples", json={}).status_code in (401, 403)
    done = client.post("/api/v1/studio/admin/voices/samples", headers=admin,
                       json={"voices": ["Kore", "Puck"], "confirm_spend": True}).json()
    assert done["made"] == ["Kore", "Puck"] and done["failed"] == []
    v = {x["name"]: x for x in client.get("/api/v1/studio/voices", headers=H(u)).json()["voices"]}
    assert v["Kore"]["sample_url"] and v["Puck"]["sample_url"] and v["Zephyr"]["sample_url"] is None
    again = client.post("/api/v1/studio/admin/voices/samples", headers=admin, json={"voices": ["Kore"]}).json()
    assert again["would_make"] == [], "never made twice"
