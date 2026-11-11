"""Automatic voice quality recovery: every talking clip is checked (Viyo's own
measurements plus an AI that listens and watches), and weak voices are fixed
automatically with the cheapest fix that can work, within a limit of tries and
money, without changing the dialogue, re-making other scenes, or charging the
creator for fixes that didn't work.

Fake AI, fake lip sync, local Postgres and real ffmpeg. These prove the
decisions, the limits and the billing. They do NOT prove that real video
models follow the corrections, that Gemini hears voices accurately, or that
pitch catches every voice change: that needs real video and audio generation.
"""
import os
import subprocess
import tempfile

import pytest

import studio_ai
import studio_fal
import studio_lipsync
import studio_pricing as pricing
import studio_voice_fix as vf
from conftest import make_user
from test_creator_studio import H, balance, env, grant, run, start, studio_schema  # noqa: F401
from test_video_drama import detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the studio SQL")

AMA = {"name": "Ama", "description": "Woman, 29, sharp and guarded", "gender": "female"}
KOFI = {"name": "Kofi", "description": "Man, 50, deep slow voice", "gender": "male"}
TALK = {"heading": "INT. WARD - NIGHT", "visual": "Ama confronts Kofi at the desk", "duration_seconds": 3,
        "dialogue": [{"character": "Ama", "text": "Where were you?", "emotion": "angry"},
                     {"character": "Kofi", "text": "Far away from here.", "emotion": "calm"}]}
QUIET = {"heading": "EXT. STREET - NIGHT", "visual": "Rain on the street", "duration_seconds": 3,
         "dialogue": [{"character": "Ama", "text": "Not again.", "emotion": "tired"}]}


def clip(tones=(300,), seconds=3.0, volume=1.0):
    """A test clip whose sound is one tone per equal part (a stand-in for each speaker's pitch)."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "v.mp4")
        part = seconds / len(tones)
        inputs, labels = [], []
        for i, f in enumerate(tones):
            inputs += ["-f", "lavfi", "-i", f"sine=f={f}:d={part}:sample_rate=48000"]
            labels.append(f"[{i + 1}:a]")
        mix = "".join(labels) + f"concat=n={len(tones)}:v=0:a=1,volume={volume}[a]"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"testsrc2=size=720x1280:rate=24:duration={seconds}", *inputs, "-filter_complex", mix,
                        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        "-shortest", out], check=True)
        return open(out, "rb").read()


def heard(lines, **over):
    """A listening answer: every line heard word for word, at its part of the 3 s clip."""
    n = len(lines)
    rows = [{"index": i, "heard": True, "heard_text": t, "speaker_ok": True,
             "start": round(0.1 + i * 3.0 / n, 2), "end": round((i + 1) * 3.0 / n - 0.1, 2)}
            for i, t in enumerate(lines)]
    for i, extra in over.pop("rows", {}).items():
        rows[i].update(extra)
    return {"lines": rows, "language_ok": True, "natural": 8, "clarity": 8, "extra_speech": False,
            "voices_distinct": True, **over}


GOOD = heard(["Where were you?", "Far away from here."])


def setup(env, scenes=(TALK,), credits=50000):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, credits)
    pid = project(client, u, language="English")["id"]
    split(client, ai, gen, u, pid, [dict(s) for s in scenes], characters=(AMA, KOFI))
    prompts = []
    real = ai.video

    def video(prompt, frame, ar, seconds=8, references=None):
        prompts.append(prompt)
        return real(prompt, frame, ar, seconds, references)
    ai.video = video
    return client, db, store, files, ai, gen, ren, u, pid, prompts


def make_video(client, u, pid, gen, sid):
    r = start(client, u, pid, "video", [sid])
    assert r.status_code == 200, r.text
    run(gen)
    return next(s for s in detail(client, u, pid)["scenes"] if s["id"] == sid)["voice_check"]


def job_out(client, u, pid):
    return client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]


# ------------------------------------------------------------------ judging (pure)

LINES = TALK["dialogue"]


def test_missing_and_partly_said_lines_are_found_from_the_transcript_not_only_the_ais_yes():
    tech = {"has_audio": True, "mean_db": -20, "peak_db": -3}
    h = studio_ai_clean(heard(["Where were you?", "Far"]))          # the AI said "heard" for both
    c = vf.evaluate(tech, h, LINES)
    assert c["status"] == "failed" and c["problems"][0]["kind"] == "incomplete_lines" and c["problems"][0]["lines"] == [1]
    h = studio_ai_clean(heard(["Where were you?", "Far away from"]))
    c = vf.evaluate(tech, h, LINES)
    assert c["status"] == "needs_work" and [p["kind"] for p in c["problems"]] == ["pronunciation"]
    assert vf.needs_fix(c), "unclear words are worth one automatic try"


def test_wrong_speaker_is_caught_by_pitch_against_the_approved_recordings():
    tech = {"has_audio": True, "mean_db": -20, "peak_db": -3}
    # Kofi's line comes out in Ama's (high) voice. The AI's ear didn't notice.
    c = vf.evaluate(tech, studio_ai_clean(GOOD), LINES, {"line_pitch": {0: 240.0, 1: 238.0},
                                                         "ref_pitch": {"ama": 240.0, "kofi": 110.0}})
    assert c["status"] == "failed" and c["problems"][0]["kind"] == "wrong_speaker"
    assert "line 2 (Kofi) sounds like Ama" in c["problems"][0]["problem"]


def test_a_different_voice_is_an_identity_problem_but_a_shout_or_whisper_is_not():
    tech = {"has_audio": True, "mean_db": -20, "peak_db": -3}
    calm = [{"character": "Kofi", "text": "Far away from here.", "emotion": "calm"}]
    h = studio_ai_clean(heard(["Far away from here."], rows={0: {"same_person": False}}))
    c = vf.evaluate(tech, h, calm, {"line_pitch": {0: 220.0}, "ref_pitch": {"kofi": 110.0}})
    assert [p["kind"] for p in c["problems"]] == ["identity"] and c["status"] == "failed"
    # The same pitch change, but the AI thinks it is the same person: a drift worth one try, not a failure.
    h = studio_ai_clean(heard(["Far away from here."]))
    c = vf.evaluate(tech, h, calm, {"line_pitch": {0: 220.0}, "ref_pitch": {"kofi": 110.0}})
    assert [p["kind"] for p in c["problems"]] == ["identity_drift"] and c["status"] == "needs_work"
    # Shouting: higher and tense, the AI's "doesn't sound like him" is cleared because it's the same person.
    shout = [{"character": "Kofi", "text": "Far away from here!", "emotion": "furious", "delivery": "shouts"}]
    h = studio_ai_clean(heard(["Far away from here!"], rows={0: {"speaker_ok": False, "same_person": True}}))
    c = vf.evaluate(tech, h, shout, {"line_pitch": {0: 160.0}, "ref_pitch": {"kofi": 110.0}})
    assert c["status"] == "passed" and c["problems"] == []
    assert "the voice changes with the feeling" in c["accepted"][0]


def test_lip_sync_needs_the_picture_and_half_the_lines_off_to_fail():
    tech = {"has_audio": True, "mean_db": -20, "peak_db": -3}
    h = studio_ai_clean(heard(["Where were you?", "Far away from here."], rows={1: {"lip_sync_ok": False}}))
    c = vf.evaluate(tech, h, LINES, {"watched": True})
    assert [p["kind"] for p in c["problems"]] == ["lip_sync"] and c["status"] == "failed"
    assert vf.evaluate(tech, h, LINES, {"watched": False})["status"] == "passed", "not judged without the picture"
    h = studio_ai_clean({**heard(["Where were you?", "Far away from here."], rows={1: {"lip_sync_ok": False}}),
                         "mouths_visible": False})
    assert vf.evaluate(tech, h, LINES, {"watched": True})["status"] == "passed", "mouths not visible"


def test_the_cheapest_fix_that_can_work_comes_first():
    sound = {"problems": [{"kind": "too_quiet"}, {"kind": "noise"}]}
    assert vf.ladder(sound, "omni", ["kling"], True, True) == [{"step": "clean", "provider": "omni"}]
    lips = {"problems": [{"kind": "lip_sync"}]}
    assert vf.ladder(lips, "omni", ["kling"], True, True) == [{"step": "resync", "provider": "sync"}]
    assert vf.ladder(lips, "omni", [], False, True)[0]["step"] == "retry", "no lip sync tool: make it again"
    bad = {"problems": [{"kind": "wrong_voice"}]}
    assert [s["step"] for s in vf.ladder(bad, "omni", ["kling"], True, True)] == ["retry", "switch", "saved"]
    assert [s["step"] for s in vf.ladder(bad, "omni", ["kling"], False, False)] == ["retry"], \
        "another model needs the scene picture; saved voices need lip sync"
    assert vf.ladder({"problems": [{"kind": "performance"}]}, "omni", [], True, True) == [], \
        "one line's feeling isn't worth a new clip"


def test_corrections_never_touch_the_lines_and_never_ask_for_more_drama():
    check = {"problems": [{"kind": "missing_lines", "lines": [1]}, {"kind": "overacted", "lines": []},
                          {"kind": "wrong_voice", "lines": [0]}]}
    text = vf.corrections(check, LINES, {"ama": {"identity": "a husky low voice", "pitch": "low"}}, "French")
    assert text.startswith("VOICE CORRECTIONS (the dialogue above is unchanged")
    assert 'line 2 "Far away from here."' in text and "Line 1 is spoken by AMA (" in text
    assert "Understated and real" in text and "louder" not in text and "more emotion" not in text


def studio_ai_clean(raw):
    import studio_voice
    return studio_voice.clean_listen(raw, len(raw["lines"]))


# ------------------------------------------------------------------ recovering (worker)

def test_a_missing_line_is_fixed_by_one_retry_with_exact_corrections_and_the_same_dialogue(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    ai.listen_result = [heard(["Where were you?", ""], rows={1: {"heard": False}}), GOOD]
    sid = detail(client, u, pid)["scenes"][0]["id"]
    before = balance(client, u)["available"]
    v = make_video(client, u, pid, gen, sid)
    assert len(prompts) == 2, "one retry"
    assert v["status"] == "fixed" and v["fixed"] is True and v["needs_approval"] is False
    assert "fixed automatically (made again with exact voice directions)" in v["message"]
    assert prompts[0].split("No subtitles")[0] in prompts[1], "the first prompt, lines included, is kept word for word"
    assert 'Every line is spoken in full, word for word, nothing skipped: line 2 "Far away from here."' in prompts[1]
    assert v["check"]["status"] == "passed" and v["first_check"]["status"] == "failed"
    rec = v["recovery"]
    assert [a["step"] for a in rec["attempts"]] == ["retry"] and rec["recovery_usd"] > 0
    # The fix worked, so it's charged with the first clip; never above the price shown.
    charged = before - balance(client, u)["available"]
    job = job_out(client, u, pid)
    assert charged == job["charged_credits"] and abs(charged - pricing.credits_for(rec["charged_usd"])) <= 1
    assert charged <= job["estimated_credits"]


def test_unnatural_and_mispronounced_speech_gets_one_try(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    ai.listen_result = [heard(["Where were you?", "Far away from"], natural=3), GOOD]
    sid = detail(client, u, pid)["scenes"][0]["id"]
    v = make_video(client, u, pid, gen, sid)
    assert v["status"] == "fixed" and len(prompts) == 2
    assert "Natural, conversational delivery" in prompts[1] and "pronounce every word properly" in prompts[1]


def test_failed_recovery_stops_at_the_limit_explains_and_charges_only_the_first_clip(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    always_wrong = heard(["Where were you?", "Far away from here."], rows={1: {"speaker_ok": False}})
    ai.listen_result = always_wrong
    sid = detail(client, u, pid)["scenes"][0]["id"]
    before = balance(client, u)["available"]
    v = make_video(client, u, pid, gen, sid)
    assert len(prompts) == 2, "no backup model and no lip sync: one retry, then stop (no endless loop)"
    assert v["status"] == "needs_approval" and v["needs_approval"] is True and v["fixed"] is False
    assert "Doesn't sound like the character" in v["message"] and "Viyo tried 1 automatic fix" in v["message"]
    assert "Listen and keep it, or change the scene" in v["message"]
    rec = v["recovery"]
    first = pricing.combine([pricing.video_cost(3.0, 3 * 5792, 1000), pricing.text_cost(2000, 500)])
    assert rec["charged_usd"] == round(first.total_usd, 4) and rec["absorbed_usd"] > 0
    assert before - balance(client, u)["available"] == first.credits, "the failed fix isn't charged"
    usage = db.execute("select provider_cost_usd from public.studio_usage order by created_at desc limit 1").fetchone()
    assert usage is None or float(usage[0]) >= first.provider_cost_usd, "what was really spent is recorded"


def test_attempt_and_money_limits_are_respected(env, monkeypatch):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    monkeypatch.setattr(studio_fal, "models", lambda: ["kling"])
    monkeypatch.setattr(studio_fal, "generate", lambda *a, **k: pytest.fail("over the money limit: never called"))
    ai.listen_result = heard(["Where were you?", ""], rows={1: {"heard": False}})
    monkeypatch.setenv("STUDIO_VOICE_FIX_MAX_USD", "0.60")      # a retry fits, then Kling doesn't
    sid = detail(client, u, pid)["scenes"][0]["id"]
    start(client, u, pid, "image", [sid])
    run(gen)
    v = make_video(client, u, pid, gen, sid)
    steps = [(a["step"], a["ran"]) for a in v["recovery"]["attempts"]]
    assert steps == [("retry", True), ("switch", False)] and v["recovery"]["attempts"][1]["skipped"] == "over the cost limit"
    assert "more than the limit for one clip" in v["message"]
    monkeypatch.setenv("STUDIO_VOICE_FIX_ATTEMPTS", "0")
    prompts.clear()
    v = make_video(client, u, pid, gen, sid)
    assert len(prompts) == 1 and v["message"].endswith("Listen and keep it, or change the scene.")
    assert "switched off" in v["message"]


def test_another_video_model_is_tried_when_the_same_one_keeps_failing(env, monkeypatch):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    sent = []
    monkeypatch.setenv("STUDIO_VOICE_FIX_MAX_USD", "5")
    monkeypatch.setattr(studio_fal, "models", lambda: ["seedance"])
    monkeypatch.setattr(studio_fal, "generate", lambda name, prompt, image, ar, secs, audio: (
        sent.append((name, prompt, audio)) or clip(), studio_fal.cost(name, 3, audio)))
    bad = heard(["Where were you?", "Far away from here."], rows={0: {"speaker_ok": False}})
    ai.listen_result = [bad, bad, GOOD]
    sid = detail(client, u, pid)["scenes"][0]["id"]
    start(client, u, pid, "image", [sid])         # another model starts from the scene picture
    run(gen)
    v = make_video(client, u, pid, gen, sid)
    assert [a["step"] for a in v["recovery"]["attempts"]] == ["retry", "switch"]
    assert sent[0][0] == "seedance" and sent[0][2] is True and '"Far away from here."' in sent[0][1]
    assert "Line 1 is spoken by AMA" in sent[0][1]
    assert v["status"] == "fixed" and v["provider"] == "seedance" and "another video model" in v["message"]
    out = db.execute("select output from public.studio_jobs where kind = 'video' order by created_at desc limit 1").fetchone()[0]
    assert out["model"] == "seedance" and out["voice"] == "fixed" and out["voice_recovery"]["attempts"]


def test_saved_voices_are_lip_synced_on_when_the_clips_own_voices_cant_be_fixed(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    gen.lipsync = studio_lipsync.FakeLipSync()
    bad = heard(["Where were you?", "Far away from here."], extra_speech=True)
    ai.listen_result = [bad, bad, GOOD]
    sid = detail(client, u, pid)["scenes"][0]["id"]
    v = make_video(client, u, pid, gen, sid)
    assert [a["step"] for a in v["recovery"]["attempts"]] == ["retry", "saved"]
    assert [x["text"] for x in ai.voice_log] == ["Where were you?", "Far away from here."], "the exact lines"
    assert len(gen.lipsync.calls) == 1 and gen.lipsync.calls[0][0] == "video"
    assert v["status"] == "fixed" and "saved character voices" in v["limitation"]


def test_wrong_lip_sync_is_resynced_not_made_again(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    gen.lipsync = studio_lipsync.FakeLipSync()
    ai.listen_result = [heard(["Where were you?", "Far away from here."],
                              rows={0: {"lip_sync_ok": False}, 1: {"lip_sync_ok": False}}), GOOD]
    sid = detail(client, u, pid)["scenes"][0]["id"]
    v = make_video(client, u, pid, gen, sid)
    assert ai.listen_watched[0] is True, "the check watched the clip, not only listened"
    assert "mouth movements match the words" in ai.listen_prompts[0]
    assert [a["step"] for a in v["recovery"]["attempts"]] == ["resync"] and len(prompts) == 1
    assert v["status"] == "fixed" and "the lip sync was redone" in v["message"]


def test_quiet_noisy_sound_is_cleaned_for_free(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    ai.video = lambda prompt, frame, ar, seconds=8, references=None: (
        prompts.append(prompt) or clip(volume=0.08), pricing.video_cost(3.0, 3 * 5792, 1000))
    ai.listen_result = [heard(["Where were you?", "Far away from here."], background_noise=True), GOOD]
    sid = detail(client, u, pid)["scenes"][0]["id"]
    v = make_video(client, u, pid, gen, sid)
    assert {p["kind"] for p in v["first_check"]["problems"]} == {"too_quiet", "noise"}
    assert [a["step"] for a in v["recovery"]["attempts"]] == ["clean"] and len(prompts) == 1, "no new clip"
    assert v["status"] == "fixed" and v["check"]["technical"]["mean_db"] > -35


def test_only_the_weak_scene_is_made_again(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env, scenes=(TALK, QUIET))
    s1, s2 = detail(client, u, pid)["scenes"]
    make_video(client, u, pid, gen, s1["id"])
    prompts.clear()
    ai.listen_result = [heard(["Not"]), heard(["Not again."])]
    v = make_video(client, u, pid, gen, s2["id"])
    assert len(prompts) == 2 and all("Not again." in p for p in prompts) and not any("Where were you?" in p for p in prompts)
    assert v["status"] == "fixed"


def test_a_passed_clip_becomes_the_voice_reference_and_a_changed_voice_is_caught_later(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env, scenes=(TALK, QUIET))
    s1, s2 = detail(client, u, pid)["scenes"]
    ai.video = lambda prompt, frame, ar, seconds=8, references=None: (
        prompts.append(prompt) or clip((300, 120)), pricing.video_cost(3.0, 3 * 5792, 1000))
    ai.listen_result = GOOD
    assert make_video(client, u, pid, gen, s1["id"])["status"] == "passed"
    refs = db.execute("select story_bible->'voice_refs' from public.studio_projects where id = %s", (pid,)).fetchone()[0]
    assert set(refs) == {"ama", "kofi"}
    assert abs(refs["ama"][0]["pitch_hz"] - 300) < 15 and abs(refs["kofi"][0]["pitch_hz"] - 120) < 10
    # Next scene: Ama now comes out at Kofi's pitch, every time.
    ai.video = lambda prompt, frame, ar, seconds=8, references=None: (
        prompts.append(prompt) or clip((120,)), pricing.video_cost(3.0, 3 * 5792, 1000))
    ai.listen_result = heard(["Not again."])
    v = make_video(client, u, pid, gen, s2["id"])
    assert ai.listen_refs[-1] == ["Ama"], "Ama's approved recording was sent with the check"
    assert "REFERENCE RECORDINGS" in ai.listen_prompts[-1] and "A different feeling is not a different person" \
        in ai.listen_prompts[-1]
    assert v["first_check"]["problems"][0]["kind"] == "wrong_speaker" and v["status"] == "needs_approval"


def test_the_creator_can_approve_a_flagged_voice_and_it_becomes_the_reference(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    ai.listen_result = heard(["Where were you?", "Far away from here."], rows={1: {"emotion_ok": False}})
    sid = detail(client, u, pid)["scenes"][0]["id"]
    v = make_video(client, u, pid, gen, sid)
    assert v["status"] == "needs_approval" and len(prompts) == 1, "a feeling note isn't worth a new clip"
    assert "isn't fixed automatically" in v["message"]
    r = client.post(f"/api/v1/studio/scenes/{sid}/voice/approve", headers=H(u))
    assert r.status_code == 200, r.text
    assert r.json()["voice_check"]["status"] == "approved" and r.json()["references_added"] == 2
    assert detail(client, u, pid)["scenes"][0]["voice_check"]["needs_approval"] is False
    other = make_user(db)
    assert client.post(f"/api/v1/studio/scenes/{sid}/voice/approve", headers=H(other)).status_code == 404


def test_the_price_shows_room_for_one_fix_and_a_creator_short_of_it_still_gets_the_clip(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    sid = detail(client, u, pid)["scenes"][0]["id"]
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "video", "target_ids": [sid]}).json()
    item = est["items"][0]
    assert item["voice_fix_credits"] > 0 and "room for one automatic voice fix" in est["note"]
    # A creator with enough for the clip but not the fix room.
    poor = make_user(db)
    grant(client, poor, 20)
    pid2 = project(client, poor, language="English")["id"]
    split(client, ai, gen, poor, pid2, [dict(TALK)], characters=(AMA, KOFI))
    have, want = balance(client, poor)["available"], item["credits"] - item["voice_fix_credits"] + 1
    assert have <= want
    if want > have:
        grant(client, poor, want - have)
    sid2 = detail(client, poor, pid2)["scenes"][0]["id"]
    ai.listen_result = heard(["Where were you?", ""], rows={1: {"heard": False}})
    prompts.clear()
    v = make_video(client, poor, pid2, gen, sid2)
    assert len(prompts) == 1 and v["status"] == "needs_approval" and "enough credits" in v["message"]
    assert balance(client, poor)["available"] >= 0


def test_a_provider_error_during_a_fix_never_fails_the_clip(env):
    client, db, store, files, ai, gen, ren, u, pid, prompts = setup(env)
    real = ai.video
    calls = []

    def flaky(prompt, frame, ar, seconds=8, references=None):
        calls.append(1)
        if len(calls) == 2:
            raise studio_ai.PolicyBlocked("refused")
        return real(prompt, frame, ar, seconds, references)
    ai.video = flaky
    ai.listen_result = heard(["Where were you?", ""], rows={1: {"heard": False}})
    sid = detail(client, u, pid)["scenes"][0]["id"]
    v = make_video(client, u, pid, gen, sid)
    assert job_out(client, u, pid)["status"] == "succeeded"
    assert v["recovery"]["attempts"][0]["status"] == "error" and v["status"] == "needs_approval"


def test_a_talking_shot_is_fixed_on_its_own_like_a_scene(env):
    from test_multi_shot import QUIET as Q, TALK as T, approve, make, plan
    from test_video_drama import DAN, MAYA
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 50000)
    pid = project(client, u, title="The Letter")["id"]
    split(client, ai, gen, u, pid, [dict(T), dict(Q)], characters=(MAYA, DAN))
    talk = plan(client, gen, u, pid)[0]
    approve(client, u, talk["id"])
    shots = talk["shots"]
    make(client, gen, u, pid, "shot_image", [s["id"] for s in shots])
    first = shots[0]
    said = [d["text"] for d in first["dialogue"]]
    ai.listen_result = [heard(said[:-1] + [""], rows={len(said) - 1: {"heard": False}}), heard(said)]
    before = len([c for c in ai.calls if c[0] == "video"])
    assert make(client, gen, u, pid, "shot_video", [first["id"]]).status_code == 200
    assert len([c for c in ai.calls if c[0] == "video"]) - before == 2, "the shot, and one retry of only that shot"
    d = detail(client, u, pid)
    assert "scenes" in d, d
    sh = d["scenes"][0]["shots"][0]
    assert sh["voice_check"]["status"] == "fixed" and sh["voice_check"]["recovery"]["attempts"][0]["step"] == "retry"
    r = client.post(f"/api/v1/studio/shots/{first['id']}/voice/approve", headers=H(u))
    assert r.status_code == 200 and r.json()["voice_check"]["status"] == "approved"
