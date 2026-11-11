"""The voice director: a Character Voice Bible cast from the story (not only
age and gender), distinct voices, established voices never replaced, inherited
by later episodes, scene-by-scene performance direction (restrained unless the
script asks for more, with subtext), the same voice direction adapted to each
video provider's real limits (the dialogue is never cut off by a backup
model's prompt limit, limits recorded), saved voices as the persistent
fallback, and a sound check of every talking clip that names the exact lines
that are wrong.

Fake AI, local Postgres and real ffmpeg. These prove what each provider is
TOLD and what is checked; they do NOT prove how the real voices sound or that
they stay identical. That needs real-model listening tests.
"""
import os
import uuid

import pytest

import creator_studio as cs
import studio_fal
import studio_voice as vd
from conftest import make_user
from test_creator_studio import H, env, grant, run, start, studio_schema  # noqa: F401
from test_video_drama import detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the studio SQL")

AMA = {"name": "Ama", "description": "Woman, 29, sharp and guarded, grew up in the market", "gender": "female"}
GRACE = {"name": "Grace", "description": "Nurse, 55, tired, kind, hiding a terrible secret", "gender": "female"}
TALK = {"heading": "INT. WARD - NIGHT", "visual": "Ama confronts Grace at the desk", "duration_seconds": 8,
        "dialogue": [{"character": "Ama", "text": "Where is my brother?", "emotion": "angry"},
                     {"character": "Grace", "text": "I don't know any Kofi.", "emotion": "calm"},
                     {"character": "Ama", "text": "You're lying.", "emotion": "furious", "delivery": "shouts"}]}


def setup(env, style="video"):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 50000)
    pid = project(client, u, production_style=style, language="English")["id"]
    db.execute("update public.studio_projects set story_bible = %s where id = %s", (
        '{"characters": [{"name": "Grace", "age": "55", "secret": "she signed the false death record", '
        '"speech": "formal, careful"}, {"name": "Ama", "age": "29", "want": "find Kofi", "speech": "blunt"}], '
        '"logline": "x", "beats": [{"purpose": "hook"}], "_assistant_approval": {"approved": true}}', pid))
    split(client, ai, gen, u, pid, [dict(TALK)], characters=(AMA, GRACE))
    return client, db, store, files, ai, gen, ren, u, pid


# ------------------------------------------------------------------ the voice bible

def test_voices_are_cast_from_the_story_and_kept_distinct(env):
    client, db, store, files, ai, gen, ren, u, pid = setup(env)
    prompt = ai.voice_prompts[-1]
    assert "from WHO THEY ARE" in prompt and "not only their age and gender" in prompt
    assert "secret: she signed the false death record" in prompt and "speech: formal, careful" in prompt
    assert "\"I don't know any Kofi.\" (calm)" in prompt, "their own lines inform the voice"
    v = client.get(f"/api/v1/studio/projects/{pid}/voices", headers=H(u)).json()
    assert set(v["voices"]) == {"ama", "grace"} and v["similar"] == []
    assert v["voices"]["ama"]["pitch"] != v["voices"]["grace"]["pitch"]


def test_voices_that_sound_alike_are_sent_back_once(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    asks = []

    def same(prompt):
        asks.append(prompt)
        style = {"pitch": "middle", "texture": "smooth", "speed": "measured", "vocal_age": "adult"} if len(asks) == 1 else None
        return {"voices": [{"character": n, "identity": f"{n} voice", "rhythm": "x", "personality": "x",
                            "emotional_range": "x", **(style or {"pitch": p, "texture": t, "speed": "measured",
                                                                  "vocal_age": "adult"})}
                           for n, p, t in (("Ama", "low", "husky"), ("Grace", "high", "thin"))]}
    ai.voice_result = same
    u = make_user(db)
    grant(client, u, 50000)
    pid = project(client, u)["id"]
    split(client, ai, gen, u, pid, [dict(TALK)], characters=(AMA, GRACE))
    assert len(asks) == 2 and "THESE VOICES ARE TOO ALIKE" in asks[1] and "Ama and Grace" in asks[1]
    voices = client.get(f"/api/v1/studio/projects/{pid}/voices", headers=H(u)).json()["voices"]
    assert voices["ama"]["pitch"] == "low" and voices["grace"]["pitch"] == "high"


def test_an_established_voice_is_never_replaced_and_the_creator_can_change_it(env, monkeypatch):
    client, db, store, files, ai, gen, ren, u, pid = setup(env)
    before = client.get(f"/api/v1/studio/projects/{pid}/voices", headers=H(u)).json()["voices"]["ama"]
    monkeypatch.setattr(cs, "_ai", ai)
    r = client.post(f"/api/v1/studio/projects/{pid}/voices/direct", headers=H(u), json={"request_id": uuid.uuid4().hex})
    assert r.status_code == 200 and r.json()["voices"]["ama"]["identity"] == before["identity"], "kept"
    r = client.patch(f"/api/v1/studio/projects/{pid}/voices/Ama", headers=H(u),
                     json={"fields": {"accent": "Kigali Kinyarwanda accent", "pitch": "low-middle"}})
    assert r.status_code == 200 and r.json()["voice"]["edited_by_creator"] is True
    v = client.get(f"/api/v1/studio/projects/{pid}/voices", headers=H(u)).json()["voices"]["ama"]
    assert v["accent"] == "Kigali Kinyarwanda accent" and v["identity"] == before["identity"]


def test_a_new_episode_keeps_every_established_voice(env):
    client, db, store, files, ai, gen, ren, u, pid = setup(env)
    ep1 = client.get(f"/api/v1/studio/projects/{pid}/voices", headers=H(u)).json()["voices"]
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u)).json()["id"]
    calls = len(ai.voice_prompts)
    split(client, ai, gen, u, ep2, [dict(TALK, heading="INT. CORRIDOR - NIGHT")], characters=(AMA, GRACE))
    assert len(ai.voice_prompts) == calls, "nobody new: no recasting at all"
    ep2v = client.get(f"/api/v1/studio/projects/{ep2}/voices", headers=H(u)).json()["voices"]
    assert ep2v["ama"]["identity"] == ep1["ama"]["identity"] and ep2v["grace"]["identity"] == ep1["grace"]["identity"]


# ------------------------------------------------------------------ performance direction

def test_each_line_gets_its_feeling_volume_and_subtext_and_stays_understated_by_default():
    a = vd.direct_line({"text": "Where is my brother?", "emotion": "angry"}, {})
    assert a["intensity"] == "held back" and a["volume"] == "conversational"
    b = vd.direct_line({"text": "You're lying.", "emotion": "furious", "delivery": "shouts"}, {})
    assert b["volume"] == "raised" and b["intensity"] == "strong"
    c = vd.direct_line({"text": "Come closer.", "delivery": "whispers"}, {})
    assert c["volume"] == "soft"
    d = vd.direct_line({"text": "I don't know any Kofi.", "emotion": "calm"}, {}, hides="she signed the false record")
    assert "hiding that she signed the false record" in d["subtext"]
    assert "felt but held back, not overplayed" in vd.line_phrase(a)


# ------------------------------------------------------------------ what each provider receives

def test_the_talking_clip_carries_each_voice_identity_and_performance(env):
    client, db, store, files, ai, gen, ren, u, pid = setup(env)
    sid = detail(client, u, pid)["scenes"][0]["id"]
    start(client, u, pid, "video", [sid])
    run(gen)
    p = ai.last_video_prompt
    v = client.get(f"/api/v1/studio/projects/{pid}/voices", headers=H(u)).json()["voices"]
    assert f"AMA (voice: {v['ama']['identity']}" in p and f"GRACE (voice: {v['grace']['identity']}" in p
    assert "hiding that she signed the false death record" in p, "subtext from the story"
    assert p.index('"Where is my brother?"') < p.index('"I don\'t know any Kofi."') < p.index('"You\'re lying."')
    assert "clearly different from each other" in p
    rec = db.execute("select made_with->'_voice' from public.studio_scenes where id = %s", (sid,)).fetchone()[0]
    assert rec["provider"] == "omni" and rec["speakers"] == ["Ama", "Grace"]
    assert any("no voice ID or voice reference" in x for x in rec["limitations"]), "the limit is recorded, not hidden"


def test_a_backup_model_gets_its_own_shorter_form_and_never_loses_the_lines():
    lines = TALK["dialogue"]
    voices = {"ama": {"character": "Ama", "identity": "a low husky voice", "accent": "Kigali"},
              "grace": {"character": "Grace", "identity": "a thin careful voice"}}
    k = vd.for_provider("kling", lines, voices, "English")
    assert "DIALOGUE in English (exact words, in order)" in k["text"] and "Ama's voice: a low husky voice; Kigali accent" in k["text"]
    long_scene = "A very long scene description. " * 200
    prompt = vd.video_prompt("kling", long_scene, k["text"])
    assert len(prompt) <= vd.CAPS["kling"]["prompt_chars"] and prompt.endswith(k["text"]), "the scene is cut, never the lines"
    fr = vd.for_provider("kling", lines, voices, "French")
    assert any("only speaks chinese, english" in x for x in fr["limitations"])
    assert any("speaker mix-ups are possible" in x for x in k["limitations"])


def test_switching_providers_keeps_the_same_voice_direction():
    voices = {"ama": {"character": "Ama", "identity": "a low husky voice"}, "grace": {"character": "Grace", "identity": "thin"}}
    omni = vd.for_provider("omni", TALK["dialogue"], voices, "English")["text"]
    seed = vd.for_provider("seedance", TALK["dialogue"], voices, "English")["text"]
    for t in (omni, seed):
        assert "a low husky voice" in t and '"You\'re lying."' in t


def test_when_google_refuses_the_backup_gets_the_dialogue_in_its_own_form(env, monkeypatch):
    client, db, store, files, ai, gen, ren, u, pid = setup(env)
    import studio_ai
    sent = {}
    monkeypatch.setattr(studio_fal, "models", lambda: ["kling"])
    monkeypatch.setattr(studio_fal, "generate", lambda name, prompt, image, ar, secs, audio: (
        sent.update(name=name, prompt=prompt) or (open(__file__, "rb").read()[:0] or _clip()), studio_fal.cost(name, secs, audio)))
    sid = detail(client, u, pid)["scenes"][0]["id"]
    start(client, u, pid, "image", [sid])        # backups start from the scene picture
    run(gen)
    ai.fail = {"video": studio_ai.PolicyBlocked("refused")}
    start(client, u, pid, "video", [sid])
    run(gen)
    assert sent["name"] == "kling" and len(sent["prompt"]) <= 2500
    assert '"You\'re lying."' in sent["prompt"] and "DIALOGUE in English" in sent["prompt"]
    rec = db.execute("select made_with->'_voice' from public.studio_scenes where id = %s", (sid,)).fetchone()[0]
    assert rec["provider"] == "kling"


def _clip():
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "v.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=720x1280:rate=24:duration=3",
                        "-f", "lavfi", "-i", "sine=f=200:d=3", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        "-shortest", out], check=True)
        return open(out, "rb").read()


def test_saved_voices_keep_the_named_voice_and_get_a_short_performance(env):
    client, db, store, files, ai, gen, ren, u, pid = setup(env, style="pictures")
    sid = detail(client, u, pid)["scenes"][0]["id"]
    start(client, u, pid, "voice", [sid])
    run(gen)
    styles = [x["style"] for x in ai.voice_log]
    assert len(styles) == 3 and all(len(s) <= 160 for s in styles)
    assert "raised volume" in styles[2] and "full intensity" in styles[2]
    assert "hiding that she signed" in styles[1]
    assert len({x["voice"] for x in ai.voice_log[:2]}) == 2, "two characters, two named voices"


# ------------------------------------------------------------------ checking the sound

def test_a_talking_clip_is_listened_to_and_a_missing_line_is_named(env):
    client, db, store, files, ai, gen, ren, u, pid = setup(env)
    ai.listen_result = {"lines": [{"index": 0, "heard": True, "heard_text": "Where is my brother?", "speaker_ok": True},
                                  {"index": 1, "heard": False, "heard_text": "", "speaker_ok": True},
                                  {"index": 2, "heard": True, "heard_text": "You're lying", "speaker_ok": False}],
                        "language_ok": True, "natural": 7, "clarity": 8, "extra_speech": False, "voices_distinct": True}
    sid = detail(client, u, pid)["scenes"][0]["id"]
    start(client, u, pid, "video", [sid])
    run(gen)
    assert ai.calls[-1][0] == "listen" and ai.calls[-1][1] > 1000, "real audio bytes were sent"
    assert "expected voice:" in ai.listen_prompts[-1] and "Where is my brother?" in ai.listen_prompts[-1]
    chk = detail(client, u, pid)["scenes"][0]["voice_check"]["check"]
    assert chk["status"] == "failed" and chk["technical"]["has_audio"] is True
    kinds = {p["kind"]: p for p in chk["problems"]}
    assert "Grace: \"I don't know any Kofi.\"" in kinds["missing_lines"]["problem"]
    assert kinds["wrong_voice"]["lines"] == [2]


def test_a_silent_clip_fails_without_asking_the_ai(env, monkeypatch):
    client, db, store, files, ai, gen, ren, u, pid = setup(env)
    import subprocess
    import tempfile

    def silent(prompt, frame, ar, seconds=8, references=None):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "v.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=720x1280:rate=24:duration=3",
                            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "3", "-c:v", "libx264",
                            "-preset", "ultrafast", "-c:a", "aac", out], check=True)
            return open(out, "rb").read(), studio_ai_cost()
    monkeypatch.setattr(ai, "video", silent)
    sid = detail(client, u, pid)["scenes"][0]["id"]
    start(client, u, pid, "video", [sid])
    run(gen)
    chk = detail(client, u, pid)["scenes"][0]["voice_check"]["check"]
    assert chk["status"] == "failed" and chk["problems"][0]["kind"] == "silent" and chk["listened"] is False
    assert not any(c[0] == "listen" for c in ai.calls)


def studio_ai_cost():
    import studio_pricing
    return studio_pricing.video_cost(3.0, 3 * 5792, 1000)


def test_a_clean_clip_passes_and_the_assistant_hears_about_failures(env, monkeypatch):
    client, db, store, files, ai, gen, ren, u, pid = setup(env)
    sid = detail(client, u, pid)["scenes"][0]["id"]
    start(client, u, pid, "video", [sid])
    run(gen)
    assert detail(client, u, pid)["scenes"][0]["voice_check"]["check"]["status"] == "passed"
    db.execute("""update public.studio_scenes set made_with = jsonb_set(made_with, '{_voice,check}',
               '{"status": "failed", "problems": [{"kind": "missing_lines", "problem": "Not said: Grace: \\"I don''t know any Kofi.\\"", "lines": [1]}]}')
               where id = %s""", (sid,))
    monkeypatch.setattr(cs, "_ai", ai)
    ai.chat_result = {"reply": "Line 2 is missing.", "proposals": [], "memory": []}
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": "How do the voices sound?", "request_id": uuid.uuid4().hex})
    assert r.status_code == 200
    prompt = ai.chat_prompts[-1]
    assert "VOICE BIBLE" in prompt and "VOICE CHECK scene 1 (OMNI): FAILED" in prompt
    assert "never promise identical" in prompt and "Google Gemini Omni: native audio yes; voice ID or reference NO" in prompt
