"""Talking-video dramas: every scene a moving clip where the characters
speak (Gemini Omni). Fake AI and local Postgres only; no provider is called."""
import base64
import types
import uuid

import studio_ai
import studio_render as render
import studio_worker as sw
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start

MAYA = {"name": "Maya", "description": "Woman in her 20s, short black hair, red jacket", "gender": "female",
        "age_category": "young_adult", "age_confidence": "high", "personality": "brave, warm"}
DAN = {"name": "Daniel", "description": "Man, 40, beard, grey coat, serious and protective", "gender": "male",
       "age_category": "middle_aged", "age_years": 40, "age_confidence": "high", "personality": "serious, protective"}


def project(client, u, **over):
    body = {"title": "Rooftop", "logline": "Two strangers meet on a rooftop.", "accept_terms": True, **over}
    r = client.post("/api/v1/studio/projects", headers=H(u), json=body)
    assert r.status_code == 200, r.text
    return r.json()


def detail(client, u, pid):
    return client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()


def split(client, ai, gen, u, pid, scenes, characters=(MAYA, DAN)):
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": "INT. ROOF - NIGHT\n\nA story.\n"})
    ai.breakdown_result = {"characters": list(characters), "scenes": scenes}
    assert start(client, u, pid, "breakdown").status_code == 200
    run(gen)


TALK = {"heading": "EXT. ROOFTOP - NIGHT", "visual": "Maya and Daniel face each other in the rain", "duration_seconds": 5,
        "dialogue": [{"character": "Maya", "text": "You came back.", "emotion": "hopeful"},
                     {"character": "Daniel", "text": "I never left.", "emotion": "tender", "delivery": "quiet"}]}
QUIET = {"heading": "EXT. CITY - NIGHT", "visual": "Wide shot of the city in the rain", "duration_seconds": 4, "dialogue": []}


def test_new_productions_are_talking_video_by_default(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 3000)
    p = project(client, u)
    assert p["production_style"] == "video"
    split(client, ai, gen, u, p["id"], [TALK, QUIET])
    scenes = detail(client, u, p["id"])["scenes"]
    assert [s["mode"] for s in scenes] == ["video", "video"]
    assert scenes[0]["speech"] == "video_model" and scenes[0]["speech_setting"] is None
    added = client.post(f"/api/v1/studio/projects/{p['id']}/scenes", headers=H(u), json={"visual": "x"}).json()
    assert added["mode"] == "video"
    pics = project(client, u, production_style="pictures")
    split(client, ai, gen, u, pics["id"], [TALK])
    assert detail(client, u, pics["id"])["scenes"][0]["mode"] == "still"


def test_a_talking_scene_gives_omni_the_lines_voices_and_faces(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 3000)
    p = project(client, u)
    split(client, ai, gen, u, p["id"], [TALK])
    start(client, u, p["id"], "portrait")
    run(gen)
    start(client, u, p["id"], "image")
    run(gen)
    cast = {c["name"]: c for c in detail(client, u, p["id"])["characters"]}
    s = detail(client, u, p["id"])["scenes"][0]
    assert start(client, u, p["id"], "video", [s["id"]]).status_code == 200
    run(gen)
    prompt = ai.last_video_prompt
    assert "speak on camera with natural lip movement, in English" in prompt
    assert 'MAYA (voice: ' + cast["Maya"]["voice_description"].rstrip(".") + '; delivery: hopeful): "You came back."' in prompt
    assert 'DANIEL (voice: ' in prompt and 'delivery: tender, quiet): "I never left."' in prompt
    assert prompt.index("You came back.") < prompt.index("I never left."), "lines in order"
    assert "No subtitles" in prompt and "No dialogue and no speech" not in prompt
    assert ai.last_video_refs == ["the scene: its setting, lighting and framing (not a character)",
                                  "Maya (keep this exact face, hair and clothes)",
                                  "Daniel (keep this exact face, hair and clothes)"]
    job = client.get(f"/api/v1/studio/projects/{p['id']}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded"


def test_clip_length_fits_the_dialogue_and_long_dialogue_is_refused_before_charging(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 3000)
    p = project(client, u)
    long_line = " ".join(["word"] * 40)
    split(client, ai, gen, u, p["id"], [
        dict(TALK, duration_seconds=3, dialogue=[{"character": "Maya", "text": " ".join(["word"] * 20)}]),
        dict(TALK, heading="INT. B", duration_seconds=3, dialogue=[{"character": "Maya", "text": long_line}])])
    s1, s2 = detail(client, u, p["id"])["scenes"]
    est = client.post(f"/api/v1/studio/projects/{p['id']}/estimate", headers=H(u),
                      json={"kind": "video", "target_ids": [s1["id"], s2["id"]]}).json()
    assert est["items"][0]["blocked"] is None
    assert "too much dialogue" in est["items"][1]["blocked"]
    before = balance(client, u)
    r = start(client, u, p["id"], "video", [s2["id"]])
    assert r.status_code == 400 and balance(client, u) == before
    start(client, u, p["id"], "video", [s1["id"]])
    run(gen)
    assert ai.calls[-1] == ("video", "9:16", 10), "20 words need about 9.8 s, not the 3 s scene length"


def test_no_separate_voice_track_for_scenes_spoken_in_the_video(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 3000)
    p = project(client, u)
    split(client, ai, gen, u, p["id"], [TALK])
    s = detail(client, u, p["id"])["scenes"][0]
    r = start(client, u, p["id"], "voice", [s["id"]])
    assert r.status_code == 400 and "speak this scene's lines in the video" in r.json()["detail"]["message"]


def test_render_uses_the_clips_own_speech_and_needs_the_talking_video(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 3000)
    p = project(client, u)
    split(client, ai, gen, u, p["id"], [TALK, QUIET])
    start(client, u, p["id"], "image")
    run(gen)
    est = client.post(f"/api/v1/studio/projects/{p['id']}/estimate", headers=H(u), json={"kind": "render"}).json()
    assert est["items"][0]["blocked"] == "Scene 1's talking video isn't made yet."
    start(client, u, p["id"], "video")
    run(gen)
    seen = {}
    real = render.render_film

    def spy(scenes, *a, **kw):
        seen["scenes"] = scenes
        return real(scenes, *a, **kw)

    monkeypatch.setattr(render, "render_film", spy)
    assert start(client, u, p["id"], "render").status_code == 200
    run(ren)
    job = client.get(f"/api/v1/studio/projects/{p['id']}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded", job
    talk, quiet = seen["scenes"]
    assert talk.video and talk.audio is None and talk.video_audio is True, "the clip's own speech, nothing on top"
    assert talk.lines == ("You came back.", "I never left."), "captions still come from the script"
    assert quiet.video_audio is True


def test_a_scene_can_use_the_saved_voices_instead(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 3000)
    p = project(client, u)
    split(client, ai, gen, u, p["id"], [TALK])
    s = detail(client, u, p["id"])["scenes"][0]
    out = client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u), json={"speech": "saved_voice"}).json()
    assert out["speech_setting"] == "saved_voice"
    assert detail(client, u, p["id"])["scenes"][0]["speech"] == "saved_voice"
    assert start(client, u, p["id"], "voice", [s["id"]]).status_code == 200
    run(gen)
    start(client, u, p["id"], "image")
    run(gen)
    start(client, u, p["id"], "video", [s["id"]])
    run(gen)
    assert "No dialogue and no speech" in ai.last_video_prompt
    seen = {}
    real = render.render_film
    monkeypatch.setattr(render, "render_film", lambda scenes, *a, **kw: seen.update(scenes=scenes) or real(scenes, *a, **kw))
    start(client, u, p["id"], "render")
    run(ren)
    (scene,) = seen["scenes"]
    assert scene.audio and scene.video_audio is False, "saved voice only; the clip's sound isn't used"


def test_a_child_in_a_talking_video_gets_a_childlike_voice_description(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 3000)
    p = project(client, u)
    lily = {"name": "Lily", "description": "A 7-year-old girl, curious", "gender": "female",
            "age_category": "young_child", "age_years": 7, "age_confidence": "high"}
    split(client, ai, gen, u, p["id"], [dict(TALK, dialogue=[{"character": "Lily", "text": "Is it raining?"}])],
          characters=[lily])
    s = detail(client, u, p["id"])["scenes"][0]
    start(client, u, p["id"], "video", [s["id"]])
    run(gen)
    assert 'LILY (voice: A natural childlike female speaking voice' in ai.last_video_prompt
    assert "adult" not in ai.last_video_prompt.split("LILY (voice:")[1].split(")")[0]


def test_next_episode_keeps_the_production_style(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    p = project(client, u)
    ep2 = client.post(f"/api/v1/studio/projects/{p['id']}/next-episode", headers=H(u)).json()
    assert ep2["production_style"] == "video"
    pics = project(client, u, production_style="pictures")
    assert client.post(f"/api/v1/studio/projects/{pics['id']}/next-episode", headers=H(u)).json()["production_style"] == "pictures"


def test_google_request_carries_labelled_reference_images():
    sent = {}

    class Interactions:
        def create(self, **kw):
            sent.update(kw)
            return types.SimpleNamespace(output_video=types.SimpleNamespace(data=base64.b64encode(b"mp4").decode()), usage=None)

    g = object.__new__(studio_ai.GoogleStudioAI)
    g.client = types.SimpleNamespace(interactions=Interactions())
    g.video("They talk.", b"frame", "9:16", 8, [("the scene", b"a"), ("Maya", b"b"), ("Daniel", b"c")])
    images = [x for x in sent["input"] if x["type"] == "image"]
    text = [x for x in sent["input"] if x["type"] == "text"][0]["text"]
    assert len(images) == 3 and base64.b64decode(images[1]["data"]) == b"b"
    assert text.startswith("Image 1: the scene. Image 2: Maya. Image 3: Daniel.") and text.endswith("They talk.")
    g.video("Rain.", b"frame", "9:16", 8)
    assert [x["type"] for x in sent["input"]] == ["image", "text"], "one image: still image-to-video"
