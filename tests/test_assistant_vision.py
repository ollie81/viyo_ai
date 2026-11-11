"""The assistant sees real pictures (and says when it can't), audits big
productions honestly, keeps its own guesses out of approved memory, never
pretends a memory was saved, rejects malformed answers, and applies a
suggestion only once. Fake AI and local Postgres: no real AI credits."""
import io
import json
import os
import uuid

import pytest
from PIL import Image

import creator_studio as cs
import studio_memory
import studio_visual
from conftest import make_user
from test_continuity import build
from test_creator_studio import H, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import detail, project

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the assistant safety SQL")


@pytest.fixture
def setup(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    monkeypatch.setattr(cs, "_files", files)
    c, d, a, g, u, pid = build(env)
    return client, db, store, files, ai, gen, u, pid


def say(client, u, pid, text):
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": text, "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    return r.json()


def reply(text="Here's what I see.", **kw):
    return {"reply": text, "proposals": kw.get("proposals", []), "memory": kw.get("memory", [])}


def png(color="red"):
    out = io.BytesIO()
    Image.new("RGB", (1600, 900), color).save(out, "PNG")
    return out.getvalue()


# ------------------------------------------------------------------ seeing

def test_a_visual_question_attaches_the_real_pictures_and_says_which(setup):
    client, db, store, files, ai, gen, u, pid = setup
    start(client, u, pid, "portrait")
    run(gen)
    for s in detail(client, u, pid)["scenes"][:2]:
        start(client, u, pid, "image", [s["id"]])
        run(gen)
    ai.chat_result = reply()
    r = say(client, u, pid, "Why does Maya look different in scene 2?")
    labels = ai.chat_images[-1]
    assert any("scene 1 picture" in label for label in labels), "the scene before, to compare"
    assert any("scene 2 picture" in label for label in labels)
    assert "Maya's approved reference picture" in labels
    assert r["seen"] == labels
    prompt = ai.chat_prompts[-1]
    assert "VISUAL MATERIAL (these images are attached to this message and you can see them):" in prompt
    assert "- Image 1: Maya's approved reference picture" in prompt


def test_attached_images_are_small(setup):
    client, db, store, files, ai, gen, u, pid = setup
    small = studio_visual.shrink(png())
    img = Image.open(io.BytesIO(small))
    assert max(img.size) <= studio_visual.MAX_SIDE and img.format == "JPEG"
    assert studio_visual.shrink(b"not an image") is None


def test_it_says_what_it_could_not_see(setup, monkeypatch):
    client, db, store, files, ai, gen, u, pid = setup
    sc = detail(client, u, pid)["scenes"]
    start(client, u, pid, "image", [sc[0]["id"]])
    run(gen)
    ai.chat_result = reply()
    say(client, u, pid, "Does scene 2 look like scene 1?")
    prompt = ai.chat_prompts[-1]
    assert "NOT seen (don't describe these as if you saw them):" in prompt
    assert "scene 2: no picture or video made yet" in prompt
    real_get = files.get
    monkeypatch.setattr(files, "get", lambda path: (_ for _ in ()).throw(OSError("gone")) if "image" in path else real_get(path))
    say(client, u, pid, "Does scene 1 look right?")
    assert "couldn't be loaded" in ai.chat_prompts[-1] and ai.chat_images[-1] == []


def test_a_plain_question_attaches_nothing_and_says_so(setup):
    client, db, store, files, ai, gen, u, pid = setup
    ai.chat_result = reply()
    say(client, u, pid, "What should the title be?")
    assert ai.chat_images[-1] == []
    assert "VISUAL MATERIAL: none attached to this message. You have NOT seen the pictures" in ai.chat_prompts[-1]


def test_a_scene_with_only_a_video_sends_one_frame(setup):
    client, db, store, files, ai, gen, u, pid = setup
    s = detail(client, u, pid)["scenes"][1]
    start(client, u, pid, "image", [s["id"]])
    run(gen)
    start(client, u, pid, "video", [s["id"]])
    run(gen)
    db.execute("update public.studio_scenes set image_path = null where id = %s", (s["id"],))
    ai.chat_result = reply()
    say(client, u, pid, "How does scene 2 look?")
    assert any("scene 2 video frame" in label for label in ai.chat_images[-1])


# ------------------------------------------------------------------ big productions

def big(env, n, pictures=True, checked=0):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 2000)
    pid = project(client, u, title=f"Big {n}", production_style="pictures")["id"]
    for i in range(n):
        path = f"{u}/{pid}/image/s{i}.png" if pictures else None
        if path:
            files.put(path, png("blue" if i % 2 else "red"), "image/png")
        ins = {"result": {"scores": {"overall": 90}, "problems": [], "observed": {}},
               "image_path": path, "signature": "x"} if i < checked else {}
        db.execute("insert into public.studio_scenes (project_id, owner_id, position, heading, visual, image_path, "
                   "inspection) values (%s, %s, %s, %s, %s, %s, %s)",
                   (pid, u, i, f"INT. ROOM {i} - NIGHT", f"Maya in room {i}", path, json.dumps(ins)))
    return client, ai, u, pid


@pytest.mark.parametrize("n", [25, 65, 130])
def test_auditing_a_big_production_is_honest_about_what_was_checked(env, monkeypatch, n):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    monkeypatch.setattr(cs, "_files", files)
    client, ai, u, pid = big(env, n, checked=10)
    ai.chat_result = reply("I checked part of it.")
    r = say(client, u, pid, "Check the entire production: do the characters look consistent in every scene?")
    prompt = ai.chat_prompts[-1]
    assert f"plan-level continuity review covers all {n} scenes" in prompt
    assert f"picture checks are current for 10 of {n} scene pictures" in prompt
    assert "A visual audit of the whole production is NOT complete" in prompt
    assert len(ai.chat_images[-1]) <= studio_visual.MAX_IMAGES
    assert f"of {n} scene pictures are attached (a sample). This is NOT a check of every scene" in prompt
    assert r["reply"]["content"] == "I checked part of it."
    if n > 120:
        assert f"(+{n - 120} more scenes not listed" in prompt


def test_a_fully_checked_production_says_so(env, monkeypatch):
    monkeypatch.setattr(cs, "_ai", env[4])
    monkeypatch.setattr(cs, "_files", env[3])
    client, ai, u, pid = big(env, 22, checked=22)
    ai.chat_result = reply()
    say(client, u, pid, "Is everything consistent?")
    assert "Every picture has a current check." in ai.chat_prompts[-1]


# ------------------------------------------------------------------ memory

def test_the_assistant_cannot_turn_its_own_guess_into_an_approved_fact(setup):
    client, db, store, files, ai, gen, u, pid = setup
    ai.chat_result = reply(memory=[{"category": "characters", "text": "Maya secretly works for the police",
                                    "from_creator": True}])
    say(client, u, pid, "Make the second scene more tense.")
    mem = client.get(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u)).json()["memory"]
    assert mem["characters"] == [] and mem["assumptions"][0]["text"] == "Maya secretly works for the police"


def test_agreeing_to_an_idea_makes_it_a_decision(setup):
    client, db, store, files, ai, gen, u, pid = setup
    ai.chat_result = reply("What if Daniel is her brother?",
                           memory=[{"category": "ideas", "text": "Daniel is Maya's brother", "from_creator": False}])
    say(client, u, pid, "Who could Daniel be?")
    ai.chat_result = reply("Great.", memory=[{"category": "decisions", "text": "Daniel is Maya's brother", "from_creator": True}])
    say(client, u, pid, "Yes, let's do that.")
    mem = client.get(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u)).json()["memory"]
    assert [x["text"] for x in mem["decisions"]] == ["Daniel is Maya's brother"] and mem["ideas"] == []


def test_a_memory_that_failed_to_save_is_not_claimed(setup, monkeypatch):
    client, db, store, files, ai, gen, u, pid = setup
    monkeypatch.setattr(cs, "_save_memory", lambda p, mem: False)
    ai.chat_result = reply("Got it, Maya is 30.", memory=[{"category": "characters", "text": "Maya is 30 years old",
                                                          "from_creator": True}])
    r = say(client, u, pid, "Maya is 30 years old.")
    assert r["memory_saved"] is False
    assert "I couldn't save this to the production's memory just now" in r["reply"]["content"]
    res = client.post(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u), json={"category": "story", "text": "x"})
    assert res.status_code == 503


# ------------------------------------------------------------------ safety

def test_malformed_answers_never_change_anything(setup):
    client, db, store, files, ai, gen, u, pid = setup
    before = len(detail(client, u, pid)["characters"])
    ai.chat_result = {"reply": "Sure.", "proposals": "add everything",
                      "memory": [{"category": "bogus", "text": "x"}, {"category": "story"}, "nope"],
                      "focus": {"scenes": ["2", True, 3]}}
    r = say(client, u, pid, "do something")
    assert r["reply"]["proposals"] == []
    assert studio_memory.count(client.get(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u)).json()["memory"]) == 0
    ai.chat_result = {"reply": "x", "proposals": [{"type": "delete_everything", "title": "x", "data": {}},
                                                  {"type": "add_character", "title": "Ama", "data": ["not", "a", "dict"]}],
                      "memory": []}
    assert say(client, u, pid, "again")["reply"]["proposals"] == []
    assert len(detail(client, u, pid)["characters"]) == before
    ai.chat_result = {"memory": []}
    bad = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                      json={"message": "hi", "request_id": uuid.uuid4().hex})
    assert bad.status_code == 502


def test_a_suggestion_is_applied_only_once_even_with_two_taps(setup, monkeypatch):
    client, db, store, files, ai, gen, u, pid = setup
    ai.chat_result = reply("Add Ama?", proposals=[{"type": "add_character", "title": "Add character: Ama", "summary": "x",
                                                   "data": {"name": "Ama", "description": "20s"}}])
    mid = say(client, u, pid, "Add a sister")["reply"]["id"]
    m = store.one("studio_messages", {"id": mid})
    assert cs._claim_proposal(m, 0, u) is True
    assert cs._claim_proposal(m, 0, u) is False, "the second tap loses"
    r = client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/0/apply", headers=H(u))
    assert r.status_code == 409
    assert sum(1 for c in detail(client, u, pid)["characters"] if c["name"] == "Ama") == 0
    # Without the database function, a local lock gives the same result.
    ai.chat_result = reply("Add Kofi?", proposals=[{"type": "add_character", "title": "Add character: Kofi", "summary": "x",
                                                    "data": {"name": "Kofi", "description": "30s"}}])
    mid2 = say(client, u, pid, "Add a brother")["reply"]["id"]
    real_rpc = store.rpc
    monkeypatch.setattr(store, "rpc", lambda name, params: (_ for _ in ()).throw(Exception("missing"))
                        if name == "studio_claim_proposal" else real_rpc(name, params))
    assert client.post(f"/api/v1/studio/assistant/messages/{mid2}/proposals/0/apply", headers=H(u)).status_code == 200
    assert client.post(f"/api/v1/studio/assistant/messages/{mid2}/proposals/0/apply", headers=H(u)).status_code == 409
    assert sum(1 for c in detail(client, u, pid)["characters"] if c["name"] == "Kofi") == 1


def test_nobody_else_can_use_or_see_another_creators_assistant(setup):
    client, db, store, files, ai, gen, u, pid = setup
    ai.chat_result = reply("Add Ama?", proposals=[{"type": "add_character", "title": "Add character: Ama", "summary": "x",
                                                   "data": {"name": "Ama", "description": "20s"}}])
    mid = say(client, u, pid, "Add a sister")["reply"]["id"]
    other = make_user(db)
    grant(client, other, 100)
    assert client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/0/apply", headers=H(other)).status_code == 404
    assert client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(other),
                       json={"message": "show me", "request_id": uuid.uuid4().hex}).status_code == 404
    mine = project(client, other, title="Mine")["id"]
    ai.chat_result = reply()
    client.post(f"/api/v1/studio/projects/{mine}/assistant/chat", headers=H(other),
                json={"message": "How do the pictures look?", "request_id": uuid.uuid4().hex})
    assert ai.chat_images[-1] == [], "no pictures from someone else's production"
