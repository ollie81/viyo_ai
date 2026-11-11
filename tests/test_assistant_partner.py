"""The Studio assistant as a filmmaking partner: whole-project understanding
(full screenplay, long-film retrieval, series, voices, job errors), long-term
memory across conversations that keeps ideas apart from decisions and that the
creator can correct, reference resolution ("her", "the next scene"), action
selection with targets, paid confirmation, thinking depth, malformed answers,
context limits and ownership. Fake AI and local Postgres: no real AI credits."""
import os
import types
import uuid

import pytest

import creator_studio as cs
import studio_ai
import studio_chat
import studio_memory
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import DAN, MAYA, TALK, detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the assistant memory SQL")

QUIET = {"heading": "INT. HOSPITAL - DAY", "visual": "Maya waits in the corridor", "duration_seconds": 4, "dialogue": []}
OUTSIDE = {"heading": "EXT. STREET - NIGHT", "visual": "Daniel walks home alone", "duration_seconds": 4, "dialogue": []}


@pytest.fixture
def partner(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    u = make_user(db)
    grant(client, u, 2000)
    p = project(client, u, title="The Return", production_style="pictures")
    split(client, ai, gen, u, p["id"], [TALK, QUIET, OUTSIDE], characters=(MAYA, DAN))
    return client, db, store, ai, gen, u, p["id"]


def say(client, u, pid, text, conv=None):
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": text, "request_id": uuid.uuid4().hex, "conversation_id": conv})
    assert r.status_code == 200, r.text
    return r.json()


def reply(text="Noted.", proposals=(), memory=(), focus=None):
    return {"reply": text, "proposals": list(proposals), "memory": list(memory), **({"focus": focus} if focus else {})}


# ------------------------------------------------------------------ understanding

def test_a_short_screenplay_is_given_whole(partner):
    client, db, store, ai, gen, u, pid = partner
    script = "EXT. ROOFTOP - NIGHT\n\nMaya waits.\n\n" + "INT. FLAT - DAY\n\nThe ending: Maya forgives Daniel.\n"
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": script})
    ai.chat_result = reply()
    say(client, u, pid, "How does it end?")
    prompt = ai.chat_prompts[-1]
    assert "Screenplay (complete," in prompt and "The ending: Maya forgives Daniel." in prompt


def long_script(n=80):
    parts = []
    for i in range(1, n + 1):
        place = "INT. HOSPITAL - DAY" if i == 57 else f"INT. ROOM {i} - NIGHT"
        line = "Maya finds the ZEPHYR letter in the hospital drawer." if i == 57 else f"Filler action number {i}. " * 20
        parts.append(f"{place}\n\n{line}\n\nMAYA\nLine {i}.\n")
    return "\n".join(parts)


def test_a_long_film_shows_an_index_of_every_section_and_the_relevant_ones_in_full():
    script = long_script()
    assert len(script) > studio_chat.SCRIPT_CHARS
    block = studio_chat.screenplay_block(script, "What happens with the ZEPHYR letter at the hospital?")
    assert "Index of every section" in block and "#80 INT. ROOM 80 - NIGHT" in block and "#1 INT. ROOM 1" in block
    assert "Maya finds the ZEPHYR letter in the hospital drawer." in block, "the section the question is about"
    assert "Full text shown below for sections" in block and "#57" in block.split("Full text shown below")[1]
    assert "don't guess" in block
    assert len(block) <= studio_chat.SCRIPT_CHARS + 2000
    # A scene the creator names by number is shown in full too.
    named = studio_chat.screenplay_block(script, "fix it", focus_numbers={23})
    assert "Filler action number 23." in named.split("Full text shown below")[1]


def test_the_assistant_sees_the_whole_production(partner):
    client, db, store, ai, gen, u, pid = partner
    db.execute("update public.studio_projects set story_bible = %s where id = %s",
               (studio_ai.json.dumps({"logline": "A nurse finds her brother.", "central_conflict": "Old guilt",
                                      "characters": [{"name": "Maya", "role": "lead", "want": "forgiveness",
                                                      "relationships": "Daniel's younger sister"}],
                                      "beats": [{"purpose": "hook", "what_happens": "Maya returns"}]}), pid))
    db.execute("update public.studio_characters set voice = 'Kore', voice_description = 'warm, low' "
               "where project_id = %s and name = 'Maya'", (pid,))
    ai.fail["image"] = studio_ai.ProviderError("the picture service is down")
    s1 = detail(client, u, pid)["scenes"][0]
    start(client, u, pid, "image", [s1["id"]])
    run(gen)
    ai.chat_result = reply()
    say(client, u, pid, "Why did the picture fail?")
    prompt = ai.chat_prompts[-1]
    assert "relationships: Daniel's younger sister" in prompt and "Beat 1 (hook): Maya returns" in prompt
    assert "voice Kore (warm, low)" in prompt
    assert "image scene 1: failed" in prompt
    assert "PRODUCTION STATUS: 3 scenes" in prompt and "Still missing:" in prompt
    assert "You came back." in prompt, "scene dialogue is in the context"


def test_other_episodes_of_the_series_are_known_but_never_someone_elses(partner):
    client, db, store, ai, gen, u, pid = partner
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u))
    if ep2.status_code == 404:
        pytest.skip("no next-episode route")
    assert ep2.status_code == 200, ep2.text
    new = ep2.json().get("project", ep2.json())
    series = db.execute("select series_id from public.studio_projects where id = %s", (pid,)).fetchone()[0]
    other = make_user(db)
    grant(client, other, 100)
    op = project(client, other, title="SECRET OTHER SHOW")
    db.execute("update public.studio_projects set series_id = %s where id = %s", (series, op["id"]))
    ai.chat_result = reply()
    say(client, u, new["id"], "What happened in the previous episode?")
    prompt = ai.chat_prompts[-1]
    assert "Other episodes of this series" in prompt and "Episode 1: The Return" in prompt
    assert "SECRET OTHER SHOW" not in prompt


def test_the_project_facts_have_a_size_limit_and_say_when_they_are_cut(monkeypatch):
    monkeypatch.setattr(studio_chat, "CONTEXT_CHARS", 500)
    p = {"title": "T", "logline": "L" * 2000, "kind": "film", "script": ""}
    text = studio_chat.project_context(p, [], [], [], [], None, [])
    assert len(text) < 700 and "cut here to fit" in text


# ------------------------------------------------------------------ memory

def test_memory_is_shared_by_every_conversation_about_the_production(partner):
    client, db, store, ai, gen, u, pid = partner
    ai.chat_result = reply(memory=[{"category": "characters", "text": "Maya is Daniel's younger sister",
                                    "from_creator": True},
                                   {"category": "preferences", "text": "No happy ending", "from_creator": True}])
    say(client, u, pid, "Maya is Daniel's younger sister, and I don't want a happy ending.")
    new = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat/new", headers=H(u)).json()["conversation"]["id"]
    ai.chat_result = reply()
    say(client, u, pid, "Remind me who Maya is.", conv=new)
    mem = ai.chat_prompts[-1].split("MEMORY (this production")[1].split("EARLIER CONVERSATION")[0]
    assert "Maya is Daniel's younger sister" in mem and "No happy ending" in mem


def test_the_assistants_own_ideas_are_never_stored_as_decisions(partner):
    client, db, store, ai, gen, u, pid = partner
    ai.chat_result = reply("What if Daniel is the villain?",
                           memory=[{"category": "decisions", "text": "Daniel is the villain", "from_creator": False}])
    say(client, u, pid, "Any twist ideas?")
    mem = client.get(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u)).json()["memory"]
    assert mem["decisions"] == [] and mem["ideas"][0]["text"] == "Daniel is the villain"
    ai.chat_result = reply()
    say(client, u, pid, "ok")
    assert "Ideas not approved yet (NOT approved: only ideas):\n- Daniel is the villain" in ai.chat_prompts[-1]
    # When the creator decides it, it moves from ideas to decisions.
    ai.chat_result = reply(memory=[{"category": "decisions", "text": "Daniel is the villain", "from_creator": True}])
    say(client, u, pid, "Yes, Daniel is the villain.")
    mem = client.get(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u)).json()["memory"]
    assert [x["text"] for x in mem["decisions"]] == ["Daniel is the villain"] and mem["ideas"] == []


def test_applied_proposals_become_approved_decisions(partner):
    client, db, store, ai, gen, u, pid = partner
    ai.chat_result = reply("Add Ama?", proposals=[{"type": "add_character", "title": "Add character: Ama",
                                                   "summary": "Maya's sister.", "data": {"name": "Ama", "description": "20s"}}])
    mid = say(client, u, pid, "Add a sister.")["reply"]["id"]
    assert client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/0/apply", headers=H(u)).status_code == 200
    mem = client.get(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u)).json()["memory"]
    assert mem["decisions"][0]["text"].startswith("Applied: Add character: Ama") and mem["decisions"][0]["source"] == "applied"


def test_the_creator_can_correct_and_delete_memory_and_nobody_else_can(partner):
    client, db, store, ai, gen, u, pid = partner
    url = f"/api/v1/studio/projects/{pid}/assistant/memory"
    r = client.post(url, headers=H(u), json={"category": "story", "text": "It is set in Kigali"})
    item = r.json()["memory"]["story"][0]
    assert client.post(url, headers=H(u), json={"category": "story", "text": "It is set in Kigali"}).status_code == 409
    fixed = client.patch(f"{url}/{item['id']}", headers=H(u), json={"text": "It is set in Lagos"}).json()["memory"]
    assert [x["text"] for x in fixed["story"]] == ["It is set in Lagos"]
    other = make_user(db)
    assert client.get(url, headers=H(other)).status_code == 404
    assert client.patch(f"{url}/{item['id']}", headers=H(other), json={"text": "x"}).status_code == 404
    assert client.delete(f"{url}/{item['id']}", headers=H(other)).status_code == 404
    assert client.delete(f"{url}/{item['id']}", headers=H(u)).json()["memory"]["story"] == []
    assert client.delete(f"{url}/nope", headers=H(u)).status_code == 404


def test_memory_rules_unit():
    m, _ = studio_memory.add({}, "questions", "Who raised Daniel?")
    m, _ = studio_memory.add(m, "story", "Who raised Daniel?")
    assert m["questions"] == [] and m["story"][0]["text"] == "Who raised Daniel?", "answered questions leave the list"
    m, added = studio_memory.apply_updates(m, [{"category": "nonsense", "text": "x"}, "bad", {"category": "story", "text": ""}])
    assert added == []
    assert studio_memory.count(m) == 1


def test_summaries_are_told_to_keep_names_decisions_and_constraints():
    p = studio_chat.summary_prompt("", [{"role": "user", "content": "Maya is 30. Decided: no happy ending."}])
    assert "every character, place and episode name" in p and "DECIDED:" in p and "IDEA:" in p


# ------------------------------------------------------------------ conversation

def test_her_and_the_next_scene_are_understood_from_the_conversation(partner):
    client, db, store, ai, gen, u, pid = partner
    ai.chat_result = reply("Scene 1 is the rooftop.", focus={"scenes": [1], "characters": ["Maya"]})
    conv = say(client, u, pid, "What happens in scene 1?")["conversation_id"]
    ai.chat_result = reply()
    say(client, u, pid, "Make her more mysterious.", conv=conv)
    assert "CURRENT FOCUS (what 'that scene', 'her', 'him', 'the next scene' refer to): scene 1; Maya" in ai.chat_prompts[-1]
    say(client, u, pid, "What happens in the next scene?", conv=conv)
    assert "CURRENT FOCUS (what 'that scene', 'her', 'him', 'the next scene' refer to): scene 2" in ai.chat_prompts[-1]


def test_focus_resolution_unit():
    names = ["Maya", "Daniel Vance"]
    assert studio_chat.resolve_focus("Why does he look different?", {"scenes": [3], "characters": ["Daniel Vance"]}, 5,
                                     names) == {"scenes": [3], "characters": ["Daniel Vance"]}
    assert studio_chat.resolve_focus("The second scene doesn't connect to the first", {}, 5, names)["scenes"] == []
    assert studio_chat.resolve_focus("scene 2 doesn't connect to scene 1", {}, 5, names)["scenes"] == [1, 2]
    assert studio_chat.resolve_focus("Daniel should be angrier", {}, 5, names)["characters"] == ["Daniel Vance"]


def test_the_thinking_depth_and_the_credit_hold_follow_the_question(partner):
    client, db, store, ai, gen, u, pid = partner
    ai.chat_result = reply()
    assert say(client, u, pid, "hi")["mode"] == "quick"
    assert say(client, u, pid, "Make her more mysterious")["mode"] == "story"
    r = say(client, u, pid, "Audit the continuity of every scene and tell me what is still missing.")
    assert r["mode"] == "deep" and ai.chat_modes[-3:] == ["quick", "story", "deep"]
    assert r["reply"]["credits"] <= cs.CHAT_HOLDS["deep"]
    assert "This needs careful analysis of the whole production" in ai.chat_prompts[-1]


def test_suggested_questions_and_status_follow_the_production(partner):
    client, db, store, ai, gen, u, pid = partner
    h = client.get(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u)).json()
    assert "What is still missing before this episode is finished?" in h["suggestions"]
    assert h["status"]["scenes"] == 3 and h["status"]["pictures"] == 0
    assert "whole production up to 30" in h["price_note"]


# ------------------------------------------------------------------ actions

def test_actions_target_the_right_character_scene_or_object(partner):
    client, db, store, ai, gen, u, pid = partner
    d = detail(client, u, pid)
    maya = next(c for c in d["characters"] if c["name"] == "Maya")
    ai.chat_result = reply("Here's the plan.", proposals=[
        {"type": "make_character_sheet", "title": "Reference pictures of Maya", "summary": "x", "data": {"character": "Maya"}},
        {"type": "record_voices", "title": "Voice scene 1", "summary": "x", "data": {"scene": 1}},
        {"type": "make_character_sheet", "title": "Zed", "summary": "x", "data": {"character": "Zed"}}])
    props = say(client, u, pid, "What do we need next?")["reply"]["proposals"]
    assert props[0]["targets"] == [maya["id"]] and props[0]["job"] == "portrait" and props[0]["paid"]
    assert props[1]["targets"] == [d["scenes"][0]["id"]] and props[1]["job"] == "voice"
    assert props[2]["status"] == "failed", "no invented characters"


def test_paid_actions_need_the_price_check_and_nothing_is_charged(partner):
    client, db, store, ai, gen, u, pid = partner
    before = balance(client, u)["available"]
    ai.chat_result = reply("Want outfits?", proposals=[{"type": "suggest_outfits", "title": "Suggest outfits",
                                                         "summary": "From the script", "data": {}}])
    r = say(client, u, pid, "Can you suggest outfits?")
    mid = r["reply"]["id"]
    res = client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/0/apply", headers=H(u))
    assert res.status_code == 409 and res.json()["detail"]["code"] == "needs_price"
    assert balance(client, u)["available"] == before - r["reply"]["credits"], "only the chat message itself"
    assert not db.execute("select count(*) from public.studio_jobs where kind = 'outfits'").fetchone()[0]


def test_free_edits_to_places_and_objects_apply_only_when_approved(partner):
    client, db, store, ai, gen, u, pid = partner
    client.post(f"/api/v1/studio/projects/{pid}/elements", headers=H(u),
                json={"kind": "prop", "name": "Black bag", "description": "plain"})
    ai.chat_result = reply("Make the bag distinctive?", proposals=[
        {"type": "update_place_object", "title": "Black bag", "summary": "Gold clasp",
         "data": {"name": "Black bag", "description": "black leather bag with a gold clasp"}}])
    mid = say(client, u, pid, "The bag looks plain.")["reply"]["id"]
    assert detail(client, u, pid)["elements"][0]["description"] == "plain", "not before Apply"
    assert client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/0/apply", headers=H(u)).status_code == 200
    assert detail(client, u, pid)["elements"][0]["description"] == "black leather bag with a gold clasp"


def test_failed_generation_is_refunded(partner):
    client, db, store, ai, gen, u, pid = partner
    before = balance(client, u)["available"]
    ai.fail["image"] = studio_ai.ProviderError("down")
    s = detail(client, u, pid)["scenes"][0]
    start(client, u, pid, "image", [s["id"]])
    run(gen)
    assert balance(client, u)["available"] == before


# ------------------------------------------------------------------ Gemini replies

def test_an_incomplete_answer_is_not_charged(partner):
    client, db, store, ai, gen, u, pid = partner
    before = balance(client, u)["available"]
    ai.chat_result = {"proposals": [], "memory": []}          # no reply text
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": "hello", "request_id": uuid.uuid4().hex})
    assert r.status_code == 502 and balance(client, u)["available"] == before


class FakeModels:
    def __init__(self, responses):
        self.responses, self.configs = list(responses), []

    def generate_content(self, model, contents, config):
        self.configs.append(config)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return types.SimpleNamespace(text=r, usage_metadata=None)


def google_with(responses):
    ai = object.__new__(studio_ai.GoogleStudioAI)
    ai.client = types.SimpleNamespace(models=FakeModels(responses))
    return ai


def test_gemini_thinking_level_with_a_budget_fallback():
    ai = google_with(['{"reply": "hi", "proposals": [], "memory": []}'])
    ai.chat("p", mode="deep")
    assert str(ai.client.models.configs[0].thinking_config.thinking_level).endswith("HIGH")
    ai = google_with([Exception("400 INVALID_ARGUMENT: thinking_level is not supported for this model"),
                      '{"reply": "hi", "proposals": [], "memory": []}'])
    data, _cost = ai.chat("p", mode="story")
    cfgs = ai.client.models.configs
    assert data["reply"] == "hi" and cfgs[1].thinking_config.thinking_budget == 4096


def test_malformed_gemini_answers_are_errors_not_replies():
    for bad in ["not json", "[1, 2]", '{"reply": "   "}']:
        with pytest.raises(studio_ai.ProviderError):
            google_with([bad]).chat("p")
    with pytest.raises(studio_ai.ProviderBusy):
        google_with([Exception("429 RESOURCE_EXHAUSTED")]).chat("p")


def test_the_thinking_level_can_be_set_per_mode(monkeypatch):
    monkeypatch.setenv("STUDIO_CHAT_THINKING_QUICK", "minimal")
    assert studio_ai.chat_thinking("quick") == ("MINIMAL", 0)
    assert studio_ai.chat_thinking("deep") == ("HIGH", 12288)
