"""The Studio assistant chat: owner-scoped conversations, persistence and
memory, project facts as context, proposals applied only on approval
through the Studio's own code, honest charging. Fake AI and local Postgres."""
import os
import uuid

import pytest

import creator_studio as cs
import studio_ai
import studio_chat
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import MAYA, TALK, detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_chat.sql")


@pytest.fixture
def chat(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    u = make_user(db)
    grant(client, u, 500)
    pid = project(client, u, title="Untitled idea")["id"]
    return client, db, ai, gen, u, pid


def say(client, u, pid, text, rid=None, conv=None):
    return client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                       json={"message": text, "request_id": rid or uuid.uuid4().hex, "conversation_id": conv})


def history(client, u, pid, conv=None):
    q = f"?conversation_id={conv}" if conv else ""
    r = client.get(f"/api/v1/studio/projects/{pid}/assistant/chat{q}", headers=H(u))
    assert r.status_code == 200, r.text
    return r.json()


def test_a_one_sentence_idea_starts_a_saved_conversation(chat):
    client, db, ai, gen, u, pid = chat
    empty = history(client, u, pid)
    assert empty["conversation"] is None and empty["messages"] == [] and len(empty["suggestions"]) >= 3
    assert "credits" in empty["price_note"]
    before = balance(client, u)["available"]
    r = say(client, u, pid, "A poor young man discovers he is the son of a billionaire.")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["message"]["role"] == "user" and out["reply"]["role"] == "assistant"
    assert "One question" in out["reply"]["content"]
    prop = out["reply"]["proposals"][0]
    assert prop["type"] == "update_story" and prop["status"] == "proposed" and not prop["paid"]
    spent = before - balance(client, u)["available"]
    assert 1 <= spent <= cs.CHAT_HOLD and out["reply"]["credits"] == spent and balance(client, u)["held"] == 0
    h = history(client, u, pid)
    assert [m["role"] for m in h["messages"]] == ["user", "assistant"] and h["conversation"]["id"] == out["conversation_id"]
    # Nothing was changed by the suggestion alone.
    assert detail(client, u, pid)["project"]["title"] == "Untitled idea"


def test_the_assistant_sees_the_real_project_and_never_the_system_prompt_in_replies(chat):
    client, db, ai, gen, u, pid = chat
    split(client, ai, gen, u, pid, [TALK], characters=(MAYA,))
    say(client, u, pid, "Change Maya's outfit in the next scene")
    prompt = ai.chat_prompts[-1]
    assert "PROJECT (saved facts):" in prompt and "- Maya:" in prompt and "1. EXT. ROOFTOP - NIGHT" in prompt
    assert "Change Maya's outfit in the next scene" in prompt and "Never say something \"was saved\"" in prompt
    assert "BE SHORT" in prompt and "under 80 words" in prompt
    h = history(client, u, pid)
    assert all("Truth rules" not in m["content"] for m in h["messages"])


def test_memory_uses_a_summary_and_recent_messages_not_the_whole_history(chat, monkeypatch):
    client, db, ai, gen, u, pid = chat
    monkeypatch.setattr(studio_chat, "SUMMARIZE_AFTER", 6)
    monkeypatch.setattr(studio_chat, "RECENT", 4)
    ai.chat_result = lambda p: {"reply": "Noted.", "proposals": [],
                                "decisions": ["Make it an action movie"] if "action movie" in p.split("CREATOR: ")[-1] else []}
    for i in range(6):
        assert say(client, u, pid, f"message number {i}" + (" I want this to be an action movie" if i == 1 else "")).status_code == 200
    last = ai.chat_prompts[-1]
    assert "Summary: the creator is developing" in last, "older messages folded into the summary"
    assert "message number 0" not in last.split("RECENT MESSAGES:")[1], "not resent word for word"
    assert "- Make it an action movie" in last.split("DECISIONS the creator made earlier:")[1].split("EARLIER CONVERSATION")[0]
    assert ("summarize",) in ai.calls


def test_applying_proposals_runs_the_real_studio_actions(chat):
    client, db, ai, gen, u, pid = chat
    split(client, ai, gen, u, pid, [TALK], characters=(MAYA,))
    ai.chat_result = {"reply": "Here are some changes.", "decisions": [], "proposals": [
        {"type": "add_character", "title": "Add character: Ama", "summary": "Maya's sister.",
         "data": {"name": "Ama", "description": "Maya's older sister, 30s, sharp and protective"}},
        {"type": "add_outfit", "title": "Hospital scrubs for Maya", "summary": "For the ward scenes.",
         "data": {"character": "maya", "name": "Scrubs", "clothing": "blue hospital scrubs", "shoes": "white clogs"}},
        {"type": "rewrite_scene", "title": "Sharper opening", "summary": "Start mid-argument.",
         "data": {"scene": 1, "visual": "Maya slams the rooftop door, rain pouring",
                  "dialogue": [{"character": "Maya", "text": "You lied to me.", "emotion": "angry"}]}},
    ]}
    reply = say(client, u, pid, "Make it stronger").json()["reply"]
    mid = reply["id"]
    for i in range(3):
        r = client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/{i}/apply", headers=H(u))
        assert r.status_code == 200, r.text
        assert r.json()["proposal"]["status"] == "applied" and r.json()["result"]
    d = detail(client, u, pid)
    assert "Ama" in [c["name"] for c in d["characters"]]
    maya = next(c for c in d["characters"] if c["name"] == "Maya")
    assert "Scrubs" in [o["name"] for o in maya["outfits"]]
    assert d["scenes"][0]["visual"].startswith("Maya slams") and d["scenes"][0]["dialogue"][0]["text"] == "You lied to me."
    # Applied once only; the decision is remembered as approved.
    assert client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/0/apply", headers=H(u)).status_code == 409
    assert any(m["proposals"] and m["proposals"][0]["status"] == "applied" for m in history(client, u, pid)["messages"])
    conv = db.execute("select memory from public.studio_conversations where project_id = %s", (pid,)).fetchone()[0]
    assert any(d.startswith("Applied: Add character: Ama") for d in conv["decisions"])


def test_a_proposal_that_names_something_missing_is_refused_honestly(chat):
    client, db, ai, gen, u, pid = chat
    ai.chat_result = {"reply": "Try this.", "decisions": [], "proposals": [
        {"type": "update_character", "title": "Update Zed", "summary": "x", "data": {"character": "Zed", "description": "y"}},
        {"type": "rewrite_scene", "title": "Scene 9", "summary": "x", "data": {"scene": 9, "visual": "z"}}]}
    mid = say(client, u, pid, "hi").json()["reply"]["id"]
    r = client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/0/apply", headers=H(u))
    assert r.status_code == 409 and "no character called" in r.json()["detail"]
    r = client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/1/apply", headers=H(u))
    assert r.status_code == 409 and "no scene 9" in r.json()["detail"]
    p = history(client, u, pid)["messages"][-1]["proposals"]
    assert p[0]["status"] == "proposed" and "no character" in p[0]["error"], "never marked as done"


def test_paid_steps_go_through_the_price_check(chat):
    client, db, ai, gen, u, pid = chat
    ai.chat_result = {"reply": "Ready to write it?", "decisions": [], "proposals": [
        {"type": "develop_story", "title": "Write the full story", "summary": "The writers' room writes the screenplay.", "data": {}}]}
    reply = say(client, u, pid, "write it").json()["reply"]
    assert reply["proposals"][0]["paid"] and reply["proposals"][0]["job"] == "script"
    r = client.post(f"/api/v1/studio/assistant/messages/{reply['id']}/proposals/0/apply", headers=H(u))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "needs_price"
    r = client.post(f"/api/v1/studio/assistant/messages/{reply['id']}/proposals/0/status", headers=H(u),
                    json={"status": "started"})
    assert r.status_code == 200 and r.json()["proposal"]["status"] == "started"


def test_failed_answers_are_free_and_a_retry_never_sends_twice(chat):
    client, db, ai, gen, u, pid = chat
    before = balance(client, u)["available"]
    ai.fail["chat"] = [studio_ai.ProviderError("down")]
    rid = uuid.uuid4().hex
    r = say(client, u, pid, "hello", rid=rid)
    assert r.status_code == 502 and "weren't charged" in r.json()["detail"]["message"]
    assert balance(client, u)["available"] == before and balance(client, u)["held"] == 0
    r = say(client, u, pid, "hello", rid=rid)
    assert r.status_code == 200
    again = say(client, u, pid, "hello", rid=rid)
    assert again.json()["reply"]["id"] == r.json()["reply"]["id"]
    roles = [m["role"] for m in history(client, u, pid)["messages"]]
    assert roles == ["user", "assistant"], "one question, one answer, one charge"


def test_conversations_are_private_and_per_project(chat):
    client, db, ai, gen, u, pid = chat
    out = say(client, u, pid, "my secret idea").json()
    other = make_user(db)
    grant(client, other, 100)
    assert client.get(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(other)).status_code == 404
    assert say(client, other, pid, "hi").status_code == 404
    assert client.post(f"/api/v1/studio/assistant/messages/{out['reply']['id']}/proposals/0/apply",
                       headers=H(other)).status_code == 404
    mine2 = project(client, u, title="Second")["id"]
    assert say(client, u, mine2, "hi", conv=out["conversation_id"]).status_code == 404, "a chat stays with its project"
    assert history(client, u, mine2)["messages"] == []


def test_a_new_chat_starts_fresh_and_old_ones_stay(chat):
    client, db, ai, gen, u, pid = chat
    first = say(client, u, pid, "first idea").json()["conversation_id"]
    new = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat/new", headers=H(u)).json()["conversation"]["id"]
    say(client, u, pid, "second idea", conv=new)
    h = history(client, u, pid)
    assert len(h["conversations"]) == 2
    assert [m["content"] for m in history(client, u, pid, first)["messages"] if m["role"] == "user"] == ["first idea"]


def test_not_enough_credits_is_clear(chat):
    client, db, ai, gen, u, pid = chat
    poor = make_user(db)
    pp = project(client, poor)["id"]
    r = say(client, poor, pp, "hi")
    assert r.status_code == 402 and ("chat",) not in ai.calls
