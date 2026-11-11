"""Episode storytelling: opening and ending hooks chosen for the story and the
genre, suspense planned as questions raised and answered, a story check of the
FINAL script with quoted evidence (and at most one targeted revision), and an
episode record so each episode continues from the previous one's actual ending
with the same secrets, knowledge and relationships, without repeating a
cliffhanger. Plus the assistant as a critic of hooks.

A scripted writers' room drives the REAL studio_story pipeline (plan, checks,
draft, doctor, evaluation, revision). These tests prove what the system asks
for, checks, remembers and passes on; they do NOT prove the real model writes
gripping episodes. scripts/eval_story.py does that against real Gemini, opt-in.
"""
import json
import os
import re
import uuid

import pytest

import creator_studio as cs
import studio_story as story
from conftest import make_user
from test_creator_studio import H, env, grant, run, start, studio_schema  # noqa: F401

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the story SQL")

CAST = [{"name": "Ama", "age": "29", "role": "lead", "want": "find her brother", "flaw": "reckless", "speech": "short",
         "look": "mustard raincoat"},
        {"name": "Kofi", "age": "24", "role": "supporting", "want": "stay hidden", "flaw": "afraid", "speech": "quiet",
         "look": "cut above his eyebrow"},
        {"name": "Grace", "age": "50s", "role": "supporting", "want": "protect her secret", "flaw": "cold", "speech": "formal",
         "look": "navy scrubs"}]


def beats(open_what, end_what):
    return [{"purpose": "hook", "what_happens": open_what, "emotional_turn": "calm -> shock", "seconds": 6},
            {"purpose": "escalation", "what_happens": "the search tightens", "emotional_turn": "hope -> fear", "seconds": 20},
            {"purpose": "reversal", "what_happens": "a lie is exposed", "emotional_turn": "trust -> doubt", "seconds": 20},
            {"purpose": "end_hook", "what_happens": end_what, "emotional_turn": "relief -> dread", "seconds": 8}]


# What the (scripted) writers plan for each episode. Episode 3's first plan
# repeats episode 2's kind of cliffhanger, to test that it is sent back.
PLANS = {
    1: {"title": "False Name", "logline": "Ama finds her missing brother admitted under a false name.",
        "central_conflict": "Ama vs the people hiding Kofi", "stakes": "Kofi's life", "ending": "Grace says Kofi is dead.",
        "characters": CAST, "beats": beats("A photo of Kofi in a hospital bed arrives from an unknown number",
                                           "Grace: 'Kofi Mensah died on Tuesday.'"),
        "hook_plan": {"opening_technique": "shocking_discovery", "opening_moment": "Ama's phone: a photo of Kofi in a hospital bed",
                      "opening_why": "her missing brother, alive, sent by a stranger", "ending_technique": "disturbing_news",
                      "ending_moment": "Grace: 'Kofi Mensah died on Tuesday. So who are you?'",
                      "viewer_question": "Why is Kofi officially dead?"},
        "questions": [{"question": "Who sent the photo?", "raised_in_beat": 1, "answered_in_beat": 0},
                      {"question": "Is it really Kofi?", "raised_in_beat": 1, "answered_in_beat": 3},
                      {"question": "Why is Kofi officially dead?", "raised_in_beat": 4, "answered_in_beat": 0}],
        "knowledge": [{"character": "Ama", "knows": "Kofi is alive under the name Joseph Owusu",
                       "does_not_know": "who gave him the false name"},
                      {"character": "Grace", "knows": "who brought Kofi in", "hides": "she signed the death record"}],
        "relationships": [{"between": "Ama and Grace", "state": "enemies", "change": "from strangers"}],
        "revelations": ["Kofi is alive", "he is registered as Joseph Owusu"],
        "next_episode": ["Ama learns why Kofi is officially dead"]},
    2: {"title": "The Record", "logline": "Ama digs into Kofi's death record.", "central_conflict": "Ama vs Grace",
        "stakes": "Kofi's freedom", "ending": "Kofi's 'killer' walks in.", "characters": CAST,
        "beats": beats("Ama answers Grace: 'I'm his sister, and he's breathing.'", "A man in a police uniform asks for Joseph Owusu"),
        "continues_from": "The very next moment: Ama answers Grace's question in the doorway.",
        "hook_plan": {"opening_technique": "confrontation", "opening_moment": "Ama: 'I'm his sister, and he's breathing.'",
                      "opening_why": "it answers the cliffhanger head-on", "ending_technique": "unexpected_person",
                      "ending_moment": "A policeman asks the desk for Joseph Owusu", "viewer_question": "Is the policeman here to kill him?"},
        "questions": [{"question": "Why is Kofi officially dead?", "raised_in_beat": 1, "answered_in_beat": 3},
                      {"question": "Who is the policeman?", "raised_in_beat": 4, "answered_in_beat": 0}],
        "knowledge": [{"character": "Ama", "knows": "Grace signed the death record", "does_not_know": "who ordered it"},
                      {"character": "Grace", "knows": "Ama knows about the record", "hides": "who paid her"}],
        "relationships": [{"between": "Ama and Grace", "state": "uneasy allies", "change": "Grace confesses"}],
        "revelations": ["Grace signed a false death record"], "next_episode": ["The policeman's identity"]},
    3: {"title": "Joseph Owusu", "logline": "The policeman comes for Kofi.", "central_conflict": "Ama vs the policeman",
        "stakes": "Kofi's life", "ending": "Ama makes a choice.", "characters": CAST,
        "beats": beats("The policeman's hand on the ward door", "Ama gives the policeman her own name instead of Kofi's"),
        "continues_from": "The policeman walks from the desk toward the ward.",
        "hook_plan": {"opening_technique": "threat", "opening_moment": "The policeman's hand on the ward door",
                      "opening_why": "the danger from the last moment is here", "ending_technique": "unexpected_person",
                      "ending_moment": "x", "viewer_question": "x"},
        "questions": [{"question": "Who is the policeman?", "raised_in_beat": 1, "answered_in_beat": 3},
                      {"question": "What will Ama's lie cost her?", "raised_in_beat": 4, "answered_in_beat": 0}],
        "knowledge": [{"character": "Ama", "knows": "the policeman ordered the false record"}],
        "revelations": ["the policeman ordered the false record"]},
}
FIXED_3 = {**PLANS[3], "hook_plan": {**PLANS[3]["hook_plan"], "ending_technique": "dangerous_decision",
                                     "ending_moment": "Ama tells the policeman: 'I'm Joseph Owusu's wife.'",
                                     "viewer_question": "What will Ama's lie cost her?"}}


def script_for(n):
    plan = PLANS[n]
    return (f"INT. WARD - NIGHT\n\n{plan['beats'][0]['what_happens']}.\n\nAMA\nEpisode {n} begins now, right here.\n\n"
            f"INT. CORRIDOR - NIGHT\n\n{plan['beats'][-1]['what_happens']}.\n\nGRACE\nThis is how episode {n} ends.\n")


class Room:
    """A scripted writers' room: answers each call of studio_story by kind."""

    def __init__(self, scores=None, revised_scores=None, eval_raw=None):
        self.prompts, self.revisions = [], []
        self.scores, self.revised_scores, self.eval_raw = scores, revised_scores, eval_raw
        self.ep3_plans = 0

    @staticmethod
    def episode(prompt):
        m = re.findall(r"^- Episode (\d+) \"", prompt, re.M)
        return (max(map(int, m)) + 1) if m else 1

    def text(self, prompt):
        self.prompts.append(("text", prompt))
        if prompt.startswith("Revise this screenplay"):
            self.revisions.append(prompt)
            return prompt.split("SCREENPLAY:\n", 1)[1].split("\n\nReturn only")[0].replace(
                "Hello Ama, good morning.", "Kofi is alive. Look.") + "\n", 900, 900
        return script_for(self.episode(prompt)), 1000, 900

    def json(self, prompt, schema):
        self.prompts.append(("json", prompt, schema))
        n = self.episode(prompt)
        if schema is story.PLAN_SCHEMA:
            if n == 3 and self.ep3_plans == 0:
                self.ep3_plans += 1
                return json.loads(json.dumps(PLANS[3])), 1500, 2000
            return json.loads(json.dumps(FIXED_3 if n == 3 else PLANS[n])), 1500, 2000
        if schema is story.DOCTOR_SCHEMA:
            draft = prompt.split("DRAFT:\n", 1)[1].split("\n\nNON-NEGOTIABLE")[0]
            plan = FIXED_3 if n == 3 else PLANS[n]
            return {"scores": {"hook": 6}, "fixes": ["sharper hook"], "screenplay": draft,
                    "opening_hook": plan["hook_plan"]["opening_moment"], "ending_hook": plan["hook_plan"]["ending_moment"],
                    "episode_summary": f"Episode {n}: {plan['logline']}", "open_threads": []}, 3000, 2500
        # The story check.
        if self.eval_raw is not None:
            return self.eval_raw, 2000, 800
        revised = "Revise" not in prompt and "Kofi is alive. Look." in prompt
        sc = self.revised_scores if revised and self.revised_scores else self.scores
        sc = sc or {k: 8 for k in story.CRITERIA}
        return {"scores": {k: {"score": v, "evidence": f"quoted line for {k}", "reason": f"{k} reason"} for k, v in sc.items()},
                "weak": [{"area": k, "problem": f"{k} is weak", "fix": f"fix the {k}"} for k, v in sc.items() if v < 7],
                "viewer_questions": [f"What happens after episode {n}?"]}, 2000, 800


def make_series(env, room, edit_ep1_end=None):
    client, db, store, files, ai, gen, ren = env
    ai.story_fns = (room.text, room.json)
    u = make_user(db)
    grant(client, u, 5000)
    r = client.post("/api/v1/studio/projects", headers=H(u), json={
        "title": "False Name", "logline": "A woman finds her missing brother admitted under a false name.",
        "genre": "Family thriller", "kind": "episode", "target_seconds": 60, "accept_terms": True})
    ep1 = r.json()["id"]
    start(client, u, ep1, "script")
    run(gen)
    if edit_ep1_end:
        p = client.get(f"/api/v1/studio/projects/{ep1}", headers=H(u)).json()["project"]
        client.patch(f"/api/v1/studio/projects/{ep1}", headers=H(u),
                     json={"script": p["script"].replace("This is how episode 1 ends.", edit_ep1_end)})
    ids = [ep1]
    for _ in range(2):
        nxt = client.post(f"/api/v1/studio/projects/{ids[-1]}/next-episode", headers=H(u)).json()["id"]
        client.patch(f"/api/v1/studio/projects/{nxt}", headers=H(u), json={"logline": "Ama keeps digging."})
        assert start(client, u, nxt, "script").status_code == 200
        run(gen)
        ids.append(nxt)
    bibles = [client.get(f"/api/v1/studio/projects/{i}", headers=H(u)).json()["project"]["story_bible"] for i in ids]
    return client, u, ids, bibles


def plan_prompts(room):
    return [p[1] for p in room.prompts if p[0] == "json" and p[2] is story.PLAN_SCHEMA]


# ------------------------------------------------------------------ three episodes

def test_a_three_episode_drama_has_hooks_at_both_ends_of_every_episode(env):
    room = Room()
    client, u, ids, bibles = make_series(env, room)
    for b in bibles:
        assert b["beats"][0]["purpose"] == "hook" and b["beats"][-1]["purpose"] == "end_hook"
        assert b["hook_plan"]["opening_moment"] and b["hook_plan"]["ending_moment"] and b["hook_plan"]["viewer_question"]
        assert b["evaluation"]["evaluated"] is True and set(b["evaluation"]["scores"]) >= {"opening_hook", "ending_hook"}
        assert b["episode_record"]["open_questions"], "every episode leaves the viewer a question"
    assert "continuity" not in bibles[0]["evaluation"]["scores"], "episode 1 has nothing to continue from"
    assert "continuity" in bibles[1]["evaluation"]["scores"] and "continuity" in bibles[2]["evaluation"]["scores"]


def test_episode_2_starts_from_episode_1s_actual_ending_and_episode_3_from_episode_2s(env):
    room = Room()
    client, u, ids, bibles = make_series(env, room, edit_ep1_end="Then tell me whose body you buried.")
    p1, p2, p3 = [p for p in plan_prompts(room) if "YOUR PREVIOUS PLAN" not in p]
    assert "THE PREVIOUS EPISODE ENDED" not in p1
    ended = p2.split("THE PREVIOUS EPISODE ENDED EXACTLY LIKE THIS (this episode starts from here):")[1]
    assert "Then tell me whose body you buried." in ended.split("Questions the viewer")[0], \
        "the creator's edit of episode 1's ending is what episode 2 continues from"
    assert "the first beat picks up that final moment" in p2
    assert bibles[1]["continues_from"]
    ended3 = p3.split("THE PREVIOUS EPISODE ENDED EXACTLY LIKE THIS")[1].split("Questions the viewer")[0]
    assert "This is how episode 2 ends." in ended3 and "episode 1 ends" not in ended3
    assert "Why is Kofi officially dead?" in p2.split("Questions the viewer is waiting to have answered:")[1].split("\n")[0]
    assert "The viewer already learned: Kofi is alive; he is registered as Joseph Owusu" in p3, "never re-explained"
    assert "Planned for this episode: Ama learns why Kofi is officially dead" in p2


def test_character_knowledge_and_secrets_carry_into_the_next_episode(env):
    room = Room()
    client, u, ids, bibles = make_series(env, room)
    p2 = plan_prompts(room)[1]
    assert "- Ama: knows Kofi is alive under the name Joseph Owusu; doesn't know who gave him the false name" in p2
    assert "- Grace: knows who brought Kofi in; hides she signed the death record" in p2
    assert "a character only learns more ON SCREEN" in p2
    assert "- Relationship Ama and Grace: enemies" in p2
    assert bibles[1]["episode_record"]["knowledge"][0]["knows"] == "Grace signed the death record"
    doctor2 = [p[1] for p in room.prompts if p[0] == "json" and p[2] is story.DOCTOR_SCHEMA][1]
    assert "doesn't know who gave him the false name" in doctor2, "the script doctor sees the series memory too"


def test_a_repeated_kind_of_cliffhanger_is_sent_back(env):
    room = Room()
    client, u, ids, bibles = make_series(env, room)
    third = [p for p in plan_prompts(room) if "YOUR PREVIOUS PLAN HAS THESE PROBLEMS" in p]
    assert len(third) == 1 and "already ended with 'unexpected person'" in third[0]
    assert [b["hook_plan"]["ending_technique"] for b in bibles] == ["disturbing_news", "unexpected_person", "dangerous_decision"]
    assert "Don't end this episode with the same kind of cliffhanger as before (disturbing news, unexpected person)" in third[0]


def test_older_episodes_without_a_record_still_give_their_ending():
    old = {"title": "Ep 1", "episode_summary": "Ama finds Kofi.", "ending_hook": "Grace: 'He died Tuesday.'",
           "open_threads": ["Why is he dead?"], "characters": CAST}
    text = story._continuity([old])
    assert "Grace: 'He died Tuesday.'" in text and "Why is he dead?" in text


# ------------------------------------------------------------------ the story check

def test_the_final_script_is_judged_with_evidence_and_a_weak_opening_is_revised_once(env):
    room = Room(scores={**{k: 8 for k in story.CRITERIA}, "opening_hook": 5},
                revised_scores={k: 8 for k in story.CRITERIA})
    brief = {"title": "T", "logline": "x", "kind": "episode", "target_seconds": 60, "genre": "drama"}
    weak_script = "INT. WARD - NIGHT\n\nAMA\nHello Ama, good morning.\n\nGRACE\nKofi Mensah died on Tuesday.\n"
    room.text = (lambda orig: (lambda p: (weak_script, 1000, 900) if not p.startswith("Revise") else orig(p)))(room.text)
    orig_json = room.json

    def json_fn(prompt, schema):
        if schema is story.DOCTOR_SCHEMA:
            return {"scores": {}, "fixes": [], "screenplay": weak_script, "opening_hook": "", "ending_hook": "",
                    "episode_summary": "", "open_threads": []}, 10, 10
        return orig_json(prompt, schema)
    text, bible, *_ = story.develop(room.text, json_fn, brief, [])
    ev = bible["evaluation"]
    assert len(room.revisions) == 1, "one targeted revision at most"
    rev = room.revisions[0]
    assert "opening_hook: opening_hook is weak FIX: fix the opening_hook" in rev
    assert "The first line is a greeting" in rev, "the no-AI check found the slow opening too"
    assert "Fix ONLY these weaknesses and keep everything else" in rev
    assert ev["revised"] is True and ev["before"]["opening_hook"] == 5 and ev["scores"]["opening_hook"]["score"] == 8
    assert "Kofi is alive. Look." in text and "Hello Ama" not in text
    assert all(v["evidence"] for v in ev["scores"].values()), "every score quotes what it judged"


def test_a_revision_that_is_not_better_is_thrown_away():
    room = Room(scores={**{k: 8 for k in story.CRITERIA}, "ending_hook": 5},
                revised_scores={**{k: 6 for k in story.CRITERIA}})
    room.text = (lambda orig: (lambda p: (script_for(1) + "\nAMA\nKofi is alive. Look.\n", 10, 10)
                               if p.startswith("Revise") else orig(p)))(room.text)
    text, bible, *_ = story.develop(room.text, room.json, {"title": "T", "logline": "x", "kind": "short"}, [])
    assert bible["evaluation"]["revised"] is False and "wasn't better" in bible["evaluation"]["revision_rejected"]
    assert text == script_for(1).strip()
    room = Room(scores={**{k: 8 for k in story.CRITERIA}, "ending_hook": 5})
    room.text = (lambda orig: (lambda p: ("INT. A\n", 1, 1) if p.startswith("Revise") else orig(p)))(room.text)
    text, bible, *_ = story.develop(room.text, room.json, {"title": "T", "logline": "x", "kind": "short"}, [])
    assert "cut off" in bible["evaluation"]["revision_rejected"] and text == script_for(1).strip()
    assert bible["evaluation"]["below_bar"] == ["ending_hook"], "still weak, and it says so"


def test_scores_without_evidence_are_never_shown_as_real():
    raw = {"scores": {"opening_hook": {"score": 10, "evidence": "", "reason": "great"},
                      "ending_hook": {"score": 9, "evidence": "GRACE: 'So who are you?'", "reason": "open question"}},
           "weak": [], "viewer_questions": []}
    ev = story.clean_evaluation(raw, has_previous=False)
    assert list(ev["scores"]) == ["ending_hook"], "an unquoted 10 is dropped, not believed"
    room = Room(eval_raw={"nonsense": True})
    text, bible, *_ = story.develop(room.text, room.json, {"title": "T", "logline": "x", "kind": "short"}, [])
    assert bible["evaluation"]["evaluated"] is False and bible["evaluation"]["scores"] == {}
    assert room.revisions == [], "no revision without a real evaluation"


def test_repeated_lines_and_a_fade_out_ending_are_caught_without_ai():
    s = ("INT. A - NIGHT\n\nAMA\nI will find him tonight.\n\nKOFI\nNo.\n\nAMA\nI will find him tonight.\n\nFADE OUT.")
    areas = [c["area"] for c in story.structure_checks(s, {"kind": "episode"}, [])]
    assert areas == ["dialogue", "ending_hook"]


# ------------------------------------------------------------------ genres

def test_each_genre_gets_its_own_techniques():
    th = story.genre_craft("Thriller")
    ro = story.genre_craft("Romance")
    my = story.genre_craft("", "A detective investigates a mystery at sea")
    assert "ticking clock" in th and "threat arriving" in th
    assert "relationship shift" in ro and "ticking clock" not in ro
    assert "mystery deepening" in my and "Plant clues" in my
    assert story.genre_craft("") .startswith("GENRE CRAFT (drama)")
    prompt = story.plan_prompt({"title": "T", "logline": "x", "genre": "Romance", "kind": "episode", "target_seconds": 60}, [])
    assert "GENRE CRAFT (romance)" in prompt and "don't force a random shock" in prompt
    assert "No greetings" in prompt and "raised_in_beat / answered_in_beat" in prompt


# ------------------------------------------------------------------ the assistant as a critic

def test_the_assistant_critiques_hooks_from_the_story_check_and_the_previous_episode(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    room = Room(scores={**{k: 8 for k in story.CRITERIA}, "ending_hook": 6},
                revised_scores={**{k: 8 for k in story.CRITERIA}, "ending_hook": 6})
    client, u, ids, bibles = make_series(env, room)
    monkeypatch.setattr(cs, "_ai", ai)
    ai.chat_result = {"reply": "The ending lands at 6/10: the policeman is a surprise, but...", "memory": [], "proposals": [
        {"type": "write_screenplay", "title": "Stronger ending", "summary": "x", "data": {"script": "INT. X\n\nNew ending.\n"}}]}
    r = client.post(f"/api/v1/studio/projects/{ids[1]}/assistant/chat", headers=H(u),
                    json={"message": "Is my ending strong enough to make people watch episode 3?", "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    prompt = ai.chat_prompts[-1]
    assert ai.chat_modes[-1] == "story"
    assert "Ending hook (unexpected person): A policeman asks the desk for Joseph Owusu" in prompt
    assert "STORY CHECK of the final script" in prompt and "ending hook 6" in prompt
    assert "Still below the bar: ending_hook" in prompt
    assert "It ended on (disturbing news): Grace: 'Kofi Mensah died on Tuesday. So who are you?'" in prompt
    assert "Left open:" in prompt and "Picks up the previous episode:" in prompt
    assert "Episode storytelling (hooks and cliffhangers)" in prompt and "never invent scores" in prompt
    script = client.get(f"/api/v1/studio/projects/{ids[1]}", headers=H(u)).json()["project"]["script"]
    assert "New ending." not in script, "a rewrite is a proposal until the creator applies it"
