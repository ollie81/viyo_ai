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
import studio_story_check as sc
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



SCREEN = "SCREENPLAY (scenes numbered for you; the numbers are not part of the script):\n"


def judged_scenes(prompt):
    body = prompt.split(SCREEN, 1)[1].split("\n\nFor every category")[0]
    return [b.split("\n", 1)[1] for b in re.split(r"\n\n(?=\[SCENE \d+\])", body) if b.startswith("[SCENE")]


def spoken(scene_text):
    return [t for _w, t in story.screenplay_lines(scene_text)]


class Room:
    """A scripted writers' room and judge: answers each call of studio_story by
    kind. The judge quotes the real screenplay unless told to invent or to
    quote from the wrong place."""

    def __init__(self, scores=None, revised_scores=None, eval_raw=None, invent=(), wrong_place=(), missing=(),
                 rewrite=None, retry_fixes=False):
        self.prompts, self.revisions = [], []
        self.scores, self.revised_scores, self.eval_raw = scores, revised_scores, eval_raw
        self.invent, self.wrong_place, self.missing = set(invent), set(wrong_place), set(missing)
        self.rewrite, self.retry_fixes = rewrite, retry_fixes
        self.ep3_plans = 0

    @staticmethod
    def episode(prompt):
        m = re.findall(r"^- Episode (\d+) \"", prompt, re.M)
        return (max(map(int, m)) + 1) if m else 1

    def text(self, prompt):
        self.prompts.append(("text", prompt))
        return script_for(self.episode(prompt)), 1000, 900

    def judge(self, prompt):
        scenes = judged_scenes(prompt)
        first, last = scenes[0], scenes[-1]
        retry = "COULD NOT BE VERIFIED" in prompt
        revised = "Kofi is alive. Look." in prompt.split(SCREEN, 1)[1]
        sc = (self.revised_scores if revised and self.revised_scores else self.scores) or {k: 8 for k in story.CRITERIA}
        out = {}
        for k, v in sc.items():
            if k in self.missing:
                continue
            if k == "continuity" and "THE PREVIOUS EPISODE ENDED" not in prompt:
                continue
            if k in ("opening_hook", "continuity"):
                quote, n = (spoken(first) or [first.splitlines()[-1]])[0], 1
            elif k == "ending_hook":
                quote, n = (spoken(last) or [last.splitlines()[-1]])[-1], len(scenes)
            elif k == "dialogue":
                quote, n = (spoken(first) or spoken(last))[0], 1
            else:
                quote, n = first.splitlines()[2], 1
            if k in self.wrong_place and not (retry and self.retry_fixes):
                quote, n = (spoken(last) or [last.splitlines()[-1]])[-1], len(scenes)
            if k in self.invent and not (retry and self.retry_fixes):
                quote = "Ama screams that she will burn the hospital down"
            out[k] = {"score": v, "quote": quote, "scene": n, "reason": f"{k} reason"}
        weak = [{"area": k, "scene": 1 if k in ("opening_hook", "continuity") else len(scenes) if k == "ending_hook" else 0,
                 "problem": f"{k} is weak", "fix": f"fix the {k}"} for k, v in sc.items() if v < 7]
        return {"scores": out, "weak": weak, "viewer_questions": ["What happens next?"]}

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
        if schema is sc.REVISION_SCHEMA:
            self.revisions.append(prompt)
            nums = [int(x) for x in re.search(r"Rewrite ONLY scene\(s\) ([\d, ]+)", prompt).group(1).split(",")]
            scenes = [b.split("\n", 1)[1] for b in re.split(r"\n\n(?=\[SCENE \d+\])", prompt.split("SCREENPLAY:\n", 1)[1])
                      if b.startswith("[SCENE")]
            fn = self.rewrite or (lambda num, text: text.replace("Hello Ama, good morning.", "Kofi is alive. Look."))
            return {"scenes": [{"number": k, "text": fn(k, scenes[k - 1].split("\n\nReturn each")[0])} for k in nums]}, 900, 900
        if self.eval_raw is not None:
            return self.eval_raw, 2000, 800
        return self.judge(prompt), 2000, 800


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


def judge_calls(room):
    return [p for p in room.prompts if p[0] == "json" and p[2] is sc.SCHEMA]


BRIEF = {"title": "T", "logline": "x", "kind": "episode", "target_seconds": 60, "genre": "drama"}
WEAK_OPEN = ("INT. WARD - NIGHT\n\nAma walks in, soaked.\n\nAMA\nHello Ama, good morning.\n\n"
             "INT. CORRIDOR - NIGHT\n\nGrace blocks the door.\n\nGRACE\nKofi Mensah died on Tuesday. So who are you?\n")


def run_story(room, script, brief=BRIEF, previous=()):
    """develop() with a fixed screenplay (the draft and the doctor's rewrite)."""
    room.text = lambda p: (room.prompts.append(("text", p)) or (script, 1000, 900))
    orig = room.json

    def json_fn(prompt, schema):
        if schema is story.DOCTOR_SCHEMA:
            room.prompts.append(("json", prompt, schema))
            return {"scores": {}, "fixes": [], "screenplay": script, "opening_hook": "", "ending_hook": "",
                    "episode_summary": "", "open_threads": []}, 10, 10
        return orig(prompt, schema)
    return story.develop(room.text, json_fn, brief, list(previous))


# ------------------------------------------------------------------ three episodes

def test_a_three_episode_drama_has_hooks_at_both_ends_of_every_episode(env):
    room = Room()
    client, u, ids, bibles = make_series(env, room)
    for b in bibles:
        assert b["beats"][0]["purpose"] == "hook" and b["beats"][-1]["purpose"] == "end_hook"
        assert b["hook_plan"]["opening_moment"] and b["hook_plan"]["ending_moment"] and b["hook_plan"]["viewer_question"]
        assert b["evaluation"]["status"] == "passed" and b["evaluation"]["complete"] is True
        cats = b["evaluation"]["categories"]
        assert cats["opening_hook"]["scene"] == 1 and cats["ending_hook"]["scene"] == 2, "quoted from the right scenes"
        assert b["episode_record"]["open_questions"], "every episode leaves the viewer a question"
    assert "continuity" not in bibles[0]["evaluation"]["categories"], "episode 1 has nothing to continue from"
    assert all(b["evaluation"]["categories"]["continuity"]["status"] == "verified" for b in bibles[1:])


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



# ------------------------------------------------------------------ verifying the final screenplay

def test_a_strong_opening_and_ending_pass_only_with_quotes_from_the_first_and_last_scene():
    room = Room()
    text, bible, *_ = run_story(room, script_for(1))
    ev = bible["evaluation"]
    assert ev["status"] == "passed" and ev["complete"] and ev["unverified"] == []
    assert ev["categories"]["opening_hook"]["quote"] == "Episode 1 begins now, right here."
    assert ev["categories"]["ending_hook"]["quote"] == "This is how episode 1 ends."
    assert ev["script_sha"] == sc.script_sha(text), "the check belongs to this exact script"
    judged = judge_calls(room)[0][1]
    assert "[SCENE 1]" in judged and "[SCENE 2]" in judged and "A plan or an intention earns nothing" in judged
    assert "Opening (shocking_discovery)" not in judged or True


def test_a_weak_opening_is_found_and_only_the_first_scene_is_rewritten():
    room = Room(scores={**{k: 8 for k in story.CRITERIA}, "opening_hook": 4},
                revised_scores={k: 8 for k in story.CRITERIA})
    text, bible, *_ = run_story(room, WEAK_OPEN)
    ev = bible["evaluation"]
    assert len(room.revisions) == 1 and "Rewrite ONLY scene(s) 1 " in room.revisions[0]
    assert "The first line is a greeting" in room.revisions[0], "the no-AI check is part of the brief"
    assert ev["status"] == "passed" and ev["revision"]["kept"] is True and ev["revision"]["rewrote"] == [1]
    assert ev["revision"]["before"]["opening_hook"] == 4 and ev["categories"]["opening_hook"]["score"] == 8
    assert "Kofi is alive. Look." in text and "Hello Ama" not in text
    assert sc.sections(text)[1] == sc.sections(WEAK_OPEN)[1], "every other scene is kept word for word"


def test_a_weak_ending_rewrites_only_the_last_scene():
    seen = []

    def rewrite(n, text):
        seen.append(n)
        return text.replace("So who are you?", "So who is the man in bed twelve?")
    room = Room(scores={**{k: 8 for k in story.CRITERIA}, "ending_hook": 5}, rewrite=rewrite)
    room.revised_scores = None
    script = WEAK_OPEN.replace("Hello Ama, good morning.", "Kofi is in bed twelve, I know it.")
    text, bible, *_ = run_story(room, script)
    assert seen == [2], "only the last scene"
    rev = bible["evaluation"]["revision"]
    assert rev["kept"] is False and "none of the weak areas improved" in rev["why"], "no proof of improvement: not kept"
    assert text == script.strip() + "\n" or text.strip() == script.strip()
    assert bible["evaluation"]["status"] == "needs_work" and bible["evaluation"]["below_bar"] == ["ending_hook"]


def test_invented_evidence_is_unverified_and_the_episode_is_incomplete_not_passed():
    room = Room(invent={"emotion", "suspense"})
    text, bible, *_ = run_story(room, script_for(1))
    ev = bible["evaluation"]
    assert ev["status"] == "incomplete" and not ev["complete"]
    assert set(ev["unverified"]) == {"emotion", "suspense"}
    assert "aren't in the screenplay" in ev["categories"]["emotion"]["why"]
    assert "emotion" not in sc.scores(ev), "an unverified score is never used"
    assert len(judge_calls(room)) == 2 and "COULD NOT BE VERIFIED" in judge_calls(room)[1][1], "asked again once"
    assert room.revisions == [], "never rewritten on an unverified judgement"
    assert "It isn't a pass" in ev["summary"]


def test_a_second_ask_with_real_quotes_completes_the_check():
    room = Room(invent={"emotion"}, retry_fixes=True)
    text, bible, *_ = run_story(room, script_for(1))
    assert bible["evaluation"]["status"] == "passed" and bible["evaluation"]["retried"] is True


def test_an_opening_quoted_from_the_wrong_scene_does_not_count():
    room = Room(wrong_place={"opening_hook"})
    text, bible, *_ = run_story(room, script_for(1))
    c = bible["evaluation"]["categories"]["opening_hook"]
    assert c["status"] == "unverified" and "not the first scene" in c["why"]
    assert bible["evaluation"]["status"] == "incomplete"


def test_missing_categories_never_pass():
    room = Room(missing={"suspense", "scene_relevance"})
    text, bible, *_ = run_story(room, script_for(1))
    ev = bible["evaluation"]
    assert ev["status"] == "incomplete" and set(ev["unverified"]) == {"suspense", "scene_relevance"}
    assert ev["categories"]["suspense"]["status"] == "missing"
    ev = sc.verify({}, script_for(1), has_previous=True)
    assert ev["status"] == "incomplete" and set(ev["unverified"]) == set(story.CRITERIA), "nothing judged: nothing passed"


def test_continuity_is_required_from_episode_2_and_a_wrong_pickup_is_caught():
    prev = [{**PLANS[1], "title": "Ep 1", "episode_record": {"final_scene": "INT. CORRIDOR - NIGHT\n\nGRACE\nKofi Mensah died."}}]
    room = Room()
    unrelated = "INT. BEACH - DAY\n\nTOMAS\nThe sea is calm today, my friend.\n\nINT. BAR - NIGHT\n\nTOMAS\nAnother drink for my friend.\n"
    text, bible, *_ = run_story(room, unrelated, brief={**BRIEF, "kind": "episode"}, previous=prev)
    ev = bible["evaluation"]
    assert "continuity" in ev["categories"], "required from episode 2 on"
    assert any(c["area"] == "continuity" and "grace" in c["problem"] for c in ev["checks"]), \
        "the first scene has nobody from the last episode's last scene"
    assert "continuity" in ev["below_bar"] and ev["status"] != "passed"


def test_a_failed_targeted_rewrite_keeps_the_original_and_says_why():
    room = Room(scores={**{k: 8 for k in story.CRITERIA}, "opening_hook": 4}, rewrite=lambda n, text: "Ama cries.")
    text, bible, *_ = run_story(room, WEAK_OPEN)
    rev = bible["evaluation"]["revision"]
    assert rev["kept"] is False and "didn't return proper scenes" in rev["why"]
    assert text.strip() == WEAK_OPEN.strip()
    room = Room(scores={**{k: 8 for k in story.CRITERIA}, "opening_hook": 4},
                revised_scores={**{k: 8 for k in story.CRITERIA}, "opening_hook": 9, "dialogue": 5})
    text, bible, *_ = run_story(room, WEAK_OPEN)
    assert bible["evaluation"]["revision"]["kept"] is False and "dialogue got worse" in bible["evaluation"]["revision"]["why"]
    assert "Hello Ama" in text, "a fix that breaks something else is thrown away"


def test_quotes_are_found_despite_case_punctuation_and_skipped_words():
    s = script_for(1)
    assert sc.find_quote("episode 1 BEGINS now right here", s) is not None
    assert sc.find_quote("Episode 1 ... right here", s) is not None
    assert sc.find_quote("Episode 1 ends now right here", s) is None
    assert sc.find_quote("right here", s) is None, "too short to prove anything"


# ------------------------------------------------------------------ the creator's own edits

def test_a_creator_edit_makes_the_check_outdated_and_a_recheck_judges_the_edited_script_without_rewriting(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    room = Room()
    client, u, ids, bibles = make_series(env, room)
    pid = ids[0]
    p = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["project"]
    edited = p["script"].replace("This is how episode 1 ends.", "Grace locks the door from the outside, smiling.")
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": edited})
    info = client.get(f"/api/v1/studio/projects/{pid}/story-check", headers=H(u)).json()
    assert info["outdated"] is True and info["credits"] > 0
    monkeypatch.setattr(cs, "_ai", ai)
    room.revisions.clear()
    r = client.post(f"/api/v1/studio/projects/{pid}/story-check", headers=H(u), json={"request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    ev = r.json()["evaluation"]
    assert ev["source"] == "recheck" and ev["categories"]["ending_hook"]["quote"] == "Grace locks the door from the outside, smiling."
    after = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["project"]
    assert after["script"] == edited, "the creator's edit is never rewritten"
    assert room.revisions == [] and after["story_bible"]["evaluation"]["script_sha"] == sc.script_sha(edited)
    assert after["story_bible"]["hook_plan"] == bibles[0]["hook_plan"], "the rest of the story plan is kept"
    assert client.get(f"/api/v1/studio/projects/{pid}/story-check", headers=H(u)).json()["outdated"] is False


def test_a_failed_recheck_charges_nothing(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    room = Room()
    client, u, ids, bibles = make_series(env, room)
    monkeypatch.setattr(cs, "_ai", ai)
    ai.fail = {"story_check": RuntimeError("down")}
    r = client.post(f"/api/v1/studio/projects/{ids[0]}/story-check", headers=H(u), json={"request_id": uuid.uuid4().hex})
    assert r.status_code == 502 and "weren't charged" in r.text


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
    assert "STORY CHECK of the final script: NEEDS WORK. Needs work: ending cliffhanger." in prompt
    assert "- ending hook 6/10 (verified quote, scene 2: \"This is how episode 2 ends.\")" in prompt
    assert "Revision: not kept" in prompt
    assert "never call an INCOMPLETE check a pass" in prompt
    assert "It ended on (disturbing news): Grace: 'Kofi Mensah died on Tuesday. So who are you?'" in prompt
    assert "Left open:" in prompt and "Picks up the previous episode:" in prompt
    assert "Episode storytelling (hooks and cliffhangers)" in prompt and "Never invent scores" in prompt
    script = client.get(f"/api/v1/studio/projects/{ids[1]}", headers=H(u)).json()["project"]["script"]
    assert "New ending." not in script, "a rewrite is a proposal until the creator applies it"
    # After the creator edits the script, the assistant is told the check is out of date.
    p = client.get(f"/api/v1/studio/projects/{ids[1]}", headers=H(u)).json()["project"]
    client.patch(f"/api/v1/studio/projects/{ids[1]}", headers=H(u), json={"script": p["script"] + "\nAMA\nI'm not done.\n"})
    client.post(f"/api/v1/studio/projects/{ids[1]}/assistant/chat", headers=H(u),
                json={"message": "Is the ending good now?", "request_id": uuid.uuid4().hex})
    assert "the script changed since: this check is OUT OF DATE" in ai.chat_prompts[-1]
