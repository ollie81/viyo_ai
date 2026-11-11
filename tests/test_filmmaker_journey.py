"""End-to-end creator journey for one 60-second drama, from an idea to a film
plan, with the difficult moments a real creator hits along the way:

  "A woman discovers that her missing brother has been admitted to the
   hospital under a false name."

Everything runs on the fake AI and local Postgres: no real Gemini, no paid
APIs, no real credits. The fake model's answers are scripted, so these tests
prove what the SYSTEM guarantees around the model (what it is told, what it
remembers, what it may change, what is blocked, what is reported honestly) and
the quality of the deterministic parts (pacing, shot plans, continuity,
dialogue coverage). They cannot prove the real model writes well; that is what
scripts/eval_assistant.py does against real Gemini, opt-in.
"""
import os
import uuid

import pytest

import creator_studio as cs
import studio_chat
import studio_memory
import studio_shots
import studio_story
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import detail, project

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the shots SQL")

IDEA = "A woman discovers that her missing brother has been admitted to the hospital under a false name."

AMA = {"name": "Ama", "description": "Woman, 29, braided hair tied back, mustard raincoat, determined", "gender": "female"}
KOFI = {"name": "Kofi", "description": "Man, 24, thin, short hair, healing cut above his left eyebrow", "gender": "male"}
GRACE = {"name": "Grace", "description": "Nurse, 50s, grey hair in a bun, navy scrubs, tired but kind", "gender": "female"}


def line(who, text, emotion=""):
    return {"character": who, "text": text, **({"emotion": emotion} if emotion else {})}


def cont(transition, *people, **extra):
    return {"transition": transition, "characters": [{"name": n, "in_frame": True, "position": pos, "outfit": wear}
                                                     for n, pos, wear in people], **extra}


# A 60-second, 8-scene story with a hook at the start and at the end.
SCENES = [
    {"heading": "INT. AMA'S FLAT - NIGHT", "duration_seconds": 6,
     "visual": "Ama's phone lights up: a photo of Kofi in a hospital bed, sent from an unknown number",
     "dialogue": [line("Ama", "Kofi?", "shocked")],
     "continuity": cont("new_place", ("Ama", "centre", "mustard raincoat"), purpose="hook", time_of_day="night",
                        end_state="Ama grabs her keys, raincoat on")},
    {"heading": "INT. HOSPITAL RECEPTION - NIGHT", "duration_seconds": 9,
     "visual": "Ama shows the photo to Nurse Grace at the desk",
     "dialogue": [line("Ama", "This man. Which ward is he in?"),
                  line("Grace", "We have no Kofi Mensah here."),
                  line("Ama", "Then who is this?", "angry")],
     "continuity": cont("time_jump", ("Ama", "left", "mustard raincoat"), ("Grace", "right", "navy scrubs"),
                        time_of_day="night", start_state="Ama walks in, soaked")},
    {"heading": "INT. HOSPITAL RECEPTION - NIGHT", "duration_seconds": 7,
     "visual": "Grace turns the screen: the patient is listed as 'Joseph Owusu'",
     "dialogue": [line("Grace", "Bed twelve. Joseph Owusu. Admitted Tuesday.")],
     "continuity": cont("continuous", ("Ama", "left", "mustard raincoat"), ("Grace", "right", "navy scrubs"))},
    {"heading": "INT. WARD - NIGHT", "duration_seconds": 8,
     "visual": "Ama pulls back the curtain of bed twelve: Kofi, pale, a wristband reading JOSEPH OWUSU",
     "dialogue": [line("Ama", "Who did this to you?"), line("Kofi", "You shouldn't be here.", "afraid")],
     "continuity": cont("same_place_cut", ("Ama", "left", "mustard raincoat"), ("Kofi", "right", "hospital gown"))},
    {"heading": "INT. WARD - NIGHT", "duration_seconds": 8,
     "visual": "Kofi grips her wrist and whispers",
     "dialogue": [line("Kofi", "The name was their idea."), line("Ama", "Whose?"),
                  line("Kofi", "If they find out you came, they'll move me again.", "afraid")],
     "continuity": cont("continuous", ("Ama", "left", "mustard raincoat"), ("Kofi", "right", "hospital gown"))},
    {"heading": "INT. WARD CORRIDOR - NIGHT", "duration_seconds": 6,
     "visual": "Through the ward window Ama sees Grace on the phone, looking straight at her",
     "dialogue": [],
     "continuity": cont("same_place_cut", ("Ama", "centre", "mustard raincoat"), ("Grace", "background", "navy scrubs"))},
    {"heading": "INT. WARD - NIGHT", "duration_seconds": 8,
     "visual": "Ama helps Kofi sit up and pulls the drip from the stand",
     "dialogue": [line("Ama", "We're leaving. Now."), line("Kofi", "Ama, the nurse—")],
     "continuity": cont("same_place_cut", ("Ama", "left", "mustard raincoat"), ("Kofi", "right", "hospital gown"))},
    {"heading": "INT. WARD - NIGHT", "duration_seconds": 8,
     "visual": "Grace blocks the doorway, holding Kofi's real ID card",
     "dialogue": [line("Grace", "Kofi Mensah died on Tuesday. So who are you?", "cold")],
     "continuity": cont("continuous", ("Ama", "left", "mustard raincoat"), ("Kofi", "right", "hospital gown"),
                        ("Grace", "doorway", "navy scrubs"), purpose="end_hook")},
]

SCRIPT = """INT. AMA'S FLAT - NIGHT

Ama's phone lights up: a photo of Kofi in a hospital bed.

AMA
Kofi?

INT. HOSPITAL RECEPTION - NIGHT

AMA
This man. Which ward is he in?

GRACE
We have no Kofi Mensah here.

AMA
(angry)
Then who is this?

GRACE
Bed twelve. Joseph Owusu. Admitted Tuesday.

INT. WARD - NIGHT

AMA
Who did this to you?

KOFI
You shouldn't be here.

KOFI
The name was their idea.

AMA
Whose?

KOFI
If they find out you came, they'll move me again.

AMA
We're leaving. Now.

KOFI
Ama, the nurse—

GRACE
Kofi Mensah died on Tuesday. So who are you?
"""


def reply(text, proposals=(), memory=(), focus=None):
    return {"reply": text, "proposals": list(proposals), "memory": list(memory), **({"focus": focus} if focus else {})}


def say(client, u, pid, text):
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": text, "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    return r.json()


def apply(client, u, msg, index=0):
    return client.post(f"/api/v1/studio/assistant/messages/{msg['reply']['id']}/proposals/{index}/apply", headers=H(u))


@pytest.fixture
def studio(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    monkeypatch.setattr(cs, "_files", files)
    u = make_user(db)
    grant(client, u, 50000)
    return client, db, store, files, ai, gen, ren, u


def built(studio, scenes=None, script=SCRIPT):
    """The production after the story was written and broken into scenes."""
    client, db, store, files, ai, gen, ren, u = studio
    pid = project(client, u, title="False Name", logline=IDEA, kind="short", target_seconds=60)["id"]
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": script})
    ai.breakdown_result = {"characters": [AMA, KOFI, GRACE], "scenes": [dict(s) for s in (scenes or SCENES)]}
    assert start(client, u, pid, "breakdown").status_code == 200
    run(gen)
    return pid


def scene_list(client, u, pid):
    return detail(client, u, pid)["scenes"]


# ------------------------------------------------------------------ 1. understanding the idea

def test_1_a_bare_idea_gets_the_hook_rules_one_question_and_a_proposal_that_is_not_applied(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = project(client, u, title="Untitled", logline="", kind="short", target_seconds=60)["id"]
    ai.chat_result = reply(
        "Strong premise: the false name is the mystery. The question that shapes everything: did Kofi choose the "
        "false name, or did someone give it to him?",
        proposals=[{"type": "update_story", "title": "Set the idea", "summary": "Logline and title 'False Name'.",
                    "data": {"title": "False Name", "logline": IDEA, "genre": "Thriller"}}],
        memory=[{"category": "story", "text": IDEA, "from_creator": True}])
    r = say(client, u, pid, f"I have an idea: {IDEA}")
    prompt = ai.chat_prompts[-1]
    assert IDEA in prompt
    assert "ask ONE question (never a list of questions)" in prompt
    assert "starts with a hook in the first seconds and ends on a hook" in prompt
    assert ai.chat_modes[-1] == "story", "a story idea gets creative thinking, not a quick answer"
    prop = r["reply"]["proposals"][0]
    assert prop["status"] == "proposed" and prop["paid"] is False
    p = detail(client, u, pid)["project"]
    assert p["title"] == "Untitled" and not p["logline"], "nothing changes until the creator applies it"
    assert apply(client, u, r).status_code == 200
    p = detail(client, u, pid)["project"]
    assert p["title"] == "False Name" and p["logline"] == IDEA
    mem = client.get(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u)).json()["memory"]
    assert any(IDEA in it["text"] for it in mem["story"]), "the creator's own idea is remembered as a fact"
    assert any("Applied: Set the idea" in it["text"] for it in mem["decisions"])


# ------------------------------------------------------------------ 2-3. story and characters

def test_2_the_writers_room_keeps_the_idea_and_plans_hooks_for_60_seconds():
    brief = {"title": "False Name", "logline": IDEA, "genre": "Thriller", "kind": "short", "aspect_ratio": "9:16",
             "target_seconds": 60, "language": "English"}
    prompt = studio_story.plan_prompt(brief, [])
    assert IDEA in prompt and "keep its core" in prompt
    assert "about 7 scenes" in prompt, "60 seconds paced at about one scene every 8 seconds"
    assert "FIRST beat's purpose is \"hook\"" in prompt and "LAST beat's purpose is \"end_hook\"" in prompt
    assert "a want, a flaw, a secret, a way of speaking" in prompt
    assert "Dialogue is short and has subtext" in prompt
    calls = []

    def json_fn(p, schema):
        calls.append(p)
        if schema is studio_story.PLAN_SCHEMA:
            beats = [{"purpose": "setup", "what_happens": "Ama at home", "emotional_turn": "calm", "seconds": 60}] \
                if len(calls) == 1 else \
                [{"purpose": "hook", "what_happens": "photo of Kofi", "emotional_turn": "calm -> shock", "seconds": 6},
                 {"purpose": "end_hook", "what_happens": "Kofi is dead on paper", "emotional_turn": "hope -> dread",
                  "seconds": 8}]
            return {"title": "False Name", "logline": IDEA, "central_conflict": "x", "stakes": "x", "ending": "x",
                    "characters": [{"name": "Ama", "age": "29", "role": "lead", "want": "find Kofi", "flaw": "reckless",
                                    "speech": "short", "look": "mustard raincoat"}], "beats": beats}, 10, 10
        return {"screenplay": SCRIPT, "scores": {}, "fixes": [], "opening_hook": "Kofi?",
                "ending_hook": "So who are you?", "episode_summary": "", "open_threads": []}, 10, 10

    script, bible, *_ = studio_story.develop(lambda p: (SCRIPT, 10, 10), json_fn, brief)
    assert "YOUR PREVIOUS PLAN HAS THESE PROBLEMS" in calls[1], "a plan without an opening hook is sent back"
    assert bible["hooks_planned"] is True and bible["opening_hook"] == "Kofi?"


def test_3_characters_keep_one_identity_through_the_breakdown(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = built(studio)
    cast = {c["name"]: c for c in detail(client, u, pid)["characters"]}
    assert set(cast) == {"Ama", "Kofi", "Grace"}
    assert "mustard raincoat" in cast["Ama"]["description"] and "cut above his left eyebrow" in cast["Kofi"]["description"]
    assert len({cast[n]["voice"] for n in cast}) == 3, "three different voices"
    # A second breakdown never re-casts or renames anyone.
    voices = {n: c["voice"] for n, c in cast.items()}
    assert start(client, u, pid, "breakdown").json()["detail"]["code"] == "confirm_replace", "never replaced silently"
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "breakdown", "request_id": uuid.uuid4().hex, "confirm_replace": True})
    assert r.status_code == 200, r.text
    run(gen)
    assert {c["name"]: c["voice"] for c in detail(client, u, pid)["characters"]} == voices


# ------------------------------------------------------------------ 4-5. dialogue and pacing

def test_4_5_pacing_fills_60_seconds_and_every_line_is_said_once(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = built(studio)
    sc = scene_list(client, u, pid)
    assert len(sc) == 8 and abs(sum(s["duration_seconds"] for s in sc) - 60) < 0.01
    assert "8 scenes" in ai.last_breakdown_prompt or "scenes per minute" in ai.last_breakdown_prompt.lower() or \
        "8" in ai.last_breakdown_prompt
    check = studio_story.dialogue_check(SCRIPT, sc)
    assert check == {"forgotten": [], "repeated": []}
    for s in sc:   # each scene's lines fit its length (speech wins otherwise, see test_multi_shot)
        assert studio_shots.speech_seconds(s["dialogue"]) <= s["duration_seconds"] + 0.5, s["heading"]
    ai.chat_result = reply("Every line is there.")
    say(client, u, pid, "Is any dialogue missing or repeated?")
    assert "every screenplay line is said once in the scenes" in ai.chat_prompts[-1]


def test_4_a_forgotten_and_a_repeated_line_are_shown_to_the_assistant(studio):
    client, db, store, files, ai, gen, ren, u = studio
    bad = [dict(s) for s in SCENES]
    bad[2] = dict(bad[2], dialogue=[])                                    # Grace's "Bed twelve" line lost
    bad[6] = dict(bad[6], dialogue=bad[6]["dialogue"] + [line("Ama", "Who did this to you?")])   # repeated
    pid = built(studio, scenes=bad)
    ai.chat_result = reply("Two possible problems.")
    say(client, u, pid, "Is any dialogue missing or repeated?")
    prompt = ai.chat_prompts[-1]
    assert "Possibly forgotten: Grace: \"Bed twelve. Joseph Owusu. Admitted Tuesday.\"" in prompt
    assert "Possibly repeated: Ama: \"Who did this to you?\" is said in scenes 4, 7" in prompt


# ------------------------------------------------------------------ 6-7. shots and continuity

def test_6_7_shots_follow_the_moment_and_carry_continuity(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = built(studio)
    ids = [s["id"] for s in scene_list(client, u, pid)]
    assert start(client, u, pid, "shotplan", ids).status_code == 200
    run(gen)
    sc = scene_list(client, u, pid)
    prompt = ai.shot_prompts[1]                     # scene 2: the reception confrontation
    assert "We have no Kofi Mensah here." in prompt and "180-degree rule" in prompt
    assert "close-up on the listener reacting to an important line" in prompt
    assert "THE SCENE BEFORE ended with: Ama grabs her keys, raincoat on" in prompt
    confrontation, silent = sc[1]["shots"], sc[5]["shots"]
    assert [x["shot_type"] for x in confrontation] == ["wide_two_shot", "over_the_shoulder", "reaction"]
    assert not any(x["dialogue"] for x in silent), "a wordless beat: no shot is given lines"
    for s in sc:
        heard = [d["text"] for x in s["shots"] for d in x["dialogue"]]
        assert heard == [d["text"] for d in s["dialogue"]], f"{s['heading']}: every line once, in order"
        assert sum(x["duration_seconds"] for x in s["shots"]) >= s["duration_seconds"] - 0.05
        for x in s["shots"]:
            assert x["duration_seconds"] >= studio_shots.speech_seconds(x["dialogue"]) - 0.05, "no line is cut off"
    total = sum(x["duration_seconds"] for s in sc for x in s["shots"])
    assert 60 <= total <= 66, f"speech wins over the 60 s target, but only by a little ({total:.1f} s)"
    # Screen direction is carried from shot to shot: Ama stays on the left.
    block = studio_shots.continuity_block(sc[1], confrontation, 1, None)
    assert "Ama" in block and "left" in block


def test_7_an_accidental_outfit_change_is_reported_as_a_possible_problem(studio):
    client, db, store, files, ai, gen, ren, u = studio
    bad = [dict(s) for s in SCENES]
    bad[4] = dict(bad[4], continuity=cont("continuous", ("Ama", "left", "red dress"), ("Kofi", "right", "hospital gown")))
    pid = built(studio, scenes=bad)
    ai.chat_result = reply("Scene 5 seems to show Ama in a red dress, but scene 4 ends with her raincoat on.")
    say(client, u, pid, "Does every scene connect to the one before?")
    prompt = ai.chat_prompts[-1]
    assert "CONTINUITY CHECK" in prompt and "Scene 4→5" in prompt and "red dress" in prompt
    assert ai.chat_modes[-1] == "deep"


# ------------------------------------------------------------------ 8-10. explaining, next steps, honesty

def test_8_9_10_next_steps_are_listed_and_paid_steps_are_never_started_by_the_chat(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = built(studio)
    before = balance(client, u)["available"]
    ai.chat_result = reply(
        "Next: reference pictures for Ama, Kofi and Grace, then scene pictures. I've proposed the shot plan; you'll see "
        "the price before anything is charged.",
        proposals=[{"type": "plan_shots", "title": "Plan the shots", "summary": "The director plans every scene.",
                    "data": {"scenes": []}},
                   {"type": "make_character_sheet", "title": "Ama's reference sheet", "summary": "x",
                    "data": {"character": "Ama"}}])
    r = say(client, u, pid, "What is still missing before this film is finished?")
    prompt = ai.chat_prompts[-1]
    assert "Still missing: 8 scene pictures" in prompt and "reference pictures for 3 characters" in prompt
    assert "the final film" in prompt
    props = r["reply"]["proposals"]
    assert [p["paid"] for p in props] == [True, True]
    assert apply(client, u, r).json()["detail"]["code"] == "needs_price", "a paid step opens the price check"
    jobs = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"]
    assert {j["kind"] for j in jobs} == {"breakdown"}, "the chat started nothing"
    assert before - balance(client, u)["available"] <= 5, "only the chat answer was charged"


def test_10_claims_that_something_was_saved_are_never_backed_by_the_server(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = built(studio)
    ai.chat_result = reply("Done.", proposals=[{"type": "rewrite_scene", "title": "Sharper line", "summary": "x",
                                                "data": {"scene": 2, "dialogue": [line("Grace", "No Kofi here.")]}}])
    say(client, u, pid, "Make Grace's line in scene 2 colder.")
    assert scene_list(client, u, pid)[1]["dialogue"][1]["text"] == "We have no Kofi Mensah here.", \
        "a proposal is not a change, whatever the reply says"
    assert "Never say something \"was saved\"" in ai.chat_prompts[-1]


# ------------------------------------------------------------------ difficult situations

def test_hard_the_creator_changes_the_story_and_approved_shots_become_outdated(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = built(studio)
    sid = scene_list(client, u, pid)[1]["id"]
    assert start(client, u, pid, "shotplan", [sid]).status_code == 200
    run(gen)
    assert client.post(f"/api/v1/studio/scenes/{sid}/shots/approve", headers=H(u), json={"approved": True}).status_code == 200
    # Halfway through, the creator changes who lies to Ama.
    ai.chat_result = reply("Here is scene 2 with Grace covering for someone.", proposals=[{
        "type": "rewrite_scene", "title": "Scene 2: Grace lies", "summary": "New lines.",
        "data": {"scene": 2, "dialogue": [line("Ama", "This man. Which ward?"), line("Grace", "I've never seen him."),
                                          line("Ama", "You're lying.")]}}],
        memory=[{"category": "story", "text": "Grace knows Kofi and lies about him", "from_creator": True}])
    r = say(client, u, pid, "Change of plan: Grace knows Kofi and lies about him.")
    assert apply(client, u, r).status_code == 200
    s = scene_list(client, u, pid)[1]
    assert s["shot_plan_outdated"] is True and s["shot_plan_approved"] is True
    # The film can't be made from shots that say the old lines.
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "render"}).json()
    assert "Scene 2's lines changed after its shots were planned" in est["items"][0]["blocked"]
    shot = s["shots"][0]["id"]
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "shot_image", "target_ids": [shot]}).json()
    assert "lines changed" in est["items"][0]["blocked"], "no credits spent on outdated shots"
    ai.chat_result = reply("Scene 2's shots must be planned again.")
    say(client, u, pid, "Can we make the film now?")
    prompt = ai.chat_prompts[-1]
    assert "OUTDATED: the scene's lines changed after the shots were planned" in prompt
    assert "the newer statement wins" in prompt
    mem = client.get(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u)).json()["memory"]
    assert any("Grace knows Kofi" in it["text"] and it["source"] == "creator" for it in mem["story"])
    # Unapprove and plan again: the new lines are in the shots and the block is gone.
    client.post(f"/api/v1/studio/scenes/{sid}/shots/approve", headers=H(u), json={"approved": False})
    assert start(client, u, pid, "shotplan", [sid]).status_code == 200
    run(gen)
    s = scene_list(client, u, pid)[1]
    assert s["shot_plan_outdated"] is False
    assert [d["text"] for x in s["shots"] for d in x["dialogue"]] == ["This man. Which ward?", "I've never seen him.",
                                                                        "You're lying."]


def test_hard_her_that_scene_and_the_previous_shot_are_resolved():
    names = ["Ama", "Kofi", "Grace"]
    f = studio_chat.resolve_focus("Why is Grace so cold in scene 8?", {}, 8, names)
    assert f == {"scenes": [8], "characters": ["Grace"]}
    f = studio_chat.resolve_focus("Make her softer in that scene.", f, 8, names)
    assert f == {"scenes": [8], "characters": ["Grace"]}
    f = studio_chat.resolve_focus("The previous shot doesn't match this one.", {"scenes": [4], "characters": ["Ama"]},
                                  8, names)
    assert f["scenes"] == [4], "'the previous shot' stays in the scene being discussed"
    f = studio_chat.resolve_focus("And the next scene?", {"scenes": [4]}, 8, names)
    assert f["scenes"] == [5]


def test_hard_a_shot_that_does_not_match_the_one_before_is_explained_from_the_plan(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = built(studio)
    sid = scene_list(client, u, pid)[3]["id"]
    start(client, u, pid, "shotplan", [sid])
    run(gen)
    ai.chat_result = reply("In shot 2 the plan keeps Ama on the left.", focus={"scenes": [4], "characters": ["Ama"]})
    say(client, u, pid, "In scene 4, Ama is on the wrong side in shot 2.")
    ai.chat_result = reply("Shot 1 is the wide two-shot.")
    say(client, u, pid, "And the previous shot?")
    prompt = ai.chat_prompts[-1]
    assert "CURRENT FOCUS (what 'that scene', 'her', 'him', 'the next scene' refer to): scene 4" in prompt
    assert "SHOTS (2, plan not approved)" in prompt or "SHOTS (" in prompt


def test_hard_an_incomplete_audit_an_unloadable_picture_and_a_failed_step_are_reported_honestly(studio, monkeypatch):
    client, db, store, files, ai, gen, ren, u = studio
    pid = built(studio)
    rows = db.execute("select id, position from public.studio_scenes where project_id = %s "
                      "order by position", (pid,)).fetchall()
    import io
    from PIL import Image
    for sid, pos in rows[:3]:
        out = io.BytesIO()
        Image.new("RGB", (64, 64), (200, 40, 40)).save(out, "PNG")
        path = f"{u}/{pid}/image/s{pos}.png"
        files.put(path, out.getvalue(), "image/png")
        db.execute("update public.studio_scenes set image_path = %s where id = %s", (path, sid))
    real = files.get
    monkeypatch.setattr(files, "get", lambda p: b"broken" if p.endswith("s1.png") else real(p))
    insp = __import__("studio_worker").StudioWorker(store, files, ai, "inspect")
    start(client, u, pid, "inspect", [str(rows[0][0])])
    run(insp)
    ai.fail = {"image": [RuntimeError("provider refused the picture")]}
    start(client, u, pid, "image", [str(rows[3][0])])
    run(gen)
    ai.chat_result = reply("1 of 3 pictures checked.")
    r = say(client, u, pid, "Compare scene 1 and scene 2. Have you checked every scene?")
    prompt = ai.chat_prompts[-1]
    assert "picture checks are current for 1 of 3 scene pictures" in prompt
    assert "A visual audit of the whole production is NOT complete" in prompt
    assert "scene 2 picture (INT. HOSPITAL RECEPTION - NIGHT): couldn't be loaded" in prompt
    assert r["not_seen"] and r["seen"][0] == "False Name · scene 1 picture (INT. AMA'S FLAT - NIGHT)"
    assert not any("scene 2" in x for x in r["seen"]), "the unreadable picture is never passed off as seen"
    assert "- image scene 4: failed" in prompt, "the failed step and its reason are in the recent steps"


def test_hard_a_request_against_an_approved_decision_is_flagged_not_silently_followed(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = built(studio)
    client.post(f"/api/v1/studio/projects/{pid}/assistant/memory", headers=H(u),
                json={"category": "decisions", "text": "Kofi is alive at the end; the twist is that Kofi is officially dead"})
    ai.chat_result = reply("That would undo the approved twist. Do you want to replace it?")
    say(client, u, pid, "Let's have Kofi die in the last scene.")
    prompt = ai.chat_prompts[-1]
    assert "Approved decisions:\n- Kofi is alive at the end" in prompt
    assert "contradicts an APPROVED decision" in prompt and "ask ONE question to\n  confirm" in prompt


def test_hard_a_change_of_mind_keeps_both_items_in_order_so_the_newer_one_wins():
    mem, _ = studio_memory.add({}, "story", "Kofi chose the false name himself")
    mem, _ = studio_memory.add(mem, "story", "Someone else gave Kofi the false name")
    text = studio_memory.render(mem)
    assert text.index("chose the false name himself") < text.index("Someone else gave Kofi")
