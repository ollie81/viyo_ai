"""The writers'-room story engine. Fake model calls only."""
import json
import types

import pytest

import studio_ai
import studio_pricing as pricing
import studio_story as story
from conftest import make_user
from test_creator_studio import H, env, grant, new_project, run, start, studio_schema  # noqa: F401

PLAN = {"title": "The Last Bus", "logline": "A nurse must choose between her job and her brother.",
        "central_conflict": "duty vs family", "stakes": "her brother's life",
        "characters": [{"name": "Ama", "age": "late 20s", "role": "lead", "want": "to save him", "flaw": "pride",
                        "speech": "clipped", "look": "scrubs, braids"}],
        "beats": [{"purpose": "hook", "what_happens": "A patient says her name", "emotional_turn": "tired -> afraid", "seconds": 5},
                  {"purpose": "end_hook", "what_happens": "Her brother's ring on the patient's hand", "emotional_turn": "relief -> dread",
                   "seconds": 5}],
        "ending": "The bus leaves without her.",
        "hook_plan": {"opening_technique": "emotional_revelation", "opening_moment": "A patient says her name",
                      "opening_why": "a stranger knows her", "ending_technique": "disturbing_news",
                      "ending_moment": "The ring", "viewer_question": "Is the patient her brother?"},
        "questions": [{"question": "Who is the patient?", "raised_in_beat": 1, "answered_in_beat": 0}],
        "knowledge": [{"character": "Ama", "knows": "a patient knows her name"}]}
DRAFT = "INT. WARD - NIGHT\n\nAMA checks a chart.\n\nPATIENT\nAma?\n" * 3


class Fake:
    def __init__(self, doctor=None, plan=PLAN):
        self.prompts, self.doctor, self.plan = [], doctor, plan

    def text(self, prompt):
        self.prompts.append(("text", prompt))
        return DRAFT, 1000, 900

    def json(self, prompt, schema):
        self.prompts.append(("json", prompt, schema))
        if schema is story.PLAN_SCHEMA:
            plan = self.plan.pop(0) if isinstance(self.plan, list) else self.plan
            return plan, 1500, 2000
        if isinstance(self.doctor, Exception):
            raise self.doctor
        return self.doctor if self.doctor is not None else {
            "scores": {"hook": 5, "dialogue": 6}, "fixes": ["hook starts mid-crisis"],
            "screenplay": DRAFT + "\nAMA\nThen I'll drive.\n", "opening_hook": "PATIENT: Ama?",
            "ending_hook": "AMA: Then I'll drive.", "episode_summary": "Ama learns the truth.",
            "open_threads": ["Who sent the patient?"]}, 3000, 2500


def test_three_steps_keep_the_idea_and_apply_the_craft():
    f = Fake()
    brief = {"title": "Night Shift", "logline": "A nurse finds her missing brother in her ward", "genre": "Drama",
             "kind": "episode", "target_seconds": 60, "aspect_ratio": "9:16", "language": "Kinyarwanda"}
    text, bible, i, o = story.develop(f.text, f.json, brief, [])
    plan_p = f.prompts[0][1]
    assert "A nurse finds her missing brother in her ward" in plan_p and "keep its core" in plan_p
    assert "Open in the middle of trouble" in plan_p and "The end hook is a cliffhanger." in plan_p
    assert "NON-NEGOTIABLE: every story starts with a hook and ends with a hook" in plan_p
    assert 'The FIRST beat\'s purpose is "hook" and the LAST beat\'s purpose is "end_hook"' in plan_p
    assert "Language of the final dialogue: Kinyarwanda" in plan_p
    assert '"logline": "A nurse must choose' in f.prompts[1][1] and "Write all dialogue in Kinyarwanda" in f.prompts[1][1]
    assert DRAFT in f.prompts[2][1] and "Score the DRAFT" in f.prompts[2][1]
    assert "The opening hook and the ending hook must each be a 9 or 10" in f.prompts[2][1]
    assert text.endswith("Then I'll drive.") and bible["rewrite_used"] is True
    assert bible["scores"] == {"hook": 5, "dialogue": 6} and bible["open_threads"] == ["Who sent the patient?"]
    assert bible["opening_hook"] == "PATIENT: Ama?" and bible["ending_hook"] == "AMA: Then I'll drive."
    assert bible["hooks_planned"] is True
    assert (i, o) == (1500 + 1000 + 3000 + 3000 + 3000, 2000 + 900 + 2500 + 2500 + 2500), \
        "plan, draft, doctor, story check, and the check asked again once because nothing could be verified"
    assert bible["evaluation"]["status"] == "incomplete", "a check that verifies nothing is never a pass"


def test_a_film_resolves_instead_of_a_cliffhanger():
    f = Fake()
    story.develop(f.text, f.json, {"title": "T", "logline": "x", "kind": "film", "target_seconds": 120}, [])
    tail = f.prompts[0][1].split("Plan ")[1]
    assert "resolves emotionally, and its final moment is still a hook" in tail and "is a cliffhanger" not in tail


def test_a_cut_off_or_failed_rewrite_keeps_the_draft():
    f = Fake(doctor={"scores": {}, "fixes": [], "screenplay": "INT.", "episode_summary": "", "open_threads": []})
    text, bible, *_ = story.develop(f.text, f.json, {"title": "T"}, [])
    assert text == DRAFT.strip() and bible["rewrite_used"] is False
    f = Fake(doctor=RuntimeError("busy"))
    text, bible, *_ = story.develop(f.text, f.json, {"title": "T"}, [])
    assert text == DRAFT.strip()


def test_an_empty_plan_is_an_error_not_a_weak_script():
    with pytest.raises(ValueError):
        story.develop(Fake(plan={"beats": [], "characters": []}).text, Fake(plan={"beats": [], "characters": []}).json,
                      {"title": "T"}, [])


def test_series_continuity_is_given_to_the_writers():
    f = Fake()
    prev = [dict(PLAN, title="Episode 1", episode_summary="Ama meets the stranger.", open_threads=["The ring"])]
    story.develop(f.text, f.json, {"title": "Episode 2", "kind": "episode"}, prev)
    p = f.prompts[0][1]
    assert "THIS IS A SERIES" in p and "Ama meets the stranger." in p and "The ring" in p
    assert "never rename or change a character" in p and "- Ama (late 20s): wants to save him" in p


def test_google_runs_the_three_calls_and_prices_them():
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(config)
            schema = getattr(config, "response_json_schema", None)
            if schema is story.PLAN_SCHEMA:
                body = json.dumps(PLAN)
            elif schema is story.DOCTOR_SCHEMA:
                body = json.dumps({"scores": {}, "fixes": [], "screenplay": DRAFT, "episode_summary": "s", "open_threads": []})
            else:
                body = DRAFT
            usage = types.SimpleNamespace(prompt_token_count=1000, candidates_token_count=500, thoughts_token_count=100)
            return types.SimpleNamespace(text=body, usage_metadata=usage)

    g = object.__new__(studio_ai.GoogleStudioAI)
    g.client = types.SimpleNamespace(models=Models())
    text, bible, cost = g.develop_script({"title": "T", "target_seconds": 60}, [])
    assert len(calls) == 4 and text == DRAFT.strip() and bible["logline"] == PLAN["logline"]
    assert cost.units == {"input_tokens": 3000, "output_tokens": 1800}, "an unreadable check isn't charged"
    assert "couldn't run" in bible["evaluation"]["error"], "an unreadable check is reported, never scored"
    assert pricing.estimate_script(60, 100).provider_cost_usd < 0.09, "worst case incl. two checks, one revision and a re-check"


def test_studio_saves_the_plan_and_episode_2_builds_on_episode_1(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 2000)
    pid = new_project(client, u)
    start(client, u, pid, "script")
    run(gen)
    p = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["project"]
    assert p["story_bible"]["episode_summary"] and p["story_bible"]["characters"][0]["name"] == "Maya"
    assert ai.last_story_previous == []
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u)).json()["id"]
    client.patch(f"/api/v1/studio/projects/{ep2}", headers=H(u), json={"logline": "Maya confronts Daniel."})
    start(client, u, ep2, "script")
    run(gen)
    assert [e["title"] for e in ai.last_story_previous] == ["Night Shift"]
    assert ai.last_story_previous[0]["open_threads"] == ["Why did Maya leave?"]
    # The scene split is told the plan's ages and looks.
    start(client, u, ep2, "breakdown")
    run(gen)
    assert "age late 20s" in studio_ai.breakdown_prompt(ai.last_breakdown_brief, "x")


def test_a_plan_without_both_hooks_is_sent_back_once():
    bad = dict(PLAN, beats=[{"purpose": "setup", "what_happens": "a calm morning", "emotional_turn": "-", "seconds": 5},
                            {"purpose": "resolution", "what_happens": "all is well", "emotional_turn": "-", "seconds": 5}])
    f = Fake(plan=[bad, PLAN])
    text, bible, *_ = story.develop(f.text, f.json, {"title": "T"}, [])
    plan_calls = [c for c in f.prompts if c[0] == "json" and c[2] is story.PLAN_SCHEMA]
    assert len(plan_calls) == 2 and "YOUR PREVIOUS PLAN HAS THESE PROBLEMS" in plan_calls[1][1]
    assert "must start with a 'hook' beat" in plan_calls[1][1]
    assert bible["beats"][0]["purpose"] == "hook" and bible["beats"][-1]["purpose"] == "end_hook"
    assert bible["hooks_planned"] is True


def test_hook_check():
    assert story.hooks_ok(PLAN)
    assert not story.hooks_ok({"beats": [{"purpose": "setup"}, {"purpose": "end_hook"}]})
    assert not story.hooks_ok({"beats": [{"purpose": "hook"}, {"purpose": "resolution"}]})
    assert not story.hooks_ok({"beats": []})
