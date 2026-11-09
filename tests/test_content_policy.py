"""Viyo makes adult fiction: only what no app may allow is refused."""
import types

import studio_ai
import studio_story as story


def result(flags=(), scores=None):
    cats = types.SimpleNamespace(**{f: True for f in flags})
    return types.SimpleNamespace(categories=cats, category_scores=types.SimpleNamespace(**(scores or {})))


def test_violence_crime_and_adult_romance_are_allowed():
    assert studio_ai.story_allowed(result(["violence", "violence_graphic", "sexual", "harassment", "illicit"]))


def test_minors_self_harm_how_to_and_real_weapon_instructions_are_refused():
    assert not studio_ai.story_allowed(result(["sexual_minors"]))
    assert not studio_ai.story_allowed(result(["self_harm_instructions"]))
    assert not studio_ai.story_allowed(result(scores={"illicit_violent": 0.95}))
    assert studio_ai.story_allowed(result(["illicit_violent"], {"illicit_violent": 0.6}))


def test_moderation_uses_the_policy(monkeypatch):
    import openai
    seen = {}

    class Fake:
        def __init__(self, **kw):
            self.moderations = self

        def create(self, model, input):
            seen["input"] = input
            flags = ["sexual_minors"] if "child" in input else ["violence"]
            return types.SimpleNamespace(results=[result(flags)])

    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-looking")
    monkeypatch.setattr(openai, "OpenAI", Fake)
    assert studio_ai.moderate_text("He holds a revolver to Julian's head.")
    assert not studio_ai.moderate_text("a child in a sexual scene")


def test_google_filters_block_only_high_severity():
    from google.genai import types as gt
    s = studio_ai.drama_safety()
    assert len(s) == 4 and all(x.threshold == gt.HarmBlockThreshold.BLOCK_ONLY_HIGH for x in s)

    seen = []

    class Models:
        def generate_content(self, model, contents, config):
            seen.append(config)
            return types.SimpleNamespace(text="INT. ROOM - NIGHT", usage_metadata=None)

    g = object.__new__(studio_ai.GoogleStudioAI)
    g.client = types.SimpleNamespace(models=Models())
    g._text("write")
    assert seen[0].safety_settings and seen[0].safety_settings[0].threshold == gt.HarmBlockThreshold.BLOCK_ONLY_HIGH


def test_writers_are_told_mature_themes_are_welcome_but_sex_stays_implied():
    p = story.plan_prompt({"title": "T"}, [])
    assert "Mature themes are welcome" in p and "never explicit" in p
    assert "Characters under 18 are never" in p
