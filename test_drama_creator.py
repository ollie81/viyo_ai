"""
Unit tests for Drama Creator's pure, network-free logic, plus the
_merged_rich_fields regression fix in studio.py (save_cast). Plain
stdlib unittest, matching this codebase's established no-test-runner
convention — see test_ads_studio.py's own docstring.

Does NOT import FastAPI endpoint code paths or hit any network —
studio.py/drama_creator.py's module-level code (building supabase_admin/
_gemini_client) only reads env vars and is safe to import without them
set.
"""
import unittest
from types import SimpleNamespace

from studio import _merged_rich_fields, _RICH_CHARACTER_FIELDS, CharacterSaveIn
from drama_creator import (
    MAX_CHARACTERS,
    PlotOut,
    _compose_four_view_sheet,
    _estimate_openai_text_cost_cents,
    _openai_text_cost_cents,
    OPENAI_TEXT_INPUT_USD_PER_1M_TOKENS,
    OPENAI_TEXT_OUTPUT_USD_PER_1M_TOKENS,
)


class MergedRichFieldsTests(unittest.TestCase):
    def _character(self, **overrides):
        base = dict(
            name="Olivia", age="mid-20s", gender="female", appearance="", clothing="", personality="",
            skin_tone="", face_shape="", eye_color="", hairstyle="", body_proportions="", voice_description="",
        )
        base.update(overrides)
        return CharacterSaveIn(**base)

    def test_incoming_value_wins_when_present(self):
        c = self._character(skin_tone="fair")
        existing = {"skin_tone": "olive"}
        result = _merged_rich_fields(c, existing, _RICH_CHARACTER_FIELDS)
        self.assertEqual(result["skin_tone"], "fair")

    def test_blank_incoming_falls_back_to_existing_row(self):
        # The exact regression this fixes: the old paste-a-script
        # screen's save request has no idea skin_tone exists, so it
        # always arrives blank — that must NOT blank out a value an
        # AI-assistant-created character already had.
        c = self._character(skin_tone="")
        existing = {"skin_tone": "olive"}
        result = _merged_rich_fields(c, existing, _RICH_CHARACTER_FIELDS)
        self.assertEqual(result["skin_tone"], "olive")

    def test_blank_incoming_with_no_existing_row_omits_the_key(self):
        # A brand-new character (insert, not update) with nothing set
        # for this field yet — the column's own DB default should
        # apply, not an explicit blank written over it.
        c = self._character(skin_tone="")
        result = _merged_rich_fields(c, existing_row=None, field_names=_RICH_CHARACTER_FIELDS)
        self.assertNotIn("skin_tone", result)

    def test_covers_every_rich_field(self):
        c = self._character(
            skin_tone="a", face_shape="b", eye_color="c", hairstyle="d",
            body_proportions="e", voice_description="f",
        )
        result = _merged_rich_fields(c, existing_row=None, field_names=_RICH_CHARACTER_FIELDS)
        for field in ("skin_tone", "face_shape", "eye_color", "hairstyle", "body_proportions", "voice_description"):
            self.assertIn(field, result)


class PlotOutTests(unittest.TestCase):
    def test_defaults_to_empty_strings_not_none(self):
        # A series with no plot_json yet (every series created before
        # this feature existed) must render as a blank-but-displayable
        # plot card, never a None that crashes string formatting.
        plot = PlotOut()
        for value in plot.model_dump().values():
            self.assertEqual(value, "")

    def test_partial_dict_fills_remaining_fields_blank(self):
        plot = PlotOut(**{"premise": "A woman returns.", "cliffhanger": "He already knew."})
        self.assertEqual(plot.premise, "A woman returns.")
        self.assertEqual(plot.main_conflict, "")
        self.assertEqual(plot.cliffhanger, "He already knew.")

    def test_has_a_dedicated_opening_hook_field(self):
        # The field the storyboard prompt builds scene 1 around — must
        # survive a round trip distinctly from premise/cliffhanger.
        plot = PlotOut(opening_hook="She finds the letter already open.")
        self.assertEqual(plot.opening_hook, "She finds the letter already open.")
        self.assertEqual(plot.premise, "")


class OpenAiCostTests(unittest.TestCase):
    def test_estimate_scales_with_prompt_length(self):
        short = _estimate_openai_text_cost_cents("hi")
        long = _estimate_openai_text_cost_cents("word " * 5000)
        self.assertGreater(long, short)

    def test_estimate_uses_real_pricing_constants_not_invented_ones(self):
        # gpt-6-astra: $10/1M input, $50/1M output — verified against
        # OpenAI's own pricing page, not guessed (see drama_creator.py's
        # own comment on where these numbers came from).
        self.assertEqual(OPENAI_TEXT_INPUT_USD_PER_1M_TOKENS, 10.0)
        self.assertEqual(OPENAI_TEXT_OUTPUT_USD_PER_1M_TOKENS, 50.0)

    def test_real_cost_reads_usage_object_correctly(self):
        usage = SimpleNamespace(input_tokens=100_000, output_tokens=10_000)
        # 100k input @ $10/1M = $1.00; 10k output @ $50/1M = $0.50 -> $1.50 -> 150 cents
        self.assertEqual(_openai_text_cost_cents(usage), 150)

    def test_none_usage_costs_nothing(self):
        self.assertEqual(_openai_text_cost_cents(None), 0)

    def test_real_cost_never_rounds_down_to_zero_for_a_nonzero_call(self):
        usage = SimpleNamespace(input_tokens=1, output_tokens=0)
        self.assertGreaterEqual(_openai_text_cost_cents(usage), 1)


class MaxCharactersTests(unittest.TestCase):
    def test_raised_from_the_original_reference_ui_cap_of_2(self):
        # Raised once a real drama (lost-heir plot needing protagonist
        # + antagonist + love interest + a separate secret-keeper)
        # showed 2 was a v1 scaffolding choice, not a real limit.
        self.assertEqual(MAX_CHARACTERS, 4)


class ComposeFourViewSheetTests(unittest.TestCase):
    def test_raises_a_clear_error_rather_than_a_placeholder_on_bad_url(self):
        # Never silently substitute a placeholder for a view that
        # failed to download/decode — see this feature's own honesty
        # notes in drama_creator.py's module docstring.
        with self.assertRaises(Exception):
            _compose_four_view_sheet(["not-a-real-url-at-all"])


if __name__ == "__main__":
    unittest.main()
