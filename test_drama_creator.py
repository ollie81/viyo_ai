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

from studio import _merged_rich_fields, _RICH_CHARACTER_FIELDS, CharacterSaveIn
from drama_creator import MAX_CHARACTERS, PlotOut, _compose_four_view_sheet


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


class MaxCharactersTests(unittest.TestCase):
    def test_matches_reference_ui_cap(self):
        self.assertEqual(MAX_CHARACTERS, 2)


class ComposeFourViewSheetTests(unittest.TestCase):
    def test_raises_a_clear_error_rather_than_a_placeholder_on_bad_url(self):
        # Never silently substitute a placeholder for a view that
        # failed to download/decode — see this feature's own honesty
        # notes in drama_creator.py's module docstring.
        with self.assertRaises(Exception):
            _compose_four_view_sheet(["not-a-real-url-at-all"])


if __name__ == "__main__":
    unittest.main()
