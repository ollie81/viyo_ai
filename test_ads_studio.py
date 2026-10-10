"""
Unit tests for Ads Studio's pure, network-free logic — the hook-scoring
formula and script-duration normalization. There is no existing test
runner anywhere in this codebase (confirmed: no test_*.py, no pytest in
requirements.txt) so this uses only the standard library's unittest,
runnable with `python3 test_ads_studio.py` or `python3 -m unittest
test_ads_studio` without adding a new dependency.

Does NOT import FastAPI/Supabase-touching endpoint code paths or hit any
network — ads_studio.py's module-level code (building supabase_admin /
_gemini_client) only reads env vars and is safe to import without them
set (both stay None, exercised nowhere in these tests).
"""
import unittest

from ads_studio import (
    HOOK_SCORE_WEIGHTS,
    _weighted_hook_score,
    _normalize_scene_durations,
    _scene_count_for_duration,
    _ad_target_dimensions,
    _veo_duration_for_scene,
)


class WeightedHookScoreTests(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(HOOK_SCORE_WEIGHTS.values()), 1.0, places=6)

    def test_all_tens_scores_ten(self):
        self.assertEqual(_weighted_hook_score(10, 10, 10, 10, 10), 10.0)

    def test_all_ones_scores_one(self):
        self.assertEqual(_weighted_hook_score(1, 1, 1, 1, 1), 1.0)

    def test_attention_weighted_highest(self):
        # Two candidates with the same total raw points (10+1+1+1+1 vs
        # 1+10+1+1+1) should NOT score equal — attention (0.30) outweighs
        # curiosity (0.25), proving the weights actually drive the
        # ranking rather than a flat average.
        attention_heavy = _weighted_hook_score(10, 1, 1, 1, 1)
        curiosity_heavy = _weighted_hook_score(1, 10, 1, 1, 1)
        self.assertGreater(attention_heavy, curiosity_heavy)

    def test_out_of_range_scores_are_clamped(self):
        # A malformed Gemini response (e.g. score of 15, or 0) must not
        # be able to skew a ranking outside the real 1-10 scale.
        clamped_high = _weighted_hook_score(15, 15, 15, 15, 15)
        clamped_low = _weighted_hook_score(0, 0, 0, 0, 0)
        self.assertEqual(clamped_high, 10.0)
        self.assertEqual(clamped_low, 1.0)

    def test_ranking_order_is_deterministic(self):
        a = _weighted_hook_score(9, 8, 7, 6, 5)
        b = _weighted_hook_score(5, 6, 7, 8, 9)
        self.assertNotEqual(a, b)
        self.assertEqual(a, _weighted_hook_score(9, 8, 7, 6, 5))


class SceneDurationNormalizationTests(unittest.TestCase):
    def test_scales_to_exact_target(self):
        scenes = [{"seconds": 2.0}, {"seconds": 2.0}, {"seconds": 2.0}]
        result = _normalize_scene_durations(scenes, 15.0)
        self.assertAlmostEqual(sum(s["seconds"] for s in result), 15.0, delta=0.5)

    def test_uneven_input_still_scales_proportionally(self):
        scenes = [{"seconds": 1.0}, {"seconds": 3.0}]
        result = _normalize_scene_durations(scenes, 8.0)
        # The 3:1 ratio between scenes should survive scaling.
        self.assertAlmostEqual(result[1]["seconds"] / result[0]["seconds"], 3.0, delta=0.2)

    def test_never_produces_non_positive_duration(self):
        scenes = [{"seconds": 0.0}, {"seconds": 0.0}]
        result = _normalize_scene_durations(scenes, 8.0)
        for s in result:
            self.assertGreater(s["seconds"], 0)


class SceneCountForDurationTests(unittest.TestCase):
    def test_monotonically_non_decreasing(self):
        counts = [_scene_count_for_duration(d) for d in (8, 15, 30, 45, 60)]
        self.assertEqual(counts, sorted(counts))

    def test_known_values(self):
        self.assertEqual(_scene_count_for_duration(8), 2)
        self.assertEqual(_scene_count_for_duration(60), 8)


class AdTargetDimensionsTests(unittest.TestCase):
    def test_vertical_720p(self):
        self.assertEqual(_ad_target_dimensions("9:16", "720p"), (720, 1280))

    def test_landscape_1080p(self):
        self.assertEqual(_ad_target_dimensions("16:9", "1080p"), (1920, 1080))

    def test_square_is_always_equal_sides(self):
        w, h = _ad_target_dimensions("1:1", "720p")
        self.assertEqual(w, h)
        w2, h2 = _ad_target_dimensions("1:1", "1080p")
        self.assertEqual(w2, h2)
        self.assertGreater(w2, w)


class VeoDurationForSceneTests(unittest.TestCase):
    def test_dialogue_picks_closest_allowed_duration(self):
        # Regression for the real bug: an 8s campaign splits into two
        # ~4s scenes, and every dialogue scene was coming back at a
        # flat 8s regardless of the plan, doubling the assembled video.
        self.assertEqual(_veo_duration_for_scene(4.0, has_dialogue=True), 4)
        self.assertEqual(_veo_duration_for_scene(3.0, has_dialogue=True), 4)
        self.assertEqual(_veo_duration_for_scene(5.0, has_dialogue=True), 4)
        self.assertEqual(_veo_duration_for_scene(5.5, has_dialogue=True), 6)
        self.assertEqual(_veo_duration_for_scene(7.0, has_dialogue=True), 6)
        self.assertEqual(_veo_duration_for_scene(8.0, has_dialogue=True), 8)
        self.assertEqual(_veo_duration_for_scene(100.0, has_dialogue=True), 8)

    def test_silent_scene_always_takes_shortest(self):
        # No speech to protect from looping — always the cheapest tier,
        # the renderer stretches/loops it to the planned length later.
        self.assertEqual(_veo_duration_for_scene(4.0, has_dialogue=False), 4)
        self.assertEqual(_veo_duration_for_scene(30.0, has_dialogue=False), 4)


if __name__ == "__main__":
    unittest.main()
