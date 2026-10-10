"""
Unit test for the _pcm_to_wav fix — Gemini TTS's returned PCM chunk has
no guaranteed alignment to the declared 16-bit sample width, and an
odd-length chunk used to get written straight into the WAV data chunk,
producing a malformed trailing sample that played back as an audible
click/pop ("weird sounds in the speech end", reported directly). Plain
stdlib unittest, matching test_ads_studio.py's own no-new-dependency
approach — no test runner existed anywhere in this codebase before that
file.
"""
import unittest
import wave
import io

from studio import _pcm_to_wav, _caption_metrics, _CAPTION_FONT_SIZE, _CAPTION_LINE_HEIGHT


class PcmToWavTests(unittest.TestCase):
    def test_even_length_is_preserved_exactly(self):
        pcm = bytes(range(20))  # 10 samples, already aligned
        wav_bytes = _pcm_to_wav(pcm, 24000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as f:
            self.assertEqual(f.readframes(f.getnframes()), pcm)

    def test_odd_length_is_truncated_not_corrupted(self):
        pcm = bytes(range(21))  # 10 full samples + 1 dangling byte
        wav_bytes = _pcm_to_wav(pcm, 24000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as f:
            frames = f.readframes(f.getnframes())
            # The dangling last byte must be dropped, not smuggled into
            # a malformed final frame — exactly the bug that produced
            # the audible artifact.
            self.assertEqual(frames, pcm[:-1])
            self.assertEqual(len(frames) % 2, 0)

    def test_wav_header_fields_are_correct(self):
        wav_bytes = _pcm_to_wav(bytes(100), 24000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as f:
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 2)
            self.assertEqual(f.getframerate(), 24000)


class CaptionMetricsTests(unittest.TestCase):
    def test_default_1080_matches_drama_studio_original_constants(self):
        # Drama Studio's only canvas is 1080 wide — the default must
        # reproduce the exact numbers every existing caller already
        # renders with, so this change is a pure no-op for it.
        font_size, line_height, border_w = _caption_metrics(1080)
        self.assertEqual(font_size, _CAPTION_FONT_SIZE)
        self.assertEqual(line_height, _CAPTION_LINE_HEIGHT)
        self.assertEqual(border_w, 3)

    def test_narrower_canvas_scales_down(self):
        # Regression for the real bug: Ads Studio's default 720p 9:16
        # output (720 wide) was burning in captions sized for a
        # 1080-wide frame, oversized relative to the actual frame and
        # prone to overflowing past the edges.
        font_size, _, _ = _caption_metrics(720)
        self.assertLess(font_size, _CAPTION_FONT_SIZE)

    def test_wider_canvas_scales_up(self):
        font_size, _, _ = _caption_metrics(1920)
        self.assertGreater(font_size, _CAPTION_FONT_SIZE)

    def test_font_size_never_collapses_to_unreadable(self):
        font_size, _, border_w = _caption_metrics(1)
        self.assertGreaterEqual(font_size, 28)
        self.assertGreaterEqual(border_w, 2)

    def test_scales_proportionally_with_width(self):
        # If font size tracks frame width, the same wrapped line
        # occupies the same fraction of the frame at any resolution —
        # this is what keeps _CAPTION_MAX_CHARS_PER_LINE (a plain
        # character count) still correct across resolutions.
        font_720, _, _ = _caption_metrics(720)
        font_1440, _, _ = _caption_metrics(1440)
        self.assertAlmostEqual(font_1440 / font_720, 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
