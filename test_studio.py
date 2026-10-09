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

from studio import _pcm_to_wav


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


if __name__ == "__main__":
    unittest.main()
