import os
import subprocess

import numpy as np
import pytest

import audio_features as af
import transcription as tr

SR = af.SAMPLE_RATE


def tone(freqs, seconds_each, amp=0.3):
    """Voiced-like signal: a harmonic tone per segment (fundamental +
    two harmonics) so autocorrelation pitch tracking has something real."""
    out = []
    for f in freqs:
        t = np.arange(int(SR * seconds_each)) / SR
        sig = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t) + 0.25 * np.sin(6 * np.pi * f * t)
        out.append(amp * sig / 1.75)
    return np.concatenate(out).astype(np.float32)


def test_pitch_tracker_finds_a_known_fundamental():
    ff = af.frame_features(tone([200.0], 2.0))
    voiced = ff.f0_hz[~np.isnan(ff.f0_hz)]
    assert voiced.size > 0.8 * ff.f0_hz.size
    assert np.median(voiced) == pytest.approx(200.0, rel=0.03)


def test_silence_is_not_speech_and_has_no_pitch():
    sig = np.concatenate([tone([180.0], 1.0), np.zeros(SR, dtype=np.float32), tone([180.0], 1.0)])
    ff = af.frame_features(sig)
    mid = (ff.times > 1.2) & (ff.times < 1.8)
    assert not ff.speech[mid].any()
    assert np.isnan(ff.f0_hz[mid]).all()


def _video_like_signal():
    """Five 30s stretches: flat delivery at normal volume, flat delivery
    much louder, animated pitch at normal volume, flat again, flat again."""
    flat = tone([150.0], 30.0, amp=0.2)
    loud_flat = tone([150.0], 30.0, amp=0.6)
    animated = tone([120, 180, 140, 220, 130, 200, 150, 240, 125, 190] * 3, 1.0, amp=0.2)
    return np.concatenate([flat, loud_flat, animated, flat, flat])


def test_animated_delivery_beats_merely_loud_delivery():
    sig = _video_like_signal()
    ff = af.frame_features(sig)
    duration = len(sig) / SR
    word_times = list(np.arange(0, duration, 0.4))
    baseline = af.build_baseline(ff, duration, word_times)

    def score(start):
        n = sum(1 for w in word_times if start <= w < start + 30)
        return af.delivery_score(af.window_stats(ff, start, start + 30, n), baseline)

    loud, loud_bd = score(30.0)
    animated, animated_bd = score(60.0)
    flat, _ = score(0.0)
    assert animated > loud, (animated_bd, loud_bd)
    assert animated > flat
    # Loudness alone moves the score only a little.
    assert loud - flat < 15


def test_decode_pcm_reads_real_media(tmp_path):
    path = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=2",
                    str(path)], check=True)
    samples = af.decode_pcm(str(path))
    assert len(samples) == pytest.approx(2 * SR, rel=0.01)


# ---------------------------------------------------------------- transcription chunking

def test_chunk_cuts_land_in_the_longest_nearby_pause():
    silences = [(580.0, 580.4), (560.0, 562.0), (1150.0, 1150.5), (1205.0, 1207.0)]
    chunks = tr.plan_chunks(1500.0, silences)
    cuts = [c[1] for c in chunks[:-1]]
    assert cuts[0] == pytest.approx(561.0)
    assert cuts[1] == pytest.approx(1150.25)
    assert chunks[0][0] == 0 and chunks[-1][1] == 1500.0
    assert all(b[0] == a[1] for a, b in zip(chunks, chunks[1:]))


def test_no_pause_nearby_falls_back_to_the_target_mark():
    chunks = tr.plan_chunks(1300.0, [])
    assert [round(c[1]) for c in chunks] == [600, 1200, 1300]


def test_short_audio_is_one_chunk():
    assert tr.plan_chunks(400.0, []) == [(0.0, 400.0)]


def test_transcribe_file_offsets_words_and_drops_out_of_range_artifacts(tmp_path):
    src = tmp_path / "speech.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=200:duration=1300", str(src)], check=True)
    seen_prompts = []

    def fake_chunk(path, prompt):
        seen_prompts.append(prompt)
        n = len(seen_prompts)
        return {"text": f"chunk {n} text.", "language": "english",
                "segments": [{"start": 1.0, "end": 2.0, "text": f"chunk {n} text."},
                             {"start": 9999.0, "end": 9999.5, "text": "artifact"}],
                "words": [{"word": "chunk", "start": 1.0, "end": 1.4},
                          {"word": str(n), "start": 1.5, "end": 1.9},
                          {"word": "ghost", "start": 9999.0, "end": 9999.4}]}

    result = tr.transcribe_file(str(src), 1300.0, str(tmp_path), fake_chunk)
    starts = [w["start"] for w in result["words"]]
    assert "ghost" not in [w["word"] for w in result["words"]]
    assert starts[0] == pytest.approx(1.0)
    assert starts[2] == pytest.approx(601.0, abs=0.1)
    assert starts[4] == pytest.approx(1201.0, abs=0.1)
    assert seen_prompts[0] is None and seen_prompts[1] == "chunk 1 text."
    assert result["language"] == "english"
    assert not any(f.endswith(".mp3") for f in os.listdir(tmp_path))
