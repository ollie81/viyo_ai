"""Local speech-to-text (faster-whisper) gives what captions need: words with timings."""
import os
import subprocess

import pytest

import transcription

SAMPLE = "/tmp/eval_media/C_live_test_7min.mp4"


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="needs the local speech sample")
def test_local_whisper_returns_timed_words_in_the_video_timeline(tmp_path):
    clip = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0", "-t", "40", "-i", SAMPLE, "-c", "copy", clip], check=True)
    out = transcription.transcribe_file(clip, 40.0, str(tmp_path), transcription.local_chunk_transcriber("small"))
    assert out["language"] == "en"
    words = out["words"]
    assert len(words) > 60, "a 40 s talk has plenty of words"
    assert all(0 <= w["start"] <= w["end"] <= 40.5 for w in words)
    assert all(a["start"] <= b["start"] for a, b in zip(words, words[1:])), "in order"
    assert "1985" in out["text"] or "85" in out["text"]


def test_cache_key_separates_engines(monkeypatch):
    monkeypatch.setenv("REPURPOSE_LOCAL_WHISPER_MODEL", "small")
    assert transcription.transcript_cache_key(True) == "faster-whisper-small"
    assert transcription.transcript_cache_key(False) == "whisper-1"
