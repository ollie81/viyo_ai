"""
Full-video transcription with word-level timestamps.

whisper-1 (OpenAI API) is used because it returns word timestamps; the
upload limit per request means long audio is split into chunks. Chunks
used to be cut every 600s exactly, which splits whatever word is being
spoken at that instant and loses it from both chunks. Cuts now move to
the middle of the longest pause near each 600s mark, and each chunk is
given the previous chunk's last words as a prompt so casing and
punctuation stay consistent across the seam.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Callable, Optional

WHISPER_MODEL = "whisper-1"
CHUNK_TARGET_SECONDS = 600.0
# How far before the target a cut may move to find a pause, and how far
# after (smaller: chunks must stay under the API's size limit).
CHUNK_SEARCH_BEFORE_SECONDS = 90.0
CHUNK_SEARCH_AFTER_SECONDS = 30.0
SILENCE_NOISE_DB = -35
SILENCE_MIN_SECONDS = 0.3
PROMPT_TAIL_CHARS = 200
FFMPEG_TIMEOUT_SECONDS = 900

ChunkTranscriber = Callable[[str, Optional[str]], dict]


def detect_silences(path: str) -> list[tuple[float, float]]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn",
         "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_SECONDS}",
         "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=FFMPEG_TIMEOUT_SECONDS,
    )
    silences = []
    start = None
    for line in proc.stderr.splitlines():
        m = re.search(r"silence_start: (-?[\d.]+)", line)
        if m:
            start = max(0.0, float(m.group(1)))
            continue
        m = re.search(r"silence_end: ([\d.]+)", line)
        if m and start is not None:
            silences.append((start, float(m.group(1))))
            start = None
    return silences


def plan_chunks(duration: float, silences: list[tuple[float, float]],
                target: float = CHUNK_TARGET_SECONDS) -> list[tuple[float, float]]:
    """[(start, end), ...] covering [0, duration] with cuts inside pauses."""
    boundaries = [0.0]
    pos = 0.0
    while duration - pos > target + CHUNK_SEARCH_AFTER_SECONDS:
        ideal = pos + target
        window = [s for s in silences
                  if ideal - CHUNK_SEARCH_BEFORE_SECONDS <= (s[0] + s[1]) / 2 <= ideal + CHUNK_SEARCH_AFTER_SECONDS
                  and (s[0] + s[1]) / 2 > pos + 1]
        if window:
            best = max(window, key=lambda s: s[1] - s[0])
            cut = (best[0] + best[1]) / 2
        else:
            cut = ideal
        boundaries.append(cut)
        pos = cut
    boundaries.append(duration)
    return list(zip(boundaries[:-1], boundaries[1:]))


def extract_chunk(path: str, start: float, end: float, out_path: str) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", path,
         "-t", f"{end - start:.3f}", "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libmp3lame", "-b:a", "64k", out_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=FFMPEG_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"audio chunk extraction failed: {proc.stderr[-300:]}")


def transcribe_file(path: str, duration: float, work_dir: str, transcribe_chunk: ChunkTranscriber,
                    on_progress: Optional[Callable[[float], None]] = None) -> dict:
    """
    Returns {"text", "segments", "words", "language", "duration"} in the
    source video's timeline. Errors from `transcribe_chunk` propagate.
    """
    chunks = plan_chunks(duration, detect_silences(path))
    all_words: list[dict] = []
    all_segments: list[dict] = []
    texts: list[str] = []
    language = None
    prompt: Optional[str] = None

    for i, (start, end) in enumerate(chunks):
        chunk_path = os.path.join(work_dir, f"chunk_{i:03d}.mp3")
        extract_chunk(path, start, end, chunk_path)
        try:
            data = transcribe_chunk(chunk_path, prompt)
        finally:
            os.remove(chunk_path)

        language = language or data.get("language")
        text = (data.get("text") or "").strip()
        if text:
            texts.append(text)
            prompt = text[-PROMPT_TAIL_CHARS:]

        chunk_len = end - start
        for seg in data.get("segments") or []:
            s, e = float(seg.get("start", 0)), float(seg.get("end", 0))
            if s >= chunk_len + 0.5:
                continue
            all_segments.append({"start": round(start + s, 3), "end": round(start + min(e, chunk_len), 3),
                                 "text": str(seg.get("text", ""))})
        for w in data.get("words") or []:
            s, e = float(w.get("start", 0)), float(w.get("end", 0))
            # whisper occasionally timestamps a word past the end of the
            # audio it was given; those are artefacts, not speech.
            if s >= chunk_len + 0.5:
                continue
            all_words.append({"word": str(w.get("word", "")).strip(),
                              "start": round(start + s, 3), "end": round(start + min(e, chunk_len), 3)})

        if on_progress:
            on_progress((i + 1) / len(chunks))

    all_words.sort(key=lambda w: w["start"])
    all_segments.sort(key=lambda s: s["start"])
    return {"text": " ".join(texts).strip(), "segments": all_segments, "words": all_words,
            "language": language, "duration": duration}


def openai_chunk_transcriber(client, model: str = WHISPER_MODEL) -> ChunkTranscriber:
    """whisper-1 verbose_json with word timestamps, errors classified."""
    import openai
    from repurpose_jobs import PermanentJobError, TransientJobError

    def run(chunk_path: str, prompt: Optional[str]) -> dict:
        kwargs = {}
        if prompt:
            kwargs["prompt"] = prompt
        try:
            with open(chunk_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model=model, file=f, response_format="verbose_json",
                    timestamp_granularities=["segment", "word"], **kwargs,
                )
        except openai.BadRequestError as e:
            raise PermanentJobError(
                "transcription_rejected",
                "We couldn't transcribe this video's audio. Try a different file.",
                f"whisper rejected chunk: {e}")
        except openai.APIError as e:
            raise TransientJobError(
                "transcription_unavailable",
                "Transcription is temporarily unavailable.", f"whisper error: {e}")
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)

    return run


# ---------------------------------------------------------------- local (faster-whisper)
#
# whisper-1 is switched off by OpenAI on 26 Feb 2027, and its replacements
# don't return word timestamps, which the captions depend on. This runs the
# open Whisper model on the worker itself instead (faster-whisper, int8 on
# CPU): same word timestamps, ~99 languages, and no per-minute API cost.
# Turned on with REPURPOSE_TRANSCRIBER=local on the worker.

LOCAL_MODEL_DEFAULT = "small"
_local_models: dict = {}


def local_model_name() -> str:
    return os.environ.get("REPURPOSE_LOCAL_WHISPER_MODEL", LOCAL_MODEL_DEFAULT)


def transcript_cache_key(local: bool) -> str:
    """Transcripts are cached per engine, so switching never mixes them."""
    return f"faster-whisper-{local_model_name()}" if local else WHISPER_MODEL


def _load_local_model(name: str):
    model = _local_models.get(name)
    if model is None:
        from faster_whisper import WhisperModel
        threads = int(os.environ.get("REPURPOSE_LOCAL_WHISPER_THREADS", "0")) or (os.cpu_count() or 2)
        model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
        _local_models[name] = model
    return model


def local_chunk_transcriber(model_name: Optional[str] = None) -> ChunkTranscriber:
    """Same output shape as whisper-1 verbose_json: text, segments, words, language."""
    from repurpose_jobs import PermanentJobError, TransientJobError

    def run(chunk_path: str, prompt: Optional[str]) -> dict:
        name = model_name or local_model_name()
        try:
            model = _load_local_model(name)
        except Exception as e:
            raise TransientJobError("transcription_unavailable", "Transcription is temporarily unavailable.",
                                    f"local whisper model {name} failed to load: {e}")
        try:
            segments, info = model.transcribe(
                chunk_path, word_timestamps=True, vad_filter=True, beam_size=1,
                initial_prompt=prompt or None, condition_on_previous_text=False)
            segs, words, texts = [], [], []
            for seg in segments:
                text = (seg.text or "").strip()
                if text:
                    texts.append(text)
                segs.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text or ""})
                for w in seg.words or []:
                    words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
        except Exception as e:
            raise PermanentJobError("transcription_rejected",
                                    "We couldn't transcribe this video's audio. Try a different file.",
                                    f"local whisper failed: {e}")
        return {"text": " ".join(texts), "segments": segs, "words": words, "language": info.language}

    return run
