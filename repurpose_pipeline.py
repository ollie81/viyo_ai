"""
One AI Repurposer job, start to finish, as run by the worker.

Stages (progress range):
  downloading 0.00-0.08 -> probing 0.08-0.10 -> transcribing 0.10-0.45
  -> analyzing 0.45-0.52 -> selecting 0.52-0.60 -> rendering 0.60-0.97
  -> finalizing 0.97-1.00

Every expensive stage is resumable: the transcript is cached by the
source file's SHA-256, the clip selection is stored in the job's
checkpoint, and clips already rendered and uploaded are skipped. A
retry after a crash or redeploy therefore re-downloads the source but
doesn't pay for transcription, the LLM, or finished renders again.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import HTTPException

import clip_selection
from audio_features import build_baseline, decode_pcm, delivery_score, frame_features, window_stats
from repurpose_jobs import JobStore, LeaseLost, PermanentJobError, ShutdownRequested, TransientJobError
from repurpose_storage import CLIPS_BUCKET, SourceMissing, SourceTooLarge, StorageUnavailable
from transcription import WHISPER_MODEL, ChunkTranscriber, transcribe_file


@dataclass
class PipelineDeps:
    storage: object
    transcribe_chunk: ChunkTranscriber
    llm: clip_selection.LLMFn
    max_input_bytes: int
    max_input_seconds: float
    transcription_model: str = WHISPER_MODEL
    min_clip_score: int = clip_selection.DEFAULT_MIN_SCORE
    # The footage critique is an extra LLM call the app already shows;
    # tests turn it off.
    critique: bool = True


class JobContext:
    """What the pipeline can do to its job: report progress, store
    checkpoints, and find out it should stop. Progress is buffered for
    the heartbeat thread; stage changes and checkpoints write through."""

    def __init__(self, store: JobStore, job: dict, state: "HeartbeatState"):
        self.store = store
        self.job = job
        self.job_id = job["id"]
        self.lease = job["lease_token"]
        self.state = state

    def check(self) -> None:
        if self.state.lease_lost:
            raise LeaseLost(self.job_id)
        if self.state.shutdown:
            raise ShutdownRequested(self.job_id)

    def stage(self, stage: str, progress: float, message: str) -> None:
        self.check()
        self.state.update(stage, progress, message)
        if not self.store.heartbeat(self.job_id, self.lease, stage, progress, message):
            self.state.lease_lost = True
            raise LeaseLost(self.job_id)

    def progress(self, progress: float, message: Optional[str] = None) -> None:
        self.state.update(None, progress, message)

    def checkpoint(self, data: dict) -> None:
        self.check()
        if not self.store.heartbeat(self.job_id, self.lease, checkpoint=data):
            self.state.lease_lost = True
            raise LeaseLost(self.job_id)
        self.job.setdefault("checkpoint", {}).update(data)

    def save_clip(self, clip: dict) -> None:
        self.check()
        if not self.store.save_clip(self.job_id, self.lease, clip):
            self.state.lease_lost = True
            raise LeaseLost(self.job_id)


class HeartbeatState:
    def __init__(self):
        self.stage: Optional[str] = None
        self.progress: Optional[float] = None
        self.message: Optional[str] = None
        self.lease_lost = False
        self.shutdown = False

    def update(self, stage, progress, message):
        if stage is not None:
            self.stage = stage
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_source(job: dict, deps: PipelineDeps, dest: str) -> int:
    try:
        if job["source_kind"] == "private_upload":
            return deps.storage.download(job["source_bucket"], job["source_path"], dest, deps.max_input_bytes)
        return deps.storage.download_url(job["source_url"], dest, deps.max_input_bytes)
    except SourceMissing as e:
        raise PermanentJobError(
            "source_missing", "We couldn't find your uploaded video. Please upload it again.", str(e))
    except SourceTooLarge as e:
        raise PermanentJobError(
            "source_too_large",
            f"This video is too large — the limit is {deps.max_input_bytes // (1024 * 1024)} MB.", str(e))
    except StorageUnavailable as e:
        raise TransientJobError("storage_unavailable", "Couldn't download your video.", str(e))


def _probe(path: str, deps: PipelineDeps) -> tuple[float, bool]:
    import repurpose as r
    try:
        duration = r._run_ffprobe_duration(path)
    except HTTPException as e:
        raise PermanentJobError("unreadable_video",
                                "This file couldn't be read as a video. It may be corrupt.", str(e.detail))
    if duration <= 0:
        raise PermanentJobError("unreadable_video", "This file couldn't be read as a video.", "zero duration")
    if duration > deps.max_input_seconds:
        raise PermanentJobError(
            "source_too_long",
            f"This video is too long — the limit is {int(deps.max_input_seconds // 60)} minutes.",
            f"{duration:.0f}s")
    return duration, r._has_audio_stream(path)


def _transcript(ctx: JobContext, deps: PipelineDeps, path: str, sha: str, duration: float,
                work_dir: str) -> dict:
    cached = ctx.store.get_transcript(sha, deps.transcription_model)
    if cached:
        ctx.stage("transcribing", 0.45, "Transcript ready")
        return cached

    def on_progress(fraction: float) -> None:
        ctx.check()
        ctx.progress(0.10 + 0.35 * fraction, f"Transcribing ({int(fraction * 100)}%)")

    transcript = transcribe_file(path, duration, work_dir, deps.transcribe_chunk, on_progress)
    ctx.store.put_transcript(sha, deps.transcription_model, transcript)
    return transcript


def _delivery_scorer(path: str, words: list[dict], duration: float):
    """Vocal-delivery scorer, or None if the audio can't be analysed — a
    missing signal lowers precision, it doesn't fail the job."""
    try:
        ff = frame_features(decode_pcm(path, timeout=900))
        baseline = build_baseline(ff, duration, [float(w["start"]) for w in words])
    except Exception as e:
        print(f"[WARN] delivery analysis unavailable: {e}")
        return None

    def score(start: float, end: float, n_words: int):
        return delivery_score(window_stats(ff, start, end, n_words), baseline)

    return score


def _select(ctx: JobContext, deps: PipelineDeps, path: str, transcript: dict, duration: float) -> dict:
    existing = (ctx.job.get("checkpoint") or {}).get("selection")
    if existing:
        return existing

    words, segments = transcript.get("words") or [], transcript.get("segments") or []
    ctx.stage("analyzing", 0.45, "Analysing delivery")
    scorer = _delivery_scorer(path, words, duration)

    ctx.stage("selecting", 0.52, "Finding the strongest moments")
    try:
        result = clip_selection.run_selection(words, segments, duration, deps.llm, scorer, deps.min_clip_score)
    except clip_selection.SelectionUnavailable as e:
        raise TransientJobError("selection_unavailable", "Clip selection is temporarily unavailable.", str(e))

    summary = result.summary()
    summary["delivery_signal"] = scorer is not None
    if not result.clips:
        raise PermanentJobError("no_strong_moments", result.note, str(summary))

    selection = {"clips": [c.to_dict() for c in result.clips], "summary": summary}
    ctx.checkpoint({"selection": selection})
    for rank, c in enumerate(result.clips, start=1):
        ctx.save_clip({
            "rank": rank, "status": "selected", "start_time": c.start, "end_time": c.end,
            "score": c.score, "score_breakdown": c.breakdown, "category": c.category,
            "title": c.title, "reason": c.reason, "hook_line": c.hook_line,
            "caption": c.caption, "hashtags": c.hashtags, "transcript": c.transcript,
        })
    return selection


def _prompt_selection(ctx: JobContext, job: dict, duration: float) -> dict:
    """Silent/music-only video with a creator description: one clip,
    packaged from the description (see repurpose._highlights_from_prompt)."""
    existing = (ctx.job.get("checkpoint") or {}).get("selection")
    if existing:
        return existing
    import repurpose as r
    ctx.stage("selecting", 0.52, "Writing your clip from your description")
    h = r._highlights_from_prompt(job["creative_prompt"], duration)
    clip = {
        "start": h.start_time, "end": h.end_time, "score": h.score, "category": "prompt",
        "title": h.suggested_title, "reason": h.reason, "hook_line": h.hook_line,
        "caption": h.caption, "hashtags": h.hashtags, "transcript": "", "breakdown": {},
    }
    selection = {"clips": [clip], "summary": {"target": 1, "selected": 1, "note": "",
                                              "from_creative_prompt": True}}
    ctx.checkpoint({"selection": selection})
    ctx.save_clip({"rank": 1, "status": "selected", "start_time": clip["start"], "end_time": clip["end"],
                   "score": clip["score"], "category": "prompt", "title": clip["title"],
                   "reason": clip["reason"], "hook_line": clip["hook_line"], "caption": clip["caption"],
                   "hashtags": clip["hashtags"], "transcript": ""})
    return selection


def _render_one(ctx: JobContext, deps: PipelineDeps, source: str, work_dir: str, rank: int,
                clip: dict, transcript: dict) -> dict:
    import repurpose as r
    words, segments = transcript.get("words") or [], transcript.get("segments") or []
    blocks = r._find_speech_blocks(segments, clip["start"], clip["end"])
    dead_air = r._dead_air_removed_seconds(blocks)
    kept_before = sum(e - s for s, e in blocks)
    blocks, filler_count = r._remove_filler_words(blocks, words)
    filler_seconds = round(kept_before - sum(e - s for s, e in blocks), 2)

    captions = os.path.join(work_dir, f"captions_{rank}.ass")
    if not r._generate_ass(words, captions, blocks, clip.get("hook_line", "")):
        captions = os.path.join(work_dir, f"captions_{rank}.srt")
        r._generate_srt(segments, captions, blocks)

    out = os.path.join(work_dir, f"clip_{rank}.mp4")
    try:
        r._render_clip(source, out, captions, blocks)
    except HTTPException as e:
        raise RuntimeError(str(e.detail))

    user_id, job_id = ctx.job["user_id"], ctx.job_id
    base = f"{user_id}/{job_id}/clip_{rank}"
    video_url = deps.storage.upload(CLIPS_BUCKET, f"{base}.mp4", out, "video/mp4")

    thumbnail_path = thumbnail_url = None
    try:
        frames = r._extract_thumbnail_candidates(out, work_dir, r.THUMBNAIL_CANDIDATE_COUNT)
        if frames:
            pick = r._pick_best_thumbnail(frames) if deps.critique else len(frames) // 2
            thumbnail_path = f"{base}_thumb.jpg"
            thumbnail_url = deps.storage.upload(CLIPS_BUCKET, thumbnail_path, frames[pick], "image/jpeg")
    except Exception as e:
        print(f"[WARN] thumbnail failed for clip {rank}: {e}")

    quote_path = quote_url = None
    try:
        card = os.path.join(work_dir, f"quote_{rank}.png")
        r._render_quote_card(clip["title"], card)
        quote_path = f"{base}_quote.png"
        quote_url = deps.storage.upload(CLIPS_BUCKET, quote_path, card, "image/png")
    except Exception as e:
        print(f"[WARN] quote card failed for clip {rank}: {e}")

    render_meta = {"dead_air_removed_seconds": dead_air,
                   "filler_words_removed_seconds": filler_seconds,
                   "filler_words_removed_count": filler_count}
    ctx.save_clip({
        "rank": rank, "status": "rendered", "start_time": clip["start"], "end_time": clip["end"],
        "score": clip["score"], "score_breakdown": clip.get("breakdown"), "category": clip.get("category"),
        "title": clip["title"], "reason": clip["reason"], "hook_line": clip.get("hook_line"),
        "caption": clip.get("caption"), "hashtags": clip.get("hashtags") or [],
        "transcript": clip.get("transcript"), "video_path": f"{base}.mp4", "video_url": video_url,
        "thumbnail_path": thumbnail_path, "thumbnail_url": thumbnail_url,
        "quote_card_path": quote_path, "quote_card_url": quote_url, "render_meta": render_meta,
    })
    for f in (out,):
        if os.path.exists(f):
            os.remove(f)
    return {"video_url": video_url, "thumbnail_url": thumbnail_url, "quote_card_url": quote_url,
            **render_meta}


def _legacy_clip(clip: dict, rendered: dict) -> dict:
    """The original RepurposeClipResult shape, plus additive fields."""
    return {
        "processed_video_url": rendered["video_url"],
        "highlight": {
            "start_time": clip["start"], "end_time": clip["end"], "reason": clip["reason"],
            "suggested_title": clip["title"], "score": clip["score"],
            "caption": clip.get("caption", ""), "hashtags": clip.get("hashtags") or [],
            "hook_line": clip.get("hook_line", ""), "category": clip.get("category"),
            "transcript": clip.get("transcript", ""), "score_breakdown": clip.get("breakdown") or {},
        },
        "dead_air_removed_seconds": rendered.get("dead_air_removed_seconds", 0.0),
        "filler_words_removed_seconds": rendered.get("filler_words_removed_seconds", 0.0),
        "filler_words_removed_count": rendered.get("filler_words_removed_count", 0),
        "quote_card_url": rendered.get("quote_card_url"),
        "thumbnail_url": rendered.get("thumbnail_url"),
    }


def process_job(ctx: JobContext, deps: PipelineDeps) -> dict:
    job = ctx.job
    with tempfile.TemporaryDirectory(prefix=f"repurpose_{ctx.job_id[:8]}_") as work:
        source = os.path.join(work, "source")
        ctx.stage("downloading", 0.0, "Downloading your video")
        _fetch_source(job, deps, source)

        ctx.stage("probing", 0.08, "Checking your video")
        duration, has_audio = _probe(source, deps)
        sha = _sha256(source)
        ctx.checkpoint({"source": {"sha256": sha, "duration": duration, "has_audio": has_audio}})

        if has_audio:
            ctx.stage("transcribing", 0.10, "Transcribing")
            transcript = _transcript(ctx, deps, source, sha, duration, work)
        elif job.get("creative_prompt"):
            transcript = {"text": "", "segments": [], "words": []}
        else:
            raise PermanentJobError(
                "no_audio",
                "This video has no audio track — the AI Repurposer needs speech to find moments. "
                "Try a video with sound, or describe the clip you want instead.")

        if transcript.get("words"):
            selection = _select(ctx, deps, source, transcript, duration)
        elif job.get("creative_prompt"):
            selection = _prompt_selection(ctx, job, duration)
        else:
            raise PermanentJobError(
                "no_speech",
                "We couldn't hear any speech in this video, so there are no moments to clip. "
                "Try a video with talking, or describe the clip you want instead.")

        existing = {c["rank"]: c for c in (ctx.store.get_job(ctx.job_id) or {}).get("clips", [])}
        clips = selection["clips"]
        results: list[dict] = []
        render_failures: list[str] = []
        for rank, clip in enumerate(clips, start=1):
            ctx.stage("rendering", 0.60 + 0.37 * (rank - 1) / len(clips),
                      f"Editing clip {rank} of {len(clips)}")
            done = existing.get(rank)
            if done and done.get("status") == "rendered" and done.get("video_url"):
                rendered = {"video_url": done["video_url"], "thumbnail_url": done.get("thumbnail_url"),
                            "quote_card_url": done.get("quote_card_url"), **(done.get("render_meta") or {})}
            else:
                try:
                    rendered = _render_one(ctx, deps, source, work, rank, clip, transcript)
                except (LeaseLost, ShutdownRequested):
                    raise
                except StorageUnavailable as e:
                    raise TransientJobError("storage_unavailable", "Couldn't save your clips.", str(e))
                except Exception as e:
                    render_failures.append(f"clip {rank}: {e}")
                    print(f"[WARN] render failed for clip {rank} of job {ctx.job_id}: {e}")
                    continue
            results.append(_legacy_clip(clip, rendered))

        if not results:
            raise TransientJobError("render_failed", "We couldn't render your clips.",
                                    "; ".join(render_failures)[:2000])

        ctx.stage("finalizing", 0.97, "Finishing up")
        feedback = None
        if deps.critique and transcript.get("text"):
            import repurpose as r
            signals = r._measure_video_signals(transcript.get("words") or [],
                                               transcript.get("segments") or [], duration)
            best = max(c["highlight"]["score"] for c in results)
            critique = r._critique_video(transcript.get("text", ""), signals, best)
            feedback = critique.model_dump() if critique else None

        summary = dict(selection.get("summary") or {})
        if render_failures:
            summary["render_failures"] = len(render_failures)
        summary["delivered"] = len(results)
        return {
            "status": "success",
            "transcript": transcript.get("text", ""),
            "clips": results,
            "feedback": feedback,
            "selection": summary,
        }
