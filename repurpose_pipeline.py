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
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import HTTPException

import audio_features
import captions
import clip_qa
import clip_render
import clip_review
import clip_selection
import creator_insights
import speakers
import video_analysis
import video_map
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
    # Whole-video analysis (pictures, voices, story map), editorial review and
    # rendering QA. Tests that only exercise job plumbing turn these off.
    analysis: bool = True
    review: bool = True
    qa: bool = True
    caption_style: str = captions.DEFAULT_STYLE
    # user_id -> creator_prior callable or None (see creator_insights)
    creator_prior_for: Optional[Callable] = None


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


class RunState:
    """Analysis shared by the stages of ONE run. Nothing here is persisted:
    after a crash it is recomputed, because the selection checkpoint already
    holds everything the expensive stages decided."""

    def __init__(self):
        self.punctuated: Optional[list] = None
        self.timeline: Optional[video_analysis.VisualTimeline] = None
        self.speaker_map: Optional[speakers.SpeakerMap] = None
        self.dims: Optional[dict] = None
        self.fps: Optional[float] = None

    def punct(self, words: list, segments: list) -> list:
        if self.punctuated is None:
            self.punctuated = clip_selection.attach_punctuation(words, segments)
        return self.punctuated

    def visual(self, path: str, duration: float) -> video_analysis.VisualTimeline:
        if self.timeline is None:
            self.timeline = video_analysis.analyze_video(path, duration)
        return self.timeline

    def dimensions(self, path: str) -> dict:
        if self.dims is None:
            self.dims = video_analysis.probe_video(path)
        return self.dims


def _analyse_media(ctx: JobContext, deps: PipelineDeps, run: RunState, path: str, words: list,
                   segments: list, duration: float, user_id: str) -> tuple:
    """Everything measured about the media. Each signal that can't be
    produced is simply absent (and recorded) — it never fails the job."""
    facts: dict = {}
    amb = clip_selection.AnalysisContext()
    sentences = clip_selection.build_sentences(run.punct(words, segments))
    ff = None
    try:
        ctx.stage("analyzing", 0.45, "Listening to the whole video")
        samples = decode_pcm(path, timeout=900)
        ff = frame_features(samples)
        baseline = build_baseline(ff, duration, [float(w["start"]) for w in words])

        def delivery(start: float, end: float, n_words: int):
            return delivery_score(window_stats(ff, start, end, n_words), baseline)

        amb.delivery = delivery
        amb.audio_quality = lambda s, e: audio_features.audio_quality_score(ff, s, e)
        facts["audio_analysed"] = True
    except Exception as e:
        samples = None
        facts["audio_analysed"] = False
        print(f"[WARN] audio analysis unavailable: {e}")

    ctx.stage("analyzing", 0.47, "Watching the whole video")
    timeline = run.visual(path, duration)
    if timeline.usable:
        amb.visual_quality = lambda s, e: video_analysis.visual_quality_score(video_analysis.window_visual(timeline, s, e))
        facts["visual"] = {"samples": int(len(timeline.times)), "shots": len(timeline.shots),
                           "camera_cuts": int(timeline.cut_before.sum()), "detector": timeline.detector,
                           "bars": {k: round(v, 3) for k, v in timeline.bars.items()}}
    else:
        facts["visual"] = {"available": False}

    if samples is not None and ff is not None:
        ctx.stage("analyzing", 0.49, "Telling the speakers apart")
        try:
            smap = speakers.estimate_speakers(samples, ff, sentences, timeline)
            run.speaker_map = smap
            facts["speakers"] = smap.summary()
            if smap.reliable and smap.n_speakers >= 2:
                amb.speaker_labels = smap.sentence_labels
        except Exception as e:
            facts["speakers"] = {"reliable": False, "note": f"unavailable: {str(e)[:80]}"}
            print(f"[WARN] speaker analysis unavailable: {e}")

    ctx.stage("analyzing", 0.51, "Understanding the story")
    vm = video_map.build_video_map(sentences, amb.speaker_labels, deps.llm,
                                   speakers_reliable=bool(amb.speaker_labels))
    amb.video_map = vm
    facts["video_map"] = ({"format": vm["format"], "topics": len(vm["topics"]), "avoided_stretches": len(vm["avoid"]),
                           "summary": vm["summary"]} if vm else {"available": False})

    if deps.creator_prior_for is not None:
        try:
            amb.creator_prior = deps.creator_prior_for(user_id)
        except Exception as e:
            print(f"[WARN] creator history unavailable: {e}")
    facts["creator_prior"] = "applied" if amb.creator_prior else "no_history"
    return amb, sentences, facts


def _review(ctx: JobContext, deps: PipelineDeps, run: RunState, path: str, result, sentences: list,
            amb, work_dir: str) -> dict:
    pool = result.clips + result.spares
    if not deps.review or not pool:
        return {"reviewed": False}
    ctx.stage("selecting", 0.57, "Reviewing the best moments like an editor")

    def frames_for(c):
        try:
            return clip_review.extract_frames(path, c, work_dir)
        except Exception:
            return []

    reviews = clip_review.review_candidates(pool, sentences, amb.speaker_labels, deps.llm,
                                            frames_for if run.timeline is not None and run.timeline.usable else None)
    before = len(result.clips)
    clip_review.apply_reviews(result, reviews, pool)
    verdicts: dict = {}
    for rv in reviews.values():
        verdicts[rv["verdict"]] = verdicts.get(rv["verdict"], 0) + 1
    return {"reviewed": True, "clips_reviewed": len(reviews), "unreviewed": len(pool) - len(reviews),
            "verdicts": verdicts, "changed_selection": before != len(result.clips) or bool(reviews)}


def _select(ctx: JobContext, deps: PipelineDeps, run: RunState, path: str, transcript: dict,
            duration: float, work_dir: str) -> dict:
    existing = (ctx.job.get("checkpoint") or {}).get("selection")
    if existing:
        return existing

    words, segments = transcript.get("words") or [], transcript.get("segments") or []
    sentences: list = []
    facts: dict = {}
    amb = clip_selection.AnalysisContext()
    if deps.analysis:
        amb, sentences, facts = _analyse_media(ctx, deps, run, path, words, segments, duration,
                                               ctx.job["user_id"])

    ctx.stage("selecting", 0.53, "Finding the strongest moments")
    try:
        result = clip_selection.run_selection(words, segments, duration, deps.llm, amb, deps.min_clip_score)
    except clip_selection.SelectionUnavailable as e:
        raise TransientJobError("selection_unavailable", "Clip selection is temporarily unavailable.", str(e))

    if result.clips and deps.analysis:
        facts["review"] = _review(ctx, deps, run, path, result, sentences, amb, work_dir)

    summary = result.summary()
    summary["analysis"] = {**(summary.get("analysis") or {}), **facts}
    summary["selected"] = len(result.clips)
    summary["spares_kept"] = len(result.spares)
    summary["candidates_rejected"] = len(result.rejected)
    summary["rejected_reasons"] = clip_selection._count_reasons(result.rejected)
    summary["note"] = result.note
    summary["delivery_signal"] = amb.delivery is not None
    if not result.clips:
        raise PermanentJobError("no_strong_moments", result.note or
                                "We didn't find a moment strong enough to stand on its own.",
                                json.dumps({"rejected": result.rejected[:10], "note": result.note,
                                            "analysis": summary.get("analysis")}, default=str)[:3500])

    selection = {"clips": [c.to_dict() for c in result.clips],
                 "spares": [c.to_dict() for c in result.spares], "summary": summary}
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


class ClipRejected(Exception):
    """A rendered clip failed quality control; the caller tries a spare."""

    def __init__(self, failures: list, report: dict):
        super().__init__(", ".join(failures))
        self.failures = failures
        self.report = report


def _clip_words(punct: list, start: float, end: float) -> list:
    return [w for w in punct if start <= (float(w["start"]) + float(w["end"])) / 2 < end]


def _render_one(ctx: JobContext, deps: PipelineDeps, run: RunState, source: str, work_dir: str, rank: int,
                clip: dict, transcript: dict) -> dict:
    import repurpose as r
    words, segments = transcript.get("words") or [], transcript.get("segments") or []
    punct = run.punct(words, segments)
    blocks = r._find_speech_blocks(segments, clip["start"], clip["end"])
    dead_air = r._dead_air_removed_seconds(blocks)
    kept_before = sum(e - s for s, e in blocks)
    blocks, filler_count = r._remove_filler_words(blocks, words)
    filler_seconds = round(kept_before - sum(e - s for s, e in blocks), 2)
    planned = clip_render_total = sum(e - s for s, e in blocks)

    overlay = clip.get("overlay_hook")
    smap = run.speaker_map
    speaker_at = smap.label_at if smap is not None and smap.reliable and smap.n_speakers >= 2 else None
    ass_text, cap_facts = captions.build_ass(punct, blocks, hook=overlay or "", style=deps.caption_style,
                                             speaker_at=speaker_at)
    ass_path = None
    if ass_text:
        ass_path = os.path.join(work_dir, f"captions_{rank}.ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_text)

    dims = run.dimensions(source)
    timeline = run.visual(source, ctx.job.get("checkpoint", {}).get("source", {}).get("duration") or 0.0) \
        if deps.analysis else None
    out = os.path.join(work_dir, f"clip_{rank}.mp4")
    plan = clip_render.plan_crop(timeline, blocks, dims["width"], dims["height"], source)
    qa_dict: dict = {}
    attempts = 0
    while True:
        attempts += 1
        try:
            render_facts = clip_render.render(source, out, blocks, plan, ass_path, run.fps)
        except clip_render.RenderError as e:
            raise RuntimeError(str(e))
        if not deps.qa:
            break
        c_words = _clip_words(punct, blocks[0][0], blocks[-1][1])
        report = clip_qa.run_qa(out, planned, render_facts, ass_text, overlay, clip.get("transcript", ""),
                                [str(w.get("text") or w.get("word")) for w in c_words], work_dir)
        qa_dict = report.to_dict()
        if report.passed:
            break
        # Caption timing: whisper's word times can sit a few hundred ms off the
        # audio. QC measured the offset, so shift the captions by it and re-render once.
        cs_fail = [f for f in report.failures if f.name == "caption_sync"]
        if (attempts == 1 and ass_text and len(report.failures) == 1 and cs_fail
                and 0.12 < abs(cs_fail[0].detail.get("offset_s", 0)) <= 0.7):
            shift = float(cs_fail[0].detail["offset_s"])
            ass_text, cap_facts = captions.build_ass(punct, blocks, hook=overlay or "", style=deps.caption_style,
                                                     speaker_at=speaker_at, shift=shift)
            cap_facts["timing_shift_s"] = shift
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_text)
            continue
        # One regeneration: a framing failure is fixed by keeping the whole
        # frame instead of cropping it.
        if attempts <= 2 and report.failures and all(f.name == "framing" for f in report.failures):
            plan = clip_render.CropPlan(mode="fit", src_w=dims["width"], src_h=dims["height"], bars=plan.bars,
                                        crop_w=plan.crop_w, crop_h=plan.crop_h,
                                        pieces=[(0.0, 0.0, planned, 0.0)], facts={**plan.facts, "retry": "fit_after_framing"})
            continue
        if os.path.exists(out):
            os.remove(out)
        raise ClipRejected([f.name for f in report.failures], qa_dict)

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
                   "filler_words_removed_count": filler_count,
                   "render": render_facts, "captions": cap_facts, "qa": qa_dict,
                   "overlay_hook": overlay, "overlay_status": clip.get("overlay_status"),
                   "review": clip.get("review") or {}}
    ctx.save_clip({
        "rank": rank, "status": "rendered", "start_time": clip["start"], "end_time": clip["end"],
        "score": clip["score"], "score_breakdown": clip.get("breakdown"), "category": clip.get("category"),
        "title": clip["title"], "reason": clip["reason"], "hook_line": clip.get("hook_line"),
        "caption": clip.get("caption"), "hashtags": clip.get("hashtags") or [],
        "transcript": clip.get("transcript"), "video_path": f"{base}.mp4", "video_url": video_url,
        "thumbnail_path": thumbnail_path, "thumbnail_url": thumbnail_url,
        "quote_card_path": quote_path, "quote_card_url": quote_url, "render_meta": render_meta,
    })
    if os.path.exists(out):
        os.remove(out)
    return {"video_url": video_url, "thumbnail_url": thumbnail_url, "quote_card_url": quote_url,
            **render_meta}


def _clip_from_row(row: dict) -> dict:
    """Rebuild a clip dict from a saved row, for clips rendered before a restart."""
    return {"start": row.get("start_time"), "end": row.get("end_time"), "score": row.get("score"),
            "category": row.get("category"), "title": row.get("title") or "", "reason": row.get("reason") or "",
            "hook_line": row.get("hook_line") or "", "caption": row.get("caption") or "",
            "hashtags": row.get("hashtags") or [], "transcript": row.get("transcript") or "",
            "breakdown": row.get("score_breakdown") or {}}


def video_analysis_fps(path: str) -> Optional[float]:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=avg_frame_rate", "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    num, _, den = out.partition("/")
    return float(num) / float(den) if den and float(den) else (float(num) if num else None)


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
    run = RunState()
    with tempfile.TemporaryDirectory(prefix=f"repurpose_{ctx.job_id[:8]}_") as work:
        source = os.path.join(work, "source")
        ctx.stage("downloading", 0.0, "Downloading your video")
        _fetch_source(job, deps, source)

        ctx.stage("probing", 0.08, "Checking your video")
        duration, has_audio = _probe(source, deps)
        sha = _sha256(source)
        ctx.checkpoint({"source": {"sha256": sha, "duration": duration, "has_audio": has_audio}})
        try:
            run.fps = video_analysis_fps(source)
        except Exception:
            run.fps = None

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
            selection = _select(ctx, deps, run, source, transcript, duration, work)
        elif job.get("creative_prompt"):
            selection = _prompt_selection(ctx, job, duration)
        else:
            raise PermanentJobError(
                "no_speech",
                "We couldn't hear any speech in this video, so there are no moments to clip. "
                "Try a video with talking, or describe the clip you want instead.")

        existing = {c["rank"]: c for c in (ctx.store.get_job(ctx.job_id) or {}).get("clips", [])}
        queue = list(selection["clips"]) + list(selection.get("spares") or [])
        target = len(selection["clips"])
        results: list[dict] = []
        render_failures: list[str] = []
        qa_rejected: list[dict] = []
        used_spans: list[tuple] = []
        for rank in range(1, target + 1):
            ctx.stage("rendering", 0.60 + 0.37 * (rank - 1) / target, f"Editing clip {rank} of {target}")
            done = existing.get(rank)
            if done and done.get("status") == "rendered" and done.get("video_url"):
                rendered = {"video_url": done["video_url"], "thumbnail_url": done.get("thumbnail_url"),
                            "quote_card_url": done.get("quote_card_url"), **(done.get("render_meta") or {})}
                used_spans.append((done.get("start_time"), done.get("end_time")))
                results.append(_legacy_clip(_clip_from_row(done), rendered))
                continue
            while queue:
                clip = queue.pop(0)
                if any(a_ is not None and abs(clip["start"] - a_) < 0.5 and abs(clip["end"] - b_) < 0.5
                       for a_, b_ in used_spans):
                    continue    # already rendered before a restart
                try:
                    rendered = _render_one(ctx, deps, run, source, work, rank, clip, transcript)
                except (LeaseLost, ShutdownRequested):
                    raise
                except ClipRejected as e:
                    qa_rejected.append({"title": clip.get("title"), "failed": e.failures,
                                        "measured": {k: v for k, v in e.report.get("checks", {}).items()
                                                     if k in e.failures}})
                    print(f"[INFO] clip '{clip.get('title')}' failed quality control: {e.failures}")
                    continue
                except StorageUnavailable as e:
                    raise TransientJobError("storage_unavailable", "Couldn't save your clips.", str(e))
                except Exception as e:
                    render_failures.append(f"clip {rank}: {e}")
                    print(f"[WARN] render failed for clip {rank} of job {ctx.job_id}: {e}")
                    continue
                used_spans.append((clip["start"], clip["end"]))
                results.append(_legacy_clip(clip, rendered))
                break

        if not results and qa_rejected and not render_failures:
            raise PermanentJobError(
                "no_clip_passed_quality_check",
                "We found moments in your video, but none came out well enough to publish, so we didn't "
                "keep any. Try a video with clearer audio or picture.", json.dumps(qa_rejected, default=str)[:3500])
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
        if qa_rejected:
            summary["quality_rejected"] = qa_rejected
            if len(results) < target:
                summary["note"] = (summary.get("note") or
                                   "Some moments were dropped because they didn't pass our quality checks.")
        summary["delivered"] = len(results)
        return {
            "status": "success",
            "transcript": transcript.get("text", ""),
            "clips": results,
            "feedback": feedback,
            "selection": summary,
        }
