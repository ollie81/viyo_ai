"""
Viyo AI Video Repurposer — Railway-compatible rebuild.

What changed from the original version, and why:

1. NO local Whisper model. The original loaded a multi-GB AI model into
   RAM on server startup — this alone would exceed Railway's free/hobby
   tier memory and likely crash the container before it ever serves a
   request. This version calls OpenAI's Whisper API instead (a normal
   HTTP request), so the server itself stays lightweight.

2. Accepts a Supabase Storage URL instead of a raw file upload.
   Previously Flutter uploaded the video file directly to Railway —
   this hit Railway's proxy timeout on anything longer than ~30s
   (upload + Whisper + FFmpeg easily exceeds that), causing the
   "Broken pipe" error. Now Flutter uploads to Supabase Storage first
   (no Railway timeout involved), then sends just the URL here.
   The backend downloads it over a server-to-server connection that
   isn't subject to Railway's inbound proxy timeout.

3. Output is uploaded to Supabase Storage, not saved to local disk.
   Railway's filesystem is ephemeral — anything written to disk is
   wiped on every redeploy or restart. This version uploads the final
   clip straight to a "processed-videos" Supabase bucket and returns
   a permanent public URL instead.

4. Long-video limits are configurable. The default is 33 minutes / 500MB,
   while Whisper transcription is performed in small audio chunks so the
   OpenAI per-file audio limit is not exceeded.

5. Uses the same JWT auth + rate limiting pattern as the rest of
   main.py, instead of being wide open.

6. Auth is handled by delegating to main.py's get_current_user_id
   (ES256 / JWKS verification) via a lazy import, avoiding duplication
   and the circular-import problem.

7. Registered as a FastAPI router, imported into main.py — kept in its
   own file so it can't accidentally break the endpoints that are
   already working.
"""

import os
import re
import json
import time
import base64
import string
import tempfile
import subprocess
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict, deque
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from openai import OpenAI
from supabase import create_client, Client

from coins import spend_on_feature

# Speaker-tracking crop is optional: this app runs fine without it (the
# reframe just falls back to a fixed center crop, same as before this
# existed), so a deploy that hasn't picked up the new opencv-python-headless
# dependency yet degrades instead of breaking every repurpose job.
try:
    import cv2
    _FACE_TRACKING_AVAILABLE = True
except ImportError:
    cv2 = None
    _FACE_TRACKING_AVAILABLE = False

router = APIRouter(prefix="/api/v1", tags=["repurpose"])

ai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Service role key — NOT the anon key. Required to upload to Storage on
# the user's behalf without needing their own Supabase session here.
# Get it from Supabase dashboard: Settings -> API Keys -> service_role.
# Keep this secret; it bypasses Row Level Security entirely.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
PROCESSED_BUCKET = "processed-videos"

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Hard limits — tune down further if Railway's plan is small.
# Long-video limits. Override with environment variables if needed.

MAX_INPUT_DURATION_SECONDS = int(os.environ.get("MAX_INPUT_DURATION_SECONDS", "1980"))  # 33 minutes

MAX_INPUT_SIZE_BYTES = int(os.environ.get("MAX_INPUT_SIZE_MB", "500")) * 1024 * 1024

MAX_OUTPUT_CLIP_SECONDS = 60
# Clip length is chosen per moment rather than padded to a fixed
# duration: a tight 18-second point lands harder than the same point
# stretched to 60. Below ~12 seconds there's rarely a complete thought,
# so that's the floor rather than a target.
MIN_OUTPUT_CLIP_SECONDS = 12

DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB

QUOTE_CARD_SIZE = "1080x1080"
# Requires fonts-dejavu-core installed in the Dockerfile — ffmpeg/libass
# only depend on the fontconfig *library*, not any actual font files, so
# without this package drawtext has nothing to render text with at all.
QUOTE_CARD_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# How many evenly-spaced frames to pull from a rendered clip as thumbnail
# candidates. 5 is enough spread to catch a genuinely different moment
# (talking head vs. a reaction vs. a b-roll cutaway) without ballooning
# the vision call's cost/latency — each candidate is one more image token
# block in the same GPT-4o-mini request.
THUMBNAIL_CANDIDATE_COUNT = 5

# Separate, stricter rate limit from the other AI endpoints — this is
# far more expensive per call (Whisper + GPT + FFmpeg render).
REPURPOSE_RATE_LIMIT = 5
REPURPOSE_RATE_WINDOW = 60 * 60 * 24  # per day
_repurpose_requests: dict = defaultdict(deque)


def _check_repurpose_rate_limit(user_id: str):
    now = time.time()
    q = _repurpose_requests[user_id]
    while q and now - q[0] > REPURPOSE_RATE_WINDOW:
        q.popleft()
    if len(q) >= REPURPOSE_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Limit reached: {REPURPOSE_RATE_LIMIT} AI repurposes per day. Try again tomorrow.",
        )
    q.append(now)


async def _get_current_user_id(authorization: str = Header(None)) -> str:
    """Delegates to main.py's get_current_user_id_no_guest (ES256 / JWKS
    verification, plus rejecting an anonymous guest session — this
    endpoint is entirely AI-driven (Whisper + GPT), same reasoning as
    every other OpenAI-calling endpoint in this backend).

    We use a lazy import instead of a module-level one to avoid a circular
    import — main.py imports repurpose at startup, so repurpose must not
    import main at module load time. Importing inside the function body is
    fine: it only runs at request time, by which point main is fully loaded.
    """
    from main import get_current_user_id_no_guest
    return await get_current_user_id_no_guest(authorization)


class RepurposeRequest(BaseModel):
    # Flutter uploads the video to Supabase Storage first, then sends
    # us the public URL. This avoids streaming a large file through
    # Railway's inbound proxy, which has a hard timeout that caused
    # the "Broken pipe" error on the previous multipart approach.
    video_url: str = Field(..., description="Public Supabase Storage URL of the source video")


class HighlightSegment(BaseModel):
    start_time: float
    end_time: float
    reason: str
    suggested_title: str
    # Deliberately no ge/le bounds here: an LLM-returned value outside
    # [0, 100] must not raise at construction time (that would discard
    # the whole batch of otherwise-valid candidates before the clamping
    # below ever runs) — same reasoning as start_time/end_time not being
    # bounded at the field level.
    score: int = 70
    # Ready-to-post copy. A rendered clip still leaves the creator
    # writing the caption themselves, which is the step that actually
    # stalls posting — these make it paste-and-publish. Both default
    # empty so an older/partial model response can't fail the batch.
    caption: str = ""
    hashtags: list[str] = []
    # The exact words the clip opens on. Used to burn the hook across
    # the opening seconds (see _generate_ass) and to show the creator
    # what the AI thinks is doing the work in the first 2 seconds.
    hook_line: str = ""


class VideoFeedback(BaseModel):
    """
    An honest read on the SOURCE video, not the clips cut from it.

    Every other tool in this space quietly returns weak clips when the
    footage is weak, leaving the creator to guess why nothing lands.
    Grounded in measured signals (see _measure_video_signals) as well as
    the transcript, so the advice cites what actually happened rather
    than generic content-coaching filler.
    """
    verdict: str = ""              # one-line overall read
    issues: list[str] = []         # concrete, fixable problems
    strengths: list[str] = []      # what genuinely worked, if anything
    score: int = 0                 # 0-100, how well this footage cuts down
    # Raw measurements the verdict was based on, so the UI can show the
    # numbers rather than asking the creator to trust a vibe.
    words_per_minute: float = 0.0
    silence_percent: float = 0.0


class RepurposeClipResult(BaseModel):
    processed_video_url: str
    highlight: HighlightSegment
    dead_air_removed_seconds: float = 0.0
    # Separate from dead_air_removed_seconds on purpose — they're
    # different edits with different causes, and folding them into one
    # number would make "dead air removed" a label that's no longer
    # accurate.
    filler_words_removed_seconds: float = 0.0
    filler_words_removed_count: int = 0
    # None if the quote-card render/upload failed — never fails the
    # whole request over the extra format, since the video clip itself
    # is the part that actually matters.
    quote_card_url: Optional[str] = None
    # None if thumbnail extraction/selection/upload failed — same
    # never-fail-the-request reasoning as quote_card_url.
    thumbnail_url: Optional[str] = None


class RepurposeResponse(BaseModel):
    status: str
    transcript: str
    # Ranked best first. Previously this endpoint rendered a single
    # "best" clip and made that call for the creator — now it returns
    # several genuinely different candidates (deduped by overlap) so
    # they can pick, which is what people actually want from an editor
    # rather than one AI verdict with no alternative.
    clips: list[RepurposeClipResult]
    # None only if the feedback pass itself failed — a missing critique
    # never fails a job that produced usable clips.
    feedback: Optional[VideoFeedback] = None


class RepurposeJobStartResponse(BaseModel):
    job_id: str
    status: str  # always "processing" — a job was just created


class RepurposeJobStatusResponse(BaseModel):
    status: str  # "processing" | "done" | "failed"
    result: Optional[RepurposeResponse] = None
    error: Optional[str] = None


class _RepurposeJob:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.status = "processing"
        self.result: Optional[RepurposeResponse] = None
        self.error: Optional[str] = None
        self.created_at = time.time()


# In-memory job store — same tradeoff already accepted by every rate
# limiter in this file (resets on restart, doesn't share state across
# instances). A real queue (Celery/Redis) would survive a restart, but
# that's new infrastructure this app doesn't have; jobs only need to
# live long enough for one client to poll them to completion, which an
# hour comfortably covers.
_REPURPOSE_JOB_TTL_SECONDS = 60 * 60
_repurpose_jobs: dict[str, _RepurposeJob] = {}


def _cleanup_expired_repurpose_jobs() -> None:
    cutoff = time.time() - _REPURPOSE_JOB_TTL_SECONDS
    expired = [jid for jid, job in _repurpose_jobs.items() if job.created_at < cutoff]
    for jid in expired:
        del _repurpose_jobs[jid]


def _validate_storage_url(url: str) -> None:
    """
    Rejects anything that isn't an HTTPS URL on this project's own
    Supabase host.

    Without this, video_url was passed straight to urlretrieve with no
    checks at all — any authenticated caller could point it at an
    internal address (cloud metadata endpoints, other Railway/internal
    services, localhost) and use this endpoint as a server-side request
    forgery proxy. Comparing against SUPABASE_URL's own host, which is
    already known server-side, closes the actual threat here.
    """
    parsed = urllib.parse.urlparse(url)
    expected_host = urllib.parse.urlparse(SUPABASE_URL).hostname
    if parsed.scheme != "https" or not expected_host or parsed.hostname != expected_host:
        raise HTTPException(
            status_code=400,
            detail="video_url must be an HTTPS Supabase Storage URL for this project.",
        )


def _download_video(url: str, dest_path: str) -> None:
    """
    Downloads the source video with a hard size cap enforced while
    streaming, not after the fact — the previous version downloaded the
    entire file with urlretrieve before ever checking MAX_INPUT_SIZE_BYTES,
    so a large or slow-drip remote file would be fully pulled down
    (bandwidth/disk/cost) before being rejected. A timeout guards against
    a remote server that simply hangs.
    """
    _validate_storage_url(url)
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
            # Re-validate after following any redirects — the initial URL
            # passing the check doesn't guarantee the final one does.
            _validate_storage_url(resp.geturl())

            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_INPUT_SIZE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Video too large — max {MAX_INPUT_SIZE_BYTES // (1024 * 1024)}MB.",
                )

            written = 0
            with open(dest_path, "wb") as out:
                while True:
                    chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_INPUT_SIZE_BYTES:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Video too large — max {MAX_INPUT_SIZE_BYTES // (1024 * 1024)}MB.",
                        )
                    out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download video from URL: {e}")


def _run_ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="Could not read video — file may be corrupt.")


def _transcribe_with_openai(path: str) -> dict:
    """
    Transcribe long videos safely.

    OpenAI's audio upload endpoint has a per-file size limit, so a long
    video's audio is split into small MP3 chunks first. Each chunk is
    transcribed separately and the timestamps are shifted back into the
    original video's timeline.
    """
    with tempfile.TemporaryDirectory() as audio_tmp:
        audio_pattern = os.path.join(audio_tmp, "audio_%03d.mp3")

        # 64 kbps mono MP3 keeps each chunk comfortably below the API limit.
        # 600 seconds (10 minutes) is intentionally conservative.
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "libmp3lame",
            "-b:a", "64k",
            "-f", "segment",
            "-segment_time", "600",
            "-reset_timestamps", "1",
            audio_pattern,
        ]
        result = subprocess.run(
            extract_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Audio extraction failed: {result.stderr[-500:]}"
            )

        chunks = sorted(
            os.path.join(audio_tmp, n)
            for n in os.listdir(audio_tmp)
            if n.endswith(".mp3")
        )

        all_segments = []
        all_words = []
        all_text = []
        offset = 0.0

        for chunk in chunks:
            chunk_duration = _run_ffprobe_duration(chunk)

            with open(chunk, "rb") as f:
                try:
                    result = ai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="verbose_json",
                        # Word timings drive the karaoke-style captions
                        # (see _generate_ass). Segment timings alone can
                        # only light up a whole line at a time, which is
                        # what makes auto-captions look auto-generated.
                        timestamp_granularities=["segment", "word"],
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Transcription failed: {e}"
                    )

            data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            text = (data.get("text") or "").strip()
            if text:
                all_text.append(text)

            for seg in data.get("segments", []) or []:
                seg = dict(seg)
                seg["start"] = float(seg.get("start", 0)) + offset
                seg["end"] = float(seg.get("end", 0)) + offset
                all_segments.append(seg)

            for word in data.get("words", []) or []:
                word = dict(word)
                word["start"] = float(word.get("start", 0)) + offset
                word["end"] = float(word.get("end", 0)) + offset
                all_words.append(word)

            offset += chunk_duration

        return {
            "text": " ".join(all_text).strip(),
            "segments": all_segments,
            "words": all_words,
        }

MAX_HIGHLIGHT_CANDIDATES = 5

# Roughly one additional highlight candidate per 6 minutes of source video —
# a 3-minute video and a 30-minute video obviously don't contain the same
# number of genuinely distinct highlight-worthy moments. Capped at
# MAX_HIGHLIGHT_CANDIDATES so a near-max-length (33-minute) upload still
# renders in a bounded amount of time (each extra candidate means another
# full render + thumbnail-selection + quote-card pass below).
_MINUTES_PER_ADDITIONAL_CLIP = 6


def _highlight_count_for_duration(duration_seconds: float) -> int:
    minutes = duration_seconds / 60
    count = 1 + int(minutes // _MINUTES_PER_ADDITIONAL_CLIP)
    return max(1, min(MAX_HIGHLIGHT_CANDIDATES, count))


# Caption/hashtag limits. Kept well under every short-form platform's
# own cap so a generated caption is never the thing that gets rejected
# at post time.
MAX_CAPTION_CHARS = 300
MAX_HASHTAGS = 6
# Burned across the opening seconds, so it has to fit a phone screen
# without covering the video.
MAX_HOOK_LINE_CHARS = 70

# Words that carry no hook on their own. A clip opening on these burns
# the two seconds that decide whether anyone keeps watching, so the
# start is nudged past them — the prompt asks the model to avoid this,
# this is what actually guarantees it.
_FILLER_OPENERS = {
    "so", "um", "uh", "erm", "ah", "oh", "eh", "hmm", "mm", "mhm",
    "and", "but", "or", "then", "well", "like", "okay", "ok", "right",
    "yeah", "yep", "yes", "no", "now", "just", "basically", "actually",
    "anyway", "anyways", "literally", "obviously", "honestly", "look",
    "listen", "i", "you", "it", "we", "they", "that", "this", "the",
    "a", "an", "is", "was", "know", "mean", "think", "guess",
}
# Never chase a hook further than this into the clip — past it, the
# model's chosen moment has effectively been abandoned.
_MAX_HOOK_SKIP_SECONDS = 2.5


def _tighten_hook_start(words: list, start_time: float, end_time: float) -> float:
    """
    Moves start_time forward past any filler the clip would otherwise
    open on, landing on the first word that can actually carry a hook.

    Returns start_time unchanged when there are no word timings, when
    the opening is already strong, or when skipping would eat more than
    _MAX_HOOK_SKIP_SECONDS or leave too short a clip.
    """
    if not words:
        return start_time

    in_clip = sorted(
        (w for w in words if float(w.get("start", 0)) >= start_time - 0.15
         and float(w.get("start", 0)) < end_time),
        key=lambda w: float(w.get("start", 0)),
    )
    if not in_clip:
        return start_time

    for word in in_clip:
        token = re.sub(r"[^a-z']", "", str(word.get("word", "")).lower())
        if token and token not in _FILLER_OPENERS:
            candidate = float(word.get("start", 0))
            # Small lead-in so the first syllable isn't clipped.
            candidate = max(start_time, candidate - 0.1)
            if candidate - start_time > _MAX_HOOK_SKIP_SECONDS:
                return start_time
            if end_time - candidate < MIN_OUTPUT_CLIP_SECONDS:
                return start_time
            return candidate

    return start_time


def _clean_hashtags(tags: list) -> list:
    """
    Normalises whatever the model returned into plain, postable tags:
    no '#', no spaces, lowercase, deduped, capped. The model is asked
    for this format already — this is the guarantee, not the request.
    """
    cleaned = []
    for tag in tags or []:
        tag = re.sub(r"[^a-z0-9]", "", str(tag).lower())
        if tag and tag not in cleaned:
            cleaned.append(tag)
        if len(cleaned) >= MAX_HASHTAGS:
            break
    return cleaned


def _overlap_fraction(a: HighlightSegment, b: HighlightSegment) -> float:
    overlap = max(0.0, min(a.end_time, b.end_time) - max(a.start_time, b.start_time))
    shorter = min(a.end_time - a.start_time, b.end_time - b.start_time)
    return overlap / shorter if shorter > 0 else 0.0


def _fallback_highlight(max_duration: float) -> HighlightSegment:
    return HighlightSegment(
        start_time=0.0,
        end_time=min(30.0, max_duration),
        reason="Default clip (AI highlight detection unavailable)",
        suggested_title="Featured Clip",
        score=50,
    )


def _format_timestamped_transcript(segments: list) -> str:
    """
    Renders Whisper's own per-segment timestamps as `[start -> end] text`
    lines. _find_highlights previously received only the flat concatenated
    transcript text with no timing information at all, yet was asked to
    return start_time/end_time in seconds — meaning every clip boundary
    was effectively a guess based on word count and assumed speaking
    pace, not grounded in the real audio. This gives the model actual
    numbers to point to instead.
    """
    lines = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{seg.get('start', 0):.1f}s -> {seg.get('end', 0):.1f}s] {text}")
    return "\n".join(lines)


def _find_highlights(
    segments: list,
    max_duration: float,
    count: int = MAX_HIGHLIGHT_CANDIDATES,
    words: Optional[list] = None,
) -> list:
    """
    Returns up to `count` ranked, non-overlapping highlight candidates
    instead of a single "best" pick — the point of multiple options is
    that they're genuinely different moments, so a candidate that mostly
    overlaps an already-accepted one is skipped rather than counted.

    Two things this asks for beyond "find something engaging": clip
    boundaries grounded in Whisper's real timestamps (see
    _format_timestamped_transcript) rather than guessed, and an opening
    line that actually hooks a viewer — the single biggest lever on
    whether a short-form clip gets watched past the first second, which
    the previous prompt never asked for at all.
    """
    if not segments:
        return [_fallback_highlight(max_duration)]

    timestamped_transcript = _format_timestamped_transcript(segments)

    prompt = (
        f"Below is a timestamped transcript of a video, {max_duration:.0f} seconds "
        "long. Each line is tagged with its EXACT start and end time in the source "
        "video — use these real timestamps for start_time/end_time. Never estimate "
        "or invent a time that isn't grounded in the lines below.\n\n"
        f"Identify the {count} best short-form clip candidates, ranked best first. "
        "They must be genuinely different moments, not overlapping variations of "
        "the same one.\n\n"
        "CLIP LENGTH: let each moment decide its own length, between "
        f"{MIN_OUTPUT_CLIP_SECONDS} and {MAX_OUTPUT_CLIP_SECONDS} seconds. Do NOT "
        "stretch a clip toward the maximum. A punchy 15-second exchange is a better "
        "clip than the same exchange padded to 45 seconds with lead-in and wind-down. "
        "Cut the moment where the idea genuinely completes.\n\n"
        "For each candidate:\n"
        "- THE FIRST 2 SECONDS DECIDE EVERYTHING. start_time must land exactly on the "
        "first word of a hook — a bold claim, a question, a number, a contradiction, "
        "or a surprising statement. Never start on filler ('so', 'um', 'yeah', 'and "
        "then', 'basically'), on a greeting, or mid-sentence. If the strongest hook "
        "sits a few seconds into a thought, start there and let the setup go.\n"
        "- hook_line: quote the exact opening words (roughly the first 2 seconds of "
        "speech) that do the hooking, copied verbatim from the transcript.\n"
        "- end_time should land at a natural end of thought (a punchline, a payoff, "
        "a conclusion) so the clip feels complete rather than cut off.\n"
        "- score (0-100) should weigh the strength of the opening hook specifically, "
        "not just whether the topic is generally interesting.\n"
        "- caption is the ready-to-post caption for this clip: 1-2 short lines in the "
        "creator's own voice, opening with a hook line, no hashtags inside it, and no "
        "quotation marks around it.\n"
        "- hashtags is 3-6 lowercase tags relevant to THIS clip's actual subject, "
        "without the # symbol. Prefer specific tags a real audience searches over "
        "generic filler like fyp or viral.\n\n"
        "Return ONLY a raw JSON array, each item an object with keys: start_time "
        "(seconds), end_time (seconds), reason (explain both why the moment is "
        "compelling and why the opening line hooks a viewer), suggested_title, "
        "score, caption, hashtags (array of strings), hook_line.\n\n"
        f"TRANSCRIPT:\n{timestamped_transcript}"
    )
    try:
        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert short-form content editor who specializes in retention-optimized hooks.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)
        if not isinstance(data, list):
            data = [data]
        candidates = [HighlightSegment(**item) for item in data]
    except Exception:
        return [_fallback_highlight(max_duration)]

    # A very short source can't yield a minimum-length clip; never
    # stretch past what actually exists.
    min_length = min(MIN_OUTPUT_CLIP_SECONDS, max_duration)

    accepted: list[HighlightSegment] = []
    for seg in candidates:
        # Clamp to the source video's real bounds and our length range
        # — never trust the model's numbers blindly.
        seg.start_time = max(0.0, min(seg.start_time, max_duration))
        seg.end_time = max(seg.start_time + 1, min(seg.end_time, max_duration))
        if seg.end_time - seg.start_time > MAX_OUTPUT_CLIP_SECONDS:
            seg.end_time = seg.start_time + MAX_OUTPUT_CLIP_SECONDS
        # Extend a too-short clip toward the end of the source, then
        # backwards if there's no room left — a 4-second fragment is
        # not a clip regardless of how the model timed it.
        if seg.end_time - seg.start_time < min_length:
            seg.end_time = min(max_duration, seg.start_time + min_length)
            seg.start_time = max(0.0, seg.end_time - min_length)

        # Guarantee the opening two seconds actually hook, rather than
        # trusting the prompt to have done it.
        seg.start_time = _tighten_hook_start(words, seg.start_time, seg.end_time)

        seg.score = max(0, min(seg.score, 100))
        seg.caption = seg.caption.strip()[:MAX_CAPTION_CHARS]
        seg.hashtags = _clean_hashtags(seg.hashtags)
        seg.hook_line = seg.hook_line.strip()[:MAX_HOOK_LINE_CHARS]

        if any(_overlap_fraction(seg, a) > 0.5 for a in accepted):
            continue
        accepted.append(seg)
        if len(accepted) >= count:
            break

    return accepted or [_fallback_highlight(max_duration)]


def _measure_video_signals(words: list, segments: list, duration: float) -> dict:
    """
    Objective numbers about the source footage, measured rather than
    guessed, so the critique below can cite real evidence instead of
    generic advice.
    """
    speech_seconds = 0.0
    for seg in segments or []:
        speech_seconds += max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
    speech_seconds = min(speech_seconds, duration)

    word_count = len(words or [])
    if not word_count:
        word_count = sum(len(str(s.get("text", "")).split()) for s in segments or [])

    wpm = (word_count / speech_seconds * 60) if speech_seconds > 0 else 0.0
    silence_percent = ((duration - speech_seconds) / duration * 100) if duration > 0 else 0.0

    return {
        "words_per_minute": round(wpm, 1),
        "silence_percent": round(max(0.0, min(silence_percent, 100.0)), 1),
        "word_count": word_count,
        "duration": round(duration, 1),
    }


def _critique_video(
    transcript: str, signals: dict, highlights: list
) -> Optional[VideoFeedback]:
    """
    Tells the creator what's actually wrong with their footage.

    Deliberately grounded in `signals` (measured) and the scores the
    highlight pass already assigned, so "your pacing drags" is backed
    by a real words-per-minute figure rather than being an opinion the
    creator has no way to check. Returns None on any failure — a
    missing critique must never fail a job that produced good clips.
    """
    if not transcript.strip():
        return None

    best_score = max((h.score for h in highlights), default=0)
    # Long transcripts get trimmed: the critique is about delivery and
    # structure, both of which are legible from a generous sample.
    sample = transcript[:6000]

    prompt = (
        "You are reviewing a creator's raw video so they can make the next one "
        "better. Be specific and honest, never flattering, and never generic. "
        "Every point must be something they could actually change next time.\n\n"
        "MEASURED FROM THE FOOTAGE:\n"
        f"- Length: {signals['duration']}s\n"
        f"- Speaking pace: {signals['words_per_minute']} words/minute "
        "(short-form delivery usually lands between 150 and 190; under 120 drags, "
        "over 220 is hard to follow)\n"
        f"- Silence/dead air: {signals['silence_percent']}% of the video\n"
        f"- Best hook score found anywhere in the video: {best_score}/100\n\n"
        "Judge these specific things: how strong the opening is, whether the "
        "delivery has energy, whether points land or ramble, and whether there "
        "are quotable moments worth clipping.\n\n"
        "Return ONLY raw JSON with keys: verdict (one honest sentence on whether "
        "this footage cuts down well), issues (array of 2-4 specific problems, each "
        "naming what to do differently — say nothing generic like 'add more energy'), "
        "strengths (array of 1-3 things that genuinely worked; empty array if "
        "nothing did), score (0-100 for how well this footage works as short-form "
        "source material).\n\n"
        f"TRANSCRIPT:\n{sample}"
    )

    try:
        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a blunt, experienced short-form video coach. You "
                               "tell creators the truth about their footage.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(re.sub(r"```json|```", "", raw).strip())

        return VideoFeedback(
            verdict=str(data.get("verdict", "")).strip()[:300],
            issues=[str(i).strip()[:200] for i in (data.get("issues") or [])][:4],
            strengths=[str(s).strip()[:200] for s in (data.get("strengths") or [])][:3],
            score=max(0, min(int(data.get("score", 0) or 0), 100)),
            words_per_minute=signals["words_per_minute"],
            silence_percent=signals["silence_percent"],
        )
    except Exception as e:
        print(f"[WARN] Video critique failed: {e}")
        return None


# A gap between Whisper segments longer than this, inside the chosen
# highlight window, is dead air worth cutting rather than a natural
# conversational pause. Padding keeps a small buffer around each kept
# block so a cut doesn't clip the start/end of a word.
DEAD_AIR_GAP_THRESHOLD_SECONDS = 0.7
DEAD_AIR_PADDING_SECONDS = 0.15


def _find_speech_blocks(segments: list, clip_start: float, clip_end: float) -> list:
    """
    Groups the Whisper segments inside [clip_start, clip_end] into
    contiguous speech blocks, treating any gap between segments longer
    than DEAD_AIR_GAP_THRESHOLD_SECONDS as dead air to cut out of the
    render. Returns a list of (start, end) tuples in the ORIGINAL
    video's timeline — a single-item list means nothing was worth
    trimming (no transcript in range, or no gap crossed the threshold).
    """
    relevant = sorted(
        (s for s in segments if s.get("end", 0) > clip_start and s.get("start", 0) < clip_end),
        key=lambda s: s.get("start", 0),
    )
    if not relevant:
        return [(clip_start, clip_end)]

    def clamp(t):
        return max(clip_start, min(t, clip_end))

    blocks = []
    block_start = clamp(relevant[0]["start"] - DEAD_AIR_PADDING_SECONDS)
    block_end = clamp(relevant[0]["end"] + DEAD_AIR_PADDING_SECONDS)

    for seg in relevant[1:]:
        seg_start = clamp(seg["start"] - DEAD_AIR_PADDING_SECONDS)
        seg_end = clamp(seg["end"] + DEAD_AIR_PADDING_SECONDS)
        if seg_start - block_end > DEAD_AIR_GAP_THRESHOLD_SECONDS:
            blocks.append((block_start, block_end))
            block_start, block_end = seg_start, seg_end
        else:
            block_end = max(block_end, seg_end)

    blocks.append((block_start, block_end))
    return blocks


def _dead_air_removed_seconds(blocks: list) -> float:
    if len(blocks) < 2:
        return 0.0
    kept = sum(b_end - b_start for b_start, b_end in blocks)
    return round((blocks[-1][1] - blocks[0][0]) - kept, 2)


# ---------------------------------------------------------
# Filler-word removal
#
# Dead-air removal above cuts silence between segments, but a filler
# word ("um", "uh") has actual sound — it sits inside otherwise-
# continuous speech, so no silence gap ever catches it. This is a
# second, finer pass over the same blocks that cuts the filler words
# themselves out of the audio and video.
#
# Deliberately narrow: only pure interjections that never carry
# meaning on their own. Words like "like" or "so" are just as often
# real content as they are filler ("I like pizza" vs. "it's, like,
# really good"), and getting that wrong silently deletes something the
# creator actually said. This only touches words that are filler in
# every context they appear in.
# ---------------------------------------------------------

_PURE_FILLER_WORDS = {
    "um", "umm", "ummm", "uh", "uhh", "uhhh", "erm", "err",
    "hmm", "hmmm", "mm", "mhm", "uh-huh", "uhhuh",
}

# Kept around a removed filler so the cut doesn't clip the tail of a
# breath or the leading edge of the next real word — much smaller than
# DEAD_AIR_PADDING_SECONDS, since a filler sits inside continuous
# speech rather than at a natural pause.
_FILLER_CUT_PADDING_SECONDS = 0.05

# Never leave a fragment shorter than this after cutting fillers out of
# a block — a sliver this short is more likely mistimed word data than
# a genuine piece of speech worth its own hard cut.
_MIN_FRAGMENT_SECONDS = 0.15


def _is_pure_filler(word_text: str) -> bool:
    cleaned = word_text.strip().strip(string.punctuation).lower()
    return cleaned in _PURE_FILLER_WORDS


def _remove_filler_words(blocks: list, words: list) -> tuple[list, int]:
    """
    Punches filler-word intervals out of speech blocks that
    _find_speech_blocks already produced. Returns (new_blocks, count) —
    count is how many filler words were actually cut, surfaced to the
    creator so "3 filler words removed" means what it says.

    Returns (blocks, 0) unchanged when there's nothing to work from (no
    word timings) or nothing to cut, so a video with no word-level data
    degrades to dead-air removal only instead of raising.
    """
    if not words:
        return blocks, 0

    filler_words = [w for w in words if _is_pure_filler(str(w.get("word", "")))]
    if not filler_words:
        return blocks, 0

    cut_intervals = sorted(
        (max(0.0, float(w.get("start", 0)) - _FILLER_CUT_PADDING_SECONDS),
         float(w.get("end", 0)) + _FILLER_CUT_PADDING_SECONDS)
        for w in filler_words
    )

    # Merge overlapping/adjacent cuts so back-to-back fillers ("um,
    # uh...") don't leave an unnecessary sliver of near-silence between
    # them.
    merged_cuts = [cut_intervals[0]]
    for start, end in cut_intervals[1:]:
        last_start, last_end = merged_cuts[-1]
        if start <= last_end:
            merged_cuts[-1] = (last_start, max(last_end, end))
        else:
            merged_cuts.append((start, end))

    new_blocks = []
    for block_start, block_end in blocks:
        pieces = [(block_start, block_end)]
        for cut_start, cut_end in merged_cuts:
            if cut_end <= block_start or cut_start >= block_end:
                continue  # this cut doesn't touch this block at all
            next_pieces = []
            for piece_start, piece_end in pieces:
                if cut_end <= piece_start or cut_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if cut_start > piece_start:
                    next_pieces.append((piece_start, cut_start))
                if cut_end < piece_end:
                    next_pieces.append((cut_end, piece_end))
            pieces = next_pieces
        new_blocks.extend((s, e) for s, e in pieces if e - s >= _MIN_FRAGMENT_SECONDS)

    if not new_blocks:
        # Cutting every filler out would leave nothing — keep the
        # original blocks rather than hand back an empty clip.
        return blocks, 0

    return new_blocks, len(filler_words)


def _word_overlaps_blocks(word_start: float, word_end: float, blocks: list) -> bool:
    """
    Whether a word survives in the edited clip at all — not just
    whether it falls within the clip's overall start/end span, which a
    word sitting inside a removed dead-air or filler gap also does.
    Without this check, a cut filler word would still flash on screen
    as a caption for a beat at the exact point the video jumps over it.
    """
    return any(word_end > b_start and word_start < b_end for b_start, b_end in blocks)


def _build_time_remap(blocks: list):
    """
    Returns a function mapping a timestamp in the ORIGINAL video's
    timeline to where it lands in the concatenated, dead-air-removed
    output — needed because burning in captions generated from the
    original timeline would otherwise drift out of sync with the trimmed
    video as soon as more than one block exists. A timestamp that falls
    inside a removed gap is clamped to the nearest kept boundary.
    """
    cumulative = []
    total = 0.0
    for b_start, b_end in blocks:
        cumulative.append(total)
        total += b_end - b_start

    def remap(t: float) -> float:
        for i, (b_start, b_end) in enumerate(blocks):
            if t < b_start:
                return cumulative[i]
            if t <= b_end:
                return cumulative[i] + (t - b_start)
        return total

    return remap


# ---------------------------------------------------------
# Speaker-tracking crop
#
# The 9:16 reframe below this has always been a fixed center crop —
# fine for a subject who stays put, but anyone who moves (walks, turns
# to something off to the side, isn't centered in the original shot to
# begin with) drifts toward the edge of frame or out of it entirely.
# This samples face position across the clip and, when detection is
# reliable enough, generates a crop that pans to follow it instead.
#
# Deliberately conservative: a full computer-vision pipeline this is
# not — it's a Haar cascade (opencv's classic, dependency-light face
# detector) sampled every _FACE_SAMPLE_INTERVAL_SECONDS, smoothed, and
# turned into a piecewise-linear ffmpeg crop expression. When detection
# is too sparse or unreliable to trust, this falls back to the exact
# same static center crop used before it existed — never worse than
# the old behavior, only sometimes better.
# ---------------------------------------------------------

# 1.0s rather than a finer interval: detection cost scales linearly
# with clip duration (confirmed ~0.12s of processing per clip-second
# at this interval on real hardware), and a pan-based crop is meant to
# catch gradual drift over several seconds, not sub-second jitter —
# which the smoothing pass would flatten out anyway. Halving this to
# 0.5s roughly doubles processing time for detection this app
# measurably doesn't need.
_FACE_SAMPLE_INTERVAL_SECONDS = 1.0
# Below this fraction of samples actually finding a face, the detected
# positions are too sparse to build a trustworthy pan from — a couple
# of lucky hits scattered across a mostly-blank clip would produce a
# crop path that lurches between them rather than following anyone.
_MIN_DETECTION_RATIO = 0.4
_MIN_DETECTIONS = 3
# Exponential smoothing on raw detections before they become a pan
# path — frame-to-frame face-box jitter is real even on a still
# subject, and panning to every wobble would be worse than not
# tracking at all.
_SMOOTHING_ALPHA = 0.35

_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None and _FACE_TRACKING_AVAILABLE:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def _detect_face_centers(
    source_path: str, start_time: float, end_time: float
) -> tuple[list, int, int]:
    """
    Samples frames in [start_time, end_time] of the ORIGINAL video and
    returns (detections, source_width, source_height), where detections
    is [(original_timestamp, face_center_x_in_original_px), ...] for
    every sample that found a face. Never raises — any failure (a
    corrupt frame, cv2 erroring on a particular codec) just means fewer
    or zero detections, which the caller already treats as "not
    reliable enough to track."
    """
    cascade = _get_face_cascade()
    if cascade is None:
        return [], 0, 0

    cap = cv2.VideoCapture(source_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return [], 0, 0

        detections = []
        t = start_time
        while t < end_time:
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if ok:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
                    )
                    if len(faces):
                        # Largest box = most likely the actual subject,
                        # not a smaller face in the background.
                        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                        detections.append((t, fx + fw / 2))
            except Exception:
                pass  # this sample just doesn't contribute a detection
            t += _FACE_SAMPLE_INTERVAL_SECONDS

        return detections, width, height
    finally:
        cap.release()


def _smooth_detections(detections: list) -> list:
    """Exponential moving average over raw (t, x) detections, in time order."""
    if not detections:
        return []
    smoothed = [(detections[0][0], detections[0][1])]
    for t, x in detections[1:]:
        prev_x = smoothed[-1][1]
        smoothed.append((t, _SMOOTHING_ALPHA * x + (1 - _SMOOTHING_ALPHA) * prev_x))
    return smoothed


def _build_pan_expression(points: list, min_x: float, max_x: float) -> str:
    """
    A piecewise-linear ffmpeg expression for crop x as a function of
    `t`, interpolating between consecutive (t, x) points and holding
    the nearest value's clamped position before the first / after the
    last point. `points` must be sorted by time and non-empty.
    """
    def clamp(x):
        return max(min_x, min(max_x, x))

    if len(points) == 1:
        return f"{clamp(points[0][1]):.1f}"

    expr = f"{clamp(points[-1][1]):.1f}"
    for i in range(len(points) - 2, -1, -1):
        t0, x0 = points[i]
        t1, x1 = points[i + 1]
        x0c, x1c = clamp(x0), clamp(x1)
        if t1 == t0:
            interp = f"{x1c:.1f}"
        else:
            slope = (x1c - x0c) / (t1 - t0)
            interp = f"({x0c:.1f}+{slope:.3f}*(t-{t0:.2f}))"
        expr = f"if(lt(t,{t1:.2f}),{interp},{expr})"
    t0, x0 = points[0]
    return f"if(lt(t,{t0:.2f}),{clamp(x0):.1f},{expr})"


def _speaker_crop_x_expr(
    source_path: str,
    highlight_start: float,
    highlight_end: float,
    blocks: list,
    target_width: int,
    target_height: int,
) -> Optional[str]:
    """
    Returns an ffmpeg crop-x expression that pans to follow the
    detected speaker, or None when tracking isn't reliable enough —
    the caller should fall back to a plain static center crop on None,
    exactly as if this function didn't exist.

    target_width/target_height are the fixed output frame size (1080x1920
    for this app's 9:16 clips) — the crop's own width, and along with
    the source dimensions this discovers, enough to reproduce the exact
    scale factor ffmpeg's own `scale=...:force_original_aspect_ratio=
    increase` computes at render time, without asking the caller to
    already know a number only ffmpeg would otherwise compute.
    """
    if not _FACE_TRACKING_AVAILABLE:
        return None

    try:
        detections, source_width, source_height = _detect_face_centers(
            source_path, highlight_start, highlight_end
        )
    except Exception:
        return None

    if source_width <= 0 or source_height <= 0:
        return None

    total_samples = max(1, round((highlight_end - highlight_start) / _FACE_SAMPLE_INTERVAL_SECONDS))
    if len(detections) < _MIN_DETECTIONS or len(detections) / total_samples < _MIN_DETECTION_RATIO:
        return None

    # Original-video timestamps -> final edited-clip timestamps, same
    # remap already used for captions — a detection inside a cut gap
    # (dead air or a removed filler word) isn't a real, current
    # position, so it's dropped rather than clamped to a boundary and
    # treated as fresh data.
    remap = _build_time_remap(blocks)
    kept = [
        (remap(t), x) for t, x in detections
        if _word_overlaps_blocks(t, t + 0.01, blocks)
    ]
    if len(kept) < _MIN_DETECTIONS:
        return None

    kept.sort(key=lambda p: p[0])
    smoothed = _smooth_detections(kept)

    # Original-video pixel positions -> the scaled space the crop
    # filter actually operates in (it runs after `scale=...increase`,
    # which uniformly enlarges by whichever axis needs it more — the
    # same "increase" factor computed here).
    scale_factor = max(target_width / source_width, target_height / source_height)
    scaled_width = source_width * scale_factor
    scaled_points = [(t, x * scale_factor - target_width / 2) for t, x in smoothed]

    min_x = 0.0
    max_x = max(0.0, scaled_width - target_width)
    return _build_pan_expression(scaled_points, min_x, max_x)


def _generate_srt(segments: list, srt_path: str, blocks: list):
    def fmt(seconds: float) -> str:
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    remap = _build_time_remap(blocks)
    clip_start, clip_end = blocks[0][0], blocks[-1][1]

    with open(srt_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            s, e = seg.get("start", 0), seg.get("end", 0)
            if e < clip_start or s > clip_end:
                continue
            rel_start = remap(s)
            rel_end = max(rel_start + 0.3, remap(e))
            text = seg.get("text", "").strip().upper()
            if not text:
                continue
            f.write(f"{idx}\n{fmt(rel_start)} --> {fmt(rel_end)}\n{text}\n\n")
            idx += 1


# Karaoke caption styling. ASS colours are &HAABBGGRR (BGR, not RGB).
# DejaVu Sans is the font actually installed in the image
# (fonts-dejavu-core in the Dockerfile) — the old SRT path asked for
# Arial-Bold, which doesn't exist here, so fontconfig silently
# substituted whatever it could find.
_ASS_FONT = "DejaVu Sans"
_ASS_IDLE_COLOUR = "&H00FFFFFF"          # white
_ASS_ACTIVE_COLOUR = "&H00FFE500"        # Viyo cyan (#00E5FF) in BGR
_ASS_WORDS_PER_PHRASE = 4
# A pause longer than this starts a new caption phrase, so a phrase
# never spans a natural break in speech.
_ASS_PHRASE_GAP_SECONDS = 0.6


def _ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:  # rounding up a whole second must carry, not print ":60.100"
        cs = 0
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    # Braces delimit override tags in ASS, and a literal newline ends the
    # Dialogue line — neither can survive inside caption text.
    return text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def _group_words_into_phrases(words: list) -> list:
    """Consecutive words chunked into short on-screen phrases."""
    phrases = []
    current = []
    for word in words:
        if current:
            gap = float(word.get("start", 0)) - float(current[-1].get("end", 0))
            if len(current) >= _ASS_WORDS_PER_PHRASE or gap > _ASS_PHRASE_GAP_SECONDS:
                phrases.append(current)
                current = []
        current.append(word)
    if current:
        phrases.append(current)
    return phrases


# How long the hook banner stays burned across the top of a clip.
# Long enough to read, short enough to be gone before it competes with
# the payoff.
HOOK_OVERLAY_SECONDS = 2.5
_HOOK_OVERLAY_COLOUR = "&H0000E5FF"  # Viyo gold (#FFE500) in BGR


# Roughly what fits across 1080px at the Hook style's 50px bold, inside
# its side margins. WrapStyle 2 means libass will NOT wrap for us — an
# unwrapped hook runs straight off both edges of the frame.
_HOOK_CHARS_PER_LINE = 30


def _hook_overlay_dialogue(hook_line: str) -> Optional[str]:
    """
    One ASS Dialogue line pinning the hook across the top of the
    opening seconds — the on-screen promise that earns the next two
    seconds of attention, which captions alone don't do.
    """
    text = _ass_escape(hook_line).upper()
    if not text:
        return None
    # \N is ASS's hard line break; reusing the quote-card wrapper keeps
    # one wrapping implementation rather than two that drift apart.
    wrapped = _wrap_quote_text(text, _HOOK_CHARS_PER_LINE).replace("\n", "\\N")
    return (
        f"Dialogue: 0,{_ass_timestamp(0)},{_ass_timestamp(HOOK_OVERLAY_SECONDS)},"
        f"Hook,,0,0,0,,{wrapped}"
    )


def _generate_ass(words: list, ass_path: str, blocks: list, hook_line: str = "") -> bool:
    """
    Writes karaoke-style captions: the whole short phrase stays on
    screen while the word currently being spoken is recoloured and
    bumped up in size, the way hand-edited short-form captions work.

    Returns False when there's nothing usable to write (no word
    timings in range), so the caller can fall back to the plain
    line-at-a-time SRT path rather than burning in nothing.
    """
    remap = _build_time_remap(blocks)

    in_range = [
        w for w in (words or [])
        if _word_overlaps_blocks(float(w.get("start", 0)), float(w.get("end", 0)), blocks)
        and _ass_escape(str(w.get("word", "")))
    ]
    # Word timings drive the karaoke captions, but a hook overlay alone
    # is still worth burning in — losing it too because Whisper returned
    # no word data would drop the more valuable of the two.
    if not in_range and not _ass_escape(hook_line):
        return False

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Fontsize 54, not the 72 this started at — full-width karaoke
        # text at 72pt was covering enough of the frame to be a real
        # problem on any clip where what's actually on screen matters
        # (a plate of food, a product, hands doing something), not just
        # a talking head. Outline/shadow trimmed to match so the smaller
        # text doesn't look proportionally heavier than before.
        f"Style: Karaoke,{_ASS_FONT},54,{_ASS_IDLE_COLOUR},{_ASS_ACTIVE_COLOUR},"
        "&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4,1,2,80,80,260,1",
        # Top-anchored (Alignment 8) so it never collides with the
        # karaoke captions running along the bottom. Shrunk to match.
        f"Style: Hook,{_ASS_FONT},50,{_HOOK_OVERLAY_COLOUR},{_HOOK_OVERLAY_COLOUR},"
        "&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4,1,8,70,70,180,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    wrote_any = False
    hook_dialogue = _hook_overlay_dialogue(hook_line)
    if hook_dialogue:
        lines.append(hook_dialogue)
        wrote_any = True

    for phrase in _group_words_into_phrases(in_range):
        texts = [_ass_escape(str(w.get("word", ""))).upper() for w in phrase]
        for i, word in enumerate(phrase):
            start = remap(float(word.get("start", 0)))
            # Hold each word until the next one starts so the phrase
            # doesn't flicker off during the gaps between words.
            if i + 1 < len(phrase):
                end = remap(float(phrase[i + 1].get("start", 0)))
            else:
                end = remap(float(word.get("end", 0)))
            end = max(start + 0.08, end)

            rendered = " ".join(
                # 106%, not 112% — proportional to the smaller base size
                # above; the point is the active word still visibly pops,
                # not that it balloons.
                f"{{\\c{_ASS_ACTIVE_COLOUR}\\fscx106\\fscy106}}{t}{{\\r}}" if j == i else t
                for j, t in enumerate(texts)
            )
            lines.append(
                f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},"
                f"Karaoke,,0,0,0,,{rendered}"
            )
            wrote_any = True

    if not wrote_any:
        return False

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return True


# Streaming-style loudness target so a clip pulled from a quiet section
# of the source doesn't play noticeably quieter than one pulled from a
# loud section — quiet audio is one of the more common reasons a short
# gets scrolled past. -1.5dB true-peak headroom avoids clipping on
# playback. Single-pass rather than ffmpeg's two-pass loudnorm
# measure-then-apply: less precise, but doesn't require decoding the
# clip twice, and "close to -16 LUFS" already fixes the actual problem
# (silence vs. speech loudness), which matters far more than being
# exact.
_LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"


def _render_clip(input_path: str, output_path: str, srt_path: Optional[str], blocks: list):
    """
    Renders the highlight window, cutting out any dead-air gaps found
    between blocks (see _find_speech_blocks) instead of a single
    continuous cut — this is the actual editing step; picking a good
    highlight window alone doesn't remove the pauses/filler air inside it.

    Single-block case (nothing to trim) uses the same fast input-seek
    approach as before. The multi-block case still seeks to the first
    block before decoding — filter-based trims only skip frames *after*
    decode, so without this the whole video up to that point would be
    decoded for nothing.
    """
    seek_offset = blocks[0][0]
    rel_blocks = [(b_start - seek_offset, b_end - seek_offset) for b_start, b_end in blocks]

    # Pans the crop to follow the detected speaker when tracking is
    # reliable enough; falls back to the plain static center crop
    # (unchanged from before this existed) on anything less than that —
    # a wrong guess about where to pan is worse than not panning.
    crop_x_expr = _speaker_crop_x_expr(input_path, blocks[0][0], blocks[-1][1], blocks, 1080, 1920)
    crop_filter = f"crop=1080:1920:x='{crop_x_expr}'" if crop_x_expr else "crop=1080:1920"
    post_filter = f"scale=1080:1920:force_original_aspect_ratio=increase,{crop_filter}"
    if srt_path and os.path.exists(srt_path):
        safe_path = srt_path.replace("\\", "/").replace(":", "\\:")
        if srt_path.endswith(".ass"):
            # ASS carries its own styling (see _generate_ass) — passing
            # force_style here would flatten the per-word highlighting
            # back into one uniform colour.
            post_filter += f",subtitles='{safe_path}'"
        else:
            style = (
                f"FontName={_ASS_FONT},FontSize=24,PrimaryColour={_ASS_ACTIVE_COLOUR},"
                "Outline=2,Bold=1,Alignment=2"
            )
            post_filter += f",subtitles='{safe_path}':force_style='{style}'"

    if len(rel_blocks) == 1:
        start, end = rel_blocks[0]
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_offset), "-i", input_path, "-t", str(end - start),
            "-vf", post_filter,
            "-af", _LOUDNORM_FILTER,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
    else:
        filter_parts = []
        for i, (b_start, b_end) in enumerate(rel_blocks):
            filter_parts.append(f"[0:v]trim=start={b_start}:end={b_end},setpts=PTS-STARTPTS[v{i}]")
            filter_parts.append(f"[0:a]atrim=start={b_start}:end={b_end},asetpts=PTS-STARTPTS[a{i}]")
        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(rel_blocks)))
        filter_parts.append(f"{concat_inputs}concat=n={len(rel_blocks)}:v=1:a=1[vcat][acat]")
        filter_parts.append(f"[vcat]{post_filter}[vout]")
        filter_parts.append(f"[acat]{_LOUDNORM_FILTER}[aout]")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_offset), "-i", input_path,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Video render failed: {result.stderr[-500:]}")


def _wrap_quote_text(text: str, max_chars_per_line: int = 22) -> str:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars_per_line and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _render_quote_card(text: str, output_path: str) -> None:
    """
    Renders a static, shareable 1080x1080 quote-card image using the
    clip's own suggested_title as the headline — reusing a field that's
    already generated per clip rather than an extra GPT call. One upload
    now produces a video AND a postable image, not just the one format.

    Text goes through a temp file (drawtext's textfile= option) rather
    than being inlined into the filter string, which sidesteps ffmpeg
    filter-syntax escaping entirely for quotes/colons/apostrophes in
    AI-generated titles — verified against exactly that kind of text.
    """
    wrapped = _wrap_quote_text(text)
    fd, text_file_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(wrapped)

        filter_graph = (
            f"drawtext=textfile={text_file_path}:fontfile={QUOTE_CARD_FONT_PATH}:"
            "fontcolor=white:fontsize=64:x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=14"
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x13132B:s={QUOTE_CARD_SIZE}",
            "-vf", filter_graph,
            "-frames:v", "1",
            output_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Quote card render failed: {result.stderr[-500:]}")
    finally:
        os.unlink(text_file_path)


def _extract_thumbnail_candidates(clip_path: str, out_dir: str, count: int) -> list[str]:
    """
    Pulls `count` evenly-spaced frames from the rendered (already cropped
    9:16) clip as thumbnail candidates. Spaced across [10%, 90%] of the
    clip rather than [0%, 100%] — the very first/last frames are the most
    likely to be a mid-cut or fade artifact, so this avoids wasting a
    candidate slot on one of those.
    """
    duration = _run_ffprobe_duration(clip_path)
    if duration <= 0:
        return []

    if count == 1:
        timestamps = [duration / 2]
    else:
        lo, hi = duration * 0.1, duration * 0.9
        step = (hi - lo) / (count - 1)
        timestamps = [lo + i * step for i in range(count)]

    paths = []
    for i, ts in enumerate(timestamps):
        frame_path = os.path.join(out_dir, f"thumb_candidate_{i}.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts), "-i", clip_path,
            "-frames:v", "1", "-q:v", "3",
            frame_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and os.path.exists(frame_path):
            paths.append(frame_path)
    return paths


def _pick_best_thumbnail(candidate_paths: list[str]) -> int:
    """
    Sends every candidate frame to GPT-4o-mini's vision input in one call
    and asks it to pick the single most scroll-stopping one — the same
    "look at the actual pixels" pattern already used by /analyze-post,
    just choosing among frames instead of critiquing one. Frames go in as
    base64 data URLs since these are local temp files, never uploaded
    anywhere unless they win — no need to touch Storage for the 4 that lose.

    Falls back to the middle candidate (index len // 2, a reasonable
    "probably not a blank intro/outro frame" guess) on any parse failure
    or out-of-range answer, so a flaky/malformed model response degrades
    to a plausible thumbnail rather than raising.
    """
    fallback_index = len(candidate_paths) // 2

    content: list[dict] = [{
        "type": "text",
        "text": (
            "These are candidate thumbnail frames from one short vertical video, "
            "in order, labeled Frame 0 through Frame "
            f"{len(candidate_paths) - 1}. Pick the single frame that would work best "
            "as a scroll-stopping thumbnail: a clear, in-focus, expressive moment "
            "(a face mid-expression, a striking visual) rather than a blurry, "
            "transitional, or blank-looking frame. "
            'Respond with ONLY a JSON object like {"best_frame": 2} — no other text.'
        ),
    }]
    for path in candidate_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    try:
        completion = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=50,
        )
        raw = (completion.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return fallback_index
        parsed = json.loads(match.group(0))
        index = int(parsed.get("best_frame"))
        if 0 <= index < len(candidate_paths):
            return index
        return fallback_index
    except Exception:
        return fallback_index


def _run_repurpose_job(job_id: str, video_url: str, user_id: str) -> None:
    """
    The actual transcribe → find-highlights → render-per-clip pipeline,
    run outside the request/response cycle (see repurpose_video below for
    why). Every step here used to run inline in the POST handler and
    raise HTTPException straight into FastAPI's own error handling; there
    is no request to raise into from a background task, so this catches
    everything itself and records it on the job instead.
    """
    job = _repurpose_jobs.get(job_id)
    if job is None:
        return  # shouldn't happen — the job is created right before this is scheduled

    try:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = os.path.join(tmp, "source.mp4")

            # Download the video from Supabase Storage.
            # This is a server-to-server download (Railway → Supabase CDN)
            # and is not subject to Railway's inbound request timeout.
            _download_video(video_url, source_path)

            duration = _run_ffprobe_duration(source_path)
            if duration > MAX_INPUT_DURATION_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Video too long — max {MAX_INPUT_DURATION_SECONDS}s.",
                )

            whisper_result = _transcribe_with_openai(source_path)
            transcript_text = whisper_result.get("text", "")
            segments = whisper_result.get("segments", [])
            words = whisper_result.get("words", [])

            # Transcription (the slow, expensive part) happens once no matter
            # how many candidate clips come out of it — only the render/upload
            # step below repeats per clip, which is cheap by comparison.
            highlight_count = _highlight_count_for_duration(duration)
            highlights = _find_highlights(
                segments, duration, count=highlight_count, words=words
            )

            clips = []
            for i, highlight in enumerate(highlights):
                # Cut dead air out of the chosen window instead of rendering
                # it as one continuous clip — this is what actually edits the
                # clip rather than just picking where to cut it.
                blocks = _find_speech_blocks(segments, highlight.start_time, highlight.end_time)
                dead_air_removed = _dead_air_removed_seconds(blocks)

                # Second, finer pass: cuts filler words ("um", "uh") out
                # of the same blocks — these have actual sound, so no
                # silence gap above ever catches them.
                kept_before_filler = sum(b_end - b_start for b_start, b_end in blocks)
                blocks, filler_words_removed_count = _remove_filler_words(blocks, words)
                kept_after_filler = sum(b_end - b_start for b_start, b_end in blocks)
                filler_words_removed_seconds = round(kept_before_filler - kept_after_filler, 2)

                # Karaoke captions when word timings came back, plain
                # line-at-a-time SRT when they didn't — burning in the
                # simpler captions beats burning in none at all.
                captions_path = os.path.join(tmp, f"captions_{i}.ass")
                if not _generate_ass(words, captions_path, blocks, highlight.hook_line):
                    captions_path = os.path.join(tmp, f"captions_{i}.srt")
                    _generate_srt(segments, captions_path, blocks)

                output_path = os.path.join(tmp, f"output_{i}.mp4")
                _render_clip(source_path, output_path, captions_path, blocks)

                # Upload the finished clip to Supabase Storage — survives
                # Railway's ephemeral filesystem across redeploys.
                storage_path = f"{user_id}/{int(time.time())}_{i}.mp4"
                with open(output_path, "rb") as f:
                    try:
                        supabase_admin.storage.from_(PROCESSED_BUCKET).upload(
                            storage_path, f, file_options={"content-type": "video/mp4"}
                        )
                    except Exception as e:
                        raise HTTPException(status_code=502, detail=f"Storage upload failed: {e}")

                public_url = supabase_admin.storage.from_(PROCESSED_BUCKET).get_public_url(storage_path)

                # A second, cheap format from the same upload — a shareable
                # quote card, not just the video. Never fails the request:
                # the video clip is what actually matters, so a render or
                # upload problem here (e.g. a stricter bucket MIME policy)
                # just means this one clip has no quote card, not a 502.
                quote_card_url = None
                try:
                    quote_card_path = os.path.join(tmp, f"quote_{i}.png")
                    _render_quote_card(highlight.suggested_title, quote_card_path)
                    quote_card_storage_path = f"{user_id}/{int(time.time())}_{i}_quote.png"
                    with open(quote_card_path, "rb") as qf:
                        supabase_admin.storage.from_(PROCESSED_BUCKET).upload(
                            quote_card_storage_path, qf, file_options={"content-type": "image/png"}
                        )
                    quote_card_url = supabase_admin.storage.from_(PROCESSED_BUCKET).get_public_url(
                        quote_card_storage_path
                    )
                except Exception as e:
                    print(f"[WARN] Quote card failed for clip {i}: {e}")

                # A GPT-4o-mini vision pick of the most scroll-stopping frame
                # from the clip itself, uploaded as the poster image — most
                # feed UIs show the thumbnail before anyone presses play, so
                # this is the single biggest lever on whether a clip gets a
                # first tap at all. Same never-fail-the-request pattern as
                # the quote card above.
                thumbnail_url = None
                try:
                    candidates = _extract_thumbnail_candidates(output_path, tmp, THUMBNAIL_CANDIDATE_COUNT)
                    if candidates:
                        best_index = _pick_best_thumbnail(candidates)
                        thumbnail_storage_path = f"{user_id}/{int(time.time())}_{i}_thumb.jpg"
                        with open(candidates[best_index], "rb") as tf:
                            supabase_admin.storage.from_(PROCESSED_BUCKET).upload(
                                thumbnail_storage_path, tf, file_options={"content-type": "image/jpeg"}
                            )
                        thumbnail_url = supabase_admin.storage.from_(PROCESSED_BUCKET).get_public_url(
                            thumbnail_storage_path
                        )
                except Exception as e:
                    print(f"[WARN] Thumbnail selection failed for clip {i}: {e}")

                clips.append(RepurposeClipResult(
                    processed_video_url=public_url,
                    highlight=highlight,
                    dead_air_removed_seconds=dead_air_removed,
                    filler_words_removed_seconds=filler_words_removed_seconds,
                    filler_words_removed_count=filler_words_removed_count,
                    quote_card_url=quote_card_url,
                    thumbnail_url=thumbnail_url,
                ))

        # Runs after the clips so it can cite the best hook score found,
        # and never blocks them: _critique_video returns None on failure
        # rather than raising.
        signals = _measure_video_signals(words, segments, duration)
        feedback = _critique_video(transcript_text, signals, highlights)

        job.result = RepurposeResponse(
            status="success",
            transcript=transcript_text,
            clips=clips,
            feedback=feedback,
        )
        job.status = "done"
    except HTTPException as e:
        job.error = str(e.detail)
        job.status = "failed"
    except Exception as e:
        job.error = f"Unexpected error: {e}"
        job.status = "failed"


@router.post("/repurpose", response_model=RepurposeJobStartResponse)
async def repurpose_video(
    req: RepurposeRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Starts video repurposing as a background job and returns immediately
    with a job_id, instead of processing everything inline and holding
    the connection open until it's done.

    Why: with clip count now scaled to source length (up to
    MAX_HIGHLIGHT_CANDIDATES), a long video's total pipeline —
    transcription plus one render + thumbnail-selection + quote-card
    pass per clip — can take several minutes. The old synchronous
    version had to fit that entire pipeline inside one HTTP round trip;
    if it ran past Railway's request-handling limit, the connection was
    killed and the client got nothing back even if the work would have
    finished. Returning a job_id immediately means the slow work no
    longer has to fit inside one request — the client polls
    GET /repurpose/{job_id} instead (see below).

    Flutter still uploads the raw video to Supabase Storage first and
    sends only the URL here, for the same reason as before: this keeps
    large file data off the Railway inbound proxy.
    """
    if supabase_admin is None:
        raise HTTPException(
            status_code=503,
            detail="Video repurposing isn't configured yet — SUPABASE_SERVICE_ROLE_KEY is missing on the server.",
        )

    _check_repurpose_rate_limit(user_id)
    # Fail fast on a malformed/malicious URL before spending any coins
    # or creating a job for it.
    _validate_storage_url(req.video_url)

    # Charged upfront — no refund path if a later step fails (e.g. video
    # too long). Simpler than partial-completion accounting for a first
    # pass, and this is already the rarest, most rate-limited call in the app.
    spend_on_feature(supabase_admin, user_id, "repurpose")

    job_id = str(uuid.uuid4())
    _repurpose_jobs[job_id] = _RepurposeJob(user_id)
    _cleanup_expired_repurpose_jobs()

    background_tasks.add_task(_run_repurpose_job, job_id, req.video_url, user_id)

    return RepurposeJobStartResponse(job_id=job_id, status="processing")


@router.get("/repurpose/{job_id}", response_model=RepurposeJobStatusResponse)
async def get_repurpose_job(
    job_id: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Polled by the client until status is "done" or "failed" — see
    repurpose_video above for why this is a separate step instead of one
    long blocking POST."""
    job = _repurpose_jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found or expired — repurpose jobs are only kept for about an hour.",
        )
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not your job.")

    return RepurposeJobStatusResponse(status=job.status, result=job.result, error=job.error)
