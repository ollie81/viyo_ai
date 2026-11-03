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
import datetime
import string
import tempfile
import subprocess
import urllib.parse
import urllib.request
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from openai import OpenAI
from supabase import create_client, Client

import clip_selection
from coins import FEATURE_COSTS
from repurpose_jobs import JobConfig, JobStore, TransientJobError, supabase_rpc
from repurpose_storage import CLIPS_BUCKET, SOURCE_BUCKET, SupabaseStorage

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

# Text/vision model for this file's chat calls (clip selection, footage
# critique, description-only packaging, thumbnail pick) — not
# transcription, which is whisper-1. Defaults to the same OPENAI_MODEL
# the rest of the backend runs on in production (gpt-4o-mini), which is
# known to work on this account. It used to be hard-coded to
# "gpt-6-luna", which was never verified: the first live jobs failed on
# every selection call, and the previous code would have hidden any such
# failure by substituting a made-up clip.
REPURPOSE_TEXT_MODEL = os.environ.get(
    "REPURPOSE_TEXT_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
# Reasoning level for CLIP SELECTION only, set alongside a selection model
# that supports it (e.g. a gpt-5.x model). Empty means the parameter isn't
# sent, which non-reasoning models like gpt-4o-mini require. Kept separate
# from the other calls in this file on purpose: those run on
# REPURPOSE_TEXT_MODEL, which can be a different model that would reject it.
REPURPOSE_SELECTION_REASONING_EFFORT = os.environ.get("REPURPOSE_SELECTION_REASONING_EFFORT", "")


def _llm_kwargs(temperature: float) -> dict:
    return {"temperature": temperature}

# How many evenly-spaced frames to pull from a rendered clip as thumbnail
# candidates. 5 is enough spread to catch a genuinely different moment
# (talking head vs. a reaction vs. a b-roll cutaway) without ballooning
# the vision call's cost/latency — each candidate is one more image token
# block in the same request.
THUMBNAIL_CANDIDATE_COUNT = 5

REPURPOSE_SELECTION_MODEL = os.environ.get("REPURPOSE_SELECTION_MODEL", REPURPOSE_TEXT_MODEL)

job_config = JobConfig.from_env(cost=FEATURE_COSTS["repurpose"]["cost"])
job_store: Optional[JobStore] = JobStore(supabase_rpc(supabase_admin), job_config) if supabase_admin else None
storage: Optional[SupabaseStorage] = SupabaseStorage(supabase_admin, SUPABASE_URL) if supabase_admin else None

_ALLOWED_UPLOAD_TYPES = {
    "video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm",
    "video/x-matroska": "mkv", "video/3gpp": "3gp", "video/x-m4v": "m4v",
    "application/octet-stream": "mp4",
}
# Supabase signed upload URLs are valid for two hours.
_SIGNED_UPLOAD_TTL_SECONDS = 2 * 60 * 60


def selection_llm(client: OpenAI) -> clip_selection.LLMFn:
    """
    The clip-selection LLM call. If the model rejects the request outright
    (HTTP 400) while a reasoning level is configured, it retries once
    without it and keeps it off for the rest of the process.
    """
    import openai

    state = {"reasoning": bool(REPURPOSE_SELECTION_REASONING_EFFORT)}

    def create(messages: list[dict]):
        # Reasoning models reject temperature and non-reasoning models
        # reject reasoning_effort, so send exactly one. Temperature 0 so
        # the same video gets the same picks.
        kwargs = ({"reasoning_effort": REPURPOSE_SELECTION_REASONING_EFFORT}
                  if state["reasoning"] else {"temperature": 0})
        return client.chat.completions.create(
            model=REPURPOSE_SELECTION_MODEL,
            messages=messages,
            max_completion_tokens=8000,
            **kwargs,
        )

    def call(messages: list[dict]) -> str:
        try:
            response = create(messages)
        except openai.BadRequestError as e:
            if not state["reasoning"]:
                raise TransientJobError("selection_unavailable",
                                        "Clip selection is temporarily unavailable.", f"LLM rejected request: {e}")
            print(f"[WARN] selection model rejected reasoning_effort ({e}); retrying without it")
            state["reasoning"] = False
            try:
                response = create(messages)
            except openai.APIError as e2:
                raise TransientJobError("selection_unavailable",
                                        "Clip selection is temporarily unavailable.", f"LLM error: {e2}")
        except openai.APIError as e:
            raise TransientJobError("selection_unavailable",
                                    "Clip selection is temporarily unavailable.", f"LLM error: {e}")
        content = response.choices[0].message.content or ""
        if not content.strip():
            print(f"[WARN] selection model returned no text "
                  f"(finish_reason={getattr(response.choices[0], 'finish_reason', None)})")
        return content

    return call


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
    # Exactly one of these. source_path is the current flow: a path in
    # the private repurpose-sources bucket from POST /repurpose/uploads.
    # video_url is a public posts-media URL, still accepted so app
    # versions released before the private bucket keep working.
    source_path: Optional[str] = Field(None, max_length=300)
    # "bunny" when source_path is "<user_id>/<bunny video id>" from POST /repurpose/bunny-uploads.
    source: Optional[str] = Field(None, max_length=10)
    video_url: Optional[str] = Field(None, description="Legacy: public Supabase Storage URL")
    # One per "Create clips" tap. Re-sending the same value (a retried
    # request after a timeout) returns the same job without a second charge.
    client_request_id: Optional[str] = Field(None, min_length=8, max_length=100)
    # Optional. Normal videos with speech never need this — highlights,
    # captions and hook lines all come from the transcript as before.
    # It only does anything when there's no speech to work with (a
    # silent video, or one with music/ambient sound Whisper can't
    # transcribe): the creator can describe what they want the clip to
    # look like instead of the upload being rejected outright. See
    # _highlights_from_prompt.
    creative_prompt: Optional[str] = Field(None, max_length=500)
    # Preferred clip length. A target, not a cut-off: a moment that needs a
    # little longer to land its hook and payoff keeps it (60 s maximum).
    clip_length: Optional[Literal["auto", "short", "medium", "long"]] = None


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
    # The exact words the clip opens on (what viewers hear first).
    hook_line: str = ""
    # The short line burned across the top of the opening seconds, built
    # only from what the speaker says. None when no faithful hook fit.
    overlay_hook: Optional[str] = None
    category: Optional[str] = None
    # What's said in the clip, word for word.
    transcript: str = ""
    # How the 0-100 score was built (content rubric, delivery,
    # structure, penalties), so the number can be explained.
    score_breakdown: dict = {}


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
    # How many clips the video could support, how many made it, and an
    # honest note when it's fewer than hoped.
    selection: Optional[dict] = None


class RepurposeJobStartResponse(BaseModel):
    job_id: str
    status: str  # always "processing" — a job was just created
    free_trial: bool = False   # this run is the account's free first video


class RepurposeJobStatusResponse(BaseModel):
    status: str  # "processing" | "done" | "failed"
    result: Optional[RepurposeResponse] = None
    error: Optional[str] = None
    # Finer-grained progress for the current app; older versions ignore it.
    stage: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None
    coins_refunded: Optional[int] = None
    # Technical detail of the latest failed attempt (keys redacted), so a
    # creator or support can report the real cause from a screenshot.
    error_detail: Optional[str] = None


class RepurposeUploadRequest(BaseModel):
    content_type: str = Field(..., max_length=100)
    size_bytes: int = Field(..., gt=0)


class RepurposeUploadResponse(BaseModel):
    # PUT the file body here (Content-Type = the type declared above).
    upload_url: str
    source_path: str
    max_bytes: int
    expires_in_seconds: int


# Before the repurpose_jobs table, job state was a JSON file per job in
# this bucket. Nothing writes these any more; they're still read (so a
# job started just before the deploy can be polled to completion) and
# swept (see repurpose_cleanup) until the last of them expires.
_REPURPOSE_JOB_TTL_SECONDS = 60 * 60
_JOB_PREFIX = "_jobs"


def _job_path(job_id: str) -> str:
    return f"{_JOB_PREFIX}/{job_id}.json"


def _read_job(job_id: str) -> Optional[dict]:
    try:
        raw = supabase_admin.storage.from_(PROCESSED_BUCKET).download(_job_path(job_id))
        return json.loads(raw)
    except Exception:
        # Covers both "genuinely doesn't exist" (404 from Storage) and
        # a transient Storage hiccup — either way the client sees the
        # same "not found or expired" the TTL case already produces,
        # since there's nothing more specific to tell them.
        return None


def _cleanup_expired_repurpose_jobs() -> None:
    """Best-effort sweep of job records past their TTL. Never raises —
    a failed listing/delete here must not block the new job that
    triggered it."""
    try:
        entries = supabase_admin.storage.from_(PROCESSED_BUCKET).list(
            _JOB_PREFIX, {"limit": 1000}
        )
    except Exception as e:
        print(f"[WARN] Could not list repurpose jobs for cleanup: {e}")
        return

    cutoff = time.time() - _REPURPOSE_JOB_TTL_SECONDS
    stale_paths = []
    for entry in entries:
        name = entry.get("name")
        created_at_raw = entry.get("created_at")
        if not name or not created_at_raw:
            continue
        try:
            created_at = datetime.datetime.fromisoformat(
                created_at_raw.replace("Z", "+00:00")
            ).timestamp()
        except (ValueError, AttributeError):
            continue
        if created_at < cutoff:
            stale_paths.append(f"{_JOB_PREFIX}/{name}")

    if stale_paths:
        try:
            supabase_admin.storage.from_(PROCESSED_BUCKET).remove(stale_paths)
        except Exception as e:
            print(f"[WARN] Could not remove expired repurpose jobs: {e}")


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


def _has_audio_stream(path: str) -> bool:
    """
    Whether the source video has any audio track at all.

    Confirmed by direct reproduction: a silent video run through the
    extraction command below (ffmpeg's segment muxer, -vn dropping the
    video stream) fails with a bare "Output file does not contain any
    stream" / "Invalid argument" — there's nothing left to write once
    the one stream that existed is dropped. That's a real, ordinary
    case (a muted recording, a video edited elsewhere with its audio
    stripped), not a corrupt file, so it gets caught here and reported
    as what it actually is instead of surfacing raw ffmpeg internals.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return bool(result.stdout.strip())


def _highlights_from_prompt(creative_prompt: str, max_duration: float) -> HighlightSegment:
    """
    The single clip for a video with no usable speech (no audio track,
    or only music/ambient sound), packaged from the creator's own
    description instead of a transcript: the opening of the video,
    capped at MAX_OUTPUT_CLIP_SECONDS.

    There is nothing to measure, so the score is 0 and the category is
    "prompt" (the app hides the score for these). If the model can't
    write the packaging, this raises a retryable error rather than
    returning a generic "Featured Clip".
    """
    clip_length = min(max_duration, MAX_OUTPUT_CLIP_SECONDS)
    prompt = (
        "A creator uploaded a video with no usable spoken audio (silent, or "
        "music/ambient sound with nothing to transcribe) and described what "
        f'they want the clip to look like: "{creative_prompt.strip()}"\n\n'
        "Write short-form packaging for this clip based on that description. "
        "Return ONLY raw JSON with keys: suggested_title (short, punchy), "
        "caption (1-2 short lines in the creator's own voice, no hashtags "
        "inside it, no quotation marks), hashtags (3-6 lowercase tags "
        "relevant to the description, without the # symbol), hook_line (a "
        "short on-screen text overlay for the opening seconds matching the "
        "vibe described, under 12 words)."
    )
    try:
        response = ai_client.chat.completions.create(
            model=REPURPOSE_TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert short-form content editor who "
                                              "specializes in retention-optimized hooks."},
                {"role": "user", "content": prompt},
            ],
            **_llm_kwargs(0.5),
        )
        data = clip_selection.parse_json_object(response.choices[0].message.content or "")
        title = str(data.get("suggested_title", "")).strip()
        if not title:
            raise ValueError("no title in response")
    except Exception as e:
        raise TransientJobError("selection_unavailable",
                                "Couldn't write your clip from the description.", str(e))
    return HighlightSegment(
        start_time=0.0,
        end_time=clip_length,
        reason="Based on your description — this video has no speech to choose moments from.",
        suggested_title=title[:clip_selection.MAX_TITLE_CHARS],
        score=0,
        category="prompt",
        caption=str(data.get("caption", "")).strip()[:clip_selection.MAX_CAPTION_CHARS],
        hashtags=clip_selection.clean_hashtags(data.get("hashtags") or []),
        hook_line=str(data.get("hook_line", "")).strip()[:clip_selection.MAX_HOOK_LINE_CHARS],
    )


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
    transcript: str, signals: dict, best_score: int
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
            model=REPURPOSE_TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a blunt, experienced short-form video coach. You "
                               "tell creators the truth about their footage. You coach for "
                               "Viyo, which also has an AI Short Drama format — numbered "
                               "episodes in a series, first 3 free, coins to unlock the rest. "
                               "If this footage has an ongoing character, story thread, or "
                               "clearly isn't the whole story, that's worth naming as a "
                               "strength suited to posting as a Short Drama episode rather "
                               "than a standalone clip — only when it's actually true of this "
                               "footage, never as a generic pitch.",
                },
                {"role": "user", "content": prompt},
            ],
            **_llm_kwargs(0.4),
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

    Each kept block is its own input, seeked straight to its start.
    Trimming blocks out of one shared input instead makes ffmpeg buffer
    every decoded frame between the branches: memory grew ~1 GB per cut
    on 1080p sources (1.3 GB for one block, 4.3 GB for four), enough to
    get the worker OOM-killed on a clip with normal dead-air removal.
    """

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

    inputs = []
    for b_start, b_end in blocks:
        inputs += ["-ss", f"{b_start:.3f}", "-t", f"{b_end - b_start:.3f}", "-i", input_path]
    filter_parts = []
    for i in range(len(blocks)):
        filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(blocks)))
    filter_parts.append(f"{concat_inputs}concat=n={len(blocks)}:v=1:a=1[vcat][acat]")
    filter_parts.append(f"[vcat]{post_filter}[vout]")
    filter_parts.append(f"[acat]{_LOUDNORM_FILTER}[aout]")

    cmd = [
        "ffmpeg", "-y",
        *inputs,
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
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                                           "detail": "low"}})

    try:
        completion = ai_client.chat.completions.create(
            model=REPURPOSE_TEXT_MODEL,
            messages=[{"role": "user", "content": content}],
            **_llm_kwargs(0),
            max_completion_tokens=50,
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


def _require_jobs() -> JobStore:
    if job_store is None or storage is None:
        raise HTTPException(
            status_code=503,
            detail="Video repurposing isn't configured yet — SUPABASE_SERVICE_ROLE_KEY is missing on the server.",
        )
    return job_store


def _max_upload_bytes() -> int:
    """The smaller of our own input cap and the private bucket's limit.
    The project-wide limit (dashboard setting) isn't readable over the
    API; if it's lower, the upload itself is rejected with HTTP 413."""
    bucket_limit = storage.bucket_file_size_limit(SOURCE_BUCKET) if storage else None
    return min(MAX_INPUT_SIZE_BYTES, bucket_limit) if bucket_limit else MAX_INPUT_SIZE_BYTES


@router.post("/repurpose/uploads", response_model=RepurposeUploadResponse)
async def create_repurpose_upload(
    req: RepurposeUploadRequest,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Step 1 of a repurpose: issues a signed upload URL for a server-chosen
    path in the private repurpose-sources bucket. The creator's video is
    never publicly readable, and nobody can write to anyone else's path.
    Nothing is charged here.
    """
    _require_jobs()
    content_type = req.content_type.split(";")[0].strip().lower()
    ext = _ALLOWED_UPLOAD_TYPES.get(content_type)
    if ext is None:
        raise HTTPException(status_code=400, detail="That file type isn't supported. Upload an MP4, MOV or WebM video.")
    max_bytes = _max_upload_bytes()
    if req.size_bytes > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Video too large — max {max_bytes // (1024 * 1024)} MB.",
        )

    path = f"{user_id}/{uuid.uuid4().hex}.{ext}"
    try:
        signed = storage.signed_upload(SOURCE_BUCKET, path)
        job_store.register_upload(user_id, SOURCE_BUCKET, path, req.size_bytes, content_type)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not prepare the upload: {e}")
    return RepurposeUploadResponse(
        upload_url=signed["url"], source_path=path, max_bytes=max_bytes,
        expires_in_seconds=_SIGNED_UPLOAD_TTL_SECONDS,
    )


class RepurposeBunnyUploadRequest(BaseModel):
    size_bytes: int = Field(..., gt=0)


class RepurposeBunnyUploadResponse(BaseModel):
    video_id: str
    library_id: str
    upload_endpoint: str
    authorization_signature: str
    authorization_expire: int
    playback_url: str
    thumbnail_url: str
    source_path: str
    max_bytes: int


@router.post("/repurpose/bunny-uploads", response_model=RepurposeBunnyUploadResponse)
async def create_repurpose_bunny_upload(
    req: RepurposeBunnyUploadRequest,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Step 1 (Bunny route): creates a video in the Bunny Stream library and
    returns a single-video upload credential. Used instead of the Supabase
    signed upload when the video is bigger than the storage plan's upload
    cap. The video is registered to this user, so only they can start a job
    from it. Nothing is charged here.
    """
    _require_jobs()
    if req.size_bytes > MAX_INPUT_SIZE_BYTES:
        raise HTTPException(status_code=400,
                            detail=f"Video too large — max {MAX_INPUT_SIZE_BYTES // (1024 * 1024)} MB.")
    import bunny_stream
    ticket = bunny_stream.create_upload("AI Repurposer source")
    path = f"{user_id}/{ticket.video_id}"
    try:
        job_store.register_upload(user_id, "bunny", path, req.size_bytes, "video/mp4")
    except Exception as e:
        bunny_stream.delete_video(ticket.video_id)
        raise HTTPException(status_code=502, detail=f"Could not prepare the upload: {e}")
    return RepurposeBunnyUploadResponse(**ticket.model_dump(), source_path=path, max_bytes=MAX_INPUT_SIZE_BYTES)


def _check_bunny_source(user_id: str, source_path: str) -> None:
    """Rejects (400/502) unless this user's Bunny video exists, is fully
    processed, and is within the size and length limits."""
    import bunny_stream
    if not source_path.startswith(f"{user_id}/"):
        raise HTTPException(status_code=400, detail=_CREATE_ERRORS["unknown_upload"][1])
    video_id = source_path.rsplit("/", 1)[-1]
    try:
        info = bunny_stream.get_video_info(video_id)
    except bunny_stream.BunnyVideoNotFound:
        raise HTTPException(status_code=400, detail="Your video hasn't finished uploading. Please try again.")
    except bunny_stream.BunnyError as e:
        raise HTTPException(status_code=502, detail=f"Could not check your video: {e}")
    if bunny_stream.is_failed(info):
        raise HTTPException(status_code=400, detail="We couldn't process that video. Try a different file.")
    if not bunny_stream.is_finished(info):
        raise HTTPException(status_code=400,
                            detail="Your video is still being prepared. Please try again in a moment.")
    # No size check on Bunny's own "storageSize": it counts every rendition Bunny
    # made, so it dwarfs the original. The declared size was capped at ticket
    # time and the download itself is capped while streaming.
    if float(info.get("length") or 0) > MAX_INPUT_DURATION_SECONDS:
        raise HTTPException(status_code=400,
                            detail=f"This video is too long — the limit is {MAX_INPUT_DURATION_SECONDS // 60} minutes.")


_CREATE_ERRORS = {
    "daily_limit": (429, None),
    "profile_not_found": (404, "Your profile couldn't be found."),
    "unknown_upload": (400, "That upload wasn't found. Please upload the video again."),
    "upload_already_used": (409, "That upload was already used. Please upload the video again."),
}


@router.post("/repurpose", response_model=RepurposeJobStartResponse)
async def repurpose_video(
    req: RepurposeRequest,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Step 2: charges the creator and queues the job, atomically, and
    returns immediately. The worker service does the processing; poll
    GET /repurpose/{job_id}. Re-sending the same client_request_id
    returns the same job without charging again.
    """
    store = _require_jobs()
    if bool(req.source_path) == bool(req.video_url):
        raise HTTPException(status_code=400, detail="Send exactly one of source_path or video_url.")

    if req.source_path and req.source == "bunny":
        _check_bunny_source(user_id, req.source_path)
        kind, bucket, path, url = "private_upload", "bunny", req.source_path, None
    elif req.source_path:
        if not req.source_path.startswith(f"{user_id}/"):
            raise HTTPException(status_code=400, detail=_CREATE_ERRORS["unknown_upload"][1])
        size = storage.object_size(SOURCE_BUCKET, req.source_path)
        if size is None:
            raise HTTPException(status_code=400, detail="Your video hasn't finished uploading. Please try again.")
        if size > MAX_INPUT_SIZE_BYTES:
            raise HTTPException(status_code=400, detail=f"Video too large — max {MAX_INPUT_SIZE_BYTES // (1024 * 1024)} MB.")
        kind, bucket, path, url = "private_upload", SOURCE_BUCKET, req.source_path, None
    else:
        _validate_storage_url(req.video_url)
        kind, bucket, path, url = "legacy_public_url", None, None, req.video_url

    job_args = dict(source_kind=kind, source_bucket=bucket, source_path=path, source_url=url,
                    creative_prompt=req.creative_prompt, client_request_id=req.client_request_id,
                    clip_length=None if req.clip_length in (None, "auto") else req.clip_length)
    # The account's free first video, when it's still available; otherwise paid.
    outcome = store.create_free_job(user_id, **job_args)
    if outcome is None:
        try:
            outcome = store.create_job(user_id, **job_args)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Could not start job: {e}")

    if not outcome.get("ok"):
        error = outcome.get("error")
        if error == "insufficient_coins":
            raise HTTPException(status_code=402, detail={
                "error": "insufficient_coins", "feature": "repurpose",
                "balance": outcome.get("balance", 0), "needed": outcome.get("needed", job_config.cost),
            })
        if error == "daily_limit":
            raise HTTPException(
                status_code=429,
                detail=f"Limit reached: {job_config.daily_limit} AI repurposes per day. Try again tomorrow.",
            )
        status, message = _CREATE_ERRORS.get(error, (400, "Could not start the job."))
        raise HTTPException(status_code=status, detail=message)

    free = bool(outcome.get("free_trial"))
    if not free and outcome.get("created") is False:
        # A re-sent tap returns the existing job; report whether that job was free.
        try:
            free = bool((store.get_job(str(outcome["job_id"])) or {}).get("free_trial"))
        except Exception:
            free = False
    return RepurposeJobStartResponse(job_id=str(outcome["job_id"]), status="processing", free_trial=free)


class BrandedClipResponse(BaseModel):
    url: str


@router.post("/repurpose/{job_id}/clips/{rank}/branded", response_model=BrandedClipResponse)
async def branded_clip(job_id: str, rank: int, user_id: str = Depends(_get_current_user_id)):
    """A copy of one finished clip with a small "Made with Viyo" tag, for
    downloading or sharing outside the app. Made once, then reused."""
    import asyncio
    import tempfile
    import clip_brand

    store = _require_jobs()
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Clip not found.")
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    if str(job.get("user_id")) != user_id:
        raise HTTPException(status_code=403, detail="Not your clip.")
    clip = next((c for c in job.get("clips") or [] if c.get("rank") == rank and c.get("video_url")), None)
    if clip is None or storage is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    src_path = f"{user_id}/{job_id}/clip_{rank}.mp4"
    out_path = f"{user_id}/{job_id}/clip_{rank}_viyo.mp4"
    if storage.object_size(CLIPS_BUCKET, out_path):
        return BrandedClipResponse(url=storage.public_url(CLIPS_BUCKET, out_path))

    def make() -> str:
        with tempfile.TemporaryDirectory(prefix="brand_") as work:
            src, out = os.path.join(work, "in.mp4"), os.path.join(work, "out.mp4")
            storage.download(CLIPS_BUCKET, src_path, src, 300 * 1024 * 1024)
            clip_brand.brand_clip(src, out)
            return storage.upload(CLIPS_BUCKET, out_path, out, "video/mp4")

    try:
        url = await asyncio.to_thread(make)
    except Exception as e:
        print(f"[WARN] branded clip {job_id}/{rank}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't add the Viyo tag right now. Try again.")
    return BrandedClipResponse(url=url)


@router.get("/repurpose/free-trial")
async def repurpose_free_trial(user_id: str = Depends(_get_current_user_id)):
    """Whether this account's next AI Repurposer run is its free first video."""
    status = _require_jobs().free_trial_status(user_id)
    return {"eligible": bool(status.get("eligible")), "used": bool(status.get("used")),
            "sold_out": bool(status.get("sold_out")),
            "max_minutes": job_config.free_trial_max_seconds // 60}


_LEGACY_STATUS = {"queued": "processing", "processing": "processing",
                  "succeeded": "done", "failed": "failed"}


@router.get("/repurpose/{job_id}", response_model=RepurposeJobStatusResponse)
async def get_repurpose_job(
    job_id: str,
    user_id: str = Depends(_get_current_user_id),
):
    store = _require_jobs()
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = store.get_job(job_id)
    if job is None:
        legacy = _read_job(job_id)
        if legacy is None:
            raise HTTPException(status_code=404, detail="Job not found or expired.")
        if legacy.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Not your job.")
        result = legacy.get("result")
        return RepurposeJobStatusResponse(
            status=legacy.get("status", "processing"),
            result=RepurposeResponse(**result) if result else None,
            error=legacy.get("error"),
        )

    if str(job.get("user_id")) != user_id:
        raise HTTPException(status_code=403, detail="Not your job.")

    status = job.get("status", "queued")
    refunded = job.get("refund_amount") if job.get("refunded_at") else None
    error = None
    if status == "failed":
        error = job.get("error") or "Processing failed."
        if refunded:
            error += f" Your {refunded} coins were refunded."
    result = job.get("result")
    detail = job.get("last_error") if (status == "failed" or job.get("error_code")) else None
    if detail:
        detail = re.sub(r"(sk|key|eyJ)[-_A-Za-z0-9.*]{6,}", "[redacted]", str(detail))[:3500]
    return RepurposeJobStatusResponse(
        status=_LEGACY_STATUS.get(status, "processing"),
        result=RepurposeResponse(**result) if result else None,
        error=error,
        stage=job.get("stage"),
        progress=job.get("progress"),
        message=job.get("status_message") or ("Waiting for a processing slot" if status == "queued" else None),
        coins_refunded=refunded,
        error_detail=detail,
    )
