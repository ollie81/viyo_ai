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

4. Hard limits on input size/duration. Video processing (transcribe +
   AI analysis + FFmpeg render) is slow and memory-heavy. Without a
   cap, a long video would likely time out the request or exceed
   Railway's memory limit. This version rejects anything over 3
   minutes / 60MB up front with a clear error, rather than trying and
   silently failing or crashing.

5. Uses the same JWT auth + rate limiting pattern as the rest of
   main.py, instead of being wide open.

6. Fixed PyJWT "alg not allowed" error: the verify_signature=False
   fallback now explicitly passes algorithms=["HS256"] as required
   by PyJWT 2.x.

7. Registered as a FastAPI router, imported into main.py — kept in its
   own file so it can't accidentally break the endpoints that are
   already working.
"""

import os
import re
import json
import time
import tempfile
import subprocess
import urllib.request
from collections import defaultdict, deque
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from openai import OpenAI
from supabase import create_client, Client

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
MAX_INPUT_DURATION_SECONDS = 180   # 3 minutes of source video
MAX_INPUT_SIZE_BYTES = 60 * 1024 * 1024  # 60 MB
MAX_OUTPUT_CLIP_SECONDS = 60

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
    """Same JWT-verification pattern as main.py's get_current_user_id —
    duplicated here rather than imported to keep this module fully
    self-contained and safe to add without touching main.py's imports.

    Fix: the verify_signature=False fallback now passes algorithms=["HS256"]
    explicitly. PyJWT 2.x requires this even when not verifying the
    signature — without it, certain alg header values are rejected with
    "The specified alg value is not allowed".
    """
    import jwt

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    secret = os.environ.get("SUPABASE_JWT_SECRET", "")

    try:
        if secret:
            payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
        else:
            # Local dev fallback only — ALLOW_INSECURE_AUTH=true must be set
            # for main.py to even start without a secret.
            # algorithms= is required by PyJWT 2.x even when verify_signature=False.
            payload = jwt.decode(
                token,
                options={"verify_signature": False},
                algorithms=["HS256"],
            )
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {e}")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")
    return user_id


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


class RepurposeResponse(BaseModel):
    status: str
    processed_video_url: str
    transcript: str
    highlight: HighlightSegment


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
    """Whisper via the OpenAI API — no local model, no GPU/RAM needed on
    the server itself."""
    with open(path, "rb") as f:
        try:
            result = ai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
    return result.model_dump() if hasattr(result, "model_dump") else dict(result)


def _find_best_highlight(transcript_text: str, max_duration: float) -> HighlightSegment:
    prompt = (
        "Analyze this video transcript and identify the single most "
        "engaging, viral-worthy clip segment, no longer than "
        f"{MAX_OUTPUT_CLIP_SECONDS} seconds. Return ONLY a raw JSON object "
        "with keys: start_time (seconds), end_time (seconds), reason, "
        "suggested_title.\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )
    try:
        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert short-form content editor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)
        seg = HighlightSegment(**data)
    except Exception:
        seg = HighlightSegment(
            start_time=0.0,
            end_time=min(30.0, max_duration),
            reason="Default clip (AI highlight detection unavailable)",
            suggested_title="Featured Clip",
        )

    # Clamp to the source video's real bounds and our max clip length —
    # never trust the model's numbers blindly.
    seg.start_time = max(0.0, min(seg.start_time, max_duration))
    seg.end_time = max(seg.start_time + 1, min(seg.end_time, max_duration))
    if seg.end_time - seg.start_time > MAX_OUTPUT_CLIP_SECONDS:
        seg.end_time = seg.start_time + MAX_OUTPUT_CLIP_SECONDS
    return seg


def _generate_srt(segments: list, srt_path: str, clip_start: float, clip_end: float):
    def fmt(seconds: float) -> str:
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(srt_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            s, e = seg.get("start", 0), seg.get("end", 0)
            if e < clip_start or s > clip_end:
                continue
            rel_start = max(0, s - clip_start)
            rel_end = max(rel_start + 0.3, e - clip_start)
            text = seg.get("text", "").strip().upper()
            if not text:
                continue
            f.write(f"{idx}\n{fmt(rel_start)} --> {fmt(rel_end)}\n{text}\n\n")
            idx += 1


def _render_clip(input_path: str, output_path: str, srt_path: Optional[str],
                  start: float, duration: float):
    filter_graph = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    if srt_path and os.path.exists(srt_path):
        safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
        style = "FontName=Arial-Bold,FontSize=24,PrimaryColour=&H00FFFF00,Outline=2,Bold=1,Alignment=2"
        filter_graph += f",subtitles='{safe_srt}':force_style='{style}'"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", input_path, "-t", str(duration),
        "-vf", filter_graph,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Video render failed: {result.stderr[-500:]}")


@router.post("/repurpose", response_model=RepurposeResponse)
async def repurpose_video(
    req: RepurposeRequest,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Accepts a Supabase Storage URL for a longer video, transcribes it,
    finds the single best highlight (capped at MAX_OUTPUT_CLIP_SECONDS),
    crops it to 9:16, burns in captions, and returns a public URL of
    the finished clip.

    Flutter uploads the raw video to Supabase Storage first, then calls
    this endpoint with just the URL — this keeps large file data off the
    Railway inbound proxy and avoids the broken-pipe timeout.
    """
    if supabase_admin is None:
        raise HTTPException(
            status_code=503,
            detail="Video repurposing isn't configured yet — SUPABASE_SERVICE_ROLE_KEY is missing on the server.",
        )

    _check_repurpose_rate_limit(user_id)

    with tempfile.TemporaryDirectory() as tmp:
        source_path = os.path.join(tmp, "source.mp4")

        # Download the video from Supabase Storage.
        # This is a server-to-server download (Railway → Supabase CDN)
        # and is not subject to Railway's inbound request timeout.
        try:
            urllib.request.urlretrieve(req.video_url, source_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not download video from URL: {e}")

        size = os.path.getsize(source_path)
        if size > MAX_INPUT_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Video too large — max {MAX_INPUT_SIZE_BYTES // (1024 * 1024)}MB.",
            )

        duration = _run_ffprobe_duration(source_path)
        if duration > MAX_INPUT_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Video too long — max {MAX_INPUT_DURATION_SECONDS}s.",
            )

        whisper_result = _transcribe_with_openai(source_path)
        transcript_text = whisper_result.get("text", "")
        segments = whisper_result.get("segments", [])

        highlight = _find_best_highlight(transcript_text, duration)
        clip_duration = highlight.end_time - highlight.start_time

        srt_path = os.path.join(tmp, "captions.srt")
        _generate_srt(segments, srt_path, highlight.start_time, highlight.end_time)

        output_path = os.path.join(tmp, "output.mp4")
        _render_clip(source_path, output_path, srt_path, highlight.start_time, clip_duration)

        # Upload the finished clip to Supabase Storage — survives Railway's
        # ephemeral filesystem across redeploys.
        storage_path = f"{user_id}/{int(time.time())}.mp4"
        with open(output_path, "rb") as f:
            try:
                supabase_admin.storage.from_(PROCESSED_BUCKET).upload(
                    storage_path, f, file_options={"content-type": "video/mp4"}
                )
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Storage upload failed: {e}")

        public_url = supabase_admin.storage.from_(PROCESSED_BUCKET).get_public_url(storage_path)

    return RepurposeResponse(
        status="success",
        processed_video_url=public_url,
        transcript=transcript_text,
        highlight=highlight,
    )
