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
import tempfile
import subprocess
import urllib.parse
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
# Long-video limits. Override with environment variables if needed.

MAX_INPUT_DURATION_SECONDS = int(os.environ.get("MAX_INPUT_DURATION_SECONDS", "1980"))  # 33 minutes

MAX_INPUT_SIZE_BYTES = int(os.environ.get("MAX_INPUT_SIZE_MB", "500")) * 1024 * 1024

MAX_OUTPUT_CLIP_SECONDS = 60

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
    """Delegates to main.py's get_current_user_id (ES256 / JWKS verification).

    We use a lazy import instead of a module-level one to avoid a circular
    import — main.py imports repurpose at startup, so repurpose must not
    import main at module load time. Importing inside the function body is
    fine: it only runs at request time, by which point main is fully loaded.
    """
    from main import get_current_user_id
    return await get_current_user_id(authorization)


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


class RepurposeClipResult(BaseModel):
    processed_video_url: str
    highlight: HighlightSegment
    dead_air_removed_seconds: float = 0.0
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

            offset += chunk_duration

        return {
            "text": " ".join(all_text).strip(),
            "segments": all_segments,
        }

MAX_HIGHLIGHT_CANDIDATES = 3


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


def _find_highlights(
    transcript_text: str,
    max_duration: float,
    count: int = MAX_HIGHLIGHT_CANDIDATES,
) -> list:
    """
    Returns up to `count` ranked, non-overlapping highlight candidates
    instead of a single "best" pick — the point of multiple options is
    that they're genuinely different moments, so a candidate that mostly
    overlaps an already-accepted one is skipped rather than counted.
    """
    prompt = (
        f"Analyze this video transcript and identify the {count} most "
        "engaging, viral-worthy clip segments, each no longer than "
        f"{MAX_OUTPUT_CLIP_SECONDS} seconds, ranked best first. The segments "
        "should be genuinely different moments, not overlapping variations "
        "of the same one. Return ONLY a raw JSON array, each item an object "
        "with keys: start_time (seconds), end_time (seconds), reason, "
        "suggested_title, score (0-100, how engaging/viral-worthy this "
        "specific clip is on its own).\n\n"
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
        if not isinstance(data, list):
            data = [data]
        candidates = [HighlightSegment(**item) for item in data]
    except Exception:
        return [_fallback_highlight(max_duration)]

    accepted: list[HighlightSegment] = []
    for seg in candidates:
        # Clamp to the source video's real bounds and our max clip
        # length — never trust the model's numbers blindly.
        seg.start_time = max(0.0, min(seg.start_time, max_duration))
        seg.end_time = max(seg.start_time + 1, min(seg.end_time, max_duration))
        if seg.end_time - seg.start_time > MAX_OUTPUT_CLIP_SECONDS:
            seg.end_time = seg.start_time + MAX_OUTPUT_CLIP_SECONDS
        seg.score = max(0, min(seg.score, 100))

        if any(_overlap_fraction(seg, a) > 0.5 for a in accepted):
            continue
        accepted.append(seg)
        if len(accepted) >= count:
            break

    return accepted or [_fallback_highlight(max_duration)]


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

    post_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    if srt_path and os.path.exists(srt_path):
        safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
        style = "FontName=Arial-Bold,FontSize=24,PrimaryColour=&H00FFFF00,Outline=2,Bold=1,Alignment=2"
        post_filter += f",subtitles='{safe_srt}':force_style='{style}'"

    if len(rel_blocks) == 1:
        start, end = rel_blocks[0]
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_offset), "-i", input_path, "-t", str(end - start),
            "-vf", post_filter,
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

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_offset), "-i", input_path,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[vout]", "-map", "[acat]",
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
        _download_video(req.video_url, source_path)

        duration = _run_ffprobe_duration(source_path)
        if duration > MAX_INPUT_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Video too long — max {MAX_INPUT_DURATION_SECONDS}s.",
            )

        whisper_result = _transcribe_with_openai(source_path)
        transcript_text = whisper_result.get("text", "")
        segments = whisper_result.get("segments", [])

        # Transcription (the slow, expensive part) happens once no matter
        # how many candidate clips come out of it — only the render/upload
        # step below repeats per clip, which is cheap by comparison.
        highlights = _find_highlights(transcript_text, duration)

        clips = []
        for i, highlight in enumerate(highlights):
            # Cut dead air out of the chosen window instead of rendering
            # it as one continuous clip — this is what actually edits the
            # clip rather than just picking where to cut it.
            blocks = _find_speech_blocks(segments, highlight.start_time, highlight.end_time)
            dead_air_removed = _dead_air_removed_seconds(blocks)

            srt_path = os.path.join(tmp, f"captions_{i}.srt")
            _generate_srt(segments, srt_path, blocks)

            output_path = os.path.join(tmp, f"output_{i}.mp4")
            _render_clip(source_path, output_path, srt_path, blocks)

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
                quote_card_url=quote_card_url,
                thumbnail_url=thumbnail_url,
            ))

    return RepurposeResponse(
        status="success",
        transcript=transcript_text,
        clips=clips,
    )
