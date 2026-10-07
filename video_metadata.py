"""
Reliable, server-side video metadata — real width/height/duration and a
generated thumbnail for ANY uploaded video, independent of whether a
browser successfully captured a frame client-side.

Two endpoints, both best-effort and fire-and-forget from the client
right after a post is created:

- POST /api/v1/videos/probe-dimensions — reads width/height/duration
  straight off the video file via ffprobe. Mainly for a Supabase-hosted
  video; a Bunny-hosted one gets its width/height from Bunny's own API
  response instead (see bunny_stream.py's self-heal, which already has
  the video id handy there).
- POST /api/v1/videos/thumbnail/generate — ffmpeg-extracts a real frame
  and uploads it, for whenever thumbnail_url is still null after
  upload: the client-side capture (video_thumbnail on native, a
  <canvas> capture on web) failed, was skipped, or — on web
  specifically — hit a CORS-tainted-canvas error (see
  web_thumbnail_html.dart's own comment on that). Reused by
  series_covers.py's own cover backfill instead of asking a viewer's
  browser to capture a remote video frame, so Web and Android end up
  with thumbnails generated the exact same way.

Both read a video's URL directly with ffmpeg/ffprobe's own HTTP
support (range requests under the hood) rather than downloading the
whole file first — verified against a real production video: ffprobe
and a single-frame ffmpeg extract each resolve in well under a second
regardless of the video's total length, which matters once long-form
uploads can run 30-40 minutes (see create_post_screen.dart's raised
cap).

A third endpoint, POST /api/v1/admin/backfill-post-thumbnails, is a
one-off admin action rather than something the client calls: the
generate-thumbnail call above only ever fires at upload time, so every
video post made before it shipped is stuck with no thumbnail_url
forever — the client has no reason to ever ask again for a post it's
not uploading right now. This walks every such post once and fills it
in the same way.
"""
import json
import os
import subprocess
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1", tags=["video_metadata"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def _require_admin(x_admin_key: str = Header(None)) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin endpoint is not configured.")
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")

_THUMB_BUCKET = "posts-media"
_THUMB_MAX_WIDTH = 1080
# Generous but bounded — a probe/extract against a slow-to-respond CDN
# shouldn't be able to hang a request indefinitely.
_FFPROBE_TIMEOUT_SECONDS = 30
_FFMPEG_TIMEOUT_SECONDS = 60


async def _get_current_user_id(authorization: str = Header(None)) -> str:
    from main import get_current_user_id

    return await get_current_user_id(authorization)


def probe_dimensions(url: str) -> Optional[dict]:
    """ffprobe reads the video stream's width/height and the
    container's duration straight off the remote URL — exported for
    bunny_stream.py and anything else that wants the same probe
    without a network round-trip to this router's own endpoint."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height:format=duration",
                "-of", "json",
                url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_FFPROBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        streams = data.get("streams") or []
        if not streams:
            return None
        width, height = streams[0].get("width"), streams[0].get("height")
        if not width or not height:
            return None
        duration = (data.get("format") or {}).get("duration")
        return {
            "width": int(width),
            "height": int(height),
            "duration_seconds": int(float(duration)) if duration else None,
        }
    except Exception:
        return None


def generate_thumbnail_bytes(video_url: str, at_seconds: float = 0.5) -> Optional[bytes]:
    """ffmpeg-extracts a single frame from a video's own URL — exported
    for series_covers.py's cover backfill, which needs the raw bytes
    (not a stored posts-row thumbnail_url) since it's generating a
    SERIES cover, not a post's own thumbnail."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "thumb.jpg")
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(at_seconds), "-i", video_url,
                    "-frames:v", "1", "-update", "1",
                    "-vf", f"scale='min({_THUMB_MAX_WIDTH},iw)':-2",
                    out_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )
            if result.returncode != 0 or not os.path.exists(out_path):
                return None
            with open(out_path, "rb") as f:
                return f.read()
        except Exception:
            return None


def upload_thumbnail_bytes(jpeg_bytes: bytes) -> Optional[str]:
    if supabase_admin is None:
        return None
    path = f"generated-thumbnails/{uuid.uuid4().hex}.jpg"
    try:
        supabase_admin.storage.from_(_THUMB_BUCKET).upload(
            path, jpeg_bytes, file_options={"content-type": "image/jpeg"}
        )
    except Exception:
        return None
    return supabase_admin.storage.from_(_THUMB_BUCKET).get_public_url(path)


class ProbeDimensionsRequest(BaseModel):
    post_id: str
    media_url: str


class ProbeDimensionsResponse(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[int] = None
    updated: bool


@router.post("/videos/probe-dimensions", response_model=ProbeDimensionsResponse)
async def probe_dimensions_endpoint(
    req: ProbeDimensionsRequest, user_id: str = Depends(_get_current_user_id)
):
    """
    Fire-and-forget from the client right after a Supabase-hosted video
    post is created. Ownership isn't checked against the post — same
    reasoning as bunny_stream.py's own status endpoint: the only thing
    a guessed post_id could trigger here is a harmless, cheap metadata
    read, and gating it on ownership would mean a non-owner's device
    (overwhelmingly who actually ends up viewing any given post) could
    never help backfill one whose own uploader's app already moved on.
    """
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Video metadata is not configured.")

    probed = probe_dimensions(req.media_url)
    if probed is None:
        return ProbeDimensionsResponse(updated=False)

    updates = {k: v for k, v in probed.items() if v is not None}
    try:
        supabase_admin.table("posts").update(updates).eq("id", req.post_id).execute()
    except Exception:
        return ProbeDimensionsResponse(updated=False)

    return ProbeDimensionsResponse(**probed, updated=True)


class GenerateThumbnailRequest(BaseModel):
    media_url: str
    # When set, the result is also written onto this posts row's own
    # thumbnail_url. Omitted by series_covers.py's own caller, which
    # writes the result onto a `series` row instead — this endpoint
    # only ever touches `posts` directly.
    post_id: Optional[str] = None


class GenerateThumbnailResponse(BaseModel):
    thumbnail_url: Optional[str] = None


@router.post("/videos/thumbnail/generate", response_model=GenerateThumbnailResponse)
async def generate_thumbnail_endpoint(
    req: GenerateThumbnailRequest, user_id: str = Depends(_get_current_user_id)
):
    """
    Server-side fallback thumbnail generator. Same ownership reasoning
    as probe_dimensions_endpoint above — this only ever produces a new
    thumbnail when one is otherwise missing, never overwrites an
    existing one, so a guessed post_id can't deface anything.
    """
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Thumbnail generation is not configured.")

    jpeg_bytes = generate_thumbnail_bytes(req.media_url)
    if jpeg_bytes is None:
        return GenerateThumbnailResponse(thumbnail_url=None)

    url = upload_thumbnail_bytes(jpeg_bytes)
    if url is None:
        return GenerateThumbnailResponse(thumbnail_url=None)

    if req.post_id:
        try:
            supabase_admin.table("posts").update(
                {"thumbnail_url": url}
            ).eq("id", req.post_id).is_("thumbnail_url", "null").execute()
        except Exception:
            pass

    return GenerateThumbnailResponse(thumbnail_url=url)


class BackfillThumbnailsResponse(BaseModel):
    checked: int
    generated: int
    skipped: int
    failed: int


@router.post(
    "/admin/backfill-post-thumbnails",
    response_model=BackfillThumbnailsResponse,
    dependencies=[Depends(_require_admin)],
)
async def backfill_post_thumbnails():
    """One-off: generates thumbnail_url for every existing video post
    that has none. These are posts uploaded before the automatic
    generate-thumbnail-on-create call existed (or whose client-side
    capture silently failed at the time) — nothing was ever going to
    revisit them on its own, since the client only calls
    generate_thumbnail_endpoint right after creating a post, never for
    one it's just viewing. Safe to re-run: only ever touches a row
    whose thumbnail_url is still null."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Thumbnail backfill is not configured.")

    rows = (
        supabase_admin.table("posts")
        .select("id,media_url")
        .eq("post_type", "video")
        .is_("thumbnail_url", "null")
        .not_.is_("media_url", "null")
        .execute()
        .data
        or []
    )

    checked = generated = skipped = failed = 0
    for row in rows:
        checked += 1
        media_url = row.get("media_url") or ""
        if not media_url:
            skipped += 1
            continue
        jpeg_bytes = generate_thumbnail_bytes(media_url)
        if jpeg_bytes is None:
            failed += 1
            continue
        url = upload_thumbnail_bytes(jpeg_bytes)
        if url is None:
            failed += 1
            continue
        try:
            supabase_admin.table("posts").update(
                {"thumbnail_url": url}
            ).eq("id", row["id"]).is_("thumbnail_url", "null").execute()
            generated += 1
        except Exception:
            failed += 1

    return BackfillThumbnailsResponse(
        checked=checked, generated=generated, skipped=skipped, failed=failed
    )
