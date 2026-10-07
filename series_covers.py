"""
Backfilling a series' missing cover image, on behalf of any viewer.

series_service.dart's SeriesService.backfillCoverIfMissing captures a
video frame in the viewer's own browser (web-only — see
web_thumbnail_html.dart) and tries to save it as the series' poster art
whenever a series is shown with no cover yet. The `series` table's own
RLS update policy is owner-only (`auth.uid() = user_id`), so that write
only ever actually lands when the viewer happens to be the series'
creator — profile screens, the upload flow's series picker. Everywhere
else a coverless series is actually seen (the Dramas tab, Discover),
the viewer is a stranger browsing someone else's content, and the RLS
write silently no-ops, leaving the placeholder tile in place forever.

Routed through here with the service-role client so the fix isn't just
"open up RLS" (which would let anyone deface anyone's series art):
  - only fills a currently-NULL cover, never overwrites one that exists
  - only accepts a URL that either IS this series' own earliest
    episode's thumbnail_url already on file, or points into this
    project's own Supabase Storage (where the client just uploaded the
    frame it captured, via the same moderated upload path every other
    post's media goes through) — never an arbitrary external image URL.
    That doesn't prove the uploaded bytes are really a frame of this
    episode's video, but it rules out pointing a series at someone
    else's asset or an off-platform image, and it only ever fires once
    per series while the cover is still empty.
"""
import io
import os
import uuid
from typing import Optional
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Depends, Header, HTTPException
from PIL import Image
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1", tags=["series"])

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


async def _get_current_user_id(authorization: str = Header(None)) -> str:
    from main import get_current_user_id

    return await get_current_user_id(authorization)


_SUPABASE_HOST = urlparse(SUPABASE_URL).netloc if SUPABASE_URL else ""

# Where a compressed cover lands, and the marker that tells the backfill
# endpoint below a cover is already our own small JPEG rather than
# someone's original multi-megabyte upload.
_COVER_BUCKET = "posts-media"
_COVER_STORAGE_PREFIX = "series-covers/"
_COVER_MAX_DIMENSION = 960
_COVER_JPEG_QUALITY = 82


def _compress_cover_image(url: str) -> Optional[bytes]:
    """Downloads a cover image and re-encodes it as a small JPEG sized
    for a poster tile. The raw art a creator picks (a phone photo, an
    AI-generated PNG) routinely comes in at 2-3MB for well under 2
    megapixels — PNG's lossless encoding doesn't compress AI-generated
    detail/noise well, and nothing before this downsamples or
    re-encodes it. On a slow connection a file that size just times
    out mid-download, which CachedNetworkImage can't tell apart from a
    genuinely broken image — it falls back to the placeholder tile
    exactly as if the cover were missing. Returns None on any failure
    (bad URL, decode error, timeout) so the caller can fall back to
    storing the original URL rather than blocking the whole cover-set
    on an image-processing hiccup.
    """
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img = img.convert("RGB")  # a poster tile is opaque — alpha buys nothing here
        if max(img.size) > _COVER_MAX_DIMENSION:
            img.thumbnail((_COVER_MAX_DIMENSION, _COVER_MAX_DIMENSION), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=_COVER_JPEG_QUALITY, optimize=True)
        return out.getvalue()
    except Exception:
        return None


def _store_compressed_cover(jpeg_bytes: bytes) -> Optional[str]:
    path = f"{_COVER_STORAGE_PREFIX}{uuid.uuid4().hex}.jpg"
    try:
        supabase_admin.storage.from_(_COVER_BUCKET).upload(
            path, jpeg_bytes, file_options={"content-type": "image/jpeg"}
        )
    except Exception:
        return None
    return supabase_admin.storage.from_(_COVER_BUCKET).get_public_url(path)


def _is_own_supabase_storage_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and bool(_SUPABASE_HOST)
        and parsed.netloc == _SUPABASE_HOST
        and "/storage/v1/object/" in parsed.path
    )


class SetSeriesCoverRequest(BaseModel):
    cover_image_url: str


class SetSeriesCoverResponse(BaseModel):
    updated: bool


@router.post("/series/{series_id}/cover", response_model=SetSeriesCoverResponse)
async def set_series_cover(
    series_id: str,
    body: SetSeriesCoverRequest,
    user_id: str = Depends(_get_current_user_id),
):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Series cover backfill is not configured.")

    series_result = (
        supabase_admin.table("series")
        .select("id,cover_image_url")
        .eq("id", series_id)
        .limit(1)
        .execute()
    )
    series_rows = series_result.data or []
    if not series_rows:
        raise HTTPException(status_code=404, detail="Series not found.")
    if series_rows[0].get("cover_image_url"):
        # Already has a cover (set by the owner, or a backfill that won
        # the race) — never clobber it.
        return SetSeriesCoverResponse(updated=False)

    episode_result = (
        supabase_admin.table("posts")
        .select("thumbnail_url")
        .eq("series_id", series_id)
        .order("episode_number", desc=False)
        .limit(1)
        .execute()
    )
    episode_rows = episode_result.data or []
    if not episode_rows:
        raise HTTPException(status_code=404, detail="Series has no episodes yet.")

    is_own_episode_thumb = body.cover_image_url == episode_rows[0].get("thumbnail_url")
    is_own_storage_upload = _is_own_supabase_storage_url(body.cover_image_url)
    if not is_own_episode_thumb and not is_own_storage_upload:
        raise HTTPException(
            status_code=400,
            detail="cover_image_url must be this series' own episode thumbnail or a Viyo-hosted upload.",
        )

    # Re-compress before storing — see _compress_cover_image's own
    # docstring for why the raw upload can't just be trusted as-is.
    # Best-effort: a compression failure stores the original URL
    # rather than failing the whole cover-set over it.
    stored_url = body.cover_image_url
    compressed = _compress_cover_image(body.cover_image_url)
    if compressed is not None:
        new_url = _store_compressed_cover(compressed)
        if new_url is not None:
            stored_url = new_url

    supabase_admin.table("series").update(
        {"cover_image_url": stored_url}
    ).eq("id", series_id).is_("cover_image_url", "null").execute()

    return SetSeriesCoverResponse(updated=True)


class BackfillCoversResponse(BaseModel):
    checked: int
    compressed: int
    skipped: int
    failed: int


@router.post(
    "/admin/backfill-series-covers",
    response_model=BackfillCoversResponse,
    dependencies=[Depends(_require_admin)],
)
async def backfill_series_covers():
    """One-off: re-compresses every series cover already in the
    database down to a small JPEG, same as set_series_cover above now
    does automatically for every NEW cover. Needed because that
    automatic compression only covers covers set after it shipped —
    a series whose cover was set earlier can still point at someone's
    original 2-3MB upload, which is exactly the kind of file that
    times out and never loads on a slow connection. Skips anything
    already under series-covers/ (our own compressed output) so
    re-running this is always safe and cheap."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Series cover backfill is not configured.")

    rows = (
        supabase_admin.table("series")
        .select("id,cover_image_url")
        .not_.is_("cover_image_url", "null")
        .execute()
        .data
        or []
    )

    checked = compressed_count = skipped = failed = 0
    for row in rows:
        checked += 1
        url = row.get("cover_image_url") or ""
        if _COVER_STORAGE_PREFIX in url:
            skipped += 1
            continue
        compressed = _compress_cover_image(url)
        if compressed is None:
            failed += 1
            continue
        new_url = _store_compressed_cover(compressed)
        if new_url is None:
            failed += 1
            continue
        try:
            supabase_admin.table("series").update({"cover_image_url": new_url}).eq("id", row["id"]).execute()
            compressed_count += 1
        except Exception:
            failed += 1

    return BackfillCoversResponse(checked=checked, compressed=compressed_count, skipped=skipped, failed=failed)
