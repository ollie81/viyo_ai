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
import os
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1", tags=["series"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def _get_current_user_id(authorization: str = Header(None)) -> str:
    from main import get_current_user_id

    return await get_current_user_id(authorization)


_SUPABASE_HOST = urlparse(SUPABASE_URL).netloc if SUPABASE_URL else ""


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

    supabase_admin.table("series").update(
        {"cover_image_url": body.cover_image_url}
    ).eq("id", series_id).is_("cover_image_url", "null").execute()

    return SetSeriesCoverResponse(updated=True)
