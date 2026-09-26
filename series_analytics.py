"""
Creator-facing analytics for a series — views, completion rate,
episode-by-episode drop-off, follows, and saves.

Genuinely infeasible until Phase 2 added `watch_progress`: before that,
this backend had no signal at all for "how much of an episode did a
viewer actually watch" — only `posts.view_count`, a one-shot counter
with no notion of completion. Every completion/drop-off number below
comes from real `watch_progress` rows (Flutter's WatchProgressService,
synced there), not a fabricated estimate — a series/episode with no
watch_progress rows yet just reports 0%, never a guessed number.

Same aggregate-raw-rows-in-Python pattern as analytics.py's admin
summary — no stored procs/DB-side aggregation available in this
codebase, so everything pulled here gets summed/averaged in Python.
Unlike analytics.py (admin-only, gated by a shared key), this is
scoped to one caller's own series — gated by ownership, not an admin
role, the same "content owner" line drawn throughout this codebase
(pin_comment, series.status).
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1", tags=["series_analytics"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# A watched fraction at or above this counts as "completed" for the
# completion-rate stat — matches WatchProgressService's own ceiling for
# when a position is "basically finished" rather than still in progress.
_COMPLETION_THRESHOLD = 0.9


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


class EpisodeAnalytics(BaseModel):
    post_id: str
    episode_number: int
    view_count: int
    like_count: int
    comment_count: int
    # Distinct viewers who have a watch_progress row for this episode —
    # a real (if partial, since only synced episodes count) measure of
    # who actually pressed play, as opposed to view_count's one-shot
    # per-screen-open increment.
    unique_viewers: int
    # Share of unique_viewers whose furthest position reached
    # _COMPLETION_THRESHOLD of the episode's duration. None (not 0)
    # when there's no watch_progress data yet, so the client can show
    # "not enough data" instead of a misleading 0%.
    completion_rate: Optional[float] = None
    average_watched_fraction: Optional[float] = None


class SeriesAnalyticsResponse(BaseModel):
    series_id: str
    total_views: int
    total_unique_viewers: int
    follower_count: int
    watchlist_count: int
    episodes: list[EpisodeAnalytics]


@router.get("/series/{series_id}/analytics", response_model=SeriesAnalyticsResponse)
async def get_series_analytics(
    series_id: str, user_id: str = Depends(_get_current_user_id_no_guest)
):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Analytics service is not configured.")

    try:
        series_result = (
            supabase_admin.table("series").select("id,user_id").eq("id", series_id).limit(1).execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load series: {e}")

    series_rows = series_result.data or []
    if not series_rows:
        raise HTTPException(status_code=404, detail="Series not found.")
    if series_rows[0]["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Only the series' own creator can view its analytics.")

    try:
        episodes = (
            supabase_admin.table("posts")
            .select("id,episode_number,view_count,like_count,comment_count")
            .eq("series_id", series_id)
            .order("episode_number", ascending=True)
            .execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load episodes: {e}")

    post_ids = [ep["id"] for ep in episodes]

    # Best-effort — a Phase 2 table that isn't migrated yet (or a
    # transient error) degrades to "no watch data" rather than failing
    # the whole analytics page, since view/like/comment/follow/save
    # counts below are all still real and worth showing on their own.
    progress_rows: list[dict] = []
    if post_ids:
        try:
            progress_rows = (
                supabase_admin.table("watch_progress")
                .select("post_id,user_id,position_ms,duration_ms")
                .in_("post_id", post_ids)
                .execute()
            ).data or []
        except Exception:
            pass

    follower_count = 0
    try:
        follower_count = (
            supabase_admin.table("series_follows")
            .select("id", count="exact")
            .eq("series_id", series_id)
            .limit(1)
            .execute()
        ).count or 0
    except Exception:
        pass

    watchlist_count = 0
    try:
        watchlist_count = (
            supabase_admin.table("watchlist")
            .select("id", count="exact")
            .eq("target_type", "series")
            .eq("target_id", series_id)
            .limit(1)
            .execute()
        ).count or 0
    except Exception:
        pass

    # Group watch_progress rows by post, deduping to one (best) row per
    # (post_id, user_id) — a viewer can have multiple synced rows over
    # time in theory, but the table's own unique constraint already
    # keeps it to one per pair, so this is just organizing, not
    # deduping duplicates that shouldn't exist.
    by_post: dict[str, list[dict]] = {}
    for row in progress_rows:
        by_post.setdefault(row["post_id"], []).append(row)

    episode_stats = []
    total_unique_viewers = 0
    for ep in episodes:
        rows = by_post.get(ep["id"], [])
        fractions = [
            (r["position_ms"] / r["duration_ms"])
            for r in rows
            if r.get("duration_ms") and r["duration_ms"] > 0
        ]
        unique_viewers = len(fractions)
        total_unique_viewers += unique_viewers

        completion_rate = None
        average_fraction = None
        if fractions:
            completed = sum(1 for f in fractions if f >= _COMPLETION_THRESHOLD)
            completion_rate = round(completed / len(fractions) * 100, 1)
            average_fraction = round(sum(fractions) / len(fractions) * 100, 1)

        episode_stats.append(EpisodeAnalytics(
            post_id=ep["id"],
            episode_number=ep.get("episode_number") or 0,
            view_count=ep.get("view_count") or 0,
            like_count=ep.get("like_count") or 0,
            comment_count=ep.get("comment_count") or 0,
            unique_viewers=unique_viewers,
            completion_rate=completion_rate,
            average_watched_fraction=average_fraction,
        ))

    return SeriesAnalyticsResponse(
        series_id=series_id,
        total_views=sum(ep.get("view_count") or 0 for ep in episodes),
        total_unique_viewers=total_unique_viewers,
        follower_count=follower_count,
        watchlist_count=watchlist_count,
        episodes=episode_stats,
    )
