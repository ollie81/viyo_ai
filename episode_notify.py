"""
New-episode notifications — fanning out when a creator publishes
another episode of a series, to the union of two audiences: everyone
following that series specifically (series_follows, a Phase 2 table —
following just one show without following everything else the creator
makes), and everyone following the creator generally (the existing
`follows` table — someone who follows the creator broadly still wants
to know about a new series episode even if they haven't explicitly
followed this particular show). Deduped so a viewer who follows both
only gets one notification.

Episodes are published by a direct Supabase insert from Flutter
(PostService.createPost), never through this backend, so there's no
publish hook to hang this on server-side. Instead the client calls this
endpoint itself right after that insert succeeds (see
upload_ai_drama_screen.dart) — best-effort and fire-and-forget, same as
series_covers.py's cover backfill: a dropped notification call should
never fail or delay the upload the creator is actually waiting on.

This is the first one-to-many notification in the app — every existing
trigger (like/comment in interactions.py, gift in gifting.py, follow in
push.py) notifies exactly one recipient. `notifications.type` is a free
string with no server-side enum, so "new_episode" needs no schema
change; neither does reading `follows`/`series_follows`, both mirrored
from Flutter's own column names since this backend has never touched
either table before. A missing `series_follows` (migration not run
yet) degrades to creator-followers only, not a failure.
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from supabase import create_client, Client

from push import send_push_to_user

router = APIRouter(prefix="/api/v1", tags=["episodes"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


@router.post("/episodes/{post_id}/notify-followers")
async def notify_new_episode(
    post_id: str, user_id: str = Depends(_get_current_user_id_no_guest)
):
    if supabase_admin is None:
        return {"notified": 0}

    try:
        post_result = (
            supabase_admin.table("posts")
            .select("id,user_id,series_id,episode_number")
            .eq("id", post_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load episode: {e}")

    rows = post_result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Episode not found.")
    post = rows[0]

    # Only the episode's own owner may trigger a fan-out to their
    # followers — otherwise any authenticated client could spam another
    # creator's followers by passing an arbitrary post_id.
    if post["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not your episode.")

    series_id = post.get("series_id")
    if not series_id:
        return {"notified": 0}

    series_title = "a series"
    try:
        series_result = (
            supabase_admin.table("series")
            .select("title")
            .eq("id", series_id)
            .limit(1)
            .execute()
        )
        if series_result.data:
            series_title = series_result.data[0].get("title") or series_title
    except Exception:
        pass  # a missing title just falls back to the generic phrase above

    actor_name = "A creator"
    try:
        profile_result = (
            supabase_admin.table("profiles")
            .select("display_name,username")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        if profile_result.data:
            profile = profile_result.data[0]
            actor_name = profile.get("display_name") or profile.get("username") or actor_name
    except Exception:
        pass

    follower_ids: set[str] = set()

    try:
        creator_follows = (
            supabase_admin.table("follows")
            .select("follower_id")
            .eq("following_id", user_id)
            .execute()
        )
        follower_ids.update(
            r["follower_id"] for r in (creator_follows.data or []) if r.get("follower_id")
        )
    except Exception as e:
        print(f"[WARN] Could not load creator followers for new-episode notify: {e}")

    try:
        series_follows = (
            supabase_admin.table("series_follows")
            .select("follower_id")
            .eq("series_id", series_id)
            .execute()
        )
        follower_ids.update(
            r["follower_id"] for r in (series_follows.data or []) if r.get("follower_id")
        )
    except Exception as e:
        # Expected until the Phase 2 migration runs (series_follows
        # doesn't exist yet) — falls back to creator-followers only,
        # not a failure.
        print(f"[INFO] series_follows not available for new-episode notify: {e}")

    if not follower_ids:
        return {"notified": 0}

    episode_number = post.get("episode_number") or 1
    message = f"{actor_name} posted Episode {episode_number} of {series_title}"

    # One batch insert for every recipient's in-app row, not a
    # per-follower loop — this is the one place in the codebase where a
    # single event fans out to many rows, so it's worth not making it N
    # round trips too.
    try:
        supabase_admin.table("notifications").insert([
            {
                "user_id": follower_id,
                "actor_id": user_id,
                "type": "new_episode",
                "message": message,
            }
            for follower_id in follower_ids
        ]).execute()
    except Exception as e:
        print(f"[WARN] Could not insert new-episode notifications: {e}")

    # Push still has to be per-recipient — send_push_to_user looks up
    # that one user's own device_tokens — but it's already best-effort
    # and silent on failure, same as every other push trigger.
    for follower_id in follower_ids:
        send_push_to_user(
            supabase_admin,
            follower_id,
            "New episode",
            message,
            {"type": "new_episode", "series_id": series_id, "post_id": post_id},
        )

    return {"notified": len(follower_ids)}
