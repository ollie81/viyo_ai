"""
Unlocking a locked AI Short Drama episode with Viyo Coins.

An episode is just a `posts` row with `series_id` + `episode_number`
set (see the add_ai_drama_series.sql migration) — this endpoint is the
one place unlocking one costs coins. The first FREE_EPISODE_COUNT
episodes of any series are free; from there, unlocking spends
`series.coin_price_per_episode` coins, split 65/35 between the
creator's earnings balance (creator_earnings) and the platform
(implicitly: the platform's share is simply never credited to anyone —
there's no platform "user" to credit it to).

Same "insert the idempotency row first, then move the money" ordering
already used by interactions.py's like_post: episode_unlocks has a
unique (user_id, post_id) constraint, so a double-tap or a retried
request after a dropped response can never charge someone twice for
the same episode.
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

from coins import debit_coins

router = APIRouter(prefix="/api/v1", tags=["episodes"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Episodes 1..FREE_EPISODE_COUNT of every series are free to watch;
# unlocking starts at episode FREE_EPISODE_COUNT + 1. Mirrored in the
# Flutter app (lib/models/series.dart) — no single source of truth
# across the two repos, same tradeoff as coins.py's FEATURE_COSTS.
FREE_EPISODE_COUNT = 3

# The creator's cut of every episode unlock. The remainder is the
# platform's — there's no platform "user" row to credit it to, so it's
# simply the part of the debit that never gets a matching credit into
# creator_earnings.
CREATOR_SHARE = 0.65


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


def _get_episode(post_id: str) -> dict:
    try:
        result = (
            supabase_admin.table("posts")
            .select("id,user_id,series_id,episode_number")
            .eq("id", post_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load episode: {e}")

    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Episode not found.")
    episode = rows[0]
    if not episode.get("series_id"):
        raise HTTPException(status_code=400, detail="This post isn't part of a series.")
    return episode


def _get_series(series_id: str) -> dict:
    try:
        result = (
            supabase_admin.table("series")
            .select("id,user_id,title,coin_price_per_episode")
            .eq("id", series_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load series: {e}")

    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Series not found.")
    return rows[0]


class UnlockEpisodeResponse(BaseModel):
    unlocked: bool
    coins_spent: int
    already_unlocked: bool = False


@router.post("/episodes/{post_id}/unlock", response_model=UnlockEpisodeResponse)
async def unlock_episode(post_id: str, user_id: str = Depends(_get_current_user_id_no_guest)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Episode unlocking is not configured.")

    episode = _get_episode(post_id)
    episode_number = episode.get("episode_number") or 1

    # The creator can always watch their own episode — no charge, no
    # unlock row needed (the paywall check on the client already skips
    # locking your own content; this is the server-side mirror of that).
    if episode["user_id"] == user_id:
        return UnlockEpisodeResponse(unlocked=True, coins_spent=0)

    if episode_number <= FREE_EPISODE_COUNT:
        return UnlockEpisodeResponse(unlocked=True, coins_spent=0)

    series = _get_series(episode["series_id"])
    price = int(series.get("coin_price_per_episode") or 0)

    # Insert the unlock row before moving any coins — the unique
    # constraint on (user_id, post_id) makes this the idempotency
    # check. A duplicate-key error means this viewer already unlocked
    # this exact episode (a double-tap, or a retried request whose
    # first response never arrived), so it's reported as already
    # unlocked rather than charged again.
    try:
        supabase_admin.table("episode_unlocks").insert({
            "user_id": user_id,
            "post_id": post_id,
            "coins_spent": price,
        }).execute()
    except Exception as e:
        if "23505" in str(e) or "duplicate" in str(e).lower():
            return UnlockEpisodeResponse(unlocked=True, coins_spent=0, already_unlocked=True)
        raise HTTPException(status_code=500, detail=f"Could not unlock episode: {e}")

    series_title = series.get("title") or "this series"

    # Raises 402/409/500 outward unchanged if the viewer can't be
    # debited — nothing else has happened yet except the unlock row
    # above, which is removed so a failed charge never leaves a "free"
    # unlock behind.
    try:
        debit_coins(
            supabase_admin, user_id, price, "episode_unlock",
            f"Unlocked {series_title} — Episode {episode_number}",
        )
    except HTTPException:
        try:
            supabase_admin.table("episode_unlocks").delete().eq("user_id", user_id).eq("post_id", post_id).execute()
        except Exception:
            pass
        raise

    creator_share = round(price * CREATOR_SHARE)

    # The viewer already has what they paid for at this point (the
    # unlock row exists) — a failure crediting the creator's earnings
    # must not undo that. Retried once, then logged loudly rather than
    # silently: unlike a notification or an analytics event, this is
    # money actually owed to the creator, so a lost credit here needs
    # to be visible for manual reconciliation, not swallowed the way a
    # best-effort push notification would be.
    for attempt in range(2):
        try:
            supabase_admin.table("creator_earnings").insert({
                "creator_id": episode["user_id"],
                "source_post_id": post_id,
                "coins": creator_share,
            }).execute()
            break
        except Exception as e:
            if attempt == 1:
                print(
                    f"[CRITICAL] Could not credit {creator_share} coins to creator "
                    f"{episode['user_id']} for episode unlock {post_id} by {user_id}: {e}"
                )

    return UnlockEpisodeResponse(unlocked=True, coins_spent=price)
