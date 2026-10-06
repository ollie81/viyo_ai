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
from subscriptions import is_active_subscriber

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

# Discount applied to each episode's price when unlocking a whole
# series at once (see unlock_series_bundle) rather than one at a time —
# a flat, global percentage, same "no single source of truth across
# repos, mirrored in Flutter" tradeoff as FREE_EPISODE_COUNT/
# CREATOR_SHARE above and coins.py's FEATURE_COSTS.
BUNDLE_DISCOUNT = 0.20

# Movie/Short Film titles are a single uploaded video, not a
# multi-episode series — there's no "episode 4" to ever reach, so the
# free-episode carve-out below would make every one of them 100% free
# by default. Mirrored in Flutter (series.dart's
# kSingleAssetContentTypes).
SINGLE_ASSET_CONTENT_TYPES = {"movie", "short_film"}

# Fallback price for a series whose coin_price_per_episode is missing
# or 0 — a row created before that column existed, or before this app
# had per-series pricing at all. Without this, `int(... or 0)` below
# would silently unlock every locked episode of that series for free,
# since 0 coins always clears the balance check. Mirrored in Flutter
# (series.dart's kDefaultEpisodeCoinPrice) — no single source of truth
# across the two repos, same tradeoff as every other constant here.
DEFAULT_EPISODE_COIN_PRICE = 30


def _is_free_episode(content_type: Optional[str], episode_number: int) -> bool:
    """
    True for the first FREE_EPISODE_COUNT episodes of a normal
    multi-episode title (series/short_drama/ai_film, or None — every
    `series` row created before content_type existed). A Movie/Short
    Film never gets this carve-out: it's priced from the first watch,
    same as any already-unlocked episode past the free window.
    """
    if content_type in SINGLE_ASSET_CONTENT_TYPES:
        return False
    return episode_number <= FREE_EPISODE_COUNT


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
            .select("id,user_id,title,coin_price_per_episode,content_type")
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

    series = _get_series(episode["series_id"])
    if _is_free_episode(series.get("content_type"), episode_number):
        return UnlockEpisodeResponse(unlocked=True, coins_spent=0)

    # An active Viyo Premium subscriber skips the coin paywall entirely
    # — no charge, and (see subscriptions.py's own module docstring)
    # deliberately no episode_unlocks row and no creator_earnings
    # credit either, since there's no real per-episode money to split
    # from a flat subscription charge.
    if is_active_subscriber(supabase_admin, user_id):
        return UnlockEpisodeResponse(unlocked=True, coins_spent=0)

    price = int(series.get("coin_price_per_episode") or DEFAULT_EPISODE_COIN_PRICE)

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


class UnlockBundleResponse(BaseModel):
    unlocked_episode_ids: list[str]
    coins_spent: int
    already_complete: bool = False


@router.post("/series/{series_id}/unlock-bundle", response_model=UnlockBundleResponse)
async def unlock_series_bundle(series_id: str, user_id: str = Depends(_get_current_user_id_no_guest)):
    """
    Unlocks every currently-locked episode of a series in one purchase,
    at BUNDLE_DISCOUNT off the per-episode price — for a binge-watcher
    who'd rather pay once than tap "unlock" on every episode. Does NOT
    just loop unlock_episode N times: that would charge full price N
    times (no way to apply a discount) and multiply this endpoint's own
    known partial-failure surface (a lost creator_earnings credit) by N
    per purchase instead of handling it once, batched.
    """
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Episode unlocking is not configured.")

    series = _get_series(series_id)
    price = int(series.get("coin_price_per_episode") or DEFAULT_EPISODE_COIN_PRICE)
    series_title = series.get("title") or "this series"

    try:
        episodes = (
            supabase_admin.table("posts")
            .select("id,episode_number")
            .eq("series_id", series_id)
            .order("episode_number", desc=False)
            .execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load episodes: {e}")

    # The owner never needs to unlock their own series — same free pass
    # unlock_episode gives them per-episode, applied here up front.
    content_type = series.get("content_type")
    if series["user_id"] == user_id:
        lockable_ids = [ep["id"] for ep in episodes if not _is_free_episode(content_type, ep.get("episode_number") or 1)]
        return UnlockBundleResponse(unlocked_episode_ids=lockable_ids, coins_spent=0, already_complete=True)

    lockable = [ep for ep in episodes if not _is_free_episode(content_type, ep.get("episode_number") or 1)]
    if not lockable:
        return UnlockBundleResponse(unlocked_episode_ids=[], coins_spent=0, already_complete=True)

    # Same subscriber bypass as unlock_episode above.
    if is_active_subscriber(supabase_admin, user_id):
        lockable_ids = [ep["id"] for ep in lockable]
        return UnlockBundleResponse(unlocked_episode_ids=lockable_ids, coins_spent=0, already_complete=True)

    lockable_ids = [ep["id"] for ep in lockable]
    try:
        already = (
            supabase_admin.table("episode_unlocks")
            .select("post_id")
            .eq("user_id", user_id)
            .in_("post_id", lockable_ids)
            .execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check existing unlocks: {e}")
    already_unlocked_ids = {row["post_id"] for row in already}

    to_unlock = [ep for ep in lockable if ep["id"] not in already_unlocked_ids]
    if not to_unlock:
        return UnlockBundleResponse(unlocked_episode_ids=lockable_ids, coins_spent=0, already_complete=True)

    per_episode_price = round(price * (1 - BUNDLE_DISCOUNT))

    # Insert each unlock row before moving any coins, same idempotency-
    # first ordering as unlock_episode above — looped rather than one
    # batch insert so a single duplicate (a race against a concurrent
    # per-episode unlock of the same post) only drops that one row
    # instead of failing the whole purchase.
    claimed_ids: list[str] = []
    for ep in to_unlock:
        try:
            supabase_admin.table("episode_unlocks").insert({
                "user_id": user_id,
                "post_id": ep["id"],
                "coins_spent": per_episode_price,
            }).execute()
            claimed_ids.append(ep["id"])
        except Exception as e:
            if "23505" in str(e) or "duplicate" in str(e).lower():
                continue
            raise HTTPException(status_code=500, detail=f"Could not unlock episode {ep['id']}: {e}")

    if not claimed_ids:
        # Every episode got claimed by something else between the
        # already-unlocked check above and these inserts — nothing left
        # to charge for.
        return UnlockBundleResponse(unlocked_episode_ids=lockable_ids, coins_spent=0, already_complete=True)

    total_cost = per_episode_price * len(claimed_ids)

    try:
        debit_coins(
            supabase_admin, user_id, total_cost, "series_bundle_unlock",
            f"Unlocked {len(claimed_ids)} episodes of {series_title} (bundle, {int(BUNDLE_DISCOUNT * 100)}% off)",
        )
    except HTTPException:
        try:
            supabase_admin.table("episode_unlocks").delete().eq("user_id", user_id).in_("post_id", claimed_ids).execute()
        except Exception:
            pass
        raise

    creator_share = round(per_episode_price * CREATOR_SHARE)
    creator_id = series["user_id"]
    for post_id in claimed_ids:
        for attempt in range(2):
            try:
                supabase_admin.table("creator_earnings").insert({
                    "creator_id": creator_id,
                    "source_post_id": post_id,
                    "coins": creator_share,
                }).execute()
                break
            except Exception as e:
                if attempt == 1:
                    print(
                        f"[CRITICAL] Could not credit {creator_share} coins to creator "
                        f"{creator_id} for bundle-unlocked episode {post_id} by {user_id}: {e}"
                    )

    return UnlockBundleResponse(unlocked_episode_ids=lockable_ids, coins_spent=total_cost, already_complete=False)
