"""
Discover Spotlight — pay coins for temporary priority placement in the
suggested-creators list (Flutter's ProfileService.getSuggestedCreators).

There's no `is_spotlighted` (or similar) column on `profiles`, and no
schema/migration access from this codebase to add one — so "currently
spotlighted" is derived from the existing `transactions` ledger instead:
a `type == "spotlight"` row within the last SPOTLIGHT_WINDOW_HOURS
counts as active, giving it a natural self-expiring window with no
cron job or manual unset needed. coins.py's spend_on_feature logs that
row automatically (type is the feature key — see coins._log_spend).

Reading "who is spotlighted right now" needs the service-role client
(same reasoning as leaderboard.py): it means reading every user's
transactions, which RLS on this personal wallet ledger blocks for a
normal user-scoped client.
"""
import datetime
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

from coins import spend_on_feature

router = APIRouter(prefix="/api/v1", tags=["discover"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


async def _get_current_user_id(authorization: str = Header(None)) -> str:
    # Guests can still see who's spotlighted — a free, passive read, same
    # posture as the weekly leaderboard.
    from main import get_current_user_id

    return await get_current_user_id(authorization)


SPOTLIGHT_WINDOW_HOURS = 24


class SpotlightResponse(BaseModel):
    active_until: str  # ISO timestamp, informational for the client


class SpotlightedIdsResponse(BaseModel):
    user_ids: list[str]  # most-recently-spotlighted first


def _window_start() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=SPOTLIGHT_WINDOW_HOURS)


@router.post("/spotlight", response_model=SpotlightResponse)
async def spotlight_me(user_id: str = Depends(_get_current_user_id_no_guest)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Spotlight service is not configured.")

    try:
        existing = (
            supabase_admin
            .table("transactions")
            .select("created_at")
            .eq("user_id", user_id)
            .eq("type", "spotlight")
            .gte("created_at", _window_start().isoformat())
            .limit(1)
            .execute()
        )
        if existing.data:
            raise HTTPException(
                status_code=400,
                detail="You're already spotlighted — this refreshes once the current one expires.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # best-effort dedupe check — worst case is one avoidable re-purchase, not a broken flow

    spend_on_feature(supabase_admin, user_id, "spotlight")

    active_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=SPOTLIGHT_WINDOW_HOURS)
    return SpotlightResponse(active_until=active_until.isoformat())


@router.get("/spotlight/active", response_model=SpotlightedIdsResponse)
async def get_active_spotlights(user_id: str = Depends(_get_current_user_id)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Spotlight service is not configured.")

    try:
        result = (
            supabase_admin
            .table("transactions")
            .select("user_id,created_at")
            .eq("type", "spotlight")
            .gte("created_at", _window_start().isoformat())
            .order("created_at", desc=True)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load spotlights: {e}")

    seen = set()
    ordered_ids = []
    for r in rows:
        uid = r.get("user_id")
        if uid and uid not in seen:
            seen.add(uid)
            ordered_ids.append(uid)

    return SpotlightedIdsResponse(user_ids=ordered_ids)
