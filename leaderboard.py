"""
Weekly Viyo Coin leaderboard.

Ranks creators by coins *earned* (not spent) over the last 7 days.
Uses the service-role client deliberately — this needs to read other
users' transaction totals to build a ranking, which RLS on
`transactions` (a personal wallet ledger) would otherwise block for a
normal user-scoped client. No OpenAI call here at all, so it isn't part
of the coins.py gating system — this is a free, passive view, same as
checking any other leaderboard.
"""
import datetime
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1", tags=["leaderboard"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def _get_current_user_id(authorization: str = Header(None)) -> str:
    # Guests can view the leaderboard too — it's a free, passive read and
    # a natural hook toward creating a real account to start earning.
    from main import get_current_user_id

    return await get_current_user_id(authorization)


_LEADERBOARD_WINDOW_DAYS = 7
_LEADERBOARD_LOOKBACK_ROWS = 5000  # generous cap for a week of an active app's earn events
_LEADERBOARD_TOP_N = 20


class LeaderboardEntry(BaseModel):
    user_id: str
    username: str
    display_name: str
    avatar_url: Optional[str] = None
    coins_earned: int
    rank: int


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]
    # The requester's own rank/total this week, even if outside the top
    # list — both None if they haven't earned anything in the window.
    my_rank: Optional[int] = None
    my_coins_earned: Optional[int] = None
    window_days: int


@router.get("/leaderboard/weekly", response_model=LeaderboardResponse)
async def weekly_leaderboard(user_id: str = Depends(_get_current_user_id)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Leaderboard service is not configured.")

    window_start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=_LEADERBOARD_WINDOW_DAYS
    )

    try:
        result = (
            supabase_admin
            .table("transactions")
            .select("user_id,amount,created_at")
            .gt("amount", 0)
            .gte("created_at", window_start.isoformat())
            .limit(_LEADERBOARD_LOOKBACK_ROWS)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load leaderboard: {e}")

    totals: dict[str, int] = {}
    for r in rows:
        uid = r.get("user_id")
        if not uid:
            continue
        totals[uid] = totals.get(uid, 0) + int(r.get("amount") or 0)

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:_LEADERBOARD_TOP_N]

    profiles_by_id: dict[str, dict] = {}
    top_ids = [uid for uid, _ in top]
    if top_ids:
        try:
            profiles_result = (
                supabase_admin
                .table("profiles")
                .select("id,username,display_name,avatar_url")
                .in_("id", top_ids)
                .execute()
            )
            profiles_by_id = {p["id"]: p for p in (profiles_result.data or [])}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load profiles: {e}")

    entries = []
    for i, (uid, total) in enumerate(top):
        profile = profiles_by_id.get(uid)
        if not profile:
            continue
        entries.append(
            LeaderboardEntry(
                user_id=uid,
                username=profile.get("username") or "",
                display_name=profile.get("display_name") or "",
                avatar_url=profile.get("avatar_url"),
                coins_earned=total,
                rank=i + 1,
            )
        )

    my_rank = None
    my_coins_earned = totals.get(user_id)
    if my_coins_earned is not None:
        for i, (uid, _) in enumerate(ranked):
            if uid == user_id:
                my_rank = i + 1
                break

    return LeaderboardResponse(
        entries=entries,
        my_rank=my_rank,
        my_coins_earned=my_coins_earned,
        window_days=_LEADERBOARD_WINDOW_DAYS,
    )
