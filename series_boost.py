"""
Series-level promotion boost — a creator spends coins to feature their
whole drama series in Discover/Trending for a fixed window, distinct
from the existing per-episode boost_post/posts.is_boosted flag (which
only ever lifts the score of the one clip it's set on, and never
expires).

No schema/migration access from this codebase (same constraint noted
throughout — see discover.py's Spotlight and coins.py's own docstring),
so "currently boosted" can't be a new series.is_boosted/boosted_until
column. Instead this follows the exact pattern Discover Spotlight
already established: spend_on_feature's own transactions row IS the
state, self-expiring by just aging out of a rolling window query.

The one wrinkle Spotlight didn't have: Spotlight is a per-USER feature,
so the spend row's own user_id is enough to answer "is this creator
spotlighted". A boost is scoped to one SERIES, and a creator can own
more than one — so alongside the real spend_on_feature call (which
logs type="boost_series", the true financial record), this also writes
a zero-amount marker transaction keyed by
f"{_BOOST_TYPE_PREFIX}{series_id}", the same "amount: 0, keyed by a
per-thing type string" trick coins.py already uses for free-taste
tracking (type=f"free_taste_{feature}").
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

from coins import spend_on_feature, refund_feature

router = APIRouter(prefix="/api/v1", tags=["series_boost"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Longer than Spotlight's 24h — a series boost is the pricier, higher-
# commitment purchase (60 coins vs. Spotlight's 25), and lifts every
# episode in the series rather than one creator-level placement.
BOOST_WINDOW_HOURS = 48
_BOOST_TYPE_PREFIX = "boost_series:"


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


def _window_start() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=BOOST_WINDOW_HOURS)


def _boost_type(series_id: str) -> str:
    return f"{_BOOST_TYPE_PREFIX}{series_id}"


class BoostSeriesResponse(BaseModel):
    series_id: str
    active_until: str
    cost: int


class ActiveSeriesBoostsResponse(BaseModel):
    series_ids: list[str]


@router.post("/series/{series_id}/boost", response_model=BoostSeriesResponse)
async def boost_series(series_id: str, user_id: str = Depends(_get_current_user_id_no_guest)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Boost service is not configured.")

    try:
        series_result = (
            supabase_admin.table("series").select("id,user_id").eq("id", series_id).limit(1).execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load series: {e}")

    rows = series_result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Series not found.")
    if rows[0]["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Only the series' own creator can boost it.")

    boost_type = _boost_type(series_id)
    try:
        existing = (
            supabase_admin.table("transactions")
            .select("id")
            .eq("type", boost_type)
            .gte("created_at", _window_start().isoformat())
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check boost status: {e}")
    if existing.data:
        raise HTTPException(status_code=400, detail="This series is already boosted.")

    charged = spend_on_feature(supabase_admin, user_id, "boost_series")

    try:
        supabase_admin.table("transactions").insert({
            "user_id": user_id,
            "amount": 0,
            "type": boost_type,
            "description": f"Boosted series {series_id}",
        }).execute()
    except Exception as e:
        # The coins are already spent and logged under the generic
        # "boost_series" type by spend_on_feature above — only the
        # per-series marker failed to write, meaning the boost won't
        # actually show as active anywhere. Refund rather than charge a
        # creator for a boost that silently never took effect.
        refund_feature(supabase_admin, user_id, "boost_series", charged)
        raise HTTPException(status_code=500, detail=f"Could not activate boost: {e}")

    active_until = datetime.now(timezone.utc) + timedelta(hours=BOOST_WINDOW_HOURS)
    return BoostSeriesResponse(series_id=series_id, active_until=active_until.isoformat(), cost=charged)


@router.get("/series/boosted/active", response_model=ActiveSeriesBoostsResponse)
async def get_active_series_boosts():
    """Public read (no auth) — every trending/browse list needs this to
    render a boosted badge/apply the ranking lift for any viewer,
    logged-in or not, same as Spotlight's own GET."""
    if supabase_admin is None:
        return ActiveSeriesBoostsResponse(series_ids=[])

    try:
        rows = (
            supabase_admin.table("transactions")
            .select("type,created_at")
            .like("type", f"{_BOOST_TYPE_PREFIX}%")
            .gte("created_at", _window_start().isoformat())
            .order("created_at", desc=True)
            .execute()
        ).data or []
    except Exception:
        return ActiveSeriesBoostsResponse(series_ids=[])

    seen: set[str] = set()
    ordered_ids: list[str] = []
    for row in rows:
        series_id = row["type"][len(_BOOST_TYPE_PREFIX):]
        if series_id and series_id not in seen:
            seen.add(series_id)
            ordered_ids.append(series_id)

    return ActiveSeriesBoostsResponse(series_ids=ordered_ids)
