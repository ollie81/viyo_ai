"""
Solo-operator analytics — no third-party analytics account needed.

Two sources feed this, both already existing:
  - `transactions`: every coin spend/credit already flows through here
    (coins.py's spend_on_feature/credit_coins) — this is where revenue
    (coin_purchase) and feature-usage (hook_check, boost_post, etc.)
    come from.
  - `analytics_events`: a new table for the pure-engagement events that
    never touch this backend at all (post created, liked, followed,
    commented, signed up) — written directly from the Flutter app (see
    AnalyticsService.track), same shape as `likes`/`follows`.

Revenue is derived by mapping a coin_purchase transaction's coin amount
back to its USD price via COIN_PACKAGES (payments.py) rather than
storing a separate USD column — relies on every package having a
distinct coin amount, true today and worth keeping true if packages
change.

Gated by a shared secret (X-Admin-Key header) rather than a per-user
admin flag — there's no admin-role concept anywhere else in this app,
and a solo operator doesn't need one built just for this.
"""
import datetime
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

from payments import COIN_PACKAGES

router = APIRouter(prefix="/api/v1/admin", tags=["analytics"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

_LOOKBACK_ROWS = 20000  # generous cap for a month of an active app's events
_COINS_TO_USD_CENTS = {pkg["coins"]: pkg["usd_cents"] for pkg in COIN_PACKAGES.values()}


def _require_admin(x_admin_key: str = Header(None)) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Analytics endpoint is not configured.")
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


class AnalyticsSummary(BaseModel):
    total_users: int
    dau: int
    wau: int
    window_days: int
    event_counts: dict[str, int]
    feature_usage: dict[str, int]
    purchase_count: int
    revenue_usd_cents: int


@router.get("/analytics/summary", response_model=AnalyticsSummary)
async def analytics_summary(days: int = 7, x_admin_key: str = Header(None)):
    _require_admin(x_admin_key)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Analytics service is not configured.")

    now = datetime.datetime.now(datetime.timezone.utc)
    window_start = now - datetime.timedelta(days=days)
    day_start = now - datetime.timedelta(days=1)

    try:
        total_users = (
            supabase_admin.table("profiles").select("id", count="exact").limit(1).execute()
        ).count or 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not count users: {e}")

    try:
        events = (
            supabase_admin
            .table("analytics_events")
            .select("user_id,event_name,created_at")
            .gte("created_at", window_start.isoformat())
            .limit(_LOOKBACK_ROWS)
            .execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load analytics_events: {e}")

    try:
        transactions = (
            supabase_admin
            .table("transactions")
            .select("user_id,amount,type,created_at")
            .gte("created_at", window_start.isoformat())
            .limit(_LOOKBACK_ROWS)
            .execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load transactions: {e}")

    active_today: set[str] = set()
    active_week: set[str] = set()
    event_counts: dict[str, int] = {}
    for e in events:
        uid = e.get("user_id")
        created_at = e.get("created_at")
        if uid and created_at:
            active_week.add(uid)
            if created_at >= day_start.isoformat():
                active_today.add(uid)
        name = e.get("event_name") or "unknown"
        event_counts[name] = event_counts.get(name, 0) + 1

    feature_usage: dict[str, int] = {}
    purchase_count = 0
    revenue_usd_cents = 0
    for t in transactions:
        uid = t.get("user_id")
        created_at = t.get("created_at")
        if uid and created_at:
            active_week.add(uid)
            if created_at >= day_start.isoformat():
                active_today.add(uid)

        t_type = t.get("type") or "unknown"
        if t_type == "coin_purchase":
            purchase_count += 1
            coins = int(t.get("amount") or 0)
            revenue_usd_cents += _COINS_TO_USD_CENTS.get(coins, 0)
        elif int(t.get("amount") or 0) < 0:
            # Only spends count as "feature usage" — coin_purchase and any
            # other credit would otherwise double up with revenue above.
            feature_usage[t_type] = feature_usage.get(t_type, 0) + 1

    return AnalyticsSummary(
        total_users=total_users,
        dau=len(active_today),
        wau=len(active_week),
        window_days=days,
        event_counts=event_counts,
        feature_usage=feature_usage,
        purchase_count=purchase_count,
        revenue_usd_cents=revenue_usd_cents,
    )
