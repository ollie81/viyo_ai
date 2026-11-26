"""
NOW: Premium is sold through NardoPay subscriptions (nardopay_payments.py, nardopay_subscriptions.sql). Lemon
Squeezy checkout was removed; only its event handling below remains, for subscriptions bought earlier.
This module also holds is_active_subscriber (the paywall check) and GET /subscription/status.

History: Viyo Premium — a real, recurring subscription (weekly, monthly, or
yearly) via Lemon Squeezy, alongside Viyo Coins' one-time top-ups
(lemonsqueezy_payments.py and the other *_payments.py files). An
active subscriber skips the per-episode coin paywall entirely (see
episodes.py's is_active_subscriber call) instead of spending coins
episode by episode.

Scope decision: a subscriber's unlock is free for the VIEWER and free
for the CREATOR too — no creator_earnings credit happens here, unlike
a coin-funded unlock_episode. Crediting creator_share in coins here
would mint coins not backed by any real purchase (subscription revenue
isn't coins, there's nothing to split per-episode). Paying creators a
share of subscription revenue is a real feature this doesn't attempt —
it would need tracking which subscriber watched which creator's
content and reconciling that against actual Lemon Squeezy payouts,
which is its own project, not a few lines here.

Reuses lemonsqueezy_payments.py's store id, API key, base URL and
webhook signing secret — one Lemon Squeezy webhook endpoint already
handles both order and subscription events (see that file's
lemonsqueezy_webhook, which delegates every subscription_* event to
handle_subscription_event below), so there's only one webhook URL to
register in the Lemon Squeezy dashboard, not two.

profiles columns this reads/writes (see the SQL migration handed over
alongside this file): is_subscribed, subscription_plan,
subscription_provider, subscription_status, subscription_id,
subscription_renews_at, subscription_ends_at. No schema/migration
access from this codebase (same limitation coins.py's own docstring
calls out), so these are written directly over PostgREST, not through
a Postgres function.

Lemon Squeezy needs a real "Viyo Premium" subscription product with
three variants (Weekly, Monthly, Yearly) created in its dashboard
first — unlike the coin purchases above, a subscription needs a real
fixed price per variant (custom_price only works for a one-time
order), so there's no way around creating those three variants for
real and setting LEMONSQUEEZY_WEEKLY_VARIANT_ID /
LEMONSQUEEZY_MONTHLY_VARIANT_ID / LEMONSQUEEZY_YEARLY_VARIANT_ID.
"""
import datetime
import os
from typing import Literal, Optional

import requests
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from lemonsqueezy_payments import (
    LEMONSQUEEZY_API_KEY,
    LEMONSQUEEZY_BASE_URL,
    LEMONSQUEEZY_STORE_ID,
    supabase_admin,
)

router = APIRouter(prefix="/api/v1", tags=["subscriptions"])

LEMONSQUEEZY_WEEKLY_VARIANT_ID = os.environ.get("LEMONSQUEEZY_WEEKLY_VARIANT_ID", "")
LEMONSQUEEZY_MONTHLY_VARIANT_ID = os.environ.get("LEMONSQUEEZY_MONTHLY_VARIANT_ID", "")
LEMONSQUEEZY_YEARLY_VARIANT_ID = os.environ.get("LEMONSQUEEZY_YEARLY_VARIANT_ID", "")

_VARIANT_BY_PLAN = {
    "weekly": LEMONSQUEEZY_WEEKLY_VARIANT_ID,
    "monthly": LEMONSQUEEZY_MONTHLY_VARIANT_ID,
    "yearly": LEMONSQUEEZY_YEARLY_VARIANT_ID,
}

# Lemon Squeezy subscription statuses that mean "currently has access" —
# everything else (cancelled, expired, past_due, unpaid, paused) does
# not. See https://docs.lemonsqueezy.com/api/subscriptions for the
# full status enum.
_ACTIVE_STATUSES = {"active", "on_trial"}


def _configured() -> bool:
    return bool(
        supabase_admin is not None
        and LEMONSQUEEZY_API_KEY
        and LEMONSQUEEZY_STORE_ID
        and LEMONSQUEEZY_WEEKLY_VARIANT_ID
        and LEMONSQUEEZY_MONTHLY_VARIANT_ID
        and LEMONSQUEEZY_YEARLY_VARIANT_ID
    )


async def _get_current_user_and_email(authorization: str = Header(None)) -> tuple[str, str]:
    from main import _decode_token

    payload = _decode_token(authorization)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")
    if payload.get("is_anonymous"):
        raise HTTPException(status_code=403, detail="Guest accounts can't subscribe — create an account first.")
    email = payload.get("email") or f"{user_id}@viyo.app"
    return user_id, email


# Premium is sold through NardoPay subscriptions (nardopay_payments.create_premium_subscription). The Lemon Squeezy
# checkout was removed; handle_subscription_event below stays so subscriptions bought earlier through Lemon Squeezy
# still receive their renewal / cancellation / expiry updates until they end.


class SubscriptionStatusResponse(BaseModel):
    is_subscribed: bool
    plan: Optional[str] = None
    provider: Optional[str] = None
    status: Optional[str] = None
    renews_at: Optional[str] = None
    ends_at: Optional[str] = None


@router.get("/subscription/status", response_model=SubscriptionStatusResponse)
async def get_subscription_status(user: tuple[str, str] = Depends(_get_current_user_and_email)):
    """Lets the app show "Premium until <date>" / a manage-subscription
    screen. The Flutter paywall gate itself doesn't call this —
    SeriesService._withUnlockState reads profiles.is_subscribed
    directly, same "read your own state, don't round-trip through the
    backend for it" pattern it already uses for episode_unlocks."""
    user_id, _ = user
    if supabase_admin is None:
        return SubscriptionStatusResponse(is_subscribed=False)
    try:
        rows = (
            supabase_admin.table("profiles")
            .select("is_subscribed, subscription_plan, subscription_provider, subscription_status, "
                    "subscription_renews_at, subscription_ends_at")
            .eq("id", user_id)
            .limit(1)
            .execute()
        ).data
    except Exception as e:
        print(f"[WARN] subscription status for {user_id} unavailable: {e}")
        rows = None
    if not rows:
        return SubscriptionStatusResponse(is_subscribed=False)
    row = rows[0]
    return SubscriptionStatusResponse(
        is_subscribed=bool(row.get("is_subscribed")) and not nardopay_premium_ended(row),
        plan=row.get("subscription_plan"),
        provider=row.get("subscription_provider"),
        status=row.get("subscription_status"),
        renews_at=row.get("subscription_renews_at"),
        ends_at=row.get("subscription_ends_at"),
    )


def is_active_subscriber(admin, user_id: str) -> bool:
    """Used by episodes.py's unlock_episode/unlock_series_bundle to
    skip the coin paywall for a subscriber — the one place this check
    needs to happen for the backend's own gate."""
    if admin is None:
        return False
    # Never let this check break a paid unlock: if the subscription columns are missing (the migration hasn't
    # been run: Postgres error 42703) or the lookup fails, the viewer is treated as not subscribed and simply
    # pays with coins as normal. Failing this way can only ever charge someone, never give premium for free.
    try:
        rows = admin.table("profiles").select("is_subscribed, subscription_provider, subscription_ends_at") \
            .eq("id", user_id).limit(1).execute().data
    except Exception as e:
        print(f"[WARN] subscriber check for {user_id} failed, treating as not subscribed: {e}")
        return False
    if not rows or not rows[0].get("is_subscribed"):
        return False
    # NardoPay Premium always has an end date (paid period + grace): past it, it gives nothing, even before the
    # worker switches it off.
    return not nardopay_premium_ended(rows[0])


def nardopay_premium_ended(row: dict) -> bool:
    if row.get("subscription_provider") != "nardopay":
        return False
    ends = row.get("subscription_ends_at")
    if not ends:
        return True
    try:
        t = datetime.datetime.fromisoformat(str(ends).replace("Z", "+00:00"))
    except ValueError:
        return True
    if t.tzinfo is None:
        t = t.replace(tzinfo=datetime.timezone.utc)
    return t <= datetime.datetime.now(datetime.timezone.utc)


def handle_subscription_event(event: dict) -> None:
    """Called from lemonsqueezy_payments.py's webhook for every
    subscription_* event (created, updated, cancelled, resumed,
    expired, paused, unpaused) — kept here instead of there so the
    plan/variant config and the profiles write live next to each
    other. Trusts meta.custom_data.user_id the same way the order
    webhook trusts it for coin purchases — Lemon Squeezy includes the
    checkout's original custom data on every subsequent event for that
    subscription, not just the first one."""
    if supabase_admin is None:
        return
    meta = event.get("meta") or {}
    custom = meta.get("custom_data") or {}
    user_id = custom.get("user_id")
    if not user_id:
        return
    plan = custom.get("plan")

    data = event.get("data") or {}
    attrs = data.get("attributes") or {}
    status = attrs.get("status")
    is_subscribed = status in _ACTIVE_STATUSES

    # An old Lemon Squeezy subscription's events never overwrite NardoPay Premium someone has paid for since.
    try:
        cur = supabase_admin.table("profiles").select("is_subscribed, subscription_provider, subscription_ends_at") \
            .eq("id", user_id).limit(1).execute().data
        if cur and cur[0].get("is_subscribed") and cur[0].get("subscription_provider") == "nardopay" \
                and not nardopay_premium_ended(cur[0]):
            return
    except Exception:
        pass

    try:
        supabase_admin.table("profiles").update({
            "is_subscribed": is_subscribed,
            "subscription_plan": plan,
            "subscription_provider": "lemonsqueezy",
            "subscription_status": status,
            "subscription_id": str(data["id"]) if data.get("id") is not None else None,
            "subscription_renews_at": attrs.get("renews_at"),
            "subscription_ends_at": attrs.get("ends_at"),
        }).eq("id", user_id).execute()
    except Exception as e:
        print(f"[WARN] Could not update subscription state for user {user_id}: {e}")
