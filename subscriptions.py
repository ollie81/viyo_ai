"""
Viyo Premium — a real, recurring subscription (weekly or monthly) via
Lemon Squeezy, alongside Viyo Coins' one-time top-ups
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
two variants (Weekly, Monthly) created in its dashboard first — unlike
the coin purchases above, a subscription needs a real fixed price per
variant (custom_price only works for a one-time order), so there's no
way around creating those two variants for real and setting
LEMONSQUEEZY_WEEKLY_VARIANT_ID / LEMONSQUEEZY_MONTHLY_VARIANT_ID.
"""
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

_VARIANT_BY_PLAN = {
    "weekly": LEMONSQUEEZY_WEEKLY_VARIANT_ID,
    "monthly": LEMONSQUEEZY_MONTHLY_VARIANT_ID,
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


class SubscriptionCheckoutRequest(BaseModel):
    plan: Literal["weekly", "monthly"]


class SubscriptionCheckoutResponse(BaseModel):
    checkout_url: str


@router.post("/subscription/checkout", response_model=SubscriptionCheckoutResponse)
async def create_subscription_checkout(
    req: SubscriptionCheckoutRequest,
    user: tuple[str, str] = Depends(_get_current_user_and_email),
):
    if not _configured():
        raise HTTPException(status_code=503, detail="Subscriptions are not configured.")
    user_id, email = user
    variant_id = _VARIANT_BY_PLAN[req.plan]

    try:
        resp = requests.post(
            f"{LEMONSQUEEZY_BASE_URL}/checkouts",
            headers={
                "Accept": "application/vnd.api+json",
                "Content-Type": "application/vnd.api+json",
                "Authorization": f"Bearer {LEMONSQUEEZY_API_KEY}",
            },
            json={
                "data": {
                    "type": "checkouts",
                    "attributes": {
                        "checkout_data": {
                            "email": email,
                            "custom": {"user_id": user_id, "plan": req.plan},
                        },
                    },
                    "relationships": {
                        "store": {"data": {"type": "stores", "id": LEMONSQUEEZY_STORE_ID}},
                        "variant": {"data": {"type": "variants", "id": variant_id}},
                    },
                }
            },
            timeout=15,
        )
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Lemon Squeezy: {e}")

    if not resp.ok:
        detail = data.get("errors", [{}])[0].get("detail", "Could not start checkout.")
        raise HTTPException(status_code=502, detail=detail)

    checkout_url = data.get("data", {}).get("attributes", {}).get("url")
    if not checkout_url:
        raise HTTPException(status_code=502, detail="Lemon Squeezy did not return a checkout URL.")
    return SubscriptionCheckoutResponse(checkout_url=checkout_url)


class SubscriptionStatusResponse(BaseModel):
    is_subscribed: bool
    plan: Optional[str] = None
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
    rows = (
        supabase_admin.table("profiles")
        .select("is_subscribed, subscription_plan, subscription_status, subscription_renews_at, subscription_ends_at")
        .eq("id", user_id)
        .limit(1)
        .execute()
    ).data
    if not rows:
        return SubscriptionStatusResponse(is_subscribed=False)
    row = rows[0]
    return SubscriptionStatusResponse(
        is_subscribed=bool(row.get("is_subscribed")),
        plan=row.get("subscription_plan"),
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
    rows = admin.table("profiles").select("is_subscribed").eq("id", user_id).limit(1).execute().data
    return bool(rows and rows[0].get("is_subscribed"))


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
