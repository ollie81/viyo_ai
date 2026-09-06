"""
Viyo Coin balance/spend logic for AI feature gating.

Shared by main.py, coach.py, and repurpose.py so every OpenAI-calling
endpoint enforces the same free-taste-then-coins rule against the same
profiles.points_balance column the rest of the app already reads from
(see the Flutter CoinService and its RPCs: claim_daily_checkin,
purchase_badge, gift_coins).

This file is the one place that writes points_balance directly instead
of through a Postgres RPC. There's no schema/migration access from
this codebase to add a new server-side function for it, so deduction
uses an optimistic-concurrency compare-and-swap UPDATE over PostgREST
instead of relying on an atomic database-side function. A lost race
(two requests from the same user at the same instant) just means the
second one sees a 409 and can retry — never a double-spend.
"""
import time
from typing import Optional

from fastapi import HTTPException

# Coin cost + free-taste allowance per AI feature. Keep this in sync
# with the matching constants in the Flutter app (ai_service.dart /
# the button UI) — there is no single source of truth across the two
# repos, so a change here needs a matching change there.
#
# Deliberately NOT included, even though they call OpenAI: content_ideas,
# voice_check, weekly_report, trending, and analyze_post (Coach feedback).
# All five only ever fire automatically — dashboard cards that load on
# their own (daily_idea_card.dart, weekly_report_card.dart,
# trending_card.dart) or a non-blocking nudge/step inside another flow
# (voice_check during posting, analyze_post right after posting). Coin
# costs only apply to something a creator deliberately presses a button
# for — charging for ambient content would just silently drain a
# balance from normal app use.
FEATURE_COSTS: dict[str, dict] = {
    "hook_check": {"cost": 5, "free_per_day": 1, "label": "Hook Check"},
    "caption_variants": {"cost": 5, "free_per_day": 1, "label": "Caption Ideas"},
    "improve_caption": {"cost": 5, "free_per_day": 1, "label": "Improve Caption"},
    "coach_message": {"cost": 10, "free_per_day": 1, "label": "Coach Message"},
    "repurpose": {"cost": 40, "free_per_day": 0, "label": "AI Repurposer"},
    "post_insight": {"cost": 5, "free_per_day": 1, "label": "Why This Worked"},
    "boost_post": {"cost": 30, "free_per_day": 0, "label": "Boost Post"},
    "spotlight": {"cost": 25, "free_per_day": 0, "label": "Discover Spotlight"},
}

_FREE_TASTE_WINDOW_SECONDS = 24 * 60 * 60
# In-memory only — same tradeoff already accepted by every rate limiter
# in this codebase (main._check_rate_limit, repurpose's per-day limiter):
# resets on restart, doesn't share state across instances. Fine until
# this backend needs a real cache/DB-backed limiter.
_last_free_use: dict[tuple, float] = {}


def _has_free_taste(user_id: str, feature: str) -> bool:
    cfg = FEATURE_COSTS[feature]
    if cfg["free_per_day"] <= 0:
        return False
    key = (user_id, feature)
    last = _last_free_use.get(key)
    return last is None or (time.time() - last) >= _FREE_TASTE_WINDOW_SECONDS


def _consume_free_taste(user_id: str, feature: str) -> None:
    _last_free_use[(user_id, feature)] = time.time()


def _log_spend(admin, user_id: str, feature: str, cost: int) -> None:
    """Best-effort — a failed history log never blocks the feature the
    creator already paid for.

    `type` is the feature key itself (e.g. "spotlight"), not a flat
    "ai_feature_spend" constant — Discover Spotlight's "am I currently
    spotlighted" check (discover.py) needs to query transactions by
    feature, since there's no dedicated column/table for that state.
    """
    try:
        admin.table("transactions").insert({
            "user_id": user_id,
            "amount": -cost,
            "type": feature,
            "description": f"Used {FEATURE_COSTS[feature]['label']}",
        }).execute()
    except Exception as e:
        print(f"[WARN] Could not log coin spend for {feature}: {e}")


def spend_on_feature(admin, user_id: str, feature: str) -> None:
    """
    Call before running the OpenAI call for a gated feature — raises
    if the creator can't use it right now, returns normally (no
    return value) if they can, having already spent the coins (or
    consumed a free taste) as a side effect.

    Raises:
        HTTPException(402) — no free taste left and balance < cost.
            detail carries {feature, balance, needed} so the client can
            show exactly how short they are.
        HTTPException(409) — a concurrent request changed the balance
            between the read and the write; safe to just retry.
        HTTPException(500) — the balance check/update itself failed.
    """
    if feature not in FEATURE_COSTS:
        raise ValueError(f"Unknown gated feature: {feature!r}")

    if admin is None:
        raise HTTPException(status_code=503, detail="Coin service is not configured.")

    if _has_free_taste(user_id, feature):
        _consume_free_taste(user_id, feature)
        return

    cost = FEATURE_COSTS[feature]["cost"]

    try:
        profile = (
            admin.table("profiles")
            .select("points_balance")
            .eq("id", user_id)
            .single()
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check coin balance: {e}")

    balance = (profile.data or {}).get("points_balance") or 0
    if balance < cost:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_coins",
                "feature": feature,
                "balance": balance,
                "needed": cost,
            },
        )

    try:
        result = (
            admin.table("profiles")
            .update({"points_balance": balance - cost})
            .eq("id", user_id)
            .eq("points_balance", balance)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not spend coins: {e}")

    if not result.data:
        raise HTTPException(
            status_code=409,
            detail={"error": "balance_changed", "feature": feature},
        )

    _log_spend(admin, user_id, feature, cost)


def credit_coins(admin, user_id: str, amount: int, type_: str, description: str) -> None:
    """
    Adds coins to a balance — the mirror image of spend_on_feature's
    compare-and-swap deduction, used when coins are created rather than
    spent (currently: payments.py, after a Stripe purchase is confirmed
    server-side). `amount` must already be a trusted, server-decided
    number; never pass through a client-supplied value.

    Raises HTTPException(409) on a lost compare-and-swap race — callers
    driven by a webhook can just let that surface as a non-2xx response,
    since the webhook sender will retry.
    """
    if admin is None:
        raise HTTPException(status_code=503, detail="Coin service is not configured.")

    try:
        profile = (
            admin.table("profiles")
            .select("points_balance")
            .eq("id", user_id)
            .single()
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check coin balance: {e}")

    balance = (profile.data or {}).get("points_balance") or 0

    try:
        result = (
            admin.table("profiles")
            .update({"points_balance": balance + amount})
            .eq("id", user_id)
            .eq("points_balance", balance)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not credit coins: {e}")

    if not result.data:
        raise HTTPException(status_code=409, detail={"error": "balance_changed"})

    try:
        admin.table("transactions").insert({
            "user_id": user_id,
            "amount": amount,
            "type": type_,
            "description": description,
        }).execute()
    except Exception as e:
        print(f"[WARN] Could not log coin credit for {user_id}: {e}")
