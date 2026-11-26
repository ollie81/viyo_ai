"""What the app may offer for real-money purchases right now: GET /api/v1/checkout/options (public, read-only).

The app shows a payment method or a Premium plan ONLY when this says it is genuinely available, so a provider that
merely appears in the code is never offered. Nothing here creates a payment, changes a setting, verifies a webhook
or touches the coin ledger: it reads the same settings each provider's own checkout and webhook endpoints require.

Coin checkouts (both the checkout AND its webhook verification must be configured, or coins could never be credited)
  stripe        STRIPE_SECRET_KEY + STRIPE_PUBLISHABLE_KEY + STRIPE_WEBHOOK_SECRET   (payments.py)
  paystack      PAYSTACK_SECRET_KEY (also signs its webhooks)                        (paystack_payments.py)
  flutterwave   FLUTTERWAVE_SECRET_KEY + FLUTTERWAVE_WEBHOOK_SECRET_HASH             (flutterwave_payments.py)
  lemonsqueezy  API key, store, coin variant + LEMONSQUEEZY_WEBHOOK_SECRET          (lemonsqueezy_payments.py)
  nardopay      nardopay_payments.checkout_problem() is None (switched on, configured, fresh exchange rate)
  The app also drops methods its platform can't run (e.g. Stripe's payment sheet isn't built for the web app).

Premium (real money, never coins): Lemon Squeezy subscription variants. A plan is purchasable only when the
subscription checkout is configured (subscriptions._configured, plus the webhook secret that activates it) AND its
real price could be read from Lemon Squeezy (GET /v1/variants/{id}: attributes.price in cents, interval,
interval_count; the store's currency from GET /v1/stores/{id}). Prices are never written in VIYO's code, so with no
readable price the plan is shown as unavailable rather than with an invented one. Prices are cached for an hour.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import requests
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["checkout"])

PLANS = ("weekly", "monthly", "yearly")
PRICE_TTL_SECONDS = 3600
_cache: dict = {"at": 0.0, "key": None, "value": None}
_lock = threading.Lock()


def coin_providers() -> list[str]:
    out: list[str] = []
    try:
        import payments as p
        if p.supabase_admin is not None and p.STRIPE_SECRET_KEY and p.STRIPE_PUBLISHABLE_KEY and p.STRIPE_WEBHOOK_SECRET:
            out.append("stripe")
    except Exception:
        pass
    try:
        import paystack_payments as ps
        if ps.supabase_admin is not None and ps.PAYSTACK_SECRET_KEY:
            out.append("paystack")
    except Exception:
        pass
    try:
        import flutterwave_payments as fw
        if fw.supabase_admin is not None and fw.FLUTTERWAVE_SECRET_KEY and fw.FLUTTERWAVE_WEBHOOK_SECRET_HASH:
            out.append("flutterwave")
    except Exception:
        pass
    try:
        import lemonsqueezy_payments as ls
        if ls._configured() and ls.LEMONSQUEEZY_WEBHOOK_SECRET:
            out.append("lemonsqueezy")
    except Exception:
        pass
    try:
        import nardopay_payments as nd
        if nd.checkout_problem() is None:
            out.append("nardopay")
    except Exception:
        pass
    return out


def _ls_get(get: Callable, path: str, key: str) -> Optional[dict]:
    try:
        r = get(f"https://api.lemonsqueezy.com/v1/{path}", timeout=10,
                headers={"Accept": "application/vnd.api+json", "Authorization": f"Bearer {key}"})
        if r.status_code != 200:
            return None
        return (r.json().get("data") or {}).get("attributes") or None
    except Exception:
        return None


def premium_plans(get: Callable = requests.get, now: Optional[float] = None) -> dict:
    """{"provider": "lemonsqueezy"|None, "plans": [{"plan", "available", "price_cents", "currency", "interval",
    "interval_count", "reason"}]}"""
    unavailable = {"provider": None, "plans": [
        {"plan": p, "available": False, "price_cents": None, "currency": None, "interval": None,
         "interval_count": None, "reason": "Premium subscriptions aren't open yet."} for p in PLANS]}
    try:
        import lemonsqueezy_payments as ls
        import subscriptions as sub
        if not sub._configured() or not ls.LEMONSQUEEZY_WEBHOOK_SECRET:
            return unavailable
        variants = dict(sub._VARIANT_BY_PLAN)
        key, store = ls.LEMONSQUEEZY_API_KEY, ls.LEMONSQUEEZY_STORE_ID
    except Exception:
        return unavailable

    now = time.time() if now is None else now
    cache_key = (store, tuple(variants.get(p) for p in PLANS))
    with _lock:
        if _cache["key"] == cache_key and now - _cache["at"] < PRICE_TTL_SECONDS and _cache["value"]:
            return _cache["value"]

    store_attrs = _ls_get(get, f"stores/{store}", key) or {}
    currency = str(store_attrs.get("currency") or "").upper() or None
    plans, all_ok = [], True
    for plan in PLANS:
        a = _ls_get(get, f"variants/{variants[plan]}", key) if currency else None
        price = a.get("price") if a else None
        ok = (isinstance(price, int) and not isinstance(price, bool) and price > 0 and a.get("is_subscription") is True)
        all_ok = all_ok and ok
        plans.append({"plan": plan, "available": ok, "price_cents": price if ok else None,
                      "currency": currency if ok else None,
                      "interval": a.get("interval") if ok else None,
                      "interval_count": a.get("interval_count") if ok else None,
                      "reason": None if ok else "This plan can't be bought right now."})
    value = {"provider": "lemonsqueezy", "plans": plans}
    if all_ok:                                   # only a complete answer is cached; a hiccup is retried next time
        with _lock:
            _cache.update(at=now, key=cache_key, value=value)
    return value


@router.get("/checkout/options")
async def checkout_options():
    import asyncio
    from payments import COIN_PACKAGES
    premium = await asyncio.to_thread(premium_plans)
    return {
        "coins": {
            "providers": coin_providers(),
            "packages": [{"id": pid, **pkg} for pid, pkg in COIN_PACKAGES.items()],
        },
        "premium": premium,
    }
