"""What the app may offer for real-money purchases right now: GET /api/v1/checkout/options (public, read-only).

The app shows a payment method or a Premium plan ONLY when this says it is genuinely available, so a provider that
merely appears in the code is never offered. Nothing here creates a payment, changes a setting, verifies a webhook
or touches the coin ledger: it reads the same settings each provider's own checkout and webhook endpoints require.

Coin checkouts (both the checkout AND its webhook verification must be configured, or coins could never be credited)
  stripe        STRIPE_SECRET_KEY + STRIPE_PUBLISHABLE_KEY + STRIPE_WEBHOOK_SECRET   (payments.py)
  paystack      PAYSTACK_SECRET_KEY (also signs its webhooks)                        (paystack_payments.py)
  flutterwave   FLUTTERWAVE_SECRET_KEY + FLUTTERWAVE_WEBHOOK_SECRET_HASH             (flutterwave_payments.py)
  nardopay      nardopay_payments.checkout_problem() is None (switched on, configured, fresh exchange rate)
  Lemon Squeezy is no longer sold through. The app also drops methods its platform can't run (e.g. Stripe's payment
  sheet isn't built for the web app).

Premium (real money, never coins): NardoPay subscriptions. The plans are listed as purchasable only when
nardopay_payments._premium_problem() is None (NardoPay switched on and configured, subscription tables set up). The
displayed USD prices are the server's PREMIUM_PLANS; NardoPay charges each plan's fixed RWF amount.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["checkout"])

PLANS = ("weekly", "monthly", "yearly")


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
        import nardopay_payments as nd
        if nd.checkout_problem() is None:
            out.append("nardopay")
    except Exception:
        pass
    return out


def premium_plans() -> dict:
    """{"provider": "nardopay"|None, "plans": [{"plan", "available", "price_cents", "currency", "interval",
    "interval_count", "reason"}]}"""
    try:
        import nardopay_payments as nd
        problem = nd._premium_problem()
        plans = nd.PREMIUM_PLANS
    except Exception:
        problem, plans = "unavailable", {}
    if problem or not plans:
        return {"provider": None, "plans": [
            {"plan": p, "available": False, "price_cents": None, "currency": None, "interval": None,
             "interval_count": None, "reason": "Premium subscriptions aren't open yet."} for p in PLANS]}
    return {"provider": "nardopay", "plans": [
        {"plan": p, "available": True, "price_cents": plans[p]["usd_cents"], "currency": "USD",
         "interval": plans[p]["interval"], "interval_count": 1, "reason": None} for p in PLANS]}


@router.get("/checkout/options")
async def checkout_options():
    from payments import COIN_PACKAGES
    return {
        "coins": {
            "providers": coin_providers(),
            "packages": [{"id": pid, **pkg} for pid, pkg in COIN_PACKAGES.items()],
        },
        "premium": premium_plans(),
    }
