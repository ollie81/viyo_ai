"""
Viyo Coin purchases — real-money top-ups via Lemon Squeezy, alongside
Stripe (payments.py), Paystack (paystack_payments.py) and Flutterwave
(flutterwave_payments.py). Same rules as all three: the client only
ever sends a package id, coins are only ever credited after Lemon
Squeezy confirms the order via webhook, and every provider shares the
one COIN_PACKAGES table in payments.py.

Lemon Squeezy is a merchant of record (it handles global tax/VAT
itself) with one hosted checkout per store — rather than a variant per
coin package, this uses a single generic "Viyo Coins" variant and
overrides its price per request with `custom_price`, the same way
Stripe's PaymentIntent amount and Paystack's `amount` are set
per-package dynamically instead of needing four separate SKUs.

Idempotency: same trick as every other provider here — no dedicated
column to key a "already credited this order" check off, so the Lemon
Squeezy order id is embedded in the transaction's description and
checked before crediting.
"""
import hashlib
import hmac
import os
from typing import Optional

import requests
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from supabase import create_client, Client

from coins import credit_coins
from payments import COIN_PACKAGES

router = APIRouter(prefix="/api/v1", tags=["payments"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
LEMONSQUEEZY_API_KEY = os.environ.get("LEMONSQUEEZY_API_KEY", "")
LEMONSQUEEZY_STORE_ID = os.environ.get("LEMONSQUEEZY_STORE_ID", "")
# One generic "Viyo Coins" product variant, reused for every package —
# its own price is overridden per request via custom_price below, so
# there's no need for one variant per coin package.
LEMONSQUEEZY_VARIANT_ID = os.environ.get("LEMONSQUEEZY_VARIANT_ID", "")
LEMONSQUEEZY_WEBHOOK_SECRET = os.environ.get("LEMONSQUEEZY_WEBHOOK_SECRET", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

LEMONSQUEEZY_BASE_URL = "https://api.lemonsqueezy.com/v1"


async def _get_current_user_and_email(authorization: str = Header(None)) -> tuple[str, str]:
    from main import _decode_token

    payload = _decode_token(authorization)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")
    if payload.get("is_anonymous"):
        raise HTTPException(status_code=403, detail="Guest accounts can't purchase coins — create an account first.")
    email = payload.get("email") or f"{user_id}@viyo.app"
    return user_id, email


class LemonSqueezyInitRequest(BaseModel):
    package_id: str = Field(..., min_length=1)


class LemonSqueezyInitResponse(BaseModel):
    checkout_url: str
    coins: int


def _configured() -> bool:
    return bool(
        supabase_admin is not None
        and LEMONSQUEEZY_API_KEY
        and LEMONSQUEEZY_STORE_ID
        and LEMONSQUEEZY_VARIANT_ID
    )


@router.post("/coins/purchase/lemonsqueezy/initialize", response_model=LemonSqueezyInitResponse)
async def initialize_lemonsqueezy_purchase(
    req: LemonSqueezyInitRequest,
    user: tuple[str, str] = Depends(_get_current_user_and_email),
):
    if not _configured():
        raise HTTPException(status_code=503, detail="Lemon Squeezy payments are not configured.")

    user_id, email = user
    package = COIN_PACKAGES.get(req.package_id)
    if package is None:
        raise HTTPException(status_code=400, detail="Unknown coin package.")

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
                        # Lemon Squeezy takes the smallest currency unit
                        # (cents) here too, same as Stripe/Paystack.
                        "custom_price": package["usd_cents"],
                        "checkout_data": {
                            "email": email,
                            "custom": {
                                "user_id": user_id,
                                "package_id": req.package_id,
                                "coins": package["coins"],
                            },
                        },
                    },
                    "relationships": {
                        "store": {"data": {"type": "stores", "id": LEMONSQUEEZY_STORE_ID}},
                        "variant": {"data": {"type": "variants", "id": LEMONSQUEEZY_VARIANT_ID}},
                    },
                }
            },
            timeout=15,
        )
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Lemon Squeezy: {e}")

    if not resp.ok:
        detail = data.get("errors", [{}])[0].get("detail", "Could not start Lemon Squeezy checkout.")
        raise HTTPException(status_code=502, detail=detail)

    checkout_url = data.get("data", {}).get("attributes", {}).get("url")
    if not checkout_url:
        raise HTTPException(status_code=502, detail="Lemon Squeezy did not return a checkout URL.")

    return LemonSqueezyInitResponse(checkout_url=checkout_url, coins=package["coins"])


def _already_credited(admin, order_id: str) -> bool:
    existing = (
        admin.table("transactions")
        .select("id")
        .eq("type", "coin_purchase")
        .ilike("description", f"%{order_id}%")
        .limit(1)
        .execute()
    )
    return bool(existing.data)


@router.post("/coins/webhook/lemonsqueezy")
async def lemonsqueezy_webhook(request: Request, x_signature: str = Header(None)):
    if not _configured() or not LEMONSQUEEZY_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Lemon Squeezy payments are not configured.")

    body = await request.body()

    # Lemon Squeezy signs the raw body with HMAC-SHA256 using the
    # webhook's signing secret — verified before trusting anything in
    # the payload, same posture as every other provider here.
    expected = hmac.new(LEMONSQUEEZY_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not x_signature or not hmac.compare_digest(expected, x_signature):
        raise HTTPException(status_code=400, detail="Invalid Lemon Squeezy signature.")

    event = await request.json()
    meta = event.get("meta") or {}
    if meta.get("event_name") != "order_created":
        return {"received": True}

    order = event.get("data") or {}
    attrs = order.get("attributes") or {}
    if attrs.get("status") != "paid":
        return {"received": True}

    custom = meta.get("custom_data") or {}
    user_id = custom.get("user_id")
    package_id = custom.get("package_id")
    coins = custom.get("coins")
    order_id = order.get("id")

    if not user_id or not coins or not order_id:
        return {"received": True}

    try:
        if _already_credited(supabase_admin, order_id):
            return {"received": True}

        label = COIN_PACKAGES.get(package_id, {}).get("label", f"{coins} coins")
        credit_coins(
            supabase_admin,
            user_id,
            int(coins),
            "coin_purchase",
            f"Purchased {label} — Lemon Squeezy order {order_id}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not credit coins: {e}")

    return {"received": True}
