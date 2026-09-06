"""
Viyo Coin purchases — real-money top-ups via Stripe.

Every other coin movement in this app (spend_on_feature, boost_post,
spotlight) only ever moves coins the platform already trusts the value
of. This is the one place coins get created from real money, so the
same "never trust a client-supplied number" rule from posts.py/coins.py
applies twice as hard here: the client only ever sends a package *id*,
never a price or coin amount — both come from COIN_PACKAGES below. The
actual credit only ever happens from the Stripe webhook, after Stripe
itself confirms the charge succeeded, never from the create-intent call
the client makes to start checkout.

Idempotency: Stripe redelivers a webhook until it gets a 2xx, and can
send the same event more than once even without a redelivery. There's
no dedicated column to key a "have we credited this payment_intent yet"
check off (no schema/migration access from this codebase), so the
payment_intent id is embedded in the transaction's description and
checked before crediting — the same schema-free-state trick discover.py
uses for "is this creator currently spotlighted".
"""
import os
from typing import Optional

import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from supabase import create_client, Client

from coins import credit_coins

router = APIRouter(prefix="/api/v1", tags=["payments"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

stripe.api_key = STRIPE_SECRET_KEY

# Coin amount + USD price per package — the single source of truth. The
# Flutter app fetches these from GET /coins/packages rather than
# hardcoding them, so a price change never needs a client release.
# Bonus coins-per-dollar increases with tier to make bigger purchases
# feel worthwhile.
COIN_PACKAGES: dict[str, dict] = {
    "starter": {"coins": 100, "usd_cents": 99, "label": "100 Coins"},
    "popular": {"coins": 550, "usd_cents": 499, "label": "550 Coins"},
    "value": {"coins": 1200, "usd_cents": 999, "label": "1,200 Coins"},
    "creator": {"coins": 3000, "usd_cents": 1999, "label": "3,000 Coins"},
}


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


class CoinPackage(BaseModel):
    id: str
    coins: int
    usd_cents: int
    label: str


class CoinPackagesResponse(BaseModel):
    packages: list[CoinPackage]


class CreateIntentRequest(BaseModel):
    package_id: str = Field(..., min_length=1)


class CreateIntentResponse(BaseModel):
    client_secret: str
    publishable_key: str
    amount_usd_cents: int
    coins: int


@router.get("/coins/packages", response_model=CoinPackagesResponse)
async def list_coin_packages():
    return CoinPackagesResponse(
        packages=[CoinPackage(id=pid, **pkg) for pid, pkg in COIN_PACKAGES.items()]
    )


@router.post("/coins/purchase/create-intent", response_model=CreateIntentResponse)
async def create_purchase_intent(
    req: CreateIntentRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    if supabase_admin is None or not STRIPE_SECRET_KEY or not STRIPE_PUBLISHABLE_KEY:
        raise HTTPException(status_code=503, detail="Payments are not configured.")

    package = COIN_PACKAGES.get(req.package_id)
    if package is None:
        raise HTTPException(status_code=400, detail="Unknown coin package.")

    try:
        intent = stripe.PaymentIntent.create(
            amount=package["usd_cents"],
            currency="usd",
            metadata={
                "user_id": user_id,
                "package_id": req.package_id,
                "coins": str(package["coins"]),
            },
            description=f"Viyo Coins — {package['label']}",
        )
    except stripe.StripeError as e:
        raise HTTPException(status_code=502, detail=f"Could not start payment: {e}")

    return CreateIntentResponse(
        client_secret=intent.client_secret,
        publishable_key=STRIPE_PUBLISHABLE_KEY,
        amount_usd_cents=package["usd_cents"],
        coins=package["coins"],
    )


def _already_credited(admin, payment_intent_id: str) -> bool:
    existing = (
        admin.table("transactions")
        .select("id")
        .eq("type", "coin_purchase")
        .ilike("description", f"%{payment_intent_id}%")
        .limit(1)
        .execute()
    )
    return bool(existing.data)


@router.post("/coins/webhook")
async def stripe_webhook(request: Request):
    if supabase_admin is None or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Payments are not configured.")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.SignatureVerificationError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid webhook signature: {e}")

    if event["type"] != "payment_intent.succeeded":
        return {"received": True}

    intent = event["data"]["object"]
    metadata = intent.get("metadata") or {}
    user_id = metadata.get("user_id")
    package_id = metadata.get("package_id")
    coins = metadata.get("coins")

    if not user_id or not coins:
        # Not one of our coin-purchase intents (or malformed metadata) —
        # ignore rather than error, so Stripe doesn't keep retrying
        # something this endpoint can never resolve.
        return {"received": True}

    try:
        if _already_credited(supabase_admin, intent["id"]):
            return {"received": True}

        label = COIN_PACKAGES.get(package_id, {}).get("label", f"{coins} coins")
        credit_coins(
            supabase_admin,
            user_id,
            int(coins),
            "coin_purchase",
            f"Purchased {label} — Stripe {intent['id']}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not credit coins: {e}")

    return {"received": True}
