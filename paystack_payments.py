"""
Viyo Coin purchases — real-money top-ups via Paystack, alongside the
Stripe flow in payments.py. Shares that file's COIN_PACKAGES (the same
packages are sellable through either provider) and the same rules:
the client only ever sends a package id, never a price or coin amount,
and coins are only ever credited from the webhook after Paystack
itself confirms the charge — never from the initialize call the client
makes to start checkout.

Paystack's checkout is a hosted page (no in-app payment sheet like
Stripe's), so the flow is: initialize a transaction server-side, hand
the client an authorization_url to open in a browser, and let the
webhook credit coins once the charge actually completes.

Idempotency: same trick as payments.py's Stripe webhook — no dedicated
column to key a "already credited this reference" check off, so the
Paystack transaction reference is embedded in the transaction's
description and checked before crediting.
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
PAYSTACK_SECRET_KEY = os.environ.get("PAYSTACK_SECRET_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

PAYSTACK_BASE_URL = "https://api.paystack.co"


async def _get_current_user_and_email(authorization: str = Header(None)) -> tuple[str, str]:
    """
    Paystack's initialize call requires a customer email, which
    get_current_user_id_no_guest doesn't expose (it only returns the
    JWT subject) — this decodes the same token again for the `email`
    claim Supabase always includes for a real (non-guest) account.
    """
    from main import _decode_token

    payload = _decode_token(authorization)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")
    if payload.get("is_anonymous"):
        raise HTTPException(status_code=403, detail="Guest accounts can't purchase coins — create an account first.")
    email = payload.get("email") or f"{user_id}@viyo.app"
    return user_id, email


class PaystackInitRequest(BaseModel):
    package_id: str = Field(..., min_length=1)


class PaystackInitResponse(BaseModel):
    authorization_url: str
    reference: str
    coins: int


@router.post("/coins/purchase/paystack/initialize", response_model=PaystackInitResponse)
async def initialize_paystack_purchase(
    req: PaystackInitRequest,
    user: tuple[str, str] = Depends(_get_current_user_and_email),
):
    if supabase_admin is None or not PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Paystack payments are not configured.")

    user_id, email = user
    package = COIN_PACKAGES.get(req.package_id)
    if package is None:
        raise HTTPException(status_code=400, detail="Unknown coin package.")

    try:
        resp = requests.post(
            f"{PAYSTACK_BASE_URL}/transaction/initialize",
            headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"},
            json={
                "email": email,
                # Paystack takes the smallest currency unit, same as
                # Stripe's usd_cents — no conversion needed.
                "amount": package["usd_cents"],
                "currency": "USD",
                "metadata": {
                    "user_id": user_id,
                    "package_id": req.package_id,
                    "coins": package["coins"],
                },
            },
            timeout=15,
        )
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Paystack: {e}")

    if not resp.ok or not data.get("status"):
        raise HTTPException(status_code=502, detail=data.get("message", "Could not start Paystack payment."))

    tx = data["data"]
    return PaystackInitResponse(
        authorization_url=tx["authorization_url"],
        reference=tx["reference"],
        coins=package["coins"],
    )


def _already_credited(admin, reference: str) -> bool:
    existing = (
        admin.table("transactions")
        .select("id")
        .eq("type", "coin_purchase")
        .ilike("description", f"%{reference}%")
        .limit(1)
        .execute()
    )
    return bool(existing.data)


@router.post("/coins/webhook/paystack")
async def paystack_webhook(request: Request, x_paystack_signature: str = Header(None)):
    if supabase_admin is None or not PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Paystack payments are not configured.")

    body = await request.body()

    # Paystack signs the raw body with HMAC-SHA512 using the secret
    # key — verified before trusting anything in the payload, same
    # posture as Stripe's signature check in payments.py.
    expected = hmac.new(PAYSTACK_SECRET_KEY.encode(), body, hashlib.sha512).hexdigest()
    if not x_paystack_signature or not hmac.compare_digest(expected, x_paystack_signature):
        raise HTTPException(status_code=400, detail="Invalid Paystack signature.")

    event = await request.json()
    if event.get("event") != "charge.success":
        return {"received": True}

    tx = event.get("data") or {}
    metadata = tx.get("metadata") or {}
    user_id = metadata.get("user_id")
    package_id = metadata.get("package_id")
    coins = metadata.get("coins")
    reference = tx.get("reference")

    if not user_id or not coins or not reference:
        return {"received": True}

    try:
        if _already_credited(supabase_admin, reference):
            return {"received": True}

        label = COIN_PACKAGES.get(package_id, {}).get("label", f"{coins} coins")
        credit_coins(
            supabase_admin,
            user_id,
            int(coins),
            "coin_purchase",
            f"Purchased {label} — Paystack {reference}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not credit coins: {e}")

    return {"received": True}
