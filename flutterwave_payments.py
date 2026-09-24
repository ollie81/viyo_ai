"""
Viyo Coin purchases — real-money top-ups via Flutterwave, alongside
Stripe (payments.py) and Paystack (paystack_payments.py). Same rules
as both: the client only ever sends a package id, coins are only ever
credited after Flutterwave confirms the charge, and every provider
shares the one COIN_PACKAGES table in payments.py.

Flutterwave's checkout is also a hosted page: initialize a payment
server-side, hand the client a `link` to open in a browser. Unlike
Stripe/Paystack, this webhook also re-verifies the transaction via
Flutterwave's own /transactions/{id}/verify endpoint before crediting
— not just checking the webhook's signature — since Flutterwave's own
docs call this out as the safer pattern (belt-and-suspenders against a
webhook payload that matched the hash but doesn't reflect the real,
fully-settled charge).
"""
import os
import uuid
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
FLUTTERWAVE_SECRET_KEY = os.environ.get("FLUTTERWAVE_SECRET_KEY", "")
# Set in the Flutterwave dashboard under Settings -> Webhooks as the
# "secret hash" — an arbitrary shared string, not derived from the API
# key, that Flutterwave echoes back verbatim in every webhook request.
FLUTTERWAVE_WEBHOOK_SECRET_HASH = os.environ.get("FLUTTERWAVE_WEBHOOK_SECRET_HASH", "")
# Where Flutterwave sends the browser after checkout finishes — a
# plain confirmation page (see payment_complete_page below), since
# this app has no registered deep link to hand back to for an
# in-browser checkout flow.
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "https://viyoai-production.up.railway.app")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

FLUTTERWAVE_BASE_URL = "https://api.flutterwave.com/v3"


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


class FlutterwaveInitRequest(BaseModel):
    package_id: str = Field(..., min_length=1)


class FlutterwaveInitResponse(BaseModel):
    payment_link: str
    tx_ref: str
    coins: int


@router.post("/coins/purchase/flutterwave/initialize", response_model=FlutterwaveInitResponse)
async def initialize_flutterwave_purchase(
    req: FlutterwaveInitRequest,
    user: tuple[str, str] = Depends(_get_current_user_and_email),
):
    if supabase_admin is None or not FLUTTERWAVE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Flutterwave payments are not configured.")

    user_id, email = user
    package = COIN_PACKAGES.get(req.package_id)
    if package is None:
        raise HTTPException(status_code=400, detail="Unknown coin package.")

    tx_ref = f"viyo-{uuid.uuid4()}"
    try:
        resp = requests.post(
            f"{FLUTTERWAVE_BASE_URL}/payments",
            headers={"Authorization": f"Bearer {FLUTTERWAVE_SECRET_KEY}"},
            json={
                "tx_ref": tx_ref,
                # Flutterwave takes a major-unit amount (dollars), unlike
                # Stripe/Paystack's smallest-unit cents.
                "amount": package["usd_cents"] / 100,
                "currency": "USD",
                "redirect_url": f"{BACKEND_PUBLIC_URL}/api/v1/coins/payment-complete",
                "customer": {"email": email},
                "meta": {
                    "user_id": user_id,
                    "package_id": req.package_id,
                    "coins": package["coins"],
                },
            },
            timeout=15,
        )
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Flutterwave: {e}")

    if not resp.ok or data.get("status") != "success":
        raise HTTPException(status_code=502, detail=data.get("message", "Could not start Flutterwave payment."))

    return FlutterwaveInitResponse(
        payment_link=data["data"]["link"],
        tx_ref=tx_ref,
        coins=package["coins"],
    )


def _already_credited(admin, tx_ref: str) -> bool:
    existing = (
        admin.table("transactions")
        .select("id")
        .eq("type", "coin_purchase")
        .ilike("description", f"%{tx_ref}%")
        .limit(1)
        .execute()
    )
    return bool(existing.data)


def _verify_transaction(transaction_id) -> dict:
    """Re-checks the charge directly with Flutterwave rather than
    trusting the webhook payload alone — see the module docstring."""
    resp = requests.get(
        f"{FLUTTERWAVE_BASE_URL}/transactions/{transaction_id}/verify",
        headers={"Authorization": f"Bearer {FLUTTERWAVE_SECRET_KEY}"},
        timeout=15,
    )
    data = resp.json()
    if not resp.ok or data.get("status") != "success":
        raise HTTPException(status_code=502, detail="Could not verify Flutterwave transaction.")
    return data["data"]


@router.post("/coins/webhook/flutterwave")
async def flutterwave_webhook(request: Request, verif_hash: str = Header(None, alias="verif-hash")):
    if supabase_admin is None or not FLUTTERWAVE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Flutterwave payments are not configured.")
    if not FLUTTERWAVE_WEBHOOK_SECRET_HASH or verif_hash != FLUTTERWAVE_WEBHOOK_SECRET_HASH:
        raise HTTPException(status_code=400, detail="Invalid Flutterwave webhook signature.")

    event = await request.json()
    tx = event.get("data") or {}
    if event.get("event") != "charge.completed" or tx.get("status") != "successful":
        return {"received": True}

    meta = tx.get("meta") or tx.get("meta_data") or {}
    user_id = meta.get("user_id")
    package_id = meta.get("package_id")
    coins = meta.get("coins")
    tx_ref = tx.get("tx_ref")
    transaction_id = tx.get("id")

    if not user_id or not coins or not tx_ref or not transaction_id:
        return {"received": True}

    package = COIN_PACKAGES.get(package_id)
    if package is None:
        return {"received": True}

    try:
        if _already_credited(supabase_admin, tx_ref):
            return {"received": True}

        # Belt-and-suspenders: confirm the charge amount/currency/status
        # directly with Flutterwave before crediting anything.
        verified = _verify_transaction(transaction_id)
        if (
            verified.get("status") != "successful"
            or verified.get("tx_ref") != tx_ref
            or round(float(verified.get("amount", 0)) * 100) != package["usd_cents"]
            or verified.get("currency") != "USD"
        ):
            raise HTTPException(status_code=400, detail="Flutterwave transaction verification mismatch.")

        credit_coins(
            supabase_admin,
            user_id,
            int(coins),
            "coin_purchase",
            f"Purchased {package['label']} — Flutterwave {tx_ref}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not credit coins: {e}")

    return {"received": True}


@router.get("/coins/payment-complete")
async def payment_complete_page():
    """
    Where Flutterwave's hosted checkout redirects the browser after
    payment — a plain confirmation page, since this app has no
    registered deep link to hand the browser back to. Coins are
    already credited (or will be within moments) by the webhook above,
    independently of this page ever being viewed.
    """
    from fastapi.responses import HTMLResponse

    return HTMLResponse(
        "<html><body style='font-family: sans-serif; text-align: center; padding: 60px 20px;'>"
        "<h2>Thanks!</h2><p>You can close this window and return to the Viyo app — "
        "your coins will appear shortly.</p></body></html>"
    )
