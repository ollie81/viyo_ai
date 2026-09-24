"""
Viyo Coin purchases — real-money top-ups via Google Play Billing, for
the Android app specifically. Google Play's Developer Program Policy
requires any digital good consumed inside an app distributed through
Google Play (coins that unlock episodes, here) to be sold through Play
Billing itself — Stripe/Paystack/Flutterwave/Lemon Squeezy (payments.py,
paystack_payments.py, flutterwave_payments.py, lemonsqueezy_payments.py)
stay exactly as they are for the web build, which isn't distributed
through Google Play and isn't subject to that rule.

Unlike every other provider here, there's no "initialize checkout" step
— the Flutter client starts the Play Billing purchase flow itself (the
in_app_purchase plugin talking directly to Google Play on-device) and
only calls this backend once *after* Google has already processed the
purchase, to verify it server-side and credit coins. Never trust the
client's own claim that a purchase succeeded: this re-checks the
purchase token directly against Google's Android Publisher API before
crediting anything, then acknowledges it — Play auto-refunds any
purchase left unacknowledged for 3 days, so skipping that step would
silently claw back every purchase a few days later.

Coin package ids (starter/popular/value/creator — see payments.py's
COIN_PACKAGES) are expected to also be the exact Google Play product
ids configured as managed in-app products in Play Console; there's no
separate mapping table since keeping the two identical is simpler than
maintaining one.
"""
import json
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from pydantic import BaseModel, Field
from supabase import create_client, Client

from coins import credit_coins
from payments import COIN_PACKAGES

router = APIRouter(prefix="/api/v1", tags=["payments"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
GOOGLE_PLAY_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", "")
# The Android app's applicationId (android/app/build.gradle) — the
# Android Publisher API is scoped per package name, not per credential.
GOOGLE_PLAY_PACKAGE_NAME = os.environ.get("GOOGLE_PLAY_PACKAGE_NAME", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

_ANDROID_PUBLISHER_SCOPE = "https://www.googleapis.com/auth/androidpublisher"
_credentials: Optional[service_account.Credentials] = None
if GOOGLE_PLAY_SERVICE_ACCOUNT_JSON:
    try:
        _credentials = service_account.Credentials.from_service_account_info(
            json.loads(GOOGLE_PLAY_SERVICE_ACCOUNT_JSON), scopes=[_ANDROID_PUBLISHER_SCOPE]
        )
    except Exception as e:
        print(f"[WARN] Could not load Google Play service account: {e}")


def _configured() -> bool:
    return bool(supabase_admin is not None and _credentials is not None and GOOGLE_PLAY_PACKAGE_NAME)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


class GooglePlayVerifyRequest(BaseModel):
    package_id: str = Field(..., min_length=1)
    purchase_token: str = Field(..., min_length=1)


class GooglePlayVerifyResponse(BaseModel):
    coins: int


def _already_credited(admin, purchase_token: str) -> bool:
    existing = (
        admin.table("transactions")
        .select("id")
        .eq("type", "coin_purchase")
        .ilike("description", f"%{purchase_token}%")
        .limit(1)
        .execute()
    )
    return bool(existing.data)


def _verify_and_acknowledge(product_id: str, purchase_token: str) -> None:
    """Confirms the purchase directly with Google — never trust the
    client's own claim that Play Billing succeeded — and acknowledges
    it (see the module docstring for why that step can't be skipped)."""
    session = AuthorizedSession(_credentials)
    base_url = (
        "https://androidpublisher.googleapis.com/androidpublisher/v3/applications/"
        f"{GOOGLE_PLAY_PACKAGE_NAME}/purchases/products/{product_id}/tokens/{purchase_token}"
    )
    try:
        resp = session.get(base_url, timeout=15)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Google Play: {e}")
    if not resp.ok:
        raise HTTPException(status_code=400, detail="Could not verify this Google Play purchase.")

    data = resp.json()
    # purchaseState: 0 = purchased, 1 = canceled, 2 = pending.
    if data.get("purchaseState") != 0:
        raise HTTPException(status_code=400, detail="This purchase has not completed successfully.")

    # acknowledgementState: 0 = not yet acknowledged, 1 = acknowledged.
    if data.get("acknowledgementState") == 0:
        try:
            ack = session.post(f"{base_url}:acknowledge", timeout=15)
            if not ack.ok:
                print(f"[WARN] Could not acknowledge Google Play purchase {purchase_token}: {ack.text}")
        except Exception as e:
            print(f"[WARN] Could not acknowledge Google Play purchase {purchase_token}: {e}")


@router.post("/coins/purchase/google-play/verify", response_model=GooglePlayVerifyResponse)
async def verify_google_play_purchase(
    req: GooglePlayVerifyRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    if not _configured():
        raise HTTPException(status_code=503, detail="Google Play payments are not configured.")

    package = COIN_PACKAGES.get(req.package_id)
    if package is None:
        raise HTTPException(status_code=400, detail="Unknown coin package.")

    # Idempotent: the client may retry this call (e.g. a dropped
    # response after Google already confirmed the purchase) without
    # ever double-crediting.
    if _already_credited(supabase_admin, req.purchase_token):
        return GooglePlayVerifyResponse(coins=package["coins"])

    _verify_and_acknowledge(req.package_id, req.purchase_token)

    credit_coins(
        supabase_admin,
        user_id,
        package["coins"],
        "coin_purchase",
        f"Purchased {package['label']} — Google Play {req.purchase_token}",
    )
    return GooglePlayVerifyResponse(coins=package["coins"])
