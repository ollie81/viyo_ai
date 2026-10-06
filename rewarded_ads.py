"""
Crediting Viyo Coins for watching a rewarded video ad.

Coins are credited the moment this endpoint is called — there is no
AdMob Server-Side Verification (SSV) wired up, since that needs a
public callback URL registered in the AdMob console for this app's
actual ad units, which isn't something this codebase can set up on its
own (see rewarded_ad_service.dart's own doc comment). Without SSV, a
determined client could call this endpoint without actually watching
an ad, so the only defense is the DAILY_AD_CAP below, enforced
server-side the same "don't trust a client-side cap" way episodes.py
enforces the paywall.

Same "dated row in the existing transactions ledger, no new
table/migration needed" trick coins.py's free-taste window uses for
counting today's usage.
"""
import datetime
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

from coins import credit_coins

router = APIRouter(prefix="/api/v1", tags=["rewarded_ads"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# How many coins one watched ad is worth, and how many times a day any
# one viewer can claim that reward. Keep in sync with
# rewarded_ad_service.dart's own doc comment — no single source of
# truth across the two repos, same tradeoff as every other
# cross-repo-mirrored constant in this backend (FREE_EPISODE_COUNT,
# FEATURE_COSTS, ...).
REWARDED_AD_COINS = 10
# Raised from 5 — coin purchases and subscriptions are both unavailable
# on Android (Google Play's anti-steering policy, see buy_coins_screen's
# own module comment), so ads are that platform's only way to earn
# coins at all. Still capped, not removed: there's no AdMob Server-Side
# Verification wired up (see this file's own module docstring), so this
# is the only defense against claiming the reward without ever
# watching an ad — a real SSV integration would be needed to lift this
# safely to "unlimited".
DAILY_AD_CAP = 30


def _today_start_utc() -> datetime.datetime:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


class RewardedAdClaimResponse(BaseModel):
    coins_earned: int
    claims_today: int
    daily_cap: int


@router.post("/coins/rewarded-ad/claim", response_model=RewardedAdClaimResponse)
async def claim_rewarded_ad(user_id: str = Depends(_get_current_user_id_no_guest)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Coin service is not configured.")

    try:
        today = (
            supabase_admin.table("transactions")
            .select("id")
            .eq("user_id", user_id)
            .eq("type", "rewarded_ad")
            .gte("created_at", _today_start_utc().isoformat())
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check today's ad claims: {e}")

    claims_today = len(today.data or [])
    if claims_today >= DAILY_AD_CAP:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_ad_cap_reached",
                "claims_today": claims_today,
                "daily_cap": DAILY_AD_CAP,
            },
        )

    credit_coins(
        supabase_admin, user_id, REWARDED_AD_COINS, "rewarded_ad",
        "Watched a rewarded ad",
    )

    return RewardedAdClaimResponse(
        coins_earned=REWARDED_AD_COINS,
        claims_today=claims_today + 1,
        daily_cap=DAILY_AD_CAP,
    )
