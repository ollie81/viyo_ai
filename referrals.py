"""
Referral signup bonus — both the new signup and whoever referred them
get a one-time coin credit once the new profile exists with a
`referred_by` on file.

Routed through this backend (service-role client) rather than a direct
client call, unlike every other `profiles` write in this app: this
credits a coin balance belonging to a DIFFERENT user than the caller
(the referrer) — the same "backend only for cross-user money movement"
line drawn everywhere else in this codebase (gifting.py,
episodes.py's creator_earnings). The client-side referral code lookup
that resolves a code to a referrer id stays a direct, read-only
Supabase query (see AuthService.createProfile) — only the actual coin
minting needs the service-role client.

No debit anywhere — these coins are minted, not moved between two
users' balances, the same "credit_coins with no matching debit" shape
payments.py uses after a real-money purchase. A referral bonus is a
growth cost the app is choosing to pay, same idea as a promotional
discount.
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

from coins import credit_coins

router = APIRouter(prefix="/api/v1", tags=["referrals"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Mirrored in invite_screen.dart's own copy ("+20 coins" / "+10 coins")
# — no single source of truth across the two repos, same tradeoff as
# every other coin amount in this app.
REFERRER_BONUS = 20
REFEREE_BONUS = 10


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


class GrantReferralBonusRequest(BaseModel):
    referrer_id: str


class GrantReferralBonusResponse(BaseModel):
    referrer_bonus: int
    referee_bonus: int


@router.post("/referrals/grant-bonus", response_model=GrantReferralBonusResponse)
async def grant_referral_bonus(
    req: GrantReferralBonusRequest, user_id: str = Depends(_get_current_user_id_no_guest)
):
    """
    Called once, right after a new profile is created with a referrer
    attached (see AuthService.createProfile) — user_id is the new
    signup (the caller, now authenticated), req.referrer_id is whoever
    referred them, already resolved client-side from their referral
    code before this call.
    """
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Referral bonus service is not configured.")

    if req.referrer_id == user_id:
        raise HTTPException(status_code=400, detail="Cannot refer yourself.")

    # Never trust a client-supplied referrer id alone for a coin-minting
    # action — require that profiles.referred_by already points to
    # exactly this referrer (set atomically with the profile's own
    # creation) before crediting anyone.
    try:
        profile = (
            supabase_admin.table("profiles")
            .select("id,referred_by")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load profile: {e}")

    rows = profile.data or []
    if not rows or rows[0].get("referred_by") != req.referrer_id:
        raise HTTPException(status_code=400, detail="No matching referral on file for this account.")

    # One-shot only — a transactions row IS the state, the same
    # "derive from the ledger, no new table" trick Discover Spotlight
    # and coins.py's free-taste tracking already use, so this can't be
    # replayed to keep minting coins for the same signup.
    try:
        existing = (
            supabase_admin.table("transactions")
            .select("id")
            .eq("user_id", user_id)
            .eq("type", "referral_bonus_referee")
            .limit(1)
            .execute()
        )
        if existing.data:
            raise HTTPException(status_code=400, detail="Referral bonus already granted for this account.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check referral bonus status: {e}")

    # Two independent mints, not a paired debit/credit — nothing to
    # roll back if the second one fails. The referee's own bonus is the
    # one this call is actually gating (checked above), so it goes
    # first; a lost referrer credit is logged loudly for manual
    # reconciliation rather than swallowed, same posture episodes.py
    # takes for a lost creator_earnings credit.
    try:
        credit_coins(supabase_admin, user_id, REFEREE_BONUS, "referral_bonus_referee", "Referral bonus — welcome to Viyo")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not grant referral bonus: {e}")

    try:
        credit_coins(
            supabase_admin, req.referrer_id, REFERRER_BONUS, "referral_bonus_referrer",
            "Referral bonus — your invite joined",
        )
    except Exception as e:
        print(
            f"[CRITICAL] Could not credit {REFERRER_BONUS} coins to referrer "
            f"{req.referrer_id} for new signup {user_id}: {e}"
        )

    return GrantReferralBonusResponse(referrer_bonus=REFERRER_BONUS, referee_bonus=REFEREE_BONUS)
