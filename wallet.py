"""
Creator wallet: earnings from episode unlocks (see episodes.py), kept
entirely separate from the general Viyo Coin spending balance
(`profiles.points_balance`). Earnings are real money owed to a
creator, not coins they can spend in-app — the whole point of a
withdrawal request is turning them into an actual payout, so crediting
them straight into points_balance instead would let a creator "cash
out" by just spending them, with no payout ever happening.

Withdrawals are manual for now (per the product brief): a request here
only marks the underlying creator_earnings rows as pending and records
a withdrawal_requests row for a human to action outside the app. No
payout integration yet.
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["wallet"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    from supabase import create_client
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Below this, a withdrawal request isn't worth the manual processing
# overhead. Purely a product choice, not a platform limit — easy to
# lower later.
MIN_WITHDRAWAL_COINS = 100


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


class WalletEarningsResponse(BaseModel):
    lifetime_coins: int
    available_coins: int
    pending_withdrawal_coins: int
    paid_coins: int
    min_withdrawal_coins: int = MIN_WITHDRAWAL_COINS


class WithdrawResponse(BaseModel):
    requested: bool
    coins: int


def _earnings_totals(creator_id: str) -> dict:
    try:
        result = (
            supabase_admin.table("creator_earnings")
            .select("coins,status")
            .eq("creator_id", creator_id)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load earnings: {e}")

    totals = {"available": 0, "pending_withdrawal": 0, "paid": 0}
    for row in rows:
        status = row.get("status") or "available"
        totals[status] = totals.get(status, 0) + int(row.get("coins") or 0)
    return totals


@router.get("/wallet/earnings", response_model=WalletEarningsResponse)
async def get_wallet_earnings(user_id: str = Depends(_get_current_user_id_no_guest)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Wallet service is not configured.")

    totals = _earnings_totals(user_id)
    return WalletEarningsResponse(
        lifetime_coins=sum(totals.values()),
        available_coins=totals["available"],
        pending_withdrawal_coins=totals["pending_withdrawal"],
        paid_coins=totals["paid"],
    )


@router.post("/wallet/withdraw", response_model=WithdrawResponse)
async def request_withdrawal(user_id: str = Depends(_get_current_user_id_no_guest)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Wallet service is not configured.")

    try:
        available = (
            supabase_admin.table("creator_earnings")
            .select("id,coins")
            .eq("creator_id", user_id)
            .eq("status", "available")
            .execute()
        )
        rows = available.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load available earnings: {e}")

    total = sum(int(r.get("coins") or 0) for r in rows)
    if total < MIN_WITHDRAWAL_COINS:
        raise HTTPException(
            status_code=400,
            detail=f"Minimum withdrawal is {MIN_WITHDRAWAL_COINS} coins — you have {total} available.",
        )

    ids = [r["id"] for r in rows]

    try:
        supabase_admin.table("withdrawal_requests").insert({
            "creator_id": user_id,
            "coins": total,
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not create withdrawal request: {e}")

    # Marked pending_withdrawal AFTER the request row exists, so a
    # failure here never leaves a withdrawal request with nothing
    # backing it — worst case some rows stay "available" and get swept
    # into the creator's next withdrawal request instead of this one.
    try:
        (
            supabase_admin.table("creator_earnings")
            .update({"status": "pending_withdrawal"})
            .in_("id", ids)
            .execute()
        )
    except Exception as e:
        print(f"[WARN] Withdrawal request created but earnings rows not marked pending for {user_id}: {e}")

    return WithdrawResponse(requested=True, coins=total)
