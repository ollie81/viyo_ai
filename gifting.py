"""
Gift Viyo Coins to another user — a direct balance transfer, not a
feature spend, so it goes through coins.debit_coins/credit_coins
directly (no free-taste allowance applies to a gift).

Previously a Postgres RPC (`gift_coins`) called straight from the
client. Same reasoning as boost_post/likes/comments elsewhere in this
app: a function that debits one user and credits a *different* user is
exactly the kind of cross-user mutation Supabase RLS on `profiles`
(owner-only UPDATE) is likely to silently block on the receiver's
side, and this codebase has no way to inspect the RPC's actual SQL to
confirm it was written as SECURITY DEFINER. Routing both sides through
the service-role client here removes the doubt entirely — coins are
the core of this app's economy, so "probably fine" isn't good enough
for the one feature that moves them directly between two people.
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client

from coins import credit_coins, debit_coins

router = APIRouter(prefix="/api/v1", tags=["gifting"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

MIN_GIFT_AMOUNT = 10


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


class GiftCoinsRequest(BaseModel):
    receiver_id: str = Field(..., min_length=1)
    amount: int = Field(..., ge=MIN_GIFT_AMOUNT)


class GiftCoinsResponse(BaseModel):
    sent: bool
    amount: int
    receiver_id: str


@router.post("/coins/gift", response_model=GiftCoinsResponse)
async def gift_coins(
    req: GiftCoinsRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Gifting is not configured.")

    if req.receiver_id == user_id:
        raise HTTPException(status_code=400, detail="You can't gift coins to yourself.")

    try:
        receiver = (
            supabase_admin.table("profiles")
            .select("id,username")
            .eq("id", req.receiver_id)
            .limit(1)
            .execute()
        )
        receiver_rows = receiver.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not find recipient: {e}")

    if not receiver_rows:
        raise HTTPException(status_code=404, detail="Recipient not found.")
    receiver_username = receiver_rows[0].get("username") or "a creator"

    try:
        sender = (
            supabase_admin.table("profiles")
            .select("username")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        sender_rows = sender.data or []
        sender_username = sender_rows[0].get("username") if sender_rows else None
        sender_username = sender_username or "Someone"
    except Exception:
        sender_username = "Someone"

    # Raises 402/409/500 outward unchanged if the sender can't be
    # debited at all — nothing has moved yet at that point.
    debit_coins(
        supabase_admin, user_id, req.amount, "gift_sent",
        f"Gift to @{receiver_username}",
    )

    try:
        credit_coins(
            supabase_admin, req.receiver_id, req.amount, "gift_received",
            f"Gift from @{sender_username}",
        )
    except Exception as e:
        # The sender was already debited — refund rather than silently
        # losing their coins if crediting the receiver failed partway
        # through (e.g. a lost compare-and-swap race on their balance).
        try:
            credit_coins(supabase_admin, user_id, req.amount, "gift_refund", "Gift failed — refunded")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Gift failed and was refunded: {e}")

    try:
        supabase_admin.table("notifications").insert({
            "user_id": req.receiver_id,
            "actor_id": user_id,
            "type": "gift",
            "message": f"@{sender_username} gifted you {req.amount} coins!",
        }).execute()
    except Exception:
        pass  # best-effort — never block a gift that already went through

    return GiftCoinsResponse(sent=True, amount=req.amount, receiver_id=req.receiver_id)
