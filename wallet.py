"""
Creator wallet: earnings from episode and movie unlocks (see episodes.py),
kept entirely separate from the general Viyo Coin spending balance
(`profiles.points_balance`). Earnings are real money owed to a creator,
not coins they can spend in-app.

Only BOUGHT coins earn cash: when a viewer unlocks a title, free coins
(check-ins, missions, ads, gifts) are spent first, and the creator's share
is taken only from the bought part (see paid_portion and
creator_payouts_schema.sql). So Viyo never pays out money it didn't take in.

Payouts are manual: a creator saves where to be paid (PayPal or mobile
money) and requests a withdrawal; the Viyo team sends the money and marks
the request paid from the admin screen (X-Admin-Key, like other admin tools).
"""
import datetime
import math
import os
import re
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["wallet"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    from supabase import create_client
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# What one earned coin is worth in cash. Below the cheapest coin price
# (3,000 coins for $19.99 = $0.00666), so payouts never exceed what came in.
COIN_USD = 0.006
# Smallest payout worth sending (PayPal and mobile money fees).
MIN_WITHDRAWAL_USD = 10.0
MIN_WITHDRAWAL_COINS = math.ceil(MIN_WITHDRAWAL_USD / COIN_USD)   # 1,667

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")


def coins_to_usd(coins: float) -> float:
    return round(float(coins) * COIN_USD, 2)


def paid_portion(price: float, balance: float, paid: float) -> float:
    """How much of a `price`-coin spend comes out of BOUGHT coins, when free
    coins are spent first. `balance` and `paid` are read before the spend."""
    free = max(float(balance) - max(float(paid), 0.0), 0.0)
    return max(0.0, min(float(price) - free, float(paid), float(price)))


_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
_PHONE = re.compile(r"^\+?[0-9]{8,15}$")


def _require_admin(x_admin_key: str = Header(None)) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin tools are not configured.")
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


class PayoutSettings(BaseModel):
    method: Literal["paypal", "mobile_money"]
    paypal_email: Optional[str] = Field(None, max_length=200)
    momo_phone: Optional[str] = Field(None, max_length=20)
    momo_network: Optional[str] = Field(None, max_length=40)
    account_name: Optional[str] = Field(None, max_length=80)
    country: Optional[str] = Field(None, max_length=60)


class PendingWithdrawal(BaseModel):
    usd_amount: float
    coins: int
    payout_destination: Optional[str] = None
    created_at: Optional[str] = None


class WalletEarningsResponse(BaseModel):
    lifetime_coins: int
    available_coins: int
    pending_withdrawal_coins: int
    paid_coins: int
    min_withdrawal_coins: int = MIN_WITHDRAWAL_COINS
    coin_usd: float = COIN_USD
    min_withdrawal_usd: float = MIN_WITHDRAWAL_USD
    available_usd: float = 0
    pending_withdrawal_usd: float = 0
    paid_usd: float = 0
    payout_settings: Optional[PayoutSettings] = None
    pending_request: Optional[PendingWithdrawal] = None


class WithdrawResponse(BaseModel):
    requested: bool
    coins: int
    usd_amount: float = 0
    payout_destination: Optional[str] = None


def _clean_settings(req: PayoutSettings) -> dict:
    """Validated, normalised payout details, or a 400 explaining what's wrong."""
    name = (req.account_name or "").strip()
    if req.method == "paypal":
        email = (req.paypal_email or "").strip().lower()
        if not _EMAIL.match(email):
            raise HTTPException(status_code=400, detail="Enter the email address of your PayPal account.")
        return {"method": "paypal", "paypal_email": email, "momo_phone": None, "momo_network": None,
                "account_name": name or None, "country": (req.country or "").strip() or None}
    phone = re.sub(r"[\s\-()]", "", req.momo_phone or "")
    if not _PHONE.match(phone):
        raise HTTPException(status_code=400, detail="Enter your mobile money number with the country code, e.g. +250 78 123 4567.")
    network = (req.momo_network or "").strip()
    if not network:
        raise HTTPException(status_code=400, detail="Choose your mobile money network, e.g. MTN or Airtel.")
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Enter the name registered on the mobile money account.")
    return {"method": "mobile_money", "paypal_email": None, "momo_phone": phone, "momo_network": network,
            "account_name": name, "country": (req.country or "").strip() or None}


def describe_destination(settings: dict) -> str:
    if settings.get("method") == "paypal":
        return f"PayPal: {settings.get('paypal_email')}"
    return (f"Mobile money ({settings.get('momo_network')}): {settings.get('momo_phone')}"
            f" — {settings.get('account_name')}")


def _load_settings(user_id: str) -> Optional[dict]:
    try:
        res = supabase_admin.table("creator_payout_settings").select("*").eq("user_id", user_id).limit(1).execute()
    except Exception as e:
        print(f"[WARN] payout settings unavailable for {user_id}: {e}")
        return None
    rows = res.data or []
    return rows[0] if rows else None


def _pending_request(user_id: str) -> Optional[dict]:
    try:
        res = (supabase_admin.table("withdrawal_requests").select("*").eq("creator_id", user_id)
               .eq("status", "pending").order("created_at", desc=True).limit(1).execute())
    except Exception as e:
        print(f"[WARN] withdrawal requests unavailable for {user_id}: {e}")
        return None
    rows = res.data or []
    return rows[0] if rows else None


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
    settings = _load_settings(user_id)
    pending = _pending_request(user_id)
    return WalletEarningsResponse(
        lifetime_coins=sum(totals.values()),
        available_coins=totals["available"],
        pending_withdrawal_coins=totals["pending_withdrawal"],
        paid_coins=totals["paid"],
        available_usd=coins_to_usd(totals["available"]),
        pending_withdrawal_usd=coins_to_usd(totals["pending_withdrawal"]),
        paid_usd=coins_to_usd(totals["paid"]),
        payout_settings=PayoutSettings(**{k: settings.get(k) for k in PayoutSettings.model_fields}) if settings else None,
        pending_request=PendingWithdrawal(
            usd_amount=float(pending.get("usd_amount") or coins_to_usd(pending.get("coins") or 0)),
            coins=int(pending.get("coins") or 0), payout_destination=pending.get("payout_destination"),
            created_at=pending.get("created_at")) if pending else None,
    )


@router.put("/wallet/payout-settings", response_model=PayoutSettings)
async def save_payout_settings(req: PayoutSettings, user_id: str = Depends(_get_current_user_id_no_guest)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Wallet service is not configured.")
    row = _clean_settings(req)
    try:
        supabase_admin.table("creator_payout_settings").upsert(
            {"user_id": user_id, **row, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            on_conflict="user_id").execute()
    except Exception as e:
        print(f"[WARN] could not save payout settings for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Couldn't save your payout details. Try again.")
    return PayoutSettings(**row)


def _rpc(name: str, params: dict) -> dict:
    """Calls one of the payout database functions (creator_payouts_safety.sql). Each runs as a single locked
    transaction, so if it fails nothing has changed. If the function isn't installed yet, withdrawals stop
    (HTTP 503) rather than falling back to the old, non-atomic steps."""
    try:
        data = supabase_admin.rpc(name, params).execute().data
    except Exception as e:
        text = str(e)
        if name in text and ("PGRST202" in text or "Could not find" in text or "does not exist" in text):
            print(f"[ERROR] {name} is missing: run creator_payouts_safety.sql")
            raise HTTPException(status_code=503, detail="Withdrawals are paused for an upgrade. Try again later.")
        print(f"[ERROR] {name} failed: {text[:300]}")
        raise HTTPException(status_code=500, detail="Couldn't complete this right now. Nothing was changed. Try again.")
    if isinstance(data, list):
        data = data[0] if data else {}
    return data or {}


@router.post("/wallet/withdraw", response_model=WithdrawResponse)
async def request_withdrawal(user_id: str = Depends(_get_current_user_id_no_guest)):
    """Reserves all of the creator's available earnings for one withdrawal, atomically (viyo_request_withdrawal):
    two requests at the same moment can't reserve the same earnings, and a failure leaves nothing reserved."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Wallet service is not configured.")
    settings = _load_settings(user_id)
    if not settings:
        raise HTTPException(status_code=400, detail="Add your payout details first (PayPal or mobile money).")
    destination = describe_destination(settings)
    r = _rpc("viyo_request_withdrawal", {"p_creator": user_id, "p_min_coins": MIN_WITHDRAWAL_COINS,
                                         "p_coin_usd": COIN_USD, "p_method": settings.get("method"),
                                         "p_destination": destination})
    if not r.get("ok"):
        if r.get("error") == "pending_exists":
            raise HTTPException(status_code=409, detail="You already have a withdrawal on its way. You can request "
                                                        "another once it's paid.")
        if r.get("error") == "below_minimum":
            total = int(r.get("coins") or 0)
            raise HTTPException(
                status_code=400,
                detail=(f"You can withdraw from ${MIN_WITHDRAWAL_USD:.0f} ({MIN_WITHDRAWAL_COINS:,} coins). "
                        f"You have ${coins_to_usd(total):.2f} ({total:,} coins)."))
        raise HTTPException(status_code=500, detail="Couldn't request the withdrawal. Nothing was changed. Try again.")
    return WithdrawResponse(requested=True, coins=int(r["coins"]), usd_amount=float(r["usd_amount"]),
                            payout_destination=destination)


# ---------------------------------------------------------------- admin: pay creators

class AdminWithdrawal(BaseModel):
    id: str
    creator_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    coins: int
    usd_amount: float
    payout_method: Optional[str] = None
    payout_destination: Optional[str] = None
    status: str
    created_at: Optional[str] = None
    paid_at: Optional[str] = None
    admin_note: Optional[str] = None


class AdminNote(BaseModel):
    note: Optional[str] = Field(None, max_length=300)


@router.get("/admin/withdrawals", response_model=list[AdminWithdrawal], dependencies=[Depends(_require_admin)])
async def admin_list_withdrawals(status: Literal["pending", "paid", "rejected"] = "pending"):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Wallet service is not configured.")
    try:
        rows = (supabase_admin.table("withdrawal_requests").select("*").eq("status", status)
                .order("created_at", desc=(status != "pending")).limit(200).execute()).data or []
        ids = list({r["creator_id"] for r in rows})
        profiles = {}
        if ids:
            for p in (supabase_admin.table("profiles").select("id,username,display_name").in_("id", ids).execute()).data or []:
                profiles[p["id"]] = p
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load withdrawals: {e}")
    return [AdminWithdrawal(
        id=str(r["id"]), creator_id=str(r["creator_id"]),
        username=(profiles.get(r["creator_id"]) or {}).get("username"),
        display_name=(profiles.get(r["creator_id"]) or {}).get("display_name"),
        coins=int(r.get("coins") or 0),
        usd_amount=float(r.get("usd_amount") or coins_to_usd(r.get("coins") or 0)),
        payout_method=r.get("payout_method"), payout_destination=r.get("payout_destination"),
        status=r.get("status") or "pending", created_at=r.get("created_at"), paid_at=r.get("paid_at"),
        admin_note=r.get("admin_note")) for r in rows]


def _settle(request_id: str, action: str, note: Optional[str]) -> dict:
    """Marks a withdrawal paid or rejected in one locked transaction (viyo_settle_withdrawal). Repeating the same
    action changes nothing and notifies no one again; the opposite action after it is refused; rejected earnings
    return to available exactly once."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Wallet service is not configured.")
    import uuid as _uuid
    try:
        rid = str(_uuid.UUID(str(request_id)))
    except ValueError:
        raise HTTPException(status_code=404, detail="Withdrawal not found.")
    r = _rpc("viyo_settle_withdrawal", {"p_request": rid, "p_action": action, "p_note": note})
    if not r.get("ok"):
        err = r.get("error")
        if err == "not_found":
            raise HTTPException(status_code=404, detail="Withdrawal not found.")
        if err in ("already_paid", "already_rejected"):
            raise HTTPException(status_code=409, detail=f"This withdrawal is already {err.split('_', 1)[1]}.")
        if err == "earnings_mismatch":
            print(f"[ERROR] withdrawal {rid}: reserved earnings ({r.get('reserved_coins')}) don't match the request "
                  f"({r.get('request_coins')}); not settled")
            raise HTTPException(status_code=409, detail="This withdrawal's reserved earnings don't match its amount. "
                                                        "It needs a manual check before it can be paid or rejected.")
        raise HTTPException(status_code=500, detail="Couldn't update the withdrawal. Nothing was changed.")
    if r.get("changed"):
        try:
            from notify import notify
            usd = float(r.get("usd_amount") or coins_to_usd(r.get("coins") or 0))
            if action == "paid":
                notify(str(r["creator_id"]), "payout_paid",
                       f"Your ${usd:.2f} payout was sent to {r.get('payout_destination') or 'you'}.",
                       entity_id=rid, push_title="Payout sent 💸", admin=supabase_admin)
            else:
                reason = f" Note: {note}" if note else ""
                notify(str(r["creator_id"]), "payout_rejected",
                       f"Your ${usd:.2f} withdrawal couldn't be sent. Check your payout details and request again.{reason}",
                       entity_id=rid, push_title="Withdrawal not sent", admin=supabase_admin)
        except Exception as e:
            print(f"[WARN] payout notification failed for {rid}: {e}")
    return {"id": rid, "status": r["status"], "changed": bool(r.get("changed"))}


@router.post("/admin/withdrawals/{request_id}/paid", dependencies=[Depends(_require_admin)])
async def admin_mark_paid(request_id: str, body: AdminNote = AdminNote()):
    """Call after sending the money in PayPal or mobile money."""
    return _settle(request_id, "paid", body.note)


@router.post("/admin/withdrawals/{request_id}/reject", dependencies=[Depends(_require_admin)])
async def admin_reject(request_id: str, body: AdminNote = AdminNote()):
    """Cancels a request (e.g. wrong payout details); the earnings go back to available."""
    return _settle(request_id, "rejected", body.note)
