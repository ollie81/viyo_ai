"""Fan donations: a viewer sends real money to a creator through NardoPay (never coins; coin gifting is gifting.py).

Rules this file keeps
  * Only a signed-in, non-guest viewer can donate, never to themselves, and only to an existing VIYO account.
  * Each donation gets its own NardoPay donation link (link_type "donation"; the donor chooses the amount on NardoPay's
    page). The exchange rate in use is locked on the donation when it starts.
  * A donation counts only when NardoPay's signed webhook for that link reports a completed payment in RWF with a
    positive amount (nardopay_payments.process_event routes it here by link code). The redirect back to VIYO proves
    nothing. Each donation and each NardoPay transaction is recorded once (nardopay_record_donation).
  * The creator receives CREATOR_DONATION_SHARE (80%) of the amount, converted at the locked rate into earnings coins
    (wallet.COIN_USD each, rounded down), as a creator_earnings row of kind 'donation', held for the holding period
    like every new earning, then withdrawn through the normal flow. VIYO keeps the rest.
  * Unknown until a real donation is tested (not in NardoPay's docs): whether NardoPay adds a fee on donation links and
    whether its webhook amount includes it. The share is computed on the amount NardoPay reports.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import nardopay_payments as np_

router = APIRouter(prefix="/api/v1", tags=["donations"])

CREATOR_DONATION_SHARE = 0.80
DONATION_REDIRECT_URL = "https://viyo.site/app"


def _ready() -> bool:
    try:
        np_.get_store().select("creator_donations", {}, limit=1)
        return True
    except Exception:
        return False


def donation_problem() -> Optional[str]:
    """Why a donation can't be started right now, or None."""
    if not np_._flag("NARDOPAY_ENABLED") or not np_._flag("NARDOPAY_LEDGER_VERIFIED"):
        return "switched off"
    if not np_._env("NARDOPAY_API_KEY") or np_.amount_multiplier() is None or np_.missing_contract(["link_url"]) \
            or np_.missing_contract() or not np_.contract_field("link_code") or not np_.webhook_url():
        return "not configured"
    if np_.rate_problem(np_.latest_rate()):
        return "rate unavailable"
    if not _ready():
        return "not set up"
    return None


class DonateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")      # no amount from the client: the donor chooses it on NardoPay
    message: str = Field("", max_length=200)


def _out(d: dict) -> dict:
    return {"donation_id": d["id"], "status": d["status"], "creator_id": d["creator_id"],
            "payment_link": d.get("payment_link") if d["status"] == "pending" else None,
            "amount_rwf": float(d["amount_rwf"]) if d.get("amount_rwf") is not None else None,
            "paid_at": d.get("paid_at")}


@router.get("/donations/availability")
async def availability():
    """Whether the app should show Donate (a yes/no only)."""
    return {"available": donation_problem() is None, "creator_share": CREATOR_DONATION_SHARE}


@router.post("/creators/{creator_id}/donate")
async def start_donation(creator_id: str, body: DonateIn, donor_id: str = Depends(np_._buyer)):
    if donation_problem():
        raise HTTPException(status_code=503, detail="Donations aren't available right now.")
    cid = np_._uuid_or_none(creator_id)
    if not cid:
        raise HTTPException(status_code=404, detail="Creator not found.")
    if cid == donor_id:
        raise HTTPException(status_code=400, detail="You can't donate to yourself.")
    store = np_.get_store()
    creator = store.select("profiles", {"id": cid}, limit=1)
    if not creator:
        raise HTTPException(status_code=404, detail="Creator not found.")
    name = (creator[0].get("display_name") or creator[0].get("username") or "a VIYO creator")[:60]
    rate = np_.latest_rate()
    d = store.insert("creator_donations", [{
        "creator_id": cid, "donor_id": donor_id, "message": body.message.strip(), "rate_id": rate["id"],
        "usd_rwf_rate": rate["rate"], "creator_share_rate": CREATOR_DONATION_SHARE}])[0]

    def fail(reason: str):
        store.update("creator_donations", {"id": d["id"]}, {"status": "failed", "failure_reason": reason[:400]})
        raise HTTPException(status_code=502, detail="NardoPay couldn't start the donation. Try again.")

    request_body = {"link_type": "donation", "title": f"Support {name} on VIYO",
                    "description": f"A donation to {name} on VIYO",
                    "metadata": {"donation_id": d["id"]}, "webhook_url": np_.webhook_url(),
                    "redirect_url": DONATION_REDIRECT_URL}
    try:
        resp = np_._http_post(np_.CREATE_LINK_URL, {"Authorization": f"Bearer {np_._env('NARDOPAY_API_KEY')}",
                                                    "Content-Type": "application/json"}, request_body)
    except Exception as e:
        print(f"[WARN] NardoPay donation link unreachable for {d['id']}: {type(e).__name__}")
        fail("unreachable")
    if not 200 <= resp.status_code < 300:
        print(f"[WARN] NardoPay donation link HTTP {resp.status_code} for {d['id']}")
        fail(f"http {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        fail("response is not JSON")
    link = np_.pick(data, np_.contract_field("link_url"))
    if not isinstance(link, str) or not link.startswith("https://") or len(link) > 2000:
        fail("no https link at the configured field")
    code = np_.link_code_of(link)
    if not code:
        fail("no link code in the donation link")          # payments are matched by it: never take one without it
    try:
        d = store.update("creator_donations", {"id": d["id"]}, {"payment_link": link, "link_code": code})[0]
    except Exception:
        fail("the link code is already used")
    return _out(d)


@router.get("/donations/{donation_id}")
async def donation_status(donation_id: str, donor_id: str = Depends(np_._buyer)):
    """The donor's own donation (the app checks it after NardoPay sends the donor back)."""
    did = np_._uuid_or_none(donation_id)
    d = np_.get_store().one("creator_donations", {"id": did}) if did else None
    if not d or d["donor_id"] != donor_id:
        raise HTTPException(status_code=404, detail="Donation not found.")
    return _out(d)


@router.get("/creator/donations")
async def my_donations(creator_id: str = Depends(np_._buyer)):
    """Donations a creator has received (paid only): who, what they wrote, and the creator's share."""
    rows = [r for r in np_.get_store().select("creator_donations", {"creator_id": creator_id}, order="paid_at",
                                              desc=True, limit=200) if r.get("status") == "paid"]
    from notify import display_name
    return {"creator_share": CREATOR_DONATION_SHARE, "donations": [
        {"donation_id": r["id"], "from": display_name(r["donor_id"]), "message": r.get("message") or "",
         "share_usd": float(r.get("share_usd") or 0), "paid_at": r.get("paid_at")} for r in rows[:50]]}


# ---------------------------------------------------------------- the webhook (called from nardopay_payments)

def find_by_link_code(code: str) -> Optional[dict]:
    if not _ready():
        return None
    return np_.get_store().one("creator_donations", {"link_code": code})


def process_payment(d: dict, ev: dict, eid: str, finish) -> tuple[int, dict]:
    """A signed webhook for one of our donation links."""
    if ev["event_type"] == np_.EVENT_FAILED:
        if d["status"] == "pending":
            np_.get_store().update("creator_donations", {"id": d["id"]},
                                   {"status": "failed", "failure_reason": "payment.failed"})
        return finish("processed", "donation payment failed")
    if ev["event_type"] != np_.EVENT_COMPLETED:
        return finish("ignored", f"event type {ev['event_type']!r} isn't handled")
    if not ev.get("transaction_id"):
        return finish("mismatch", "no transaction id")
    if ev.get("currency") != "RWF":
        return finish("mismatch", f"currency {ev.get('currency')!r} is not RWF")
    unit = np_.amount_multiplier()
    if unit is None or ev.get("amount") is None or ev["amount"] <= 0:
        return finish("mismatch", "no positive amount")
    import wallet
    amount = ev["amount"] / unit
    out = np_._rpc("nardopay_record_donation", {
        "p_donation": str(d["id"]), "p_transaction": ev["transaction_id"], "p_event": eid, "p_amount": str(amount),
        "p_currency": ev["currency"], "p_hold_days": wallet.EARNINGS_HOLD_DAYS,
        "p_coin_usd": str(wallet.COIN_USD)}) or {}
    result = out.get("result")
    if result == "duplicate":
        return finish("processed", "donation already recorded")
    if result in ("other_transaction", "transaction_used_elsewhere"):
        print(f"[WARN] NardoPay duplicate donation payment: {d['id']}, transaction {ev['transaction_id']}")
        return finish("duplicate_payment", f"{result}: not recorded again; refund it in NardoPay")
    if result != "recorded":
        raise RuntimeError(f"donation not recorded ({result})")
    try:
        from notify import display_name, notify
        notify(str(out["creator_id"]), "donation",
               f"{display_name(str(out['donor_id']))} sent you a donation. You earned "
               f"${float(out.get('share_usd') or 0):.2f}.",
               actor_id=str(out["donor_id"]), entity_id=d["id"], push_title="New donation 💛")
    except Exception as e:
        print(f"[WARN] donation notification failed: {e}")
    return finish("processed", "donation recorded")
