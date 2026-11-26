"""
Viyo Coin purchases on the WEB through NardoPay. A separate module: Stripe, Paystack, Flutterwave, Lemon Squeezy and
Google Play (the Android flow) are untouched. Plan: VIYO_NardoPay_Integration_Plan_v3; tables and atomic steps:
nardopay_orders.sql.

Rules
  * The client sends a package id and nothing else. Price and coins come from payments.COIN_PACKAGES; the RWF
    amount from the USD/RWF rate in use (BNR's daily published average, fetched automatically by nardopay_fx, or a
    manual admin override), locked on the order with its id so a later rate never changes it.
  * Coins are credited only by a signed webhook (or reconciliation of a payment already verified from one), never
    by a redirect or anything the client says, and only through coins.credit_coins with external_ref
    "nardopay:<transaction id>" (the atomic viyo_coin_apply ledger).
  * Before crediting: valid signature (hex HMAC-SHA256 of the exact raw body), the payment's order exists (found
    by the paid link's code, saved on the order when the link was made), amount equals the order's RWF amount, currency is RWF, a transaction id is present, and the
    order is claimed for that transaction (one order, one transaction, ever).

Undocumented details are never guessed
  The NardoPay field NAMES used to read the create-link response and the webhook body are taken from settings filled
  in from NardoPay's documentation (CONTRACT below), not written into this file. While any required one is missing,
  orders aren't created and nothing is credited; signed webhooks are still stored and can be reprocessed once they
  are set. The unit of `amount` (whole francs or minor units) is a setting too.

Where NardoPay reports payments
  NardoPay only sends a webhook for a link created through the API with a webhook_url (documented in its dashboard,
  Developers > Webhooks). NARDOPAY_WEBHOOK_URL is that address (this server's /api/v1/webhooks/nardopay, https);
  without it no link is created, because no payment could ever be confirmed.

Switches (all off by default: nothing goes live by deploying this)
  NARDOPAY_ENABLED=true         allow creating orders
  NARDOPAY_LEDGER_VERIFIED=true set only after checking viyo_coin_apply and transactions_external_ref_key exist in
                                production (credit_coins falls back to a path without duplicate protection otherwise)
"""
from __future__ import annotations

import datetime
import decimal
import hashlib
import hmac
import json
import math
import os
import re
import uuid
from typing import Any, Optional

import requests
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import nardopay_fx
from coins import credit_coins
from payments import COIN_PACKAGES

router = APIRouter(prefix="/api/v1", tags=["payments"])

CREATE_LINK_URL = "https://mczqwqsvumfsneoknlep.supabase.co/functions/v1/create-payment-link-api"   # documented
EVENT_COMPLETED, EVENT_FAILED = "payment.completed", "payment.failed"                               # documented
EVENT_RENEWED = "subscription.renewed"        # documented: a completed transaction whose link is a subscription
# Seen in a real signed NardoPay delivery (25 Nov 2026): its test webhook is {"event": "webhook.test", "test": true, ...}.
# A payload marked test=true, or a webhook.test event, is never a real payment: it is recorded and ignored, never
# credited, whatever order, amount and currency it carries.
TEST_FLAG, TEST_EVENTS = "test", ("webhook.test",)

# Where each value is in NardoPay's JSON, as a dotted path (e.g. "data.id"), set from NardoPay's documentation.
CONTRACT = {
    "link_url": "NARDOPAY_LINK_URL_FIELD",            # create-link response: the payment page URL ("url")
    "event_type": "NARDOPAY_FIELD_EVENT_TYPE",        # webhook: payment.completed / payment.failed ("event")
    "transaction_id": "NARDOPAY_FIELD_TRANSACTION_ID",
    "amount": "NARDOPAY_FIELD_AMOUNT",
    "currency": "NARDOPAY_FIELD_CURRENCY",
}
# How a payment finds its order: by the link's code (NardoPay sends "link_code"; it doesn't send metadata back,
# seen in a real payment on 25 Nov 2026), or by an order id the webhook carries. At least one must be set.
ORDER_MATCH = {"link_code": "NARDOPAY_FIELD_LINK_CODE", "order_id": "NARDOPAY_FIELD_ORDER_ID"}
OPTIONAL_CONTRACT = {"event_id": "NARDOPAY_FIELD_EVENT_ID", **ORDER_MATCH}
LINK_CODE_IN_URL = re.compile(r"/[ps]/([A-Za-z0-9_-]{6,64})(?:[/?#]|$)")   # /p/ payment, /s/ subscription links
AMOUNT_UNITS = {"rwf": 1, "rwf_minor": 100}         # NARDOPAY_AMOUNT_UNIT: what `amount` counts (unverified)

RATE_MIN_DEFAULT, RATE_MAX_DEFAULT = 1000.0, 3000.0
LARGE_CHANGE = 0.10
RECONCILE_MIN_AGE_SECONDS = 300

# VIYO Premium as a recurring NardoPay subscription (real money, never coins). The RWF amounts and billing cycles are
# the owner's NardoPay plans; the USD price is what VIYO displays. No trial.
PREMIUM_PLANS = {
    "weekly": {"rwf": 5900, "usd_cents": 399, "billing_cycle": "weekly", "interval": "week", "name": "VIYO Premium Weekly"},
    "monthly": {"rwf": 14800, "usd_cents": 999, "billing_cycle": "monthly", "interval": "month",
                "name": "VIYO Premium Monthly"},
    "yearly": {"rwf": 88600, "usd_cents": 5999, "billing_cycle": "yearly", "interval": "year", "name": "VIYO Premium Yearly"},
}
PREMIUM_REDIRECT_URL = "https://viyo.site/app"

# Coins bought through NardoPay: NardoPay adds a payment fee on top for the customer on payment links (about 17%,
# seen on its checkout page on 26 Nov 2026; Premium subscription links show no fee). VIYO says so on Buy Coins and
# adds 20% bonus coins to every package. Bonus coins are FREE coins (type "purchase_bonus", not "coin_purchase"):
# they never count as bought coins, so they never earn creators cash.
NARDOPAY_BONUS_RATE = 0.20


def bonus_coins_for(coins: int) -> int:
    return int(round(int(coins) * NARDOPAY_BONUS_RATE))


def customer_fee_percent() -> float:
    """The fee NardoPay adds for the customer on coin payment links, as shown in VIYO (NARDOPAY_CUSTOMER_FEE_PERCENT,
    default 17). Also bounds the extra amount a payment may report on top of the price."""
    try:
        return max(0.0, float(_env("NARDOPAY_CUSTOMER_FEE_PERCENT", "17") or 17))
    except ValueError:
        return 17.0      # where NardoPay sends the customer back (not proof of payment)


# ====================================================================== settings and helpers

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str) -> bool:
    return _env(name).lower() == "true"


def contract_field(key: str) -> str:
    return _env(CONTRACT.get(key) or OPTIONAL_CONTRACT.get(key) or "")


def missing_contract(keys=None) -> list[str]:
    out = [CONTRACT[k] for k in (keys or CONTRACT) if not _env(CONTRACT[k])]
    if keys is None and not any(_env(v) for v in ORDER_MATCH.values()):
        out.append("NARDOPAY_FIELD_LINK_CODE (or NARDOPAY_FIELD_ORDER_ID)")
    return out


def link_code_of(url: str) -> Optional[str]:
    """The code in a NardoPay payment link (https://nardopay.com/p/<code>?pay=1), or None."""
    m = LINK_CODE_IN_URL.search(url or "")
    return m.group(1) if m else None


def amount_multiplier() -> Optional[int]:
    return AMOUNT_UNITS.get(_env("NARDOPAY_AMOUNT_UNIT").lower())


def pick(obj: Any, path: str) -> Any:
    """The value at a dotted path ("data.metadata.order_id") in parsed JSON, or None."""
    cur = obj
    for part in (path or "").split("."):
        if not part:
            return None
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def rwf_for(usd_cents: int, rate: float) -> int:
    """Whole Rwandan francs for a USD price, rounded up (the customer never pays less than the price)."""
    return int(math.ceil(decimal.Decimal(usd_cents) * decimal.Decimal(str(rate)) / 100))


def sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def signature_ok(body: bytes, header: Optional[str], secret: str) -> bool:
    """X-NardoPay-Signature: hex HMAC-SHA256 of the exact raw body, compared in constant time."""
    if not secret or not header:
        return False
    return hmac.compare_digest(sign(body, secret), header.strip().lower())


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_time(v) -> datetime.datetime:
    t = datetime.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def _uuid_or_none(v) -> Optional[str]:
    try:
        return str(uuid.UUID(str(v)))
    except (ValueError, TypeError, AttributeError):
        return None


# ====================================================================== data access (the Studio's Store interface)

_store = None
_admin = None


def get_admin():
    """The Supabase service-role client (used by coins.credit_coins)."""
    global _admin
    if _admin is None:
        url, key = _env("SUPABASE_URL"), _env("SUPABASE_SERVICE_ROLE_KEY")
        if not (url and key):
            raise HTTPException(status_code=503, detail="Payments are not configured.")
        from supabase import create_client
        _admin = create_client(url, key)
    return _admin


def get_store():
    global _store
    if _store is None:
        from studio_store import SupabaseStore
        _store = SupabaseStore(get_admin())
    return _store


def _rpc(name: str, params: dict):
    return get_store().rpc(name, params)


# ====================================================================== exchange rate

RATE_HISTORY_LOOKBACK = 400


def _rate_rows() -> list[dict]:
    return get_store().select("nardopay_rates", {}, order="set_at", desc=True, limit=RATE_HISTORY_LOOKBACK)


def latest_rate() -> Optional[dict]:
    """The rate checkout uses now (nardopay_fx.current_rate): a held manual override, else the newer of the latest
    manual rate and BNR's latest published rate. Rejected automatic rates never count."""
    return nardopay_fx.current_rate(_rate_rows(), _now())


def rate_problem(rate: Optional[dict]) -> Optional[str]:
    """Why the current rate can't be used, or None. An automatic rate's age counts from its publication date."""
    return nardopay_fx.problem(rate, _now())


def validate_new_rate(value: float, previous: Optional[dict], confirm_large_change: bool) -> None:
    lo = float(_env("NARDOPAY_RATE_MIN", str(RATE_MIN_DEFAULT)) or RATE_MIN_DEFAULT)
    hi = float(_env("NARDOPAY_RATE_MAX", str(RATE_MAX_DEFAULT)) or RATE_MAX_DEFAULT)
    if not (lo <= value <= hi):
        raise HTTPException(status_code=422, detail=f"A rate must be between {lo:g} and {hi:g} RWF per USD.")
    if previous and not confirm_large_change:
        prev = float(previous["rate"])
        if abs(value - prev) / prev > LARGE_CHANGE:
            raise HTTPException(status_code=409, detail=f"That's more than a {int(LARGE_CHANGE * 100)}% change from "
                                                        f"{prev:g}. Send confirm_large_change if it's right.")


def require_admin(x_admin_key: str = Header(None)) -> None:
    key = _env("ADMIN_API_KEY")
    if not key or not x_admin_key or not hmac.compare_digest(x_admin_key, key):
        raise HTTPException(status_code=403, detail="Invalid admin key.")


class RateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rate: float = Field(..., gt=0)
    set_by: str = Field(..., min_length=1, max_length=120)
    note: str = Field("", max_length=400)
    confirm_large_change: bool = False
    hold_hours: int = Field(0, ge=0, le=24 * 30)


@router.get("/admin/nardopay/rate", dependencies=[Depends(require_admin)])
async def get_rate():
    rows = _rate_rows()
    r = nardopay_fx.current_rate(rows, _now())
    out = {"rate": r, "usable": rate_problem(r) is None, "problem": rate_problem(r),
           "auto_updates": nardopay_fx._env("NARDOPAY_RATE_AUTO").lower() in ("on", "true", "1", "yes"),
           "sources": nardopay_fx.sources(),
           "rejected": [x for x in rows if x.get("status") == "rejected"][:10]}
    try:
        out["recent_runs"] = get_store().select("nardopay_rate_runs", {}, order="ran_at", desc=True, limit=10)
    except Exception:
        out["recent_runs"] = []                       # nardopay_rates_auto.sql not run yet
    return out


@router.post("/admin/nardopay/rate", dependencies=[Depends(require_admin)])
async def set_rate(body: RateIn):
    """Adds a manual rate (rates are never edited or deleted, so the history stays). It is used until BNR publishes a
    newer rate, or, with hold_hours, for that long whatever BNR publishes."""
    validate_new_rate(body.rate, latest_rate(), body.confirm_large_change)
    row = {"rate": body.rate, "set_by": body.set_by.strip(), "note": body.note.strip()}
    if body.hold_hours:
        row["hold_until"] = (_now() + datetime.timedelta(hours=body.hold_hours)).isoformat()
    return {"rate": get_store().insert("nardopay_rates", [row])[0]}


@router.post("/admin/nardopay/rate/fetch", dependencies=[Depends(require_admin)])
async def fetch_rate_now():
    """Fetches BNR's latest rate now (same checks as the daily update). Never touches payment settings."""
    import asyncio
    out = await asyncio.to_thread(nardopay_fx.update, get_store())
    r = latest_rate()
    return {"result": out, "rate": r, "usable": rate_problem(r) is None, "problem": rate_problem(r)}


# ====================================================================== orders

async def _buyer(authorization: str = Header(None)) -> str:
    from main import _decode_token
    payload = _decode_token(authorization)
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Token missing subject")
    if payload.get("is_anonymous"):
        raise HTTPException(status_code=403, detail="Guest accounts can't buy coins. Create an account first.")
    return uid


class CreateOrderIn(BaseModel):
    model_config = ConfigDict(extra="forbid")      # a price, coins or amount from the client is refused, not ignored
    package_id: str = Field(..., min_length=1, max_length=40)


def _order_out(o: dict) -> dict:
    return {"order_id": o["id"], "status": o["status"], "package_id": o["package_id"], "coins": o["coins"],
            "bonus_coins": bonus_coins_for(o["coins"]),
            "usd_cents": o["usd_cents"], "rwf_amount": o["rwf_amount"], "currency": o["currency"],
            "payment_link": o.get("payment_link") if o["status"] in ("pending", "expired") else None,
            "created_at": o.get("created_at"), "credited_at": o.get("credited_at")}


def webhook_url() -> Optional[str]:
    """NARDOPAY_WEBHOOK_URL if it's an https address, else None."""
    u = _env("NARDOPAY_WEBHOOK_URL")
    return u if u.startswith("https://") and len(u) <= 500 and " " not in u else None


def _http_post(url: str, headers: dict, body: dict):
    return requests.post(url, headers=headers, json=body, timeout=15)


@router.post("/coins/purchase/nardopay/create")
async def create_order(body: CreateOrderIn, user_id: str = Depends(_buyer)):
    """A server-side pending order, then a NardoPay payment link for exactly its amount. Web checkout only: the
    Android app keeps Google Play Billing and never shows this."""
    if not _flag("NARDOPAY_ENABLED"):
        raise HTTPException(status_code=503, detail="NardoPay checkout isn't available.")
    if not _flag("NARDOPAY_LEDGER_VERIFIED"):
        raise HTTPException(status_code=503, detail="NardoPay checkout isn't available yet.")
    api_key, unit = _env("NARDOPAY_API_KEY"), amount_multiplier()
    if not api_key or unit is None or missing_contract(["link_url"]) or missing_contract() or not webhook_url():
        raise HTTPException(status_code=503, detail="NardoPay checkout isn't configured.")
    package = COIN_PACKAGES.get(body.package_id)
    if package is None:
        raise HTTPException(status_code=400, detail="Unknown coin package.")
    rate = latest_rate()
    problem = rate_problem(rate)
    if problem:
        raise HTTPException(status_code=503, detail=f"NardoPay checkout is paused: {problem}")
    rwf = rwf_for(package["usd_cents"], float(rate["rate"]))
    ttl = float(_env("NARDOPAY_ORDER_TTL_HOURS", "24") or 24)
    order = get_store().insert("nardopay_orders", [{
        "user_id": user_id, "package_id": body.package_id, "coins": int(package["coins"]),
        "usd_cents": int(package["usd_cents"]), "rate_id": rate["id"], "usd_rwf_rate": rate["rate"],
        "rwf_amount": rwf, "expires_at": (_now() + datetime.timedelta(hours=ttl)).isoformat()}])[0]

    def fail(reason: str, status: int = 502, message: str = "NardoPay couldn't start the payment. Try again."):
        get_store().update("nardopay_orders", {"id": order["id"]},
                           {"status": "failed", "failure_reason": reason[:400], "updated_at": _now().isoformat()})
        raise HTTPException(status_code=status, detail=message)

    request_body = {"link_type": "payment", "product_name": "VIYO Coins", "amount_mode": "fixed",
                    "amount": rwf * unit, "description": f"VIYO Coins purchase: {package['label']}",
                    "metadata": {"order_id": order["id"]}, "webhook_url": webhook_url(),
                    "redirect_url": PREMIUM_REDIRECT_URL}      # back to VIYO after paying (not proof of payment)
    try:
        resp = _http_post(CREATE_LINK_URL, {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                          request_body)
    except Exception as e:
        print(f"[WARN] NardoPay create-link unreachable for order {order['id']}: {type(e).__name__}")
        fail("unreachable")
    if not 200 <= resp.status_code < 300:
        print(f"[WARN] NardoPay create-link HTTP {resp.status_code} for order {order['id']}")
        fail(f"http {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        fail("response is not JSON")
    link = pick(data, contract_field("link_url"))
    if not isinstance(link, str) or not link.startswith("https://") or len(link) > 2000:
        fail("no https payment link at the configured field")
    code = link_code_of(link)
    if contract_field("link_code") and not code:
        # Payments are matched by link code: a link whose code can't be read could never be credited.
        fail("no link code in the payment link")
    fields = {"payment_link": link, "updated_at": _now().isoformat()}
    if code:
        fields["link_code"] = code
    try:
        order = get_store().update("nardopay_orders", {"id": order["id"]}, fields)[0]
    except Exception:
        fail("the payment link's code is already used by another order")
    return _order_out(order)


@router.get("/coins/purchase/nardopay/orders/{order_id}")
async def order_status(order_id: str, user_id: str = Depends(_buyer)):
    """The buyer's own order (the app polls this after checkout; the page NardoPay returns to credits nothing)."""
    oid = _uuid_or_none(order_id)
    o = get_store().one("nardopay_orders", {"id": oid}) if oid else None
    if not o or o["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    return _order_out(o)


# ====================================================================== webhook

def _parse_event(payload: dict) -> dict:
    """The values this module needs, read only from the configured (documented) fields."""
    amt = pick(payload, contract_field("amount"))
    try:
        amount = decimal.Decimal(str(amt)) if amt is not None and not isinstance(amt, bool) else None
    except decimal.InvalidOperation:
        amount = None
    cur = pick(payload, contract_field("currency"))
    tx = pick(payload, contract_field("transaction_id"))
    ev = pick(payload, contract_field("event_id")) if contract_field("event_id") else None
    order_id = _uuid_or_none(pick(payload, contract_field("order_id"))) if contract_field("order_id") else None
    if order_id is None and contract_field("link_code"):
        code = pick(payload, contract_field("link_code"))
        if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_-]{6,64}", code):
            row = get_store().one("nardopay_orders", {"link_code": code})
            order_id = row["id"] if row else None
    return {"event_type": str(pick(payload, contract_field("event_type")) or "")[:80],
            "order_id": order_id,
            "transaction_id": str(tx)[:200] if tx not in (None, "") else None,
            "amount": amount, "currency": str(cur).strip().upper()[:10] if cur not in (None, "") else None,
            "provider_event_id": str(ev)[:200] if ev not in (None, "") else None,
            "link_code": _link_code_in(payload)}


def _link_code_in(payload: dict) -> Optional[str]:
    if not contract_field("link_code"):
        return None
    code = pick(payload, contract_field("link_code"))
    return code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_-]{6,64}", code) else None


def is_test_event(payload) -> bool:
    """True for NardoPay's test deliveries: the payload says test=true (true, "true", 1), or its event type is
    webhook.test (read from the configured field, or the top-level "event" key the test webhook uses)."""
    if not isinstance(payload, dict):
        return False
    flag = payload.get(TEST_FLAG)
    if flag is True or (isinstance(flag, str) and flag.strip().lower() in ("true", "1", "yes")) or \
            (isinstance(flag, int) and not isinstance(flag, bool) and flag == 1):
        return True
    kinds = {str(payload.get("event") or "").strip().lower()}
    if contract_field("event_type"):
        kinds.add(str(pick(payload, contract_field("event_type")) or "").strip().lower())
    return bool(kinds & set(TEST_EVENTS))


def verify_payment(order: dict, ev: dict, fee_allowed: bool = False) -> Optional[str]:
    """Why a completed payment doesn't match its order, or None if it does. With fee_allowed (coin orders), the reported
    amount may be the price OR the price plus NardoPay's customer fee (at most customer_fee_percent() on top, rounded
    up, plus 1 franc for rounding): NardoPay's docs don't say which of the two its webhook reports. Never less."""
    unit = amount_multiplier()
    if unit is None:
        return "amount unit not configured"
    if not ev.get("transaction_id"):
        return "no transaction id"
    if ev.get("currency") != "RWF":
        return f"currency {ev.get('currency')!r} is not RWF"
    if ev.get("amount") is None:
        return "no amount"
    paid = decimal.Decimal(str(ev["amount"]))
    price = decimal.Decimal(int(order["rwf_amount"]) * unit)
    if paid == price:
        return None
    if fee_allowed and paid > price:
        max_fee = decimal.Decimal(math.ceil(int(order["rwf_amount"]) * customer_fee_percent() / 100) + 1) * unit
        if paid <= price + max_fee:
            return None
    return f"amount {ev.get('amount')} doesn't match {int(order['rwf_amount']) * unit}"


def _credit(order: dict, transaction_id: str) -> None:
    package = COIN_PACKAGES.get(order["package_id"]) or {}
    credit_coins(get_admin(), order["user_id"], int(order["coins"]), "coin_purchase",
                 f"Purchased {package.get('label', str(order['coins']) + ' coins')} — NardoPay {transaction_id}",
                 external_ref=f"nardopay:{transaction_id}")
    bonus = bonus_coins_for(order["coins"])
    if bonus > 0:
        # Free coins, credited once per payment (its own reference), never counted as bought.
        credit_coins(get_admin(), order["user_id"], bonus, "purchase_bonus",
                     f"Bonus {bonus} coins with your purchase — NardoPay {transaction_id}",
                     external_ref=f"nardopay-bonus:{transaction_id}")


def process_event(event_row: dict, payload: dict) -> tuple[int, dict]:
    """Processes one CLAIMED event. Returns (http status, body)."""
    eid = event_row["id"]
    ev = _parse_event(payload)

    def finish(status: str, detail: str, http: int = 200):
        _rpc("nardopay_finish_event", {"p_event": eid, "p_status": status, "p_detail": detail})
        return http, {"received": True, "result": status}

    if is_test_event(payload):           # defence in depth: handle_signed already stops these
        return finish("ignored", "test event: never credited")
    if not ev["order_id"] and ev.get("link_code"):
        sub = get_store().one("nardopay_subscriptions", {"link_code": ev["link_code"]}) if _subscriptions_ready() else None
        if sub:
            return _process_subscription_payment(sub, ev, eid, finish)
        import donations
        donation = donations.find_by_link_code(ev["link_code"])
        if donation:
            return donations.process_payment(donation, ev, eid, finish)
    if ev["event_type"] not in (EVENT_COMPLETED, EVENT_FAILED):
        return finish("ignored", f"event type {ev['event_type']!r} isn't handled")
    order = get_store().one("nardopay_orders", {"id": ev["order_id"]}) if ev["order_id"] else None
    if not order:
        return finish("ignored", "no order for this payment's order id")
    if ev["event_type"] == EVENT_FAILED:
        if order["status"] == "pending":
            get_store().update("nardopay_orders", {"id": order["id"]},
                               {"status": "failed", "failure_reason": "payment.failed", "updated_at": _now().isoformat()})
        return finish("processed", "payment failed")

    problem = verify_payment(order, ev, fee_allowed=True)
    if problem:
        if order["status"] in ("pending", "expired", "failed"):
            get_store().update("nardopay_orders", {"id": order["id"]},
                               {"status": "mismatch", "failure_reason": problem[:400], "updated_at": _now().isoformat()})
        print(f"[WARN] NardoPay payment for order {order['id']} not credited: {problem}")
        return finish("mismatch", problem)
    if not _flag("NARDOPAY_LEDGER_VERIFIED"):
        return finish("error", "ledger not verified: nothing credited; will be retried", 503)

    claim = _rpc("nardopay_claim_order", {"p_order": order["id"], "p_transaction": ev["transaction_id"],
                                          "p_event": eid}) or {}
    result = claim.get("result")
    if result in ("other_transaction", "transaction_used_elsewhere"):
        print(f"[WARN] NardoPay duplicate payment: order {order['id']}, transaction {ev['transaction_id']}")
        return finish("duplicate_payment", f"{result}: this payment wasn't credited; refund it in NardoPay")
    if result not in ("claimed", "retry"):
        return finish("ignored", f"order can't be credited ({result})")
    _credit(order, ev["transaction_id"])
    if not _rpc("nardopay_finish_order", {"p_order": order["id"], "p_transaction": ev["transaction_id"]}):
        raise RuntimeError("order could not be marked credited")
    return finish("processed", "credited")


# ====================================================================== Premium subscriptions

def _subscriptions_ready() -> bool:
    """False until nardopay_subscriptions.sql has been run (then subscription payments simply aren't matched)."""
    try:
        get_store().select("nardopay_subscriptions", {}, limit=1)
        return True
    except Exception:
        return False


def premium_grace_hours() -> int:
    return max(0, int(_env("NARDOPAY_PREMIUM_GRACE_HOURS", "48") or 48))


def _process_subscription_payment(sub: dict, ev: dict, eid: str, finish) -> tuple[int, dict]:
    """A signed webhook for one of our subscription links. Premium is extended only for a completed payment of exactly
    the plan's RWF amount; each NardoPay transaction counts once."""
    if ev["event_type"] == EVENT_FAILED:
        return finish("processed", "subscription payment failed: Premium not extended")
    if ev["event_type"] not in (EVENT_COMPLETED, EVENT_RENEWED):
        return finish("ignored", f"event type {ev['event_type']!r} isn't handled")
    problem = verify_payment(sub, ev)
    if problem:
        print(f"[WARN] NardoPay subscription payment for {sub['id']} not counted: {problem}")
        return finish("mismatch", problem)
    out = _rpc("nardopay_record_subscription_payment", {
        "p_subscription": sub["id"], "p_transaction": ev["transaction_id"], "p_event": eid,
        "p_event_type": ev["event_type"], "p_amount": str(ev["amount"]), "p_currency": ev["currency"],
        "p_grace_hours": premium_grace_hours()}) or {}
    if out.get("result") == "duplicate":
        return finish("processed", "subscription payment already counted")
    if out.get("result") != "recorded":
        raise RuntimeError(f"subscription payment not recorded ({out.get('result')})")
    return finish("processed", "Premium extended")


class PremiumSubscribeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")      # the price and cycle come from PREMIUM_PLANS, never the client
    plan: str = Field(..., pattern="^(weekly|monthly|yearly)$")


def _premium_problem() -> Optional[str]:
    """Why a Premium subscription can't be started right now, or None."""
    if not _flag("NARDOPAY_ENABLED") or not _flag("NARDOPAY_LEDGER_VERIFIED"):
        return "switched off"
    if not _env("NARDOPAY_API_KEY") or amount_multiplier() is None or missing_contract(["link_url"]) \
            or missing_contract() or not contract_field("link_code") or not webhook_url():
        return "not configured"
    if not _subscriptions_ready():
        return "not set up"
    return None


def _sub_out(s: dict) -> dict:
    return {"subscription_id": s["id"], "status": s["status"], "plan": s["plan"], "usd_cents": s["usd_cents"],
            "payment_link": s.get("payment_link") if s["status"] == "pending" else None,
            "current_period_end": s.get("current_period_end")}


@router.post("/subscription/nardopay/create")
async def create_premium_subscription(body: PremiumSubscribeIn, user_id: str = Depends(_buyer)):
    """Creates this user's NardoPay subscription link for the chosen plan (web only). Premium starts only when
    NardoPay's signed payment webhook for this link arrives, never from the redirect back to VIYO."""
    problem = _premium_problem()
    if problem:
        raise HTTPException(status_code=503, detail="Premium checkout isn't available right now.")
    rows = get_store().select("profiles", {"id": user_id}, limit=1)
    prof = rows[0] if rows else {}
    ends = prof.get("subscription_ends_at")
    if prof.get("is_subscribed") and (not ends or _parse_time(ends) > _now()):
        raise HTTPException(status_code=409, detail="You already have VIYO Premium.")
    plan = PREMIUM_PLANS[body.plan]
    unit = amount_multiplier()
    sub = get_store().insert("nardopay_subscriptions", [{
        "user_id": user_id, "plan": body.plan, "rwf_amount": plan["rwf"], "usd_cents": plan["usd_cents"]}])[0]

    def fail(reason: str):
        get_store().update("nardopay_subscriptions", {"id": sub["id"]},
                           {"status": "failed", "failure_reason": reason[:400], "updated_at": _now().isoformat()})
        raise HTTPException(status_code=502, detail="NardoPay couldn't start the subscription. Try again.")

    request_body = {"link_type": "subscription", "plan_name": plan["name"], "amount": plan["rwf"] * unit,
                    "billing_cycle": plan["billing_cycle"], "trial_days": 0,
                    "description": f"{plan['name']}: unlimited episodes on VIYO",
                    "metadata": {"subscription_id": sub["id"]}, "webhook_url": webhook_url(),
                    "redirect_url": PREMIUM_REDIRECT_URL}
    try:
        resp = _http_post(CREATE_LINK_URL, {"Authorization": f"Bearer {_env('NARDOPAY_API_KEY')}",
                                            "Content-Type": "application/json"}, request_body)
    except Exception as e:
        print(f"[WARN] NardoPay subscription link unreachable for {sub['id']}: {type(e).__name__}")
        fail("unreachable")
    if not 200 <= resp.status_code < 300:
        print(f"[WARN] NardoPay subscription link HTTP {resp.status_code} for {sub['id']}")
        fail(f"http {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        fail("response is not JSON")
    link = pick(data, contract_field("link_url"))
    if not isinstance(link, str) or not link.startswith("https://") or len(link) > 2000:
        fail("no https link at the configured field")
    code = link_code_of(link)
    if not code:
        fail("no link code in the subscription link")       # payments are matched by it: never sell without one
    try:
        sub = get_store().update("nardopay_subscriptions", {"id": sub["id"]},
                                 {"payment_link": link, "link_code": code, "updated_at": _now().isoformat()})[0]
    except Exception:
        fail("the link code is already used")
    return _sub_out(sub)


@router.get("/subscription/nardopay/{subscription_id}")
async def premium_subscription_status(subscription_id: str, user_id: str = Depends(_buyer)):
    """The buyer's own subscription (the app checks this after checkout until NardoPay's payment is verified)."""
    sid = _uuid_or_none(subscription_id)
    s = get_store().one("nardopay_subscriptions", {"id": sid}) if sid else None
    if not s or s["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Subscription not found.")
    return _sub_out(s)


def handle_signed(raw: bytes, payload: Optional[dict]) -> tuple[int, dict]:
    """A delivery whose signature is valid: record it (or find its earlier record), claim it, process it."""
    configured = not missing_contract() and amount_multiplier() is not None
    ev = _parse_event(payload) if (configured and isinstance(payload, dict)) else {}
    rec = _rpc("nardopay_record_event", {
        "p_hash": hashlib.sha256(raw).hexdigest(), "p_signature_ok": True, "p_event_type": ev.get("event_type"),
        "p_order_id": ev.get("order_id"), "p_transaction_id": ev.get("transaction_id"),
        "p_amount": str(ev["amount"]) if ev.get("amount") is not None else None, "p_currency": ev.get("currency"),
        "p_provider_event_id": ev.get("provider_event_id"), "p_raw": raw.decode("utf-8", "replace")}) or {}
    row = rec.get("event") or {}
    if row.get("status") in ("processed", "ignored", "mismatch", "duplicate_payment"):
        return 200, {"received": True, "result": row["status"], "duplicate": True}
    if not isinstance(payload, dict):
        _rpc("nardopay_finish_event", {"p_event": row["id"], "p_status": "ignored", "p_detail": "body is not a JSON object"})
        return 200, {"received": True, "result": "ignored"}
    if is_test_event(payload):
        # A test delivery (webhook.test, or anything marked test=true, payment.completed included) is recorded and
        # finished as ignored at once, even before the field settings are filled in: it can never credit coins.
        _rpc("nardopay_finish_event", {"p_event": row["id"], "p_status": "ignored", "p_detail": "test event: never credited"})
        return 200, {"received": True, "result": "ignored", "test": True}
    if not configured:
        return 202, {"received": True, "result": "stored", "detail": "NardoPay fields not configured yet"}
    if not _rpc("nardopay_claim_event", {"p_event": row["id"]}):
        return 409, {"received": False, "detail": "This delivery is being processed. Try again shortly."}
    try:
        return process_event(row, payload)
    except Exception as e:
        print(f"[ERROR] NardoPay event {row['id']} failed: {type(e).__name__}")
        _rpc("nardopay_finish_event", {"p_event": row["id"], "p_status": "error", "p_detail": type(e).__name__})
        return 500, {"received": False, "detail": "Couldn't process this payment yet. It will be retried."}


@router.post("/webhooks/nardopay")
async def nardopay_webhook(request: Request, x_nardopay_signature: str = Header(None, alias="X-NardoPay-Signature")):
    secret = _env("NARDOPAY_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=503, detail="NardoPay webhooks are not configured.")
    raw = await request.body()                               # the exact bytes NardoPay signed
    if not signature_ok(raw, x_nardopay_signature, secret):
        raise HTTPException(status_code=400, detail="Invalid NardoPay signature.")
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = None
    status, body = handle_signed(raw, payload)
    return JSONResponse(body, status_code=status)


# ====================================================================== reconciliation and maintenance

def reconcile(min_age_seconds: int = RECONCILE_MIN_AGE_SECONDS) -> dict:
    """Finishes payments that were VERIFIED and claimed but not marked credited (a crash in between). It re-checks the
    stored event (signature, transaction, amount, currency) and never touches a pending or unverified order.
    credit_coins is safe to repeat: the ledger credits each "nardopay:<tx>" once."""
    out = {"finished": 0, "skipped": [], "expired": 0, "purged": 0}
    if not _flag("NARDOPAY_LEDGER_VERIFIED"):
        out["skipped"].append({"reason": "ledger not verified"})
    else:
        for c in _rpc("nardopay_reconcile_candidates", {"p_min_age_seconds": int(min_age_seconds)}) or []:
            ev = {"transaction_id": c.get("event_transaction_id"), "currency": c.get("event_currency"),
                  "amount": c.get("event_amount")}
            problem = None if c.get("signature_ok") is True else "event signature not verified"
            problem = problem or (None if c.get("event_transaction_id") == c.get("transaction_id")
                                  else "event and order transactions differ")
            problem = problem or verify_payment(c, ev, fee_allowed=(c.get("kind") or "coins") == "coins")
            if problem:
                out["skipped"].append({"order_id": c["order_id"], "reason": problem})
                continue
            _credit({"user_id": c["user_id"], "coins": c["coins"], "package_id": c["package_id"]}, c["transaction_id"])
            if _rpc("nardopay_finish_order", {"p_order": c["order_id"], "p_transaction": c["transaction_id"]}):
                _rpc("nardopay_finish_event", {"p_event": c["event_id"], "p_status": "processed",
                                               "p_detail": "credited by reconciliation"})
                out["finished"] += 1
    out["expired"] = int(_rpc("nardopay_expire_orders", {}) or 0)
    try:
        out["premium_ended"] = int(_rpc("nardopay_end_lapsed_subscriptions", {}) or 0)
    except Exception:
        out["premium_ended"] = 0         # nardopay_subscriptions.sql not run yet
    out["purged"] = int(_rpc("nardopay_purge_raw", {
        "p_days": int(_env("NARDOPAY_RAW_BODY_DAYS", "30") or 30),
        "p_problem_days": int(_env("NARDOPAY_RAW_BODY_PROBLEM_DAYS", "90") or 90)}) or 0)
    return out


def reprocess_stored(limit: int = 100) -> dict:
    """Processes signed events that were stored but not handled (fields not configured yet, a crash, the ledger
    gate), while their raw body is still kept. Same checks as a live delivery."""
    rows = [r for r in get_store().select("nardopay_webhook_events", {"signature_ok": True}, order="received_at", limit=500)
            if r.get("status") in ("received", "error") and r.get("raw_body")][:limit]
    out = {"processed": 0, "results": []}
    for r in rows:
        try:
            payload = json.loads(r["raw_body"])
        except ValueError:
            continue
        status, body = handle_signed(r["raw_body"].encode("utf-8"), payload)
        out["results"].append({"event_id": r["id"], "http": status, "result": body.get("result")})
        out["processed"] += 1
    return out


class TestLinkIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_rwf: int = Field(100, ge=100, le=2000)


def https_fields(obj, prefix: str = "", depth: int = 0) -> list[dict]:
    """Every https URL in a JSON response, with its dotted path ("data.url"): shown to the admin so the link field
    setting comes from what NardoPay really returns, not from a guess."""
    out = []
    if depth > 4:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += https_fields(v, f"{prefix}.{k}" if prefix else str(k), depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:10]):
            out += https_fields(v, f"{prefix}.{i}" if prefix else str(i), depth + 1)
    elif isinstance(obj, str) and obj.startswith("https://") and len(obj) <= 2000:
        out.append({"field": prefix, "url": obj})
    return out


@router.post("/admin/nardopay/test-link", dependencies=[Depends(require_admin)])
async def admin_test_link(body: TestLinkIn):
    """A real NardoPay payment link for the owner to pay once, so the webhook NardoPay really sends can be seen
    (nardopay_webhook_events.raw_body). It belongs to NO VIYO order (no order_id in its metadata), so it can never
    credit coins: it's stored, then ignored once the field settings exist. Works while checkout is still off, and
    shows which response field holds the link (for NARDOPAY_LINK_URL_FIELD)."""
    api_key, unit, hook = _env("NARDOPAY_API_KEY"), amount_multiplier(), webhook_url()
    missing = [n for n, ok in (("NARDOPAY_API_KEY", api_key), ("NARDOPAY_WEBHOOK_URL", hook)) if not ok]
    if missing:
        raise HTTPException(status_code=503, detail=f"Set these first in Railway: {', '.join(missing)}.")
    # The amount unit isn't confirmed yet: whole francs unless NARDOPAY_AMOUNT_UNIT says otherwise.
    amount = body.amount_rwf * (unit or 1)
    request_body = {"link_type": "payment", "product_name": "VIYO test payment", "amount_mode": "fixed",
                    "amount": amount, "description": "VIYO webhook test (not a coin purchase)",
                    "metadata": {"viyo_test": "admin"}, "webhook_url": hook}
    try:
        resp = _http_post(CREATE_LINK_URL, {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                          request_body)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NardoPay couldn't be reached ({type(e).__name__}).")
    try:
        data = resp.json()
    except ValueError:
        data = None
    found = https_fields(data) if isinstance(data, (dict, list)) else []
    configured = pick(data, contract_field("link_url")) if contract_field("link_url") and data is not None else None
    if not 200 <= resp.status_code < 300 or not found:
        raise HTTPException(status_code=502, detail={
            "message": f"NardoPay answered HTTP {resp.status_code} without a payment link.",
            "response_fields": sorted(data.keys())[:30] if isinstance(data, dict) else []})
    link = configured if isinstance(configured, str) and configured.startswith("https://") else found[0]["url"]
    return {"link": link, "links_found": found[:10], "link_field_setting": contract_field("link_url") or None,
            "amount_rwf": body.amount_rwf, "amount_sent": amount}


@router.post("/admin/nardopay/reconcile", dependencies=[Depends(require_admin)])
async def admin_reconcile():
    return reconcile()


@router.post("/admin/nardopay/reprocess", dependencies=[Depends(require_admin)])
async def admin_reprocess():
    return reprocess_stored()


@router.get("/admin/nardopay/duplicate-payments", dependencies=[Depends(require_admin)])
async def duplicate_payments():
    """Payments that weren't credited because their order was already paid: refund them in NardoPay."""
    rows = get_store().select("nardopay_webhook_events", {"status": "duplicate_payment"}, order="received_at", desc=True,
                              limit=200)
    return {"payments": [{k: r.get(k) for k in ("id", "order_id", "transaction_id", "amount", "currency", "detail",
                                                 "received_at")} for r in rows]}


# ====================================================================== availability and the reconciliation schedule

def checkout_problem() -> Optional[str]:
    """Why web checkout can't take an order right now (for logs), or None."""
    if not _flag("NARDOPAY_ENABLED") or not _flag("NARDOPAY_LEDGER_VERIFIED"):
        return "switched off"
    if not _env("NARDOPAY_API_KEY") or amount_multiplier() is None or missing_contract() or not webhook_url():
        return "not configured"
    try:
        return rate_problem(latest_rate())
    except Exception:
        return "rates unavailable"


@router.get("/coins/purchase/nardopay/availability")
async def availability():
    """Whether the web app should offer "Pay with NardoPay". Only a yes/no: no setting is revealed. The Android app
    never shows NardoPay (it keeps Google Play Billing), whatever this says."""
    return {"available": checkout_problem() is None}


def reconciler_should_run() -> bool:
    return _flag("NARDOPAY_ENABLED") or bool(_env("NARDOPAY_WEBHOOK_SECRET"))


def reconcile_forever(interval_seconds: int = 600, stop=None) -> None:
    """Runs reconcile() every interval (finishing verified payments, expiring unpaid orders, purging old raw webhook
    bodies). Errors are logged and never stop the loop."""
    import threading
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            if reconciler_should_run():
                out = reconcile()
                if out["finished"] or out["skipped"] or out["purged"]:
                    print(f"[INFO] NardoPay reconcile: {out['finished']} finished, {len(out['skipped'])} skipped, "
                          f"{out['expired']} expired, {out['purged']} raw bodies purged")
        except Exception as e:
            print(f"[WARN] NardoPay reconcile failed: {type(e).__name__}: {str(e)[:160]}")
        stop.wait(interval_seconds)


def start_reconciler() -> bool:
    """Starts the reconciliation loop in a background thread (called from the worker service). Returns whether it
    started; NARDOPAY_RECONCILE=off turns it off."""
    import threading
    if _env("NARDOPAY_RECONCILE", "on").lower() in ("off", "false", "0", "no"):
        print("[INFO] NardoPay reconciliation off (NARDOPAY_RECONCILE=off)")
        return False
    interval = int(_env("NARDOPAY_RECONCILE_SECONDS", "600") or 600)
    threading.Thread(target=reconcile_forever, args=(max(60, interval),), name="nardopay-reconcile", daemon=True).start()
    print(f"[INFO] NardoPay reconciliation scheduled every {max(60, interval)}s (acts only when NardoPay is set up)")
    return True
