"""Brand campaigns: businesses pay creators for sponsored posts, through VIYO (brand_campaigns.sql).

Rules (agreed with the owner):
  * A business writes a brief with a fixed fee per creator and a number of creators, and pays the whole budget upfront
    by card through NardoPay. Amounts are in US dollars; VIYO converts them with its own rate (no RWF from the client)
    and shows NardoPay's customer fee before payment. Campaign credit is used first.
  * Creators at level Rising or above (creator_levels) apply; the business accepts or declines.
  * An accepted creator submits one of their own posts, which is labelled Sponsored. The business approves or rejects
    it with a reason; undecided submissions are approved after 7 days.
  * On approval the creator earns 80% of the fee (VIYO keeps 20%), held like every new earning.
  * Unused money goes back to the business as campaign credit, once. NardoPay has no refund API: a money refund is
    done by an admin on request.
A campaign counts as paid only when NardoPay's signed payment webhook for its link arrives (never from a redirect)."""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import nardopay_payments as np_

router = APIRouter(prefix="/api/v1/brand-campaigns", tags=["brand-campaigns"])

CREATOR_SHARE = 0.80
FEE_MIN_CENTS = 500             # $5 per creator
FEE_MAX_CENTS = 500000          # $5,000 per creator
SLOTS_MAX = 50
DAYS_MIN, DAYS_MAX = 3, 90
AUTO_APPROVE_DAYS = 7
MIN_LEVEL_RANK = 2              # Rising
MIN_LEVEL_NAME = "Rising"
REDIRECT_URL = "https://viyo.site/app"

ERRORS = {
    "not_found": (404, "Not found."),
    "not_allowed": (403, "You can't do that."),
    "campaign_not_open": (409, "This campaign isn't taking creators any more."),
    "wrong_state": (409, "This has already changed. Refresh to see the latest."),
    "no_slots": (409, "Every place in this campaign is taken."),
    "deadline_passed": (409, "The campaign's deadline has passed."),
    "no_post": (400, "Choose the post you made for this campaign."),
    "not_your_post": (400, "That post isn't yours."),
    "post_used": (409, "That post is already submitted to another campaign."),
    "reason_required": (400, "Tell the creator why (a short reason)."),
}


def _ready() -> bool:
    try:
        np_.get_store().select("brand_campaigns", {}, limit=1)
        return True
    except Exception:
        return False


def payment_problem() -> Optional[str]:
    """Why a business can't pay for a campaign right now, or None (same checks as donations)."""
    if not np_._flag("NARDOPAY_ENABLED") or not np_._flag("NARDOPAY_LEDGER_VERIFIED"):
        return "switched off"
    if not np_._env("NARDOPAY_API_KEY") or np_.amount_multiplier() is None or np_.missing_contract(["link_url"]) \
            or np_.missing_contract() or not np_.contract_field("link_code") or not np_.webhook_url():
        return "not configured"
    if np_.rate_problem(np_.latest_rate()):
        return "rate unavailable"
    return None


def _store():
    if not _ready():
        raise HTTPException(status_code=503, detail="Brand campaigns aren't set up yet.")
    return np_.get_store()


def _rpc(name: str, params: dict) -> dict:
    out = np_._rpc(name, params) or {}
    if isinstance(out, dict) and out.get("ok") is False:
        code, msg = ERRORS.get(out.get("error"), (409, "That can't be done right now."))
        raise HTTPException(status_code=code, detail=msg)
    return out


def _usd(cents) -> float:
    return round(int(cents or 0) / 100, 2)


def creator_share_cents(fee_cents: int) -> int:
    return int(fee_cents * CREATOR_SHARE)


def _campaign_out(c: dict, owner_view: bool = False) -> dict:
    out = {
        "id": c["id"], "brand_name": c["brand_name"], "title": c["title"], "brief": c["brief"],
        "deliverable": c.get("deliverable") or "", "link": c.get("link"),
        "fee_usd": _usd(c["fee_cents"]), "creator_gets_usd": _usd(creator_share_cents(c["fee_cents"])),
        "slots": c["slots"], "status": c["status"], "deadline": c["deadline"], "created_at": c.get("created_at"),
    }
    if owner_view:
        out.update({
            "budget_usd": _usd(c["budget_cents"]), "credit_used_usd": _usd(c.get("credit_used_cents")),
            "pay_usd": _usd(c.get("pay_cents")), "paid": c.get("transaction_id") is not None or c["status"] != "awaiting_payment",
            "payment_link": c.get("payment_link") if c["status"] == "awaiting_payment" else None,
            "credited_back_usd": _usd(c.get("credited_back_cents")), "failure_reason": c.get("failure_reason") or "",
        })
    return out


def _app_out(a: dict, campaign: Optional[dict] = None, with_creator: bool = False) -> dict:
    out = {"id": a["id"], "campaign_id": a["campaign_id"], "status": a["status"], "pitch": a.get("pitch") or "",
           "post_id": a.get("post_id"), "decision_note": a.get("decision_note") or "",
           "auto_approved": bool(a.get("auto_approved")), "submitted_at": a.get("submitted_at"),
           "decided_at": a.get("decided_at"), "created_at": a.get("created_at")}
    if a.get("submitted_at") and a["status"] == "submitted":
        import datetime as _dt
        try:
            t = _dt.datetime.fromisoformat(str(a["submitted_at"]).replace("Z", "+00:00"))
            out["auto_approves_at"] = (t + _dt.timedelta(days=AUTO_APPROVE_DAYS)).isoformat()
        except ValueError:
            pass
    if campaign is not None:
        out["campaign"] = _campaign_out(campaign)
    if with_creator:
        from notify import display_name
        out["creator"] = {"id": a["creator_id"], "name": display_name(a["creator_id"])}
    return out


def _level(uid: str) -> dict:
    import creator_levels
    return creator_levels.level_for(np_.get_store(), uid)


# ---------------------------------------------------------------- for everyone

@router.get("/availability")
async def availability():
    ready = _ready()
    problem = payment_problem()
    return {"available": ready, "can_pay": ready and problem is None, "creator_share": CREATOR_SHARE,
            "fee_min_usd": _usd(FEE_MIN_CENTS), "fee_max_usd": _usd(FEE_MAX_CENTS), "slots_max": SLOTS_MAX,
            "days_min": DAYS_MIN, "days_max": DAYS_MAX, "auto_approve_days": AUTO_APPROVE_DAYS,
            "min_level": MIN_LEVEL_NAME,
            "fee_percent": np_.customer_fee_percent() if problem is None else None}


# ---------------------------------------------------------------- the business

class CampaignIn(BaseModel):
    model_config = ConfigDict(extra="forbid")       # no RWF amount, rate or status from the client
    brand_name: str = Field(..., min_length=1, max_length=80)
    title: str = Field(..., min_length=1, max_length=120)
    brief: str = Field(..., min_length=10, max_length=2000)
    deliverable: str = Field("", max_length=500)
    link: str = Field("", max_length=300)
    fee_usd_cents: int = Field(..., ge=FEE_MIN_CENTS, le=FEE_MAX_CENTS)
    slots: int = Field(..., ge=1, le=SLOTS_MAX)
    days: int = Field(14, ge=DAYS_MIN, le=DAYS_MAX)


@router.get("/credit")
async def credit(uid: str = Depends(np_._buyer)):
    _store()
    cents = int(np_._rpc("brand_credit_balance", {"p_owner": uid}) or 0)
    return {"credit_usd": _usd(cents)}


@router.post("")
async def create_campaign(body: CampaignIn, uid: str = Depends(np_._buyer)):
    store = _store()
    link = body.link.strip()
    if link and not link.startswith("https://"):
        raise HTTPException(status_code=400, detail="The link must start with https://")
    budget = body.fee_usd_cents * body.slots
    credit_cents = int(np_._rpc("brand_credit_balance", {"p_owner": uid}) or 0)
    needs_payment = budget > max(credit_cents, 0)
    rate = None
    if needs_payment:
        if payment_problem():
            raise HTTPException(status_code=503, detail="Card payments for campaigns aren't available right now.")
        rate = np_.latest_rate()
    out = _rpc("brand_campaign_create", {
        "p_owner": uid, "p_brand": body.brand_name.strip(), "p_title": body.title.strip(), "p_brief": body.brief.strip(),
        "p_deliverable": body.deliverable.strip(), "p_link": link, "p_fee_cents": body.fee_usd_cents,
        "p_slots": body.slots, "p_days": body.days, "p_rate_id": str(rate["id"]) if rate else "",
        "p_rate": str(rate["rate"]) if rate else ""})
    c = out["campaign"]
    if c["status"] != "awaiting_payment":
        return _campaign_out(c, owner_view=True)

    def fail(reason: str):
        store.update("brand_campaigns", {"id": c["id"]}, {"failure_reason": reason[:400]})
        np_._rpc("brand_campaign_close", {"p_campaign": c["id"], "p_actor": uid, "p_mode": "cancel"})
        raise HTTPException(status_code=502, detail="NardoPay couldn't start the payment. Try again.")

    request_body = {"link_type": "payment", "product_name": f"VIYO campaign: {c['title']}"[:120],
                    "amount_mode": "fixed", "amount": int(c["price_rwf"]) * np_.amount_multiplier(),
                    "description": f"Brand campaign on VIYO: {body.slots} creator(s) x ${body.fee_usd_cents / 100:.2f}",
                    "metadata": {"brand_campaign_id": c["id"]}, "webhook_url": np_.webhook_url(),
                    "redirect_url": REDIRECT_URL}
    try:
        resp = np_._http_post(np_.CREATE_LINK_URL, {"Authorization": f"Bearer {np_._env('NARDOPAY_API_KEY')}",
                                                    "Content-Type": "application/json"}, request_body)
    except Exception as e:
        print(f"[WARN] NardoPay campaign link unreachable for {c['id']}: {type(e).__name__}")
        fail("unreachable")
    if not 200 <= resp.status_code < 300:
        print(f"[WARN] NardoPay campaign link HTTP {resp.status_code} for {c['id']}")
        fail(f"http {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        fail("response is not JSON")
    link_url = np_.pick(data, np_.contract_field("link_url"))
    if not isinstance(link_url, str) or not link_url.startswith("https://") or len(link_url) > 2000:
        fail("no https link at the configured field")
    code = np_.link_code_of(link_url)
    if not code:
        fail("no link code in the payment link")
    try:
        c = store.update("brand_campaigns", {"id": c["id"]}, {"payment_link": link_url, "link_code": code})[0]
    except Exception:
        fail("the link code is already used")
    return _campaign_out(c, owner_view=True)


def _owned(cid: str, uid: str) -> dict:
    cid = np_._uuid_or_none(cid)
    c = _store().one("brand_campaigns", {"id": cid}) if cid else None
    if not c or c["owner_id"] != uid:
        raise HTTPException(status_code=404, detail="Campaign not found.")
    return c


@router.get("/mine")
async def my_campaigns(uid: str = Depends(np_._buyer)):
    store = _store()
    rows = store.select("brand_campaigns", {"owner_id": uid}, order="created_at", desc=True, limit=50)
    out = []
    for c in rows:
        apps = store.select("brand_campaign_applications", {"campaign_id": c["id"]}, order="created_at", limit=500)
        item = _campaign_out(c, owner_view=True)
        item["counts"] = {s: sum(1 for a in apps if a["status"] == s) for s in
                          ("applied", "accepted", "submitted", "approved", "rejected", "declined", "expired", "withdrawn")}
        out.append(item)
    cents = int(np_._rpc("brand_credit_balance", {"p_owner": uid}) or 0)
    return {"campaigns": out, "credit_usd": _usd(cents)}


@router.get("/{campaign_id}")
async def campaign_detail(campaign_id: str, uid: str = Depends(np_._buyer)):
    """The business's own campaign with every application (the app also checks it after payment)."""
    c = _owned(campaign_id, uid)
    apps = _store().select("brand_campaign_applications", {"campaign_id": c["id"]}, order="created_at", limit=500)
    return {**_campaign_out(c, owner_view=True), "applications": [_app_out(a, with_creator=True) for a in apps]}


@router.post("/{campaign_id}/close")
async def close_campaign(campaign_id: str, uid: str = Depends(np_._buyer)):
    c = _owned(campaign_id, uid)
    mode = "cancel" if c["status"] == "awaiting_payment" else "close"
    out = _rpc("brand_campaign_close", {"p_campaign": c["id"], "p_actor": uid, "p_mode": mode})
    return {"closed": True, "credited_usd": _usd(out.get("credited_cents"))}


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str = Field("", max_length=300)


def _act(app_id: str, uid: Optional[str], action: str, post_id: Optional[str] = None, note: str = "") -> dict:
    import creator_levels
    import wallet
    aid = np_._uuid_or_none(app_id)
    if not aid:
        raise HTTPException(status_code=404, detail="Not found.")
    store = _store()
    a = store.one("brand_campaign_applications", {"id": aid})
    if not a:
        raise HTTPException(status_code=404, detail="Not found.")
    hold = creator_levels.hold_days_for(store, str(a["creator_id"])) if action in ("approve", "auto_approve") else 0
    out = _rpc("brand_campaign_act", {"p_application": aid, "p_actor": uid, "p_action": action, "p_post": post_id,
                                      "p_note": note.strip(), "p_hold_days": hold, "p_coin_usd": str(wallet.COIN_USD),
                                      "p_sponsor": None})
    _notify(action, out)
    return out


def _notify(action: str, out: dict) -> None:
    try:
        from notify import notify
        title = out.get("title") or "your campaign"
        brand = out.get("brand_name") or "A brand"
        if action == "accept":
            notify(str(out["creator_id"]), "brand_deal", f"{brand} picked you for \"{title}\". Make your post and "
                   f"submit it before the deadline.", entity_id=str(out["campaign_id"]), push_title="Brand deal 🤝")
        elif action in ("approve", "auto_approve"):
            notify(str(out["creator_id"]), "brand_deal", f"{brand} approved your post for \"{title}\". You earned "
                   f"${float(out.get('share_usd') or 0):.2f}.", entity_id=str(out["campaign_id"]),
                   push_title="Brand deal paid 💸")
        elif action == "reject":
            notify(str(out["creator_id"]), "brand_deal", f"{brand} didn't approve your post for \"{title}\".",
                   entity_id=str(out["campaign_id"]))
        elif action == "apply" or action == "submit":
            notify(str(out["owner_id"]), "brand_campaign",
                   ("A creator applied to " if action == "apply" else "A creator submitted their post for ") +
                   f"\"{title}\".", entity_id=str(out["campaign_id"]))
    except Exception as e:
        print(f"[WARN] brand campaign notification failed: {e}")


# ---------------------------------------------------------------- creators

@router.get("/creator/open")
async def open_campaigns(uid: str = Depends(np_._buyer)):
    """Open campaigns, whether this creator can apply, and their own application to each."""
    store = _store()
    level = _level(uid)
    rows = [c for c in store.select("brand_campaigns", {"status": "open"}, order="created_at", desc=True, limit=100)
            if c["owner_id"] != uid]
    mine = {a["campaign_id"]: a for a in store.select("brand_campaign_applications", {"creator_id": uid},
                                                      order="created_at", desc=True, limit=500)}
    return {"eligible": level["rank"] >= MIN_LEVEL_RANK, "level": level["name"], "min_level": MIN_LEVEL_NAME,
            "creator_share": CREATOR_SHARE,
            "campaigns": [{**_campaign_out(c), "my_application": _app_out(mine[c["id"]]) if c["id"] in mine else None}
                          for c in rows]}


@router.get("/creator/deals")
async def my_deals(uid: str = Depends(np_._buyer)):
    store = _store()
    apps = store.select("brand_campaign_applications", {"creator_id": uid}, order="created_at", desc=True, limit=100)
    out = []
    for a in apps:
        c = store.one("brand_campaigns", {"id": a["campaign_id"]})
        if c:
            out.append(_app_out(a, campaign=c))
    return {"deals": out}


@router.get("/creator/posts")
async def my_posts(uid: str = Depends(np_._buyer)):
    """The creator's recent posts, to pick the one made for a campaign."""
    try:
        rows = np_.get_store().select("posts", {"user_id": uid}, order="created_at", desc=True, limit=30)
    except Exception:
        rows = []
    return {"posts": [{"id": p["id"], "caption": (p.get("caption") or "")[:120], "thumbnail_url": p.get("thumbnail_url"),
                       "post_type": p.get("post_type"), "created_at": p.get("created_at")}
                      for p in rows if not p.get("is_archived") and not p.get("is_private")]}


class ApplyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pitch: str = Field("", max_length=500)


@router.post("/{campaign_id}/apply")
async def apply(campaign_id: str, body: ApplyIn, uid: str = Depends(np_._buyer)):
    store = _store()
    cid = np_._uuid_or_none(campaign_id)
    c = store.one("brand_campaigns", {"id": cid}) if cid else None
    if not c or c["status"] != "open":
        raise HTTPException(status_code=404, detail="This campaign isn't open.")
    if c["owner_id"] == uid:
        raise HTTPException(status_code=400, detail="You can't apply to your own campaign.")
    level = _level(uid)
    if level["rank"] < MIN_LEVEL_RANK:
        raise HTTPException(status_code=403, detail=f"Brand deals are for {MIN_LEVEL_NAME} creators and above. "
                                                    f"You're {level['name']}: see what's left in your Creator Hub.")
    if store.one("brand_campaign_applications", {"campaign_id": cid, "creator_id": uid}):
        raise HTTPException(status_code=409, detail="You've already applied.")
    try:
        a = store.insert("brand_campaign_applications", [{"campaign_id": cid, "creator_id": uid,
                                                          "pitch": body.pitch.strip()}])[0]
    except Exception:
        raise HTTPException(status_code=409, detail="You've already applied.")
    _notify("apply", {"owner_id": c["owner_id"], "campaign_id": c["id"], "title": c["title"]})
    return _app_out(a)


class SubmitIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    post_id: str


@router.post("/applications/{app_id}/submit")
async def submit(app_id: str, body: SubmitIn, uid: str = Depends(np_._buyer)):
    pid = np_._uuid_or_none(body.post_id)
    if not pid:
        raise HTTPException(status_code=400, detail="Choose the post you made for this campaign.")
    out = _act(app_id, uid, "submit", post_id=pid)
    return _app_out(out["application"])


@router.post("/applications/{app_id}/withdraw")
async def withdraw(app_id: str, uid: str = Depends(np_._buyer)):
    out = _act(app_id, uid, "withdraw")
    return _app_out(out["application"])


# The business's decisions; registered after submit / withdraw so those paths match first.
@router.post("/applications/{app_id}/{action}")
async def decide(app_id: str, action: Literal["accept", "decline", "approve", "reject"], body: DecisionIn = DecisionIn(),
                 uid: str = Depends(np_._buyer)):
    out = _act(app_id, uid, action, note=body.note)
    return _app_out(out["application"])


# ---------------------------------------------------------------- the webhook (called from nardopay_payments)

def find_by_link_code(code: str) -> Optional[dict]:
    if not _ready():
        return None
    return np_.get_store().one("brand_campaigns", {"link_code": code})


def process_payment(c: dict, ev: dict, eid: str, finish) -> tuple[int, dict]:
    if ev["event_type"] == np_.EVENT_FAILED:
        return finish("processed", "campaign payment failed")
    if ev["event_type"] != np_.EVENT_COMPLETED:
        return finish("ignored", f"event type {ev['event_type']!r} isn't handled")
    if not ev.get("transaction_id"):
        return finish("mismatch", "no transaction id")
    if not c.get("price_rwf"):
        return finish("mismatch", "this campaign has no price to pay")
    problem = np_.verify_payment({"rwf_amount": c["price_rwf"]}, ev, fee_allowed=True)
    if problem:
        print(f"[WARN] NardoPay campaign {c['id']} not recorded: {problem}")
        return finish("mismatch", problem)
    unit = np_.amount_multiplier()
    out = np_._rpc("brand_campaign_record_payment", {
        "p_campaign": str(c["id"]), "p_transaction": ev["transaction_id"], "p_event": eid,
        "p_amount": str(ev["amount"] / unit), "p_currency": ev["currency"]}) or {}
    result = out.get("result")
    if result == "duplicate":
        return finish("processed", "campaign payment already recorded")
    if result in ("other_transaction", "transaction_used_elsewhere"):
        print(f"[WARN] NardoPay duplicate campaign payment: {c['id']}, transaction {ev['transaction_id']}")
        return finish("duplicate_payment", f"{result}: not recorded again; refund it in NardoPay")
    if result == "credited":
        return finish("processed", "paid after cancelling: added as campaign credit")
    if result != "recorded":
        raise RuntimeError(f"campaign payment not recorded ({result})")
    return finish("processed", "campaign paid")


# ---------------------------------------------------------------- maintenance (from the reconcile loop)

def maintain() -> dict:
    """Auto-approves submissions older than 7 days, closes campaigns past their deadline and cancels unpaid ones after
    24 hours; each returns unused money as credit. Safe to repeat."""
    out = {"auto_approved": 0, "expired": 0, "abandoned": 0}
    if not _ready():
        return out
    due = np_._rpc("brand_campaigns_due", {}) or {}
    for aid in due.get("auto_approve") or []:
        try:
            _act(str(aid), None, "auto_approve")
            out["auto_approved"] += 1
        except Exception as e:
            print(f"[WARN] brand deal auto-approve {aid} failed: {str(e)[:160]}")
    for mode, key in (("expire", "expire"), ("cancel", "abandon")):
        for cid in due.get(key) or []:
            r = np_._rpc("brand_campaign_close", {"p_campaign": str(cid), "p_actor": None, "p_mode": mode}) or {}
            if r.get("ok"):
                out["expired" if mode == "expire" else "abandoned"] += 1
    return out
