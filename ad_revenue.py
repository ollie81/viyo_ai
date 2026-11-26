"""Ad revenue sharing (ad_revenue.sql): creators get half of what VIYO's ads earn on their content.

Rules (agreed with the owner):
  * Each month an admin enters the real totals from the AdMob report: in-feed ads (interstitials) and episode-unlock
    ads (rewarded). Nothing is estimated.
  * CREATOR_SHARE (50%) of each total goes to creators, split by the ads that ran on each creator's content that
    month: Google-verified unlock ads (2 per episode unlocked with ads), and in-feed ads next to any of their posts
    (dramas, videos, long videos...), counted once per viewer per post per day, never on their own posts.
  * Every creator takes part (no level needed). Ads watched for coins aren't tied to content and aren't shared.
  * Paid as an earning of kind 'ads', once per creator per month, held like every new earning (their level's hold).
Since creators share the real total, extra counts can only move money between creators, never raise what VIYO pays."""
from __future__ import annotations

import datetime as _dt
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/api/v1", tags=["ad-revenue"])

CREATOR_SHARE = 0.50
DAILY_IMPRESSION_CAP = 40             # in-feed ads counted per viewer per day


def _store():
    import nardopay_payments as np_
    return np_.get_store()


def _rpc(name: str, params: dict):
    return _store().rpc(name, params)


def _ready() -> bool:
    try:
        _store().select("ad_revenue_months", {}, limit=1)
        return True
    except Exception:
        return False


async def _user(authorization: str = Header(None)) -> str:
    from main import _decode_token
    payload = _decode_token(authorization)
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Token missing subject")
    return uid


def _require_admin(x_admin_key: str = Header(None)) -> None:
    import hmac
    key = os.environ.get("ADMIN_API_KEY", "")
    if not key:
        raise HTTPException(status_code=503, detail="Admin tools are not configured.")
    if not x_admin_key or not hmac.compare_digest(x_admin_key, key):
        raise HTTPException(status_code=403, detail="Invalid admin key.")


def _uuid(v) -> Optional[str]:
    import uuid
    try:
        return str(uuid.UUID(str(v)))
    except (ValueError, TypeError):
        return None


def _month(s: str) -> _dt.date:
    try:
        y, m = s.split("-")
        return _dt.date(int(y), int(m), 1)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Month must look like 2026-11.")


def _this_month() -> _dt.date:
    t = _dt.datetime.now(_dt.timezone.utc).date()
    return t.replace(day=1)


# ---------------------------------------------------------------- the app reports an in-feed ad next to a post

class ImpressionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    post_id: str


@router.post("/ads/impression")
async def impression(body: ImpressionIn, uid: str = Depends(_user)):
    """Called after an in-feed ad was actually shown next to a post. Counted once per viewer per post per day."""
    pid = _uuid(body.post_id)
    if not pid or not _ready():
        return {"counted": False}
    try:
        out = _rpc("ad_impression_record", {"p_viewer": uid, "p_post": pid, "p_daily_cap": DAILY_IMPRESSION_CAP}) or {}
    except Exception:
        return {"counted": False}
    return {"counted": bool(out.get("counted"))}


# ---------------------------------------------------------------- creators

def creator_summary(creator_id: str) -> dict:
    """This month's ads so far (no dollars until the month is closed) and the creator's closed months."""
    import wallet
    if not _ready():
        return {"available": False}
    now = _rpc("ad_revenue_counts", {"p_month": _this_month().isoformat(), "p_creator": creator_id}) or {}
    mine = (now.get("creators") or [{}])[0] if now.get("creators") else {}
    rows = _store().select("ad_revenue_payouts", {"creator_id": creator_id}, order="month", desc=True, limit=12)
    return {
        "available": True,
        "creator_share": CREATOR_SHARE,
        "this_month": {"month": _this_month().isoformat()[:7], "feed_ads": int(mine.get("interstitial") or 0),
                       "unlock_ads": int(mine.get("unlock") or 0)},
        "months": [{"month": str(r["month"])[:7], "feed_ads": int(r.get("interstitial_count") or 0),
                    "unlock_ads": int(r.get("unlock_count") or 0), "usd": round(float(r.get("usd") or 0), 2),
                    "coins": int(r.get("coins") or 0),
                    "credited_usd": wallet.coins_to_usd(int(r.get("coins") or 0))} for r in rows],
        "note": "Paid once a month from VIYO's real ad earnings: half goes to creators, shared by the ads shown on "
                "their content.",
    }


@router.get("/creator/ad-revenue")
async def my_ad_revenue(uid: str = Depends(_user)):
    return creator_summary(uid)


# ---------------------------------------------------------------- admin: see a month, then close it

@router.get("/admin/ad-revenue/{month}", dependencies=[Depends(_require_admin)])
async def admin_month(month: str):
    m = _month(month)
    counts = _rpc("ad_revenue_counts", {"p_month": m.isoformat(), "p_creator": None}) or {}
    closed = _store().one("ad_revenue_months", {"month": m.isoformat()})
    return {"month": month, "closed": closed is not None, "summary": closed,
            "feed_ads": int(counts.get("interstitial_total") or 0), "unlock_ads": int(counts.get("unlock_total") or 0),
            "creators": len(counts.get("creators") or []), "creator_share": CREATOR_SHARE,
            "can_close": m < _this_month() and closed is None}


class CloseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feed_usd: float = Field(..., ge=0, le=1_000_000)       # AdMob: in-feed (interstitial) ad unit, that month
    unlock_usd: float = Field(..., ge=0, le=1_000_000)     # AdMob: episode-unlock (rewarded) ad unit, that month
    note: str = Field("", max_length=300)


@router.post("/admin/ad-revenue/{month}/close", dependencies=[Depends(_require_admin)])
async def admin_close(month: str, body: CloseIn):
    import creator_levels
    import wallet
    m = _month(month)
    if m >= _this_month():
        raise HTTPException(status_code=409, detail="A month can be closed only after it's over.")
    store = _store()
    counts = _rpc("ad_revenue_counts", {"p_month": m.isoformat(), "p_creator": None}) or {}
    holds = {c["creator_id"]: creator_levels.hold_days_for(store, str(c["creator_id"]))
             for c in counts.get("creators") or []}
    out = _rpc("ad_revenue_close", {
        "p_month": m.isoformat(), "p_interstitial_usd": f"{body.feed_usd:.2f}", "p_unlock_usd": f"{body.unlock_usd:.2f}",
        "p_share": str(CREATOR_SHARE), "p_coin_usd": str(wallet.COIN_USD),
        "p_default_hold": wallet.EARNINGS_HOLD_DAYS, "p_holds": holds, "p_note": body.note.strip()}) or {}
    if not out.get("ok"):
        err = out.get("error")
        raise HTTPException(status_code=409, detail={"already_closed": "This month is already closed.",
                                                     "month_not_over": "A month can be closed only after it's over."}
                            .get(err, "The month couldn't be closed."))
    _notify(m, store)
    return out


def _notify(m: _dt.date, store) -> None:
    try:
        from notify import notify
        for r in store.select("ad_revenue_payouts", {"month": m.isoformat()}, limit=5000):
            if int(r.get("coins") or 0) > 0:
                notify(str(r["creator_id"]), "ad_revenue",
                       f"Your ad revenue for {m.strftime('%B %Y')} is in: ${float(r['usd']):.2f}.",
                       push_title="Ad revenue 📺")
    except Exception as e:
        print(f"[WARN] ad revenue notifications failed: {e}")
