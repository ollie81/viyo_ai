"""Paid promotion of published ads inside VIYO.

A creator pays (in VIYO Coins, the existing wallet and ledger) to show one of their own published, public ads
to more people in Discover. Rules this file keeps:

  * Switched off until approved: PROMOTIONS_ENABLED must be "true" before anyone can start one.
  * Transparent price, shown first (GET .../quote): a fixed number of coins per 1,000 impressions, the same
    for everyone and every country. The quote says "up to" so many impressions: delivery depends on how many
    people use Discover, and results (views, likes, sales) are never promised.
  * Budget limit: the whole budget is taken when the promotion starts and can never be exceeded; each counted
    impression uses price/1000 coins of it; spending is paced evenly over the chosen days.
  * Whatever isn't used goes back to the creator when the promotion ends: at its end date, when the budget is
    used up, when the creator stops it, or when the ad stops being public. Refunded exactly once.
  * Only real, counted reporting: an impression is counted once per viewer per promotion per day (never the
    creator's own views), a click (the Visit button) likewise. Engagement during the promotion is reported as
    the post's own counts, labelled as all engagement in that period (paid and organic can't be split per view).
  * Promoted content is always labelled "Promoted" in the app, and only an ad post (post_kind "ad") that its
    owner published publicly can be promoted. Drama playback never shows promotions.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

import ads_studio as core
from coins import credit_coins, debit_coins
from creator_studio import creator

router = APIRouter(prefix="/api/v1", tags=["ads_promotion"])

COINS_PER_1000 = int(os.environ.get("PROMO_COINS_PER_1000", "50"))   # ~US$0.50 per 1,000 impressions
MIN_BUDGET, MAX_BUDGET = 100, 20000
MIN_DAYS, MAX_DAYS = 1, 14
PLACEMENTS = {"for_you": "Discover: For You", "ads": "Discover: Ads"}
MAX_ACTIVE_PER_USER = 5


def enabled() -> bool:
    return os.environ.get("PROMOTIONS_ENABLED", "false").lower() == "true"


def _db():
    if core.supabase_admin is None:
        raise HTTPException(status_code=503, detail="Promotion is not configured.")
    return core.supabase_admin


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _parse(ts) -> Optional[_dt.datetime]:
    if not ts:
        return None
    if isinstance(ts, _dt.datetime):
        return ts
    return _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


async def _viewer(authorization: str = Header(None)) -> str:
    from main import get_current_user_id
    return await get_current_user_id(authorization)


def quote(budget: int, days: int) -> dict:
    if not MIN_BUDGET <= budget <= MAX_BUDGET:
        raise HTTPException(status_code=400, detail=f"Budget: {MIN_BUDGET} to {MAX_BUDGET} coins.")
    if not MIN_DAYS <= days <= MAX_DAYS:
        raise HTTPException(status_code=400, detail=f"Length: {MIN_DAYS} to {MAX_DAYS} days.")
    return {"budget_coins": budget, "days": days, "coins_per_1000_impressions": COINS_PER_1000,
            "max_impressions": budget * 1000 // COINS_PER_1000,
            "per_day_coins": round(budget / days, 1),
            "terms": ["You pay the budget now. Each person who sees your ad in Discover (once a day) uses "
                      f"{COINS_PER_1000} coins per 1,000.",
                      "Unused coins come back when the promotion ends, when you stop it, or if your ad stops "
                      "being public.",
                      "Delivery is up to the number above, depending on how many people use Discover. Views, likes "
                      "and sales are not guaranteed.",
                      "Your ad is labelled \"Promoted\" while it runs."]}


def _spent_coins(p: dict) -> int:
    return math.ceil(int(p.get("spent_millicoins") or 0) / 1000)


def _promo_out(p: dict) -> dict:
    spent = min(_spent_coins(p), int(p.get("budget_coins") or 0))
    return {"id": p["id"], "post_id": p.get("post_id"), "status": p.get("status"), "budget_coins": p.get("budget_coins"),
            "coins_per_1000_impressions": p.get("price_per_1000"), "days": p.get("days"),
            "placements": p.get("placements") or list(PLACEMENTS), "start_at": p.get("start_at"), "end_at": p.get("end_at"),
            "impressions": int(p.get("impressions") or 0), "clicks": int(p.get("clicks") or 0),
            "coins_spent": spent, "coins_refunded": p.get("refunded_coins"), "ended_reason": p.get("ended_reason")}


def _post(post_id: str) -> Optional[dict]:
    rows = _db().table("posts").select("*").eq("id", post_id).limit(1).execute().data or []
    return rows[0] if rows else None


def settle(p: dict, reason: Optional[str] = None) -> dict:
    """Ends a promotion if it should end (or [reason] is given) and returns the unused coins, exactly once."""
    if p.get("status") != "active":
        return p
    if reason is None:
        post = _post(p["post_id"])
        if _now() >= (_parse(p.get("end_at")) or _now()):
            reason = "finished"
        elif int(p.get("spent_millicoins") or 0) >= int(p["budget_coins"]) * 1000:
            reason = "budget_used"
        elif not post or post.get("is_private") or post.get("is_archived") or post.get("user_id") != p.get("owner_id"):
            reason = "ad_not_public"
        else:
            return p
    refund = max(0, int(p["budget_coins"]) - _spent_coins(p))
    patch = {"status": "ended", "ended_reason": reason, "ended_at": _now().isoformat(), "refunded_coins": refund}
    won = _db().table("ad_promotions").update(patch).eq("id", p["id"]).eq("status", "active").execute().data
    if not won:                                     # someone else ended it first: no second refund
        rows = _db().table("ad_promotions").select("*").eq("id", p["id"]).limit(1).execute().data or [p]
        return rows[0]
    if refund:
        credit_coins(_db(), p["owner_id"], refund, "ad_promotion_refund",
                     f"Unused promotion budget ({reason.replace('_', ' ')})", external_ref=f"promo-refund:{p['id']}")
    return {**p, **patch}


# ---------------------------------------------------------------- creator

class PromoteIn(BaseModel):
    post_id: str
    budget_coins: int
    days: int = 3
    placements: list[str] = Field(default_factory=lambda: list(PLACEMENTS))


@router.get("/ads-maker/promotions/quote")
async def get_quote(budget: int = 500, days: int = 3, user_id: str = Depends(creator)):
    """The full price and what it buys, before anything is paid."""
    return {**quote(budget, days), "enabled": enabled()}


@router.post("/ads-maker/promotions")
async def start_promotion(body: PromoteIn, user_id: str = Depends(creator)):
    if not enabled():
        raise HTTPException(status_code=503, detail="Paid promotion isn't open yet.")
    q = quote(body.budget_coins, body.days)
    placements = list(dict.fromkeys(p for p in body.placements if p in PLACEMENTS))
    if not placements:
        raise HTTPException(status_code=400, detail="Pick where to show it.")
    post = _post(body.post_id)
    if not post or post.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    if post.get("post_kind") != "ad" or post.get("is_private") or post.get("is_archived"):
        raise HTTPException(status_code=400, detail="Only an ad you published publicly can be promoted.")
    mine = _db().table("ad_promotions").select("*").eq("owner_id", user_id).eq("status", "active").execute().data or []
    mine = [settle(p) for p in mine]
    if any(p.get("status") == "active" and p.get("post_id") == post["id"] for p in mine):
        raise HTTPException(status_code=400, detail="This ad is already being promoted.")
    if len([p for p in mine if p.get("status") == "active"]) >= MAX_ACTIVE_PER_USER:
        raise HTTPException(status_code=400, detail=f"Up to {MAX_ACTIVE_PER_USER} promotions at a time.")
    debit_coins(_db(), user_id, q["budget_coins"], "ad_promotion", "Ad promotion budget")   # 402 if not enough
    start = _now()
    try:
        row = _db().table("ad_promotions").insert({
            "owner_id": user_id, "post_id": post["id"], "campaign_id": post.get("ad_campaign_id"),
            "budget_coins": q["budget_coins"], "price_per_1000": COINS_PER_1000, "days": q["days"],
            "placements": placements, "status": "active", "start_at": start.isoformat(),
            "end_at": (start + _dt.timedelta(days=q["days"])).isoformat(), "impressions": 0, "clicks": 0,
            "spent_millicoins": 0}).execute().data[0]
    except Exception:
        credit_coins(_db(), user_id, q["budget_coins"], "ad_promotion_refund", "Promotion couldn't start")
        raise HTTPException(status_code=500, detail="The promotion couldn't start. Your coins were returned.")
    return _promo_out(row)


@router.get("/ads-maker/promotions")
async def my_promotions(user_id: str = Depends(creator)):
    rows = (_db().table("ad_promotions").select("*").eq("owner_id", user_id).order("created_at", desc=True)
            .execute().data) or []
    out = []
    for p in rows:
        p = settle(p)
        post = _post(p["post_id"]) or {}
        out.append({**_promo_out(p), "caption": post.get("caption") or "", "thumbnail_url": post.get("thumbnail_url"),
                    "post_engagement": {k: int(post.get(c) or 0) for k, c in
                                        (("views", "view_count"), ("likes", "like_count"),
                                         ("comments", "comment_count"), ("shares", "share_count"))}})
    return {"enabled": enabled(), "promotions": out,
            "note": "Impressions and clicks are counted only from the promotion. The post's views, likes, comments "
                    "and shares are all of its engagement (paid and organic can't be split per view)."}


def _owned(promo_id: str, user_id: str) -> dict:
    import uuid as _u
    try:
        pid = str(_u.UUID(promo_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found.")
    rows = _db().table("ad_promotions").select("*").eq("id", pid).limit(1).execute().data or []
    if not rows or rows[0].get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    return rows[0]


@router.post("/ads-maker/promotions/{promo_id}/stop")
async def stop_promotion(promo_id: str, user_id: str = Depends(creator)):
    """Stops now and returns the unused coins."""
    p = settle(_owned(promo_id, user_id), reason="stopped")
    return _promo_out(p)


# ---------------------------------------------------------------- serving and counting

def _today() -> str:
    return _now().date().isoformat()


def _spent_today_millicoins(p: dict) -> int:
    rows = (_db().table("ad_promotion_events").select("*").eq("promotion_id", p["id"]).eq("day", _today())
            .eq("kind", "impression").execute().data) or []
    return len(rows) * int(p.get("price_per_1000") or COINS_PER_1000)


def eligible(placement: str, viewer: str) -> list[dict]:
    """Active promotions that can be shown here now: not the viewer's own, the ad still public, within today's
    even share of the budget. Least-delivered first, so promotions share the space fairly."""
    if placement not in PLACEMENTS:
        return []
    rows = _db().table("ad_promotions").select("*").eq("status", "active").execute().data or []
    out = []
    for p in rows:
        p = settle(p)
        if p.get("status") != "active" or p.get("owner_id") == viewer or placement not in (p.get("placements") or []):
            continue
        daily = int(p["budget_coins"]) * 1000 / max(1, int(p.get("days") or 1))
        if _spent_today_millicoins(p) >= daily:
            continue
        out.append(p)
    out.sort(key=lambda p: int(p.get("spent_millicoins") or 0) / max(1, int(p["budget_coins"])))
    return out


@router.get("/promotions/serve")
async def serve(placement: str = "for_you", limit: int = 2, viewer: str = Depends(_viewer)):
    """Promoted ad posts for this placement (always shown labelled "Promoted"). Empty when promotion is off."""
    if not enabled():
        return {"promotions": []}
    picks = eligible(placement, viewer)[:max(0, min(int(limit), 3))]
    return {"promotions": [{"promotion_id": p["id"], "post_id": p["post_id"]} for p in picks]}


def _count(promo_id: str, viewer: str, kind: str) -> dict:
    import uuid as _u
    try:
        pid = str(_u.UUID(promo_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found.")
    rows = _db().table("ad_promotions").select("*").eq("id", pid).limit(1).execute().data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found.")
    p = settle(rows[0])
    if p.get("status") != "active" or p.get("owner_id") == viewer:
        return {"counted": False}
    seen = (_db().table("ad_promotion_events").select("*").eq("promotion_id", p["id"]).eq("viewer_id", viewer)
            .eq("day", _today()).eq("kind", kind).execute().data) or []
    if seen:
        return {"counted": False}
    if kind == "impression":
        price = int(p.get("price_per_1000") or COINS_PER_1000)
        cur = int(p.get("spent_millicoins") or 0)
        if cur + price > int(p["budget_coins"]) * 1000:
            settle(p, reason="budget_used")
            return {"counted": False}
        if _spent_today_millicoins(p) + price > int(p["budget_coins"]) * 1000 / max(1, int(p.get("days") or 1)):
            return {"counted": False}
        try:
            _db().table("ad_promotion_events").insert({"promotion_id": p["id"], "viewer_id": viewer, "day": _today(),
                                                       "kind": kind}).execute()
        except Exception:
            return {"counted": False}               # the unique key: counted already
        won = (_db().table("ad_promotions").update({"spent_millicoins": cur + price,
                                                    "impressions": int(p.get("impressions") or 0) + 1})
               .eq("id", p["id"]).eq("spent_millicoins", cur).execute().data)
        if not won:                                  # lost a race: re-read and add once
            fresh = _db().table("ad_promotions").select("*").eq("id", p["id"]).limit(1).execute().data[0]
            _db().table("ad_promotions").update({
                "spent_millicoins": min(int(fresh["spent_millicoins"]) + price, int(p["budget_coins"]) * 1000),
                "impressions": int(fresh.get("impressions") or 0) + 1}).eq("id", p["id"]).execute()
        return {"counted": True}
    try:
        _db().table("ad_promotion_events").insert({"promotion_id": p["id"], "viewer_id": viewer, "day": _today(),
                                                   "kind": kind}).execute()
    except Exception:
        return {"counted": False}
    _db().table("ad_promotions").update({"clicks": int(p.get("clicks") or 0) + 1}).eq("id", p["id"]).execute()
    return {"counted": True}


@router.post("/promotions/{promo_id}/impression")
async def impression(promo_id: str, viewer: str = Depends(_viewer)):
    """Counts one view of a promoted ad (once per person per day; never the advertiser's own)."""
    return _count(promo_id, viewer, "impression")


@router.post("/promotions/{promo_id}/click")
async def click(promo_id: str, viewer: str = Depends(_viewer)):
    """Counts one tap on the promoted ad's Visit button (once per person per day). Clicks cost nothing extra."""
    return _count(promo_id, viewer, "click")
