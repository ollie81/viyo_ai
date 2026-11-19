"""Fraud and abuse detection: finds suspicious patterns for a person to review. It never takes coins, money or an
account away by itself.

Money is already protected server-side (verified purchases and webhooks, the locked coin ledger, verified
rewarded ads with caps, view counting once per person every few hours). This adds:
  * Detectors over recent records, each writing at most one open flag per person/post and reason:
      - view_burst: an account watching an implausible number of different videos in a day;
      - event_flood: an account sending far more activity than a person can;
      - like_ring: a post whose likes come mostly from accounts that do nothing else;
      - free_coin_velocity: an account collecting far more free coins (missions, check-ins, ads, bonuses) a day
        than normal use allows.
  * While a flag is open, that account's activity doesn't count toward trending or recommendations' quality
    numbers (the account itself keeps working normally).
  * Review (admin, X-Admin-Key): list, clear or confirm, with a note. The person can see flags about them and
    appeal in their own words; an appeal is shown to the reviewer.
Thresholds are deliberately high so ordinary heavy users aren't flagged.
"""
from __future__ import annotations

import datetime as _dt
import os
from collections import defaultdict
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

import activity

router = APIRouter(prefix="/api/v1", tags=["fraud"])

VIEW_BURST_POSTS_PER_DAY = 400
EVENT_FLOOD_PER_DAY = 5000
LIKE_RING_MIN_LIKERS = 10
LIKE_RING_SHARE = 0.6
FREE_COINS_PER_DAY = 1500
FREE_COIN_TYPES_EXCLUDED = {"coin_purchase", "ad_promotion_refund", "studio_refund"}


def _admin(x_admin_key: str = Header(None)) -> None:
    key = os.environ.get("ADMIN_API_KEY", "")
    if not key or x_admin_key != key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


def _open_flag(db, subject_type: str, subject_id: str, reason: str, detail: dict, score: float) -> bool:
    existing = (db.table("fraud_flags").select("*").eq("subject_type", subject_type).eq("subject_id", subject_id)
                .eq("reason", reason).eq("status", "open").execute().data) or []
    if existing:
        return False
    db.table("fraud_flags").insert({"subject_type": subject_type, "subject_id": subject_id, "reason": reason,
                                    "detail": detail, "score": score, "status": "open"}).execute()
    return True


def scan(db, now: Optional[_dt.datetime] = None) -> dict:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    since = (now - _dt.timedelta(days=1)).isoformat()
    events = db.table("viewer_events").select("*").gte("created_at", since).execute().data or []
    created = []
    per_user_posts, per_user_count = defaultdict(set), defaultdict(int)
    likers = defaultdict(set)
    active_kinds = defaultdict(set)
    for e in events:
        u = e.get("user_id")
        per_user_count[u] += 1
        active_kinds[u].add(e["kind"])
        if e["kind"] in ("view", "progress") and e.get("post_id"):
            per_user_posts[u].add(e["post_id"])
        if e["kind"] == "like" and e.get("post_id"):
            likers[e["post_id"]].add(u)
    for u, posts in per_user_posts.items():
        if len(posts) >= VIEW_BURST_POSTS_PER_DAY and _open_flag(
                db, "user", u, "view_burst", {"distinct_videos_24h": len(posts)}, len(posts) / VIEW_BURST_POSTS_PER_DAY):
            created.append(("user", u, "view_burst"))
    for u, n in per_user_count.items():
        if n >= EVENT_FLOOD_PER_DAY and _open_flag(db, "user", u, "event_flood", {"events_24h": n}, n / EVENT_FLOOD_PER_DAY):
            created.append(("user", u, "event_flood"))
    for pid, us in likers.items():
        if len(us) < LIKE_RING_MIN_LIKERS:
            continue
        idle = [u for u in us if active_kinds[u] <= {"like"}]
        share = len(idle) / len(us)
        if share >= LIKE_RING_SHARE and _open_flag(db, "post", pid, "like_ring",
                                                   {"likers_24h": len(us), "likers_doing_nothing_else": len(idle)}, share):
            created.append(("post", pid, "like_ring"))
    try:
        tx = db.table("transactions").select("*").gte("created_at", since).execute().data or []
    except Exception:
        tx = []
    free = defaultdict(int)
    for t in tx:
        amt = int(t.get("amount") or 0)
        if amt > 0 and (t.get("type") or "") not in FREE_COIN_TYPES_EXCLUDED and not str(t.get("type") or "").startswith("coin_purchase"):
            free[t.get("user_id")] += amt
    for u, coins in free.items():
        if coins >= FREE_COINS_PER_DAY and _open_flag(db, "user", u, "free_coin_velocity", {"free_coins_24h": coins},
                                                      coins / FREE_COINS_PER_DAY):
            created.append(("user", u, "free_coin_velocity"))
    return {"flags_created": [{"subject_type": a, "subject_id": b, "reason": c} for a, b, c in created],
            "checked_events": len(events), "checked_transactions": len(tx)}


@router.post("/admin/fraud/scan", dependencies=[Depends(_admin)])
async def run_scan():
    """Run the detectors over the last 24 hours (safe to run often; open flags aren't duplicated)."""
    return scan(activity._db())


@router.get("/admin/fraud/flags", dependencies=[Depends(_admin)])
async def list_flags(status: Literal["open", "cleared", "confirmed"] = "open"):
    rows = activity._db().table("fraud_flags").select("*").eq("status", status).order("created_at", desc=True).execute().data or []
    return rows


class ResolveIn(BaseModel):
    status: Literal["cleared", "confirmed"]
    note: str = Field("", max_length=1000)


@router.post("/admin/fraud/flags/{flag_id}/resolve", dependencies=[Depends(_admin)])
async def resolve(flag_id: str, body: ResolveIn):
    db = activity._db()
    rows = db.table("fraud_flags").select("*").eq("id", flag_id).limit(1).execute().data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found.")
    db.table("fraud_flags").update({"status": body.status, "reviewed_note": body.note,
                                    "reviewed_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}).eq("id", flag_id).execute()
    return {**rows[0], "status": body.status, "reviewed_note": body.note}


async def _user(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest
    return await get_current_user_id_no_guest(authorization)


def _reason_text(reason: str) -> str:
    return {"view_burst": "An unusually large number of videos watched in one day.",
            "event_flood": "Far more app activity than one person usually makes.",
            "free_coin_velocity": "An unusually large number of free coins collected in one day.",
            "like_ring": "Many of this post's likes came from accounts that did nothing else."}.get(reason, reason)


@router.get("/account/reviews")
async def my_reviews(user_id: str = Depends(_user)):
    """Reviews about this person's account or posts, so nothing is decided about them in secret."""
    db = activity._db()
    mine = db.table("fraud_flags").select("*").eq("subject_type", "user").eq("subject_id", user_id).execute().data or []
    post_ids = [p["id"] for p in (db.table("posts").select("id").eq("user_id", user_id).execute().data or [])]
    posts = db.table("fraud_flags").select("*").eq("subject_type", "post").in_("subject_id", post_ids).execute().data or [] if post_ids else []
    return [{"id": f["id"], "about": f["subject_type"], "subject_id": f["subject_id"], "reason": _reason_text(f["reason"]),
             "status": f["status"], "appeal": f.get("appeal"), "reviewed_note": f.get("reviewed_note"),
             "created_at": f.get("created_at"),
             "effect": "While this is open, this activity doesn't count toward trending. Nothing else changes."}
            for f in mine + posts]


class AppealIn(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000)


@router.post("/account/reviews/{flag_id}/appeal")
async def appeal(flag_id: str, body: AppealIn, user_id: str = Depends(_user)):
    db = activity._db()
    rows = db.table("fraud_flags").select("*").eq("id", flag_id).limit(1).execute().data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found.")
    f = rows[0]
    owns = f["subject_type"] == "user" and f["subject_id"] == user_id
    if f["subject_type"] == "post":
        p = db.table("posts").select("*").eq("id", f["subject_id"]).limit(1).execute().data or []
        owns = bool(p) and p[0].get("user_id") == user_id
    if not owns:
        raise HTTPException(status_code=404, detail="Not found.")
    db.table("fraud_flags").update({"appeal": body.text.strip(),
                                    "appealed_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}).eq("id", flag_id).execute()
    return {"appealed": True}
