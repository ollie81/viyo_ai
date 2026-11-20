"""Viewer activity: the one reliable record of what viewers do, used by recommendations, trending, creator
analytics and fraud checks.

  * Written only by this backend (service role). The app sends events here; the server checks each one, looks up
    the post's series and creator itself (never trusts the app for those), and drops what it can't verify.
  * Likes, comments, shares and views are also recorded server-side by interactions.py when they happen, so they
    don't depend on the app sending them.
  * Privacy: a viewer can turn personalization off (their history is then not used for their feed) and can erase
    their own history. Events hold ids, a watched fraction and, for searches, the search text only.
"""
from __future__ import annotations

import datetime as _dt
import os
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["activity"])

supabase_admin = None
if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
    from supabase import create_client
    supabase_admin = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# What the app may report itself (likes, comments, shares and views are recorded by the server when they happen).
CLIENT_KINDS = {"impression", "progress", "complete", "save", "not_interested", "not_interested_creator", "search"}
SERVER_KINDS = {"view", "like", "comment", "share", "follow"}
MAX_BATCH = 50
MAX_PER_HOUR = 600             # per viewer; more than this is dropped (and is itself a fraud signal)
_recent: dict[str, deque] = defaultdict(deque)


def _db():
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Activity is not configured.")
    return supabase_admin


async def _viewer(authorization: str = Header(None)) -> str:
    from main import get_current_user_id
    return await get_current_user_id(authorization)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def record(user_id: str, kind: str, post_id: Optional[str] = None, series_id: Optional[str] = None,
           creator_id: Optional[str] = None, value: Optional[float] = None, query: Optional[str] = None) -> None:
    """Server-side recording (best-effort: an activity record must never break the action it describes)."""
    try:
        _db().table("viewer_events").insert({
            "user_id": user_id, "kind": kind, "post_id": post_id, "series_id": series_id, "creator_id": creator_id,
            "value": value, "query": (query or "")[:100] or None, "created_at": _now().isoformat()}).execute()
    except Exception as e:
        print(f"[WARN] activity record {kind}: {str(e)[:120]}")


def personalized(user_id: str) -> bool:
    try:
        rows = _db().table("viewer_prefs").select("*").eq("user_id", user_id).limit(1).execute().data or []
    except Exception:
        return True
    return bool(rows[0].get("personalized", True)) if rows else True


class EventIn(BaseModel):
    kind: str
    post_id: Optional[str] = None
    series_id: Optional[str] = None
    creator_id: Optional[str] = None      # only for not_interested_creator
    value: Optional[float] = None
    query: Optional[str] = Field(None, max_length=200)


class EventsIn(BaseModel):
    events: list[EventIn] = Field(..., max_length=MAX_BATCH)


def _allowed(user_id: str, n: int) -> int:
    q, now = _recent[user_id], time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    room = max(0, MAX_PER_HOUR - len(q))
    for _ in range(min(n, room)):
        q.append(now)
    return min(n, room)


@router.post("/events")
async def post_events(body: EventsIn, user_id: str = Depends(_viewer)):
    """What the viewer did in the app (impressions, watch progress, completions, saves, not interested,
    searches). Unknown kinds, unknown posts and anything past the hourly limit are dropped, not stored."""
    events = [e for e in body.events if e.kind in CLIENT_KINDS]
    events = events[:_allowed(user_id, len(events))]
    post_ids = list({e.post_id for e in events if e.post_id})
    posts = {}
    if post_ids:
        rows = _db().table("posts").select("id,user_id,series_id,is_private,is_archived").in_("id", post_ids).execute().data or []
        posts = {r["id"]: r for r in rows}
    stored = 0
    for e in events:
        if e.kind == "search":
            q = (e.query or "").strip()
            if len(q) >= 2:
                record(user_id, "search", query=q)
                stored += 1
            continue
        if e.kind == "not_interested_creator":
            if e.creator_id and e.creator_id != user_id:
                record(user_id, e.kind, creator_id=e.creator_id)
                stored += 1
            continue
        p = posts.get(e.post_id or "")
        if not p or ((p.get("is_private") or p.get("is_archived")) and p.get("user_id") != user_id):
            continue
        value = None
        if e.kind == "progress" and e.value is not None:
            value = max(0.0, min(1.0, float(e.value)))
        record(user_id, e.kind, post_id=p["id"], series_id=p.get("series_id"), creator_id=p.get("user_id"), value=value)
        stored += 1
    return {"stored": stored, "dropped": len(body.events) - stored}


class PrefsIn(BaseModel):
    personalized: bool


@router.get("/viewer/prefs")
async def get_prefs(user_id: str = Depends(_viewer)):
    return {"personalized": personalized(user_id)}


@router.put("/viewer/prefs")
async def put_prefs(body: PrefsIn, user_id: str = Depends(_viewer)):
    db = _db()
    if db.table("viewer_prefs").select("*").eq("user_id", user_id).limit(1).execute().data:
        db.table("viewer_prefs").update({"personalized": body.personalized, "updated_at": _now().isoformat()}) \
            .eq("user_id", user_id).execute()
    else:
        db.table("viewer_prefs").insert({"user_id": user_id, "personalized": body.personalized}).execute()
    return {"personalized": body.personalized}


@router.delete("/viewer/history")
async def clear_history(user_id: str = Depends(_viewer)):
    """Erases this viewer's own activity history (recommendations start fresh)."""
    _db().table("viewer_events").delete().eq("user_id", user_id).execute()
    return {"cleared": True}
