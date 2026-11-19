"""Creator analytics and growth advice, from real records only.

  * Unique viewers, completion, watch time and episode-to-episode retention come from viewer_events and
    watch_progress (where each viewer actually got to); followers from follows; earnings from creator_earnings.
  * Watch time is the sum of how far each viewer got in each video (their furthest point), so it never counts
    the same seconds twice and never guesses.
  * Advice is a short list of rules applied to those numbers, each shown with the numbers it's based on, and only
    when enough people watched (MIN_SAMPLE). It describes what the data shows; it never promises growth.
"""
from __future__ import annotations

import datetime as _dt
from collections import defaultdict

from fastapi import APIRouter, Depends, Header, HTTPException

import activity

router = APIRouter(prefix="/api/v1/creator", tags=["creator_analytics"])

MIN_SAMPLE = 20
COMPLETE_AT = 0.9


async def _creator(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest
    return await get_current_user_id_no_guest(authorization)


def _day(ts) -> str:
    return str(ts or "")[:10]


def analytics(db, user_id: str, days: int = 30) -> dict:
    now = _dt.datetime.now(_dt.timezone.utc)
    days = max(7, min(int(days), 90))
    since = (now - _dt.timedelta(days=days)).isoformat()
    posts = [p for p in (db.table("posts").select("*").eq("user_id", user_id).execute().data or []) if not p.get("is_archived")]
    series = {s["id"]: s for s in (db.table("series").select("*").eq("user_id", user_id).execute().data or [])}
    by_post = {p["id"]: p for p in posts}
    events = [e for e in (db.table("viewer_events").select("*").eq("creator_id", user_id).gte("created_at", since)
                          .execute().data or []) if e.get("user_id") != user_id]
    try:
        wp = [r for r in (db.table("watch_progress").select("*").in_("post_id", list(by_post)).execute().data or [])
              if r.get("user_id") != user_id] if by_post else []
    except Exception:
        wp = []

    viewers, completers, liked = defaultdict(set), defaultdict(set), defaultdict(set)
    furthest = {}                                     # (user, post) -> seconds reached
    for e in events:
        pid, u = e.get("post_id"), e.get("user_id")
        if pid not in by_post:
            continue
        if e["kind"] in ("view", "progress", "complete"):
            viewers[pid].add(u)
        if e["kind"] == "complete" or (e["kind"] == "progress" and (e.get("value") or 0) >= COMPLETE_AT):
            completers[pid].add(u)
        if e["kind"] in ("like", "comment", "share", "save"):
            liked[pid].add(u)
        if e["kind"] == "progress" and e.get("value") is not None and by_post[pid].get("duration_seconds"):
            furthest[(u, pid)] = max(furthest.get((u, pid), 0), float(e["value"]) * float(by_post[pid]["duration_seconds"]))
    for r in wp:
        pid, u = r.get("post_id"), r.get("user_id")
        if pid not in by_post or str(r.get("updated_at") or now.isoformat()) < since:
            continue
        viewers[pid].add(u)
        pos, dur = (r.get("position_ms") or 0) / 1000, (r.get("duration_ms") or 0) / 1000
        furthest[(u, pid)] = max(furthest.get((u, pid), 0), pos)
        if dur and pos / dur >= COMPLETE_AT:
            completers[pid].add(u)
    watch_seconds = defaultdict(float)
    for (u, pid), sec in furthest.items():
        watch_seconds[pid] += sec

    productions, advice = [], []

    def summary(pids):
        v = set().union(*(viewers[p] for p in pids)) if pids else set()
        c = set().union(*(completers[p] for p in pids)) if pids else set()
        e = set().union(*(liked[p] for p in pids)) if pids else set()
        return v, c, e

    groups = defaultdict(list)
    for p in posts:
        groups[("series", p["series_id"]) if p.get("series_id") else ("video", p["id"])].append(p)
    for (kind, gid), lst in groups.items():
        lst.sort(key=lambda p: int(p.get("episode_number") or 0))
        pids = [p["id"] for p in lst]
        v, c, e = summary(pids)
        per_ep = []
        if kind == "series":
            first = viewers[pids[0]] if pids else set()
            for p in lst:
                vv = viewers[p["id"]]
                per_ep.append({"episode": p.get("episode_number"), "viewers": len(vv),
                               "completion": round(len(completers[p["id"]]) / len(vv), 3) if vv else None,
                               "retention_from_first": round(len(vv & first) / len(first), 3) if first else None})
        title = (series.get(gid) or {}).get("title") if kind == "series" else (lst[0].get("caption") or "").split("\n")[0][:80]
        prod = {"kind": kind, "id": gid, "title": title or "", "unique_viewers": len(v),
                "completion_rate": round(sum(len(completers[p]) for p in pids) / max(1, sum(len(viewers[p]) for p in pids)), 3)
                if any(viewers[p] for p in pids) else None,
                "watch_minutes": round(sum(watch_seconds[p] for p in pids) / 60, 1),
                "engaged_viewers": len(e), "episodes": per_ep, "views_counter": sum(int(p.get("view_count") or 0) for p in lst)}
        productions.append(prod)
        name = f'"{prod["title"]}"' if prod["title"] else "this production"
        if len(v) >= MIN_SAMPLE:
            if kind == "series" and len(per_ep) >= 2 and per_ep[1]["retention_from_first"] is not None \
                    and per_ep[0]["viewers"] >= MIN_SAMPLE and per_ep[1]["retention_from_first"] < 0.4:
                advice.append({"production_id": gid, "text": f"Most viewers of {name} stop after episode 1: only "
                               f"{per_ep[1]['retention_from_first']:.0%} go on to episode 2. A stronger cliffhanger at the "
                               f"end of episode 1 usually helps.", "based_on": f"{per_ep[0]['viewers']} episode-1 viewers"})
            if prod["completion_rate"] is not None and prod["completion_rate"] < 0.3:
                advice.append({"production_id": gid, "text": f"Viewers often leave {name} before the end "
                               f"({prod['completion_rate']:.0%} finish). A faster opening or a shorter cut may keep them.",
                               "based_on": f"{len(v)} viewers"})
            elif prod["completion_rate"] is not None and prod["completion_rate"] >= 0.7 and len(v) < 200:
                advice.append({"production_id": gid, "text": f"People who start {name} finish it "
                               f"({prod['completion_rate']:.0%}). It's worth sharing it to reach more people.",
                               "based_on": f"{len(v)} viewers"})
        if kind == "series" and len(lst) <= 2 and len(v) >= 5:
            advice.append({"production_id": gid, "text": f"{name} has {len(lst)} episode(s). For You brings back "
                           "series viewers are in the middle of, so more episodes give your viewers a reason to return.",
                           "based_on": f"{len(v)} viewers, {len(lst)} episodes"})
    productions.sort(key=lambda p: p["unique_viewers"], reverse=True)

    try:
        follows = db.table("follows").select("*").eq("following_id", user_id).execute().data or []
    except Exception:
        follows = []
    fdays = defaultdict(int)
    for f in follows:
        if f.get("created_at") and str(f["created_at"]) >= since:
            fdays[_day(f["created_at"])] += 1
    try:
        earn = db.table("creator_earnings").select("*").eq("creator_id", user_id).execute().data or []
    except Exception:
        earn = []
    edays = defaultdict(int)
    for e in earn:
        if str(e.get("created_at") or "") >= since:
            edays[_day(e["created_at"])] += int(e.get("coins") or 0)
    all_v, all_c, all_e = summary(list(by_post))
    if not advice and len(all_v) < MIN_SAMPLE:
        advice.append({"production_id": None, "text": f"Advice appears once at least {MIN_SAMPLE} people have watched "
                       f"a production (so far: {len(all_v)}).", "based_on": "not enough viewers yet"})
    return {
        "days": days,
        "totals": {"unique_viewers": len(all_v), "completers": len(all_c), "engaged_viewers": len(all_e),
                   "watch_minutes": round(sum(watch_seconds.values()) / 60, 1),
                   "followers": len(follows), "new_followers": sum(fdays.values()),
                   "earned_coins": sum(edays.values())},
        "followers_by_day": [{"day": d, "count": n} for d, n in sorted(fdays.items())],
        "earnings_by_day": [{"day": d, "coins": n} for d, n in sorted(edays.items())],
        "productions": productions,
        "advice": advice[:6],
        "note": "Unique viewers, completion and watch time count each person once and only what they actually "
                "watched; the creator's own viewing is excluded. Older activity from before these records existed "
                "isn't included.",
    }


@router.get("/analytics")
async def get_analytics(days: int = 30, user_id: str = Depends(_creator)):
    return analytics(activity._db(), user_id, days)
