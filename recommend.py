"""VIYO For You: a personal, explainable ranking. Plain statistics, no paid AI.

Philosophy (in code, not only in words): help viewers find great stories, help good creators get found.
  * Quality, not raw views: completion rate and like rate from distinct viewers, smoothed (Bayesian) so a title
    with 3 viewers can't look perfect and a title with huge view counts gets no automatic advantage.
  * Personal: genres, kinds of title and creators the viewer finished, liked, shared or saved; their searches;
    a series they're in the middle of comes back as its next episode ("Continue watching").
  * Fresh and fair: newer titles rank higher; small creators with good early numbers get a lift; every fifth
    place goes to something few people have seen yet; at most two titles per creator in any eight in a row.
  * Respectful: "Not interested" removes a title (or a creator, for 30 days); finished titles don't come back;
    titles shown again and again without being opened sink; turning personalization off removes all history.
  * New accounts get a mix of high-quality, fresh and diverse titles.

A recommendation unit is a series (shown as the viewer's next episode) or a standalone video/photo post.
"""
from __future__ import annotations

import datetime as _dt
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

import activity

router = APIRouter(prefix="/api/v1", tags=["recommend"])

POOL_DAYS = 120                 # standalone posts older than this aren't candidates
HISTORY_DAYS = 90
NI_CREATOR_DAYS = 30
SMALL_CREATOR_VIEWERS = 200     # fewer distinct viewers than this, across their work: a "new creator"
EXPLORE_EVERY = 5
CREATOR_WINDOW, CREATOR_MAX = 8, 2
CACHE_SECONDS = 120
COMPLETE_AT = 0.9

W = {"quality": 1.4, "affinity": 1.0, "fresh": 0.6, "continue": 1.6, "new_creator": 0.3, "seen": 0.35,
     "popular": 0.05}


@dataclass
class Item:
    key: str                     # "s:<series id>" or "p:<post id>"
    kind: str                    # series | video | photo | ad
    post_id: str                 # what to play/show (for a series: the viewer's next episode)
    creator_id: str
    title: str = ""
    text: str = ""
    genre: str = ""
    content_type: str = ""
    published: Optional[_dt.datetime] = None
    post_ids: list = field(default_factory=list)   # every post of the unit (series: all episodes)


@dataclass
class Stats:
    viewers: int = 0
    completers: int = 0
    likers: int = 0
    sharers: int = 0
    savers: int = 0
    not_interested: int = 0
    impressions: int = 0


@dataclass
class Viewer:
    user_id: Optional[str] = None
    personalized: bool = True
    genre_w: dict = field(default_factory=dict)
    type_w: dict = field(default_factory=dict)
    creator_w: dict = field(default_factory=dict)
    follows: set = field(default_factory=set)
    terms: list = field(default_factory=list)
    completed: set = field(default_factory=set)      # post ids watched to the end
    started: set = field(default_factory=set)        # post ids opened
    progress_series: dict = field(default_factory=dict)  # series id -> highest episode number reached
    hidden_keys: set = field(default_factory=set)    # not interested (unit keys)
    hidden_creators: set = field(default_factory=set)
    seen_recent: Counter = field(default_factory=Counter)  # unit key -> impressions in the last 3 days
    blocked: set = field(default_factory=set)

    @property
    def is_new(self) -> bool:
        return not (self.genre_w or self.type_w or self.creator_w or self.follows or self.started)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def quality(st: Stats) -> float:
    """0..1. Completion from distinct viewers (prior 40% over 10 viewers), likes/shares/saves per viewer
    (prior 5% over 20), and "not interested" against impressions."""
    comp = (st.completers + 0.4 * 10) / (st.viewers + 10)
    love = (st.likers + st.sharers + st.savers + 0.05 * 20) / (st.viewers + 20)
    neg = st.not_interested / (st.impressions + st.viewers + 20)
    return max(0.0, min(1.0, 0.6 * comp + 0.4 * min(1.0, love * 4) - 2.0 * neg))


def freshness(item: Item, now: _dt.datetime) -> float:
    if not item.published:
        return 0.3
    days = max(0.0, (now - item.published).total_seconds() / 86400)
    return 0.5 ** (days / (14 if item.kind == "series" else 7))


def affinity(item: Item, v: Viewer) -> tuple[float, str]:
    if not v.personalized:
        return 0.0, ""
    score, why = 0.0, ""
    g = v.genre_w.get((item.genre or "").lower(), 0.0)
    t = v.type_w.get(item.content_type or item.kind, 0.0)
    c = v.creator_w.get(item.creator_id, 0.0) + (0.6 if item.creator_id in v.follows else 0.0)
    score += 0.6 * g + 0.3 * t + c
    if item.creator_id in v.follows:
        why = "From a creator you follow"
    elif g > 0.3 and item.genre:
        why = f"Because you watch {item.genre}"
    text = _norm(f"{item.title} {item.text} {item.genre}")
    hits = [term for term in v.terms if term in text]
    if hits:
        score += min(0.6, 0.25 * len(hits))
        why = why or f'Matches your search "{hits[0]}"'
    return max(-1.5, min(2.0, score)), why


def score_item(item: Item, st: Stats, v: Viewer, now: _dt.datetime, creator_viewers: int) -> tuple[float, str]:
    q = quality(st)
    aff, why = affinity(item, v)
    s = W["quality"] * q + W["affinity"] * aff + W["fresh"] * freshness(item, now)
    s += W["popular"] * math.log1p(st.viewers)            # a little, never the main reason
    if v.personalized and item.kind == "series" and item.key[2:] in v.progress_series:
        s += W["continue"]
        why = "Continue watching"
    if creator_viewers < SMALL_CREATOR_VIEWERS and q >= 0.4:
        s += W["new_creator"]
        why = why or "New creator"
    s -= W["seen"] * min(3, v.seen_recent.get(item.key, 0))
    if item.kind != "series" and item.post_id in v.started:
        s -= 0.8                                           # opened before: rarely worth showing again
    if not why:
        why = "Popular with viewers who finish it" if q >= 0.55 and st.viewers >= 5 else "Fresh on VIYO" \
            if freshness(item, now) > 0.6 else "Picked for you"
    return s, why


def eligible(item: Item, v: Viewer) -> bool:
    if item.creator_id in v.blocked or item.creator_id in v.hidden_creators or item.key in v.hidden_keys:
        return False
    if v.user_id and item.creator_id == v.user_id:
        return False
    if item.kind != "series" and item.post_id in v.completed:
        return False
    return True


def diversify(scored: list, stats: dict, limit: int, rng: random.Random) -> list:
    """Greedy: best score first, at most CREATOR_MAX per creator in any CREATOR_WINDOW, and every
    EXPLORE_EVERY-th place for a little-seen title (picked at random among the decent ones)."""
    remaining = sorted(scored, key=lambda x: x[1], reverse=True)
    out = []
    while remaining and len(out) < limit:
        recent = [it.creator_id for it, _, _ in out[-(CREATOR_WINDOW - 1):]]
        ok = [x for x in remaining if recent.count(x[0].creator_id) < CREATOR_MAX] or remaining
        pick = ok[0]
        if (len(out) + 1) % EXPLORE_EVERY == 0:
            explore = [x for x in ok if stats[x[0].key].viewers < 20 and quality(stats[x[0].key]) >= 0.3]
            if explore:
                pick = rng.choice(explore[:10])
                pick = (pick[0], pick[1], pick[2] if pick[2] not in ("Picked for you", "Fresh on VIYO") else "Something new")
        out.append(pick)
        remaining = [x for x in remaining if x[0].key != pick[0].key]
    return out


def rank(items: list, stats: dict, v: Viewer, now: _dt.datetime, creator_viewers: dict, limit: int = 60,
         seed: Optional[int] = None) -> list[dict]:
    rng = random.Random(seed if seed is not None else hash((v.user_id, now.date().isoformat())))
    scored = []
    for it in items:
        if not eligible(it, v):
            continue
        s, why = score_item(it, stats.get(it.key, Stats()), v, now, creator_viewers.get(it.creator_id, 0))
        scored.append((it, s, why))
    stats_d = defaultdict(Stats, stats)
    picked = diversify([x for x in scored if x[0].kind != "ad"], stats_d, limit, rng)
    # Ads: the best few, at the 4th place and then at most one in every eight (their own Ads category holds them
    # all), never pushing the feed past [limit].
    ads = sorted([x for x in scored if x[0].kind == "ad"], key=lambda x: x[1], reverse=True)
    for n, ad in enumerate(ads):
        pos = 3 + 8 * n
        if pos >= len(picked):
            break
        picked.insert(pos, ad)
    picked = picked[:limit]
    return [{"post_id": it.post_id, "series_id": it.key[2:] if it.kind == "series" else None,
             "kind": it.kind, "reason": why, "score": round(s, 3)} for it, s, why in picked]


# ---------------------------------------------------------------- loading from the database

def _ts(v) -> Optional[_dt.datetime]:
    if not v:
        return None
    try:
        return _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def load_items(db, now: _dt.datetime) -> tuple[list, dict]:
    """Candidate units and a post id -> unit key map."""
    since = (now - _dt.timedelta(days=POOL_DAYS)).isoformat()
    posts = (db.table("posts").select("*").eq("is_private", False).eq("is_archived", False).execute().data) or []
    series = {s["id"]: s for s in (db.table("series").select("*").execute().data or [])}
    items, key_of = [], {}
    eps = defaultdict(list)
    for p in posts:
        if not p.get("media_url"):
            continue
        if p.get("series_id"):
            eps[p["series_id"]].append(p)
            continue
        if str(p.get("created_at") or "") < since:
            continue
        kind = "ad" if p.get("post_kind") == "ad" else ("video" if p.get("post_type") == "video" else "photo")
        it = Item(key=f"p:{p['id']}", kind=kind, post_id=p["id"], creator_id=p["user_id"],
                  title=(p.get("caption") or "")[:120], text=p.get("caption") or "", published=_ts(p.get("created_at")),
                  post_ids=[p["id"]])
        items.append(it)
        key_of[p["id"]] = it.key
    for sid, lst in eps.items():
        s = series.get(sid)
        if not s:
            continue
        lst.sort(key=lambda e: int(e.get("episode_number") or 0))
        it = Item(key=f"s:{sid}", kind="series", post_id=lst[0]["id"], creator_id=s.get("user_id") or lst[0]["user_id"],
                  title=s.get("title") or "", text=s.get("description") or "", genre=s.get("genre") or "",
                  content_type=s.get("content_type") or "short_drama",
                  published=max((_ts(e.get("created_at")) for e in lst), default=None, key=lambda d: d or now),
                  post_ids=[e["id"] for e in lst])
        it._episodes = lst
        items.append(it)
        for e in lst:
            key_of[e["id"]] = it.key
    return items, key_of


def load_stats(db, key_of: dict, now: _dt.datetime, excluded_users: set) -> tuple[dict, dict]:
    """Per unit: distinct viewers, completers, likers, sharers, savers, not-interested and impressions over the
    last 30 days, from viewer_events (accounts under fraud review excluded). Per creator: distinct viewers."""
    since = (now - _dt.timedelta(days=30)).isoformat()
    rows = (db.table("viewer_events").select("user_id,kind,post_id,creator_id,value").gte("created_at", since)
            .execute().data) or []
    sets = defaultdict(lambda: defaultdict(set))
    creator_viewers = defaultdict(set)
    for r in rows:
        if r.get("user_id") in excluded_users:
            continue
        key = key_of.get(r.get("post_id") or "")
        kind = r["kind"]
        if key:
            if kind in ("view", "progress"):
                sets[key]["viewers"].add(r["user_id"])
            if kind == "complete" or (kind == "progress" and (r.get("value") or 0) >= COMPLETE_AT):
                sets[key]["completers"].add(r["user_id"])
                sets[key]["viewers"].add(r["user_id"])
            for k, name in (("like", "likers"), ("share", "sharers"), ("save", "savers"),
                            ("not_interested", "not_interested"), ("impression", "impressions")):
                if kind == k:
                    sets[key][name].add(r["user_id"])
        if r.get("creator_id") and kind in ("view", "progress", "complete"):
            creator_viewers[r["creator_id"]].add(r["user_id"])
    stats = {k: Stats(**{n: len(v) for n, v in d.items()}) for k, d in sets.items()}
    return stats, {c: len(v) for c, v in creator_viewers.items()}


def load_viewer(db, user_id: str, items: list, key_of: dict, now: _dt.datetime) -> Viewer:
    v = Viewer(user_id=user_id, personalized=activity.personalized(user_id))
    try:
        v.blocked = {r["blocked_id"] for r in (db.table("blocks").select("*").eq("blocker_id", user_id).execute().data or [])}
    except Exception:
        pass
    since = (now - _dt.timedelta(days=HISTORY_DAYS)).isoformat()
    events = (db.table("viewer_events").select("*").eq("user_id", user_id).gte("created_at", since).execute().data) or []
    by_key = {it.key: it for it in items}
    three_days = (now - _dt.timedelta(days=3)).isoformat()
    ni_since = (now - _dt.timedelta(days=NI_CREATOR_DAYS)).isoformat()
    weights = {"complete": 3.0, "like": 2.0, "share": 2.0, "save": 2.0, "comment": 1.5, "view": 0.6,
               "progress": 0.6, "not_interested": -3.0}
    genre, ctype, creator = defaultdict(float), defaultdict(float), defaultdict(float)
    for e in events:
        k, pid = e["kind"], e.get("post_id")
        key = key_of.get(pid or "")
        if k == "not_interested" and key:
            v.hidden_keys.add(key)
        if k == "not_interested_creator" and e.get("creator_id") and str(e.get("created_at")) >= ni_since:
            v.hidden_creators.add(e["creator_id"])
        if k == "search" and e.get("query"):
            v.terms += [t for t in _norm(e["query"]).split() if len(t) >= 3]
        if k == "impression" and key and str(e.get("created_at")) >= three_days:
            v.seen_recent[key] += 1
        if k in ("view", "progress", "complete") and pid:
            v.started.add(pid)
            if k == "complete" or (k == "progress" and (e.get("value") or 0) >= COMPLETE_AT):
                v.completed.add(pid)
        w = weights.get(k)
        it = by_key.get(key) if key else None
        if w and it:
            genre[(it.genre or "").lower()] += w
            ctype[it.content_type or it.kind] += w
            creator[it.creator_id] += w / 3
    try:
        for r in db.table("watch_progress").select("*").eq("user_id", user_id).execute().data or []:
            pid = r.get("post_id")
            v.started.add(pid)
            dur = r.get("duration_ms") or 0
            if dur and (r.get("position_ms") or 0) / dur >= COMPLETE_AT:
                v.completed.add(pid)
    except Exception:
        pass
    try:
        v.follows = {r["following_id"] for r in (db.table("follows").select("*").eq("follower_id", user_id).execute().data or [])}
    except Exception:
        pass

    def scale(d):
        top = max((abs(x) for x in d.values()), default=0)
        return {k: x / top for k, x in d.items()} if top else {}
    genre.pop("", None)
    v.genre_w, v.type_w, v.creator_w = scale(genre), scale(ctype), {k: min(1.0, x / 3) for k, x in creator.items()}
    v.terms = list(dict.fromkeys(reversed(v.terms)))[:15]
    # Series: the next episode for this viewer (the first one not finished); a finished series stays out
    # until a new episode arrives.
    for it in items:
        if it.kind != "series":
            continue
        eps = getattr(it, "_episodes", [])
        done = [e for e in eps if e["id"] in v.completed]
        touched = [e for e in eps if e["id"] in v.started]
        nxt = next((e for e in eps if e["id"] not in v.completed), None)
        if nxt is None:
            v.hidden_keys.add(it.key)
            continue
        it.post_id = nxt["id"]
        if touched:
            v.progress_series[it.key[2:]] = max(int(e.get("episode_number") or 0) for e in (done or touched))
    if not v.personalized:
        v.genre_w, v.type_w, v.creator_w, v.terms, v.progress_series = {}, {}, {}, [], {}
    return v


def flagged_users(db) -> set:
    try:
        rows = db.table("fraud_flags").select("*").eq("subject_type", "user").eq("status", "open").execute().data or []
        return {r["subject_id"] for r in rows}
    except Exception:
        return set()


_cache: dict[str, tuple[float, list]] = {}


def for_you(user_id: str, limit: int = 60) -> list[dict]:
    db = activity._db()
    now = _dt.datetime.now(_dt.timezone.utc)
    items, key_of = load_items(db, now)
    stats, creator_viewers = load_stats(db, key_of, now, flagged_users(db))
    viewer = load_viewer(db, user_id, items, key_of, now)
    return rank(items, stats, viewer, now, creator_viewers, limit=limit)


@router.get("/feed/for-you")
async def get_for_you(cursor: int = 0, limit: int = 20, user_id: str = Depends(activity._viewer)):
    """The viewer's For You page: post ids in order, with a short reason each. Ranked on the server; the same
    ranking is kept for a couple of minutes so paging doesn't reshuffle."""
    limit = max(1, min(int(limit), 40))
    hit = _cache.get(user_id)
    if cursor == 0 or not hit or time.time() - hit[0] > CACHE_SECONDS:
        ranked = for_you(user_id)
        _cache[user_id] = (time.time(), ranked)
        if len(_cache) > 5000:
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:1000]:
                _cache.pop(k, None)
    ranked = _cache[user_id][1]
    page = ranked[cursor:cursor + limit]
    return {"items": page, "next_cursor": cursor + len(page) if cursor + len(page) < len(ranked) else None}
