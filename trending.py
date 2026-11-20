"""Discovery and trending: what's catching on, growing fast, just released, and which smaller creators are making
work people finish. Plain counting over viewer_events, no paid AI.

Fairness and honesty rules in the maths:
  * Counted in distinct people, not raw counters: one account watching 50 times is one viewer. The creator's own
    activity and accounts under fraud review don't count.
  * Diminishing returns (log) and a fame damper: a creator who already reaches many people gets less extra lift
    from each new viewer than a small creator does, so popularity can't become a permanent advantage.
  * Quality matters: a title people start but don't finish ranks below one they finish and like.
  * No more than two titles from the same creator in any list.
  * Regions and languages: VIYO doesn't record a viewer's region or a title's language yet, so those lists
    aren't produced (the response says so) rather than guessed.
"""
from __future__ import annotations

import datetime as _dt
import math
import time
from collections import defaultdict

from fastapi import APIRouter, Depends

import activity
import recommend as rec

router = APIRouter(prefix="/api/v1", tags=["trending"])

WEIGHTS = {"view": 1.0, "progress": 1.0, "complete": 2.0, "like": 1.5, "comment": 2.0, "share": 3.0, "save": 2.0}
MIN_PEOPLE = 3                  # fewer distinct people than this: not "trending", just early
PER_CREATOR = 2
CACHE_SECONDS = 300
_cache: dict = {}


def _people(rows, key_of, owners, excluded, since, until=None):
    """Per unit: {user: best weight} from events in [since, until)."""
    out = defaultdict(dict)
    for r in rows:
        ts = str(r.get("created_at") or "")
        if ts < since or (until and ts >= until):
            continue
        key = key_of.get(r.get("post_id") or "")
        w = WEIGHTS.get(r["kind"])
        u = r.get("user_id")
        if not key or not w or u in excluded or owners.get(key) == u:
            continue
        out[key][u] = max(out[key].get(u, 0), w)
    return out


def _cap(ordered, by_key, n=12):
    seen, out = defaultdict(int), []
    for key, score in ordered:
        it = by_key[key]
        if seen[it.creator_id] >= PER_CREATOR:
            continue
        seen[it.creator_id] += 1
        out.append((key, score))
        if len(out) >= n:
            break
    return out


def compute(db, now: _dt.datetime) -> dict:
    items, key_of = rec.load_items(db, now)
    items = [it for it in items if it.kind != "ad"]
    by_key = {it.key: it for it in items}
    owners = {it.key: it.creator_id for it in items}
    excluded = rec.flagged_users(db)
    since10 = (now - _dt.timedelta(days=10)).isoformat()
    rows = (db.table("viewer_events").select("user_id,kind,post_id,created_at").gte("created_at", since10)
            .execute().data) or []
    stats, creator_viewers = rec.load_stats(db, key_of, now, excluded)
    d2, d3, d7 = (now - _dt.timedelta(days=n) for n in (2, 3, 7))
    recent = _people(rows, key_of, owners, excluded, d2.isoformat())
    last3 = _people(rows, key_of, owners, excluded, d3.isoformat())
    before = _people(rows, key_of, owners, excluded, (d3 - _dt.timedelta(days=7)).isoformat(), d3.isoformat())

    def fame(it):
        return 1 + 0.15 * math.log1p(creator_viewers.get(it.creator_id, 0))

    trending, growing, rising = [], [], []
    for key, people in recent.items():
        if len(people) < MIN_PEOPLE or key not in by_key:
            continue
        it = by_key[key]
        q = rec.quality(stats.get(key, rec.Stats()))
        trending.append((key, math.log1p(sum(people.values())) * (0.5 + q) / fame(it)))
    for key, people in last3.items():
        if len(people) < MIN_PEOPLE or key not in by_key:
            continue
        per_day_now = len(people) / 3
        per_day_before = len(before.get(key, {})) / 7
        growth = (per_day_now + 1) / (per_day_before + 1)
        if growth >= 1.5:
            growing.append((key, growth))
    new = [(it.key, rec.quality(stats.get(it.key, rec.Stats())) + rec.freshness(it, now)) for it in items
           if it.published and (now - it.published).days <= 14]
    best_by_creator = {}
    for it in items:
        st = stats.get(it.key, rec.Stats())
        q = rec.quality(st)
        if creator_viewers.get(it.creator_id, 0) < rec.SMALL_CREATOR_VIEWERS and st.viewers >= MIN_PEOPLE and q >= 0.4:
            if q > best_by_creator.get(it.creator_id, ("", 0))[1]:
                best_by_creator[it.creator_id] = (it.key, q)
    rising = sorted(best_by_creator.values(), key=lambda x: x[1], reverse=True)
    genre_people = defaultdict(set)
    for key, people in _people(rows, key_of, owners, excluded, d7.isoformat()).items():
        g = (by_key[key].genre or "").strip() if key in by_key else ""
        if g:
            genre_people[g] |= set(people)

    def out(lst, label):
        return [{"post_id": by_key[k].post_id, "series_id": k[2:] if by_key[k].kind == "series" else None,
                 "kind": by_key[k].kind, "title": by_key[k].title, "reason": label, "score": round(s, 3)}
                for k, s in lst]
    return {
        "trending": out(_cap(sorted(trending, key=lambda x: x[1], reverse=True), by_key), "Trending"),
        "fast_growing": out(_cap(sorted(growing, key=lambda x: x[1], reverse=True), by_key), "Growing fast"),
        "new_releases": out(_cap(sorted(new, key=lambda x: x[1], reverse=True), by_key), "New"),
        "rising_creators": out(_cap(rising, by_key), "Rising creator"),
        "genres": [{"genre": g, "viewers": len(p)} for g, p in sorted(genre_people.items(), key=lambda x: -len(x[1]))[:8]],
        "regions": [],
        "note": "Trending counts distinct people (creators' own activity and accounts under review excluded). "
                "Region and language lists need region/language data VIYO doesn't record yet.",
    }


@router.get("/discover/trending")
async def get_trending(user_id: str = Depends(activity._viewer)):
    """Trending, fast-growing, new releases, rising creators and popular genres (cached for five minutes)."""
    hit = _cache.get("all")
    if not hit or time.time() - hit[0] > CACHE_SECONDS:
        _cache["all"] = (time.time(), compute(activity._db(), _dt.datetime.now(_dt.timezone.utc)))
    return _cache["all"][1]
