"""Creator levels: New, Rising, Pro and Star, worked out from real records (creator_levels.sql).

Business rules (agreed with the owner):
  * every creator earns, at every level, the same share (episodes.CREATOR_SHARE, 65%); levels never change who can earn;
  * the only perk is a shorter earnings hold for Pro and Star (SHORT_HOLD_DAYS instead of wallet.EARNINGS_HOLD_DAYS);
  * Pro and Star also need a clean record: no refunded or reversed earning in the last CLEAN_DAYS days, so the
    shorter hold only goes to creators whose earnings haven't needed refund protection.
If the numbers can't be read, a creator is treated as New: the normal hold, never a shorter one."""
from __future__ import annotations

import os
import time

SHORT_HOLD_DAYS = max(0, int(os.environ.get("CREATOR_PRO_HOLD_DAYS", "7") or 7))
CLEAN_DAYS = 90

# Each level needs everything listed (followers, people who watched in the last 30 days, account age in days,
# money earned net of refunds in USD, and for Pro and Star a clean record).
LEVELS = (
    {"key": "new", "name": "New", "followers": 0, "viewers_30d": 0, "account_days": 0, "earned_usd": 0, "clean": False},
    {"key": "rising", "name": "Rising", "followers": 100, "viewers_30d": 500, "account_days": 0, "earned_usd": 0,
     "clean": False},
    {"key": "pro", "name": "Pro", "followers": 1000, "viewers_30d": 5000, "account_days": 60, "earned_usd": 50,
     "clean": True},
    {"key": "star", "name": "Star", "followers": 10000, "viewers_30d": 50000, "account_days": 90, "earned_usd": 500,
     "clean": True},
)


def hold_days(level_key: str) -> int:
    import wallet
    return min(SHORT_HOLD_DAYS, wallet.EARNINGS_HOLD_DAYS) if level_key in ("pro", "star") else wallet.EARNINGS_HOLD_DAYS


def _requirements(level: dict, m: dict) -> list[dict]:
    import wallet
    earned_usd = wallet.coins_to_usd(int(m.get("earned_coins") or 0))
    age = m.get("account_days")
    out = []
    if level["followers"]:
        out.append({"key": "followers", "label": f"{level['followers']:,} followers",
                    "have": int(m.get("followers") or 0), "need": level["followers"]})
    if level["viewers_30d"]:
        out.append({"key": "viewers_30d", "label": f"{level['viewers_30d']:,} people watched in the last 30 days",
                    "have": int(m.get("viewers_30d") or 0), "need": level["viewers_30d"]})
    if level["account_days"]:
        out.append({"key": "account_days", "label": f"Account at least {level['account_days']} days old",
                    "have": int(age) if age is not None else 0, "need": level["account_days"]})
    if level["earned_usd"]:
        out.append({"key": "earned_usd", "label": f"${level['earned_usd']:,} earned",
                    "have": earned_usd, "need": level["earned_usd"]})
    if level["clean"]:
        out.append({"key": "clean", "label": f"No refunded earnings in the last {CLEAN_DAYS} days",
                    "have": int(m.get("reversals_90d") or 0), "need": 0})
    for r in out:
        r["met"] = r["have"] == 0 if r["key"] == "clean" else r["have"] >= r["need"]
    return out


def evaluate(m: dict | None) -> dict:
    """The creator's level, its perk, and what the next level still needs. m = creator_level_metrics output."""
    import wallet
    m = m or {}
    current = LEVELS[0]
    for lvl in LEVELS[1:]:
        if all(r["met"] for r in _requirements(lvl, m)):
            current = lvl
        else:
            break                     # levels are climbed in order
    i = LEVELS.index(current)
    nxt = LEVELS[i + 1] if i + 1 < len(LEVELS) else None
    days = hold_days(current["key"])
    return {
        "key": current["key"],
        "name": current["name"],
        "rank": i + 1,
        "of": len(LEVELS),
        "hold_days": days,
        "perks": ([f"Earnings available after {days} days instead of {wallet.EARNINGS_HOLD_DAYS}"]
                  if days < wallet.EARNINGS_HOLD_DAYS else []),
        "next": ({"key": nxt["key"], "name": nxt["name"], "requirements": _requirements(nxt, m),
                  "perk": (f"Earnings available after {hold_days(nxt['key'])} days instead of "
                           f"{wallet.EARNINGS_HOLD_DAYS}") if hold_days(nxt["key"]) < days else None}
                 if nxt else None),
        "share_note": "Every level earns the same share of what viewers pay.",
    }


def metrics(db, creator_id: str) -> dict | None:
    """db: a Supabase client (rpc(...).execute()) or a Studio store (rpc(...) returns the data)."""
    try:
        r = db.rpc("creator_level_metrics", {"p_creator": creator_id})
        data = r.execute().data if hasattr(r, "execute") else r
        return data if isinstance(data, dict) and data else None
    except Exception:
        return None


def level_for(db, creator_id: str) -> dict:
    """The creator's level, or None-safe: unreadable numbers count as New (normal hold)."""
    m = metrics(db, creator_id)
    out = evaluate(m)
    out["ready"] = m is not None
    return out


_cache: dict[str, tuple[float, int]] = {}
CACHE_SECONDS = 600


def hold_days_for(db, creator_id: str) -> int:
    """How long a new earning for this creator is held. Cached for a few minutes (an unlock shouldn't wait on it);
    on any problem, the normal hold."""
    import wallet
    now = time.monotonic()
    hit = _cache.get(creator_id)
    if hit and now - hit[0] < CACHE_SECONDS:
        return hit[1]
    m = metrics(db, creator_id) if db is not None else None
    days = evaluate(m)["hold_days"] if m is not None else wallet.EARNINGS_HOLD_DAYS
    _cache[creator_id] = (now, days)
    return days


def forget(creator_id: str) -> None:
    _cache.pop(creator_id, None)
