"""Creator Hub: one place for a creator to see how their productions do, what they've earned, how earning on
VIYO actually works, and how to get paid.

Everything here is read from real records, never estimated:
  * views, likes, comments and shares are the posts' own counters; followers are follows rows;
  * earnings are creator_earnings rows (the coins actually credited when a viewer unlocked an episode or movie
    with bought coins), per production via each row's source post;
  * how earning works is generated from the same constants the money code uses (episodes.CREATOR_SHARE,
    FREE_EPISODE_COUNT, BUNDLE_DISCOUNT; wallet.COIN_USD, MIN_WITHDRAWAL_USD), including what does not earn,
    so the explanation can never promise more than the platform pays.
Read-only: nothing here changes a balance, a post or a payout.
"""
from __future__ import annotations

import datetime as _dt
import os
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

router = APIRouter(prefix="/api/v1/creator", tags=["creator_hub"])

supabase_admin = None
if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
    from supabase import create_client
    supabase_admin = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINGLE_ASSET_TYPES = {"movie", "short_film"}


async def _user(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest
    return await get_current_user_id_no_guest(authorization)


def _db():
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Creator Hub is not configured.")
    return supabase_admin


def _stats(rows: list[dict]) -> dict:
    return {k: sum(int(r.get(c) or 0) for r in rows) for k, c in
            (("views", "view_count"), ("likes", "like_count"), ("comments", "comment_count"), ("shares", "share_count"))}


def how_you_earn() -> dict:
    """The real rules, from the constants the money code uses."""
    import episodes
    import wallet
    share = round(episodes.CREATOR_SHARE * 100)
    return {
        "earns": [
            {"title": "Paid episodes and movies",
             "detail": f"You choose the coin price per episode. When a viewer unlocks one with coins they bought, you "
                       f"get {share}% of the bought coins they paid. The first {episodes.FREE_EPISODE_COUNT} episodes "
                       f"of a series are always free; a movie or short film is paid from the first watch. Viewers "
                       f"who unlock a whole series at once get {round(episodes.BUNDLE_DISCOUNT * 100)}% off, and you "
                       f"get {share}% of what they paid."},
        ],
        "does_not_earn": [
            "Unlocks paid with free coins (daily check-ins, missions, rewarded ads, gifts) or by watching ads.",
            "Premium subscribers watching your episodes (subscription revenue isn't shared yet).",
            "Free titles and free episodes, and views, likes or shares on their own: they help people find you, "
            "but they aren't paid.",
        ],
        "payout": {
            "usd_per_coin": wallet.COIN_USD,
            "min_usd": wallet.MIN_WITHDRAWAL_USD,
            "min_coins": wallet.MIN_WITHDRAWAL_COINS,
            "detail": f"Earned coins are worth ${wallet.COIN_USD:.3f} each. From ${wallet.MIN_WITHDRAWAL_USD:.0f} "
                      f"({wallet.MIN_WITHDRAWAL_COINS:,} coins) you can ask for a withdrawal to PayPal or mobile "
                      "money. The VIYO team sends it by hand and marks it paid; you see its status in your Wallet.",
        },
        "note": "VIYO pays only from coins viewers actually bought, so earnings depend on paid unlocks of your work.",
    }


def hub(user_id: str) -> dict:
    import wallet
    db = _db()
    posts = [p for p in (db.table("posts").select("*").eq("user_id", user_id).execute().data or [])
             if not p.get("is_archived")]
    series = db.table("series").select("*").eq("user_id", user_id).execute().data or []
    try:
        earnings = db.table("creator_earnings").select("*").eq("creator_id", user_id).execute().data or []
    except Exception:
        earnings = []
    try:
        followers = len(db.table("follows").select("*").eq("following_id", user_id).execute().data or [])
    except Exception:
        followers = None

    by_post = defaultdict(int)
    for e in earnings:
        if e.get("source_post_id"):
            by_post[e["source_post_id"]] += int(e.get("coins") or 0)
    episodes_of = defaultdict(list)
    for p in posts:
        if p.get("series_id"):
            episodes_of[p["series_id"]].append(p)

    productions = []
    for s in series:
        eps = episodes_of.get(s["id"], [])
        public = [p for p in eps if not p.get("is_private")]
        price = int(s.get("coin_price_per_episode") or 0)
        kind = s.get("content_type") or "short_drama"
        productions.append({
            "kind": "series", "id": s["id"], "title": s.get("title") or "", "cover_url": s.get("cover_image_url"),
            "content_type": kind, "status": s.get("status"), "episodes": len(eps), "public_episodes": len(public),
            "coin_price": price, "earns": price > 0, **_stats(eps),
            "earned_coins": sum(by_post.get(p["id"], 0) for p in eps),
            "created_at": s.get("created_at")})
    standalone = [p for p in posts if not p.get("series_id") and (p.get("post_kind") or "post") == "post"
                  and p.get("post_type") == "video"]
    for p in sorted(standalone, key=lambda x: int(x.get("view_count") or 0), reverse=True)[:20]:
        productions.append({
            "kind": "video", "id": p["id"], "title": (p.get("caption") or "").split("\n")[0][:80],
            "cover_url": p.get("thumbnail_url"), "content_type": "video", "status": "private" if p.get("is_private") else "public",
            "episodes": 1, "public_episodes": 0 if p.get("is_private") else 1, "coin_price": 0, "earns": False,
            **_stats([p]), "earned_coins": by_post.get(p["id"], 0), "created_at": p.get("created_at")})
    productions.sort(key=lambda x: (x["views"], x["earned_coins"]), reverse=True)

    totals = {"available": 0, "pending_withdrawal": 0, "paid": 0}
    for e in earnings:
        st = e.get("status") or "available"
        totals[st] = totals.get(st, 0) + int(e.get("coins") or 0)
    since = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=30)).isoformat()
    last_30 = sum(int(e.get("coins") or 0) for e in earnings if str(e.get("created_at") or "") >= since)
    try:
        settings = wallet._load_settings(user_id)
        pending = wallet._pending_request(user_id)
    except Exception:
        settings, pending = None, None
    lifetime = sum(totals.values())
    return {
        "audience": {**_stats(posts), "followers": followers, "productions": len(series) + len(standalone)},
        "earnings": {"lifetime_coins": lifetime, "lifetime_usd": wallet.coins_to_usd(lifetime),
                     "available_coins": totals.get("available", 0),
                     "available_usd": wallet.coins_to_usd(totals.get("available", 0)),
                     "pending_coins": totals.get("pending_withdrawal", 0), "paid_coins": totals.get("paid", 0),
                     "paid_usd": wallet.coins_to_usd(totals.get("paid", 0)),
                     "last_30_days_coins": last_30, "can_withdraw": totals.get("available", 0) >= wallet.MIN_WITHDRAWAL_COINS,
                     "payout_details_saved": bool(settings), "withdrawal_pending": bool(pending)},
        "productions": productions,
        "how_you_earn": how_you_earn(),
    }


@router.get("/hub")
async def get_hub(user_id: str = Depends(_user)):
    """The signed-in creator's own audience, productions and earnings, and the real earning rules."""
    return hub(user_id)
