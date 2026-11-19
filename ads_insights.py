"""Advertising insights and the experiment lab, for a creator's own published ads.

Insights
  * Real numbers only: each ad post's views, likes, comments and shares (the post's own counters), likes and
    comments per day (from the likes/comments rows' own dates), follower growth per day (from follows), and
    views/shares per day only from daily snapshots actually taken (no estimated or filled-in days).
  * Organic and paid are kept apart: an ad that was boosted (paid placement) is counted under "paid", every
    other ad under "organic". One boosted ad's views can't yet be split into paid and organic; the result says so.
  * A snapshot of each of the creator's ad posts is saved (once a day) whenever they open their insights,
    so trends build up from real counts.

Experiment lab
  * An experiment makes 2-6 versions of an ad for one brand, each in a different tone (funny, emotional,
    cinematic, storytelling, product demonstration, character-led). Creating it charges nothing: each version
    is an ordinary draft, paid only when the creator generates it, after seeing the price.
  * The comparison uses only real engagement of the published versions (likes + comments + shares per view).
    A version is called "more engaging so far" only when every compared version has at least MIN_VIEWS views
    and the difference is statistically clear (two-proportion z-test, 95%). Otherwise: not enough data yet.
    It never predicts sales or promises results.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import ads_studio as core
from ads_brands import _db, _uuid, owned_brand
from creator_studio import creator

router = APIRouter(prefix="/api/v1/ads-maker", tags=["ads_maker"])

MIN_VIEWS = 200          # each compared version needs at least this many views before any winner is named
Z_95 = 1.96
TONES = {
    "funny": ("Funny", "social", "Light, playful humour with a comic twist. Never mock anyone."),
    "emotional": ("Emotional", "lifestyle", "Warm and heartfelt: a real feeling the audience knows."),
    "cinematic": ("Cinematic", "product_cinematic", "Big, beautiful shots and slow, premium pacing."),
    "storytelling": ("Storytelling", "auto", "A clear little story with a beginning, a turn and an end."),
    "demo": ("Product demonstration", "demo", "Show exactly how it works or what it does."),
    "character": ("Character-led", "character", "The brand's character carries the ad."),
}


def _today() -> _dt.date:
    return _dt.datetime.now(_dt.timezone.utc).date()


def _day(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    try:
        return _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return str(ts)[:10]


def _ad_posts(user_id: str) -> list[dict]:
    rows = _db().table("posts").select("*").eq("user_id", user_id).eq("post_kind", "ad").execute().data or []
    return [p for p in rows if not p.get("is_archived")]


def _stats(p: dict) -> dict:
    v, l, c, s = (int(p.get(k) or 0) for k in ("view_count", "like_count", "comment_count", "share_count"))
    return {"views": v, "likes": l, "comments": c, "shares": s,
            "engagement_rate": round((l + c + s) / v, 4) if v else None}


def _snapshot(posts: list[dict], user_id: str) -> None:
    """Today's counts for each ad post, saved once a day (best-effort: insights still work without it)."""
    day = _today().isoformat()
    try:
        have = {r["post_id"] for r in (_db().table("ad_post_stats_daily").select("*").eq("owner_id", user_id)
                                       .eq("day", day).execute().data or [])}
        for p in posts:
            if p["id"] in have:
                continue
            s = _stats(p)
            _db().table("ad_post_stats_daily").insert({
                "post_id": p["id"], "owner_id": user_id, "day": day, "views": s["views"], "likes": s["likes"],
                "comments": s["comments"], "shares": s["shares"], "boosted": bool(p.get("is_boosted"))}).execute()
    except Exception as e:
        print(f"[WARN] ad insights snapshot: {str(e)[:120]}")


def _by_day(rows: list[dict], since: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        d = _day(r.get("created_at"))
        if d and d >= since:
            out[d] += 1
    return dict(out)


def _snapshot_deltas(user_id: str, since: str) -> dict[str, dict]:
    """Views and shares gained per day, only between two real snapshots of the same post."""
    try:
        rows = _db().table("ad_post_stats_daily").select("*").eq("owner_id", user_id).execute().data or []
    except Exception:
        return {}
    per_post: dict[str, list] = defaultdict(list)
    for r in rows:
        per_post[r["post_id"]].append(r)
    out: dict[str, dict] = defaultdict(lambda: {"views": 0, "shares": 0})
    for snaps in per_post.values():
        snaps.sort(key=lambda r: str(r["day"]))
        for prev, cur in zip(snaps, snaps[1:]):
            d = str(cur["day"])[:10]
            if d < since:
                continue
            out[d]["views"] += max(0, int(cur.get("views") or 0) - int(prev.get("views") or 0))
            out[d]["shares"] += max(0, int(cur.get("shares") or 0) - int(prev.get("shares") or 0))
    return dict(out)


def _followers(user_id: str, since: str) -> dict:
    try:
        rows = _db().table("follows").select("*").eq("following_id", user_id).execute().data or []
    except Exception:
        return {"total": None, "gained": None, "by_day": []}
    dated = [r for r in rows if r.get("created_at")]
    by = _by_day(dated, since)
    return {"total": len(rows), "gained": sum(by.values()) if dated or not rows else None,
            "by_day": [{"day": d, "count": n} for d, n in sorted(by.items())]}


def insights(user_id: str, days: int = 30) -> dict:
    days = max(7, min(int(days), 90))
    since = (_today() - _dt.timedelta(days=days - 1)).isoformat()
    posts = _ad_posts(user_id)
    _snapshot(posts, user_id)
    ids = [p["id"] for p in posts]
    likes, comments = [], []
    for pid in ids:
        likes += _db().table("likes").select("*").eq("post_id", pid).execute().data or []
        comments += _db().table("comments").select("*").eq("post_id", pid).execute().data or []
    lk, cm, deltas = _by_day(likes, since), _by_day(comments, since), _snapshot_deltas(user_id, since)
    trend = []
    for i in range(days):
        d = (_today() - _dt.timedelta(days=days - 1 - i)).isoformat()
        trend.append({"day": d, "likes": lk.get(d, 0), "comments": cm.get(d, 0),
                      "views": deltas[d]["views"] if d in deltas else None,
                      "shares": deltas[d]["shares"] if d in deltas else None})
    totals = {"organic": {"ads": 0, "views": 0, "likes": 0, "comments": 0, "shares": 0},
              "paid": {"ads": 0, "views": 0, "likes": 0, "comments": 0, "shares": 0}}
    out_posts = []
    for p in posts:
        s = _stats(p)
        bucket = totals["paid" if p.get("is_boosted") else "organic"]
        bucket["ads"] += 1
        for k in ("views", "likes", "comments", "shares"):
            bucket[k] += s[k]
        out_posts.append({"post_id": p["id"], "campaign_id": p.get("ad_campaign_id"), "caption": p.get("caption") or "",
                          "thumbnail_url": p.get("thumbnail_url"), "public": not p.get("is_private"),
                          "paid": bool(p.get("is_boosted")), "published_at": p.get("created_at"), **s})
    for b in totals.values():
        b["engagement_rate"] = round((b["likes"] + b["comments"] + b["shares"]) / b["views"], 4) if b["views"] else None
    out_posts.sort(key=lambda x: x["views"], reverse=True)
    notes = ["Views and shares per day appear from the days your insights were opened (daily snapshots); "
             "likes and comments per day are exact."]
    if totals["paid"]["ads"]:
        notes.append("Boosted ads are counted as paid. A boosted ad's views can't yet be split into paid and organic.")
    return {"days": days, "totals": totals, "posts": out_posts, "trend": trend,
            "followers": _followers(user_id, since), "notes": notes}


@router.get("/insights")
async def get_insights(days: int = 30, user_id: str = Depends(creator)):
    """The creator's own published ads: real counts, trends and follower growth. Organic and paid apart."""
    return insights(user_id, days)


# ---------------------------------------------------------------- experiment lab

def compare(variants: list[dict]) -> dict:
    """Which published version is more engaging so far, only when the data is clear enough. Each variant:
    {"label", "views", "engagements"}."""
    live = [v for v in variants if v.get("views") is not None]
    if len(live) < 2:
        return {"verdict": "not_enough_data", "message": "Publish at least two versions to compare them."}
    short = [v for v in live if v["views"] < MIN_VIEWS]
    if short:
        need = ", ".join(f"{v['label']} ({MIN_VIEWS - v['views']} more views)" for v in short)
        return {"verdict": "not_enough_data",
                "message": f"Not enough data yet. Each version needs at least {MIN_VIEWS} views: {need}."}
    ranked = sorted(live, key=lambda v: v["engagements"] / v["views"], reverse=True)
    a, b = ranked[0], ranked[1]
    p1, p2 = a["engagements"] / a["views"], b["engagements"] / b["views"]
    pooled = (a["engagements"] + b["engagements"]) / (a["views"] + b["views"])
    se = math.sqrt(pooled * (1 - pooled) * (1 / a["views"] + 1 / b["views"])) if 0 < pooled < 1 else 0
    z = (p1 - p2) / se if se else 0.0
    if z < Z_95:
        return {"verdict": "no_clear_difference", "z": round(z, 2),
                "message": "No clear difference yet between the versions. Keep them running, or try a bigger change."}
    return {"verdict": "leader", "leader": a["label"], "z": round(z, 2),
            "message": f"{a['label']} is more engaging so far ({p1:.1%} vs {p2:.1%} of views engaged). "
                       "This is engagement on VIYO, not a prediction of sales."}


class ExperimentIn(BaseModel):
    name: str = Field("", max_length=80)
    tones: list[str]
    aspect_ratio: str = "9:16"
    use_character: bool = True
    idea_id: Optional[str] = None


def _experiment_out(e: dict, user_id: str) -> dict:
    camps = [c for c in (_db().table("ad_campaigns").select("*").eq("experiment_id", e["id"]).execute().data or [])
             if c.get("user_id") == user_id]
    variants, cmp_in = [], []
    for c in camps:
        label = TONES.get(c.get("variant_tone") or "", ("Version",))[0]
        post = None
        if c.get("published_post_id"):
            rows = _db().table("posts").select("*").eq("id", c["published_post_id"]).limit(1).execute().data or []
            post = rows[0] if rows and rows[0].get("user_id") == user_id and not rows[0].get("is_archived") else None
        s = _stats(post) if post else None
        variants.append({"campaign_id": c["id"], "tone": c.get("variant_tone"), "label": label,
                         "status": c.get("status") or "draft", "published": post is not None,
                         "public": bool(post) and not post.get("is_private"), "stats": s,
                         "thumbnail_url": core._resolve(c.get("thumbnail_url"))})
        if post and not post.get("is_private"):
            cmp_in.append({"label": label, "views": s["views"], "engagements": s["likes"] + s["comments"] + s["shares"]})
    order = list(TONES)
    variants.sort(key=lambda v: order.index(v["tone"]) if v["tone"] in order else 99)
    return {"id": e["id"], "name": e.get("name") or "", "brand_id": e.get("brand_id"), "created_at": e.get("created_at"),
            "variants": variants, "comparison": compare(cmp_in)}


def _owned_experiment(experiment_id: str, user_id: str) -> dict:
    rows = _db().table("ad_experiments").select("*").eq("id", _uuid(experiment_id)).limit(1).execute().data or []
    if not rows or rows[0].get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    return rows[0]


@router.post("/brands/{brand_id}/experiments")
async def create_experiment(brand_id: str, body: ExperimentIn, user_id: str = Depends(creator)):
    """2-6 versions of an ad in different tones, as drafts. Nothing is charged until each one is generated."""
    import ads_brands
    b = owned_brand(brand_id, user_id)
    if b.get("archived"):
        raise HTTPException(status_code=400, detail="This brand is archived. Restore it first.")
    tones = list(dict.fromkeys(t for t in body.tones))
    if not 2 <= len(tones) <= 6 or any(t not in TONES for t in tones):
        raise HTTPException(status_code=400, detail="Pick 2 to 6 different versions.")
    base_brief = ""
    if body.idea_id:
        rows = _db().table("ad_ideas").select("*").eq("id", _uuid(body.idea_id)).limit(1).execute().data or []
        if not rows or rows[0].get("owner_id") != user_id or rows[0].get("brand_id") != b["id"]:
            raise HTTPException(status_code=404, detail="Not found.")
        i = rows[0]
        base_brief = f"{i.get('title')}. Opening: {i.get('hook')}. {i.get('concept')} "
    name = body.name.strip() or f"{b.get('name') or 'Brand'}: {' vs '.join(TONES[t][0] for t in tones)}"[:80]
    exp = _db().table("ad_experiments").insert({"owner_id": user_id, "brand_id": b["id"], "name": name}).execute().data[0]
    for t in tones:
        label, style, tone = TONES[t]
        c = await ads_brands.new_ad_from_brand(b["id"], ads_brands.NewAdIn(
            ad_style=style, aspect_ratio=body.aspect_ratio,
            use_character=body.use_character or t == "character"), user_id)
        _db().table("ad_campaigns").update({
            "experiment_id": exp["id"], "variant_tone": t,
            "creative_brief": (base_brief + f"Tone for this version ({label}): {tone}")[:1500]}).eq("id", c["id"]).execute()
    return _experiment_out(exp, user_id)


@router.get("/experiments")
async def list_experiments(user_id: str = Depends(creator)):
    rows = (_db().table("ad_experiments").select("*").eq("owner_id", user_id).order("created_at", desc=True)
            .execute().data) or []
    return [_experiment_out(e, user_id) for e in rows]


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, user_id: str = Depends(creator)):
    return _experiment_out(_owned_experiment(experiment_id, user_id), user_id)


@router.get("/experiments-tones")
async def experiment_tones(user_id: str = Depends(creator)):
    return [{"key": k, "label": v[0], "description": v[2]} for k, v in TONES.items()]
