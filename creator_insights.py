"""
Learning from how a creator's published clips actually perform — without
pretending to know what will go viral.

What it does: compares how THIS creator's audience responded (completion,
rewatches, shares, saves, comments) to clips of different kinds (category,
length) and nudges future selection toward the kinds that have done better
for them, by at most +-MAX_POINTS on a 0-100 score. What it never does:
claim a clip will perform, or act on thin data. With fewer than MIN_CLIPS
clips that each have MIN_VIEWS views, it returns "insufficient_data" and
changes nothing.

Data comes from repurpose_clip_performance (see repurpose_insights_schema.sql),
one row per published clip. Nothing in the app writes to it yet; until the
post-metrics are wired in, the prior is inert and selection is unchanged.
"""
from __future__ import annotations

from typing import Callable, Optional

MIN_VIEWS = 200
MIN_CLIPS = 8
MAX_POINTS = 4.0
SHRINK = 10.0          # a bucket with n clips counts n/(n+SHRINK) of its measured difference
DURATION_BUCKETS = ((0, 25, "short"), (25, 45, "medium"), (45, 10_000, "long"))


def engagement_index(row: dict) -> Optional[float]:
    """0-1 blend of how the audience responded; None if too few views to trust."""
    views = float(row.get("views") or 0)
    if views < MIN_VIEWS:
        return None

    def rate(key):
        return min(1.0, float(row.get(key) or 0) / views)

    completion = min(1.0, float(row.get("completion_rate") or 0))
    rewatch = min(1.0, float(row.get("rewatch_rate") or 0))
    actions = min(1.0, (rate("shares") + rate("saves")) * 8)    # ~12% of viewers acting = full marks
    comments = min(1.0, rate("comments") * 20)
    return 0.4 * completion + 0.15 * rewatch + 0.3 * actions + 0.15 * comments


def _bucket(duration: float) -> str:
    for lo, hi, name in DURATION_BUCKETS:
        if lo <= duration < hi:
            return name
    return "long"


def build_prior(rows: list) -> Optional[Callable]:
    """A creator_prior callable for clip_selection.AnalysisContext, or None."""
    scored = [(r, engagement_index(r)) for r in rows or []]
    scored = [(r, e) for r, e in scored if e is not None]
    if len(scored) < MIN_CLIPS:
        return None
    overall = sum(e for _, e in scored) / len(scored)
    by_cat: dict = {}
    by_len: dict = {}
    for r, e in scored:
        by_cat.setdefault(r.get("category") or "other", []).append(e)
        by_len.setdefault(_bucket(float(r.get("duration") or 0)), []).append(e)

    def effect(group: list) -> float:
        n = len(group)
        diff = sum(group) / n - overall
        return diff * n / (n + SHRINK)

    cat_fx = {k: effect(v) for k, v in by_cat.items()}
    len_fx = {k: effect(v) for k, v in by_len.items()}
    spread = max([abs(v) for v in list(cat_fx.values()) + list(len_fx.values())] + [1e-9])

    def prior(candidate) -> tuple:
        raw = cat_fx.get(candidate.category, 0.0) + len_fx.get(_bucket(candidate.duration), 0.0)
        points = max(-MAX_POINTS, min(MAX_POINTS, raw / (2 * spread) * MAX_POINTS))
        return round(points, 2), {"status": "applied", "clips_learned_from": len(scored),
                                  "category_effect": round(cat_fx.get(candidate.category, 0.0), 3),
                                  "length_effect": round(len_fx.get(_bucket(candidate.duration), 0.0), 3)}

    return prior


def load_prior(admin, user_id: str) -> Optional[Callable]:
    """Reads this creator's history; any failure means 'no prior', never a failed job."""
    try:
        res = admin.table("repurpose_clip_performance").select("*").eq("user_id", user_id).limit(500).execute()
        return build_prior(res.data or [])
    except Exception as e:
        print(f"[INFO] creator history unavailable: {e}")
        return None
