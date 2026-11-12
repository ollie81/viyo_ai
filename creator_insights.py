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


# ------------------------------------------------------------------ what the creator did with their clips

FEEDBACK_MIN_EVENTS = 5       # fewer than this and nothing changes
FEEDBACK_MAX_POINTS = 3.0
FEEDBACK_WEIGHTS = {"downloaded": 1.0, "shared": 1.0, "exported": 1.0, "kept": 1.0, "not_useful": -1.5,
                    "regenerated": 0.0}   # changing the edits says nothing about the moment itself


def build_feedback_prior(events: list) -> Optional[Callable]:
    """A creator_prior from THIS creator's own actions on earlier clips: kinds and lengths of clip they keep and
    download get a small nudge up, kinds they mark "not useful" a small nudge down (at most +-3 points on 0-100).
    A download is treated as "they wanted it", never as "it was published" or "it performed"."""
    rows = [e for e in events or [] if e.get("event") in FEEDBACK_WEIGHTS and FEEDBACK_WEIGHTS[e["event"]] != 0]
    if len(rows) < FEEDBACK_MIN_EVENTS:
        return None
    cat: dict = {}
    length: dict = {}
    for e in rows:
        w = FEEDBACK_WEIGHTS[e["event"]]
        cat[e.get("category") or "other"] = cat.get(e.get("category") or "other", 0.0) + w
        if e.get("duration"):
            b = _bucket(float(e["duration"]))
            length[b] = length.get(b, 0.0) + w
    n = len(rows)

    def prior(candidate) -> tuple:
        c = cat.get(candidate.category, 0.0) / (n + SHRINK)
        d = length.get(_bucket(candidate.duration), 0.0) / (n + SHRINK)
        points = max(-FEEDBACK_MAX_POINTS, min(FEEDBACK_MAX_POINTS, (c + d) * FEEDBACK_MAX_POINTS * 2))
        return round(points, 2), {"status": "applied", "source": "creator_feedback", "events": n}
    return prior


def combine(*priors) -> Optional[Callable]:
    """Several priors added together (each already capped)."""
    ps = [p for p in priors if p is not None]
    if not ps:
        return None
    if len(ps) == 1:
        return ps[0]

    def prior(candidate) -> tuple:
        total, info = 0.0, []
        for p in ps:
            pts, i = p(candidate)
            total += pts
            info.append(i)
        return round(total, 2), {"status": "applied", "parts": info}
    return prior


def preferred_length(events: list) -> Optional[str]:
    """The clip length this creator keeps choosing to download, once it's clear (60% of at least 8 downloads)."""
    lens = [_bucket(float(e["duration"])) for e in events or []
            if e.get("event") in ("downloaded", "shared", "exported", "kept") and e.get("duration")]
    if len(lens) < 8:
        return None
    best = max(set(lens), key=lens.count)
    return best if lens.count(best) >= 0.6 * len(lens) else None
