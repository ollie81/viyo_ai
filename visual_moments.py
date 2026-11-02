"""
"No talking" mode: Shorts from videos with no speech (cooking, crafts,
gameplay, travel, music-only), chosen by what is SEEN, not what is said.

  1. Candidate windows come from the whole-video visual pass
     (video_analysis): built on shot boundaries, so a clip starts and ends on
     a cut, and long continuous shots are scanned with a sliding window.
  2. Each window is measured: on-screen activity (picture change between
     samples), sound energy if there is sound (a sizzle, a beat drop), and
     picture quality (sharpness, exposure). Dark or blurry windows are dropped.
  3. The best few go to a vision model as three still frames each, with the
     creator's optional description. It rates how appealing and how complete
     the moment is, says what happens, and writes a short on-screen title that
     only describes what is visible.
  4. Final 0-100 score blends that judgement with the measurements. Weak ones
     are dropped; fewer clips is a valid answer. No fallback clip is ever made up.
"""
from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import clip_selection as cs
import video_analysis as va

MIN_SECONDS = 10.0
TARGET_SECONDS = 20.0
MAX_SECONDS = 35.0
SLIDE_STEP = 8.0
MAX_REVIEW = 8
MIN_SCORE = 56
MIN_QUALITY = 3.0          # 0-10 measured picture quality
MIN_BRIGHTNESS = 25.0      # median luma; below this the picture is too dark to post
MIN_SHARPNESS = 15.0       # median Laplacian variance at 320px; below this it is blurry
MAX_TITLE_WORDS = 7
REVIEW_BATCH = 4


@dataclass
class Window:
    start: float
    end: float
    cuts: int = 0
    activity: float = 0.0       # 0-1, relative to this video
    sound: Optional[float] = None
    quality: Optional[float] = None
    review: dict = field(default_factory=dict)
    score: int = 0
    unusable: str = ""         # "too_dark" / "too_blurry" when measured so

    @property
    def duration(self) -> float:
        return self.end - self.start


def target_count(duration: float) -> int:
    return int(max(1, min(4, round(duration / 90.0))))


def _activity_series(tl: va.VisualTimeline) -> np.ndarray:
    """Picture change between consecutive samples (cuts excluded), scaled so
    this video's busy moments sit near 1."""
    n = len(tl.times)
    change = np.zeros(n)
    for k in range(1, n):
        if not tl.cut_before[k]:
            change[k] = va._bhattacharyya(tl.hist[k - 1], tl.hist[k])
    ref = float(np.percentile(change, 90)) if n else 0.0
    return np.clip(change / ref, 0, 1) if ref > 1e-6 else change


def _sound_series(ff, times: np.ndarray) -> Optional[np.ndarray]:
    if ff is None or getattr(ff, "times", np.zeros(0)).size == 0:
        return None
    lo_db, hi_db = np.percentile(ff.rms_db, 10), np.percentile(ff.rms_db, 95)
    if hi_db - lo_db < 3:
        return None
    out = np.zeros(len(times))
    for i, t in enumerate(times):
        a, b = np.searchsorted(ff.times, [t, t + 1.0])
        if b > a:
            out[i] = (float(np.mean(ff.rms_db[a:b])) - lo_db) / (hi_db - lo_db)
    return np.clip(out, 0, 1)


def build_windows(tl: va.VisualTimeline, duration: float) -> list[Window]:
    """Shot-aligned windows of MIN-MAX seconds; long shots are scanned."""
    shots = tl.shots or []
    if not shots:
        shots = [va.Shot(0.0, duration, 0, max(0, len(tl.times) - 1))]
    wins: list[Window] = []
    for i, s in enumerate(shots):
        # windows that start at this cut and run over whole shots
        end, cuts = s.end, 0
        j = i
        while end - s.start < TARGET_SECONDS and j + 1 < len(shots) and shots[j + 1].end - s.start <= MAX_SECONDS:
            j += 1
            end, cuts = shots[j].end, cuts + 1
        if MIN_SECONDS <= end - s.start <= MAX_SECONDS:
            wins.append(Window(s.start, min(end, duration), cuts))
        # long single shots: slide through them
        if s.end - s.start > MAX_SECONDS:
            t = s.start
            while t + TARGET_SECONDS <= s.end:
                wins.append(Window(t, t + TARGET_SECONDS, 0))
                t += SLIDE_STEP
    if not wins and duration >= 6.0:
        wins.append(Window(0.0, min(duration, MAX_SECONDS), 0))
    return wins


def measure(wins: list[Window], tl: va.VisualTimeline, ff=None) -> None:
    act = _activity_series(tl)
    snd = _sound_series(ff, tl.times)
    for w in wins:
        sl = tl._slice(w.start, w.end)
        w.activity = float(np.mean(act[sl])) if len(act[sl]) else 0.0
        w.sound = float(np.mean(snd[sl])) if snd is not None and len(snd[sl]) else None
        v = va.window_visual(tl, w.start, w.end)
        w.quality, _ = va.visual_quality_score(v)
        if v.get("available"):
            dark_share = float(np.mean(tl.brightness[sl] < MIN_BRIGHTNESS)) if len(tl.brightness[sl]) else 0.0
            if dark_share > 0.25:
                w.unusable = "too_dark"
            elif v["sharpness_median"] < MIN_SHARPNESS:
                w.unusable = "too_blurry"


def _measured(w: Window) -> float:
    """0-1 from what was measured: activity, a little variety, sound."""
    variety = min(w.cuts, 4) / 4.0
    parts = [(0.65, w.activity), (0.15, variety)]
    if w.sound is not None:
        parts.append((0.2, w.sound))
    total = sum(p for p, _ in parts)
    return sum(p * v for p, v in parts) / total


def shortlist(wins: list[Window], k: int = MAX_REVIEW) -> list[Window]:
    """Best measured, non-overlapping, never dark or blurry."""
    ok = [w for w in wins if not w.unusable and (w.quality is None or w.quality >= MIN_QUALITY)]
    ok.sort(key=_measured, reverse=True)
    picked: list[Window] = []
    for w in ok:
        if all(min(w.end, p.end) - max(w.start, p.start) <= 2.0 for p in picked):
            picked.append(w)
        if len(picked) >= k:
            break
    return sorted(picked, key=lambda w: w.start)


PROMPT = """You are a short-form video editor choosing moments from a video with NO talking
(for example cooking, crafts, gameplay, travel, a performance). Each candidate below comes with
three still frames from its beginning, middle and end. {guidance}

For each candidate return:
- appeal 0-10: would this stop a scroller? (satisfying action, a transformation, a reveal,
  something beautiful or surprising scores high; static, empty, dark or repetitive scores low)
- complete 0-10: does it feel like a whole moment with a payoff (the pour, the flip, the reveal),
  rather than a random slice?
- what: one sentence saying what happens, ONLY from what is visible
- title: an on-screen title of at most 7 words describing what is visible. No claims, no numbers
  or facts you cannot see, no clickbait that isn't true.
- caption: one ready-to-post line; hashtags: 3-5 lowercase tags without #
- verdict: "publish" or "reject"

Return ONLY JSON: {{"reviews": [{{"id": int, "appeal": int, "complete": int, "what": str,
"title": str, "caption": str, "hashtags": [str], "verdict": "publish|reject"}}]}}

CANDIDATES
{blocks}
"""


def _clean_title(t: str) -> str:
    t = re.sub(r"\s+", " ", str(t or "").strip().strip('"“”'))
    return " ".join(t.split()[:MAX_TITLE_WORDS])


def review(wins: list[Window], llm: Callable, frames_for: Callable[[Window], list],
           guidance: str = "") -> int:
    """Fills w.review; returns how many were reviewed. Raises
    SelectionUnavailable if no batch could be reviewed at all."""
    note = (f'The creator described the video as: "{guidance.strip()[:400]}". Use it only to understand '
            "what you see; never state anything from it that the frames don't show.") if guidance.strip() else ""
    items = list(enumerate(wins))
    batches = [items[i:i + REVIEW_BATCH] for i in range(0, len(items), REVIEW_BATCH)]

    def run(batch) -> dict:
        blocks = "\n".join(f"--- CANDIDATE {i}: {w.duration:.0f}s" for i, w in batch)
        content: list = [{"type": "text", "text": PROMPT.format(guidance=note, blocks=blocks)}]
        for i, w in batch:
            imgs = frames_for(w)
            if imgs:
                content.append({"type": "text", "text": f"Frames for CANDIDATE {i}:"})
                content.extend({"type": "image_url", "image_url": {"url": u, "detail": "low"}} for u in imgs)
        try:
            data = cs.parse_json_object(llm([{"role": "user", "content": content}]))
        except Exception as e:
            print(f"[WARN] visual review unavailable for a batch: {e}")
            return {}
        out = {}
        for r in (data.get("reviews") if isinstance(data, dict) else None) or []:
            try:
                out[int(r["id"])] = r
            except (KeyError, TypeError, ValueError):
                continue
        return out

    got: dict = {}
    with ThreadPoolExecutor(max_workers=min(2, max(1, len(batches)))) as pool:
        for part in pool.map(run, batches):
            got.update(part)
    if wins and not got:
        raise cs.SelectionUnavailable("visual review returned nothing usable")

    def n(v):
        try:
            return max(0, min(10, int(round(float(v)))))
        except (TypeError, ValueError):
            return None

    for i, w in enumerate(wins):
        r = got.get(i)
        if not r:
            continue
        w.review = {"appeal": n(r.get("appeal")), "complete": n(r.get("complete")),
                    "what": str(r.get("what", ""))[:300], "title": _clean_title(r.get("title")),
                    "caption": str(r.get("caption", ""))[:300],
                    "hashtags": cs.clean_hashtags(r.get("hashtags")),
                    "verdict": "reject" if str(r.get("verdict", "")).lower() == "reject" else "publish"}
    return len(got)


def score(w: Window, rank: float) -> int:
    """0-100: the reviewer's read of appeal and completeness, the measured
    activity rank among candidates, and picture quality."""
    rv = w.review
    if not rv or rv.get("appeal") is None:
        return 0
    q = (w.quality / 10.0) if w.quality is not None else 0.6
    s = 0.55 * rv["appeal"] / 10 + 0.20 * (rv.get("complete") or 0) / 10 + 0.15 * rank + 0.10 * q
    return int(round(100 * s))


@dataclass
class VisualResult:
    clips: list
    spares: list
    target: int
    rejected: list
    note: str
    analysis: dict


def select(tl: va.VisualTimeline, duration: float, llm: Callable, frames_for: Callable,
           ff=None, guidance: str = "", refine_cut: Optional[Callable] = None,
           min_score: int = MIN_SCORE) -> VisualResult:
    target = target_count(duration)
    wins = build_windows(tl, duration)
    measure(wins, tl, ff)
    short = shortlist(wins)
    analysis = {"mode": "visual", "windows": len(wins), "reviewed": 0,
                "dark_or_blurry": sum(1 for w in wins if w.unusable or (w.quality is not None and w.quality < MIN_QUALITY))}
    if not short:
        return VisualResult([], [], target, [], "Every part of this video was too dark or blurry to use.", analysis)

    # End exactly on the cut, not up to a second into the next shot.
    if refine_cut is not None:
        for w in short:
            nxt = [s.start for s in tl.shots if w.end - 1.5 <= s.start <= w.end + 0.5]
            if nxt:
                w.end = max(w.start + MIN_SECONDS, refine_cut(nxt[0]) - 0.04)

    analysis["reviewed"] = review(short, llm, frames_for, guidance)
    measured = sorted(_measured(w) for w in short)
    rejected, passed = [], []
    for w in short:
        rank = (measured.index(_measured(w)) + 1) / len(measured)
        w.score = score(w, rank)
        rv = w.review
        if not rv:
            rejected.append({"start": w.start, "reason": "not_reviewed"})
        elif rv["verdict"] == "reject":
            rejected.append({"start": w.start, "score": w.score, "reason": "review_rejected", "what": rv.get("what")})
        elif w.score < min_score:
            rejected.append({"start": w.start, "score": w.score, "reason": "below_quality_floor", "what": rv.get("what")})
        elif not rv.get("title"):
            rejected.append({"start": w.start, "score": w.score, "reason": "no_title"})
        else:
            passed.append(w)
    passed.sort(key=lambda w: -w.score)
    clips, spares = passed[:target], passed[target:target + 2]
    note = "" if len(clips) >= target else (
        f"Found {len(clips)} of up to {target} possible clips — the other moments weren't strong enough."
        if clips else "No moment in this video was visually strong enough to make a Short.")
    analysis["rejected_reasons"] = {}
    for r in rejected:
        analysis["rejected_reasons"][r["reason"]] = analysis["rejected_reasons"].get(r["reason"], 0) + 1
    return VisualResult([to_clip(w) for w in clips], [to_clip(w) for w in spares], target, rejected, note, analysis)


def to_clip(w: Window) -> dict:
    rv = w.review
    return {
        "start": round(w.start, 3), "end": round(w.end, 3), "duration": round(w.duration, 3),
        "score": w.score, "category": "visual", "title": rv.get("title", ""), "reason": rv.get("what", ""),
        "hook_line": "", "caption": rv.get("caption", ""), "hashtags": rv.get("hashtags", []),
        "transcript": "", "overlay_hook": rv.get("title") or None, "overlay_status": "visual_title",
        "breakdown": {"mode": "visual", "appeal": rv.get("appeal"), "complete": rv.get("complete"),
                      "activity": round(w.activity, 3), "sound": None if w.sound is None else round(w.sound, 3),
                      "picture_quality": w.quality, "cuts": w.cuts},
        "review": {"status": "reviewed", "verdict": rv.get("verdict"), "what": rv.get("what")},
    }
