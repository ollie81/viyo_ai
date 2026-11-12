"""
Editorial review of finalist clips before any render time is spent on them.

A second, independent pass over each finalist — with the 20-30 seconds
BEFORE it, the sentence after it, the proposed hook and, when available,
three frames from the source — asking the questions a human editor would:
does this make sense on its own, is the opening strong, is the ending
satisfying, does the hook say what the speaker actually says, and does the
picture work?

The verdict is applied by code, not trusted blindly:
  reject  -> the clip is dropped and a spare takes its place
  fix     -> the clip is kept but loses its overlay hook if the hook was the problem
  publish -> the clip goes on
A clip is rejected outright when it can't stand alone, or when its opening
or ending is judged weak. A review that can't run leaves the clip
"unreviewed" — it is not treated as passed, and the job says so.
"""
from __future__ import annotations

import base64
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import clip_selection as cs

CONTEXT_SENTENCES_BEFORE = 6
MIN_OPENING = 5
MIN_ENDING = 5
BATCH = 4
FRAME_HEIGHT = 360

PROMPT = """You are a strict senior editor doing final review of short clips cut from a longer video.
For each clip you get: what was said just BEFORE it (context the viewer will NOT see), the clip's own
transcript, the sentence AFTER it, the on-screen hook (if any) and sometimes 3 frames from the clip.

{context}Judge each clip as a viewer who has never seen the rest of the video, but who WILL see the post's title
and caption, so the general topic is known:
- standalone: false ONLY if the viewer could not follow the clip's main point without hearing something
  else first (e.g. it answers a question never asked, or leans on a specific thing never named). Passing
  references to the speaker's own experiences, jokes, or things the topic already implies do NOT make a
  clip non-standalone.
- opening 0-10: would the first two seconds make someone keep watching?
- ending 0-10: does it close the thought satisfyingly, rather than stop abruptly or trail off?
- hook_accurate: true/false whether the on-screen hook says only what the speaker actually says (null if no hook).
- visual: if frames are given, what is on screen and does it work vertically (a person talking, slides,
  screen share, b-roll, a static image, text-heavy)? Mention if a face is cut off, the frame is dark/blurry,
  or there is distracting text. "visual_ok" is true/false (null if no frames).
- verdict: "publish" if it is worth posting as it is, "fix" if it is usable only after a small change
  (say what), "reject" if it should not be posted.
- issues: concrete problems, empty if none. If standalone is false you MUST say exactly what the viewer
  would be missing (e.g. "refers to 'the study' never introduced"). Be specific and honest; do not praise by default.

Return ONLY JSON: {{"reviews": [{{"id": int, "standalone": bool, "opening": int, "ending": int,
"hook_accurate": bool|null, "visual_ok": bool|null, "visual": "short note", "verdict": "publish|fix|reject",
"issues": [str]}}]}}

CLIPS
{clips}
"""


def _clip_block(c: cs.Candidate, idx: int, sentences: list, labels: Optional[list]) -> str:
    before = sentences[max(0, c.start_sentence - CONTEXT_SENTENCES_BEFORE): c.start_sentence]
    after = sentences[c.end_sentence + 1: c.end_sentence + 2]

    def line(s):
        who = f"[{labels[s.id]}] " if labels and labels[s.id] else ""
        return f"{who}{s.text}"

    text = c.transcript
    st = getattr(c, "structure", None) or {}
    if st.get("joined"):
        # Two parts of the video, joined; the viewer sees "Later in the video" between them.
        a, b = st["joined"]
        dropped = set(st.get("dropped") or [])
        first = " ".join(s.text for s in sentences[c.start_sentence:a] if s.id not in dropped)
        text = f"{first} [ON SCREEN: \"LATER IN THE VIDEO\"] {' '.join(s.text for s in sentences[a:b + 1])}"
    return (f"--- CLIP {idx} (title: {c.title})\n"
            f"BEFORE (not shown to viewer): {' / '.join(line(s) for s in before) or '(start of video)'}\n"
            f"CLIP: {text}\n"
            f"AFTER: {' / '.join(line(s) for s in after) or '(end of video)'}\n"
            f"ON-SCREEN HOOK: {c.overlay_hook or '(none)'}\n")


def extract_frames(source_path: str, c: cs.Candidate, work_dir: str, n: int = 3) -> list[str]:
    """Data-URL JPEGs from the source at evenly spaced points in the clip."""
    urls = []
    for k in range(n):
        t = c.start + (c.end - c.start) * (0.15 + 0.35 * k)
        out = os.path.join(work_dir, f"rv_{int(c.start * 1000)}_{k}.jpg")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", source_path,
                            "-frames:v", "1", "-vf", f"scale=-2:{FRAME_HEIGHT}", "-q:v", "5", out],
                           capture_output=True, timeout=60)
        if r.returncode == 0 and os.path.exists(out):
            with open(out, "rb") as f:
                urls.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii"))
            os.remove(out)
    return urls


def review_candidates(cands: list, sentences: list, labels: Optional[list], llm: Callable[[list], str],
                      frames_for: Optional[Callable[[cs.Candidate], list]] = None,
                      video_summary: Optional[str] = None) -> dict[int, dict]:
    """Returns {index in `cands`: review}. Missing index = not reviewed."""
    results: dict[int, dict] = {}

    def run(batch: list[tuple[int, cs.Candidate]]) -> dict[int, dict]:
        blocks = "\n".join(_clip_block(c, i, sentences, labels) for i, c in batch)
        context = f"WHAT THE WHOLE VIDEO IS ABOUT: {video_summary}\n\n" if video_summary else ""
        content: list = [{"type": "text", "text": PROMPT.format(clips=blocks, context=context)}]
        if frames_for is not None:
            for i, c in batch:
                imgs = frames_for(c)
                if imgs:
                    content.append({"type": "text", "text": f"Frames for CLIP {i}:"})
                    content.extend({"type": "image_url", "image_url": {"url": u, "detail": "low"}} for u in imgs)
        try:
            data = cs.parse_json_object(llm([{"role": "user", "content": content}]))
        except Exception as e:
            print(f"[WARN] clip review unavailable for a batch: {e}")
            return {}
        out = {}
        for r in (data.get("reviews") if isinstance(data, dict) else None) or []:
            try:
                idx = int(r["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if idx in dict(batch):
                out[idx] = _clean(r)
        return out

    items = list(enumerate(cands))
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    with ThreadPoolExecutor(max_workers=min(3, max(1, len(batches)))) as pool:
        for part in pool.map(run, batches):
            results.update(part)
    return results


def _clean(r: dict) -> dict:
    def b(v):
        return v if isinstance(v, bool) else None

    def n(v):
        try:
            return max(0, min(10, int(round(float(v)))))
        except (TypeError, ValueError):
            return None

    verdict = str(r.get("verdict", "")).lower()
    return {"standalone": b(r.get("standalone")), "opening": n(r.get("opening")), "ending": n(r.get("ending")),
            "hook_accurate": b(r.get("hook_accurate")), "visual_ok": b(r.get("visual_ok")),
            "visual": str(r.get("visual", ""))[:200],
            "verdict": verdict if verdict in ("publish", "fix", "reject") else "unknown",
            "issues": [str(i)[:200] for i in (r.get("issues") or []) if i][:5]}


def apply_reviews(result: cs.SelectionResult, reviews: dict[int, dict], pool: list) -> cs.SelectionResult:
    """
    Applies verdicts to `pool` (clips + spares, in rank order) and rebuilds
    the selection from what passed: spares replace rejected clips.
    """
    kept: list[cs.Candidate] = []
    for i, c in enumerate(pool):
        rv = reviews.get(i)
        if rv is None:
            c.review = {"status": "unreviewed"}
            kept.append(c)
            continue
        c.review = {"status": "reviewed", **rv}
        reason = reject_reason(rv)
        if reason == "review_not_standalone" and rv.get("verdict") == "fix":
            result.fixable.append(c)
        if reason:
            result.rejected.append({"title": c.title, "score": c.score, "reason": reason,
                                    "verdict": rv.get("verdict"), "standalone": rv.get("standalone"),
                                    "opening": rv.get("opening"), "ending": rv.get("ending"),
                                    "issues": rv.get("issues"), "opens": c.transcript[:90]})
            continue
        if rv.get("hook_accurate") is False and c.overlay_hook:
            c.overlay_hook, c.overlay_status = None, "removed_by_review"
        kept.append(c)
    result.clips = kept[: result.target]
    result.spares = kept[result.target:]
    if len(result.clips) >= result.target:
        result.note = ""
    else:
        result.note = cs._selection_note(result.clips, result.target, kept, result.rejected)
    return result


def reject_reason(rv: dict) -> Optional[str]:
    if rv.get("verdict") == "reject":
        return "review_rejected"
    # "Not standalone" alone is a judgement call the model makes cheaply; it only
    # rejects when the same review also withholds "publish" or names a concrete problem.
    if rv.get("standalone") is False and (rv.get("verdict") != "publish" or rv.get("issues")):
        return "review_not_standalone"
    if rv.get("opening") is not None and rv["opening"] < MIN_OPENING:
        return "review_weak_opening"
    if rv.get("ending") is not None and rv["ending"] < MIN_ENDING:
        return "review_weak_ending"
    if rv.get("visual_ok") is False:
        return "review_bad_visual"
    return None


RECOVER_MAX = 4
RECOVER_SENTENCES_BEFORE = (3, 5)


def _raw_from(c: cs.Candidate, start_sentence: int) -> dict:
    return {"start_sentence": start_sentence, "end_sentence": c.end_sentence, "category": c.category,
            "scores": dict(c.rubric), "opens_mid_conversation": False, "title": c.title, "reason": c.reason,
            "overlay_hook": "", "caption": c.caption, "hashtags": list(c.hashtags), "topic": c.topic}


def recover_with_context(result: cs.SelectionResult, sentences: list, words: list, video_seconds: float,
                         ctx: cs.AnalysisContext, labels: Optional[list], llm: Callable,
                         frames_for: Optional[Callable] = None, video_summary: Optional[str] = None) -> dict:
    """
    A clip the editor marks "fix — the viewer is missing the setup" is what a
    human editor would start a few sentences earlier. Do that (never inventing
    anything: the added words are the speaker's own), score the longer clip like
    any other, and have it reviewed again. It only joins the selection if it
    then passes every check; otherwise it stays rejected.
    """
    if len(result.clips) >= result.target or not result.fixable:
        return {"attempted": 0, "recovered": 0}
    tried: list[cs.Candidate] = []
    for c in sorted(result.fixable, key=lambda x: -x.score)[:RECOVER_MAX]:
        for extra in RECOVER_SENTENCES_BEFORE:
            s0 = max(0, c.start_sentence - extra)
            if s0 == c.start_sentence:
                continue
            ext, why = cs.evaluate_candidate(_raw_from(c, s0), sentences, words, video_seconds, ctx)
            if ext is None or ext.duration > cs.MAX_CLIP_SECONDS or cs.over_length_choice(ext, ctx.length_pref):
                continue
            ext.review = {"recovered_with_context_sentences": c.start_sentence - ext.start_sentence}
            tried.append(ext)
            break
    if not tried:
        return {"attempted": 0, "recovered": 0}
    reviews = review_candidates(tried, sentences, labels, llm, frames_for, video_summary)
    recovered = 0
    for i, ext in enumerate(tried):
        rv = reviews.get(i)
        if rv is None or reject_reason(rv) or ext.score < result.min_score:
            result.rejected.append({"title": ext.title, "score": ext.score, "reason": "recovery_failed",
                                    "issues": (rv or {}).get("issues")})
            continue
        if len(result.clips) >= result.target or cs._conflicts(ext, result.clips):
            continue
        ext.review = {"status": "reviewed", **rv, **ext.review}
        result.clips.append(ext)
        recovered += 1
    result.clips.sort(key=lambda x: -x.score)
    if len(result.clips) >= result.target:
        result.note = ""
    elif result.clips:
        result.note = cs._selection_note(result.clips, result.target, result.clips, result.rejected)
    return {"attempted": len(tried), "recovered": recovered}
