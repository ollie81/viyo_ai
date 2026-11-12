"""The AI Shorts judge on the OpenAI Decisions API.

ONE Decisions request per job evaluates every finalist clip (clips + spares)
with structured questions instead of a free-text review:

  A hook        score   Poor / Average / Good / Excellent - do the first seconds hold attention?
  B standalone  predicate - can a viewer follow it without the rest of the video?
  C story       score   - is there an interesting, useful or entertaining moment?
  D suitable    score   - would it work as a short on its own?
  E ending      score   - does it end naturally, without cutting off the point?
  F visual      score   - do the frames suit a vertical short? (only with frames)
  G hook_true   predicate - does the on-screen hook say only what is said? (only with a hook)
  H join_true   predicate - do parts joined from different moments keep the meaning? (only joined clips)

Shared input (the cacheable prefix, identical for every candidate): what the
whole video is, its numbered transcript, and the finalist clips as JSON. The
questions only name a clip, so each clip's text is sent once, not once per
question. Frames go only in the visual questions, at 360 px, three per clip.

Answers are validated (decisions.validate) and turned into the same review
record the rest of the pipeline already uses (standalone / opening / ending /
hook_accurate / visual_ok / join_faithful / verdict), plus the scores, their
level labels and plain explanations for the creator. Thresholds live in
THRESHOLDS; set them from labelled examples, not intuition. A low-confidence or
borderline answer is marked, and (when a second evaluator is available) those
clips alone get the existing editorial review as a second opinion.

An AI score is a judgement of the clip, not a prediction of views.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import clip_selection as cs
import decisions as dc

LEVELS = {
    "hook": [("Poor", "The first seconds are filler, a greeting, or hard to follow; a viewer would scroll."),
             ("Average", "It starts on topic but without anything that pulls the viewer in."),
             ("Good", "It opens on a clear claim, question, or tension that makes the viewer want more."),
             ("Excellent", "The first line is surprising, vivid, or high-stakes; very hard to scroll past.")],
    "story": [("Poor", "Nothing happens: small talk, logistics, or a fragment with no point."),
              ("Average", "A point is made but it is ordinary or forgettable."),
              ("Good", "A real story, insight, opinion with a reason, or genuinely useful advice."),
              ("Excellent", "A moment people would share: a twist, a strong emotion, or a striking idea.")],
    "suitable": [("Poor", "It does not work as a short on its own."),
                 ("Average", "It works as a short but needs the viewer's patience."),
                 ("Good", "A complete, watchable short as it is."),
                 ("Excellent", "A complete, tight short with a clear beginning, middle and end.")],
    "ending": [("Poor", "It stops mid-thought, before the point, or on an unrelated line."),
               ("Average", "It ends, but trails off or lingers after the point."),
               ("Good", "It ends on the point, answer or punchline."),
               ("Excellent", "It ends on a strong, satisfying final line.")],
    "visual": [("Poor", "Dark, blurry, mostly text, or nobody and nothing worth watching in frame."),
               ("Average", "Watchable but static or cluttered."),
               ("Good", "A clear subject that will crop well to vertical."),
               ("Excellent", "Clear, well-lit, expressive subject; looks made for vertical.")],
}

THRESHOLDS = {
    "standalone": 0.5,        # P(standalone) below this: the clip needs missing context
    "hook_true": 0.5,         # below: the on-screen hook is removed
    "join_true": 0.6,         # below: the joined part is undone
    "min_opening": 1.0,       # hook score below "Average" (index 1): weak opening
    "min_ending": 1.0,        # ending score below "Average": abrupt ending
    "min_visual": 0.75,       # visual score clearly "Poor"
    "low_confidence": 0.5,    # a score answer below this confidence is flagged
    "borderline": 0.15,       # a predicate within this of its threshold is flagged
}
WEIGHTS = {"hook": 0.25, "story": 0.3, "suitable": 0.25, "ending": 0.2}
BLEND = 0.4                   # share of the final ranking from Decisions (the rest: the measured rubric)
MAX_CANDIDATES = 12           # cost guard: finalists evaluated per job
MAX_VISUAL = 8                # ...of which this many get frames
MAX_TRANSCRIPT_WORDS = 14000


def _transcript(sentences: list, labels: Optional[list]) -> str:
    words = sum(len(s.text.split()) for s in sentences)
    keep = 18 if words > MAX_TRANSCRIPT_WORDS else None
    out = []
    for s in sentences:
        t = " ".join(s.text.split()[:keep]) if keep else s.text
        who = f"{labels[s.id]} " if labels and labels[s.id] else ""
        out.append(f"S{s.id} {who}{t}")
    return "\n".join(out)


def _clip_json(i: int, c: cs.Candidate, sentences: list) -> dict:
    st = c.structure or {}
    before = sentences[max(0, c.start_sentence - 3): c.start_sentence]
    after = sentences[c.end_sentence + 1: c.end_sentence + 2]
    d = {"id": i, "title": c.title, "seconds": round(c.duration, 1), "sentences": f"S{c.start_sentence}-S{c.end_sentence}",
         "transcript": c.transcript, "said_just_before (viewer does not hear)": " ".join(s.text for s in before),
         "said_just_after": " ".join(s.text for s in after)}
    if c.overlay_hook:
        d["on_screen_hook"] = c.overlay_hook
    if st.get("joined"):
        d["joined_parts"] = ("Parts from different moments of the video, played back to back; viewers see "
                             "'LATER IN THE VIDEO' where they meet: " + "; ".join(f"S{a}-S{b}" for a, b in st.get("ranges") or []))
    return d


def build_request(pool: list, sentences: list, labels: Optional[list], video_map: Optional[dict],
                  frames: dict) -> tuple[list, list]:
    """(shared input, questions) for the finalists in `pool` (list of (index, candidate))."""
    vm = video_map or {}
    shared = [
        {"role": "developer", "content": [dc.text(
            "You judge short vertical clips cut from one long video, as a strict senior short-form editor. "
            "Judge each clip as a viewer who has never seen the rest of the video. Base every answer only on "
            "what is said and shown; loud or excited is not the same as interesting.")]},
        {"role": "user", "content": [
            dc.text(f"THE WHOLE VIDEO: {vm.get('summary') or 'not summarised'} (format: {vm.get('format', 'unknown')})."
                    f"\nNUMBERED TRANSCRIPT:\n{_transcript(sentences, labels)}"),
            dc.json_part({"clips": [_clip_json(i, c, sentences) for i, c in pool]}),
        ]},
    ]
    qs = []
    for i, c in pool:
        ref = f"Clip id {i} in the clips JSON (\"{c.title}\")."
        qs.append(dc.score(f"c{i}_hook", f"{ref} How strongly do its first two seconds capture attention?",
                           LEVELS["hook"]))
        qs.append(dc.predicate(f"c{i}_standalone", f"{ref} Can a viewer who has not seen the rest of the video "
                               "follow its main point? Passing references to the speaker's own life or to the "
                               "general topic are fine; answering a question nobody asked, or leaning on a "
                               "specific thing never named in the clip, is not."))
        qs.append(dc.score(f"c{i}_story", f"{ref} How valuable is what happens in it?", LEVELS["story"]))
        qs.append(dc.score(f"c{i}_suitable", f"{ref} How well does it work as a standalone short?",
                           LEVELS["suitable"]))
        qs.append(dc.score(f"c{i}_ending", f"{ref} How well does it end?", LEVELS["ending"]))
        if frames.get(i):
            qs.append(dc.score(f"c{i}_visual", [dc.text(f"{ref} These frames are from the clip, before it is "
                                                        "cropped to vertical. How well do they suit a vertical short?")]
                               + [dc.image(u) for u in frames[i]], LEVELS["visual"]))
        if c.overlay_hook:
            qs.append(dc.predicate(f"c{i}_hook_true", f"{ref} Does its on_screen_hook say only what the speaker "
                                   "actually says in the clip (no new facts, numbers or claims)?"))
        if (c.structure or {}).get("joined"):
            qs.append(dc.predicate(f"c{i}_join_true", f"{ref} Heard back to back, do its joined_parts keep exactly "
                                   "what the speaker meant and belong to the same story?"))
    return shared, qs


def _explain(key: str, a: Optional[dc.Answer]) -> Optional[str]:
    if a is None or a.refused:
        return None
    if a.type == "score":
        label = a.level or LEVELS[key][int(round(a.score))][0]
        desc = dict(LEVELS[key]).get(label, "")
        return f"{label}: {desc}"
    return None


def to_review(i: int, c: cs.Candidate, res: dc.Result) -> dict:
    """One clip's answers -> the pipeline's review record (+ scores, explanations, flags)."""
    T = THRESHOLDS
    a = {k: res.get(f"c{i}_{k}") for k in ("hook", "standalone", "story", "suitable", "ending", "visual",
                                           "hook_true", "join_true")}
    refused = [k for k, v in a.items() if v is not None and v.refused]
    scored = {k: v for k, v in a.items() if v is not None and not v.refused}
    core = ("hook", "standalone", "story", "suitable", "ending")
    if any(k not in scored for k in core):
        return {"status": "unreviewed", "decisions": {"refused": refused, "missing": [k for k in core if k not in scored]}}
    p_alone = scored["standalone"].probability
    flags = []
    for k in ("hook", "story", "suitable", "ending", "visual"):
        v = scored.get(k)
        if v is not None and v.confidence is not None and v.confidence < T["low_confidence"]:
            flags.append(f"low_confidence_{k}")
    if abs(p_alone - T["standalone"]) < T["borderline"]:
        flags.append("borderline_standalone")
    to10 = lambda s, n=4: int(round(10 * s / (n - 1)))           # noqa: E731
    composite = sum(WEIGHTS[k] * scored[k].score / 3 for k in WEIGHTS) * 100 * (0.5 + 0.5 * p_alone)
    standalone = p_alone >= T["standalone"]
    issues = []
    if not standalone:
        issues.append(f"Needs context the viewer won't have (chance it stands alone: {p_alone:.0%}).")
    if scored["hook"].score < T["min_opening"]:
        issues.append("Weak opening.")
    if scored["ending"].score < T["min_ending"]:
        issues.append("Ends abruptly or off the point.")
    visual = scored.get("visual")
    visual_ok = None if visual is None else visual.score >= T["min_visual"]
    if visual_ok is False:
        issues.append("The picture doesn't suit a vertical short.")
    hook_true = scored.get("hook_true")
    join_true = scored.get("join_true")
    verdict = "reject" if (not standalone or scored["suitable"].score < 0.75) else \
        ("fix" if issues else "publish")
    explanations = {k: e for k in ("hook", "story", "suitable", "ending", "visual") if (e := _explain(k, scored.get(k)))}
    explanations["standalone"] = f"{p_alone:.0%} chance a new viewer can follow it on its own."
    return {
        "status": "reviewed", "source": "decisions",
        "standalone": standalone, "opening": to10(scored["hook"].score), "ending": to10(scored["ending"].score),
        "hook_accurate": None if hook_true is None else hook_true.probability >= T["hook_true"],
        "visual_ok": visual_ok, "visual": explanations.get("visual", ""),
        "join_faithful": None if join_true is None else join_true.probability >= T["join_true"],
        "verdict": verdict, "issues": issues,
        "decisions": {"scores": {k: round(v.score, 2) for k, v in scored.items() if v.type == "score"},
                      "probabilities": {k: round(v.probability, 3) for k, v in scored.items() if v.type == "predicate"},
                      "confidence": {k: round(v.confidence, 3) for k, v in scored.items() if v.confidence is not None},
                      "composite": int(round(composite)), "explanations": explanations,
                      "flags": flags, "refused": refused, "needs_second_look": bool(flags or refused)},
    }


def review(pool: list, sentences: list, labels: Optional[list], video_map: Optional[dict],
           frames_for: Optional[Callable], client: dc.Client, cache_key: Optional[str] = None,
           second_opinion: Optional[Callable] = None) -> tuple[dict, dict]:
    """Evaluates `pool` (finalists in rank order). Returns ({pool index: review}, facts). Raises
    decisions.DecisionsUnavailable when the API can't be used (the caller falls back)."""
    picked = list(enumerate(pool))[:MAX_CANDIDATES]
    frames = {}
    if frames_for is not None:
        for i, c in picked[:MAX_VISUAL]:
            try:
                f = frames_for(c)
            except Exception:
                f = []
            if f:
                frames[i] = f[:3]
    shared, questions = build_request(picked, sentences, labels, video_map, frames)
    res = client.decide(shared, questions, cache_key=cache_key)
    reviews = {i: to_review(i, c, res) for i, c in picked}
    facts = {"requests": 1, "questions": len(questions), "candidates": len(picked), "with_frames": len(frames),
             "dropped_answers": len(res.dropped), "usage": res.usage, "cost_usd": res.cost_usd, "seconds": res.seconds,
             "second_opinions": 0}
    # A second look for low-confidence or borderline clips, with the existing editorial review (one batched call).
    unsure = [i for i, rv in reviews.items() if (rv.get("decisions") or {}).get("needs_second_look")
              or rv.get("status") == "unreviewed"]
    if unsure and second_opinion is not None:
        try:
            extra = second_opinion([pool[i] for i in unsure])
        except Exception as e:
            print(f"[WARN] second opinion unavailable: {e}")
            extra = {}
        for k, i in enumerate(unsure):
            rv2 = extra.get(k)
            if not rv2:
                continue
            base = reviews[i]
            merged = {**base, **{key: rv2.get(key) for key in ("standalone", "opening", "ending", "verdict", "issues")
                                 if rv2.get(key) is not None}}
            merged["status"] = "reviewed"
            merged.setdefault("decisions", {})["second_opinion"] = {k2: rv2.get(k2) for k2 in ("standalone", "opening",
                                                                                              "ending", "verdict")}
            reviews[i] = merged
            facts["second_opinions"] += 1
    # Clips Decisions couldn't judge (refusals, missing answers) stay "unreviewed": never treated as passed.
    reviews = {i: rv for i, rv in reviews.items() if rv.get("status") == "reviewed"}
    return reviews, facts


def rerank(pool: list, reviews: dict) -> tuple[list, dict]:
    """Blends each clip's measured score with its Decisions composite and re-orders the pool (reviews re-keyed)."""
    rows = []
    for i, c in enumerate(pool):
        rv = reviews.get(i)
        comp = ((rv or {}).get("decisions") or {}).get("composite")
        if comp is not None:
            blended = int(round((1 - BLEND) * c.score + BLEND * comp))
            c.breakdown = {**(c.breakdown or {}), "decisions": comp, "before_decisions": c.score}
            c.score = blended
        rows.append((c, rv))
    rows.sort(key=lambda x: -x[0].score)
    return [c for c, _ in rows], {k: rv for k, (_, rv) in enumerate(rows) if rv is not None}


def make_client() -> Optional[dc.Client]:
    return dc.Client() if dc.enabled() else None


def env_thresholds() -> None:
    """REPURPOSE_DECISIONS_THRESHOLDS='standalone=0.55,low_confidence=0.6' tunes thresholds without a deploy."""
    for part in os.environ.get("REPURPOSE_DECISIONS_THRESHOLDS", "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() in THRESHOLDS:
                try:
                    THRESHOLDS[k.strip()] = float(v)
                except ValueError:
                    pass


env_thresholds()
