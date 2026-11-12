"""Retention analysis for one clip: where a viewer is likely to lose
interest, measured from the words and their timing (no AI call).

For every sentence of the clip it measures:
  * pace       - words per second; a slow sentence drags (< 1.8 w/s over 3 s);
  * length     - a single sentence over 12 s is a long explanation;
  * repetition - a sentence whose content words were mostly already said
                 earlier in the clip (it restates a point);
  * restating  - "in other words", "what I mean is", "like I said"...;
and for the clip as a whole: how fast it gets to its first words, whether its
first line is a warm-up, whether it ends on a complete thought, and a 0-100
drop-off risk.

Edits are bounded and meaning-preserving:
  * APPLIED automatically: cutting a sentence that only restates what the
    clip already said (most of its content words were already heard, it adds
    no number, and the next sentence doesn't lean on it). At most two, at
    most a fifth of the clip, never the first or last sentence of a part.
  * SUGGESTED only (the creator decides): slow stretches, long explanations,
    a weak opening or ending. Changing those means changing what is said.
"""
from __future__ import annotations

import re

import clip_edit
import clip_text

SLOW_WPS, SLOW_MIN_SECONDS = 1.8, 3.0
LONG_SECONDS = 12.0
REPEAT_COVERAGE = 0.7
MAX_AUTO_CUTS, MAX_CUT_SHARE = 2, 0.2
_RESTATE = re.compile(r"^(in other words|what i mean is|what i'm saying is|basically what|to put it another way|"
                      r"like i said|as i said|again,? (the|it|this)|so again)\b", re.I)


def _content(text: str) -> set:
    return set(clip_text.content_tokens(text))


def analyze(ranges: list, sentences: list, words: list, clip_start: float) -> dict:
    """`ranges`: the clip's sentence ranges (inclusive, in order). Returns the report, including `cuts` (sentence
    ids safe to remove) and `suggestions` (for the creator)."""
    ids = [i for a, b in ranges for i in range(a, b + 1)]
    if not ids:
        return {"risk": None, "sentences": [], "cuts": [], "suggestions": []}
    edges = {a for a, _b in ranges} | {b for _a, b in ranges}
    total = sum(sentences[b].end - sentences[a].start for a, b in ranges)
    out_t, t = {}, 0.0
    for a, b in ranges:                                    # each sentence's time in the clip
        for i in range(a, b + 1):
            out_t[i] = round(t + sentences[i].start - sentences[a].start, 2)
        t += sentences[b].end - sentences[a].start
    heard: set = set()
    rows, cuts, cut_secs = [], [], 0.0
    for i in ids:
        s = sentences[i]
        dur = max(0.01, s.end - s.start)
        n = len(s.text.split())
        wps = n / dur
        toks = _content(s.text)
        coverage = len(toks & heard) / len(toks) if toks else 1.0
        flags = []
        if dur >= SLOW_MIN_SECONDS and wps < SLOW_WPS:
            flags.append("slow")
        if dur > LONG_SECONDS:
            flags.append("long_explanation")
        restates = bool(_RESTATE.match(s.text.strip()))
        repeats = bool(toks) and len(toks) >= 3 and coverage >= REPEAT_COVERAGE
        if restates:
            flags.append("restates")
        if repeats:
            flags.append("repeats")
        rows.append({"sentence": i, "at": out_t[i], "seconds": round(dur, 2), "words_per_second": round(wps, 2),
                     "flags": flags})
        # A restated point is cut only when it adds nothing new and the clip still reads on without it.
        nxt = i + 1 if i + 1 in out_t else None
        if (repeats or (restates and coverage >= 0.5)) and i not in edges and len(cuts) < MAX_AUTO_CUTS \
                and not re.search(r"\d", s.text) and not (nxt is not None and clip_edit.leans_back(sentences[nxt].text)) \
                and cut_secs + dur <= MAX_CUT_SHARE * total:
            cuts.append(i)
            cut_secs += dur
            rows[-1]["cut"] = True
        heard |= toks
    first = next((w for w in words if float(w["start"]) >= clip_start - 0.05), None)
    first_word_at = round(float(first["start"]) - clip_start, 2) if first else None
    warm_up = clip_edit.slow_opening(words, clip_start)
    last = sentences[ids[-1]]
    weak_end = not last.terminal or bool(clip_edit._SIGN_OFF.match(last.text.strip()))
    flagged = [r for r in rows if r["flags"] and not r.get("cut")]
    risk = min(100, int(round(
        25 * warm_up + 15 * weak_end + 100 * sum(r["seconds"] for r in flagged) / max(1.0, total) * 0.8)))
    suggestions = []
    if warm_up:
        suggestions.append({"kind": "opening", "at": 0.0, "text": "The first seconds are a warm-up. A preview of a "
                            "stronger line is used when one exists; otherwise consider starting later."})
    for r in flagged:
        if "slow" in r["flags"]:
            suggestions.append({"kind": "slow", "at": r["at"], "text": f"Slow stretch at {r['at']:.0f}s "
                                f"({r['words_per_second']} words/s). Viewers may scroll here."})
        elif "long_explanation" in r["flags"]:
            suggestions.append({"kind": "long_explanation", "at": r["at"], "text": f"A {r['seconds']:.0f}s "
                                f"explanation at {r['at']:.0f}s. Shorter clips usually keep more viewers."})
        elif "repeats" in r["flags"] or "restates" in r["flags"]:
            suggestions.append({"kind": "repeats", "at": r["at"], "text": f"At {r['at']:.0f}s the speaker repeats "
                                "an earlier point; it was kept because the clip needs it to make sense."})
    if weak_end:
        suggestions.append({"kind": "ending", "at": round(total, 1), "text": "The clip doesn't end on a complete "
                            "thought."})
    return {"risk": risk, "first_word_at": first_word_at, "warm_up_opening": warm_up, "weak_ending": weak_end,
            "sentences": rows, "cuts": cuts, "cut_seconds": round(cut_secs, 2), "suggestions": suggestions[:6]}


def cut_blocks(blocks: list, sentences: list, cut_ids: list) -> list:
    """Removes whole sentences (from just before their first word to just before the next one) from the blocks."""
    out = list(blocks)
    for i in sorted(cut_ids):
        a = sentences[i].start - 0.02
        b = (sentences[i + 1].start - 0.02) if i + 1 < len(sentences) else sentences[i].end + 0.1
        out = clip_edit._cut(out, a, b)
    return out
