"""
A real on-screen hook for every clip, without inventing anything.

Selection asks for an overlay hook along with each moment, but that first
suggestion is often missing or fails the grounding check
(clip_text.vet_overlay_hook), which left many clips with no hook at all.
For each clip still without one:

  1. one batched LLM call asks for three short hook options built ONLY from
     what the speaker says in that clip; each is vetted the same way and the
     first that passes is used;
  2. otherwise the strongest short line the speaker actually says later in
     the clip is used verbatim (a quote can't misrepresent the clip).

A clip only stays hook-less when it has no usable line at all.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import clip_selection as cs
import clip_text

MAX_WORDS = clip_text.MAX_OVERLAY_WORDS
MIN_WORDS = 3
OPTIONS = 3

PROMPT = """Write on-screen hooks for short vertical videos. A hook is the line of text shown across the
top for the first 2-3 seconds that makes a scroller stop and watch.

For EACH clip below, write {n} different hook options. Rules for every option:
- at most {max_words} words;
- use ONLY facts, claims and numbers the speaker actually says in that clip's transcript; you may
  rephrase, tease or quote, but never add anything new and never put words in their mouth;
- don't just repeat the clip's first sentence;
- specific beats generic ("Fresh breath changes everything" beats "Kissing tips").

Return ONLY JSON: {{"hooks": [{{"id": int, "options": [str, ...]}}]}}

CLIPS
{clips}
"""


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def quote_hook(transcript: str, opening_words: list[str]) -> Optional[str]:
    """The punchiest short line the speaker says, verbatim — never the first one."""
    sents = _sentences(transcript)[1:]
    best, best_score = None, -1.0
    for s in sents:
        words = s.split()
        # A long sentence can still give a hook: its first clause, if that stands up.
        if len(words) > MAX_WORDS:
            clause = re.split(r"[,;:—–]\s*", s)[0].split()
            if not (MIN_WORDS <= len(clause) <= MAX_WORDS):
                continue
            words = clause
        if len(words) < MIN_WORDS:
            continue
        line = " ".join(words).rstrip(",;:")
        if clip_text.opens_as_reply(line):
            continue
        toks = [clip_text._norm(w) for w in words]
        score = (2.0 * sum(t in clip_text.EMPHASIS_STRONG for t in toks)
                 + 1.0 * sum(t in ("you", "your") for t in toks)
                 + 1.5 * any(re.search(r"\d", t) for t in toks)
                 + 1.0 * line.endswith(("!", "?"))
                 - 0.15 * len(words))
        text, why = clip_text.vet_overlay_hook(line, transcript, opening_words)
        if text and score > best_score:
            best, best_score = text, score
    return best


def _ask(cands: list, llm: Callable) -> dict:
    blocks = "\n".join(f"--- CLIP {i} (title: {c.title})\nTRANSCRIPT: {c.transcript}" for i, c in cands)
    try:
        data = cs.parse_json_object(llm([{"role": "user", "content": PROMPT.format(
            n=OPTIONS, max_words=MAX_WORDS, clips=blocks)}]))
    except Exception as e:
        print(f"[WARN] hook writing unavailable: {e}")
        return {}
    out = {}
    for item in (data.get("hooks") if isinstance(data, dict) else None) or []:
        try:
            out[int(item["id"])] = [str(o) for o in (item.get("options") or [])][:OPTIONS + 2]
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fill_missing_hooks(clips: list, llm: Optional[Callable]) -> dict:
    """Gives every clip without a vetted overlay hook one, if a faithful one exists."""
    missing = [(i, c) for i, c in enumerate(clips) if not c.overlay_hook]
    stats = {"already_had": len(clips) - len(missing), "written": 0, "quoted": 0, "none": 0}
    if not missing:
        return stats
    options = _ask(missing, llm) if llm is not None else {}
    for i, c in missing:
        opening = c.transcript.split()[:14]
        chosen = None
        for opt in options.get(i, []):
            text, _ = clip_text.vet_overlay_hook(opt, c.transcript, opening)
            if text:
                chosen = text
                break
        if chosen:
            c.overlay_hook, c.overlay_status = chosen, "written"
            stats["written"] += 1
            continue
        q = quote_hook(c.transcript, opening)
        if q:
            c.overlay_hook, c.overlay_status = q, "quoted"
            stats["quoted"] += 1
        else:
            stats["none"] += 1
    return stats
