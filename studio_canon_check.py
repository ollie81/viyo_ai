"""Story Intelligence stage 5: continuity problems, judged by VIYO and resolved by the creator.

Where problems come from
  The chapter job already finds two kinds, both with evidence, without any AI judgement (studio_canon):
    claim   something the chapter shows that the story memory says can't be true at that point (a dead character
            speaking, a secret known before it is learned, an object in two places): canon.check;
    fact    a fact the chapter establishes that contradicts an approved or creator-locked fact (Amani is 19 here,
            24 in the memory): canon.conflicts_for, found while the chapter is read back.
  This module turns both into one list of problems, each with a stable id, the chapter's words, and the memory's
  words.

VIYO's judgement (one AI call, only when there are problems)
  For each problem the model says whether it looks like a mistake ("error"), something the story does on purpose
  ("intended": a ghost, a lie, a dream not marked as one, a misdirection), or "unclear", and why, quoting the
  chapter. A judgement of "intended" whose quote isn't in the chapter is downgraded to "unclear": VIYO can't call
  something deliberate without showing where the chapter says so. The judgement is advice only: nothing in the
  memory changes until the creator decides.

What the creator can decide
  fix_chapter    the memory is right, the chapter is wrong: the chapter needs a new version (VIYO writes the note
                 for it); a "fact" problem's proposed fact is rejected so it never reaches the memory;
  chapter_right  the chapter is right, the memory is wrong: the memory's fact is retired (a creator-locked fact only
                 when the creator also confirms unlocking it); a "fact" problem's proposed fact is then approvable;
  intended       it's deliberate: an exception is recorded for that rule, that character and this chapter (or to the
                 end of the story), with the creator's reason, so it's never reported again there. Only for claim
                 problems: two facts that disagree can't both be canon.

A chapter can be approved once every problem is decided and none is waiting for a new version, or when the creator
explicitly approves it anyway (the problems stay on record).

Pure logic here; the AI call runs in studio_worker.do_chapter, the decisions in studio_novel_api.
"""
from __future__ import annotations

import hashlib
from typing import Optional

ACTIONS = ("fix_chapter", "chapter_right", "intended")
VERDICTS = ("error", "intended", "unclear")
CONTEXT_CHARS = 700          # how much of the chapter around a quote the judge sees


def _pid(*parts) -> str:
    return hashlib.sha1("|".join(str(p or "") for p in parts).encode()).hexdigest()[:12]


def problems_from(extraction: dict) -> list[dict]:
    """One list of continuity problems from a chapter's read-back (its issues and its conflicts with the memory)."""
    out = []
    for i in extraction.get("issues") or []:
        ev = i.get("evidence") or {}
        out.append({"id": _pid("claim", i.get("rule"), i.get("subject_id"), i.get("quote"), ev.get("id")),
                    "source": "claim", "rule": i.get("rule") or "", "subject": i.get("subject") or "",
                    "subject_id": i.get("subject_id"), "quote": i.get("quote") or "", "reason": i.get("reason") or "",
                    "memory_fact_id": ev.get("id"), "memory_statement": ev.get("statement") or "",
                    "memory_quote": ev.get("source_quote") or "", "memory_locked": ev.get("status") == "creator_locked"})
    for c in extraction.get("conflicts") or []:
        for w in c.get("with") or []:
            ev = w.get("fact") or {}
            out.append({"id": _pid("fact", c.get("fact"), ev.get("id")), "source": "fact", "rule": "conflicting_facts",
                        "subject": "", "subject_id": None, "quote": "", "reason":
                        f"\"{c.get('statement') or ''}\" contradicts the memory: {w.get('reason') or ''}.",
                        "proposed_fact_id": c.get("fact"), "proposed_statement": c.get("statement") or "",
                        "memory_fact_id": ev.get("id"), "memory_statement": ev.get("statement") or "",
                        "memory_quote": ev.get("source_quote") or "", "memory_locked": bool(w.get("blocking"))})
    seen, unique = set(), []
    for p in out:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)
    return unique


def context(text: str, quote: str, chars: int = CONTEXT_CHARS) -> str:
    """The part of the chapter around a quote (the whole start of the chapter if the quote isn't found)."""
    from studio_story_check import _norm, find_quote
    if not quote or find_quote(quote, text) is None:
        return text[:chars]
    low = text.lower()
    at = low.find(quote.lower()[:40])
    if at < 0:                                     # the quote matched after normalising: fall back to its position
        at = int(find_quote(quote, text) / max(1, len(_norm(text))) * len(text))
    start = max(0, at - chars // 2)
    return ("…" if start else "") + text[start:start + chars] + ("…" if start + chars < len(text) else "")


_STR = {"type": "string"}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"judgements": {"type": "array", "items": {"type": "object", "properties": {
        "id": _STR, "verdict": _STR, "intent": _STR, "why": _STR, "quote": _STR, "fix": _STR},
        "required": ["id", "verdict", "why"]}}},
    "required": ["judgements"],
}


def judge_prompt(n: int, text: str, problems: list[dict], memory: str = "") -> str:
    lines = [
        f"You check continuity in chapter {n} of a novel. The story memory (what earlier chapters established) "
        "flagged the problems below. For EACH one decide:",
        "- verdict: \"error\" (a continuity mistake the writer should fix), \"intended\" (the chapter does it on "
        "purpose and makes that clear: a ghost or vision, a lie, a dream, a flashback, a misdirection the reader is "
        "meant to notice), or \"unclear\".",
        "- intent: for \"intended\", which kind (ghost, vision, lie, dream, flashback, misdirection, other).",
        "- why: one or two plain sentences.",
        "- quote: words copied EXACTLY from the chapter (at least 4 words) that support your verdict. For "
        "\"intended\" this is required: the words that show it's deliberate.",
        "- fix: for \"error\", one sentence saying what the next version of the chapter should change.",
        "Judge only from the chapter and the memory shown. If the chapter doesn't make the intention clear to a "
        "reader, it's an error or unclear, not intended.",
    ]
    if memory:
        lines += ["", memory[:3000]]
    for p in problems:
        lines += ["", f"PROBLEM {p['id']}: {p['reason']}"]
        if p.get("memory_statement"):
            lines.append(f"The memory says: {p['memory_statement']}"
                         + (f" (from: \"{p['memory_quote']}\")" if p.get("memory_quote") else ""))
        if p.get("proposed_statement"):
            lines.append(f"This chapter establishes: {p['proposed_statement']}")
        if p.get("quote"):
            lines.append(f"The chapter's words: \"{p['quote']}\"")
        lines.append("Around it in the chapter: " + context(text, p.get("quote") or p.get("proposed_statement") or ""))
    return "\n".join(lines)


def apply_judgements(problems: list[dict], raw: dict, text: str) -> list[dict]:
    """Attaches VIYO's judgement to each problem. An "intended" without a quote found in the chapter becomes
    "unclear"; an unknown verdict becomes "unclear"; a problem the model skipped gets no judgement."""
    from studio_story_check import find_quote
    by_id = {str(j.get("id")): j for j in (raw or {}).get("judgements") or [] if isinstance(j, dict)}
    out = []
    for p in problems:
        j = by_id.get(p["id"])
        if not j:
            out.append(p)
            continue
        verdict = str(j.get("verdict") or "").lower()
        quote = str(j.get("quote") or "").strip()
        found = bool(quote) and find_quote(quote, text) is not None
        note = ""
        if verdict not in VERDICTS:
            verdict, note = "unclear", "VIYO's answer wasn't one of error / intended / unclear."
        elif verdict == "intended" and not found:
            verdict, note = "unclear", "VIYO thought it might be deliberate but couldn't show where the chapter says so."
        out.append({**p, "judgement": {"verdict": verdict, "intent": str(j.get("intent") or "")[:40] if verdict == "intended" else "",
                                       "why": str(j.get("why") or "")[:500], "quote": quote[:400] if found else "",
                                       "fix": str(j.get("fix") or "")[:400] if verdict == "error" else "",
                                       **({"note": note} if note else {})}})
    return out


def carry_decisions(new: list[dict], old: list[dict]) -> list[dict]:
    """Keeps decisions already made when the problem list is rebuilt for the same chapter version."""
    done = {p["id"]: p.get("resolution") for p in old or [] if p.get("resolution")}
    return [{**p, "resolution": done[p["id"]]} if p["id"] in done else p for p in new]


def open_problems(problems: list[dict]) -> list[dict]:
    """Problems that stop approval: undecided, or decided as "fix the chapter" (it needs a new version)."""
    return [p for p in problems or [] if not p.get("resolution") or p["resolution"].get("action") == "fix_chapter"]


def allowed_actions(p: dict) -> list[str]:
    acts = ["fix_chapter"]
    if p.get("memory_fact_id"):
        acts.append("chapter_right")
    if p.get("source") == "claim" and p.get("rule") in _EXCEPTABLE:
        acts.append("intended")
    return acts


_EXCEPTABLE = ("dead_character_active", "knowledge_leak", "object_two_places", "object_location_conflict",
               "character_two_places")


def fix_note(problems: list[dict]) -> str:
    """The note for the chapter's next version: every problem the creator wants fixed, in plain words."""
    items = [p for p in problems or [] if (p.get("resolution") or {}).get("action") == "fix_chapter"]
    if not items:
        return ""
    lines = ["Fix these continuity problems (the story memory is right):"]
    for p in items:
        fix = (p.get("judgement") or {}).get("fix") or ""
        lines.append(f"- {p['reason']}" + (f" {fix}" if fix else "")
                     + (f" Keep: {p['memory_statement']}" if p.get("memory_statement") else ""))
    return "\n".join(lines)[:1500]


def summary(problems: list[dict]) -> dict:
    v = [((p.get("judgement") or {}).get("verdict")) for p in problems or []]
    return {"total": len(problems or []), "open": len(open_problems(problems)), "errors": v.count("error"),
            "intended": v.count("intended"), "unclear": v.count("unclear"),
            "decided": len([p for p in problems or [] if p.get("resolution")])}


def find(problems: list[dict], pid: str) -> Optional[dict]:
    return next((p for p in problems or [] if p["id"] == pid), None)
