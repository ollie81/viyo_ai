"""Story Intelligence stage 4: VIYO judges its own chapter before the creator sees it, and fixes only what is weak.

The review
  A demanding editor scores the chapter on nine areas: originality, character, emotional depth, plot (cause and
  effect, coherence with the plan), dialogue, literary quality, cinematic potential, the opening and the ending.
  Every score needs a reason AND a quote copied from the chapter, and the quote is looked up in the text
  (studio_story_check.find_quote): a score whose quote isn't there doesn't count (UNVERIFIED). The opening must be
  quoted from the first passage and the ending from the last. Dialogue isn't judged in a chapter with no dialogue.
  Weak areas say WHY they are weak and give a concrete fix, tied to a numbered passage.
  Status: "passed" (every area verified and at or above the bar), "needs_work" (complete, something below it), or
  "incomplete" (something missing or unverified: never a pass).

The revision
  Only a "needs_work" chapter is revised, at most once, and only the passages a verified weak area names are
  rewritten; every other passage is kept word for word, and an area that passed is never a target. Good writing is
  not rewritten to chase a number. The revised chapter is reviewed again and kept only if every targeted area
  improved, no other area dropped by more than one point, and the review is no less complete; otherwise the
  original is kept and the creator is told why.

Pure logic: passages, prompts, schemas, verification and the keep/discard rule. The calls run in
studio_worker.do_chapter.
"""
from __future__ import annotations

import re
from typing import Optional

AREAS = ("originality", "character", "emotion", "plot", "dialogue", "prose", "cinematic", "opening", "ending")
LABELS = {"originality": "Originality", "character": "Character development", "emotion": "Emotional depth",
          "plot": "Plot coherence", "dialogue": "Dialogue", "prose": "Literary quality",
          "cinematic": "Cinematic potential", "opening": "Opening", "ending": "Ending"}
BAR = 7
PASSAGE_WORDS = 150          # passages are paragraphs grouped to at least this many words
MAX_TARGET_PASSAGES = 4      # a revision touches at most this many passages


def passages(text: str) -> list[str]:
    """The chapter as numbered passages (paragraphs grouped to ~PASSAGE_WORDS words each)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text or "") if p.strip()]
    out, cur = [], []
    for p in paras:
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= PASSAGE_WORDS:
            out.append("\n\n".join(cur))
            cur = []
    if cur:
        if out and sum(len(x.split()) for x in cur) < PASSAGE_WORDS // 3:
            out[-1] = out[-1] + "\n\n" + "\n\n".join(cur)
        else:
            out.append("\n\n".join(cur))
    return out


def join(ps: list[str]) -> str:
    return "\n\n".join(ps)


def numbered(ps: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(ps, start=1))


def has_dialogue(text: str) -> bool:
    return len(re.findall(r"[\"“”«»]|—\s?[A-Z]", text or "")) >= 4


def required(text: str) -> list[str]:
    return [a for a in AREAS if a != "dialogue" or has_dialogue(text)]


_STR = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "object", "properties": {a: {"type": "object", "properties": {
            "score": {"type": "integer", "description": "1-10. 7 = ready for a published novel; 9 = exceptional."},
            "quote": {"type": "string", "description": "Words copied EXACTLY from the chapter (at least 4) that "
                                                       "show what you judged."},
            "passage": {"type": "integer", "description": "The number of the passage the quote is from."},
            "reason": _STR}, "required": ["score", "quote", "passage", "reason"]} for a in AREAS}},
        "weak": {"type": "array", "items": {"type": "object", "properties": {
            "area": {"type": "string", "enum": list(AREAS)},
            "passage": {"type": "integer", "description": "The passage with the problem (0 = the whole chapter)."},
            "why": {"type": "string", "description": "Why it is weak, concretely."},
            "fix": {"type": "string", "description": "A concrete instruction for rewriting that passage."}},
            "required": ["area", "passage", "why", "fix"]}},
        "strengths": {"type": "array", "items": _STR},
    },
    "required": ["scores", "weak"],
}

REVISION_SCHEMA = {
    "type": "object",
    "properties": {"passages": {"type": "array", "items": {"type": "object", "properties": {
        "number": {"type": "integer"}, "text": _STR}, "required": ["number", "text"]}}},
    "required": ["passages"],
}


def review_prompt(outline: dict, plan: dict, n: int, ps: list[str]) -> str:
    total = len(outline.get("chapters") or [])
    return "\n".join([
        f"You are a demanding literary editor. Judge chapter {n} of {total} of the novel "
        f"\"{outline.get('title') or ''}\" ({outline.get('genre') or ''}; {outline.get('tone') or ''}).",
        f"What the chapter was meant to do: {plan.get('purpose') or ''} — {plan.get('summary') or ''}",
        f"The feeling it should leave: {plan.get('emotion') or ''}",
        "",
        "Score each area from 1 to 10. Be honest: 7 means ready for a published novel, 9 is exceptional, and most "
        "first drafts are 5-7. Don't inflate.",
        "- originality: fresh images, situations and turns, not cliché;",
        "- character: people with inner lives who want things, choose and change;",
        "- emotion: feeling earned through moments, gestures and subtext, not announced;",
        "- plot: cause and effect, the chapter does what it was meant to, nothing contradicts itself;",
        "- dialogue: each voice distinct, subtext, no speeches that explain the plot;",
        "- prose: precise, varied, sensory, the point of view held;",
        "- cinematic: scenes that could be filmed: clear action, places, images;",
        "- opening: the first passage pulls the reader in (quote from passage 1);",
        f"- ending: the last passage pulls the reader on (quote from passage {len(ps)}).",
        "For every score give a reason and a quote copied EXACTLY from the chapter, with its passage number.",
        "Then list what is WEAK (only areas really below 7): say why it is weak and give a concrete fix for the "
        "passage that has the problem. Don't list anything that is already good. Never suggest changing the plot "
        "the chapter was meant to follow or any fact of the story.",
        "",
        "THE CHAPTER, IN NUMBERED PASSAGES:",
        numbered(ps),
    ])


def verify(raw, ps: list[str]) -> dict:
    """The editor's judgement, checked against the chapter: only quoted, found, correctly placed scores count."""
    from studio_story_check import find_quote
    raw = raw if isinstance(raw, dict) else {}
    got = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    text = join(ps)
    cats = {}
    for a in required(text):
        v = got.get(a)
        if not isinstance(v, dict) or not isinstance(v.get("score"), (int, float)) or isinstance(v.get("score"), bool):
            cats[a] = {"status": "missing", "why": "The review gave no score for this."}
            continue
        item = {"score": max(1, min(10, int(v["score"]))), "quote": str(v.get("quote") or "")[:300],
                "reason": str(v.get("reason") or "")[:600]}
        quote = item["quote"]
        where = next((i for i, p in enumerate(ps, start=1) if find_quote(quote, p) is not None), None)
        if where is None:
            cats[a] = {**item, "status": "unverified", "why": "Its quote isn't in the chapter."}
        elif a == "opening" and where != 1:
            cats[a] = {**item, "status": "unverified", "why": "The opening must be quoted from the first passage."}
        elif a == "ending" and where != len(ps):
            cats[a] = {**item, "status": "unverified", "why": "The ending must be quoted from the last passage."}
        else:
            cats[a] = {**item, "status": "verified", "passage": where}
    weak = []
    for w in raw.get("weak") or []:
        a = w.get("area")
        if a not in cats or cats[a].get("status") != "verified" or cats[a]["score"] >= BAR:
            continue                     # only areas that are really, verifiably below the bar
        p = int(w.get("passage") or 0)
        if a == "opening":
            p = 1
        elif a == "ending":
            p = len(ps)
        weak.append({"area": a, "passage": p if 0 <= p <= len(ps) else 0, "why": str(w.get("why") or "")[:600],
                     "fix": str(w.get("fix") or "")[:600]})
    unverified = [a for a, c in cats.items() if c["status"] != "verified"]
    below = [a for a, c in cats.items() if c["status"] == "verified" and c["score"] < BAR]
    status = "incomplete" if unverified else ("needs_work" if below else "passed")
    return {"status": status, "categories": cats, "unverified": unverified, "below_bar": below, "weak": weak,
            "strengths": [str(s)[:300] for s in raw.get("strengths") or []][:6], "summary": summary(status, cats, below, unverified)}


def summary(status: str, cats: dict, below: list, unverified: list) -> str:
    if status == "passed":
        return "VIYO's review passed every area, each backed by words from the chapter."
    if status == "needs_work":
        return "Needs work: " + ", ".join(LABELS[a].lower() for a in below) + "."
    return ("The review is incomplete (" + ", ".join(f"{LABELS[a].lower()}: {cats[a]['why'].rstrip('.').lower()}"
                                                     for a in unverified[:3]) + "), so it isn't a pass.")


def scores(ev: dict) -> dict:
    return {a: c["score"] for a, c in (ev.get("categories") or {}).items() if c.get("status") == "verified"}


_RANK = {"passed": 3, "needs_work": 2, "incomplete": 1}


def targets(ev: dict) -> list[int]:
    """The passages a revision may touch: those named by verified weak areas, at most MAX_TARGET_PASSAGES.
    A problem with the whole chapter (passage 0) can't be fixed by a passage edit and isn't targeted."""
    out = []
    for w in ev.get("weak") or []:
        if w["passage"] and w["passage"] not in out:
            out.append(w["passage"])
    return sorted(out[:MAX_TARGET_PASSAGES])


def revision_prompt(outline: dict, plan: dict, n: int, ps: list[str], ev: dict, memory: str = "") -> str:
    which = targets(ev)
    lines = [f"Revise chapter {n} of the novel \"{outline.get('title') or ''}\". Rewrite ONLY passages "
             f"{', '.join(map(str, which))}; every other passage stays exactly as it is.",
             "Fix only the problems named below. Keep every event, fact, name, relationship and the point of view; "
             "keep the passage about as long; keep what already works in it.",
             f"What the chapter must do: {plan.get('summary') or ''}", "", "PROBLEMS TO FIX:"]
    for w in ev.get("weak") or []:
        if w["passage"] in which:
            lines.append(f"- passage {w['passage']} ({LABELS[w['area']].lower()}): {w['why']} FIX: {w['fix']}")
    if memory:
        lines += ["", memory]
    lines += ["", "THE CHAPTER, IN NUMBERED PASSAGES:", numbered(ps), "",
              "Return the rewritten passages only, each with its number."]
    return "\n".join(lines)


def splice(ps: list[str], raw: dict, allowed: list[int]) -> Optional[list[str]]:
    """The chapter with only the allowed passages replaced; None if a replacement isn't usable (missing, cut short
    or ballooned)."""
    out = list(ps)
    got = {int(p.get("number") or 0): str(p.get("text") or "").strip() for p in (raw or {}).get("passages") or []}
    if not allowed or any(k not in got for k in allowed):
        return None
    for k in allowed:
        new, old = got[k], ps[k - 1]
        if not (0.5 * len(old) <= len(new) <= 2.5 * len(old)):
            return None
        out[k - 1] = re.sub(r"^\[\d+\]\s*", "", new)
    return out


def improved(before: dict, after: dict) -> tuple[bool, str]:
    """Keep the revision only if it is demonstrably better."""
    if _RANK.get(after.get("status"), 0) < _RANK.get(before.get("status"), 0):
        return False, "the new review was less complete"
    b, a = scores(before), scores(after)
    for k in set(b) - set(a):
        return False, f"{LABELS[k].lower()} could no longer be verified"
    for k in set(b) & set(a):
        if a[k] < b[k] - 1:
            return False, f"{LABELS[k].lower()} got worse ({b[k]} → {a[k]})"
    targeted = {w["area"] for w in before.get("weak") or [] if w["passage"] in targets(before)}
    for k in targeted:
        if k in a and k in b and a[k] <= b[k]:
            return False, f"{LABELS[k].lower()} didn't improve ({b[k]} → {a[k]})"
    if not targeted:
        return False, "nothing was targeted"
    return True, "every targeted area improved and nothing else got worse"


def report(first: dict, final: dict, revision: Optional[dict]) -> dict:
    """What the creator sees: the final review, and what happened to the revision."""
    return {**final, "first": {"status": first["status"], "scores": scores(first), "summary": first["summary"]},
            "revision": revision}

