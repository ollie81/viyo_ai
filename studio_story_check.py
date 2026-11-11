"""Strict story check of an episode's FINAL screenplay.

A model judges the screenplay, but its judgement only counts when it can be
verified against the actual text:

- Every category needs a score, a reason and a QUOTE copied from the screenplay.
  The quote is searched for in the screenplay (ignoring case, punctuation and
  spacing; "..." may skip words). A quote that isn't there is invented: the
  category is UNVERIFIED and its score is not used.
- The quote must come from the right place: the opening hook from the first
  scene, the ending hook from the last scene, continuity from the first scene,
  dialogue from a spoken line.
- Every required category must be verified (continuity only from episode 2 on).
  Anything missing or unverified makes the check INCOMPLETE; it never passes.
- Checks that need no AI (studio_story.structure_checks) count too.

Status: "passed" (complete, every category at or above the bar, no failed
checks), "needs_work" (complete, something below the bar), "incomplete".

A weak episode gets at most one targeted revision: only the scenes with the
problem are rewritten (the rest is kept word for word), the result is checked
again, and it is kept only if every targeted area improved, nothing else got
worse and the check is no less complete.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from typing import Optional

CRITERIA = ("opening_hook", "progression", "emotion", "suspense", "dialogue", "scene_relevance", "ending_hook", "continuity")
LABELS = {"opening_hook": "Opening hook", "progression": "Story progression", "emotion": "Emotional engagement",
          "suspense": "Suspense", "dialogue": "Dialogue", "scene_relevance": "Every scene matters",
          "ending_hook": "Ending cliffhanger", "continuity": "Continues the last episode"}
THRESHOLD = {"opening_hook": 8, "ending_hook": 8, "continuity": 8}      # the rest: 7
WHERE = {"opening_hook": "first", "ending_hook": "last", "continuity": "first"}
MIN_QUOTE_WORDS = 3

_HEAD = re.compile(r"\n(?=[ \t]*(?:\d+\s*[.)]?\s*)?(?:INT\.|EXT\.|INT/EXT|EXT/INT|I/E)(?:\s|$))", re.I)

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "object", "properties": {k: {"type": "object", "properties": {
            "score": {"type": "integer", "description": "1-10."},
            "quote": {"type": "string", "description": "Words copied EXACTLY from the screenplay (at least 4 words) "
                                                       "that show what you judged. Never paraphrase."},
            "scene": {"type": "integer", "description": "The number of the scene the quote is from."},
            "reason": {"type": "string"}}, "required": ["score", "quote", "scene", "reason"]} for k in CRITERIA}},
        "weak": {"type": "array", "items": {"type": "object", "properties": {
            "area": {"type": "string", "enum": list(CRITERIA)},
            "scene": {"type": "integer", "description": "The scene with the problem (0 if it's the whole episode)."},
            "problem": {"type": "string"}, "fix": {"type": "string", "description": "A concrete rewrite instruction."}},
            "required": ["area", "scene", "problem", "fix"]}},
        "viewer_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scores", "weak", "viewer_questions"],
}
REVISION_SCHEMA = {
    "type": "object",
    "properties": {"scenes": {"type": "array", "items": {"type": "object", "properties": {
        "number": {"type": "integer"}, "text": {"type": "string", "description": "The whole rewritten scene, heading included."}},
        "required": ["number", "text"]}}},
    "required": ["scenes"],
}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sections(script: str) -> list[str]:
    """The screenplay's scenes (split at scene headings); text before the first heading joins scene 1."""
    parts = [p for p in _HEAD.split("\n" + (script or "").strip()) if p.strip()]
    if len(parts) > 1 and not re.match(r"\s*(?:\d+\s*[.)]?\s*)?(INT\.|EXT\.|INT/EXT|EXT/INT|I/E)", parts[0], re.I):
        parts = [parts[0] + "\n" + parts[1]] + parts[2:]
    return [p.strip("\n") for p in parts] or [(script or "").strip()]


def script_sha(script: str) -> str:
    return hashlib.sha1(" ".join((script or "").split()).encode()).hexdigest()[:16]


def _norm(text: str) -> str:
    t = (text or "").lower().replace("’", "'").replace("‘", "'").replace("—", " ").replace("–", " ")
    return " ".join(re.findall(r"[a-z0-9']+", t))


def find_quote(quote: str, script: str) -> Optional[int]:
    """Where the quote is in the screenplay (character offset in the normalised
    text), or None. '...' may skip words; each piece must appear in order."""
    pieces = [_norm(p) for p in re.split(r"\.\.\.|…", quote or "")]
    pieces = [p for p in pieces if p]
    if sum(len(p.split()) for p in pieces) < MIN_QUOTE_WORDS:
        return None
    hay, start, first = " " + _norm(script) + " ", 0, None
    for p in pieces:
        at = hay.find(" " + p + " ", start)
        if at < 0:
            return None
        first = at if first is None else first
        start = at + len(p) + 1
    return first


def _section_index(script: str, pos: int) -> int:
    bounds, offset = [], 0
    for s in sections(script):
        offset += len(" " + _norm(s))
        bounds.append(offset)
    return next((i for i, b in enumerate(bounds) if pos < b), len(bounds) - 1)


def _spoken(quote: str, script: str) -> bool:
    import studio_story
    q = _norm(re.split(r"\.\.\.|…", quote or "")[0])
    return any(q and q in _norm(text) for _who, text in studio_story.screenplay_lines(script))


def required(has_previous: bool) -> list[str]:
    return [k for k in CRITERIA if k != "continuity" or has_previous]


def verify(raw, script: str, has_previous: bool, checks: Optional[list] = None) -> dict:
    """The model's judgement, verified against the screenplay."""
    raw = raw if isinstance(raw, dict) else {}
    got = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    secs = sections(script)
    last = len(secs) - 1
    cats = {}
    for k in required(has_previous):
        v = got.get(k)
        if not isinstance(v, dict) or not isinstance(v.get("score"), (int, float)) or isinstance(v.get("score"), bool):
            cats[k] = {"status": "missing", "why": "The check gave no score for this."}
            continue
        item = {"score": max(1, min(10, int(v["score"]))), "quote": str(v.get("quote") or "")[:300],
                "reason": str(v.get("reason") or "")[:300]}
        pos = find_quote(item["quote"], script)
        if not item["quote"].strip():
            item.update(status="unverified", why="No quote from the screenplay was given.")
        elif pos is None:
            item.update(status="unverified", why="The quoted words aren't in the screenplay (invented or paraphrased).")
        else:
            n = _section_index(script, pos)
            item["scene"] = n + 1
            where = WHERE.get(k)
            if where == "first" and n != 0:
                item.update(status="unverified", why=f"The quote is from scene {n + 1}, not the first scene.")
            elif where == "last" and n != last:
                item.update(status="unverified", why=f"The quote is from scene {n + 1}, not the last scene.")
            elif k == "dialogue" and not _spoken(item["quote"], script):
                item.update(status="unverified", why="The quote isn't a line anyone says.")
            elif not item["reason"].strip():
                item.update(status="unverified", why="No reason was given.")
            else:
                item["status"] = "verified"
        cats[k] = item
    weak = []
    for w in raw.get("weak") or []:
        if isinstance(w, dict) and w.get("area") in CRITERIA:
            sc = w.get("scene") if isinstance(w.get("scene"), int) and 1 <= w.get("scene") <= len(secs) else 0
            weak.append({"area": w["area"], "scene": sc, "problem": str(w.get("problem") or "")[:300],
                         "fix": str(w.get("fix") or "")[:300]})
    checks = list(checks or [])
    unverified = [k for k, c in cats.items() if c["status"] != "verified"]
    below = sorted({k for k, c in cats.items() if c["status"] == "verified" and c["score"] < THRESHOLD.get(k, 7)} |
                   {c["area"] for c in checks})
    status = "incomplete" if unverified else "needs_work" if below else "passed"
    return {"status": status, "complete": not unverified, "categories": cats, "unverified": unverified,
            "below_bar": below, "weak": weak, "checks": checks, "scenes": len(secs),
            "viewer_questions": [str(q)[:200] for q in (raw.get("viewer_questions") or [])[:6] if str(q).strip()],
            "script_sha": script_sha(script), "checked_at": _now(),
            "summary": summary(status, cats, unverified, below)}


def summary(status: str, cats: dict, unverified: list, below: list) -> str:
    if status == "passed":
        return "Every part of the story check passed, each backed by a line from the script."
    if status == "needs_work":
        return "Needs work: " + ", ".join(LABELS.get(k, k).lower() for k in below) + "."
    return ("Incomplete: " + ", ".join(f"{LABELS.get(k, k).lower()} ({cats[k]['why'].rstrip('.').lower()})"
                                       for k in unverified[:4]) + ". It isn't a pass.")


def scores(ev: dict) -> dict:
    """Verified scores only."""
    return {k: c["score"] for k, c in (ev.get("categories") or {}).items() if c.get("status") == "verified"}


_RANK = {"passed": 3, "needs_work": 2, "incomplete": 1}


def improved(before: dict, after: dict, targets: list[str]) -> tuple[bool, str]:
    """Is the revision demonstrably better?"""
    if _RANK.get(after.get("status"), 0) < _RANK.get(before.get("status"), 0):
        return False, "the new check was less complete"
    b, a = scores(before), scores(after)
    for k in set(b) & set(a):
        if a[k] < b[k] - 1:
            return False, f"{LABELS.get(k, k).lower()} got worse ({b[k]} -> {a[k]})"
    for k in set(b) - set(a):
        return False, f"{LABELS.get(k, k).lower()} could no longer be verified"
    fixed = [k for k in targets if (k in a and k in b and a[k] > b[k]) or
             (k in {c["area"] for c in before.get("checks") or []} and k not in {c["area"] for c in after.get("checks") or []})]
    if not fixed:
        return False, "none of the weak areas improved"
    return True, ", ".join(LABELS.get(k, k).lower() for k in fixed) + " improved"


def evaluation_prompt(brief: dict, plan: dict, script: str, continuity: str, retry: bool = False) -> str:
    hp = plan.get("hook_plan") or {}
    numbered = "\n\n".join(f"[SCENE {i}]\n{s}" for i, s in enumerate(sections(script), start=1))
    return f"""You are the head of story at a short-drama studio deciding whether this episode is ready for
viewers. Judge the FINAL screenplay below, exactly as written. A plan or an intention earns nothing: only what the
scenes actually show and say counts.

{continuity}WHAT THE WRITERS PLANNED (check whether the screenplay really delivers it):
- Opening ({hp.get('opening_technique', '')}): {hp.get('opening_moment', '')}
- Ending ({hp.get('ending_technique', '')}): {hp.get('ending_moment', '')}
Format: {brief.get('kind')} of about {int(brief.get('target_seconds') or 90)} seconds; genre {brief.get('genre') or 'drama'}.

SCREENPLAY (scenes numbered for you; the numbers are not part of the script):
{numbered}

For every category give a score 1-10, a reason, the scene number, and a QUOTE: words copied EXACTLY from the
screenplay, at least 4 words, never paraphrased. Quotes are checked against the script; an invented quote fails.
- opening_hook: quote SCENE 1. Does the first scene itself (not the plan) make leaving impossible in 3-5 seconds?
- progression: does every scene change the situation?
- emotion: do we feel for someone? Quote the moment.
- suspense: do questions build and get answered in a good order?
- dialogue: quote a spoken line. Natural, with subtext, not repeated?
- scene_relevance: is any scene filler? Name it in weak if so.
- ending_hook: quote the LAST scene. Does the final moment leave a question the viewer needs answered, grown from the story?
{'- continuity: quote SCENE 1. Does it pick up the previous episode' + "'" + 's actual ending, keep who knows what, and not re-explain?' if continuity else '- continuity: leave it out (first episode).'}
A 9-10 means a viewer could not stop watching; 5 means forgettable. List each weak area with its scene number and a
concrete fix, and the viewer's questions after the last moment.{chr(10) + 'YOUR LAST ANSWER COULD NOT BE VERIFIED: copy the quotes letter for letter from the screenplay.' if retry else ''}"""


def revision_prompt(script: str, weak: list[dict], numbers: list[int], continuity: str, hook_rule: str) -> str:
    secs = sections(script)
    numbered = "\n\n".join(f"[SCENE {i}]\n{s}" for i, s in enumerate(secs, start=1))
    fixes = "\n".join(f"- scene {w['scene'] or 'all'} ({w['area']}): {w['problem']} FIX: {w['fix']}" for w in weak[:8])
    return f"""Rewrite ONLY scene(s) {', '.join(map(str, numbers))} of this screenplay to fix these problems:
{fixes}

Keep the same characters, names, events that later scenes depend on, language and length. Don't contradict any other
scene. Every other scene stays exactly as it is.

{continuity}{hook_rule}

SCREENPLAY:
{numbered}

Return each rewritten scene in full, heading included, with its scene number."""


def splice(script: str, replacements: dict) -> Optional[str]:
    """The screenplay with only the given scenes replaced (by number); None if a
    replacement isn't a proper scene."""
    secs = sections(script)
    for n, text in replacements.items():
        t = (text or "").strip()
        if not (1 <= n <= len(secs)) or not re.match(r"\s*(INT\.|EXT\.|INT/EXT|EXT/INT|I/E)", t, re.I) or len(t) < 0.4 * len(secs[n - 1]):
            return None
        secs[n - 1] = t
    return "\n\n".join(secs) + "\n"


def target_scenes(script: str, weak: list[dict]) -> list[int]:
    """Which scenes a revision may touch: the opening/ending/continuity scene,
    or the scenes the check named. Empty when the problem is the whole episode."""
    last = len(sections(script))
    out = set()
    for w in weak:
        where = WHERE.get(w["area"])
        if where == "first":
            out.add(1)
        elif where == "last":
            out.add(last)
        elif w.get("scene"):
            out.add(w["scene"])
        else:
            return []
    return sorted(out)


def check(j, brief: dict, plan: dict, script: str, continuity: str, has_previous: bool, structure: list) -> dict:
    """One verified check; asked again once if the quotes couldn't be verified."""
    ev = verify(j(evaluation_prompt(brief, plan, script, continuity), SCHEMA), script, has_previous, structure)
    if ev["status"] == "incomplete":
        again = verify(j(evaluation_prompt(brief, plan, script, continuity, retry=True), SCHEMA), script, has_previous,
                       structure)
        again["retried"] = True
        if len(again["unverified"]) <= len(ev["unverified"]):
            ev = again
    return ev


def improve(j, brief: dict, plan: dict, script: str, ev: dict, continuity: str, has_previous: bool,
            structure_fn, hook_rule: str) -> tuple[str, dict]:
    """At most one targeted revision of a weak (complete) episode."""
    if ev["status"] != "needs_work":
        return script, ev
    targets = ev["below_bar"]
    weak = [w for w in ev["weak"] if w["area"] in targets] + \
        [{"area": c["area"], "scene": 1 if WHERE.get(c["area"]) == "first" else len(sections(script))
          if WHERE.get(c["area"]) == "last" else 0, "problem": c["problem"], "fix": c["fix"]} for c in ev["checks"]]
    for k in targets:
        if not any(w["area"] == k for w in weak) and k in ev["categories"]:
            weak.append({"area": k, "scene": ev["categories"][k].get("scene") or 0,
                         "problem": ev["categories"][k].get("reason") or "below the bar",
                         "fix": f"Make the {LABELS.get(k, k).lower()} stronger."})
    numbers = target_scenes(script, weak) or list(range(1, len(sections(script)) + 1))
    record = {"attempted": True, "areas": targets, "scenes": numbers, "whole": numbers == list(range(1, len(sections(script)) + 1))}
    try:
        out = j(revision_prompt(script, weak, numbers, continuity, hook_rule), REVISION_SCHEMA)
        reps = {int(s["number"]): s["text"] for s in (out or {}).get("scenes") or []
                if isinstance(s, dict) and isinstance(s.get("number"), int) and s["number"] in numbers}
        new = splice(script, reps) if reps else None
        if not new:
            return script, {**ev, "revision": {**record, "kept": False, "why": "The rewrite didn't return proper scenes."}}
        ev2 = verify(j(evaluation_prompt(brief, plan, new, continuity), SCHEMA), new, has_previous, structure_fn(new))
        ok, why = improved(ev, ev2, targets)
        if not ok:
            return script, {**ev, "revision": {**record, "kept": False, "why": f"Not kept: {why}."}}
        return new, {**ev2, "revision": {**record, "kept": True, "why": why, "rewrote": sorted(reps),
                                         "before": scores(ev)}}
    except Exception as e:
        return script, {**ev, "revision": {**record, "kept": False, "why": f"The rewrite failed ({str(e)[:120]})."}}


def outdated(ev: dict, script: str) -> bool:
    return bool(ev.get("script_sha")) and ev["script_sha"] != script_sha(script)
