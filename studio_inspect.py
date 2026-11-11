"""Visual continuity inspection: a vision model looks at a scene picture and
compares it with what it should show.

It gets the scene picture, the characters' approved reference pictures, the
previous scene's picture (for a continuous scene), and the place/object
pictures, plus the scene text, the continuity record (who holds what, wears
what, where they stand, the light and time) and what the previous picture was
seen to show. It returns scores, problems (each with severity, certainty and
whether the screenplay means the change), recommended fixes, and what it
actually sees (the observed state the next scene inherits).

A vision model can be wrong. Uncertain observations are marked uncertain, and
nothing is changed without the creator: a fix makes a new picture of that one
scene only after they confirm the price.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from typing import Optional

KINDS = ("character", "outfit", "location", "prop", "position", "lighting", "transition", "story", "camera")
SEVERITIES = ("high", "medium", "low")
CERTAINTY = ("certain", "likely", "uncertain")
SCORES = ("overall", "character", "outfit", "location", "props", "transition", "lighting", "camera")
ATTENTION_SCORE = 70
MAX_IMAGES = 6

_S = {"type": "string"}
_SCORE = {"type": "integer", "description": "0-100; omit when it doesn't apply (e.g. no previous scene)."}
SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "object", "properties": {k: _SCORE for k in SCORES}, "required": ["overall"]},
        "problems": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "severity": {"type": "string", "enum": list(SEVERITIES)},
            "certainty": {"type": "string", "enum": list(CERTAINTY),
                          "description": "uncertain when the picture doesn't show it clearly."},
            "characters": {"type": "array", "items": _S},
            "expected": {"type": "string", "description": "What it should be (from the references, record or text)."},
            "seen": {"type": "string", "description": "What the picture shows."},
            "intentional": {"type": "boolean", "description": "true if the screenplay/record means this change."},
            "explanation": {"type": "string", "description": "In plain words, why it matters."},
            "fix": {"type": "string", "description": "What a new picture should change, as a short instruction."},
        }, "required": ["kind", "severity", "certainty", "expected", "seen", "intentional", "fix"]}},
        "observed": {"type": "object", "description": "What the picture actually shows.", "properties": {
            "characters": {"type": "array", "items": {"type": "object", "properties": {
                "name": _S, "outfit": _S, "position": _S, "facing": _S, "expression": _S,
                "holding": {"type": "array", "items": _S}, "visible_state": _S}}},
            "objects": {"type": "array", "items": {"type": "object", "properties": {"name": _S, "where": _S, "held_by": _S}}},
            "location": _S, "lighting": _S, "weather": _S, "time_of_day": _S}},
        "summary": {"type": "string", "description": "One or two plain sentences for the creator."},
    },
    "required": ["scores", "problems", "summary"],
}


def _t(x, n=300) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()[:n]


def clean(raw) -> dict:
    """Valid fields only; scores clamped to 0-100; unknown values dropped."""
    raw = raw if isinstance(raw, dict) else {}
    scores = {}
    for k in SCORES:
        v = (raw.get("scores") or {}).get(k) if isinstance(raw.get("scores"), dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            scores[k] = max(0, min(100, int(round(v))))
    problems = []
    for p in (raw.get("problems") or [])[:12]:
        if not isinstance(p, dict) or p.get("kind") not in KINDS:
            continue
        problems.append({
            "kind": p["kind"],
            "severity": p.get("severity") if p.get("severity") in SEVERITIES else "medium",
            "certainty": p.get("certainty") if p.get("certainty") in CERTAINTY else "uncertain",
            "characters": [_t(c, 60) for c in (p.get("characters") or [])[:4] if _t(c, 60)],
            "expected": _t(p.get("expected")), "seen": _t(p.get("seen")),
            "intentional": bool(p.get("intentional")),
            "explanation": _t(p.get("explanation")), "fix": _t(p.get("fix"), 400)})
    obs = raw.get("observed") if isinstance(raw.get("observed"), dict) else {}
    observed = {
        "characters": [{k: (_t(c.get(k), 120) if k != "holding" else [_t(h, 60) for h in (c.get("holding") or [])[:4]])
                        for k in ("name", "outfit", "position", "facing", "expression", "holding", "visible_state")
                        if c.get(k)} for c in (obs.get("characters") or [])[:8] if isinstance(c, dict) and c.get("name")],
        "objects": [{k: _t(o.get(k), 120) for k in ("name", "where", "held_by") if o.get(k)}
                    for o in (obs.get("objects") or [])[:10] if isinstance(o, dict) and o.get("name")],
        **{k: _t(obs.get(k), 160) for k in ("location", "lighting", "weather", "time_of_day") if obs.get(k)}}
    if "overall" not in scores:
        raise ValueError("no overall score")
    return {"scores": scores, "problems": problems, "observed": observed, "summary": _t(raw.get("summary"), 400)}


def errors(result: dict) -> list[dict]:
    """Problems that are mistakes (not changes the screenplay means)."""
    return [p for p in result.get("problems") or [] if not p.get("intentional")]


def needs_attention(result: dict) -> bool:
    if not result or "scores" not in result:
        return False
    serious = [p for p in errors(result) if p["severity"] == "high" and p["certainty"] != "uncertain"]
    return bool(serious) or result["scores"].get("overall", 100) < ATTENTION_SCORE


def correction_text(result: dict) -> str:
    """The fixes for a new picture, from the mistakes found (uncertain ones as 'check')."""
    out = []
    for p in errors(result)[:6]:
        if p.get("fix"):
            out.append(("Check: " if p["certainty"] == "uncertain" else "") + p["fix"])
    return " ".join(out)[:600]


def signature(scene: dict, prev: Optional[dict], outfit_ids: list, identity_versions: list) -> str:
    """Changes when anything the check depends on changes: the picture, the
    previous picture, the scene text and record, outfits, identities."""
    key = json.dumps([scene.get("image_path"), (prev or {}).get("image_path") if prev else None, scene.get("visual"),
                      scene.get("heading"), scene.get("continuity") or {}, sorted(map(str, outfit_ids)),
                      sorted(map(str, identity_versions))], sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def record(result: dict, sig: str, image_path: str, cost_usd: float) -> dict:
    return {"result": result, "signature": sig, "image_path": image_path, "cost_usd": round(cost_usd, 5),
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def current(scene: dict, sig: str) -> Optional[dict]:
    """The saved check, if it is still valid for this picture."""
    ins = scene.get("inspection") or {}
    if ins.get("signature") == sig and ins.get("image_path") == scene.get("image_path") and ins.get("result"):
        return ins
    return None


def prompt(scene: dict, prev: Optional[dict], cast_lines: list[str], world: str, labels: list[str],
           prev_observed: Optional[dict], transition: str) -> str:
    """The text for the vision model; images are sent before it, labelled."""
    c = scene.get("continuity") or {}
    pc = (prev or {}).get("continuity") or {}
    roles = "\n".join(f"Image {i}: {label}" for i, label in enumerate(labels, start=1))
    prev_part = ""
    if prev is not None:
        seen = json.dumps(prev_observed, ensure_ascii=False)[:1200] if prev_observed else "(not checked)"
        prev_part = (f"\nPREVIOUS SCENE ({transition.replace('_', ' ')} into this one):\n"
                     f"- It ended with: {pc.get('end_state') or '(not recorded)'}\n"
                     f"- Its picture was seen to show: {seen}\n")
    record_txt = json.dumps({k: c[k] for k in ("transition", "transition_note", "time_of_day", "location", "objects",
                                               "characters", "start_state", "camera", "lighting",
                                               "intentional_changes") if c.get(k)}, ensure_ascii=False)[:3000]
    return f"""You are the continuity supervisor of a film. Look at Image 1 (this scene's picture) and compare it with
what it should show. Be strict about real mistakes and honest about doubt.

IMAGES:
{roles}

SCENE {scene.get('position', 0) + 1}: {scene.get('heading') or ''}
What we should see: {scene.get('visual') or ''}
Continuity record (what the script means here): {record_txt or '(none)'}

CHARACTERS (permanent look; clothes as listed for this scene):
{chr(10).join(cast_lines) or '(none in frame)'}
{world}{prev_part}
CHECK: faces and appearance against the reference pictures; clothing and accessories; positions and which way people
face; the place; important objects and who holds them; light, weather and time of day; whether it follows the
previous scene; whether it shows what the scene text says.

RULES:
- A change listed in intentional_changes, or a time jump / new place / flashback, is intentional: report it with
  intentional true, it is not a mistake.
- If the picture doesn't show something clearly (small, dark, turned away), mark certainty "uncertain". Never guess.
- Score 0-100 (100 = perfect). Omit a score that doesn't apply.
- Each fix is a short instruction for a new picture of this scene (e.g. "Maya's dress must be red, as in scene 2").
- observed: what the picture actually shows, so the next scene can follow it."""


def _ranges(nums: list[int]) -> str:
    """[1, 2, 3, 7, 9, 10] -> '1-3, 7, 9-10'."""
    out, start, prev = [], None, None
    for n in sorted(nums):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ", ".join(out)


def audit(scenes: list[dict], jobs: list[dict]) -> dict:
    """The whole-production picture audit, from every scene (not a sample):
    what is checked, outdated, queued/running, failed, never checked or has no
    picture; every mistake found; and whether the audit is complete."""
    ordered = sorted(scenes, key=lambda s: s.get("position") or 0)
    latest = {}
    for j in sorted([j for j in jobs if j.get("kind") == "inspect"], key=lambda j: str(j.get("created_at") or "")):
        latest[j.get("scene_id")] = j
    rows, problems, cost = [], [], 0.0
    buckets = {k: [] for k in ("checked", "outdated", "queued", "failed", "unchecked", "no_picture")}
    scores = []
    for i, s in enumerate(ordered, start=1):
        ins = s.get("inspection") or {}
        job = latest.get(s.get("id")) or {}
        if not s.get("image_path"):
            state = "no_picture"
        elif job.get("status") in ("queued", "running"):
            state = "queued"
        elif ins.get("result") and ins.get("image_path") == s.get("image_path"):
            state = "checked"
        elif job.get("status") == "failed":
            state = "failed"
        elif ins.get("result"):
            state = "outdated"
        else:
            state = "unchecked"
        buckets[state].append(i)
        cost += float(ins.get("cost_usd") or 0)
        row = {"scene": i, "scene_id": s.get("id"), "state": state}
        if state == "checked":
            r = ins["result"]
            row.update(overall=r["scores"].get("overall"), needs_attention=needs_attention(r))
            scores.append(r["scores"].get("overall") or 0)
            for p in errors(r):
                problems.append({"scene": i, "scene_id": s.get("id"), **{k: p.get(k) for k in
                                 ("kind", "severity", "certainty", "expected", "seen", "fix", "characters")}})
        if state == "failed":
            row["error"] = (job.get("error_message") or "")[:200]
        rows.append(row)
    with_pictures = len(ordered) - len(buckets["no_picture"])
    return {"total": len(ordered), "with_pictures": with_pictures, "scenes": rows, "problems": problems,
            **{k: len(v) for k, v in buckets.items()}, "lists": {k: _ranges(v) for k, v in buckets.items() if v},
            "complete": with_pictures > 0 and len(buckets["checked"]) == with_pictures,
            "average_score": round(sum(scores) / len(scores)) if scores else None,
            "attention": [r["scene"] for r in rows if r.get("needs_attention")],
            "cost_usd": round(cost, 4)}


def audit_lines(a: dict) -> list[str]:
    """The audit for the assistant: covers every scene, says what is missing."""
    if not a["total"]:
        return []
    out = [f"VISUAL AUDIT (every scene of the production): {a['checked']} of {a['with_pictures']} scene pictures checked"
           f"{' (complete)' if a['complete'] else ' (NOT complete)'}; {a['queued']} queued or running; {a['failed']} "
           f"failed; {a['outdated']} outdated; {a['unchecked']} never checked; {a['no_picture']} scenes have no picture."]
    for k, label in (("queued", "Being checked now"), ("failed", "Check failed (not checked)"),
                     ("outdated", "Checked before the picture changed"), ("unchecked", "Never checked"),
                     ("no_picture", "No picture yet")):
        if a["lists"].get(k):
            out.append(f"  {label}: scenes {a['lists'][k]}")
    if a["average_score"] is not None:
        out.append(f"  Average score of checked pictures: {a['average_score']}/100. Need attention: scenes "
                   f"{_ranges(a['attention']) or 'none'}.")
    for p in a["problems"][:30]:
        out.append(f"  - Scene {p['scene']} [{p['kind']}, {p['severity']}, {p['certainty']}]: expected {p['expected']}; "
                   f"seen {p['seen']}. Fix: {p['fix']}")
    if len(a["problems"]) > 30:
        out.append(f"  (+{len(a['problems']) - 30} more problems; ask about a scene to see them)")
    return out
