"""Shot-to-shot visual continuity: every shot picture is compared with the shot
before it and with the approved references, by the same vision checker as
scene pictures (studio_inspect), with rules for how film shots change.

A cut is not a mistake: a close-up after a wide shot, a reverse angle, an
insert, a time jump or an approved outfit change all change the picture on
purpose. What must not change without a reason: who the person is, what they
wear, the place, the light, the props, where people stand (screen direction)
and what has already happened.

The check returns a per-shot report: which checks passed and failed, how
serious each failure is, why, and the fix. A critical identity or story
contradiction fails the shot whatever the average score is. The creator can
accept a flagged change as intentional; the acceptance belongs to that exact
picture and is dropped when a new one is made.

A vision model can be wrong. Nothing here proves two AI pictures show the
same person; it reports what the model sees, with its certainty.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from typing import Optional

import studio_inspect as inspect_mod

# The checks a creator sees, in order, and the checker's problem kinds/scores behind each.
CHECKS = (
    ("identity", "Same person (face, hair, age, build)", ("character",), "character"),
    ("clothing", "Clothes and accessories", ("outfit",), "outfit"),
    ("location", "Same place", ("location",), "location"),
    ("props", "Props and objects", ("prop",), "props"),
    ("lighting", "Light, time of day, weather", ("lighting",), "lighting"),
    ("action", "Follows what just happened", ("story", "transition"), "transition"),
    ("camera", "Camera and screen direction", ("position", "camera"), "camera"),
)
CRITICAL_KINDS = ("character", "story")    # never hidden by a good average
MAX_AUTO_FIX = 1                           # automatic corrected pictures per shot job (cost control)
MAX_IMAGES = inspect_mod.MAX_IMAGES

CUT_RULES = {
    "establishing_wide": "a wide view of the whole place; people small in frame",
    "wide_two_shot": "both people in frame, full or three-quarter body",
    "medium": "waist up",
    "close_up": "the face fills the frame; the background is mostly out of view or out of focus",
    "extreme_close_up": "one detail (eyes, a hand, an object) fills the frame",
    "over_the_shoulder": "seen past one person's shoulder (back of their head in the foreground) onto the other",
    "reaction": "the listener's face as they react; usually a close-up or medium",
    "pov": "what a character sees, from their eyes; that character is not in frame",
    "insert": "an object or detail up close; people may be absent",
}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def signature(shot: dict, prev_path: Optional[str], scene: dict, outfit_ids: list, identity_versions: list) -> str:
    """Changes when anything the check depends on changes."""
    key = json.dumps([shot.get("image_path"), prev_path, shot.get("shot_type"), shot.get("description"),
                      shot.get("characters") or [], shot.get("continuity") or "", scene.get("continuity") or {},
                      sorted(map(str, outfit_ids)), sorted(map(str, identity_versions))], sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def current(shot: dict, sig: str) -> Optional[dict]:
    ins = shot.get("inspection") or {}
    if ins.get("signature") == sig and ins.get("image_path") == shot.get("image_path") and ins.get("result"):
        return ins
    return None


def prompt(scene: dict, scene_number: int, shot: dict, index: int, shots: list[dict], prev: Optional[dict],
           cast_lines: list[str], world: str, labels: list[str]) -> str:
    """The vision check for one shot picture. prev: {"label", "shot_type", "description", "observed",
    "transition", "same_scene"} for the picture it follows, or None."""
    c = scene.get("continuity") or {}
    roles = "\n".join(f"Image {i}: {label}" for i, label in enumerate(labels, start=1))
    kind = shot.get("shot_type") or "medium"
    nxt = shots[index + 1] if index + 1 < len(shots) else None
    before = ""
    if prev:
        seen = json.dumps(prev.get("observed"), ensure_ascii=False)[:1200] if prev.get("observed") else "(not checked)"
        how = "the shot just before, in the same scene: a CUT to a new framing" if prev.get("same_scene") else \
            f"the last shot of the previous scene ({(prev.get('transition') or 'continuous').replace('_', ' ')})"
        before = (f"\nTHE PICTURE IT FOLLOWS ({how}):\n- Framing: {(prev.get('shot_type') or '').replace('_', ' ')} – "
                  f"{CUT_RULES.get(prev.get('shot_type'), '')}\n- What it shows: {prev.get('description') or ''}\n"
                  f"- It was seen to show: {seen}\n")
    record = json.dumps({k: c[k] for k in ("transition", "time_of_day", "lighting", "weather", "location", "objects",
                                           "characters", "start_state", "action", "end_state", "intentional_changes")
                         if c.get(k)}, ensure_ascii=False)[:2500]
    lines = "; ".join(f"{d.get('character')}: \"{d.get('text')}\"" for d in shot.get("dialogue") or [])
    return f"""You are the continuity supervisor of a film. Image 1 is shot {index + 1} of {len(shots)} in scene
{scene_number}. Compare it with the picture it follows and the approved references. Be strict about real mistakes
and honest about doubt.

IMAGES:
{roles}

THIS SHOT: {kind.replace('_', ' ')} ({CUT_RULES.get(kind, '')}); camera {shot.get('camera_angle') or 'eye level'},
{shot.get('camera_movement') or 'static'}. What we should see: {shot.get('description') or ''}
Who is in frame and where: {json.dumps(shot.get('characters') or [], ensure_ascii=False)}
What must carry over (director's note): {shot.get('continuity') or '(none)'}
Lines in this shot: {lines or '(none)'}
{'Next shot: ' + (nxt.get('shot_type') or '').replace('_', ' ') + ' – ' + (nxt.get('description') or '') if nxt else 'This is the last shot of the scene.'}

SCENE {scene_number}: {scene.get('heading') or ''}. {scene.get('visual') or ''}
Scene record (what the script means here): {record or '(none)'}

CHARACTERS (permanent look; clothes as listed for this scene):
{chr(10).join(cast_lines) or '(none in frame)'}
{world}{before}
HOW SHOTS CHANGE (these are NOT mistakes):
- A cut changes the framing: wide -> medium -> close-up, a reverse angle, an over-the-shoulder, a reaction, an insert.
  Compare the PERSON, clothes, place, light and objects, never the framing itself.
- A reverse angle shows the other side of the room: the background is different, but it must be the same room
  (same walls, colours, furniture style, light). Screen direction holds: whoever was on the left stays on the left.
- A close-up or insert may hide the background, the outfit or other people: that is not a missing item. Mark what
  you can't see as "uncertain", never as a mistake.
- Objects may move when the action moves them (picked up, handed over, put down): check the action says so.
- People enter and leave when the scene says so; a new place, a time jump (e.g. day to night) or a change listed in
  intentional_changes is intentional. Report those with intentional true.

CHECK: same person (face, hair, age, skin, build) against the references; clothes and accessories; same place;
props and who holds them; light, time of day and weather; whether it follows what just happened; camera framing as
planned and the 180-degree rule (screen direction); whether it contradicts the scene.

RULES:
- kind: character, outfit, location, prop, position (screen direction / where people stand), lighting, camera
  (the framing is not what was planned), story (contradicts what happened or the screenplay), transition.
- Score 0-100 (100 = perfect) for overall, character, outfit, location, props, lighting, transition and camera;
  omit a score that doesn't apply.
- Each fix is a short instruction for a new picture of THIS shot (e.g. "Ama's raincoat must be mustard, as in shot 2").
- observed: what the picture actually shows, so the next shot can follow it."""


def report(ins: Optional[dict], image_path: Optional[str] = None) -> dict:
    """The creator-facing continuity report of one shot picture."""
    ins = ins or {}
    result = ins.get("result")
    if not image_path:
        return {"status": "no_picture", "checks": []}
    if not result:
        return {"status": "unchecked", "checks": []}
    if ins.get("image_path") != image_path:
        return {"status": "outdated", "checks": [], "note": "Checked before the picture changed."}
    scores = result.get("scores") or {}
    probs = result.get("problems") or []
    checks, critical = [], []
    for key, label, kinds, score_key in CHECKS:
        mine = [p for p in probs if p.get("kind") in kinds]
        mistakes = [p for p in mine if not p.get("intentional")]
        serious = [p for p in mistakes if p.get("certainty") != "uncertain"]
        worst = next((s for s in ("high", "medium", "low") if any(p["severity"] == s for p in serious)), None)
        item = {"check": key, "label": label, "score": scores.get(score_key),
                "passed": not serious and (scores.get(score_key) is None or scores.get(score_key) >= inspect_mod.ATTENTION_SCORE),
                "severity": worst,
                "problems": [{k: p.get(k) for k in ("severity", "certainty", "expected", "seen", "explanation", "fix",
                                                    "characters", "intentional")} for p in mine]}
        if not item["passed"] and worst is None:
            item["severity"] = "medium"        # a low score with no named problem
        if [p for p in serious if p["severity"] == "high" and p["kind"] in CRITICAL_KINDS]:
            critical.append(key)
        checks.append(item)
    failed = [c for c in checks if not c["passed"]]
    high = any(c["severity"] == "high" for c in failed)
    status = "failed" if critical or high else "attention" if failed or scores.get("overall", 100) < inspect_mod.ATTENTION_SCORE \
        else "passed"
    accepted = ins.get("accepted") if (ins.get("accepted") or {}).get("image_path") == image_path else None
    out = {"status": "accepted" if accepted and status != "passed" else status, "checks": checks,
           "overall": scores.get("overall"), "critical": critical, "summary": result.get("summary") or "",
           "fix": inspect_mod.correction_text(result), "at": ins.get("at")}
    if accepted:
        out["accepted"] = {"note": accepted.get("note") or "", "at": accepted.get("at")}
    return out


def accept(ins: dict, image_path: str, note: str = "") -> dict:
    """The creator says the flagged change is intentional (for this picture only)."""
    return {**(ins or {}), "accepted": {"image_path": image_path, "note": (note or "").strip()[:300], "at": _now()}}


def better(a: dict, b: dict) -> bool:
    """Is report b better than report a? (for keeping the best of two attempts)"""
    rank = {"passed": 3, "accepted": 3, "attention": 2, "failed": 1}
    ra, rb = rank.get(a.get("status"), 0), rank.get(b.get("status"), 0)
    if rb != ra:
        return rb > ra
    return (b.get("overall") or 0) > (a.get("overall") or 0)


def chat_line(rep: dict) -> str:
    """One short phrase for the assistant's shot list."""
    s = rep.get("status")
    if s in ("no_picture", "unchecked"):
        return "" if s == "no_picture" else " [continuity not checked]"
    if s == "outdated":
        return " [continuity check outdated]"
    failed = [c for c in rep["checks"] if not c["passed"]]
    if s == "passed":
        return f" [continuity passed {rep.get('overall')}/100]"
    if s == "accepted":
        return f" [continuity flagged but ACCEPTED by the creator as intentional: {', '.join(c['check'] for c in failed)}]"
    bits = []
    for c in failed[:3]:
        p = next((x for x in c["problems"] if not x.get("intentional")), {})
        bits.append(f"{c['check']} ({c['severity']}): expected {p.get('expected') or '?'}, seen {p.get('seen') or '?'}"
                    + (" (uncertain)" if p.get("certainty") == "uncertain" else ""))
    return f" [continuity {s.upper()}{' – critical: ' + ', '.join(rep['critical']) if rep.get('critical') else ''}: " \
        + "; ".join(bits) + (f". Fix: {rep['fix'][:200]}" if rep.get("fix") else "") + "]"


def memory(project: dict, cast: list[dict], outfits: list[dict], elements: list[dict], scenes: list[dict],
           shots_by: dict, wearing: Optional[dict] = None) -> dict:
    """The production's continuity memory, assembled from what is saved
    (identities, approved references, outfits, places, scene records, shot
    plans and what each checked picture was seen to show). Structured, not
    conversation history."""
    import studio_identity as ident
    out_names = {o["id"]: o for o in outfits}
    chars = []
    for c in cast:
        views = c.get("reference_views") or {}
        chars.append({
            "id": c["id"], "name": c["name"],
            "approved_references": [k for k, v in views.items() if isinstance(v, dict) and v.get("approved")],
            "references": sorted(views) or (["portrait"] if c.get("reference_path") else []),
            "identity": ident.identity_text(c["name"], c.get("identity"), c.get("description") or ""),
            "identity_locked": bool(c.get("identity_locked_at")),
            "outfits": [{"id": o["id"], "name": o["name"], "default": bool(o.get("is_default"))}
                        for o in outfits if o["character_id"] == c["id"] and (o.get("status") or "saved") == "saved"]})
    places = [{"id": e["id"], "name": e["name"], "description": e.get("description") or "",
               "reference": "approved" if e.get("status") == "approved" else "made" if e.get("reference_path") else "none",
               **{k: v for k, v in (e.get("details") or {}).items() if k in ("time_of_day", "mood", "layout", "lighting")}}
              for e in elements if e.get("kind") == "location"]
    props = [{"id": e["id"], "name": e["name"], "description": e.get("description") or "",
              "owner": (e.get("details") or {}).get("owner")} for e in elements if e.get("kind") == "prop"]
    scene_rows = []
    for n, s in enumerate(sorted(scenes, key=lambda x: x.get("position") or 0), start=1):
        rec = s.get("continuity") or {}
        seen = ((s.get("inspection") or {}).get("result") or {}).get("observed") or {}
        rows = sorted(shots_by.get(s["id"], []), key=lambda r: r.get("position") or 0)
        shot_rows = []
        for k, sh in enumerate(rows):
            rep = report(sh.get("inspection"), sh.get("image_path"))
            shot_rows.append({
                "shot": k + 1, "id": sh["id"], "type": sh.get("shot_type"), "camera_angle": sh.get("camera_angle"),
                "camera_movement": sh.get("camera_movement"), "characters": sh.get("characters") or [],
                "previous": k if k else None, "next": k + 2 if k + 1 < len(rows) else None,
                "start": sh.get("description") or "", "carry_over": sh.get("continuity") or "",
                "seen": ((sh.get("inspection") or {}).get("result") or {}).get("observed") or {},
                "continuity": rep["status"], "approved": bool(sh.get("approved_at"))})
        scene_rows.append({
            "scene": n, "id": s["id"], "heading": s.get("heading") or "", "situation": rec.get("purpose") or s.get("visual") or "",
            "transition": rec.get("transition") or "", "time_of_day": rec.get("time_of_day") or "",
            "lighting": rec.get("lighting") or "", "weather": rec.get("weather") or "",
            "characters": [{"name": x.get("name"), "position": x.get("position"), "facing": x.get("facing"),
                            "outfit": ((wearing or {}).get(s["id"]) or {}).get(x.get("name")) or x.get("outfit"),
                            "emotion": x.get("emotion_end") or x.get("emotion_start"), "holding": x.get("holding_at_end") or x.get("holding"),
                            "enters": x.get("enters"), "leaves": x.get("leaves")} for x in rec.get("characters") or []],
            "start": rec.get("start_state") or "", "happening": rec.get("action") or "", "end": rec.get("end_state") or "",
            "objects": rec.get("objects") or [], "intentional_changes": rec.get("intentional_changes") or [],
            "picture_seen": seen, "shots": shot_rows})
    return {"characters": chars, "locations": places, "props": props, "scenes": scene_rows,
            "outfit_names": {k: v["name"] for k, v in out_names.items()}}
