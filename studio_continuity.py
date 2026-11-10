"""Scene continuity: one film, not unrelated pictures.

The breakdown (the "continuity director") reads the whole screenplay and gives
every scene a structured record: who is in frame, where they stand, what they
hold and wear, how each one feels, how the scene starts and ends, the camera,
the light, and how it connects to the scene before (continuous, a cut in the
same place, a new place, a time jump, a flashback).

That record is the handoff: the next scene's picture and video are told the
previous scene's ending state, and a free rule-based review compares every
pair of consecutive scenes before the film is assembled.

Text can't guarantee identical faces or rooms. The real anchors are the
reference pictures (characters, outfits, places, objects, the previous shot)
and, for video, the scene picture as the first frame; this module decides
which of those matter for each shot and what the prompt says about them.
"""
from __future__ import annotations

import re
from typing import Optional

TRANSITIONS = ("continuous", "same_place_cut", "new_place", "time_jump", "flashback")
CARRY = ("continuous", "same_place_cut")   # the previous shot's look carries over
_S = {"type": "string"}

SCHEMA = {
    "type": "object",
    "properties": {
        "purpose": {"type": "string", "description": "What this scene does for the story, in one sentence."},
        "transition": {"type": "string", "enum": list(TRANSITIONS),
                       "description": "How it follows the scene before: continuous (the very next moment), "
                                      "same_place_cut (same place, a little later or another angle), new_place, "
                                      "time_jump, flashback. The first scene is new_place."},
        "transition_note": {"type": "string", "description": "For new_place, time_jump or flashback: what changed "
                                                             "(e.g. 'next morning', 'she has driven to the hospital')."},
        "time_of_day": {"type": "string", "description": "e.g. night, dawn, afternoon."},
        "location": {"type": "string", "description": "The place, as named in the heading."},
        "objects": {"type": "array", "items": _S, "description": "Important objects visible or used."},
        "characters": {"type": "array", "items": {"type": "object", "properties": {
            "name": _S,
            "in_frame": {"type": "boolean", "description": "false if only heard (phone, off-screen voice)."},
            "position": {"type": "string", "description": "Where they stand/sit and relative to others, e.g. "
                                                          "'left of frame, by the door, facing Daniel'."},
            "facing": _S,
            "holding": {"type": "array", "items": _S, "description": "What is in their hands at the START, with which "
                                                                     "hand if it matters."},
            "holding_at_end": {"type": "array", "items": _S, "description": "What is in their hands at the END."},
            "outfit": {"type": "string", "description": "What they wear in this scene (same as before unless the "
                                                        "screenplay changes it)."},
            "emotion_start": _S, "emotion_end": _S,
            "enters": {"type": "boolean"}, "leaves": {"type": "boolean"},
        }, "required": ["name", "in_frame"]}},
        "start_state": {"type": "string", "description": "The first moment: who is where, doing what, physical state "
                                                         "(wet, injured, out of breath)."},
        "action": {"type": "string", "description": "What happens during the scene: who moves, who stays still."},
        "end_state": {"type": "string", "description": "The last moment, which the next scene starts from."},
        "camera": {"type": "string", "description": "Framing and angle, e.g. 'medium two-shot, eye level'."},
        "camera_movement": {"type": "string", "description": "e.g. 'static', 'slow push in', 'handheld follow'."},
        "lighting": _S,
        "from_previous": {"type": "string", "description": "What carries over from the previous scene's end."},
        "setup_next": {"type": "string", "description": "What this scene sets up for the next one."},
        "intentional_changes": {"type": "array", "items": _S,
                                "description": "Changes the screenplay MEANS to make from the previous scene "
                                               "(e.g. 'Maya changes into the black dress', 'night becomes dawn')."},
    },
    "required": ["purpose", "transition", "characters", "start_state", "action", "end_state", "camera"],
}


def _t(x, n=300) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()[:n]


def _list(x, n=8, m=120) -> list[str]:
    return [_t(v, m) for v in (x or [])[:n] if _t(v, m)] if isinstance(x, list) else []


def clean(raw) -> dict:
    """Keeps only known fields, trimmed. {} for nothing usable."""
    if not isinstance(raw, dict):
        return {}
    out = {k: _t(raw.get(k), 600 if k in ("start_state", "action", "end_state") else 300)
           for k in ("purpose", "transition_note", "time_of_day", "location", "start_state", "action", "end_state",
                     "camera", "camera_movement", "lighting", "from_previous", "setup_next")}
    t = _t(raw.get("transition"), 30)
    out["transition"] = t if t in TRANSITIONS else ""
    out["objects"] = _list(raw.get("objects"))
    out["intentional_changes"] = _list(raw.get("intentional_changes"), 8, 200)
    chars = []
    for c in (raw.get("characters") or [])[:8]:
        if not isinstance(c, dict) or not _t(c.get("name"), 60):
            continue
        chars.append({"name": _t(c.get("name"), 60), "in_frame": c.get("in_frame") is not False,
                      "position": _t(c.get("position"), 200), "facing": _t(c.get("facing"), 80),
                      "holding": _list(c.get("holding"), 4, 80), "holding_at_end": _list(c.get("holding_at_end"), 4, 80),
                      "outfit": _t(c.get("outfit"), 200), "emotion_start": _t(c.get("emotion_start"), 60),
                      "emotion_end": _t(c.get("emotion_end"), 60), "enters": bool(c.get("enters")),
                      "leaves": bool(c.get("leaves"))})
    out["characters"] = chars
    return {k: v for k, v in out.items() if v not in ("", [], None)}


def merge(old: Optional[dict], patch: Optional[dict]) -> dict:
    """A creator's edit (e.g. a new end state) on top of the saved record."""
    base = dict(old or {})
    for k, v in (patch or {}).items():
        base[k] = v
    return clean(base)


def of(scene: Optional[dict]) -> dict:
    return (scene or {}).get("continuity") or {}


def _same(a: str, b: str) -> bool:
    return _t(a).lower() == _t(b).lower() and _t(a) != ""


def transition(prev: Optional[dict], scene: dict) -> str:
    """The saved transition, or a guess for scenes made before continuity
    records existed (same heading = same place)."""
    t = of(scene).get("transition")
    if t:
        return t
    if not prev:
        return "new_place"
    return "same_place_cut" if _same(prev.get("heading"), scene.get("heading")) else "new_place"


def carries_look(prev: Optional[dict], scene: dict) -> bool:
    """Should the previous shot's picture be sent as a reference?"""
    return bool(prev) and transition(prev, scene) in CARRY


def in_frame_names(scene: dict) -> Optional[list[str]]:
    """Names the continuity record puts in frame, or None when there's no record."""
    chars = of(scene).get("characters")
    if not chars:
        return None
    return [c["name"] for c in chars if c.get("in_frame", True)]


def _who(chars: list[dict], key: str) -> list[str]:
    out = []
    for c in chars:
        if not c.get("in_frame", True):
            continue
        # Positions and facing are the scene's START; the end only has what is held and felt.
        start = key == "holding"
        bits = [c.get("position") if start else "", f"facing {c['facing']}" if start and c.get("facing") else ""]
        hold = c.get(key) or []
        if hold:
            bits.append("holding " + ", ".join(hold))
        emo = c.get("emotion_start" if key == "holding" else "emotion_end")
        if emo:
            bits.append(f"feeling {emo}")
        bits = [b for b in bits if b]
        if bits:
            out.append(f"{c['name']}: " + "; ".join(bits))
    return out


def previous_end(prev: Optional[dict]) -> str:
    if not prev:
        return ""
    c = of(prev)
    parts = [c.get("end_state") or ""]
    parts += _who(c.get("characters") or [], "holding_at_end")
    if c.get("time_of_day"):
        parts.append(f"time: {c['time_of_day']}")
    return " | ".join(p for p in parts if p)


def image_block(prev: Optional[dict], scene: dict) -> str:
    """CONTINUITY text for the scene picture (its opening frame)."""
    c = of(scene)
    t = transition(prev, scene)
    lines = []
    if prev is None:
        lines.append("This is the opening shot of the film.")
    elif t in CARRY:
        end = previous_end(prev)
        lines.append("This shot follows straight on from the previous scene"
                     + (" (the very next moment)" if t == "continuous" else " (same place, a moment later)")
                     + ". Keep the same place, light, weather, clothes and what people hold unless listed as a change.")
        if end:
            lines.append(f"The previous scene ended with: {end}.")
    else:
        note = c.get("transition_note") or {"new_place": "a different place", "time_jump": "later in time",
                                            "flashback": "a flashback"}.get(t, "")
        lines.append(f"New sequence ({t.replace('_', ' ')}{': ' + note if note else ''}). Do not copy the previous "
                     "scene's position or place; characters keep their faces and, unless listed as a change, their clothes.")
    if c.get("start_state"):
        lines.append(f"This scene starts with: {c['start_state']}.")
    who = _who(c.get("characters") or [], "holding")
    if who:
        lines.append("Positions at the start: " + " / ".join(who) + ".")
    heard = [x["name"] for x in c.get("characters") or [] if not x.get("in_frame", True)]
    if heard:
        lines.append(f"Only heard, NOT in the picture: {', '.join(heard)}.")
    if c.get("intentional_changes"):
        lines.append("Intended changes from the previous scene: " + "; ".join(c["intentional_changes"]) + ".")
    look = ", ".join(x for x in (c.get("camera"), c.get("lighting"), c.get("time_of_day")) if x)
    if look:
        lines.append(f"Camera and light: {look}.")
    return "CONTINUITY:\n" + "\n".join(f"- {x}" for x in lines) + "\n"


def video_block(prev: Optional[dict], scene: dict) -> str:
    """BEGINNING / DURING / END direction for the clip."""
    c = of(scene)
    if not c:
        return ""
    chars = c.get("characters") or []
    parts = ["The first frame is the scene picture: keep its composition, the characters' faces, clothes and the place."]
    if c.get("start_state"):
        parts.append(f"BEGINNING: {c['start_state']}.")
    if c.get("action"):
        parts.append(f"DURING: {c['action']}.")
    if c.get("end_state"):
        parts.append(f"END: {c['end_state']}.")
    cam = ", ".join(x for x in (c.get("camera"), c.get("camera_movement")) if x)
    if cam:
        parts.append(f"CAMERA: {cam}.")
    keep = []
    for x in chars:
        if not x.get("in_frame", True):
            continue
        if x.get("outfit"):
            keep.append(f"{x['name']} wears {x['outfit']}")
        held = x.get("holding") or []
        ends = x.get("holding_at_end") or []
        if held and set(map(str.lower, held)) == set(map(str.lower, ends)):
            keep.append(f"{x['name']} keeps holding {', '.join(held)}")
    keep.append("nobody else appears; no new objects or actions beyond what is described")
    parts.append("MUST NOT CHANGE: " + "; ".join(keep) + ".")
    heard = [x["name"] for x in chars if not x.get("in_frame", True)]
    if heard:
        parts.append(f"{', '.join(heard)} is only heard, never seen.")
    return " ".join(parts)


# ------------------------------------------------------------ review

def _held(c: dict, key: str) -> set:
    return {h.strip().lower() for h in c.get(key) or [] if h.strip()}


def _mentions(text: str, thing: str) -> bool:
    words = [w for w in re.findall(r"[a-z]{3,}", thing.lower()) if w not in ("the", "her", "his", "their", "and")]
    return bool(words) and any(re.search(rf"\b{re.escape(w)}", text.lower()) for w in words[-1:])


def _intended(scene: dict, *words: str) -> bool:
    text = " ".join(of(scene).get("intentional_changes") or []).lower()
    return any(w.lower() in text for w in words if w)


def review(scenes: list[dict], outfits_by_scene: dict, outfit_names: dict, char_names: dict,
           location_of: dict) -> list[dict]:
    """Compares every pair of consecutive scenes. Free: no AI call.

    outfits_by_scene: {scene id: {character id: outfit id}} (what each wears now)
    outfit_names / char_names: id -> name; location_of: {scene id: location id or None}.
    Each issue: {scene, previous, kind, severity ('fix'|'check'), message, fix}."""
    issues = []
    ordered = sorted(scenes, key=lambda s: s.get("position") or 0)
    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        n, pn = i + 1, i
        t = transition(prev, cur)
        cp, cc = of(prev), of(cur)

        def add(kind, severity, message, fix):
            issues.append({"scene": n, "previous": pn, "scene_id": cur["id"], "kind": kind, "severity": severity,
                           "message": message, "fix": fix})

        if t in CARRY:
            # Outfits: the same character wearing something else with no reason.
            po, co = outfits_by_scene.get(prev["id"]) or {}, outfits_by_scene.get(cur["id"]) or {}
            for cid in set(po) & set(co):
                if po[cid] and co[cid] and po[cid] != co[cid]:
                    who = char_names.get(cid, "A character")
                    new = outfit_names.get(co[cid], "another outfit")
                    if not _intended(cur, who.split(" ")[0], new, "outfit", "change", "dress", "clothes"):
                        add("outfit", "fix", f"{who} wears \"{outfit_names.get(po[cid], 'one outfit')}\" in scene {pn} "
                                             f"but \"{new}\" in scene {n}, with no change in the story.",
                            f"Set {who}'s outfit in scene {n} to the one from scene {pn}, or add the change to the script.")
            # Place: a continuous scene somewhere else.
            pl, cl = location_of.get(prev["id"]), location_of.get(cur["id"])
            if pl and cl and pl != cl and t == "continuous":
                add("location", "fix", f"Scene {n} is marked as the very next moment, but it's in a different place "
                                       f"from scene {pn}.",
                    f"Mark scene {n} as a new place, or set it in the same place as scene {pn}.")
            # Time of day.
            if cp.get("time_of_day") and cc.get("time_of_day") and not _same(cp["time_of_day"], cc["time_of_day"]) \
                    and not _intended(cur, cc["time_of_day"], "time", "dawn", "morning", "night"):
                add("time", "fix", f"Scene {pn} is at {cp['time_of_day']} but scene {n} is at {cc['time_of_day']}, "
                                   "with no time jump.",
                    f"Mark scene {n} as a time jump, or keep it at {cp['time_of_day']}.")
            # Objects in hand.
            cur_chars = {x["name"].lower(): x for x in cc.get("characters") or []}
            text = " ".join([cc.get("start_state", ""), cc.get("action", ""), cur.get("visual") or ""])
            for x in cp.get("characters") or []:
                y = cur_chars.get(x["name"].lower())
                if not y or x.get("leaves"):
                    continue
                for obj in _held(x, "holding_at_end") - _held(y, "holding"):
                    if not _mentions(text, obj):
                        add("object", "fix", f"{x['name']} ends scene {pn} holding {obj}, but starts scene {n} without "
                                             "it and the scene doesn't show it being put down.",
                            f"Add {obj} to {x['name']}'s hands at the start of scene {n}, or show it being put down.")
                # Feelings.
                e1, e2 = x.get("emotion_end"), y.get("emotion_start")
                if e1 and e2 and not _same(e1, e2) and t == "continuous" and not _intended(cur, e2):
                    add("emotion", "check", f"{x['name']} ends scene {pn} {e1} but starts scene {n} {e2}.",
                        "Check that the change of feeling is meant; otherwise carry it over.")
            # People who left coming back without entering.
            left = {x["name"].lower() for x in cp.get("characters") or [] if x.get("leaves")}
            for y in cc.get("characters") or []:
                if y["name"].lower() in left and y.get("in_frame", True) and not y.get("enters"):
                    add("presence", "fix", f"{y['name']} leaves in scene {pn} but is back in scene {n} without "
                                           "entering.",
                        f"Show {y['name']} coming back in scene {n}, or remove them from it.")

        # Pictures made from an older version of the scene before.
        made = cur.get("made_with") or {}
        if made.get("_previous_shot") and prev.get("image_path") and made["_previous_shot"] != prev["image_path"] \
                and cur.get("image_path"):
            add("stale", "fix", f"Scene {n}'s picture was made from an older picture of scene {pn}, which has "
                                "changed since.",
                f"Make a new picture for scene {n} so it follows the new scene {pn}.")
    return issues
