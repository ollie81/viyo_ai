"""The multi-shot director: a scene becomes an ordered list of camera shots.

The AI director reads the whole scene (story purpose, the continuity record,
who says what, how it starts and ends, the scene before) and plans 1-4 shots
the way a film director would: an establishing or wide two-shot, a medium on
the speaker, an over-the-shoulder, a reaction close-up, an insert of an object,
choosing by what the moment needs (a conversation, an emotional beat, action,
a quiet moment), never one fixed pattern. Each shot says why it was chosen.

Every shot then gets its own opening picture and its own video clip (the video
provider makes one shot per request), each told the scene's continuity record
and exactly what the previous shot showed, and the render joins the shots of
a scene in order. Scenes without an approved shot plan work as before: one
picture and one clip for the whole scene.
"""
from __future__ import annotations

import re
from typing import Optional

# Shot type -> (how it's described to the image/video model, the reference-view camera for faces)
SHOT_TYPES = {
    "establishing_wide": ("Establishing wide shot: the whole place, characters small in the frame", "wide"),
    "wide_two_shot": ("Wide two-shot: both characters in frame, full bodies or knees up", "wide"),
    "medium": ("Medium shot: waist up", "front"),
    "close_up": ("Close-up: head and shoulders, the face fills most of the frame", "close_up"),
    "extreme_close_up": ("Extreme close-up: eyes, a hand or a small detail fills the frame", "close_up"),
    "over_the_shoulder": ("Over-the-shoulder shot: from behind one character's shoulder (soft, in the foreground) "
                          "onto the other character's face", "front"),
    "reaction": ("Reaction shot: close-up of the listener reacting, not speaking", "close_up"),
    "pov": ("Point-of-view shot: what the character sees, from their eyes", "wide"),
    "insert": ("Insert shot: a close view of an important object or detail", "close_up"),
}
MOVEMENTS = ("static", "slow push in", "pull out", "pan", "tilt", "tracking", "handheld", "dolly", "crane")
MIN_SHOT = 1.5          # seconds; shorter shots feel like a glitch
MAX_SHOTS = 4

_S = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {"shots": {"type": "array", "items": {"type": "object", "properties": {
        "type": {"type": "string", "enum": list(SHOT_TYPES)},
        "angle": {"type": "string", "description": "e.g. eye level, low angle, high angle, from Maya's side."},
        "movement": {"type": "string", "description": "static, slow push in, pull out, pan, tilt, tracking, handheld."},
        "duration_seconds": {"type": "number"},
        "description": {"type": "string", "description": "What we see in this shot, as one coherent image."},
        "characters": {"type": "array", "items": {"type": "object", "properties": {
            "name": _S, "position": {"type": "string", "description": "Where in frame: left, right, centre, background."},
            "facing": _S, "action": _S}, "required": ["name"]}},
        "line_indices": {"type": "array", "items": {"type": "integer"},
                         "description": "Which of the scene's lines (0-based) are heard in this shot, in order."},
        "audio": {"type": "string", "description": "Sound besides the lines (e.g. rain, a door slams), or empty."},
        "frame_prompt": {"type": "string", "description": "The opening frame of this shot, for the image model."},
        "video_prompt": {"type": "string", "description": "What moves during the shot, for the video model."},
        "continuity": {"type": "string", "description": "What must stay the same from the previous shot."},
        "reason": {"type": "string", "description": "Why this shot here, in one plain sentence for the creator."},
    }, "required": ["type", "duration_seconds", "description", "characters", "line_indices", "frame_prompt",
                    "video_prompt", "reason"]}}},
    "required": ["shots"],
}


def _t(x, n=400) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()[:n]


def planner_prompt(project: dict, scene: dict, number: int, total: int, prev: Optional[dict],
                   prev_last_shot: Optional[dict], cast_lines: list[str]) -> str:
    c = scene.get("continuity") or {}
    lines = [d for d in scene.get("dialogue") or [] if (d.get("text") or "").strip()]
    talk = "\n".join(f"{i}. {d.get('character')}: \"{d.get('text')}\"{' (' + d['emotion'] + ')' if d.get('emotion') else ''}"
                     for i, d in enumerate(lines)) or "(no dialogue)"
    before = ""
    if prev is not None:
        pc = prev.get("continuity") or {}
        before = (f"\nTHE SCENE BEFORE ended with: {pc.get('end_state') or prev.get('visual') or '(not recorded)'}"
                  + (f"\nIts last shot: {prev_last_shot.get('shot_type')} – {prev_last_shot.get('description')}"
                     if prev_last_shot else ""))
    return f"""You are the director of a {project.get('kind') or 'short drama'} ({project.get('aspect_ratio')}, about
{project.get('target_seconds')} seconds in total, {total} scenes). Plan the camera shots for scene {number}.

SCENE {number}: {scene.get('heading') or ''} · {scene.get('duration_seconds')} seconds
What happens: {scene.get('visual') or ''}
Purpose: {c.get('purpose') or '(not given)'} · How it starts: {c.get('start_state') or '(not given)'}
What happens during it: {c.get('action') or '(not given)'} · How it ends: {c.get('end_state') or '(not given)'}
Time/light: {', '.join(x for x in (c.get('time_of_day'), c.get('lighting'), c.get('weather')) if x) or '(not given)'}
Positions, objects, outfits: {c.get('characters') or '(not given)'}{before}

LINES:
{talk}

CHARACTERS (permanent look):
{chr(10).join(cast_lines) or '(none)'}

HOW TO DIRECT:
- Choose 1 to {MAX_SHOTS} shots from what the moment needs, not a fixed pattern. Every shot must add something:
  a new piece of information, a reaction, a reveal. No two shots in a row with the same type and framing.
- A conversation usually opens on a wide two-shot (who is where), then a medium or over-the-shoulder on the speaker,
  a close-up on the listener reacting to an important line, and back to the speaker. An emotional beat lingers on a
  close-up. Action uses wider shots and movement. A quiet moment may need a single held shot.
- Insert shots show an important object (a letter, a phone). Use point-of-view when we should see what a character sees.
- Keep the 180-degree rule: once a character is on the left, they stay on the left in every shot of this scene.
- Continuity: same place, light, clothes, objects in the same hands, positions that follow from the previous shot.
  The first shot starts from how the scene starts (and the scene before, if it's continuous).
- Every line of the scene is heard in exactly one shot, in order (line_indices). A shot's duration must fit its lines.
- The durations add up to the scene's {scene.get('duration_seconds')} seconds; no shot under {MIN_SHOT} seconds.
- reason: one plain sentence for the creator, e.g. "A close-up on Daniel so we see the lie land."."""


def _norm_names(names: list[str], cast_names: list[str]) -> list[str]:
    lower = {n.lower(): n for n in cast_names}
    first = {n.split(" ")[0].lower(): n for n in cast_names}
    out = []
    for n in names:
        k = (n or "").strip().lower()
        hit = lower.get(k) or first.get(k.split(" ")[0] if k else "")
        if hit and hit not in out:
            out.append(hit)
    return out


def clean_plan(raw, scene: dict, cast_names: list[str]) -> list[dict]:
    """Valid shots only, in order: known types, characters from the cast, every
    line heard exactly once in order, durations that add up to the scene."""
    shots_in = (raw or {}).get("shots") if isinstance(raw, dict) else None
    lines = [d for d in scene.get("dialogue") or [] if (d.get("text") or "").strip()]
    total = float(scene.get("duration_seconds") or 6)
    shots = []
    for s in (shots_in or [])[:MAX_SHOTS]:
        if not isinstance(s, dict) or s.get("type") not in SHOT_TYPES or not _t(s.get("description")):
            continue
        chars = []
        for c in s.get("characters") or []:
            if isinstance(c, dict) and _norm_names([c.get("name")], cast_names):
                chars.append({"name": _norm_names([c.get("name")], cast_names)[0], "position": _t(c.get("position"), 80),
                              "facing": _t(c.get("facing"), 80), "action": _t(c.get("action"), 200)})
        mov = _t(s.get("movement"), 40).lower()
        dur = s.get("duration_seconds")
        shots.append({"shot_type": s["type"], "camera_angle": _t(s.get("angle"), 80) or "eye level",
                      "camera_movement": mov if mov in MOVEMENTS else ("static" if not mov else mov[:40]),
                      "duration_seconds": float(dur) if isinstance(dur, (int, float)) and not isinstance(dur, bool) and dur > 0 else 0.0,
                      "description": _t(s.get("description"), 600), "characters": chars,
                      "lines_wanted": [i for i in s.get("line_indices") or [] if isinstance(i, int) and 0 <= i < len(lines)],
                      "audio": _t(s.get("audio"), 200), "frame_prompt": _t(s.get("frame_prompt"), 900) or _t(s.get("description"), 600),
                      "video_prompt": _t(s.get("video_prompt"), 900) or _t(s.get("description"), 600),
                      "continuity": _t(s.get("continuity"), 400), "reason": _t(s.get("reason"), 300)})
    if not shots:
        raise ValueError("the director returned no usable shots")
    shots = shots[:max(1, int(total // MIN_SHOT))]
    # Lines: each heard once, in order; the director's choice when it's consistent.
    nxt = 0
    for i, s in enumerate(shots):
        take = [k for k in s.pop("lines_wanted") if k >= nxt]
        if take:
            end = max(take)
            s["line_indices"] = list(range(nxt, end + 1))
            nxt = end + 1
        else:
            s["line_indices"] = []
    if nxt < len(lines):
        shots[-1]["line_indices"] += list(range(nxt, len(lines)))
    for s in shots:
        s["dialogue"] = [lines[k] for k in s["line_indices"]]
    _fit_durations(shots, total)
    return shots


def speech_seconds(lines: list) -> float:
    """How long a shot's lines take to say. One rule everywhere for shots: the
    plan gives each shot at least this, the shot's clip is requested at the
    shot's length (so the model paces the lines to fit), and the render never
    cuts a shot shorter than this. (A whole-scene clip keeps its own, roomier
    estimate in studio_worker.speech_seconds.)"""
    if not lines:
        return 0.0
    words = sum(len((d.get("text") or "").split()) for d in lines)
    return words / 2.4 + 0.4 * len(lines) + 0.5


def clip_seconds(shot: dict) -> Optional[int]:
    """The clip length to ask the video model for: the shot's length (or its
    lines' length if longer), within what the model accepts; None if it can't fit."""
    import math
    need = max(float(shot.get("duration_seconds") or 3), speech_seconds(shot.get("dialogue") or []))
    if need > 15:
        return None
    return max(3, min(15, math.ceil(need)))


def _fit_durations(shots: list[dict], total: float) -> None:
    """Scale to the scene's length; every shot at least MIN_SHOT and long enough
    for its lines. Speech wins: if the lines need more time than the scene has,
    the shots run longer rather than cut a line."""
    need = [max(MIN_SHOT, speech_seconds(s["dialogue"])) for s in shots]
    given = [s["duration_seconds"] or n for s, n in zip(shots, need)]
    scale = total / sum(given) if sum(given) else 1
    durs = [max(n, g * scale) for g, n in zip(given, need)]
    over = sum(durs) - total
    if over > 0:      # take it from the shots with spare time
        spare = [d - n for d, n in zip(durs, need)]
        room = sum(spare)
        durs = [d - (over * sp / room if room else 0) for d, sp in zip(durs, spare)]
    for s, d in zip(shots, durs):
        s["duration_seconds"] = round(max(MIN_SHOT, d), 2)


def default_plan(scene: dict, cast_names: list[str]) -> dict:
    """A plausible plan (used by the fake AI in tests): coverage chosen from the
    scene's kind, not one pattern for everything."""
    lines = [d for d in scene.get("dialogue") or [] if (d.get("text") or "").strip()]
    speakers = _norm_names([d.get("character") for d in lines], cast_names)
    present = _norm_names([c.get("name") for c in (scene.get("continuity") or {}).get("characters") or []
                           if c.get("in_frame", True)], cast_names) or speakers or cast_names[:1]
    side = {n: ("left" if i == 0 else "right") for i, n in enumerate(present)}
    who = [{"name": n, "position": side[n]} for n in present]
    if len(speakers) >= 2:
        a, b = speakers[0], speakers[1]
        shots = [{"type": "wide_two_shot", "description": f"{a} and {b} face each other", "characters": who,
                  "line_indices": [0], "reason": "Shows where both of them are before they speak."},
                 {"type": "over_the_shoulder", "description": f"Over {b}'s shoulder onto {a}",
                  "characters": [{"name": a, "position": side.get(a, "left")}, {"name": b, "position": side.get(b, "right")}],
                  "line_indices": list(range(1, max(2, len(lines) - 1))), "reason": f"Stays with {a} while the argument builds."},
                 {"type": "reaction", "description": f"Close-up of {b} reacting",
                  "characters": [{"name": b, "position": side.get(b, "right")}],
                  "line_indices": list(range(max(2, len(lines) - 1), len(lines))), "reason": f"We need to see the line land on {b}."}]
    elif lines:
        n = speakers[0] if speakers else present[0]
        shots = [{"type": "medium", "description": f"{n} speaks", "characters": [{"name": n, "position": "centre"}],
                  "line_indices": list(range(len(lines))), "reason": "One person speaking: stay with them."},
                 {"type": "close_up", "description": f"Close on {n}'s face", "characters": [{"name": n, "position": "centre"}],
                  "line_indices": [], "reason": "Holds the feeling after the words."}]
    elif "phone" in (scene.get("visual") or "").lower() or "letter" in (scene.get("visual") or "").lower():
        shots = [{"type": "medium", "description": scene.get("visual") or "", "characters": who, "line_indices": [],
                  "reason": "Where we are and who is there."},
                 {"type": "insert", "description": "The object up close", "characters": [], "line_indices": [],
                  "reason": "The object matters to the story: show it."}]
    else:
        shots = [{"type": "establishing_wide", "description": scene.get("visual") or "", "characters": who,
                  "line_indices": [], "reason": "A quiet moment: one held shot sets the mood."}]
    for s in shots:
        s.update(duration_seconds=0, frame_prompt=s["description"], video_prompt=s["description"], movement="static")
    return {"shots": shots}


def camera_text(shot: dict) -> str:
    base = SHOT_TYPES.get(shot.get("shot_type"), ("Medium shot", "front"))[0]
    extra = ", ".join(x for x in (shot.get("camera_angle"), shot.get("camera_movement")) if x and x != "static")
    return base + (f" ({extra})" if extra else "")


def reference_camera(shot: dict) -> str:
    return SHOT_TYPES.get(shot.get("shot_type"), ("", "front"))[1]


def continuity_block(scene: dict, shots: list[dict], index: int, prev_scene_end: str = "") -> str:
    """What this shot must keep from the scene and from the previous shot."""
    c = scene.get("continuity") or {}
    shot = shots[index]
    out = ["SHOT CONTINUITY (one scene filmed in several shots; everything matches across them):"]
    look = ", ".join(x for x in (c.get("time_of_day"), c.get("lighting"), c.get("weather")) if x)
    if look:
        out.append(f"- Same light and time in every shot: {look}.")
    sides = {}
    for s in shots:
        for ch in s.get("characters") or []:
            pos = (ch.get("position") or "").lower()
            if ch["name"] not in sides and ("left" in pos or "right" in pos):
                sides[ch["name"]] = "left" if "left" in pos else "right"
    if sides:
        out.append("- Screen direction (keep it): " + "; ".join(f"{n} on the {sd}" for n, sd in sides.items()) + ".")
    if index == 0:
        start = c.get("start_state") or prev_scene_end
        if start:
            out.append(f"- First shot of the scene: it starts with {start}.")
    else:
        p = shots[index - 1]
        out.append(f"- The previous shot ({p.get('shot_type', '').replace('_', ' ')}) showed: {p.get('description')}.")
        pos = "; ".join(f"{ch['name']}: {ch.get('position') or ''} {ch.get('action') or ''}".strip()
                        for ch in p.get("characters") or [])
        if pos:
            out.append(f"- Where people were in it: {pos}.")
    if shot.get("continuity"):
        out.append(f"- Keep: {shot['continuity']}.")
    return "\n".join(out) + "\n"


def uses_shots(scene: dict, shots: list[dict]) -> bool:
    """The scene is made from its shots (approved plan) instead of one clip."""
    return bool(scene.get("shot_plan_approved")) and bool(shots)


def shot_problem(scene_number: int, shots: list[dict], talking: bool) -> Optional[str]:
    """Why a scene made of shots can't be rendered yet."""
    for s in sorted(shots, key=lambda x: x.get("position") or 0):
        n = (s.get("position") or 0) + 1
        if not (s.get("image_path") or s.get("video_path")):
            return f"Scene {scene_number}, shot {n} has no picture yet."
        if talking and s.get("dialogue") and not s.get("video_path"):
            return f"Scene {scene_number}, shot {n}'s talking video isn't made yet."
    return None
