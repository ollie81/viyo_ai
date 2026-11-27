"""VIYO action intelligence: action scenes planned as connected sequences, not unrelated clips.

What it does (pure code, no AI call, no cost by itself)
  detect       which scenes are action (a fight, a chase, an escape, a fall...): strong action words, or several
               weaker ones. A quiet "Maya walks home" stays a drama scene.
  sequences    consecutive action scenes form one sequence with a story purpose and a beat for each scene:
               setup -> escalation -> reversal -> climax, and the scene right after is its consequence. A single
               action scene carries the whole arc inside its own shots.
  planning     an action scene may have up to MAX_ACTION_SHOTS shots (drama keeps studio_shots.MAX_SHOTS). The
               director chooses how many; the quote reserves the upper bound and returns what isn't used.
  spatial      each shot can say where people start and end, which way they move and what changes in the place;
               the review checks the 180-degree rule, matching ends and starts, and chase direction.
  feasibility  what current video models do badly (fast contact between several people, hands, crashes, crowds,
               transformations, readable text, long continuous choreography...) is flagged on the plan with a
               simpler cinematic alternative, before anything is made.

Honest limits
  Detection is word-based: the creator can always re-plan or edit shots. Feasibility rules come from what image-to-
  video models are generally known to struggle with; they were not measured on our providers (that needs the live
  benchmark). Nothing here makes a clip look good; it gives the director and the models clearer, safer instructions.

ENABLED (env STUDIO_ACTION_INTELLIGENCE, default on) switches all of this, and the per-shot prompt context in
studio_shots, back to the earlier behaviour: the benchmark compares the two in one run.
"""
from __future__ import annotations

import os
import re
from typing import Optional

ENABLED = os.environ.get("STUDIO_ACTION_INTELLIGENCE", "on").lower() not in ("off", "0", "false", "no")

MAX_ACTION_SHOTS = 8
BRIEF_CHARS = 3200          # the action brief added to the director's prompt (for the planning hold)
BEATS = ("setup", "escalation", "reversal", "climax", "consequence")
BEAT_TEXT = {
    "setup": "SETUP: show the geography and the goal before anything happens: who is where, what they want, the "
             "threat. A wide or establishing shot first.",
    "escalation": "ESCALATION: the danger grows in clear steps; each shot is one readable action and its effect.",
    "reversal": "REVERSAL: the situation turns: the plan fails, the hunter becomes the hunted, a weapon is lost.",
    "climax": "CLIMAX: the decisive action, shown clearly (cause, impact, reaction), then hold on the result.",
    "consequence": "CONSEQUENCE: the cost on the people: injury, loss, relief, a changed relationship. Slower, closer.",
    "whole": "ONE-SCENE ACTION: the whole arc in this scene's shots: set up the geography, escalate, turn, the "
             "decisive moment, then the consequence on a face.",
}
DIRECTIONS = ("left_to_right", "right_to_left", "toward_camera", "away_from_camera", "none")

# Words that make a scene action on their own, and weaker ones that need company.
_STRONG = r"(fight|fights|fighting|punch\w*|kick\w*|chase\w*|explo\w+|gunfire|shoot\w*|shot at|fires at|stab\w*|" \
          r"sword|swords|blade|duel\w*|attack\w*|ambush\w*|tackle\w*|grapple\w*|wrestle\w*|brawl\w*|crash\w*|" \
          r"disarm\w*|leaps?\b|leaping|dodge\w*|ducks under|strikes?|struck|blast\w*|battle\w*|martial|kung fu)"
_WEAK = r"(run|runs|running|sprint\w*|jump\w*|climb\w*|falls?|falling|grab\w*|escape\w*|flee\w*|races?|racing|" \
        r"throws?|thrown|swing\w*|smash\w*|pursu\w+|hides? from|gun|guns|pistol|rifle|weapon|knife|car|motorbike|" \
        r"rooftop|roofs|helicopter|danger)"


def _text(scene: dict) -> str:
    c = scene.get("continuity") or {}
    return " ".join(str(x or "") for x in (scene.get("heading"), scene.get("visual"), c.get("action"),
                                           c.get("purpose"), c.get("start_state"), c.get("end_state"))).lower()


def action_score(scene: dict) -> int:
    t = _text(scene)
    return 2 * len(re.findall(rf"\b{_STRONG}", t)) + len(re.findall(rf"\b{_WEAK}\b", t))


def is_action(scene: dict, project: Optional[dict] = None) -> bool:
    """An action scene: one strong action word, or at least two weaker ones."""
    return ENABLED and action_score(scene) >= 2


def max_shots(scene: dict, project: Optional[dict] = None) -> int:
    import studio_shots
    return MAX_ACTION_SHOTS if is_action(scene, project) else studio_shots.MAX_SHOTS


def reserve_shots(scene: dict) -> int:
    """The most shots an action scene may get (the quote reserves this; unused shots are returned)."""
    import studio_shots
    dur = float(scene.get("duration_seconds") or 6)
    return max(1, min(MAX_ACTION_SHOTS, int(dur // studio_shots.MIN_SHOT) or 1))


# ------------------------------------------------------------------ sequences

def _beats(n: int) -> list[str]:
    if n == 1:
        return ["whole"]
    if n == 2:
        return ["setup", "climax"]
    if n == 3:
        return ["setup", "escalation", "climax"]
    rev = max(2, round(n * 2 / 3) - 1)
    return ["setup"] + ["escalation"] * (rev - 1) + ["reversal"] + ["escalation"] * (n - rev - 2) + ["climax"]


def sequences(scenes: list[dict], project: Optional[dict] = None) -> list[dict]:
    """Consecutive action scenes as sequences, each with a beat per scene and the scene after as its consequence."""
    ordered = sorted(scenes, key=lambda s: s.get("position") or 0)
    out, run = [], []
    for k, s in enumerate(ordered + [None]):
        if s is not None and is_action(s, project):
            run.append(s)
            continue
        if run:
            beats = _beats(len(run))
            purpose = next(((r.get("continuity") or {}).get("purpose") for r in run
                            if (r.get("continuity") or {}).get("purpose")), "")
            seq = {"index": len(out) + 1, "scenes": [r["id"] for r in run], "beats": dict(zip([r["id"] for r in run], beats)),
                   "purpose": purpose or "", "consequence_scene": s["id"] if s is not None else None}
            out.append(seq)
            run = []
    return out


def scene_beat(scene: dict, scenes: list[dict], project: Optional[dict] = None) -> Optional[dict]:
    """Where this scene sits in an action sequence (or None): its beat, the beats around it and the purpose."""
    for seq in sequences(scenes, project):
        ids = seq["scenes"]
        if scene["id"] in ids:
            k = ids.index(scene["id"])
            return {"sequence": seq["index"], "beat": seq["beats"][scene["id"]], "position": k + 1, "of": len(ids),
                    "before": seq["beats"][ids[k - 1]] if k else None,
                    "after": seq["beats"][ids[k + 1]] if k + 1 < len(ids) else "consequence",
                    "purpose": seq["purpose"]}
        if seq["consequence_scene"] == scene["id"]:
            return {"sequence": seq["index"], "beat": "consequence", "position": len(ids) + 1, "of": len(ids) + 1,
                    "before": "climax", "after": None, "purpose": seq["purpose"]}
    return None


# ------------------------------------------------------------------ the director's brief

def planner_block(scene: dict, beat: Optional[dict]) -> str:
    """Added to the shot planner's prompt for an action scene (or the consequence after one)."""
    if not ENABLED or beat is None:
        return ""
    b = beat["beat"]
    seq = (f"Action sequence {beat['sequence']}, scene {beat['position']} of {beat['of']}: this scene is the "
           f"{b.upper()}" + (f" (after the {beat['before']}" if beat.get("before") else "")
           + (f", before the {beat['after']})" if beat.get("after") else (")" if beat.get("before") else "")) + ".")
    purpose = (f"What this sequence changes in the story: {beat['purpose']}." if beat.get("purpose")
               else "Say in each reason what the action changes in the story (a goal won or lost, a cost, a turn). "
                    "No action for its own sake.")
    if b == "consequence":
        return f"""
ACTION CONTEXT: {seq}
- {BEAT_TEXT['consequence']}
- {purpose}
"""
    return f"""
ACTION SCENE ({seq})
- {BEAT_TEXT.get(b, BEAT_TEXT['whole'])}
- {purpose}
- Use as many shots as the action needs, up to {MAX_ACTION_SHOTS}; fewer if the moment is simple. Each shot is ONE
  readable beat: a clear starting pose, one action, a clear result. Cause, impact, reaction.
- Coverage: a wide that shows who is where; mediums for the moves; close-ups and inserts for impacts and hands;
  a reaction shot for the cost. Vary sizes and angles between consecutive shots; movement follows the action.
- SPATIAL CONTINUITY: for every shot give each character's position at the start and at the end (end_position),
  which way they move (movement_direction: {', '.join(DIRECTIONS)}), and anything that changes in the place
  (environment_change: a door breaks, a fire starts, glass on the floor). The next shot starts where this one ended.
  Keep screen direction: a chase keeps running the same way across shots unless a shot shows the turn.
- beat: label every shot setup, escalation, reversal, climax or consequence, in order.
- WHAT VIDEO MODELS DO BADLY (avoid or simplify): several people in fast physical contact in one shot, hands doing
  precise things, cars crashing or flipping, explosions next to faces, crowds, transformations, readable text on
  screens, and long continuous choreography. Prefer: one attacker per shot, the impact implied by a cut and a
  reaction, the approach then the aftermath, an insert for the hands, sound for what we don't see.
"""


# ------------------------------------------------------------------ feasibility

_RISKS = [
    ("multi_contact", r"\b(grapple\w*|wrestle\w*|tangle\w*|brawl\w*|melee|pile on|struggle\w* (with|for)|fights? (with )?(two|three|several|all))\b",
     "high", "Several people in fast physical contact in one shot: bodies and limbs merge in generated video.",
     "Split it: one attacker per shot, cut on the impact, then a reaction."),
    ("hands", r"\b(pick\w* the lock|unlock\w*|types?|typing|ties?|tying|loads? the (gun|pistol|rifle)|reloads?|fingers|"
              r"defus\w*|wire\w*|cuts? the wire|sews?|sewing)\b",
     "medium", "Precise hand work: generated hands often lose fingers or do the action wrongly.",
     "Show it as a short static insert, or keep the hands out of frame and show the result."),
    ("vehicle", r"\b(crash\w*|collid\w*|flip\w*|rolls? over|swerv\w*|car chase|high[- ]speed)\b",
     "high", "Vehicles crashing or moving fast: shapes deform and motion breaks.",
     "Show the approach, cut on impact with sound, then the aftermath."),
    ("explosion", r"\b(explo\w+|blast\w*|fireball|on fire|flames? (engulf|swallow))\b",
     "medium", "Explosions or fire close to people: faces and bodies distort.",
     "Keep the blast wide and away from faces; show the character's reaction in its own shot."),
    ("crowd", r"\b(crowd|army|armies|dozens|hundreds|mob|horde|stadium)\b",
     "medium", "Crowds: generated people repeat, merge or lose detail.",
     "Fewer people in frame; imply the crowd with sound and off-screen voices."),
    ("transform", r"\b(transform\w*|morph\w*|shape-?shift\w*|turns? into|power(s)? (surge|activate)|glow\w* with power)\b",
     "high", "Transformations and powers changing a body: identity drifts during the change.",
     "Cut away on a flash of light; show before and after in separate shots."),
    ("fall", r"\b(falls? from|jumps? (from|between|across) (the )?(roof|building|rooftop)|leaps? (from|between|across))\b",
     "medium", "Long jumps or falls: bodies float or the landing breaks.",
     "Split into take-off and landing shots; cut during the flight."),
    ("weapon_precision", r"\b(sword clash|blades? (meet|clash)|bullets? (hit|strike)|through the scope|aims? at)\b",
     "medium", "Weapons meeting or bullets landing precisely: timing and contact are unreliable.",
     "Show the swing, cut, then the impact on a reaction or an object."),
    ("text", r"\b(reads? (the|a) (screen|message|sign|note)|message says|screen shows|the sign says|text on)\b",
     "medium", "Readable text: generated text is usually garbled.",
     "Don't show the words; show the glow on the face, or put the words in captions."),
    ("camera", r"\b(360|orbit\w*|spins? around|one continuous take|oner)\b",
     "medium", "Complex camera moves: geometry and faces warp.",
     "Use a simpler move: a push in, a pan or a tracking shot alongside."),
]
LONG_ACTION_SECONDS = 6.0


def feasibility(shot: dict, scene: Optional[dict] = None) -> list[dict]:
    """What in this shot current video models struggle with, and a simpler cinematic alternative."""
    text = " ".join(str(shot.get(k) or "") for k in ("description", "frame_prompt", "video_prompt")).lower()
    text += " " + " ".join(str(c.get("action") or "") for c in shot.get("characters") or []).lower()
    out = []
    for key, pattern, risk, issue, alternative in _RISKS:
        if re.search(pattern, text):
            out.append({"kind": key, "risk": risk, "issue": issue, "alternative": alternative})
    acts = len(re.findall(rf"\b{_STRONG}", text)) + len(re.findall(rf"\b{_WEAK}\b", text))
    if float(shot.get("duration_seconds") or 0) > LONG_ACTION_SECONDS and acts >= 3:
        out.append({"kind": "long_choreography", "risk": "high",
                    "issue": f"{acts} actions in one {float(shot['duration_seconds']):.0f}-second shot: long "
                             "choreography breaks in generated video.",
                    "alternative": "Split it into shorter shots, one action each."})
    in_frame = len(shot.get("characters") or [])
    if in_frame >= 3 and acts >= 2:
        out.append({"kind": "many_people", "risk": "medium",
                    "issue": f"{in_frame} people moving in one action shot: they may merge or swap looks.",
                    "alternative": "Keep two people in frame at most; cover the others in their own shots."})
    return out


# ------------------------------------------------------------------ reviews of an action plan

def _side(pos: str) -> Optional[str]:
    p = (pos or "").lower()
    return "left" if "left" in p else "right" if "right" in p else None


def spatial_review(shots: list[dict]) -> list[dict]:
    """Positions and movement across consecutive shots: screen side kept, ends meet starts, chases keep direction."""
    out = []
    side, last_dir = {}, None
    for i, s in enumerate(shots, start=1):
        moved = set()
        for c in s.get("characters") or []:
            start, end = _side(c.get("position")), _side(c.get("end_position"))
            if start and c["name"] in side and side[c["name"]] != start:
                out.append({"shot": i, "kind": "screen_direction",
                            "problem": f"{c['name']} jumps from the {side[c['name']]} to the {start} of the frame "
                                       f"between shots {i - 1} and {i}.",
                            "fix": f"Keep {c['name']} on the {side[c['name']]}, or show the move inside a shot."})
            if end or start:
                side[c["name"]] = end or start
            if end and start and end != start:
                moved.add(c["name"])
        a = (s.get("_action") or {}).get("movement_direction") or "none"
        if a in ("left_to_right", "right_to_left") and last_dir in ("left_to_right", "right_to_left") and a != last_dir:
            out.append({"shot": i, "kind": "movement_direction",
                        "problem": f"The movement reverses between shots {i - 1} and {i} ({last_dir.replace('_', ' ')} "
                                   f"then {a.replace('_', ' ')}): the chase seems to turn back.",
                        "fix": "Keep the direction, or show the turn in a shot."})
        if a != "none":
            last_dir = a
    return out


def action_review(scene: dict, shots: list[dict]) -> list[dict]:
    """Coverage and structure of an action scene's plan."""
    out = []
    types = [s.get("shot_type") for s in shots]
    if len(shots) >= 3 and not ({"establishing_wide", "wide_two_shot"} & set(types)):
        out.append({"shot": 1, "kind": "geography", "problem": "No wide shot: the viewer can't tell who is where.",
                    "fix": "Open the action on a wide that shows the place and the positions."})
    if len(shots) >= 3 and types[-1] not in ("reaction", "close_up", "extreme_close_up") and \
            not any(t == "reaction" for t in types[len(types) // 2:]):
        out.append({"shot": len(shots), "kind": "consequence", "problem": "The action ends without a reaction.",
                    "fix": "End on a close-up or reaction: what it cost them."})
    beats = [(s.get("_action") or {}).get("beat") for s in shots]
    ranks = [BEATS.index(b) for b in beats if b in BEATS]
    if ranks and ranks != sorted(ranks):
        out.append({"shot": 1 + next(i for i in range(1, len(ranks)) if ranks[i] < ranks[i - 1]), "kind": "beats",
                    "problem": "The beats go backwards (for example a climax before the set-up).",
                    "fix": "Order the shots setup, escalation, reversal, climax, consequence."})
    for i, s in enumerate(shots, start=1):
        for f in feasibility(s, scene):
            if f["risk"] == "high":
                out.append({"shot": i, "kind": f"feasibility_{f['kind']}", "problem": f["issue"], "fix": f["alternative"]})
    return out + spatial_review(shots)
