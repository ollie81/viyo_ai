"""The whole-film report: a free review of the finished cut, not of single pictures.

What it looks at
  duplicates      two moments of the film that show (nearly) the same picture: the same file used twice, or frames
                  whose fingerprints match. Fingerprints (a 64-bit difference hash of one frame per shot) are taken by
                  the render job from the files it already has on disk: ffmpeg only, no AI, no cost.
  missing         a planned shot or scene with no picture or clip, a line of dialogue no shot carries, a film rendered
                  before the shots or scenes changed.
  camera          the same shot size over and over: across scene cuts, scenes that all open the same way, one shot
                  type or one camera move dominating, a film made mostly of single-shot scenes (a slideshow).
  pacing          the film much longer or shorter than planned, every shot the same length, long static moments,
                  lines spoken faster than people can follow, lines longer than their scene.
  screen_time     a character who speaks but is never seen, a planned character missing from a scene's shots, a main
                  character seen less than a minor one.
  continuity      what the existing free checks already found (scene-to-scene review, picture and shot checks), and
                  what hasn't been checked yet.
  ending          the last scene: made, long enough, carries the main character, and the story check's verified ending
                  score when there is one.

What it never does
  It never remakes anything and never calls a paid service. Every problem names the scene (and shot), says what is
  wrong and what to do, and which route the fix takes: "edit" (free: change the plan, the length or the lines),
  "free_fix" (the shot failed VIYO's own check, so its one free fix applies), "remake" (a paid remake, its exact price
  shown before the creator confirms) or "story" (change the screenplay). The creator decides.

Honest limits
  Fingerprints catch the same or nearly the same picture, not two different pictures of the same idea. Screen time
  counts shots, not seconds of a face on screen. The ending check is structural; how good an ending FEELS is only the
  story check's (verified) score. Nothing here judges how good the acting or the pictures are.
"""
from __future__ import annotations

import subprocess
from collections import Counter
from typing import Optional

HASH_DUPLICATE = 5            # bits of 64: at most this many differ -> the same picture
WORDS_PER_SECOND_MAX = 3.3    # faster than this and most viewers can't follow (about 200 words a minute)
LENGTH_TOLERANCE = 0.25       # of the planned length
STATIC_SECONDS = 10.0         # one unchanging picture, no lines, for this long, reads as a pause in the film
TYPE_SHARE_MAX = 0.45         # one shot size for more of the film than this
MOVE_SHARE_MAX = 0.7
SINGLE_SHOT_SHARE_MAX = 0.6
SAME_OPENING_RUN = 3          # this many scenes in a row opening with the same shot size
MIN_ENDING_SECONDS = 3.0

CATEGORIES = ("duplicates", "missing", "camera", "pacing", "screen_time", "continuity", "ending")
LABELS = {"duplicates": "Duplicated shots", "missing": "Missing shots", "camera": "Camera variety",
          "pacing": "Pacing", "screen_time": "Character screen time", "continuity": "Continuity",
          "ending": "Ending"}
ROUTES = {"edit": "Free edit", "free_fix": "Free fix (VIYO's check found a defect)",
          "remake": "Paid remake (price shown first)", "story": "Change the screenplay"}
_SEV = {"high": 0, "medium": 1, "low": 2}


# ------------------------------------------------------------------ fingerprints (free, ffmpeg)

def frame_hash(path: str, video: bool = False, seconds: Optional[float] = None) -> Optional[str]:
    """A 64-bit difference hash of one frame (the middle of a clip, or the picture), as 16 hex digits.
    None when the file can't be read: a missing fingerprint never fails a render."""
    if not path:
        return None
    px = b""
    # The middle of the piece; a clip shorter than its scene (the render holds its last frame) falls back to its start.
    for at in ([max(0.0, seconds / 2)] if video and seconds else []) + [None]:
        cmd = ["ffmpeg", "-v", "error"] + (["-ss", f"{at:.2f}"] if at else [])
        cmd += ["-i", path, "-frames:v", "1", "-vf", "scale=9:8:flags=area,format=gray", "-f", "rawvideo", "-"]
        try:
            px = subprocess.run(cmd, capture_output=True, timeout=30).stdout
        except Exception:
            px = b""
        if len(px) >= 72:
            break
    if len(px) < 72:
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
    return f"{bits:016x}"


def distance(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# ------------------------------------------------------------------ helpers

def _first(name: str) -> str:
    return (name or "").strip().split(" ")[0].lower()


def _lines(items) -> list:
    return [d for d in items or [] if (d.get("text") or "").strip()]


def _where(scene: int, shot: Optional[int] = None) -> str:
    return f"Scene {scene}" + (f", shot {shot}" if shot else "")


def _units(scenes: list[dict], shots: dict) -> list[dict]:
    """The film's pieces in order: every shot of a directed scene, or the scene itself as one piece."""
    out = []
    for i, s in enumerate(scenes, start=1):
        rows = shots.get(s["id"]) or []
        if rows:
            for k, sh in enumerate(rows, start=1):
                out.append({"scene": i, "shot": k, "scene_id": s["id"], "shot_id": sh.get("id"),
                            "type": sh.get("shot_type") or "", "move": (sh.get("camera_movement") or "static").lower(),
                            "seconds": float(sh.get("duration_seconds") or 0),
                            "names": {_first(c.get("name")) for c in sh.get("characters") or [] if c.get("name")},
                            "made": bool(sh.get("video_path") or sh.get("image_path")),
                            "needs_clip": s.get("mode") == "video", "has_clip": bool(sh.get("video_path")),
                            "path": sh.get("video_path") or sh.get("image_path")})
        else:
            cam = ((s.get("continuity") or {}).get("camera") or "").lower()
            out.append({"scene": i, "shot": None, "scene_id": s["id"], "shot_id": None,
                        "type": _type_from_words(cam), "move": ((s.get("continuity") or {}).get("camera_movement")
                                                                or "").lower(),
                        "seconds": float(s.get("duration_seconds") or 0), "names": None,
                        "made": bool(s.get("video_path") or s.get("image_path")),
                        "needs_clip": s.get("mode") == "video", "has_clip": bool(s.get("video_path")),
                        "path": s.get("video_path") or s.get("image_path")})
    return out


def _type_from_words(text: str) -> str:
    for key, words in (("extreme_close_up", "extreme close"), ("close_up", "close"), ("over_the_shoulder", "over the shoulder"),
                       ("establishing_wide", "establishing"), ("wide_two_shot", "two-shot"), ("medium", "medium"),
                       ("wide_two_shot", "wide"), ("pov", "point of view"), ("insert", "insert")):
        if words in text:
            return key
    return ""


def _nice(t: str) -> str:
    return (t or "single shot").replace("_", " ")


def _a(words: str) -> str:
    return ("an " if words[:1] in "aeiou" else "a ") + words


# ------------------------------------------------------------------ the checks

def _duplicates(units, segments, add):
    seen = {}
    for u in units:
        if u["path"]:
            if u["path"] in seen:
                a = seen[u["path"]]
                add("duplicates", "high", u["scene"], u["shot"],
                    f"{_where(u['scene'], u['shot'])} uses exactly the same picture as {_where(a['scene'], a['shot'])}.",
                    "Make a new picture for one of them, or cut one if the moment is repeated by mistake.", "remake")
            else:
                seen[u["path"]] = u
    marks = [g for g in segments or [] if g.get("hash")]
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            a, b = marks[i], marks[j]
            if a.get("path") and a.get("path") == b.get("path"):
                continue                                   # already reported as the same file
            if distance(a["hash"], b["hash"]) <= HASH_DUPLICATE:
                next_to = j == i + 1
                add("duplicates", "medium", b["scene"], b.get("shot"),
                    f"{_where(b['scene'], b.get('shot'))} looks almost identical to {_where(a['scene'], a.get('shot'))}"
                    + (" right before it: the cut shows nothing new." if next_to else "."),
                    "Change the framing or the moment of one of them. If the repeat is deliberate (a callback), "
                    "keep it.", "edit" if next_to else "remake")


def _missing(scenes, units, shots, render, segments, add):
    for u in units:
        if not u["made"]:
            add("missing", "high", u["scene"], u["shot"], f"{_where(u['scene'], u['shot'])} has no picture yet.",
                "Make it before the film is rendered.", "remake")
        elif u["needs_clip"] and not u["has_clip"]:
            add("missing", "medium", u["scene"], u["shot"],
                f"{_where(u['scene'], u['shot'])} is a video scene but has only a still picture.",
                "Make its clip, or switch the scene to a still picture.", "remake")
    for i, s in enumerate(scenes, start=1):
        rows = shots.get(s["id"]) or []
        if not rows:
            continue
        carried = Counter(" ".join((d.get("text") or "").split()).lower() for sh in rows for d in _lines(sh.get("dialogue")))
        for d in _lines(s.get("dialogue")):
            key = " ".join(d["text"].split()).lower()
            if carried[key] > 0:
                carried[key] -= 1
            else:
                add("missing", "high", i, None,
                    f"Scene {i}: the line \"{d['text'][:80]}\" isn't in any shot, so it isn't in the film.",
                    "Give the line to a shot in the shot plan.", "edit")
    if render and segments is not None:
        # Each rendered piece remembers the file it used: any difference means the film is older than the plan.
        used = [(g.get("scene_id"), g.get("shot_id"), g.get("path")) for g in segments]
        now = [(u["scene_id"], u["shot_id"], u["path"]) for u in units]
        changed = sorted({u["scene"] for k, u in enumerate(units) if k >= len(used) or used[k] != now[k]})
        if changed or len(used) != len(now):
            where = (f"scene{'s' if len(changed) > 1 else ''} {', '.join(map(str, changed[:6]))}" if changed
                     else "the shot plan")
            add("missing", "medium", changed[0] if changed else None, None,
                f"The film was rendered before {where} changed, so it doesn't show the latest version.",
                "Render the film again (only the render is charged).", "edit")


def _camera(scenes, units, shots, add):
    typed = [u for u in units if u["type"]]
    jumps = [(a, b) for a, b in zip(units, units[1:])
             if a["scene"] != b["scene"] and a["type"] and a["type"] == b["type"] and a["names"] is not None
             and b["names"] is not None and a["names"] == b["names"] and a["names"]]
    if len(jumps) > 3:          # one item, not dozens
        add("camera", "medium", jumps[0][1]["scene"], None,
            f"{len(jumps)} cuts between scenes go from one {_nice(jumps[0][0]['type'])} to the same size of the same "
            f"people (into scenes {', '.join(str(b['scene']) for _a, b in jumps[:8])}"
            f"{'…' if len(jumps) > 8 else ''}): the cuts feel like jumps.",
            "Open those scenes on a different shot size.", "edit")
    else:
        for a, b in jumps:
            add("camera", "low", b["scene"], b["shot"],
                f"Scene {a['scene']} ends and scene {b['scene']} starts on the same {_nice(a['type'])} of the same "
                "people: the cut feels like a jump.", f"Open scene {b['scene']} on a different shot size.", "edit")
    openings = [(u["scene"], u["type"]) for u in units if u["shot"] in (1, None) and u["type"]]
    run = 1
    for k in range(1, len(openings) + 1):
        if k < len(openings) and openings[k][1] == openings[k - 1][1]:
            run += 1
            continue
        if run >= SAME_OPENING_RUN:
            first = openings[k - run][0]
            add("camera", "medium", first, None,
                f"Scenes {first} to {openings[k - 1][0]} all open on {_a(_nice(openings[k - 1][1]))}.",
                "Vary how scenes open: a close detail, a reaction, the middle of an action.", "edit")
        run = 1
    if len(typed) >= 8:
        kind, n = Counter(u["type"] for u in typed).most_common(1)[0]
        if n / len(typed) > TYPE_SHARE_MAX:
            add("camera", "medium", None, None,
                f"{round(100 * n / len(typed))}% of the shots are {_nice(kind)}s.",
                "Mix in other sizes where the moment allows: wides to show the place, close-ups for feelings.", "edit")
    moved = [u for u in units if u["move"]]
    if len(moved) >= 8:
        move, n = Counter(u["move"] for u in moved).most_common(1)[0]
        if n / len(moved) > MOVE_SHARE_MAX:
            add("camera", "low", None, None, f"{round(100 * n / len(moved))}% of the shots use the same camera move "
                f"({move}).", "Let the camera move with the story: still for tension, moving for action.", "edit")
    if len(scenes) >= 5:
        single = sum(1 for s in scenes if not shots.get(s["id"]))
        if single / len(scenes) > SINGLE_SHOT_SHARE_MAX:
            add("camera", "medium", None, None,
                f"{single} of {len(scenes)} scenes are one single shot: the film can feel like a slideshow.",
                "Let the director plan shots for the longer and talking scenes (more shots cost more; the price is "
                "shown first).", "edit")


def _pacing(project, scenes, units, shots, render, add):
    planned = float(project.get("target_seconds") or 0)
    actual = float((render or {}).get("duration_seconds") or 0) or sum(float(s.get("duration_seconds") or 0) for s in scenes)
    if planned and abs(actual - planned) > planned * LENGTH_TOLERANCE:
        add("pacing", "low", None, None,
            f"The film runs {actual:.0f} s; {planned:.0f} s was planned.",
            "Trim or extend scenes, or change the planned length.", "edit")
    lengths = [u["seconds"] for u in units if u["seconds"] > 0]
    if len(lengths) >= 6:
        mean = sum(lengths) / len(lengths)
        spread = (sum((x - mean) ** 2 for x in lengths) / len(lengths)) ** 0.5
        if mean and spread / mean < 0.15:
            add("pacing", "low", None, None, f"Almost every shot lasts about {mean:.1f} s: the rhythm never changes.",
                "Hold emotional moments longer and cut action and reactions shorter.", "edit")
    for i, s in enumerate(scenes, start=1):
        lines = _lines(s.get("dialogue"))
        secs = float(s.get("duration_seconds") or 0)
        if not shots.get(s["id"]) and not lines and s.get("mode") != "video" and secs >= STATIC_SECONDS:
            add("pacing", "medium", i, None, f"Scene {i} holds one still picture for {secs:.0f} s with no lines.",
                "Shorten it, plan more shots, or make it a video clip.", "edit")
        words = sum(len(d["text"].split()) for d in lines)
        if lines and secs:
            if words / secs > WORDS_PER_SECOND_MAX * 1.5:
                add("pacing", "high", i, None,
                    f"Scene {i} has {words} words in {secs:.0f} s: the lines can't fit.",
                    "Make the scene longer or cut lines.", "edit")
            elif words / secs > WORDS_PER_SECOND_MAX:
                add("pacing", "medium", i, None,
                    f"Scene {i} is rushed: {words} words in {secs:.0f} s.",
                    "Give it a few more seconds, or trim a line.", "edit")


def _screen_time(scenes, units, cast, presence, add):
    names = {_first(c.get("name")): c.get("name") for c in cast if c.get("name")}
    speaks = Counter()
    for s in scenes:
        for d in _lines(s.get("dialogue")):
            speaks[_first(d.get("character"))] += 1
    seen = Counter()
    for u in units:
        if u["names"] is not None:
            seen.update(n for n in u["names"] if n in names)
        else:                     # a single-shot scene shows whoever the scene puts in frame
            seen.update(_first(n) for n in presence.get(u["scene_id"], []) if _first(n) in names)
    for i, s in enumerate(scenes, start=1):
        rows = [u for u in units if u["scene_id"] == s["id"] and u["names"] is not None]
        if not rows:
            continue
        shown = set().union(*(u["names"] for u in rows))
        for n in presence.get(s["id"], []):
            f = _first(n)
            if f in names and f not in shown:
                add("screen_time", "medium", i, None,
                    f"Scene {i}: {names[f]} is in the scene but in none of its shots.",
                    f"Put {names[f]} in a shot, or take them out of the scene.", "edit")
    for f, full in names.items():
        if speaks[f] and not seen[f]:
            add("screen_time", "medium", None, None,
                f"{full} speaks {speaks[f]} line{'s' if speaks[f] != 1 else ''} but is never seen.",
                f"Show {full} at least once, unless they're meant to be only a voice.", "edit")
    if speaks:
        lead = speaks.most_common(1)[0][0]
        if lead in names:
            more = [names[f] for f in names if f != lead and seen[f] > seen[lead] and speaks[f] < speaks[lead]]
            if more and seen[lead] < max(seen[f] for f in names if f != lead) * 0.5:
                add("screen_time", "low", None, None,
                    f"{names[lead]} has the most lines but is seen in fewer shots than {more[0]}.",
                    f"Give {names[lead]} more coverage: singles when they speak, reactions when others do.", "edit")
    total = sum(seen.values()) or 1
    return {names[f]: {"shots": seen[f], "lines": speaks[f], "share": round(seen[f] / total, 2)} for f in names}


def _continuity(issues, audit, shot_checks, units, add):
    for x in issues or []:
        add("continuity", "medium" if x.get("severity") == "fix" else "low", x.get("scene"), None,
            x.get("message") or "A continuity problem.", x.get("fix") or "Check the scene.", "edit")
    for p in (audit or {}).get("problems") or []:
        add("continuity", p.get("severity") if p.get("severity") in _SEV else "medium", p.get("scene"), None,
            f"Scene {p.get('scene')}: expected {p.get('expected') or '?'}, the picture shows {p.get('seen') or '?'}.",
            p.get("fix") or "Correct the picture.", "free_fix" if p.get("severity") in ("high", "medium") else "remake")
    for u in units:
        rep = (shot_checks or {}).get(u["shot_id"]) if u["shot_id"] else None
        if rep and rep.get("status") in ("failed", "attention"):
            bad = [c for c in rep.get("checks") or [] if not c.get("passed")]
            what = ", ".join((c.get("label") or c.get("check") or "").lower() for c in bad[:3]) or "the picture"
            add("continuity", "high" if rep["status"] == "failed" else "medium", u["scene"], u["shot"],
                f"{_where(u['scene'], u['shot'])} failed VIYO's picture check: {what}.",
                "Report it as a defect: it's remade free once if the check confirms it.", "free_fix")
    unchecked = [u for u in units if u["made"] and u["shot_id"] and
                 ((shot_checks or {}).get(u["shot_id"]) or {}).get("status") in (None, "unchecked", "outdated")]
    return {"scene_checks": {k: (audit or {}).get(k) for k in ("checked", "unchecked", "outdated", "total")},
            "shots_not_checked": len(unchecked)}


def _ending(project, scenes, units, cast, presence, add):
    if not scenes:
        return
    n = len(scenes)
    last = scenes[-1]
    tail = [u for u in units if u["scene"] == n]
    if not any(u["made"] for u in tail):
        add("ending", "high", n, None, "The last scene isn't made, so the film has no ending.",
            "Make the last scene.", "remake")
    secs = sum(u["seconds"] for u in tail)
    if 0 < secs < MIN_ENDING_SECONDS:
        add("ending", "medium", n, None, f"The last scene lasts {secs:.1f} s: the ending goes by too fast.",
            "Hold the final moment longer.", "edit")
    speaks = Counter(_first(d.get("character")) for s in scenes for d in _lines(s.get("dialogue")))
    names = {_first(c.get("name")): c.get("name") for c in cast if c.get("name")}
    if speaks and len(scenes) >= 3:
        lead = speaks.most_common(1)[0][0]
        in_end = {_first(x) for s in scenes[-2:] for x in presence.get(s["id"], [])}
        if lead in names and lead not in in_end:
            add("ending", "low", n, None, f"{names[lead]}, who drives the story, isn't in the last two scenes.",
                "Check that the ending resolves (or deliberately leaves open) their story.", "story")
    ev = (project.get("story_bible") or {}).get("evaluation") or {}
    cat = (ev.get("categories") or {}).get("ending_hook") or {}
    if cat.get("status") == "verified" and isinstance(cat.get("score"), (int, float)):
        if cat["score"] < 8:
            add("ending", "medium", n, None,
                f"The story check scored the ending {cat['score']}/10: {cat.get('reason') or 'below the bar'}.",
                "Strengthen the last scene in the screenplay, then remake only that scene.", "story")
        return {"story_check_score": cat["score"]}
    return {"story_check_score": None, "note": "No verified story-check score for the ending yet."}


# ------------------------------------------------------------------ the report

def build(project: dict, scenes: list[dict], shots: dict, cast: list[dict], presence: dict,
          segments: Optional[list] = None, render: Optional[dict] = None, continuity_issues: Optional[list] = None,
          audit: Optional[dict] = None, shot_checks: Optional[dict] = None) -> dict:
    """scenes: ordered; shots: {scene id: ordered shot rows} for directed scenes only; presence: {scene id: names the
    scene puts in frame}; segments: the latest render's fingerprints (None: not rendered with fingerprints)."""
    problems = []

    def add(cat, sev, scene, shot, problem, fix, route):
        problems.append({"category": cat, "severity": sev, "scene": scene, "shot": shot, "problem": problem,
                         "fix": fix, "route": route, "route_label": ROUTES[route]})
    units = _units(scenes, shots)
    _duplicates(units, segments, add)
    _missing(scenes, units, shots, render, segments, add)
    _camera(scenes, units, shots, add)
    _pacing(project, scenes, units, shots, render, add)
    screen = _screen_time(scenes, units, cast, presence, add)
    checks = _continuity(continuity_issues, audit, shot_checks, units, add)
    ending = _ending(project, scenes, units, cast, presence, add) or {}
    problems.sort(key=lambda p: (_SEV[p["severity"]], p["scene"] or 0, p["shot"] or 0))
    typed = [u["type"] for u in units if u["type"]]
    lengths = [u["seconds"] for u in units if u["seconds"] > 0]
    notes = []
    if not render:
        notes.append("The film hasn't been rendered yet: this reviews the plan and what's made so far.")
    if segments is None:
        notes.append("Frame comparison runs when the film is rendered; until then only reused files are found.")
    if checks["shots_not_checked"]:
        notes.append(f"{checks['shots_not_checked']} shot picture(s) haven't been checked against the references.")
    categories = []
    for c in CATEGORIES:
        mine = [p for p in problems if p["category"] == c]
        worst = min((_SEV[p["severity"]] for p in mine), default=None)
        status = "ok" if worst is None else "problem" if worst == 0 else "attention"
        if c == "duplicates" and segments is None and status == "ok":
            status = "not_checked"
        categories.append({"key": c, "label": LABELS[c], "status": status, "count": len(mine)})
    return {
        "status": "problems" if any(p["severity"] == "high" for p in problems) else
                  "attention" if problems else "ok",
        "summary": _summary(problems),
        "categories": categories, "problems": problems, "notes": notes,
        "render": {"version": render.get("version"), "duration_seconds": float(render.get("duration_seconds") or 0)}
        if render else None,
        "metrics": {"scenes": len(scenes), "pieces": len(units), "directed_scenes": sum(1 for s in scenes if shots.get(s["id"])),
                    "shot_types": dict(Counter(typed)), "distinct_shot_types": len(set(typed)),
                    "largest_type_share": round(Counter(typed).most_common(1)[0][1] / len(typed), 2) if typed else None,
                    "average_shot_seconds": round(sum(lengths) / len(lengths), 2) if lengths else None,
                    "duplicate_pairs": sum(1 for p in problems if p["category"] == "duplicates"),
                    "missing": sum(1 for p in problems if p["category"] == "missing"),
                    "fingerprinted": sum(1 for g in segments or [] if g.get("hash")),
                    "screen_time": screen, "ending": ending, **checks},
        "free": True,
        "note": "This report is free and changes nothing. Fixes happen only when you choose them; a paid remake shows "
                "its price first.",
    }


def _summary(problems: list) -> str:
    if not problems:
        return "No problems found in the checks this report runs."
    high = sum(1 for p in problems if p["severity"] == "high")
    cats = Counter(LABELS[p["category"]].lower() for p in problems)
    top = ", ".join(f"{k} ({v})" for k, v in cats.most_common(3))
    return (f"{len(problems)} thing{'s' if len(problems) != 1 else ''} to look at"
            + (f", {high} serious" if high else "") + f": {top}.")
