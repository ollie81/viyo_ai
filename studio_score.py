"""VIYO score: the film's music planned as cues that follow the story, not one track looped under everything.

The cue sheet (pure code, no AI call, no cost)
  Every scene gets a musical mood from what happens in it: action (studio_action), a strong emotional beat, the
  aftermath of action, the genre's own colour (tension for a thriller, tenderness for a romance...), and the finale.
  Consecutive scenes with the same mood share one cue; a new place or a time jump may start a new one. Some scenes
  are planned SILENT: a shock, grief, a revelation, or a scene that asks for stillness. Silence is a choice, not a gap.
  The number of cues is capped by the film's length (one per 12 s, MAX_CUES at most: $0.48), so the cost stays small
  and known. A planned silence is never merged away.

Character themes
  The two characters with the most lines get a theme described in words (an instrument and a shape, chosen from their
  personality), and every cue they appear in is told to carry it. Each cue is generated separately, so the same
  melody across cues is NOT guaranteed: the theme is a shared description, not a recorded motif.

Generation and cost
  Each cue is one original instrumental Lyria track (studio_music, $0.08). The quote holds for the cues still to be
  made; a cue whose instructions haven't changed is reused on the next render (no new charge). A cue that fails is
  left silent and isn't charged; the film is still made. The creator's own uploaded music always wins, and "music
  off" makes none.

ENABLED (env STUDIO_SCORE, default on) switches back to the single looped track.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Optional

ENABLED = os.environ.get("STUDIO_SCORE", "on").lower() not in ("off", "0", "false", "no")

MAX_CUES = 6
SECONDS_PER_CUE = 12        # at most one cue per 12 seconds of film (shorter feels choppy), and MAX_CUES

MOODS = {
    "action": ("driving percussion and low strings, urgent ostinato, rising energy", 5),
    "tension": ("sparse pulsing synth and low strings, held notes, unresolved harmony", 3),
    "dread": ("dark drones, distant metallic textures, very slow, unsettling", 3),
    "mystery": ("muted piano and soft pizzicato, questioning motifs, restrained", 2),
    "tender": ("warm strings and soft piano, gentle, intimate", 2),
    "emotional": ("expressive solo strings over sustained chords, swelling then receding", 4),
    "aftermath": ("slow, exhausted, a single instrument over a soft pad, breathing room", 2),
    "light": ("playful pizzicato and light percussion, bouncy, bright", 2),
    "reflective": ("soft piano and ambient pads, unhurried, open", 2),
    "finale": ("the main theme returns, fuller, resolving (or suspended on an open chord for a cliffhanger)", 4),
}
GENRE_MOOD = {"thriller": "tension", "spy_thriller": "tension", "survival_thriller": "tension", "mystery": "mystery",
              "horror": "dread", "romance": "tender", "comedy": "light", "family": "reflective", "drama": "reflective",
              "revenge": "tension", "action": "tension", "adventure": "reflective", "superhero": "tension",
              "martial_arts": "tension", "science_fiction": "mystery"}
_SILENT_WORDS = re.compile(r"\b(silence|silent|stillness|frozen in shock|says nothing|without a word|holds? (his|her|their) breath)\b", re.I)
_SHOCK = re.compile(r"\b(shocked|stunned|devastated|grief|sobbing|heartbroken|horrified|numb)\b", re.I)
_THEME_SHAPES = [
    (r"warm|kind|gentle|loving|caring", "a warm solo cello phrase that rises gently"),
    (r"brave|bold|determined|fierce|strong", "a bold brass call of three rising notes"),
    (r"mysterious|secret|quiet|reserved|cold", "a low, sparse piano figure that never quite resolves"),
    (r"young|playful|curious|hopeful|bright", "a light flute or guitar melody that skips upward"),
    (r"serious|protective|stern|tired|old", "a steady low-string line that moves step by step"),
    (r"angry|cruel|dangerous|ruthless", "a dark, repeated two-note figure in the low brass"),
]


def _lines(scene):
    return [d for d in scene.get("dialogue") or [] if (d.get("text") or "").strip()]


def scene_mood(scene: dict, project: dict, after_action: bool, last: bool) -> str:
    import studio_action
    import studio_look
    text = " ".join([scene.get("visual") or ""] + [f"{d.get('emotion') or ''} {d.get('text') or ''}" for d in _lines(scene)])
    if _SILENT_WORDS.search(scene.get("visual") or "") or (_SHOCK.search(text) and len(_lines(scene)) <= 2):
        return "silence"
    if studio_action.is_action(scene, project) or (not studio_action.ENABLED and studio_action.action_score(scene) >= 2):
        return "action"
    if last:
        return "finale"
    if after_action:
        return "aftermath"
    import studio_shots
    if studio_shots.STRONG.search(text):
        return "emotional"
    return GENRE_MOOD.get(studio_look.genre_key(project.get("genre") or "", project.get("logline") or ""), "reflective")


def themes(cast: list[dict], scenes: list[dict]) -> dict:
    """Up to two main characters (most lines) -> a theme described in words."""
    count = Counter()
    for s in scenes:
        for d in _lines(s):
            count[(d.get("character") or "").split(" ")[0].lower()] += 1
    out = {}
    for c in sorted(cast, key=lambda c: -count[(c.get("name") or "").split(" ")[0].lower()])[:2]:
        if not count[(c.get("name") or "").split(" ")[0].lower()]:
            continue
        p = f"{c.get('personality') or ''} {c.get('description') or ''}".lower()
        shape = next((sh for pat, sh in _THEME_SHAPES if re.search(pat, p)), "a simple, memorable four-note melody")
        out[c["name"]] = shape
    return out


def sheet(project: dict, scenes: list[dict], cast: list[dict], present: Optional[dict] = None) -> list[dict]:
    """The cue sheet: [{index, mood, scenes: [ids], positions, seconds, intensity, characters, silent}]."""
    if not scenes:
        return []
    import studio_action
    ordered = sorted(scenes, key=lambda s: s.get("position") or 0)
    moods, prev_action = [], False
    for k, s in enumerate(ordered):
        m = scene_mood(s, project, prev_action, k == len(ordered) - 1)
        moods.append(m)
        prev_action = m == "action"
    cues = []
    for s, m in zip(ordered, moods):
        t = (s.get("continuity") or {}).get("transition")
        if cues and cues[-1]["mood"] == m and t not in ("time_jump", "flashback"):
            cues[-1]["scenes"].append(s)
        else:
            cues.append({"mood": m, "scenes": [s]})
    total = sum(float(s.get("duration_seconds") or 0) for s in ordered)
    limit = max(1, min(MAX_CUES, math.ceil(total / SECONDS_PER_CUE)))
    def secs(c):
        return sum(float(s.get("duration_seconds") or 0) for s in c["scenes"])
    while sum(1 for c in cues if c["mood"] != "silence") > limit:
        # Merge the shortest music cue with a music cue right next to it (the more intense mood wins). A planned
        # silence is never merged away; if no two music cues touch, the cap can't be met and we stop.
        pairs = [(i, i + 1) for i in range(len(cues) - 1)
                 if cues[i]["mood"] != "silence" and cues[i + 1]["mood"] != "silence"]
        if not pairs:
            break
        a, b = min(pairs, key=lambda ab: min(secs(cues[ab[0]]), secs(cues[ab[1]])))
        mood = max((cues[a]["mood"], cues[b]["mood"]), key=lambda m: MOODS[m][1])
        cues = cues[:a] + [{"mood": mood, "scenes": cues[a]["scenes"] + cues[b]["scenes"]}] + cues[b + 1:]
    th = themes(cast, ordered)
    out = []
    for n, c in enumerate(cues):
        who = [] if c["mood"] == "silence" else \
            sorted({x for s in c["scenes"] for x in (present or {}).get(s["id"], [])} & set(th))
        out.append({"index": n, "mood": c["mood"], "silent": c["mood"] == "silence",
                    "scenes": [s["id"] for s in c["scenes"]], "positions": [s.get("position", 0) + 1 for s in c["scenes"]],
                    "seconds": round(sum(float(s.get("duration_seconds") or 0) for s in c["scenes"]), 2),
                    "intensity": 0 if c["mood"] == "silence" else MOODS[c["mood"]][1], "characters": who,
                    "themes": {k: th[k] for k in who}})
    return out


def cue_prompt(project: dict, cue: dict) -> str:
    """The Lyria prompt for one cue (studio_music adds the shared rules and the story world's instruments)."""
    style, _i = MOODS[cue["mood"]]
    themes_text = " ".join(f"{name}'s theme: {shape}; weave it in when it fits." for name, shape in cue["themes"].items())
    return (f"This is cue {cue['index'] + 1} of the film's score, under scenes {', '.join(map(str, cue['positions']))} "
            f"(about {round(cue['seconds'])} seconds). Mood: {cue['mood']}. Intensity {cue['intensity']} of 5. "
            f"Sound: {style}. {themes_text}").strip()


def cue_key(project: dict, cue: dict) -> str:
    """What the cue was made from: the same key means the saved audio can be reused."""
    return hashlib.sha256((cue_prompt(project, cue) + "|" + (project.get("genre") or "") + "|"
                           + (project.get("logline") or "")).encode()).hexdigest()[:16]


def to_make(project: dict, cues: list[dict]) -> list[dict]:
    """The music cues whose audio isn't saved yet (what the render will generate and charge)."""
    saved = {c.get("key"): c for c in ((project.get("story_bible") or {}).get("score") or {}).get("cues") or []
             if c.get("path")}
    return [c for c in cues if not c["silent"] and cue_key(project, c) not in saved]


def timeline(durations: list[float], dissolves: list[float]) -> list[tuple[float, float]]:
    """(start, end) of each scene in the finished film, with dissolve overlaps taken out."""
    out, t = [], 0.0
    for k, d in enumerate(durations):
        start = max(0.0, t - (dissolves[k] if k else 0.0))
        out.append((start, start + d))
        t = start + d
    return out
