"""VIYO look bible: one visual identity per production, the same in every picture and clip.

What a look is
  lens, depth of field, camera language, light, colour palette, contrast and grade, texture, framing, and what to
  avoid. It's built by code (no AI call, no cost) from the genre, the creator's own visual style and the story
  world's visual identity (its palette and textures), and the creator can edit any part of it.

When it applies
  A production keeps one look from its first picture to its last: the look is saved (frozen) on the production the
  first time a picture or clip is made with it, and every later picture and clip gets the same text. A production
  that already has pictures made before the look bible existed gets no look unless the creator saves one, so a
  half-made film is never given a new look half way through. An episode of a series uses the series' look.
  The scene's own time of day, weather and light (its continuity record) still apply: the look sets the style, not
  the hour.

ENABLED (env STUDIO_LOOK, default on) switches it off.

Honest limits
  The look is an instruction to the image and video models; they follow it more or less closely. Consistency is
  helped by giving every shot the identical text, but it isn't measured on real pictures here (that needs the live
  benchmark). Lens and depth of field are described in words; the models have no real lens.
"""
from __future__ import annotations

import os
import re
from typing import Optional

ENABLED = os.environ.get("STUDIO_LOOK", "on").lower() not in ("off", "0", "false", "no")

FIELDS = ("lens", "depth_of_field", "camera", "light", "palette", "contrast_and_grade", "texture", "framing", "avoid")
LABELS = {"lens": "Lens", "depth_of_field": "Depth of field", "camera": "Camera", "light": "Light",
          "palette": "Colour palette", "contrast_and_grade": "Contrast and grade", "texture": "Texture",
          "framing": "Framing", "avoid": "Avoid"}

_BASE = {"lens": "35mm-equivalent, natural perspective; 50-85mm feel for close-ups",
         "depth_of_field": "shallow on faces in close shots, deeper in wides so the place reads",
         "camera": "steady and motivated: it moves only when a character or the story moves",
         "light": "naturalistic, motivated by real sources in the scene (windows, lamps, the sky)",
         "palette": ["warm skin tones", "neutral mid-tones", "one accent colour that recurs"],
         "contrast_and_grade": "medium contrast, gentle highlight roll-off, true blacks",
         "texture": "fine film grain, clean but not plastic",
         "framing": "rule of thirds, eyes near the upper third, room to look into",
         "avoid": ["flat even lighting", "oversaturated colours", "a different look from shot to shot"]}

# Genre presets: only what differs from the base.
PRESETS = {
    "drama": {"light": "soft, motivated window light; faces modelled with gentle shadow",
              "palette": ["warm skin tones", "muted earth colours", "soft daylight blue"]},
    "romance": {"light": "soft, warm and flattering; golden hour where the story allows",
                "palette": ["warm gold", "blush pink", "soft cream"], "depth_of_field": "shallow, dreamy backgrounds",
                "contrast_and_grade": "low contrast, lifted shadows, warm grade"},
    "family": {"light": "warm practical light at home, natural daylight outside",
               "palette": ["warm wood tones", "faded fabric colours", "evening amber"]},
    "thriller": {"light": "low-key, hard side light, pools of light in darkness",
                 "palette": ["cold teal shadows", "sodium amber", "desaturated skin"],
                 "contrast_and_grade": "high contrast, deep blacks, teal-and-amber grade",
                 "camera": "controlled; slow creeping push-ins for tension, handheld only when danger breaks"},
    "mystery": {"light": "low-key with motivated practicals; faces half in shadow",
                "palette": ["deep green", "old brass", "fog grey"], "contrast_and_grade": "medium-high contrast, cool grade"},
    "revenge": {"light": "hard, directional light; strong shadows",
                "palette": ["blood red accent", "charcoal", "cold steel blue"],
                "contrast_and_grade": "high contrast, crushed blacks"},
    "horror": {"light": "very low-key; single sources; darkness at the frame edges",
               "palette": ["sickly green", "cold blue-black", "candle amber"],
               "contrast_and_grade": "high contrast, desaturated, deep shadows",
               "framing": "negative space where something could appear; off-centre subjects"},
    "comedy": {"light": "bright, even and warm", "palette": ["sunny yellow", "fresh green", "clean white"],
               "contrast_and_grade": "medium-low contrast, bright and saturated", "depth_of_field": "deeper, so reactions read"},
    "action": {"camera": "dynamic: tracking and handheld in the action, locked-off for impacts and reactions",
               "light": "hard, high-contrast light with strong rim light separating bodies from the background",
               "palette": ["steel blue", "fire orange", "gunmetal grey"],
               "contrast_and_grade": "high contrast, punchy teal-and-orange grade",
               "lens": "wide 24-28mm for movement and geography, long lenses for impacts"},
    "adventure": {"light": "big natural light: sun, sky and weather", "palette": ["sun-bleached sand", "jungle green", "sky blue"],
                  "lens": "wide for landscapes that dwarf the characters, normal for faces"},
    "superhero": {"light": "heroic rim and top light; powers glow as practical light sources",
                  "palette": ["bold primary accent", "night blue", "power glow colour"],
                  "contrast_and_grade": "high contrast, saturated accents", "framing": "low angles for heroes, symmetry for iconic moments"},
    "martial_arts": {"light": "clean, sculpting light that shows every movement",
                     "palette": ["dojo wood", "red lacquer", "ink black"],
                     "camera": "wide enough to see full bodies in the fights; still frames between them",
                     "framing": "full-body framing for technique, symmetry for duels"},
    "spy_thriller": {"light": "cool, controlled; neon and screen light at night",
                     "palette": ["steel grey", "cold cyan", "a single warm accent"],
                     "contrast_and_grade": "medium-high contrast, cool and clean",
                     "framing": "frames within frames: doorways, glass, reflections"},
    "survival_thriller": {"light": "harsh natural light; dusk and darkness are threats",
                          "palette": ["bleached earth", "storm grey", "frost white"],
                          "camera": "handheld and close, as if surviving with them",
                          "texture": "rough grain, weathered and dirty"},
    "science_fiction": {"light": "motivated by technology: screens, panels, hard white work light",
                        "palette": ["clinical white", "deep space blue", "signal amber"],
                        "contrast_and_grade": "clean, high dynamic range, cool grade",
                        "framing": "strong geometry and symmetry in built spaces"},
}


def genre_key(genre: str, logline: str = "") -> str:
    """The look preset that fits the genre (the writers' room's genre words, then drama)."""
    import studio_story
    text = f"{genre or ''} {logline or ''}".lower()
    for g, words in studio_story._GENRE_WORDS.items():
        if any(w in text for w in words) and g in PRESETS:
            return g
    return "drama"


def default_look(project: dict, world: Optional[dict] = None) -> dict:
    """A look built from the genre, the creator's visual style and the story world (free)."""
    key = genre_key(project.get("genre") or "", project.get("logline") or "")
    look = {**_BASE, **PRESETS.get(key, {})}
    look["palette"] = list(look["palette"])
    vis = (world or {}).get("visual") or {}
    if (vis.get("palette") or "").strip():                 # the world's own colours win over the genre's
        look["palette"] = [p.strip() for p in re.split(r"[;,]", vis["palette"]) if p.strip()][:5] or look["palette"]
    if (vis.get("textures") or "").strip():
        look["texture"] = f"{look['texture']}; world textures: {vis['textures'].strip()[:200]}"
    style = (project.get("visual_style") or "").strip()
    return {**look, "genre": key, "style": style[:400], "source": "default", "edited": []}


def clean(body: dict, current: dict) -> dict:
    """A creator's edit: only known fields, short text, lists for palette and avoid."""
    out = dict(current)
    edited = set(current.get("edited") or [])
    for k in FIELDS + ("style",):
        if k not in body:
            continue
        v = body[k]
        if k in ("palette", "avoid"):
            items = v if isinstance(v, list) else re.split(r"[;,\n]", str(v or ""))
            out[k] = [str(x).strip()[:80] for x in items if str(x).strip()][:8]
        else:
            out[k] = re.sub(r"\s+", " ", str(v or "")).strip()[:400 if k == "style" else 240]
        edited.add(k)
    out["edited"] = sorted(edited)
    out["source"] = "creator" if edited else out.get("source", "default")
    return out


def block(look: Optional[dict]) -> str:
    """The look as one fenced block for the image and video models: identical text for every shot."""
    if not ENABLED or not look:
        return ""
    parts = []
    if look.get("style"):
        parts.append(f"Creator's style: {look['style']}")
    for k in FIELDS:
        v = look.get(k)
        if v:
            parts.append(f"{LABELS[k]}: {', '.join(v) if isinstance(v, list) else v}")
    return ("LOOK OF THE FILM (identical in every picture and clip of this production; the scene's own time of day, "
            "weather and light still apply):\n" + "\n".join(f"- {p}" for p in parts) + "\n")


# ------------------------------------------------------------------ film-wide repetition (for the director)

def pattern(shots: list[dict]) -> tuple:
    return tuple(s.get("shot_type") or s.get("type") or "" for s in shots)


def recent_block(recent: list[tuple]) -> str:
    """Tells the director how the last scenes were filmed, so this one doesn't repeat them."""
    recent = [p for p in recent if p]
    if not ENABLED or not recent:
        return ""
    lines = "\n".join(f"- {' -> '.join(t.replace('_', ' ') for t in p)}" for p in recent)
    return (f"\nHOW THE LAST SCENES WERE FILMED (don't repeat the same pattern or open the same way again unless the "
            f"story asks for it):\n{lines}\n- If this scene starts in a new place or after a time jump, you may frame "
            "its first shot to echo the previous scene's last shot (a match cut: the same shape, position or "
            "movement), when that serves the story.\n")


def repetition(shots: list[dict], recent: list[tuple]) -> list[dict]:
    """The plan repeats the last scenes: the same shot pattern as both of the last two, or the same opening as the
    last three."""
    recent = [p for p in recent if p]
    if not ENABLED or not shots:
        return []
    me = pattern(shots)
    out = []
    if len(recent) >= 2 and len(me) >= 2 and recent[-1] == me and recent[-2] == me:
        out.append({"shot": 0, "kind": "repetition",
                    "problem": "This scene is filmed exactly like the two before it (" +
                               " -> ".join(t.replace("_", " ") for t in me) + ").",
                    "fix": "Change the pattern: open differently, or change the order and sizes."})
    elif len(recent) >= 3 and all(p and p[0] == me[0] for p in recent[-3:]):
        out.append({"shot": 1, "kind": "repetition",
                    "problem": f"The last three scenes also open on {me[0].replace('_', ' ')}.",
                    "fix": "Open this scene another way: a detail, a reaction, the middle of an action."})
    return out
