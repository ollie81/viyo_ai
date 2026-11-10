"""Places and objects that stay the same in every scene.

The AI reads the script and lists the locations that recur (Vance mansion,
the rooftop) and the objects that matter or come back (a black leather bag,
a phone, keys, a revolver), with the scenes each appears in. Each gets one
reference picture. Every scene picture and video that uses a place or an
object is given that picture and its fixed description, so the mansion and
the bag look the same every time instead of being reinvented per scene.

A scene's place is the one the creator chose, else the one the AI assigned,
else the location whose name matches the scene heading. Its objects are the
chosen list, else the objects named in the scene.
"""
from __future__ import annotations

import re
from typing import Optional

ELEMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "locations": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Short, as the scene headings call it, e.g. 'Vance Mansion - Living Room'."},
            "description": {"type": "string", "description": "What it looks like: architecture, furniture, colours, materials, "
                                                            "era, light sources. Concrete and visual."},
            "time_of_day": {"type": "string"}, "mood": {"type": "string"},
            "scenes": {"type": "array", "items": {"type": "integer"}, "description": "Scene numbers (1 = first)."},
        }, "required": ["name", "description", "scenes"]}},
        "props": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string", "description": "e.g. 'Maya's black leather bag', 'the revolver', 'car keys'."},
            "description": {"type": "string", "description": "Exact look: colour, material, size, condition, details. "
                                                            "No brand names or logos."},
            "owner": {"type": "string", "description": "The character it belongs to, if any."},
            "scenes": {"type": "array", "items": {"type": "integer"}},
        }, "required": ["name", "description", "scenes"]}},
    },
    "required": ["locations", "props"],
}


def extract_prompt(script: str, scenes: list[dict], cast: list[dict], style: str) -> str:
    shots = "\n".join(f"{i}. {s.get('heading') or ''}: {s.get('visual') or ''}" for i, s in enumerate(scenes, start=1))
    who = ", ".join(c["name"] for c in cast) or "(none)"
    return f"""You are the production designer and props master of a film. From the screenplay and scene list, list:

1. LOCATIONS: every distinct place the story happens in (one entry per place; the same room seen twice is one
   location). Describe each so a set builder could rebuild it exactly: layout, furniture, colours, materials, era,
   light sources.
2. OBJECTS: every object that matters to the story or appears in more than one scene: things characters carry,
   use, give or fight over (a bag, a phone, keys, a letter, a weapon, a ring, a car). Describe the exact look so it
   is drawn the same every time. Skip ordinary background items.

Describe each object as it IS, on its own: shape, size, colour, material, permanent marks (a crack, a scratch, a
sticker). Never what happens to it in a scene: not where it lies (a puddle, a table, a pocket), not water, mud,
blood or weather on it, not what its screen shows, not who is calling. Describe each location as the empty set,
with no events or people in it.

For each, list the scene numbers it appears in. Only what the screenplay contains; no brand names or logos.
Visual style: {style or 'cinematic, realistic'}. Characters: {who}.

SCENES:
{shots or '(no scene list yet)'}

SCREENPLAY:
{script[:14000]}"""


def clean(text, n=800) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:n]


def picture_prompt(element: dict, style: str, aspect_ratio: str) -> str:
    d = element.get("details") or {}
    if element["kind"] == "location":
        frame = "Vertical 9:16" if aspect_ratio == "9:16" else "Wide 16:9"
        extra = "; ".join(x for x in (d.get("time_of_day"), d.get("mood")) if x)
        return (f"Location reference for a film: {element['name']}. {element.get('description') or ''}"
                f"{(' (' + extra + ')') if extra else ''}.\nEstablishing shot of the empty set, no people, every "
                f"important detail visible, {frame} frame, cinematic light. Style: {style or 'cinematic, realistic'}. "
                "No text, logos or watermarks.")
    owner = f" (belongs to {d['owner']})" if d.get("owner") else ""
    return (f"Prop reference photo for a film: {element['name']}{owner}. {element.get('description') or ''}.\n"
            "The object alone, centred, fully visible, on a plain neutral background, soft studio light, sharp detail. "
            "Show only the object itself: ignore any situation in the description (where it lies, water, mud, rain, "
            "what is happening). Any screen is dark and switched off. "
            f"Style: {style or 'cinematic, realistic'}. No people, no text, no logos or brand names.")


def _words(name: str) -> list[str]:
    """Distinctive words of a name, for matching it in scene text."""
    stop = {"the", "a", "an", "of", "and", "s", "int", "ext", "day", "night", "room", "living", "his", "her", "their"}
    return [w for w in re.findall(r"[a-z0-9']+", name.lower().replace("'s", "")) if len(w) > 2 and w not in stop]


def _mentions(text: str, name: str) -> bool:
    text = text.lower()
    if re.search(rf"\b{re.escape(name.lower())}\b", text):
        return True
    words = _words(name)
    return bool(words) and all(re.search(rf"\b{re.escape(w)}", text) for w in words)


def scene_location(scene: dict, elements: list[dict]) -> Optional[dict]:
    places = [e for e in elements if e["kind"] == "location"]
    if scene.get("location_id"):
        return next((e for e in places if e["id"] == scene["location_id"]), None)
    heading = scene.get("heading") or ""
    for e in places:
        if _mentions(heading, e["name"]):
            return e
    for e in places:
        if _mentions(scene.get("visual") or "", e["name"]):
            return e
    return None


def scene_props(scene: dict, elements: list[dict]) -> list[dict]:
    props = [e for e in elements if e["kind"] == "prop"]
    chosen = scene.get("props")
    if isinstance(chosen, list):
        return [e for e in props if e["id"] in chosen]
    text = " ".join([scene.get("visual") or ""] + [d.get("text") or "" for d in scene.get("dialogue") or []])
    return [e for e in props if _mentions(text, e["name"]) or
            any(_mentions(text, w) for w in _words(e["name"])[-1:])]


def world_block(location: Optional[dict], props: list[dict]) -> str:
    lines = []
    if location:
        d = location.get("details") or {}
        extra = "; ".join(x for x in (d.get("time_of_day"), d.get("mood")) if x)
        lines.append(f"LOCATION (the same place every time it appears: keep its look, materials and colours, but show only "
                     f"the part of it this shot would see; in a close-up it may be only a small background detail): "
                     f"{location['name']}. {location.get('description') or ''}{(' (' + extra + ')') if extra else ''}")
    if props:
        lines.append("OBJECTS (exactly the same in every scene: same colour, material and shape):\n" +
                     "\n".join(f"- {p['name']}: {p.get('description') or ''}" for p in props))
    return ("\n".join(lines) + "\n") if lines else ""


def reference_labels(location: Optional[dict], props: list[dict], limit: int) -> list[tuple[str, str]]:
    """(label, path) of the place's and objects' pictures, the place first."""
    out = []
    if location and location.get("reference_path") and len(out) < limit:
        out.append((f"the location, {location['name']} (how the place looks; frame the shot as the SCENE says, "
                    f"not like this picture)", location["reference_path"]))
    for p in props:
        if p.get("reference_path") and len(out) < limit:
            out.append((f"the object {p['name']} (keep it exactly like this)", p["reference_path"]))
    return out
