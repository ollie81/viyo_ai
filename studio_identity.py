"""Viyo Studio's Character Consistency Engine.

A character is created once and stays the same person in every scene:

* identity: a permanent, structured profile (basic, face, body). Scene
  text never changes it; only the creator does, while it is unlocked.
* reference sheet: seven views of the SAME person. The full-body front view
  is made first and becomes the anchor; every other view is generated from
  the anchor image ("this exact person, turned to the left"), never as an
  unrelated new portrait. Each view is approved or redone on its own.
* outfits: what the character wears in part of the story, each with its own
  picture made from the anchor. A scene names the outfit, or keeps the one
  from the scene before, so clothes stay the same across connected scenes.
* prompt builder: every scene picture and video gets the permanent identity
  first, then the outfit, then the scene's action, camera and lighting, and
  the reference views that suit the camera angle.

Pure functions only: the worker and API call these.
"""
from __future__ import annotations

import re
from typing import Optional

# ------------------------------------------------------------------ identity

IDENTITY_FIELDS: dict[str, list[tuple[str, str]]] = {
    "basic": [
        ("age", "Age or age range"), ("height_cm", "Height (cm)"), ("build", "Body build and proportions"),
        ("skin_tone", "Skin tone"), ("posture", "Posture"), ("distinguishing", "Distinguishing features"),
    ],
    "face": [
        ("face_shape", "Face shape"), ("eyes", "Eyes: shape, colour, spacing"), ("eyebrows", "Eyebrows"),
        ("nose", "Nose"), ("lips", "Lips"), ("jaw_chin", "Jawline and chin"), ("cheekbones", "Cheekbones"),
        ("ears", "Ears"), ("hairline", "Hairline"), ("hair", "Hair: colour, texture, style"),
        ("facial_hair", "Facial hair"), ("marks", "Freckles, scars, moles, other marks"),
    ],
    "body": [
        ("proportions", "Body proportions"), ("shoulders", "Shoulders"), ("physique", "General physique"),
        ("neck_limbs", "Neck and limb proportions"), ("hands", "Hands"), ("other", "Other distinguishing features"),
    ],
}
SECTION_LABEL = {"basic": "Basic identity", "face": "Face", "body": "Body"}
TEXT_MAX = 200


def _field_schema(key: str) -> dict:
    if key == "height_cm":
        return {"type": "integer", "description": "Height in centimetres, e.g. 168. A child is shorter."}
    return {"type": "string"}


IDENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        **{sec: {"type": "object", "properties": {k: _field_schema(k) for k, _ in fields},
                 "required": [k for k, _ in fields if k not in ("facial_hair", "marks", "distinguishing", "other")]}
           for sec, fields in IDENTITY_FIELDS.items()},
        "default_outfit": {"type": "string", "description": "The clothes and accessories they usually wear in this story."},
    },
    "required": ["basic", "face", "body", "default_outfit"],
}


def identity_prompt(name: str, description: str, context: str = "") -> str:
    return f"""You are the casting and costume department of a film studio. Turn this character into a precise,
permanent identity profile so artists can draw them exactly the same way in every shot, from every angle.

Character: {name}
What the story says about them: {description or '(nothing yet: invent a fitting, specific look)'}
{context}
Rules:
- Be concrete and visual (e.g. "almond-shaped dark-brown eyes, set slightly wide", not "nice eyes").
- Keep everything the story already says. Fill the gaps with choices that suit the character.
- Height in centimetres must suit their age and build.
- Use "none" for things they do not have (e.g. facial hair, scars).
- Fictional person only: never describe a real or famous person.
- default_outfit: the clothes and accessories they wear in most of the story, not their body."""


def clean_identity(raw) -> dict:
    """Only known fields, short strings, a sane height."""
    out: dict = {}
    raw = raw if isinstance(raw, dict) else {}
    for sec, fields in IDENTITY_FIELDS.items():
        src = raw.get(sec) if isinstance(raw.get(sec), dict) else {}
        clean = {}
        for k, _ in fields:
            v = src.get(k)
            if k == "height_cm":
                try:
                    h = int(float(v))
                except (TypeError, ValueError):
                    continue
                if 40 <= h <= 250:
                    clean[k] = h
            elif isinstance(v, (str, int, float)) and str(v).strip():
                clean[k] = re.sub(r"\s+", " ", str(v)).strip()[:TEXT_MAX]
        out[sec] = clean
    outfit = raw.get("default_outfit")
    if isinstance(outfit, str) and outfit.strip():
        out["default_outfit"] = outfit.strip()[:800]
    return out


def merge_identity(current: dict, patch: dict) -> dict:
    """The creator's edit on top of the saved identity (empty value = clear)."""
    merged = {sec: dict((current or {}).get(sec) or {}) for sec in IDENTITY_FIELDS}
    for sec in IDENTITY_FIELDS:
        for k, v in ((patch or {}).get(sec) or {}).items():
            if v in (None, ""):
                merged[sec].pop(k, None)
            else:
                merged[sec][k] = v
    if "default_outfit" in (patch or {}):
        merged["default_outfit"] = patch["default_outfit"] or ""
    elif (current or {}).get("default_outfit"):
        merged["default_outfit"] = current["default_outfit"]
    return clean_identity(merged)


def fallback_identity(description: str, age_label: Optional[str] = None) -> dict:
    """Without a model call: the description kept as the distinguishing
    features, so the prompt builder still has something permanent."""
    basic = {"distinguishing": description[:TEXT_MAX]} if description.strip() else {}
    if age_label:
        basic["age"] = age_label
    return clean_identity({"basic": basic, "face": {}, "body": {}})


def has_identity(identity: Optional[dict]) -> bool:
    identity = identity or {}
    return any(identity.get(sec) for sec in IDENTITY_FIELDS)


def identity_text(name: str, identity: Optional[dict], description: str = "") -> str:
    """The permanent identity as one compact block for image and video prompts."""
    identity = identity or {}
    if not has_identity(identity):
        return f"{name}: {description}".strip().rstrip(":") + "."
    parts = []
    for sec, fields in IDENTITY_FIELDS.items():
        vals = identity.get(sec) or {}
        bits = []
        for k, label in fields:
            v = vals.get(k)
            if v in (None, "") or str(v).lower() == "none":
                continue
            bits.append(f"{label.split(':')[0].split(' (')[0].lower()} {v}{' cm' if k == 'height_cm' else ''}")
        if bits:
            parts.append(f"{SECTION_LABEL[sec]}: " + "; ".join(bits))
    return f"{name}. " + ". ".join(parts) + "."


# ------------------------------------------------------------ reference sheet

VIEWS = ("front_full", "face_closeup", "left_profile", "right_profile", "back_full", "expressions", "wardrobe")
VIEW_LABEL = {
    "front_full": "Front, full body", "face_closeup": "Face close-up", "left_profile": "Left profile",
    "right_profile": "Right profile", "back_full": "Back, full body", "expressions": "Expressions",
    "wardrobe": "Clothes and accessories",
}
ANCHOR = "front_full"
_VIEW_SHOT = {
    "front_full": "Full-body shot, head to feet, standing straight and facing the camera, arms relaxed at the sides.",
    "face_closeup": "Close-up of the face and shoulders, facing the camera, neutral expression, every facial detail sharp.",
    "left_profile": "Full-body shot in exact left side profile (the camera sees the character's left side), standing straight.",
    "right_profile": "Full-body shot in exact right side profile (the camera sees the character's right side), standing straight.",
    "back_full": "Full-body shot from directly behind, head to feet; the back of the head, hair and clothes clearly visible.",
    "expressions": "A 2x2 grid of head-and-shoulders close-ups of the same face: happy, angry, sad, afraid. Same lighting in each.",
    "wardrobe": "Full-body front shot that shows the whole outfit and every accessory clearly: fabric, colours, shoes, jewellery.",
}


def view_prompt(view: str, name: str, identity: Optional[dict], description: str, style: str,
                outfit: str = "", from_anchor: bool = True) -> str:
    who = identity_text(name, identity, description)
    clothes = outfit or (identity or {}).get("default_outfit") or ""
    same = ("Reference image 1 is the approved front view of this character. Draw THE SAME PERSON: identical face, "
            "hair, skin tone, height, body proportions and marks. Only the camera angle changes. "
            if from_anchor else "This is the first, defining picture of this character. ")
    clothes_line = f"Clothes and accessories: {clothes}. " if clothes else ""
    if view == "expressions":
        clothes_line = ""
    return (f"Character reference sheet view: {VIEW_LABEL[view]}. {same}\n"
            f"PERMANENT IDENTITY: {who}\n{clothes_line}{_VIEW_SHOT[view]} "
            f"Plain light-grey studio background, soft even light, no other people. "
            f"Style: {style or 'cinematic, realistic'}. No text, labels, captions or watermarks.")


def outfit_prompt(name: str, identity: Optional[dict], description: str, style: str, outfit_name: str,
                  outfit_description: str, parts: Optional[dict] = None) -> str:
    wear = outfit_text({"name": outfit_name, "description": outfit_description, "parts": parts or {}}, outfit_name)
    return (f"Outfit reference for {name}: \"{outfit_name}\". Reference image 1 is the approved front view of this "
            "character. Draw THE SAME PERSON (identical face, hair, skin tone, height, proportions and marks); "
            f"only the clothes, shoes, accessories and makeup change. Now wearing: {wear}.\nPERMANENT IDENTITY: "
            f"{identity_text(name, identity, description)}\nFull-body front shot, head to feet, the whole outfit "
            "and every accessory visible. Plain light-grey studio background, soft even light, no other people. "
            f"Style: {style or 'cinematic, realistic'}. No text, labels, captions or watermarks.")


def view_order(requested: list[str]) -> list[str]:
    """The anchor first, then the sheet order, no repeats."""
    wanted = {v for v in requested if v in VIEWS}
    return [v for v in VIEWS if v in wanted]


def approved(views: dict, view: str) -> bool:
    return ((views or {}).get(view) or {}).get("status") == "approved"


def sheet_status(views: dict) -> dict:
    """What the Cast tab shows about a sheet."""
    views = views or {}
    anchor = (views.get(ANCHOR) or {}).get("path")
    out = {}
    for v in VIEWS:
        e = views.get(v) or {}
        if not e.get("path"):
            out[v] = "missing"
        elif v != ANCHOR and anchor and e.get("anchor") and e["anchor"] != anchor:
            out[v] = "outdated"     # made from an older front view: kept, but may not match
        else:
            out[v] = e.get("status") or "draft"
    return out


def ready_to_lock(views: dict) -> Optional[str]:
    """None if the identity can be approved and locked, else why not."""
    st = sheet_status(views)
    if st[ANCHOR] == "missing":
        return "Create the reference sheet first."
    missing = [VIEW_LABEL[v] for v in ("front_full", "face_closeup") if st[v] == "missing"]
    if missing:
        return "Create these views first: " + ", ".join(missing) + "."
    outdated = [VIEW_LABEL[v] for v, s in st.items() if s == "outdated"]
    if outdated:
        return "These views were made from an older front view. Redo them first: " + ", ".join(outdated) + "."
    return None


# ------------------------------------------------------------ camera & refs

CAMERAS = ("auto", "close_up", "front", "side", "back", "full_body", "wide")
_CAMERA_WORDS = [
    ("back", r"\b(from behind|from the back|back to (the )?camera|walks away|walking away|over (his|her|their) shoulder)\b"),
    ("side", r"\b(profile|side view|from the side|side-on)\b"),
    ("close_up", r"\b(close[- ]?up|extreme close|cu\b|tight on|face fills)\b"),
    ("wide", r"\b(wide shot|establishing|aerial|long shot|crowd)\b"),
    ("full_body", r"\b(full[- ]body|head to toe|full shot)\b"),
]


def infer_camera(visual: str) -> str:
    text = (visual or "").lower()
    for cam, pattern in _CAMERA_WORDS:
        if re.search(pattern, text):
            return cam
    return "front"


def scene_camera(scene: dict) -> str:
    cam = scene.get("camera") or "auto"
    return infer_camera(scene.get("visual") or scene.get("heading") or "") if cam == "auto" else cam


_CAMERA_VIEWS = {
    "close_up": ["face_closeup", "front_full", "expressions"],
    "front": ["face_closeup", "front_full"],
    "side": ["face_closeup", "left_profile", "right_profile"],
    "back": ["back_full", "face_closeup"],
    "full_body": ["front_full", "face_closeup"],
    "wide": ["front_full", "face_closeup"],
}


def references_for(character: dict, camera: str, outfit: Optional[dict], limit: int) -> list[tuple[str, str]]:
    """(label, file path) of the best reference pictures for this shot.
    Approved views first; a sheet-less character falls back to its portrait."""
    if limit <= 0:
        return []
    name = character["name"]
    views = character.get("reference_views") or {}
    status = sheet_status(views)
    out: list[tuple[str, str]] = []

    def add(label, path):
        if path and path not in {p for _, p in out} and len(out) < limit:
            out.append((label, path))

    usable = [v for v in _CAMERA_VIEWS.get(camera, _CAMERA_VIEWS["front"]) if status.get(v) in ("approved", "draft")]
    usable.sort(key=lambda v: status[v] != "approved")
    if usable:
        add(f"{name}: {VIEW_LABEL[usable[0]].lower()} (keep this exact face, hair, skin and body)", views[usable[0]]["path"])
    sig = (outfit or {}).get("image_signature")
    if outfit and outfit.get("reference_path") and (not sig or sig == outfit_signature(outfit) or outfit.get("scene_only")):
        if outfit.get("scene_only"):
            add(f"{name} in their outfit (keep it, except this scene's changes described in the text)", outfit["reference_path"])
        else:
            add(f"{name} wearing the outfit for this scene (keep these exact clothes)", outfit["reference_path"])
    for v in usable[1:]:
        add(f"{name}: {VIEW_LABEL[v].lower()} (same person)", views[v]["path"])
    if not out and character.get("reference_path"):
        add(f"{name} (keep this exact face, hair and clothes)", character["reference_path"])
    return out


def split_limit(n_characters: int, total: int, per_character: int = 3) -> list[int]:
    """How many reference pictures each character gets (speakers first)."""
    if n_characters <= 0:
        return []
    base, extra = divmod(total, n_characters)
    return [min(per_character, base + (1 if i < extra else 0)) for i in range(n_characters)]


# ------------------------------------------------------------ outfits

OUTFIT_PARTS = ("clothing", "shoes", "accessories", "makeup")
PART_LABEL = {"clothing": "Clothing", "shoes": "Shoes", "accessories": "Accessories", "makeup": "Makeup"}


def clean_parts(raw) -> dict:
    """Only the four outfit parts, as short text. Face, hair colour, body
    and height are identity, never outfit."""
    raw = raw if isinstance(raw, dict) else {}
    return {k: re.sub(r"\s+", " ", str(raw[k])).strip()[:300] for k in OUTFIT_PARTS
            if isinstance(raw.get(k), (str, int, float)) and str(raw[k]).strip()}


def effective_outfit(outfit: Optional[dict], override: Optional[dict]) -> Optional[dict]:
    """The outfit worn in one scene: the chosen outfit with this scene's
    one-off changes on top, part by part."""
    override = clean_parts(override)
    if not outfit and not override:
        return outfit
    base = dict(outfit or {"name": "", "description": ""})
    parts = dict(clean_parts(base.get("parts")))
    parts.update(override)
    base["parts"] = parts
    base["scene_only"] = override
    return base


def outfit_text(outfit: Optional[dict], fallback: str = "") -> str:
    """What the character wears, part by part."""
    if not outfit:
        return fallback
    parts = clean_parts(outfit.get("parts"))
    bits = [f"{PART_LABEL[k].lower()}: {parts[k]}" for k in OUTFIT_PARTS if parts.get(k)]
    if outfit.get("description"):
        bits.insert(0, outfit["description"])
    text = "; ".join(bits) or outfit.get("name") or fallback
    if outfit.get("scene_only"):
        text += " (for this scene only: " + "; ".join(f"{PART_LABEL[k].lower()} {v}"
                                                      for k, v in outfit["scene_only"].items()) + ")"
    return text


def outfit_signature(outfit: dict) -> str:
    """Which parts a picture was made from: if any part changes later, the
    picture is out of date (the face and body are never re-made for it)."""
    import hashlib
    parts = clean_parts(outfit.get("parts"))
    key = "|".join([outfit.get("description") or ""] + [parts.get(k, "") for k in OUTFIT_PARTS])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


OUTFIT_SUGGEST_SCHEMA = {
    "type": "object",
    "properties": {"outfits": {"type": "array", "items": {"type": "object", "properties": {
        "character": {"type": "string", "description": "Exactly as in the character list."},
        "name": {"type": "string", "description": "Short outfit name, e.g. 'Hospital scrubs', 'Wedding', 'Nightwear'."},
        "clothing": {"type": "string"}, "shoes": {"type": "string"},
        "accessories": {"type": "string"}, "makeup": {"type": "string", "description": "Empty if none."},
        "scenes": {"type": "array", "items": {"type": "integer"}, "description": "Scene numbers (1 = first) it is worn in."},
        "reason": {"type": "string", "description": "One short sentence: why this outfit fits the story here."},
    }, "required": ["character", "name", "clothing", "scenes"]}}},
    "required": ["outfits"],
}


def outfit_suggest_prompt(script: str, cast: list[dict], scenes: list[dict], style: str) -> str:
    who = "\n".join(f"- {c['name']}: usual clothes: {(c.get('identity') or {}).get('default_outfit') or c.get('description') or '?'}"
                    for c in cast)
    shots = "\n".join(f"{i}. {s.get('heading') or ''}: {s.get('visual') or ''}" for i, s in enumerate(scenes, start=1))
    return f"""You are the costume designer of a film. Suggest the outfits each character needs, based on the story:
where they are, the time of day, the occasion, what happens to them (rain, a fight, a wedding, a hospital).

CHARACTERS:
{who}

SCENES:
{shots or '(no scene list yet)'}

SCREENPLAY:
{script[:12000]}

Rules:
- Only suggest a new outfit when the story gives a reason to change clothes. Most scenes keep the same outfit.
- Describe clothing, shoes, accessories and makeup separately and concretely (colours, fabrics, condition).
- Never change the character's face, hair colour, skin, body or height: only what they wear.
- List the scene numbers each outfit is worn in.
- Visual style: {style or 'cinematic, realistic'}. Write in English."""


def resolve_outfits(scenes: list[dict], outfits: list[dict], cast: list[dict], in_scene) -> dict:
    """{scene id: {character id: outfit row or None}}. A scene that names an
    outfit uses it; otherwise the character keeps the outfit they wore in the
    scene before (the same outfit period); otherwise their default outfit."""
    outfits = [o for o in outfits if (o.get("status") or "saved") == "saved"]
    by_id = {o["id"]: o for o in outfits}
    default = {}
    for o in sorted(outfits, key=lambda o: (not o.get("is_default"), o.get("position") or 0)):
        default.setdefault(o["character_id"], o)
    current: dict = {}
    out: dict = {}
    for s in sorted(scenes, key=lambda s: s.get("position") or 0):
        chosen = s.get("outfits") or {}
        row = {}
        for c in in_scene(s, cast):
            oid = chosen.get(c["id"])
            o = by_id.get(oid) if oid else None
            if o and o["character_id"] == c["id"]:
                current[c["id"]] = o
            row[c["id"]] = current.get(c["id"]) or default.get(c["id"])
        out[s["id"]] = row
    return out


# ------------------------------------------------------------ prompts

def cast_block(entries: list[tuple[dict, Optional[dict]]]) -> str:
    """Each character's permanent identity, then what they wear in this scene."""
    lines = []
    for c, outfit in entries:
        lines.append("- " + identity_text(c["name"], c.get("identity"), c.get("description") or ""))
        wear = outfit_text(outfit, (c.get("identity") or {}).get("default_outfit") or "")
        if wear:
            lines.append(f"  Wearing in this scene: {wear}.")
    return "\n".join(lines)


_CAMERA_TEXT = {
    "close_up": "Close-up shot.", "front": "Medium shot, characters facing the camera.",
    "side": "Side-on shot: characters seen in profile.", "back": "Shot from behind the characters.",
    "full_body": "Full-body shot, head to feet.", "wide": "Wide shot.",
}


def scene_prompt(action: str, entries: list[tuple[dict, Optional[dict]]], camera: str, style: str,
                 aspect_ratio: str, extra: str = "") -> str:
    """Permanent identity first and fenced off, so scene text can't change it."""
    block = cast_block(entries)
    frame = "Vertical 9:16 frame." if aspect_ratio == "9:16" else "Wide 16:9 frame."
    who = (f"CHARACTERS (permanent identity: never change their face, hair, skin tone, height, body proportions or "
           f"marks, whatever the scene says; clothes change only as listed):\n{block}\n\n" if block else "")
    return (f"{who}SCENE: {action}\nCAMERA: {_CAMERA_TEXT.get(camera, '')} {frame}\n"
            f"STYLE AND LIGHT: {style or 'cinematic, realistic, film still'}.\n{extra}"
            "If the scene text seems to describe a character differently, keep the permanent identity.").strip()
