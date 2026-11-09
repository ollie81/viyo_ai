"""The small Viyo mark on Studio covers.

Image models draw logos badly (wrong shape, wrong colours, misspelt), so
the AI paints the poster and this adds the real Viyo mark afterwards: the
V logo and a small "VIYO" wordmark in one corner, with a soft shadow so it
reads on any picture without spoiling it.
"""
from __future__ import annotations

import io
import os
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
MARK_PATH = os.path.join(ROOT, "assets", "viyo_mark.png")
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")


def _font(size: int):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def stamp(image_bytes: bytes, corner: str = "top_right", opacity: float = 0.92) -> bytes:
    """The poster with the Viyo mark in a corner, as PNG bytes."""
    poster = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = poster.size
    mark = Image.open(MARK_PATH).convert("RGBA")
    mh = max(18, int(min(w, h) * 0.055))
    mark = mark.resize((int(mark.width * mh / mark.height), mh), Image.LANCZOS)
    font = _font(max(11, int(mh * 0.62)))
    text = "VIYO"
    tw = int(font.getlength(text))
    gap = max(4, mh // 5)
    bw, bh = mark.width + gap + tw, mh
    badge = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    badge.alpha_composite(mark, (0, 0))
    d = ImageDraw.Draw(badge)
    ascent, descent = font.getmetrics()
    d.text((mark.width + gap, (bh - ascent - descent) // 2 + descent // 3), text, font=font, fill=(255, 255, 255, 255))
    if opacity < 1:
        a = badge.getchannel("A").point(lambda v: int(v * opacity))
        badge.putalpha(a)
    # A soft shadow so the mark reads on light and dark pictures alike.
    pad = max(6, mh // 3)
    shadow = Image.new("RGBA", (bw + pad * 2, bh + pad * 2), (0, 0, 0, 0))
    sa = badge.getchannel("A").point(lambda v: int(v * 0.75))
    shadow.paste((0, 0, 0, 255), (pad, pad), sa)
    shadow = shadow.filter(ImageFilter.GaussianBlur(pad / 2))
    margin = int(min(w, h) * 0.04)
    x = w - bw - margin if corner.endswith("right") else margin
    y = margin if corner.startswith("top") else h - bh - margin
    poster.alpha_composite(shadow, (x - pad, y - pad + max(1, mh // 14)))
    poster.alpha_composite(badge, (x, y))
    out = io.BytesIO()
    poster.convert("RGB").save(out, "PNG", optimize=True)
    return out.getvalue()


COVER_CONCEPT_SCHEMA = {
    "type": "object",
    "properties": {
        "focus": {"type": "string", "enum": ["characters", "setting", "object", "symbol"],
                  "description": "What the poster is built around."},
        "characters": {"type": "array", "items": {"type": "string"},
                       "description": "Exactly as in the cast list; empty if the poster shows no people. Any number."},
        "composition": {"type": "string", "description": "The image: who/what, where in the frame, the moment, the pose."},
        "setting": {"type": "string", "description": "The place and time shown, true to the story."},
        "colour_and_light": {"type": "string"},
        "title_style": {"type": "string", "description": "The lettering style that fits the genre and mood."},
    },
    "required": ["focus", "characters", "composition", "setting", "colour_and_light", "title_style"],
}


def _story(project: dict) -> tuple[str, str, str]:
    bible = project.get("story_bible") or {}
    title = (project.get("title") or bible.get("title") or "Untitled").strip()
    story = bible.get("logline") or project.get("logline") or ""
    extra = "; ".join(x for x in (bible.get("premise_twist"), bible.get("central_conflict"), bible.get("tone"),
                                  bible.get("theme"), bible.get("ending")) if x)
    return title, story, extra


def concept_prompt(project: dict, cast: list[dict], note: str = "") -> str:
    """The art director decides what the poster shows, from the story."""
    title, story, extra = _story(project)
    names = "\n".join(f"- {c['name']}: {(c.get('description') or '')[:160]}" for c in cast[:12]) or "- (no characters)"
    return f"""You are the art director of a film studio. Design the poster for "{title}" ({project.get('genre') or 'drama'},
visual style: {project.get('visual_style') or 'cinematic, realistic'}).

THE STORY: {story}
{('MORE ABOUT IT: ' + extra) if extra else ''}
{('SCRIPT OPENING: ' + (project.get('script') or '')[:1500]) if project.get('script') else ''}
CAST:
{names}
{('THE CREATOR WANTS: ' + note) if note.strip() else ''}

Decide what makes the strongest poster for THIS story. It can show one character, several, the whole cast, or no
people at all (a place, an object or a symbol) if that tells the story better. Be true to the story's world, era
and culture. The title will be lettered on the poster."""


def cover_prompt(project: dict, cast: list[dict], refs: list[str], identity_text=None,
                 concept: Optional[dict] = None, note: str = "") -> str:
    """The poster brief, from the story and the art director's concept."""
    title, story, extra = _story(project)
    genre = project.get("genre") or "drama"
    concept = concept or {}
    who = []
    for c in cast:
        line = identity_text(c["name"], c.get("identity"), c.get("description") or "") if identity_text \
            else f"{c['name']}: {c.get('description') or ''}"
        who.append(f"- {line}")
    roles = " ".join(f"Reference image {i}: {label}." for i, label in enumerate(refs, start=1))
    people = (f"CHARACTERS ON THE POSTER (draw these exact people, never different ones):\n" + "\n".join(who) + "\n" + roles
              if who else "No people on this poster.")
    design = "\n".join(f"{k.replace('_', ' ').upper()}: {v}" for k, v in concept.items()
                       if k in ("composition", "setting", "colour_and_light") and v)
    lettering = concept.get("title_style") or "a beautiful display typeface that suits the genre"
    return f"""Design the poster (cover art) for a {genre} called "{title}".
It must make people stop scrolling and want to watch: bold, beautiful, cinematic key art like the best
streaming posters, true to the story's world.

THE STORY: {story}
{('MOOD AND CONFLICT: ' + extra) if extra else ''}
{('THE CREATOR WANTS: ' + note) if note.strip() else ''}
{design}
{people}

Composition: 3:4 portrait, a clear focal point that reads even as a small phone thumbnail.
TITLE LETTERING: write the title exactly as "{title}" (same spelling and capitals), large and perfectly legible,
in {lettering}, integrated into the art (usually the lower third). No other words, no taglines, no credits.
Keep the top-right corner calm and free of faces and lettering (a small logo goes there).
No logos, watermarks or brand names of any kind. Fictional people only."""
