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


def cover_prompt(project: dict, cast: list[dict], refs: list[str], identity_text=None) -> str:
    """The poster brief, from what the story is about."""
    bible = project.get("story_bible") or {}
    title = (project.get("title") or bible.get("title") or "Untitled").strip()
    genre = project.get("genre") or "drama"
    who = []
    for c in cast[:3]:
        line = identity_text(c["name"], c.get("identity"), c.get("description") or "") if identity_text \
            else f"{c['name']}: {c.get('description') or ''}"
        who.append(f"- {line}")
    roles = " ".join(f"Reference image {i}: {label}." for i, label in enumerate(refs, start=1))
    story = bible.get("logline") or project.get("logline") or ""
    extra = "; ".join(x for x in (bible.get("central_conflict"), bible.get("tone"), bible.get("theme")) if x)
    vertical = "3:4 portrait"
    return f"""Design the poster (cover art) for a short {genre} series called "{title}".
It must make people stop scrolling and want to watch: a bold, beautiful, cinematic key-art image like the best
streaming-drama posters, with real emotion and tension between the characters.

THE STORY: {story}
{('MOOD AND CONFLICT: ' + extra) if extra else ''}
MAIN CHARACTERS (draw these exact people, never different ones):
{chr(10).join(who) if who else '- (no characters yet: show the story world)'}
{roles}

Composition: {vertical}. The main character(s) large and close, a moment that captures the heart of the story,
dramatic lighting and colour that fits the genre, a rich background that hints at the setting.
TITLE LETTERING: write the title exactly as "{title}" (same spelling and capitals), large, in a beautiful
display typeface that suits the genre (e.g. elegant script for romance, sharp metallic for thriller), integrated
into the art in the lower third, perfectly legible. No other words, no taglines, no credits.
Keep the top-right corner calm and free of faces and lettering (a small logo goes there).
No logos, watermarks or brand names of any kind. Fictional people only."""
