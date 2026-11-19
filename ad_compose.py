"""Frame layout and FFmpeg rendering for Ads Maker scenes and the end card.

Every element has its own area of the frame, computed per output size:

  safe area     margins every app's buttons/captions leave clear (more at the bottom for 9:16)
  caption band  the only place captions go: inside the safe area, lower part of the frame
  content box   where a real advertiser asset (screenshot, product photo, logo) is shown, whole and
                undistorted, ABOVE the caption band, so captions can never cover it
  brand mark    a small logo in the top corner during the story scenes (when there is a logo)
  end card      logo, brand name, website and call to action, stacked and centred, no captions

Captions are measured in real pixels with the actual font (PIL), wrapped to the band's width, sized down
within readable limits, and split into consecutive pages when they still don't fit; nothing is cut off
or truncated. Text is drawn at fixed pixel positions from that layout, so what QC checks is exactly what
was drawn. Generated images are cropped to fill (never stretched); real assets are fitted, never cropped.

Pure functions plus thin ffmpeg wrappers: no network, no AI, nothing advertiser-specific.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from PIL import ImageFont

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FPS = 30
LINE_SPACING = 1.32


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def intersects(self, o: "Box") -> bool:
        return self.x < o.x2 and o.x < self.x2 and self.y < o.y2 and o.y < self.y2

    def inside(self, o: "Box") -> bool:
        return self.x >= o.x and self.y >= o.y and self.x2 <= o.x2 and self.y2 <= o.y2

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def orientation(w: int, h: int) -> str:
    return "portrait" if h > w * 1.1 else "landscape" if w > h * 1.1 else "square"


def frame_layout(w: int, h: int) -> dict:
    """The fixed areas of a w x h frame (see the module docstring)."""
    o = orientation(w, h)
    side = max(24, round(w * (0.06 if o != "landscape" else 0.05)))
    top = round(h * (0.07 if o == "portrait" else 0.08))
    bottom = round(h * (0.14 if o == "portrait" else 0.08))     # 9:16 apps put their own UI at the bottom
    safe = Box(side, top, w - 2 * side, h - top - bottom)
    band_top = round(h * {"portrait": 0.70, "square": 0.70, "landscape": 0.72}[o])
    band_w = safe.w if o != "landscape" else min(safe.w, round(w * 0.72))
    caption = Box((w - band_w) // 2, band_top, band_w, safe.y2 - band_top)
    gap = round(h * 0.02)
    content = Box(safe.x, safe.y, safe.w, caption.y - gap - safe.y)
    m = round(min(w, h) * 0.13)
    brand_mark = Box(safe.x, safe.y, m, m)
    # When the brand mark is on screen, a real asset is shown below it so the two never touch.
    content_below_mark = Box(safe.x, brand_mark.y2 + gap, safe.w, caption.y - gap - (brand_mark.y2 + gap))
    return {"orientation": o, "safe": safe, "caption": caption, "content": content, "brand_mark": brand_mark,
            "content_below_mark": content_below_mark, "max_lines": 3 if o == "portrait" else 2}


@lru_cache(maxsize=256)
def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def text_width(text: str, size: int, bold: bool = True) -> int:
    return int(round(_font(size, bold).getlength(text)))


def _break_long_word(word: str, size: int, max_w: int) -> list[str]:
    """A word wider than the line (usually a web address): split after '/', '.' or '-', else by letters."""
    if text_width(word, size) <= max_w:
        return [word]
    parts, cur = [], ""
    for ch in word:
        if cur and text_width(cur + ch, size) > max_w:
            cut = max(cur.rfind("/"), cur.rfind("."), cur.rfind("-"))
            if cut > 0:
                parts.append(cur[: cut + 1])
                cur = cur[cut + 1:]
            else:
                parts.append(cur)
                cur = ""
        cur += ch
    if cur:
        parts.append(cur)
    return parts


def wrap(text: str, size: int, max_w: int) -> list[str]:
    words = []
    for w in text.split():
        words += _break_long_word(w, size, max_w)
    lines, cur = [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if not cur or text_width(cand, size) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


@dataclass
class CaptionPage:
    font_size: int
    lines: list = field(default_factory=list)        # [(text, Box)]
    box: Optional[Box] = None

    @property
    def text(self) -> str:
        return " ".join(t for t, _ in self.lines)


def font_limits(w: int, h: int) -> tuple[int, int]:
    """(largest, smallest) caption size for this frame: big enough to read on a phone, never oversized."""
    m = min(w, h)
    return max(24, round(m * 0.060)), max(20, round(m * 0.044))


def layout_caption(text: str, w: int, h: int, scale: float = 1.0) -> list[CaptionPage]:
    """Fits [text] in the caption band: the largest size (within limits) at which it fits in two lines;
    at the smallest size it's split into several pages shown one after another. Never truncated."""
    text = " ".join((text or "").split())
    if not text:
        return []
    lay = frame_layout(w, h)
    band, max_lines = lay["caption"], lay["max_lines"]
    big, small = font_limits(w, h)
    big, small = max(small, round(big * scale)), max(18, round(small * scale))
    pad = lambda s: round(s * 0.35)   # noqa: E731  room for the outline and the backing box
    for size in range(big, small - 1, -2):
        lines = wrap(text, size, band.w - 2 * pad(size))
        if len(lines) <= max_lines and _block_h(len(lines), size) <= band.h:
            return [_place(lines, size, band)]
    size = small
    lines = wrap(text, size, band.w - 2 * pad(size))
    per = max(1, min(max_lines, int(band.h // (size * LINE_SPACING)) or 1))
    return [_place(lines[i:i + per], size, band) for i in range(0, len(lines), per)]


def _block_h(n_lines: int, size: int) -> int:
    return round(size + (n_lines - 1) * size * LINE_SPACING + size * 0.3)


def _place(lines: list[str], size: int, band: Box) -> CaptionPage:
    """Lines centred horizontally, the block sitting at the bottom of the band (just above the apps' UI)."""
    total = _block_h(len(lines), size)
    y0 = band.y2 - total
    page = CaptionPage(font_size=size)
    xs, ys, x2s, y2s = [], [], [], []
    for i, line in enumerate(lines):
        lw = text_width(line, size)
        x = band.x + (band.w - lw) // 2
        y = y0 + round(i * size * LINE_SPACING)
        b = Box(x, y, lw, round(size * 1.2))
        page.lines.append((line, b))
        xs.append(b.x), ys.append(b.y), x2s.append(b.x2), y2s.append(b.y2)
    page.box = Box(min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys))
    return page


def time_pages(pages: list[CaptionPage], start: float, end: float) -> list[tuple[CaptionPage, float, float]]:
    """Spreads a caption's pages over its time, in proportion to their length."""
    if not pages:
        return []
    total = sum(max(1, len(p.text)) for p in pages)
    out, t = [], start
    for p in pages:
        d = (end - start) * max(1, len(p.text)) / total
        out.append((p, t, t + d))
        t += d
    return out


# ---------------------------------------------------------------- drawing

def _textfile(tmp: str, name: str, text: str) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path.replace("\\", "/").replace(":", "\\:")


def caption_filters(timed: list, tmp: str, prefix: str) -> list[str]:
    """drawtext filters at the layout's exact pixel positions, white with an outline and a soft backing."""
    out = []
    for pi, (page, t0, t1) in enumerate(timed):
        border = max(2, round(page.font_size * 0.07))
        for li, (line, b) in enumerate(page.lines):
            tf = _textfile(tmp, f"{prefix}_c{pi}_{li}.txt", line)
            out.append(
                f"drawtext=fontfile={FONT_BOLD}:textfile='{tf}':fontsize={page.font_size}:fontcolor=white:"
                f"borderw={border}:bordercolor=black:box=1:boxcolor=black@0.38:boxborderw={round(page.font_size * 0.18)}:"
                f"x={b.x}:y={b.y}:enable='between(t\\,{t0:.3f}\\,{t1:.3f})'"
            )
    return out


def _ff(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed: {r.stderr[-600:]}")


def _encode(out: str, audio: bool = True) -> list[str]:
    return ["-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"] + \
        (["-c:a", "aac", "-ar", "44100", "-ac", "2"] if audio else []) + ["-movflags", "+faststart", out]


def render_image_scene(image: str, mode: str, w: int, h: int, duration: float, audio: str, timed: list,
                       out: str, tmp: str, prefix: str, logo: Optional[str] = None, pan: str = "center") -> dict:
    """A still scene. mode "cover": a generated image cropped to fill, with a slow push-in (never stretched).
    mode "fit": a real asset shown whole in the content box over a blurred copy of itself, captions below it.
    [logo]: the brand mark in its corner. Returns what was drawn where (for QC)."""
    lay = frame_layout(w, h)
    frames = max(1, int(round(duration * FPS)))
    graph = []
    if mode == "fit":
        c = lay["content_below_mark"] if logo else lay["content"]
        graph.append(f"[0:v]split=2[a][b]")
        graph.append(f"[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=24:2,"
                     f"eq=brightness=-0.18:saturation=0.8[bg]")
        graph.append(f"[b]scale={c.w}:{c.h}:force_original_aspect_ratio=decrease[fg]")
        graph.append(f"[bg][fg]overlay=x={c.x}+({c.w}-overlay_w)/2:y={c.y}+({c.h}-overlay_h)/2,setsar=1[v0]")
        shown = c
    else:
        step = 0.10 / frames
        x_expr = f"(iw-iw/zoom)*(on/{frames})" if pan == "left_right" else "iw/2-(iw/zoom/2)"
        graph.append(f"[0:v]scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
                     f"zoompan=z='min(zoom+{step:.8f},1.10)':x='{x_expr}':y='ih/2-(ih/zoom/2)':d={frames}:"
                     f"s={w}x{h}:fps={FPS},setsar=1[v0]")
        shown = Box(0, 0, w, h)
    inputs = ["-loop", "1", "-framerate", str(FPS), "-i", image, "-i", audio]
    last = "v0"
    drawn = {"captions": [p.box.as_dict() for p, _, _ in timed], "content": shown.as_dict(), "texts": [p.text for p, _, _ in timed]}
    if logo:
        m = lay["brand_mark"]
        inputs += ["-loop", "1", "-i", logo]
        graph.append(f"[2:v]scale={m.w}:{m.h}:force_original_aspect_ratio=decrease[lg]")
        graph.append(f"[{last}][lg]overlay=x={m.x}:y={m.y}[v1]")
        last = "v1"
        drawn["brand_mark"] = m.as_dict()
    caps = caption_filters(timed, tmp, prefix)
    graph.append(f"[{last}]" + (",".join(caps) if caps else "null") + "[vout]")
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(graph), "-map", "[vout]", "-map", "1:a:0",
           "-t", f"{duration:.3f}"] + _encode(out)
    _ff(cmd, "Scene render")
    return drawn


def render_video_scene(video: str, w: int, h: int, timed: list, out: str, tmp: str, prefix: str,
                       duration: Optional[float] = None, audio: Optional[str] = None) -> dict:
    """A generated video clip cropped to fill, captions in the band. [audio]: a separate voice track
    (the clip loops to [duration]); without it the clip's own sound is kept as rendered."""
    caps = caption_filters(timed, tmp, prefix)
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1" + ("," + ",".join(caps) if caps else "")
    if audio:
        cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", video, "-i", audio, "-vf", vf, "-map", "0:v:0",
               "-map", "1:a:0", "-t", f"{duration:.3f}"] + _encode(out)
    else:
        cmd = ["ffmpeg", "-y", "-i", video, "-vf", vf, "-map", "0:v:0", "-map", "0:a:0?"] + _encode(out)
    _ff(cmd, "Video scene render")
    return {"captions": [p.box.as_dict() for p, _, _ in timed], "content": Box(0, 0, w, h).as_dict(),
            "texts": [p.text for p, _, _ in timed]}


# ---------------------------------------------------------------- end card

def _fit_size(text: str, start: int, smallest: int, max_w: int) -> int:
    s = start
    while s > smallest and text_width(text, s) > max_w:
        s -= 1
    return s


def end_card_layout(name: str, display_url: str, cta: str, w: int, h: int, has_logo: bool) -> dict:
    """Logo, brand name, website and call to action stacked in the centre of the safe area, each sized to
    fit the width (a long address wraps onto a second line instead of being cut). Missing items are left
    out, never filled in."""
    lay = frame_layout(w, h)
    safe = lay["safe"]
    m = min(w, h)
    inner = round(safe.w * 0.92)
    gap = round(h * 0.028)
    items = []
    if has_logo:
        lh = round(h * (0.22 if lay["orientation"] == "portrait" else 0.30))
        lw = min(inner, round(w * (0.5 if lay["orientation"] == "portrait" else 0.32)))
        items.append(("logo", None, None, lw, lh))
    if name:
        s = _fit_size(name, round(m * (0.085 if not has_logo else 0.07)), round(m * 0.04), inner)
        for line in wrap(name, s, inner)[:2]:
            items.append(("name", line, s, text_width(line, s), round(s * 1.25)))
    if display_url:
        s = _fit_size(display_url, round(m * 0.052), round(m * 0.034), inner)
        for line in wrap(display_url, s, inner):
            items.append(("url", line, s, text_width(line, s), round(s * 1.3)))
    if cta:
        s = _fit_size(cta, round(m * 0.05), round(m * 0.034), inner - round(m * 0.08))
        lines = wrap(cta, s, inner - round(m * 0.08))[:2]
        pad_x, pad_y = round(s * 0.9), round(s * 0.55)
        bw = max(text_width(t, s) for t in lines) + 2 * pad_x
        bh = round(len(lines) * s * LINE_SPACING) + 2 * pad_y
        items.append(("cta", lines, s, bw, bh))
    total = sum(it[4] for it in items) + gap * max(0, len(items) - 1)
    y = safe.y + max(0, (safe.h - total) // 2)
    placed = []
    for kind, text, size, iw, ih in items:
        placed.append({"kind": kind, "text": text, "size": size, "box": Box(safe.x + (safe.w - iw) // 2, y, iw, ih)})
        y += ih + gap
    return {"items": placed, "safe": safe, "fits": total <= safe.h}


def render_end_card(name: str, display_url: str, cta: str, w: int, h: int, duration: float, audio: str,
                    out: str, tmp: str, logo: Optional[str] = None, background: Optional[str] = None,
                    plain: bool = False) -> dict:
    """The closing card, built only from the advertiser's own facts. [plain]: a solid background (used as
    an automatic correction when the image background made the text hard to read)."""
    lay = end_card_layout(name, display_url, cta, w, h, bool(logo))
    inputs, graph = [], []
    if background and not plain:
        inputs += ["-loop", "1", "-framerate", str(FPS), "-i", background]
        graph.append(f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=30:3,"
                     f"eq=brightness=-0.42:saturation=0.6,setsar=1[v0]")
    else:
        inputs += ["-f", "lavfi", "-i", f"color=c=0x12121a:s={w}x{h}:r={FPS}"]
        graph.append("[0:v]setsar=1[v0]")
    inputs += ["-i", audio]
    last, n = "v0", 2
    texts = []
    for it in lay["items"]:
        b = it["box"]
        if it["kind"] == "logo" and logo:
            inputs += ["-loop", "1", "-i", logo]
            graph.append(f"[{n}:v]scale={b.w}:{b.h}:force_original_aspect_ratio=decrease[lg]")
            graph.append(f"[{last}][lg]overlay=x={b.x}+({b.w}-overlay_w)/2:y={b.y}+({b.h}-overlay_h)/2[v{n}]")
            last, n = f"v{n}", n + 1
    draws = []
    for i, it in enumerate(lay["items"]):
        b, s = it["box"], it["size"]
        if it["kind"] in ("name", "url"):
            tf = _textfile(tmp, f"end_{i}.txt", it["text"])
            color = "white" if it["kind"] == "name" else "0xdfe7ff"
            draws.append(f"drawtext=fontfile={FONT_BOLD}:textfile='{tf}':fontsize={s}:fontcolor={color}:"
                         f"borderw={max(2, round(s * 0.06))}:bordercolor=black@0.7:x={b.x}:y={b.y}")
            texts.append(it["text"])
        elif it["kind"] == "cta":
            draws.append(f"drawbox=x={b.x}:y={b.y}:w={b.w}:h={b.h}:color=white@0.96:t=fill")
            for li, line in enumerate(it["text"]):
                tf = _textfile(tmp, f"end_{i}_{li}.txt", line)
                lw = text_width(line, s)
                ty = b.y + round(s * 0.55) + round(li * s * LINE_SPACING)
                draws.append(f"drawtext=fontfile={FONT_BOLD}:textfile='{tf}':fontsize={s}:fontcolor=0x111111:"
                             f"x={b.x + (b.w - lw) // 2}:y={ty}")
                texts.append(line)
    graph.append(f"[{last}]" + (",".join(draws) if draws else "null") + "[vout]")
    audio_idx = 1
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(graph), "-map", "[vout]", "-map", f"{audio_idx}:a:0",
           "-t", f"{duration:.3f}"] + _encode(out)
    _ff(cmd, "End card render")
    return {"items": [{"kind": it["kind"], "text": it["text"], "box": it["box"].as_dict()} for it in lay["items"]],
            "texts": texts, "fits": lay["fits"]}


def concat(clips: list[str], out: str) -> None:
    """Joins scene clips (same size/fps/audio format) without re-encoding the video."""
    lst = out + ".txt"
    with open(lst, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    _ff(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", out], "Concat")


def silence(duration: float, out: str) -> None:
    _ff(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
         "-t", f"{duration:.3f}", out], "Silence")
