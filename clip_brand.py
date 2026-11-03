"""
"Made with Viyo" tag for AI Repurposer clips that leave the app.

Clips posted inside Viyo stay clean; this makes a tagged copy for
downloading or sharing to TikTok, Reels or Shorts, where every view then
carries Viyo's name. The tag sits top-left, above the hook text and below
the area the big apps cover with their own top bar.
"""
from __future__ import annotations

import os
import subprocess

TAG_TEXT = "Made with Viyo · viyo.site"
FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class BrandError(Exception):
    pass


def _drawtext(height: int) -> str:
    scale = height / 1920.0
    size = max(14, round(30 * scale))
    x, y = round(48 * scale), round(150 * scale)
    pad = max(6, round(14 * scale))
    font = f"fontfile='{FONT_FILE}':" if os.path.exists(FONT_FILE) else ""
    text = TAG_TEXT.replace(":", r"\:").replace("'", r"\'")
    return (f"drawtext={font}text='{text}':fontsize={size}:fontcolor=white@0.92:"
            f"x={x}:y={y}:box=1:boxcolor=black@0.45:boxborderw={pad}:"
            f"shadowcolor=black@0.6:shadowx=1:shadowy=1")


def _height(path: str) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height",
                        "-of", "csv=p=0", path], capture_output=True, text=True, timeout=60)
    try:
        return int(r.stdout.strip().split("\n")[0])
    except ValueError:
        raise BrandError("not a readable video")


def brand_clip(src: str, out: str, timeout: float = 180) -> None:
    """Writes a copy of src with the tag burned in. Audio is copied untouched."""
    h = _height(src)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", _drawtext(h), "-map", "0:v:0", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", out]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise BrandError("tagging took too long")
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        raise BrandError(r.stderr[-400:] or "ffmpeg failed")
