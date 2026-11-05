"""
Creator Studio render: scenes -> one finished MP4 at the project's aspect
ratio (1080x1920 vertical or 1920x1080 landscape).

Each scene becomes a clip of identical format (30 fps, H.264 yuv420p,
AAC 48 kHz stereo), so the clips are joined without re-encoding:
  * still scene  - the image with a slow push-in, voice on top
  * video scene  - the clip cropped to the frame, held on its last frame
                   if the voice runs longer
  * captions     - spoken lines burned in, timed across the scene
Audio is loudness-normalised per scene so levels match across the film.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FPS = 30


@dataclass
class SceneInput:
    duration: float
    image: Optional[str] = None
    video: Optional[str] = None
    audio: Optional[str] = None
    audio_seconds: Optional[float] = None
    lines: tuple = ()


class RenderError(Exception):
    pass


def canvas(aspect_ratio: str) -> tuple[int, int]:
    return (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)


def _run(cmd: list[str], step: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RenderError(f"{step} failed: {r.stderr[-600:]}")


def probe(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RenderError(f"can't read {os.path.basename(path)}")
    info = json.loads(r.stdout)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"duration": float(info.get("format", {}).get("duration") or 0), "width": v.get("width"),
            "height": v.get("height"), "has_audio": any(s.get("codec_type") == "audio" for s in info.get("streams", []))}


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:").replace("%", "\\%")


def _wrap(text: str, width_chars: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:3])


def _caption_filter(lines: tuple, duration: float, w: int, h: int, tmp: str, idx: int) -> str:
    if not lines:
        return ""
    weights = [max(1, len(t.split())) for t in lines]
    total = sum(weights)
    t0, parts = 0.15, []
    span = max(0.5, duration - 0.3)
    size = int(w * (0.05 if h > w else 0.032))
    for n, (text, wt) in enumerate(zip(lines, weights)):
        t1 = t0 + span * wt / total
        txt_path = os.path.join(tmp, f"cap_{idx}_{n}.txt")
        with open(txt_path, "w") as f:
            f.write(_wrap(text, 26 if h > w else 48))
        parts.append(
            f"drawtext=fontfile={FONT}:textfile='{txt_path}':fontsize={size}:fontcolor=white:borderw={max(2, size // 12)}"
            f":bordercolor=black@0.85:line_spacing={size // 4}:x=(w-text_w)/2:y=h*{0.72 if h > w else 0.82}-text_h/2"
            f":enable='between(t,{t0:.2f},{t1:.2f})'")
        t0 = t1
    return "," + ",".join(parts)


def render_scene(s: SceneInput, w: int, h: int, out: str, tmp: str, idx: int, captions: bool) -> float:
    voice = s.audio_seconds or 0.0
    d = round(max(s.duration, voice + 0.5, 1.5), 2)
    caps = _caption_filter(s.lines, d, w, h, tmp, idx) if captions else ""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if s.video:
        cmd += ["-i", s.video]
        v = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS},"
             f"tpad=stop_mode=clone:stop_duration={d},trim=duration={d},setsar=1,format=yuv420p{caps}[v]")
    elif s.image:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", str(d), "-i", s.image]
        # Slow push-in: scale up a little over the scene and keep it centred.
        bw, bh = int(w * 1.25) // 2 * 2, int(h * 1.25) // 2 * 2
        frames = max(1, int(d * FPS))
        v = (f"[0:v]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
             f"zoompan=z='1+0.10*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={FPS},"
             f"setsar=1,format=yuv420p{caps}[v]")
    else:
        raise RenderError("a scene has no image or video")
    audio_idx = 1
    if s.audio:
        cmd += ["-i", s.audio]
        a = f"[{audio_idx}:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=duration={d},loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]"
    elif s.video and probe(s.video)["has_audio"]:
        a = f"[0:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=duration={d},loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]"
    else:
        cmd += ["-f", "lavfi", "-t", str(d), "-i", "anullsrc=r=48000:cl=stereo"]
        a = f"[{audio_idx}:a]atrim=duration={d}[a]"
    cmd += ["-filter_complex", f"{v};{a}", "-map", "[v]", "-map", "[a]", "-t", str(d),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-maxrate", "4M", "-bufsize", "8M",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", out]
    _run(cmd, f"scene {idx + 1}")
    return d


def render_film(scenes: list[SceneInput], aspect_ratio: str, out: str, tmp: str, captions: bool = True) -> dict:
    if not scenes:
        raise RenderError("no scenes to render")
    w, h = canvas(aspect_ratio)
    clips = []
    for i, s in enumerate(scenes):
        clip = os.path.join(tmp, f"scene_{i:03d}.mp4")
        render_scene(s, w, h, clip, tmp, i, captions)
        clips.append(clip)
    listing = os.path.join(tmp, "clips.txt")
    with open(listing, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy",
          "-movflags", "+faststart", out], "joining scenes")
    info = probe(out)
    sha = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return {"duration": round(info["duration"], 2), "width": info["width"], "height": info["height"],
            "bytes": os.path.getsize(out), "sha256": sha.hexdigest()}


def join_audio(wavs: list[str], out: str, gap: float = 0.25) -> float:
    """Dialogue lines -> one scene track with a short pause between lines."""
    if len(wavs) == 1:
        _run(["ffmpeg", "-y", "-v", "error", "-i", wavs[0], "-ar", "48000", "-ac", "1", out], "voice track")
    else:
        inputs, labels = [], []
        for i, wv in enumerate(wavs):
            inputs += ["-i", wv]
            labels.append(f"[{i}:a]aresample=48000,apad=pad_dur={gap}[a{i}]")
        chain = ";".join(labels) + ";" + "".join(f"[a{i}]" for i in range(len(wavs))) + f"concat=n={len(wavs)}:v=0:a=1[out]"
        _run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", chain, "-map", "[out]", "-ac", "1", out], "voice track")
    return probe(out)["duration"]
