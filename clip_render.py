"""
Professional 9:16 rendering of a clip.

  * Letterbox / pillarbox bars are cropped away first (measured, not assumed).
  * Framing follows the people: faces were located across the WHOLE source
    by video_analysis (YuNet). The clip is planned shot by shot — the crop
    re-centres instantly at each camera cut or edit point (cuts are refined
    to frame accuracy), holds still while the subject is still, and only
    glides when they actually move (dead-band pan, no jitter).
  * When the picture has no people (slides, screen share, b-roll) the full
    frame is kept, fitted to the width over a blurred copy of itself —
    nothing important is cropped off.
  * Audio: rumble filter, micro-fades at every cut (no clicks), and
    TWO-PASS loudness normalisation to -14 LUFS (measured, then applied
    linearly), so levels are accurate rather than approximate.
  * Each kept block is its own seeked input (bounded memory); encode is
    libx264 high profile, CRF 20, faststart.

Every decision is returned as facts (crop mode, tracked share, how many
sampled faces would be cut off) so QC can verify it instead of trusting it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

import video_analysis as va

OUT_W, OUT_H = 1080, 1920
TARGET_LUFS = -14.0
NO_FACE_FIT_BELOW = 0.15        # face ratio under this: fit the frame instead of cropping
FACE_PAD = 0.7                  # keep this multiple of the face width around the face centre
DEADBAND = 0.18                 # fraction of crop width the subject may drift before the crop moves
GLIDE_SECONDS = 1.2
CUT_REFINE_FPS = 30


class RenderError(Exception):
    pass


@dataclass
class CropPlan:
    mode: str = "center"                 # "track" | "center" | "fit"
    src_w: int = 0
    src_h: int = 0
    bars: Optional[tuple] = None         # (x, y, w, h) crop of the source, or None
    crop_w: int = 0
    crop_h: int = 0
    pieces: list = field(default_factory=list)   # (t0, x0, t1, x1) in OUTPUT time, source px of the bar-cropped frame
    facts: dict = field(default_factory=dict)

    def x_at(self, t: float) -> float:
        for t0, x0, t1, x1 in self.pieces:
            if t < t1:
                if t1 <= t0:
                    return x1
                return x0 + (x1 - x0) * max(0.0, min(1.0, (t - t0) / (t1 - t0)))
        return self.pieces[-1][3] if self.pieces else max(0.0, (self.src_w - self.crop_w) / 2)


# ------------------------------------------------------------------ planning

def refine_cut(path: str, approx: float, window: float = 1.1) -> float:
    """Frame-accurate time of the camera cut near `approx` (sampled at 1 fps,
    so it can be up to a second off). Falls back to `approx`."""
    if cv2 is None:
        return approx
    try:
        t0 = max(0.0, approx - window)
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{window * 2:.3f}", "-i", path, "-an",
             "-vf", f"fps={CUT_REFINE_FPS},scale=160:-2,setsar=1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True, timeout=60)
        raw = proc.stdout
        # Frame height: derive from the byte count (160 wide).
        info = va.probe_video(path)
        h = int(round(160 * info["height"] / info["width"] / 2)) * 2
        size = 160 * h * 3
        n = len(raw) // size
        if n < 3:
            return approx
        prev, best_d, best_i = None, 0.0, -1
        for i in range(n):
            frame = np.frombuffer(raw[i * size:(i + 1) * size], dtype=np.uint8).reshape(h, 160, 3)
            hist = va._hist(frame)
            if prev is not None:
                d = va._bhattacharyya(prev, hist)
                if d > best_d:
                    best_d, best_i = d, i
            prev = hist
        if best_i >= 1 and best_d > va.CUT_DISTANCE:
            return t0 + best_i / CUT_REFINE_FPS
    except Exception:
        pass
    return approx


def _output_map(blocks: list):
    cum, total = [], 0.0
    for b0, b1 in blocks:
        cum.append(total)
        total += b1 - b0

    def to_out(t: float) -> Optional[float]:
        for i, (b0, b1) in enumerate(blocks):
            if b0 <= t < b1:
                return cum[i] + (t - b0)
        return None

    return to_out, cum, total


def _main_face(faces: list) -> Optional[va.Face]:
    return max(faces, key=lambda f: f.w * f.h) if faces else None


def plan_crop(timeline: Optional[va.VisualTimeline], blocks: list, src_w: int, src_h: int,
              source_path: Optional[str] = None, refine: bool = True) -> CropPlan:
    plan = CropPlan(src_w=src_w, src_h=src_h)
    rect = va.crop_rect_for_bars(timeline, src_w, src_h) if timeline is not None and timeline.usable else None
    bx, by, bw, bh = rect if rect else (0, 0, src_w, src_h)
    plan.bars = rect
    to_out, cum, total = _output_map(blocks)
    start, end = blocks[0][0], blocks[-1][1]

    vis = va.window_visual(timeline, start, end) if timeline is not None and timeline.usable else {"available": False}
    face_ratio = vis.get("face_ratio", 0.0) if vis.get("available") else 0.0
    plan.facts = {"bars_cropped": bool(rect), "visual_available": bool(vis.get("available")),
                  "face_ratio": round(face_ratio, 2)}

    # Source narrower than 9:16 is simply scaled to cover.
    crop_h = bh
    crop_w = int(round(crop_h * OUT_W / OUT_H / 2) * 2)
    if crop_w >= bw:
        plan.mode, plan.crop_w, plan.crop_h = "center", bw, int(round(bw * OUT_H / OUT_W / 2) * 2)
        plan.crop_h = min(plan.crop_h, bh)
        plan.pieces = [(0.0, 0.0, total, 0.0)]
        return plan
    plan.crop_w, plan.crop_h = crop_w, crop_h
    centre = (bw - crop_w) / 2.0

    if not vis.get("available") or face_ratio < NO_FACE_FIT_BELOW:
        # No people to follow. Wide enough content is fitted, not cropped.
        plan.mode = "fit" if vis.get("available") else "center"
        plan.pieces = [(0.0, centre, total, centre)]
        plan.facts["reason"] = "no_faces" if vis.get("available") else "no_visual_analysis"
        return plan

    # Boundaries where the framing may jump: camera cuts and block joins.
    bounds = [0.0]
    for ci in cum[1:]:
        bounds.append(ci)
    for ct in timeline.cuts_between(start, end):
        o = to_out(ct)
        if o is None:
            continue
        if refine and source_path:
            r = refine_cut(source_path, ct)
            o2 = to_out(r)
            o = o2 if o2 is not None else o
        bounds.append(o)
    bounds = sorted(set(round(b, 3) for b in bounds if 0 <= b < total))
    bounds.append(total)

    # Samples (output time, face centre x in bar-cropped px, face box)
    samples = []
    t_idx = timeline.times
    lo, hi = np.searchsorted(t_idx, [start, end])
    for k in range(int(lo), int(hi)):
        o = to_out(float(t_idx[k]))
        if o is None:
            continue
        f = _main_face(timeline.faces[k])
        if f is None:
            continue
        group = [x for x in timeline.faces[k] if x.w * x.h >= 0.35 * f.w * f.h]
        xs0 = min(g.cx - g.w / 2 for g in group) * src_w - bx
        xs1 = max(g.cx + g.w / 2 for g in group) * src_w - bx
        # Two people side by side who both fit in the crop: frame them together.
        if len(group) >= 2 and (xs1 - xs0) <= crop_w * (1 - 2 * 0.06):
            cx = (xs0 + xs1) / 2
        else:
            cx = f.cx * src_w - bx
        samples.append((o, cx, f))

    pieces: list = []
    tracked = 0.0
    last_x = centre
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = [(o, cx) for o, cx, _ in samples if a - 1e-6 <= o < b]
        if not seg:
            pieces.append((a, last_x, b, last_x))
            continue
        tracked += b - a
        first = float(np.median([cx for _, cx in seg[:3]]))
        cur = float(np.clip(first - crop_w / 2, 0, bw - crop_w))
        pieces.append((a, cur, a, cur))
        tcur = a
        for o, cx in seg:
            want = float(np.clip(cx - crop_w / 2, 0, bw - crop_w))
            if abs(want - cur) > DEADBAND * crop_w:
                t_move = min(max(o, tcur), b)
                t_end = min(t_move + GLIDE_SECONDS, b)
                if t_move > tcur:
                    pieces.append((tcur, cur, t_move, cur))
                pieces.append((t_move, cur, t_end, want))
                cur, tcur = want, t_end
        if b > tcur:
            pieces.append((tcur, cur, b, cur))
        last_x = cur
    pieces = [p for p in pieces if p[2] > p[0] or p[0] == p[2]]
    plan.pieces = pieces
    plan.mode = "track"

    # How many sampled faces would be cut off or touch the crop edge?
    cut_off = 0
    for o, cx, f in samples:
        x = plan.x_at(o)
        fl, fr = (f.cx - f.w / 2) * src_w - bx, (f.cx + f.w / 2) * src_w - bx
        if fl < x - 1 or fr > x + crop_w + 1:
            cut_off += 1
    plan.facts.update({"tracked_share": round(tracked / total, 2) if total else 0.0,
                       "face_samples": len(samples), "faces_cut_off": cut_off,
                       "faces_cut_off_share": round(cut_off / len(samples), 3) if samples else 0.0,
                       "crop_segments": len(bounds) - 1})
    return plan


def x_expression(plan: CropPlan) -> str:
    """ffmpeg expression for crop x as a function of t (output time)."""
    pieces = plan.pieces
    if not pieces:
        return f"{(plan.src_w - plan.crop_w) / 2:.1f}"
    expr = f"{pieces[-1][3]:.1f}"
    for t0, x0, t1, x1 in reversed(pieces):
        if t1 <= t0:
            continue
        slope = (x1 - x0) / (t1 - t0)
        seg = f"{x0:.1f}" if abs(slope) < 1e-9 else f"({x0:.1f}+{slope:.3f}*(t-{t0:.3f}))"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return expr


# ------------------------------------------------------------------ rendering

ZOOM = 1.08            # a cut inside one shot alternates between the normal frame and this gentle punch-in
MIN_ZOOM_BLOCK = 1.2   # a piece shorter than this keeps the previous framing (no flicker)


def jump_zoom_windows(blocks: list, timeline: Optional[va.VisualTimeline], plan: CropPlan,
                      joins: tuple = ()) -> list:
    """Zoom windows from jump cuts only (see zoom_windows)."""
    return zoom_windows(blocks, timeline, plan, joins)[0]


HOLD_MAX = 7.0          # one unchanging framing longer than this gets a gentle reframe at the next sentence
MIN_GAP_AFTER = 2.5     # ...unless another framing change comes within this many seconds anyway


def zoom_windows(blocks: list, timeline: Optional[va.VisualTimeline], plan: CropPlan, joins: tuple = (),
                 sentence_starts: tuple = ()) -> tuple[list, dict]:
    """Punch-in windows (output time) for a talking-head clip: alternated on cuts inside one shot (jump cuts)
    and, when one framing holds for over HOLD_MAX seconds, at the start of the next sentence (visual pacing).
    A camera cut or a joined part resets to the normal frame. `sentence_starts`: output times. Returns
    (windows, facts)."""
    facts = {"jump_cuts": 0, "pacing_reframes": 0}
    if plan.mode not in ("track", "center") or not (plan.facts.get("face_ratio") or 0) >= 0.5:
        return [], facts
    to_out, cum, total = _output_map(blocks)
    events = []                                   # (time, kind)
    for i in range(1, len(blocks)):
        prev_end, nxt = blocks[i - 1][1], blocks[i][0]
        camera = timeline is not None and timeline.usable and bool(timeline.cuts_between(prev_end - 0.05, nxt + 0.05))
        if camera or any(abs(nxt - j) < 0.01 for j in joins):
            events.append((cum[i], "reset"))
        elif blocks[i][1] - blocks[i][0] >= MIN_ZOOM_BLOCK:
            events.append((cum[i], "jump"))
    if timeline is not None and timeline.usable:
        for b0, b1 in blocks:
            for ct in timeline.cuts_between(b0, b1):
                o = to_out(ct)
                if o is not None:
                    events.append((o, "reset"))
    events.sort()
    changes = sorted({t for t, _k in events} | {0.0})
    zoomed, since, start = False, 0.0, None
    out = []

    def toggle(t):
        nonlocal zoomed, start
        if zoomed:
            out.append((round(start, 3), round(t, 3)))
        else:
            start = t
        zoomed = not zoomed

    queue = sorted(events + [(t, "sentence") for t in sentence_starts if 0 < t < total - 1.0])
    for t, kind in queue:
        if kind == "reset":
            if zoomed:
                toggle(t)
            since = t
        elif kind == "jump":
            toggle(t)
            since = t
            facts["jump_cuts"] += 1
        elif kind == "sentence" and t - since >= HOLD_MAX:
            nxt_change = min([c for c in changes if c > t + 1e-6] or [total])
            if nxt_change - t >= MIN_GAP_AFTER:
                toggle(t)
                since = t
                facts["pacing_reframes"] += 1
    if zoomed:
        toggle(total)
    return [w for w in out if w[1] - w[0] >= 0.5], facts


def snap_to_cuts(blocks: list, timeline: Optional[va.VisualTimeline], words: list, path: Optional[str] = None,
                 tolerance: float = 0.35) -> tuple[list, int]:
    """A kept block that starts just before a camera cut, or ends just after one, would flash a few frames of the
    other shot. Moves that edge onto the cut, only when no word is spoken in the sliver. Returns (blocks, moved)."""
    if timeline is None or not timeline.usable:
        return blocks, 0
    mids = [(float(w["start"]) + float(w["end"])) / 2 for w in words]
    out, moved = [], 0
    for b0, b1 in blocks:
        for ct in timeline.cuts_between(b0 - 0.01, b0 + tolerance):
            c = refine_cut(path, ct) if path else ct
            if b0 < c < b0 + tolerance and not any(b0 <= m < c for m in mids):
                b0, moved = c, moved + 1
                break
        for ct in timeline.cuts_between(b1 - tolerance - 1.0, b1):
            c = refine_cut(path, ct) if path else ct
            if b1 - tolerance < c < b1 and not any(c <= m < b1 for m in mids):
                b1, moved = c, moved + 1
                break
        if b1 - b0 >= 0.3:
            out.append((b0, b1))
    return out or blocks, moved


def cutaways(timeline: Optional[va.VisualTimeline], min_seconds: float = 1.5) -> list:
    """Shots in the source with no person on screen (product shots, screen shares, scenery): the B-roll the
    video actually contains. [(start, end)] in source time; [] when there's no visual analysis."""
    if timeline is None or not timeline.usable:
        return []
    out = []
    for sh in timeline.shots:
        if sh.end - sh.start < min_seconds:
            continue
        idx = range(sh.first, sh.last + 1)
        faces = sum(1 for k in idx if k < len(timeline.faces) and timeline.faces[k])
        sharp = float(np.median(timeline.sharpness[sh.first:sh.last + 1])) if sh.last >= sh.first else 0.0
        if faces <= 0.2 * len(idx) and sharp > 0:
            out.append((round(sh.start, 2), round(sh.end, 2)))
    return out


def _zoom_filter(zoom: list) -> str:
    """Punch in on the 1080x1920 output during the given windows (centred across, kept high so faces stay in)."""
    on = "+".join(f"between(t,{a:.3f},{b - 0.001:.3f})" for a, b in zoom)
    zw, zh = int(round(OUT_W * ZOOM / 2) * 2), int(round(OUT_H * ZOOM / 2) * 2)
    return (f"scale=w='if({on},{zw},{OUT_W})':h='if({on},{zh},{OUT_H})':eval=frame:flags=bicubic,"
            f"crop={OUT_W}:{OUT_H}:x='(iw-{OUT_W})/2':y='(ih-{OUT_H})*0.35'")


def _video_filter(plan: CropPlan, ass_path: Optional[str], out_fps: Optional[float],
                  zoom: Optional[list] = None) -> str:
    parts = []
    if out_fps:
        parts.append(f"fps={out_fps:g}")
    if plan.bars:
        x, y, w, h = plan.bars
        parts.append(f"crop={w}:{h}:{x}:{y}")
    if plan.mode == "fit":
        chain = ",".join(parts) + "," if parts else ""
        graph = (f"[vcat]{chain}split[bgsrc][fgsrc];"
                 f"[bgsrc]scale=216:384:force_original_aspect_ratio=increase,crop=216:384,boxblur=6:2,"
                 f"scale={OUT_W}:{OUT_H}:flags=bicubic[bg];"
                 f"[fgsrc]scale={OUT_W}:-2:flags=lanczos[fg];"
                 f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    else:
        if plan.mode == "track":
            crop = f"crop={plan.crop_w}:{plan.crop_h}:x='{x_expression(plan)}':y=0"
        else:
            crop = f"crop={plan.crop_w}:{plan.crop_h}:(iw-{plan.crop_w})/2:0"
        parts += [crop, f"scale={OUT_W}:{OUT_H}:flags=lanczos"]
        if zoom:
            parts.append(_zoom_filter(zoom))
        parts.append("setsar=1")
        graph = "[vcat]" + ",".join(parts)
    if ass_path:
        safe = ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        graph += f",subtitles='{safe}'"
    return graph


# Peak control. Measured on real uploads: loudnorm's linear mode let speech
# transients through at up to +2.8 dBTP, and the AAC encoder adds up to
# ~1.7 dB on top of a clean signal. Two lookahead limiters at -3 dBFS
# (one alone still missed isolated transients) keep the encoded clip under
# the QC ceiling. auto-level stays off so the limiter never raises the gain.
_LIMIT = "alimiter=limit=0.71:attack=2:release=60:level=false"
PEAK_LIMITER = f"{_LIMIT},{_LIMIT}"

# A stereo track with one channel effectively silent (a lav mic wired to one
# side) plays in one ear only on headphones, and the AAC encoder overshoots
# badly on it. Copy the live channel to both sides before normalising.
DEAD_CHANNEL_DB = 30.0


def channel_fix(inputs: list, filter_parts: list, concat_label: str) -> str:
    """'' normally; a pan filter (with trailing comma) when one stereo channel is silent."""
    graph = ";".join(filter_parts + [f"[{concat_label}]astats=measure_overall=none:"
                                     f"measure_perchannel=RMS_level[aout]"])
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *inputs, "-filter_complex", graph,
                            "-map", "[aout]", "-vn", "-f", "null", "-"], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return ""
    levels = []
    for m in re.finditer(r"RMS level dB:\s*(\S+)", r.stderr):
        try:
            levels.append(float(m.group(1)))
        except ValueError:            # "-inf" parses; anything else is ignored
            pass
    if r.returncode != 0 or len(levels) != 2:
        return ""
    left, right = levels
    if not np.isfinite(left) and not np.isfinite(right):
        return ""
    if not np.isfinite(right) or left - right > DEAD_CHANNEL_DB:
        return "pan=stereo|c0=c0|c1=c0,"
    if not np.isfinite(left) or right - left > DEAD_CHANNEL_DB:
        return "pan=stereo|c0=c1|c1=c1,"
    return ""


def measure_loudness(inputs: list, filter_parts: list, concat_label: str, pre: str = "") -> Optional[dict]:
    """First loudnorm pass on the concatenated audio only."""
    graph = ";".join(filter_parts + [f"[{concat_label}]{pre}highpass=f=70,loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11:"
                                     f"print_format=json[aout]"])
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *inputs, "-filter_complex", graph, "-map", "[aout]",
                        "-vn", "-f", "null", "-"], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return None
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        vals = {k: float(d[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except (KeyError, ValueError):
        return None
    if not all(np.isfinite(v) for v in vals.values()):   # silence yields -inf
        return None
    return vals


def render(source: str, out: str, blocks: list, plan: CropPlan, ass_path: Optional[str] = None,
           src_fps: Optional[float] = None, timeout: float = 900, has_audio: bool = True,
           zoom: Optional[list] = None, crf: int = 20, preset: str = "fast") -> dict:
    if not has_audio:
        return _render_silent(source, out, blocks, plan, ass_path, src_fps, timeout, zoom, crf, preset)
    inputs: list = []
    for b0, b1 in blocks:
        inputs += ["-ss", f"{b0:.3f}", "-t", f"{b1 - b0:.3f}", "-i", source]
    vparts, aparts = [], []
    for i, (b0, b1) in enumerate(blocks):
        d = b1 - b0
        vparts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
        fade = min(0.03, d / 4)
        aparts.append(f"[{i}:a]asetpts=PTS-STARTPTS,afade=t=in:st=0:d={fade:.3f},"
                      f"afade=t=out:st={max(0.0, d - fade):.3f}:d={fade:.3f}[a{i}]")
    n = len(blocks)
    cat_a = "".join(f"[a{i}]" for i in range(n))
    a_only = aparts + [f"{cat_a}concat=n={n}:v=0:a=1[acat]"]
    pre = channel_fix(inputs, a_only, "acat")
    measured = measure_loudness(inputs, a_only, "acat", pre)

    out_fps = 30 if (src_fps and src_fps > 31) else None
    if measured:
        ln = (f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11:measured_I={measured['input_i']}:"
              f"measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}:"
              f"measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true")
    else:
        ln = f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11"
    graph = ";".join(vparts + aparts + [
        "".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vcat][acat0]",
        _video_filter(plan, ass_path, out_fps, zoom) + "[vout]",
        f"[acat0]{pre}highpass=f=70,{ln},{PEAK_LIMITER}[aout]"])
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RenderError(r.stderr[-600:])
    return {"loudness_two_pass": bool(measured), "mono_channel_copied": bool(pre), "measured_lufs_before": measured["input_i"] if measured else None,
            "target_lufs": TARGET_LUFS, "crop_mode": plan.mode, "zoom_windows": len(zoom or []), **plan.facts}


def _render_silent(source: str, out: str, blocks: list, plan: CropPlan, ass_path: Optional[str],
                   src_fps: Optional[float], timeout: float, zoom: Optional[list] = None, crf: int = 20,
                   preset: str = "fast") -> dict:
    """Source without an audio track: render the picture and add a silent
    stereo track, so every clip plays the same way in players and apps."""
    inputs: list = []
    for b0, b1 in blocks:
        inputs += ["-ss", f"{b0:.3f}", "-t", f"{b1 - b0:.3f}", "-i", source]
    n = len(blocks)
    total = sum(b1 - b0 for b0, b1 in blocks)
    inputs += ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
    vparts = [f"[{i}:v]setpts=PTS-STARTPTS[v{i}]" for i in range(n)]
    out_fps = 30 if (src_fps and src_fps > 31) else None
    graph = ";".join(vparts + ["".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vcat]",
                               _video_filter(plan, ass_path, out_fps, zoom) + "[vout]"])
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[vout]", "-map", f"{n}:a",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", "-ar", "44100", "-shortest", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RenderError(r.stderr[-600:])
    return {"loudness_two_pass": False, "measured_lufs_before": None, "target_lufs": None,
            "silent_source": True, "crop_mode": plan.mode, **plan.facts}


# ------------------------------------------------------------------ masters, captions, joining

MASTER_CRF = 17     # the caption-free master is encoded once more when captions are burned on: keep it near-lossless
# Two encodes per clip instead of one: both use the faster x264 preset, so the pair costs about what the old
# single "fast" encode did (measured: a 44 s clip, 71 s CPU before; see the PR notes).
MASTER_PRESET = "veryfast"
BURN_PRESET, BURN_CRF = "veryfast", 19


def burn_captions(master: str, out: str, ass_path: Optional[str], timeout: float = 600,
                  encode: Optional[dict] = None) -> None:
    """Captions burned onto a caption-free master; the sound is copied untouched unless `encode` (a platform's
    export settings, see platforms.PLATFORMS) asks for another sample rate."""
    vf = "setsar=1"
    if ass_path:
        safe = ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        vf = f"subtitles='{safe}'"
    enc = encode or {}
    extra = []
    if enc.get("fps"):
        vf = f"fps={enc['fps']}," + vf
    if enc.get("maxrate"):
        extra += ["-maxrate", enc["maxrate"], "-bufsize", enc.get("bufsize") or enc["maxrate"]]
    if enc.get("keyint"):
        extra += ["-g", str(enc["keyint"]), "-keyint_min", str(enc["keyint"]), "-sc_threshold", "0"]
    audio = ["-c:a", "aac", "-b:a", "192k", "-ar", str(enc["audio_rate"])] \
        if enc.get("audio_rate") and enc["audio_rate"] != 44100 else ["-c:a", "copy"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", master, "-vf", vf, "-c:v", "libx264", "-preset", BURN_PRESET,
           "-crf", str(BURN_CRF), *extra,
           "-profile:v", "high", "-pix_fmt", "yuv420p", *audio, "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RenderError(r.stderr[-600:])


def join_clips(first: str, second: str, out: str, timeout: float = 300) -> None:
    """Two pieces rendered with the same settings, played back to back (no re-encode)."""
    listing = out + ".txt"
    with open(listing, "w") as f:
        f.write(f"file '{first}'\nfile '{second}'\n")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy",
                        "-movflags", "+faststart", out], capture_output=True, text=True, timeout=timeout)
    os.remove(listing)
    if r.returncode != 0:
        raise RenderError(r.stderr[-600:])
