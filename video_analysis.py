"""
Whole-video visual analysis for the AI Repurposer — local, no paid API.

One pass over the source at SAMPLE_FPS frames/second (downscaled to
FRAME_W px wide) records, per sample: sharpness, brightness, faces
(YuNet; Haar fallback), a colour histogram, and whether the picture cut
to something different. From that:
  * shots (runs between camera cuts) and shot clusters (same camera /
    same person on screen) — the basis for speaker estimation;
  * letterbox / pillarbox bars to crop away;
  * per-window visual quality for clip scoring.

Everything degrades to "unknown" rather than guessing: a detector that
fails or a video that can't be decoded yields an empty timeline, which
callers treat as "no visual signal", never as a good or bad score.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a pinned dependency
    cv2 = None

SAMPLE_FPS = 1.0
FRAME_W = 320
YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yunet.onnx")
FACE_SCORE_MIN = 0.7
# Histogram distance (Bhattacharyya, 0 = identical, 1 = unrelated) above
# (measured on real footage: true cuts score 0.33-0.38, everything else<=0.12)
# which consecutive samples count as a camera cut.
CUT_DISTANCE = 0.2
# Two shots show "the same camera" when their mean histograms are this close.
SAME_CAMERA_DISTANCE = 0.22
# A row/column is "black bar" when it is this dark and this uniform.
BAR_MEAN_MAX = 18
BAR_STD_MAX = 7
MIN_BAR_FRACTION = 0.03


@dataclass
class Face:
    cx: float   # centre, 0-1 of frame width
    cy: float
    w: float    # size, 0-1 of frame width / height
    h: float
    score: float


@dataclass
class Shot:
    start: float
    end: float
    first: int           # sample index range [first, last]
    last: int
    cluster: int = -1    # same value = same camera / same person on screen


@dataclass
class VisualTimeline:
    times: np.ndarray
    sharpness: np.ndarray
    brightness: np.ndarray
    faces: list = field(default_factory=list)       # list[list[Face]] per sample
    hist: np.ndarray = field(default_factory=lambda: np.zeros((0, 48)))
    cut_before: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    shots: list = field(default_factory=list)
    frame_w: int = 0
    frame_h: int = 0
    bars: dict = field(default_factory=dict)        # fractions of the frame: top/bottom/left/right
    detector: str = "none"

    @property
    def usable(self) -> bool:
        return len(self.times) >= 3

    def _slice(self, start: float, end: float) -> slice:
        lo, hi = np.searchsorted(self.times, [start, end])
        return slice(int(lo), max(int(hi), int(lo) + 1))

    def cuts_between(self, start: float, end: float) -> list:
        """Times (seconds) of detected camera cuts inside [start, end)."""
        if not self.usable:
            return []
        sl = self._slice(start, end)
        return [float(t) for t, c in zip(self.times[sl], self.cut_before[sl]) if c and t > start]

    def shot_at(self, t: float) -> Optional[Shot]:
        for s in self.shots:
            if s.start <= t < s.end + 1e-6:
                return s
        return self.shots[-1] if self.shots else None


def probe_video(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:stream_side_data=rotation:stream_tags=rotate",
         "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True, timeout=60).stdout
    info = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    w, h = int(info.get("width", 0) or 0), int(info.get("height", 0) or 0)
    rot = abs(int(float(info.get("rotation", info.get("TAG:rotate", 0)) or 0))) % 180
    if rot == 90:
        w, h = h, w
    return {"width": w, "height": h}


_yunet = None


def _get_yunet(frame_w: int, frame_h: int):
    global _yunet
    if cv2 is None or not hasattr(cv2, "FaceDetectorYN") or not os.path.exists(YUNET_PATH):
        return None
    try:
        if _yunet is None:
            _yunet = cv2.FaceDetectorYN.create(YUNET_PATH, "", (frame_w, frame_h), FACE_SCORE_MIN, 0.3, 50)
        _yunet.setInputSize((frame_w, frame_h))
        return _yunet
    except Exception:
        return None


_haar = None


def detect_faces(frame_bgr: np.ndarray) -> tuple[list, str]:
    """Faces in a BGR frame, as normalised Face objects, and the detector used."""
    h, w = frame_bgr.shape[:2]
    det = _get_yunet(w, h)
    if det is not None:
        try:
            _, res = det.detect(frame_bgr)
            faces = []
            if res is not None:
                for r in res:
                    x, y, fw, fh, score = float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[14])
                    x, y = max(0.0, x), max(0.0, y)
                    fw, fh = min(fw, w - x), min(fh, h - y)
                    if fw > 4 and fh > 4 and score >= FACE_SCORE_MIN:
                        faces.append(Face((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h, score))
            return faces, "yunet"
        except Exception:
            pass
    if cv2 is None:
        return [], "none"
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    found = _haar.detectMultiScale(gray, 1.1, 4, minSize=(max(20, w // 16), max(20, w // 16)))
    return [Face((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h, 0.6) for x, y, fw, fh in found], "haar"


def _hist(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    parts = [cv2.calcHist([hsv], [c], None, [16], r).flatten() for c, r in ((0, [0, 180]), (1, [0, 256]), (2, [0, 256]))]
    h = np.concatenate(parts).astype(np.float32)
    return h / max(h.sum(), 1e-6) * 3  # each channel sums to 1


def _bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a / max(a.sum(), 1e-9), b / max(b.sum(), 1e-9)
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)))))


def _bars(frame_bgr: np.ndarray) -> tuple[float, float, float, float]:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape

    def run(lines_mean, lines_std, total):
        n = 0
        for m, s in zip(lines_mean, lines_std):
            if m <= BAR_MEAN_MAX and s <= BAR_STD_MAX:
                n += 1
            else:
                break
        return n / total

    rm, rs = g.mean(axis=1), g.std(axis=1)
    cm, cs_ = g.mean(axis=0), g.std(axis=0)
    return (run(rm, rs, h), run(rm[::-1], rs[::-1], h), run(cm, cs_, w), run(cm[::-1], cs_[::-1], w))


def analyze_video(path: str, duration: float, fps: float = SAMPLE_FPS,
                  timeout: float = 1800) -> VisualTimeline:
    """Never raises: any failure returns an unusable (empty) timeline."""
    empty = VisualTimeline(np.zeros(0), np.zeros(0), np.zeros(0))
    if cv2 is None:
        return empty
    try:
        dims = probe_video(path)
        if dims["width"] <= 0 or dims["height"] <= 0:
            return empty
        fw = FRAME_W
        fh = int(round(fw * dims["height"] / dims["width"] / 2)) * 2
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-an",
             "-vf", f"fps={fps},scale={fw}:{fh}:flags=fast_bilinear,setsar=1",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        size = fw * fh * 3
        times, sharp, bright, faces, hists, bar_rows = [], [], [], [], [], []
        detector = "none"
        i = 0
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(fh, fw, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            bright.append(float(gray.mean()))
            f, detector = detect_faces(frame)
            faces.append(f)
            hists.append(_hist(frame))
            bar_rows.append(_bars(frame))
            times.append(i / fps)
            i += 1
        proc.stdout.close()
        proc.wait(timeout=30)
        if len(times) < 3:
            return empty

        hist = np.array(hists)
        cut = np.zeros(len(times), dtype=bool)
        for k in range(1, len(times)):
            cut[k] = _bhattacharyya(hist[k - 1], hist[k]) > CUT_DISTANCE
        bars_arr = np.median(np.array(bar_rows), axis=0)
        bars = {"top": float(bars_arr[0]), "bottom": float(bars_arr[1]),
                "left": float(bars_arr[2]), "right": float(bars_arr[3])}
        tl = VisualTimeline(np.array(times), np.array(sharp), np.array(bright), faces, hist, cut,
                            [], fw, fh, bars, detector)
        tl.shots = build_shots(tl)
        return tl
    except Exception as e:
        print(f"[WARN] visual analysis unavailable: {e}")
        return empty


def build_shots(tl: VisualTimeline) -> list:
    """Shots between cuts, with cluster ids shared by shots from the same camera."""
    if not tl.usable:
        return []
    starts = [0] + [i for i in range(1, len(tl.times)) if tl.cut_before[i]]
    shots = []
    for n, a in enumerate(starts):
        b = (starts[n + 1] - 1) if n + 1 < len(starts) else len(tl.times) - 1
        shots.append(Shot(float(tl.times[a]), float(tl.times[b]) + 1.0 / SAMPLE_FPS, a, b))
    centroids: list[np.ndarray] = []
    for s in shots:
        h = tl.hist[s.first: s.last + 1].mean(axis=0)
        best, best_d = -1, 9.0
        for ci, c in enumerate(centroids):
            d = _bhattacharyya(h, c)
            if d < best_d:
                best, best_d = ci, d
        if best >= 0 and best_d <= SAME_CAMERA_DISTANCE:
            s.cluster = best
        else:
            centroids.append(h)
            s.cluster = len(centroids) - 1
    return shots


def crop_rect_for_bars(tl: VisualTimeline, src_w: int, src_h: int) -> Optional[tuple[int, int, int, int]]:
    """(x, y, w, h) in source pixels that removes letterbox/pillarbox bars,
    or None when there are no meaningful bars."""
    if not tl.usable or not tl.bars:
        return None
    t, b, l, r = (tl.bars.get(k, 0.0) for k in ("top", "bottom", "left", "right"))
    if max(t + b, l + r) < MIN_BAR_FRACTION * 2 and max(t, b, l, r) < MIN_BAR_FRACTION:
        return None
    # A hair inside the detected edge, so anti-aliased bar pixels never show.
    pad_y, pad_x = 0.004, 0.004
    x0 = int(round((l + (pad_x if l else 0)) * src_w / 2) * 2)
    y0 = int(round((t + (pad_y if t else 0)) * src_h / 2) * 2)
    x1 = src_w - int(round((r + (pad_x if r else 0)) * src_w / 2) * 2)
    y1 = src_h - int(round((b + (pad_y if b else 0)) * src_h / 2) * 2)
    w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    if w < src_w * 0.4 or h < src_h * 0.4:
        return None
    return x0, y0, w, h


# ---------------------------------------------------------------- per-window scores

def window_visual(tl: VisualTimeline, start: float, end: float) -> dict:
    """Measured facts about the picture in [start, end]; None when unknown."""
    if not tl.usable:
        return {"available": False}
    sl = tl._slice(start, end)
    sharp = tl.sharpness[sl]
    bright = tl.brightness[sl]
    fs = tl.faces[sl.start: sl.stop]
    if len(sharp) == 0:
        return {"available": False}
    face_present = [bool(f) for f in fs]
    biggest = [max((x.w for x in f), default=0.0) for f in fs]
    return {
        "available": True,
        "samples": int(len(sharp)),
        "sharpness_median": float(np.median(sharp)),
        "brightness_median": float(np.median(bright)),
        "face_ratio": float(np.mean(face_present)),
        "face_width_median": float(np.median([b for b in biggest if b > 0])) if any(b > 0 for b in biggest) else 0.0,
        "multi_face_ratio": float(np.mean([len(f) >= 2 for f in fs])),
        "cuts": len(tl.cuts_between(start, end)),
        "bars": dict(tl.bars),
    }


def visual_quality_score(v: dict) -> tuple[Optional[float], dict]:
    """0-10 from measured sharpness, exposure, subject presence and size.
    (None, ...) when there is no visual signal — never a made-up number."""
    if not v.get("available"):
        return None, {"available": False}
    sharp = v["sharpness_median"]
    s_sharp = float(np.clip((np.log10(max(sharp, 1.0)) - 1.0) / 1.5, 0, 1))      # ~10 -> 0, ~300 -> 1
    b = v["brightness_median"]
    s_exposure = float(np.clip(1.0 - abs(b - 125) / 100.0, 0, 1))
    s_subject = float(np.clip(v["face_ratio"], 0, 1))
    fw = v["face_width_median"]
    s_size = 0.0 if fw == 0 else float(np.clip(fw / 0.08, 0, 1)) if fw < 0.08 else float(np.clip((0.7 - fw) / 0.3, 0, 1))
    # Talking-head formats live and die on the subject being visible and
    # reasonably large; screen recordings/b-roll have no face, so their
    # score leans on sharpness and exposure alone.
    if v["face_ratio"] >= 0.3:
        score = 10 * (0.35 * s_sharp + 0.2 * s_exposure + 0.25 * s_subject + 0.2 * s_size)
    else:
        score = 10 * (0.6 * s_sharp + 0.4 * s_exposure)
    return round(score, 2), {"sharpness": round(s_sharp, 2), "exposure": round(s_exposure, 2),
                             "subject": round(s_subject, 2), "subject_size": round(s_size, 2)}
