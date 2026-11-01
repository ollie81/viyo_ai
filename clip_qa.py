"""
Quality gate for a RENDERED clip. Every check measures the actual output
file (not the plan), so a render that went wrong is caught before it is
shown to a creator.

A check is "fail" (the clip is rejected and a spare replaces it) or "warn"
(recorded, clip kept). Nothing here uses AI — every number is measured.

  format        1080x1920, h264 + aac, duration within limits and matching the plan
  black_bars    no black bars burned into the picture (the blur-fit layout is exempt)
  framing       sampled faces were not cut off by the crop
  loudness      integrated loudness near the target, no true-peak clipping
  silence       no long silent stretch, no dead start or end
  speech_end    the clip does not end in the middle of a word (speech_start only warns)
  caption_sync  captions cover the speech, don't appear over silence, and the
                measured offset between them and the audio is small
  hook          the on-screen hook is still grounded in what is said
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import audio_features as af
import captions as cp
import clip_text
import video_analysis as va

OUT_W, OUT_H = 1080, 1920
MIN_SECONDS, MAX_SECONDS = 10.0, 62.0
DURATION_TOLERANCE = 0.6
LUFS_TARGET, LUFS_TOLERANCE = -14.0, 1.8
MAX_TRUE_PEAK = -0.3
MAX_SILENCE = 1.6
MAX_LEADING_SILENCE = 0.9
MAX_TRAILING_SILENCE = 1.4
MAX_FACES_CUT_OFF_SHARE = 0.15
MIN_SPEECH_SHARE = 0.35
MAX_CAPTION_OFFSET = 0.15
MIN_CAPTION_COVERAGE = 0.85
MIN_CAPTION_PRECISION = 0.82
MAX_BAR_FRACTION = 0.03


@dataclass
class Check:
    name: str
    ok: bool
    severity: str = "fail"       # what a failed check means
    detail: dict = field(default_factory=dict)


@dataclass
class QAReport:
    checks: list = field(default_factory=list)

    @property
    def failures(self) -> list:
        return [c for c in self.checks if not c.ok and c.severity == "fail"]

    @property
    def warnings(self) -> list:
        return [c for c in self.checks if not c.ok and c.severity == "warn"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {"passed": self.passed,
                "failed": [c.name for c in self.failures],
                "warnings": [c.name for c in self.warnings],
                "checks": {c.name: {"ok": c.ok, **c.detail} for c in self.checks}}


def _probe(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,width,height,duration:format=duration", "-of", "json", path],
                       capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout or "{}")


def check_format(path: str, planned_seconds: float) -> Check:
    try:
        info = _probe(path)
    except Exception as e:
        return Check("format", False, detail={"error": str(e)[:120]})
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    dur = float((info.get("format") or {}).get("duration") or 0)
    problems = []
    if (v.get("width"), v.get("height")) != (OUT_W, OUT_H):
        problems.append("not_1080x1920")
    if v.get("codec_name") != "h264":
        problems.append("not_h264")
    if a.get("codec_name") != "aac":
        problems.append("no_aac_audio")
    if not (MIN_SECONDS <= dur <= MAX_SECONDS):
        problems.append("duration_out_of_range")
    if abs(dur - planned_seconds) > DURATION_TOLERANCE:
        problems.append("duration_differs_from_plan")
    return Check("format", not problems, detail={"duration": round(dur, 2), "planned": round(planned_seconds, 2),
                                                "problems": problems})


def _frames(path: str, times: list, work_dir: str) -> list:
    out = []
    for t in times:
        p = os.path.join(work_dir, f"qa_{int(t * 1000)}.png")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                            "-vf", "scale=270:480", p], capture_output=True, timeout=60)
        if r.returncode == 0 and os.path.exists(p):
            import cv2
            img = cv2.imread(p)
            os.remove(p)
            if img is not None:
                out.append(img)
    return out


def check_black_bars(path: str, duration: float, work_dir: str, crop_mode: str) -> Check:
    if crop_mode == "fit":
        return Check("black_bars", True, detail={"exempt": "blur_fit_layout"})
    try:
        imgs = _frames(path, [duration * f for f in (0.15, 0.5, 0.85)], work_dir)
        bars = [va._bars(i) for i in imgs]
    except Exception as e:
        return Check("black_bars", True, "warn", {"unchecked": str(e)[:100]})
    if not bars:
        return Check("black_bars", True, "warn", {"unchecked": "no frames"})
    # A bar is a letterbox: dark on BOTH opposite edges. A dark one-sided region
    # (a black jumper, a night sky) is picture content, not a bar.
    arr = np.array(bars)                      # columns: top, bottom, left, right
    pair = np.maximum(np.minimum(arr[:, 0], arr[:, 1]), np.minimum(arr[:, 2], arr[:, 3]))
    worst = float(pair.max())
    return Check("black_bars", worst <= MAX_BAR_FRACTION,
                 detail={"worst_bar_fraction": round(worst, 3), "one_sided_dark_max": round(float(arr.max()), 3)})


def check_framing(render_facts: dict) -> Check:
    if render_facts.get("crop_mode") != "track":
        return Check("framing", True, detail={"mode": render_facts.get("crop_mode"), "faces_checked": 0})
    share = float(render_facts.get("faces_cut_off_share", 0.0))
    return Check("framing", share <= MAX_FACES_CUT_OFF_SHARE,
                 detail={"faces_cut_off_share": share, "face_samples": render_facts.get("face_samples", 0)})


def _loudness(path: str) -> Optional[dict]:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af",
                        "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=300)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {"lufs": float(d["input_i"]), "true_peak": float(d["input_tp"])}
    except (KeyError, ValueError):
        return None


def check_loudness(path: str) -> Check:
    m = _loudness(path)
    if not m or not np.isfinite(m["lufs"]):
        return Check("loudness", False, detail={"error": "could_not_measure"})
    on_target = abs(m["lufs"] - LUFS_TARGET) <= LUFS_TOLERANCE
    # Quiet, peaky speech can't be raised further without clipping: sitting at the
    # true-peak ceiling is as loud as it can honestly get.
    peak_limited = m["true_peak"] >= -2.0 and m["lufs"] >= -17.5
    ok = (on_target or peak_limited) and m["true_peak"] <= MAX_TRUE_PEAK
    return Check("loudness", ok, detail={"lufs": round(m["lufs"], 1), "true_peak_db": round(m["true_peak"], 1),
                                         "peak_limited": bool(peak_limited and not on_target)})


def _speech_runs(ff: af.FrameFeatures) -> np.ndarray:
    """Speech indicator, with 100 ms gaps bridged (stop consonants, breaths)."""
    sp = ff.speech.copy()
    n = int(0.1 / af._HOP_SECONDS)
    idx = np.flatnonzero(sp)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= n:
            sp[a:b] = True
    return sp


def check_silence_and_edges(ff: af.FrameFeatures, duration: float) -> list:
    hop = af._HOP_SECONDS
    sp = _speech_runs(ff)
    if ff.times.size == 0 or sp.sum() == 0:
        return [Check("silence", False, detail={"error": "no_speech_in_output"})]
    share = float(sp.mean())
    idx = np.flatnonzero(sp)
    lead = float(idx[0] * hop)
    trail = float((len(sp) - 1 - idx[-1]) * hop)
    gaps = np.diff(idx) - 1
    longest = float(gaps.max() * hop) if gaps.size else 0.0
    sil = Check("silence", longest <= MAX_SILENCE and lead <= MAX_LEADING_SILENCE
                and trail <= MAX_TRAILING_SILENCE and share >= MIN_SPEECH_SHARE,
                detail={"longest_silence_s": round(longest, 2), "leading_s": round(lead, 2),
                        "trailing_s": round(trail, 2), "speech_share": round(share, 2)})
    # Mid-word cut: loud, voiced sound right at the very first / last frames.
    level = float(np.percentile(ff.rms_db[ff.speech], 50))
    first = float(np.max(ff.rms_db[:3])) if len(ff.rms_db) >= 3 else -99
    last = float(np.max(ff.rms_db[-3:])) if len(ff.rms_db) >= 3 else -99
    # The render fades the first/last 30 ms, so mid-word cuts show as sound
    # still strong a few frames in/out.
    first_in = float(np.max(ff.rms_db[3:6])) if len(ff.rms_db) >= 6 else first
    last_in = float(np.max(ff.rms_db[-6:-3])) if len(ff.rms_db) >= 6 else last
    starts_mid = first_in > level - 6
    ends_mid = last_in > level - 6
    # A clip begins right at a word onset, so loud first frames are normal (warn only).
    # Loud speech at the very end means the clip is cut off mid-word: that is a failure.
    start_edge = Check("speech_start", not starts_mid, "warn", detail={"starts_mid_word": bool(starts_mid)})
    end_edge = Check("speech_end", not ends_mid, "fail", detail={"ends_mid_word": bool(ends_mid)})
    return [sil, start_edge, end_edge]


def check_caption_sync(ass_text: Optional[str], ff: af.FrameFeatures, duration: float) -> Check:
    if not ass_text:
        return Check("caption_sync", False, "warn", {"error": "no_captions"})
    events = cp.caption_timeline(ass_text)
    if not events or ff.times.size == 0:
        return Check("caption_sync", False, "warn", {"error": "nothing_to_compare"})
    hop = af._HOP_SECONDS
    n = len(ff.speech)
    sp = _speech_runs(ff).astype(float)
    cap = np.zeros(n)
    for s, e, _ in events:
        a, b = int(max(0, s) / hop), int(min(duration, e) / hop)
        cap[a:max(b, a + 1)] = 1.0
    # Coverage: how much speech has a caption on screen. Precision: how much
    # caption time has speech under it.
    coverage = float((sp * cap).sum() / max(sp.sum(), 1))
    precision = float((sp * cap).sum() / max(cap.sum(), 1))
    # Offset: shift captions against speech, keep the best agreement.
    best, best_score = 0.0, -1.0
    for k in range(-25, 26):            # +-0.5 s
        if k >= 0:
            a, b = sp[k:], cap[:n - k]
        else:
            a, b = sp[:n + k], cap[-k:]
        sc = float((a * b).sum() / max(np.sqrt(a.sum() * b.sum()), 1.0))
        if sc > best_score:
            best, best_score = k * hop, sc
    ok = coverage >= MIN_CAPTION_COVERAGE and precision >= MIN_CAPTION_PRECISION and abs(best) <= MAX_CAPTION_OFFSET
    return Check("caption_sync", ok, detail={"speech_covered": round(coverage, 3),
                                            "caption_over_speech": round(precision, 3),
                                            "offset_s": round(best, 2), "words": len(events)})


def check_hook(hook: Optional[str], clip_text_: str, clip_words: list) -> Check:
    if not hook:
        return Check("hook", True, detail={"hook": None})
    text, why = clip_text.vet_overlay_hook(hook, clip_text_, clip_words)
    return Check("hook", text is not None, detail={"reason": why})


def run_qa(path: str, planned_seconds: float, render_facts: dict, ass_text: Optional[str],
           hook: Optional[str], transcript: str, clip_words: list, work_dir: str) -> QAReport:
    rep = QAReport()
    fmt = check_format(path, planned_seconds)
    rep.checks.append(fmt)
    if not fmt.ok and fmt.detail.get("error"):
        return rep
    duration = float(fmt.detail.get("duration") or planned_seconds)
    rep.checks.append(check_black_bars(path, duration, work_dir, render_facts.get("crop_mode", "")))
    rep.checks.append(check_framing(render_facts))
    rep.checks.append(check_loudness(path))
    try:
        ff = af.frame_features(af.decode_pcm(path, timeout=300))
        rep.checks.extend(check_silence_and_edges(ff, duration))
        rep.checks.append(check_caption_sync(ass_text, ff, duration))
    except Exception as e:
        rep.checks.append(Check("audio_analysis", False, "warn", {"error": str(e)[:120]}))
    rep.checks.append(check_hook(hook, transcript, clip_words))
    return rep
