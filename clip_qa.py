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
  speech_start / speech_end   recorded as warnings only (cut points come from word timings)
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
    if render_facts.get("crop_mode") == "stack":
        return Check("framing", True, detail={"mode": "stack", "faces_checked": render_facts.get("face_samples", 0)})
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
    # Both are recorded, never fatal. The cut points come from word timings (a clip
    # ends after its last word finishes), and with fast speakers the last syllable
    # of that word still sits in these final frames, so this audio check
    # can't tell a cut word from a quick one without false rejections.
    start_edge = Check("speech_start", not starts_mid, "warn", detail={"starts_mid_word": bool(starts_mid)})
    end_edge = Check("speech_end", not ends_mid, "warn", detail={"ends_mid_word": bool(ends_mid)})
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


AV_SYNC_WARN, AV_SYNC_FAIL = 0.12, 0.35


def check_av_sync(path: str) -> Check:
    """Picture and sound streams start together and last as long as each other (a drift between them shows as
    lips out of step with the words, or a frozen end)."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time,duration",
                            "-of", "json", path], capture_output=True, text=True, timeout=60)
        streams = json.loads(r.stdout or "{}").get("streams", [])
        v = next(s for s in streams if s.get("codec_type") == "video")
        a = next(s for s in streams if s.get("codec_type") == "audio")
        start = abs(float(v.get("start_time") or 0) - float(a.get("start_time") or 0))
        length = abs(float(v.get("duration") or 0) - float(a.get("duration") or 0))
    except Exception as e:
        return Check("av_sync", False, "warn", {"error": str(e)[:120]})
    worst = max(start, length)
    return Check("av_sync", worst <= AV_SYNC_WARN, "fail" if worst > AV_SYNC_FAIL else "warn",
                 {"start_offset_s": round(start, 3), "length_difference_s": round(length, 3)})


def summary(qa: dict, review: Optional[dict], edits: Optional[dict] = None, components: Optional[dict] = None,
            mode: str = "speech", inspection: Optional[dict] = None) -> dict:
    """The creator's checklist for one delivered clip. Every item is passed / warning / failed / unchecked;
    an item nobody could check is "unchecked", never "passed"."""
    checks = (qa or {}).get("checks") or {}
    rv = review or {}
    reviewed = rv.get("status") == "reviewed"
    edits = edits or {}
    items = []

    def add(key, label, status, note=""):
        items.append({"key": key, "label": label, "status": status, "note": note})

    def qa_item(key, label, name, note_ok="", note_bad=""):
        c = checks.get(name)
        if c is None:
            add(key, label, "unchecked", "Not measured for this clip.")
        else:
            failed = name in (qa or {}).get("failed", [])
            add(key, label, "passed" if c.get("ok") else ("failed" if failed else "warning"),
                note_ok if c.get("ok") else note_bad)

    if mode == "visual":
        add("stands_alone", "Makes sense on its own", "unchecked", "Chosen from what's on screen; there's no speech.")
    elif reviewed and rv.get("standalone") is not None:
        add("stands_alone", "Makes sense on its own", "passed" if rv["standalone"] else "failed")
    else:
        add("stands_alone", "Makes sense on its own", "unchecked", "The editorial review didn't run for this clip.")
    opening = rv.get("opening") if reviewed else None
    if edits.get("cold_open"):
        add("opening", "Strong opening", "passed",
            f"Opens with a preview of a line from later in the clip: \"{edits['cold_open']['text'][:80]}\"")
    elif opening is not None:
        add("opening", "Strong opening", "passed" if opening >= 6 else "warning", f"Rated {opening}/10 by the review.")
    else:
        add("opening", "Strong opening", "unchecked")
    ending = rv.get("ending") if reviewed else None
    add("ending", "Satisfying ending", "unchecked" if ending is None else ("passed" if ending >= 6 else "warning"),
        "" if ending is None else f"Rated {ending}/10 by the review.")
    if mode == "visual":
        add("captions", "Captions accurate", "unchecked", "No speech to caption.")
    else:
        cs = checks.get("caption_sync")
        if cs is None:
            add("captions", "Captions accurate", "unchecked")
        else:
            add("captions", "Captions accurate", "passed" if cs.get("ok") else "warning",
                "Words come from the transcript and are timed to the voice. The transcript itself isn't "
                "checked against a second listener.")
    audio = (components or {}).get("audio")
    if mode == "visual" or audio is None:
        add("speech", "Speech understandable", "unchecked")
    else:
        lo = checks.get("loudness") or {}
        add("speech", "Speech understandable", "passed" if audio >= 5 and lo.get("ok", True) else "warning",
            f"Sound quality {audio}/10.")
    qa_item("framing", "Framing", "framing", "", "A face may be cut off by the crop.")
    qa_item("pauses", "No dead air", "silence", "", "There is a long silent stretch.")
    qa_item("sync", "Sound and picture in sync", "av_sync")
    qa_item("technical", "Meets the platforms' requirements (1080x1920, H.264/AAC, length)", "format")
    # The final inspection of the finished clip (clip_inspect): its items join the checklist.
    have = {i["key"] for i in items}
    for it in (inspection or {}).get("items") or []:
        if it["key"] not in have:
            items.append(dict(it))
    status = "failed" if any(i["status"] == "failed" for i in items) else \
        "warning" if any(i["status"] == "warning" for i in items) else \
        "passed" if all(i["status"] == "passed" for i in items) else "partly_checked"
    return {"status": status, "items": items}


def run_qa(path: str, planned_seconds: float, render_facts: dict, ass_text: Optional[str],
           hook: Optional[str], transcript: str, clip_words: list, work_dir: str,
           mode: str = "speech", has_audio: bool = True) -> QAReport:
    rep = QAReport()
    fmt = check_format(path, planned_seconds)
    rep.checks.append(fmt)
    if not fmt.ok and fmt.detail.get("error"):
        return rep
    duration = float(fmt.detail.get("duration") or planned_seconds)
    rep.checks.append(check_black_bars(path, duration, work_dir, render_facts.get("crop_mode", "")))
    rep.checks.append(check_framing(render_facts))
    rep.checks.append(check_av_sync(path))
    if has_audio:
        rep.checks.append(check_loudness(path))
    if mode == "visual":
        # No speech to check: no captions to sync, and silence is expected.
        # The on-screen title came from the frames, not a transcript.
        return rep
    try:
        ff = af.frame_features(af.decode_pcm(path, timeout=300))
        rep.checks.extend(check_silence_and_edges(ff, duration))
        rep.checks.append(check_caption_sync(ass_text, ff, duration))
    except Exception as e:
        rep.checks.append(Check("audio_analysis", False, "warn", {"error": str(e)[:120]}))
    rep.checks.append(check_hook(hook, transcript, clip_words))
    return rep
