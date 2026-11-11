"""Automatic voice quality recovery for talking clips.

After a talking clip is made, its voices are judged on two kinds of evidence:

* Measurements Viyo makes itself, which don't depend on an AI's opinion:
  ffmpeg sound levels (silent, too quiet, clipping), when speech is heard
  (ffmpeg silence detection: cut off at the end, far too fast), how many of
  the script's words appear in the transcript (missing or garbled lines),
  and each line's pitch compared with the character's approved recordings
  (a different person, or the wrong character speaking).
* An AI that listens to (and, when the clip is sent, watches) the clip:
  who speaks, naturalness, feeling, pronunciation, lip sync, noise.

Neither is trusted alone where they disagree: an AI "wrong voice" on a line
whose pitch matches the character's approved recording is softened, and a
big pitch change on a whispered, shouted or crying line is accepted as
performance, not counted as a new voice.

Then the cheapest fix that can work is tried, at most a few times:
  clean    the sound only (ffmpeg: noise reduction, level) - free
  resync   lip sync the clip to its own sound (only lip sync was wrong)
  retry    the same video model again, with exact corrections
  switch   another video model that can speak this language
  saved    the saved character voices, lip synced onto the clip
The dialogue text is never changed: corrections are added beside it.

Limits that are NOT hidden: none of the video models takes a voice reference
or a voice ID, so a character's voice in a new clip can only be described,
then checked against the approved recordings. Pitch is a coarse measure (it
catches a different sex or age, not two similar adults). Lip sync and
naturalness are an AI's judgement. Identical voices are not guaranteed.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

import studio_voice as voice_mod

# Problems worth paying to fix automatically. Others ("performance": one
# line's feeling) are shown to the creator; a re-made clip for them costs
# more than it is likely to help.
HARD = {"no_audio", "silent", "missing_lines", "incomplete_lines", "cut_off", "wrong_voice", "wrong_speaker",
        "identity", "language", "extra_speech", "lip_sync", "clipping"}
FIXABLE = HARD | {"unnatural", "clarity", "pronunciation", "same_voice", "identity_drift", "overacted", "rushed",
                  "lip_sync_some", "too_quiet", "noise"}
SOUND_ONLY = {"too_quiet", "noise"}

MAX_ATTEMPTS = 2           # automatic re-makes per clip (the cheap sound clean-up isn't counted)
MAX_USD = 2.00             # the most automatic re-makes may spend on one clip (Viyo's limit, not the creator's)


def max_attempts() -> int:
    try:
        return max(0, min(3, int(os.environ.get("STUDIO_VOICE_FIX_ATTEMPTS", MAX_ATTEMPTS))))
    except ValueError:
        return MAX_ATTEMPTS


def max_usd() -> float:
    try:
        return max(0.0, float(os.environ.get("STUDIO_VOICE_FIX_MAX_USD", MAX_USD)))
    except ValueError:
        return MAX_USD


def enabled() -> bool:
    return os.environ.get("STUDIO_VOICE_FIX", "on").lower() not in ("off", "false", "0") and max_attempts() > 0


def charge_all() -> bool:
    """STUDIO_VOICE_FIX_BILLING=all charges every re-make (within the price shown); the default charges a re-make
    only when its result is the one kept and it passed."""
    return os.environ.get("STUDIO_VOICE_FIX_BILLING", "").lower() == "all"


# ------------------------------------------------------------------ measuring

def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", (text or "").lower())


def recall(script: str, heard: str) -> float:
    """The share of the script's words found in the transcript, in order (1.0 = every word said)."""
    want, got = words(script), words(heard)
    if not want:
        return 1.0
    i = hits = 0
    for w in want:
        j = i
        while j < len(got) and got[j] != w:
            j += 1
        if j < len(got):
            hits += 1
            i = j + 1
    return hits / len(want)


def pitch_hz(samples, rate: int = 16000) -> Optional[float]:
    """Median voice pitch (Hz) of a mono 16-bit segment by autocorrelation, voiced frames only; None if unvoiced."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    if x.size < rate * 0.2:
        return None
    frame, hop = int(rate * 0.04), int(rate * 0.02)
    lo, hi = int(rate / 450), int(rate / 60)       # 60-450 Hz: speaking voices, child to bass
    out = []
    for k in range(0, x.size - frame - hi, hop):
        f = x[k:k + frame + hi]
        f = f - f.mean()
        energy = float(np.dot(f[:frame], f[:frame]))
        if energy < 1e3 * frame:                    # too quiet to be speech
            continue
        ac = np.array([np.dot(f[:frame], f[lag:lag + frame]) for lag in range(lo, hi)])
        top = float(ac.max())
        # The shortest lag that is (nearly) as strong as the best one: the period, not a multiple of it.
        best = next(j for j in range(len(ac)) if ac[j] >= 0.9 * top and
                    (j == 0 or ac[j] >= ac[j - 1]) and (j + 1 >= len(ac) or ac[j] >= ac[j + 1]))
        if ac[best] / energy > 0.45:
            out.append(rate / (lo + best))
    return float(np.median(out)) if len(out) >= 3 else None


def semitones(a: float, b: float) -> float:
    import math
    return abs(12 * math.log2(a / b))


def read_wav(path: str):
    import wave

    import numpy as np
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data, rate


def speech_spans(wav_path: str) -> dict:
    """When there is sound loud enough to be speech (ffmpeg silencedetect): total seconds, first and last moment."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav_path, "-af",
                          "silencedetect=noise=-35dB:d=0.25", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=120).stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    total = (int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))) if dur else None
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", out)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", out)]
    if total is None:
        return {}
    silent = 0.0
    for i, a in enumerate(starts):
        b = ends[i] if i < len(ends) else total
        silent += max(0.0, b - max(0.0, a))
    trailing = len(starts) > len(ends)          # still silent when the clip ends
    return {"clip_seconds": round(total, 2), "speech_seconds": round(max(0.0, total - silent), 2),
            "sound_at_end": not trailing}


def emotive(line: dict) -> bool:
    """Whispered, shouted, crying, terrified...: the voice may sound very different and still be the same person."""
    t = f"{line.get('emotion') or ''} {line.get('delivery') or ''}"
    return bool(voice_mod._LOUD.search(t) or voice_mod._SOFT.search(t) or voice_mod._STRONG.search(t) or
                re.search(r"\b(cry|crying|cries|sob|tearful|weeping|panic|scared|afraid)", t, re.I))


# ------------------------------------------------------------------ judging

def evaluate(tech: dict, heard: Optional[dict], lines: list[dict], measures: Optional[dict] = None) -> dict:
    """The full voice check: the listening verdict plus Viyo's own measurements, with the status recomputed."""
    check = voice_mod.verdict(tech, heard, lines)
    m = measures or {}
    problems = list(check["problems"])
    notes = []
    if tech.get("has_audio") and not tech.get("silent") and tech.get("mean_db") is not None and tech["mean_db"] < -35:
        problems.append({"kind": "too_quiet", "problem": f"The voices are very quiet ({tech['mean_db']:.0f} dB).",
                         "lines": []})
    if heard is not None:
        by = {x["index"]: x for x in heard["lines"]}
        # Words: what the transcript actually contains, not only the AI's yes/no.
        garbled, partial = [], []
        for i, d in enumerate(lines):
            x = by.get(i)
            if not x or not x["heard"]:
                continue
            r = recall(d.get("text") or "", x.get("heard_text") or "")
            if x.get("heard_text") and r < 0.5:
                garbled.append(i)
            elif (x.get("heard_text") and r < 0.8) or not x["pronunciation_ok"]:
                partial.append(i)
        if garbled:
            problems.append({"kind": "incomplete_lines", "problem": "Said only in part or with other words: " + "; ".join(
                f"{lines[i].get('character')}: \"{voice_mod._t(lines[i].get('text'), 60)}\"" for i in garbled[:4]),
                "lines": garbled})
        if partial:
            problems.append({"kind": "pronunciation", "problem": "Words unclear or mispronounced on line(s) " +
                             ", ".join(str(i + 1) for i in partial), "lines": partial})
        # Timing: speech still going when the clip ends, with the last line incomplete.
        if m.get("sound_at_end") and lines:
            last = len(lines) - 1
            x = by.get(last)
            if x and (not x["heard"] or recall(lines[last].get("text") or "", x.get("heard_text") or "") < 0.8):
                problems.append({"kind": "cut_off", "problem": "The last line is cut off by the end of the clip.",
                                 "lines": [last]})
        # Rushed: the clip had time for the lines, but the speech was squeezed into much less of it.
        if m.get("expected_seconds") and m.get("speech_seconds") is not None and \
                (m.get("clip_seconds") or 0) >= 0.9 * m["expected_seconds"] and \
                m["speech_seconds"] < 0.45 * m["expected_seconds"] and not any(p["kind"] in ("missing_lines", "silent")
                                                                              for p in problems):
            problems.append({"kind": "rushed", "problem": f"The lines are rushed ({m['speech_seconds']:.1f} s of speech "
                             f"for about {m['expected_seconds']:.1f} s of dialogue).", "lines": []})
        # Lip sync (the AI watching the clip).
        if m.get("watched") and heard.get("mouths_visible", True):
            off = [i for i, x in by.items() if x["heard"] and not x.get("lip_sync_ok", True)]
            if off:
                hard = len(off) * 2 >= max(1, len(lines))
                problems.append({"kind": "lip_sync" if hard else "lip_sync_some", "problem":
                                 "Mouths don't match the words on line(s) " + ", ".join(str(i + 1) for i in off),
                                 "lines": off})
        if heard.get("background_noise"):
            problems.append({"kind": "noise", "problem": "Background noise gets in the way of the words.", "lines": []})
        if heard.get("overacted"):
            problems.append({"kind": "overacted", "problem": "The delivery is more exaggerated than the scene calls for.",
                             "lines": []})
        # Identity: the approved recordings, the measured pitch and the AI's ear together.
        pitches, refs = m.get("line_pitch") or {}, m.get("ref_pitch") or {}
        wrong_voice = next((p for p in problems if p["kind"] == "wrong_voice"), None)
        cleared, drift, identity, wrong_speaker = [], [], [], []
        for i, d in enumerate(lines):
            k = voice_mod.key(d.get("character") or "")
            x = by.get(i) or {}
            hz, ref = pitches.get(i), refs.get(k)
            shift = semitones(hz, ref) if hz and ref else None
            limit = 8.0 if emotive(d) else 4.5
            # The wrong character speaking: the line's pitch sits on another character's approved voice.
            others = [(semitones(hz, r), n) for n, r in refs.items() if n != k and hz and r]
            if shift is not None and others:
                near, who = min(others)
                if near + 3 < shift and shift > 4.5:
                    wrong_speaker.append((i, who))
                    continue
            if wrong_voice and i in wrong_voice["lines"]:
                if x.get("same_person") is True and emotive(d):
                    cleared.append(i)                   # a feeling, not a new voice
                    notes.append(f"line {i + 1}: the voice changes with the feeling"
                                 + (f" ({shift:.0f} semitones)" if shift is not None else "") + ", accepted")
                elif shift is not None and shift <= 2.0 and x.get("same_person") is not False:
                    cleared.append(i)                   # pitch matches the approved voice: only doubt
                    drift.append(i)
                continue
            if x.get("same_person") is False and not emotive(d):
                identity.append(i)
            elif shift is not None and shift > limit:
                (identity if x.get("same_person") is False else drift).append(i)
            elif shift is not None and shift > 4.5:
                notes.append(f"line {i + 1}: the voice changes with the feeling ({shift:.0f} semitones), accepted")
        if wrong_voice and cleared:
            wrong_voice["lines"] = [i for i in wrong_voice["lines"] if i not in cleared]
            if not wrong_voice["lines"]:
                problems.remove(wrong_voice)
        if wrong_speaker:
            problems.append({"kind": "wrong_speaker", "problem": "The wrong character seems to speak: " + ", ".join(
                f"line {i + 1} ({lines[i].get('character')}) sounds like {who.title()}" for i, who in wrong_speaker),
                "lines": [i for i, _w in wrong_speaker]})
        if identity:
            problems.append({"kind": "identity", "problem": "Not the same voice as before: " + ", ".join(
                f"line {i + 1} ({lines[i].get('character')})" for i in identity), "lines": identity})
        if drift:
            problems.append({"kind": "identity_drift", "problem": "The voice drifts from the approved one on line(s) " +
                             ", ".join(str(i + 1) for i in drift), "lines": drift})
    status = check["status"]
    if heard is not None or problems:
        status = "failed" if any(p["kind"] in HARD for p in problems) else "needs_work" if problems else \
            ("passed" if heard is not None else check["status"])
    out = {**check, "status": status, "problems": problems}
    if m:
        out["measured"] = {k: v for k, v in m.items() if k in ("speech_seconds", "expected_seconds", "sound_at_end",
                                                               "watched", "line_pitch", "ref_pitch")}
    if notes:
        out["accepted"] = notes
    return out


def needs_fix(check: Optional[dict]) -> bool:
    return bool(check) and any(p["kind"] in FIXABLE for p in check.get("problems") or [])


def rank(check: Optional[dict]) -> tuple:
    """Higher is better: passed > needs work > failed > not checked; then fewer serious, then fewer problems."""
    if not check:
        return (0, 0, 0)
    order = {"passed": 4, "needs_work": 3, "failed": 1, "unchecked": 0}
    ps = check.get("problems") or []
    return (order.get(check.get("status"), 0), -sum(p["kind"] in HARD for p in ps), -len(ps))


# ------------------------------------------------------------------ choosing the fix

def ladder(check: dict, provider: str, backups: list[str], lipsync_on: bool, has_frame: bool) -> list[dict]:
    """The fixes to try, cheapest that can work first. `backups`: other video models usable for this language."""
    kinds = {p["kind"] for p in check.get("problems") or [] if p["kind"] in FIXABLE}
    if not kinds:
        return []
    if kinds <= SOUND_ONLY:
        return [{"step": "clean", "provider": provider}]
    if kinds <= {"lip_sync", "lip_sync_some"} and lipsync_on:
        return [{"step": "resync", "provider": "sync"}]
    steps = [{"step": "retry", "provider": provider}]
    if has_frame:
        steps += [{"step": "switch", "provider": b} for b in backups if b != provider][:1]
    if lipsync_on:
        steps.append({"step": "saved", "provider": "tts"})
    return steps


def corrections(check: dict, lines: list[dict], voices: dict, language: str) -> str:
    """Exact instructions added to the next try. The dialogue itself is never changed."""
    out = []
    kinds = {p["kind"]: p for p in check.get("problems") or []}

    def who(i):
        d = lines[i]
        prof = voices.get(voice_mod.key(d.get("character") or ""), {})
        return f"{(d.get('character') or '').upper()} ({voice_mod.profile_text(prof, short=True) or 'their own voice'})"

    for k in ("missing_lines", "incomplete_lines", "cut_off"):
        if k in kinds:
            out.append("Every line is spoken in full, word for word, nothing skipped: " + "; ".join(
                f"line {i + 1} \"{lines[i].get('text')}\"" for i in kinds[k]["lines"] if i < len(lines)) + ".")
    for k in ("wrong_voice", "wrong_speaker", "identity", "identity_drift"):
        if k in kinds:
            out += [f"Line {i + 1} is spoken by {who(i)}, nobody else." for i in kinds[k]["lines"] if i < len(lines)]
    if "same_voice" in kinds:
        out.append("Each character has their own clearly different voice, as described.")
    if "extra_speech" in kinds:
        out.append("Nobody says anything that is not in the lines: no extra words, no background voices.")
    if "language" in kinds:
        out.append(f"Every word is spoken in {language}, keeping each character's accent.")
    if "pronunciation" in kinds or "clarity" in kinds:
        out.append("Speak clearly and pronounce every word properly.")
    if "unnatural" in kinds:
        out.append("Natural, conversational delivery with real breathing and pauses; not robotic.")
    if "overacted" in kinds:
        out.append("Understated and real: no more intense than each line's feeling.")
    if "rushed" in kinds or "cut_off" in kinds:
        out.append("An unhurried pace; the last line ends before the clip ends.")
    if "lip_sync" in kinds or "lip_sync_some" in kinds:
        out.append("The speaker's mouth movements match their words exactly, in time.")
    if {"clipping", "noise", "too_quiet"} & set(kinds):
        out.append("Clean, clear recording at a normal level: no distortion, hiss or background noise.")
    if {"no_audio", "silent"} & set(kinds):
        out.append("The characters must be heard speaking these lines out loud, with sound.")
    return ("VOICE CORRECTIONS (the dialogue above is unchanged; fix only this): " + " ".join(out)) if out else ""


def step_cost(step: dict, seconds: float, language: str, text: str):
    """What one fix could cost (provider cost, before the check)."""
    import studio_fal
    import studio_lipsync
    import studio_pricing as pricing
    s = step["step"]
    if s == "clean":
        return None
    if s == "resync":
        return studio_lipsync.estimate_lipsync(seconds)
    if s == "saved":
        return pricing.combine([pricing.estimate_voice(text), studio_lipsync.estimate_lipsync(seconds)])
    if step["provider"] in studio_fal.MODELS:
        return studio_fal.cost(step["provider"], seconds, True)
    return pricing.estimate_video(seconds)


def reserve(seconds: float, language: str, text: str, has_frame: bool, with_check: bool = True):
    """Room in the price for ONE automatic fix (the dearest likely one), so a fix that works can be charged
    without asking again. Only what is used is charged; nothing if the fix doesn't work (by default)."""
    import studio_fal
    import studio_lipsync
    import studio_pricing as pricing
    if not enabled():
        return None
    backups = [b for b in studio_fal.models() if studio_fal.usable(b, True, language)] if has_frame else []
    steps = ladder({"problems": [{"kind": "wrong_voice"}]}, "omni", backups, studio_lipsync.enabled(), has_frame)
    steps = steps[:max_attempts()]
    costs = [c for c in (step_cost(s, seconds, language, text) for s in steps) if c is not None]
    if not costs:
        return None
    dearest = max(costs, key=lambda c: c.total_usd)
    return pricing.combine([dearest, pricing.estimate_listen(seconds, video=True)]) if with_check else dearest


# ------------------------------------------------------------------ the creator's view

def outcome(first: dict, kept: dict, attempts: list[dict], stopped: str = "") -> dict:
    """What the creator sees: status, whether the automatic fix worked, whether their approval is needed, limits."""
    fs, ks = (first or {}).get("status"), (kept or {}).get("status")
    tried = [a for a in attempts if a.get("ran")]
    if fs in (None, "unchecked"):
        return {"status": "unchecked", "fixed": False, "needs_approval": False,
                "message": "The voices couldn't be checked automatically. Please listen to this clip."}
    if fs == "passed":
        return {"status": "passed", "fixed": False, "needs_approval": False, "message": "The voices passed the check."}
    if ks == "passed":
        return {"status": "fixed", "fixed": True, "needs_approval": False,
                "message": f"The voices had a problem ({_first_problem(first)}); fixed automatically "
                           f"({_how(tried[-1]) if tried else 'cleaned up'})."}
    left = _first_problem(kept)
    if not tried:
        why = stopped or "No automatic fix could be tried."
        return {"status": "needs_approval", "fixed": False, "needs_approval": True,
                "message": f"Voice problem: {left}. {why} Listen and keep it, or change the scene."}
    better = rank(kept) > rank(first)
    return {"status": "needs_approval", "fixed": False, "needs_approval": True,
            "message": (f"Voice problem: {left}. Viyo tried {len(tried)} automatic fix"
                        f"{'es' if len(tried) != 1 else ''}" + (", which improved it" if better else "") +
                        f". {stopped + ' ' if stopped else ''}Listen and keep it, or change the scene.")}


def _first_problem(check: Optional[dict]) -> str:
    ps = (check or {}).get("problems") or []
    serious = [p for p in ps if p["kind"] in HARD] or ps
    return serious[0]["problem"].rstrip(".") if serious else "unknown"


def _how(a: dict) -> str:
    return {"clean": "the sound was cleaned up", "resync": "the lip sync was redone",
            "retry": "made again with exact voice directions", "switch": f"made with another video model",
            "saved": "the saved character voices were used, lip synced"}.get(a.get("step"), a.get("step", ""))
