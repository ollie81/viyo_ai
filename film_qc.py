"""The final check of a rendered Studio film, before it is saved, charged or offered to publish.

A production only counts as made when the finished file is usable: it plays, has the right shape, is about as
long as planned, has sound where people speak, has no long stretch of black picture in the middle, and its sound
and picture stay together. When it fails, the render job fails (the creator's credits for it are returned) and
the message says what was wrong and which scene to fix. Uses the same measurements as the Ads Maker's check
(ad_qc: ffprobe, blackdetect, silencedetect); nothing here calls a paid service.

Kept deliberately to faults that make a film unusable, so a dark night scene or a quiet moment without
dialogue is never failed: black must be nearly total and last at least BLACK_FAIL_SECONDS (fades the film asked
for are allowed), and silence only fails in a scene where someone speaks.
"""
from __future__ import annotations

from typing import Optional

import ad_qc

BLACK_FAIL_SECONDS = 2.0
SYNC_TOLERANCE = 0.5          # seconds between the picture and the sound track
MIN_LENGTH_SHARE = 0.85       # of the planned length
VOICE_SILENCE_SHARE = 0.9     # a speaking scene that is this silent has lost its voices


def plan_from_inputs(inputs) -> list[dict]:
    """What the check needs from each rendered scene (studio_render.SceneInput)."""
    # A dissolve overlaps a scene with the one before it: the film is that much shorter than the scenes added up.
    return [{"duration": max(0.5, float(i.duration) - float(getattr(i, "dissolve_in", 0) or 0)),
             "speaks": bool(getattr(i, "lines", ()) and (i.audio or i.video)),
             "fade_in": float(getattr(i, "fade_in", 0) or 0), "fade_out": float(getattr(i, "fade_out", 0) or 0)}
            for i in inputs]


def _orientation_ok(aspect: str, w: int, h: int) -> bool:
    if aspect == "9:16":
        return h > w
    if aspect == "16:9":
        return w > h
    if aspect == "1:1":
        return w and abs(w - h) <= max(2, w * 0.02)
    return w > 0 and h > 0


def _spans(plan: list[dict], total: float) -> list[tuple[float, float]]:
    planned = sum(p["duration"] for p in plan) or 1.0
    scale = total / planned if total else 1.0
    out, t = [], 0.0
    for p in plan:
        d = p["duration"] * scale
        out.append((t, t + d))
        t += d
    return out


def _scene_at(spans, t: float) -> int:
    for i, (a, b) in enumerate(spans):
        if a <= t < b:
            return i + 1
    return len(spans)


def inspect_film(path: str, plan: list[dict], aspect_ratio: str,
                 black: Optional[list] = None, silent: Optional[list] = None) -> dict:
    """{"passed", "items": [{"key", "label", "status", "note"}]}. [black]/[silent] can be given (tests)."""
    items = []

    def add(key, label, ok, note=""):
        items.append({"key": key, "label": label, "status": "passed" if ok else "failed", "note": note})

    info = ad_qc.probe(path)
    total = info["duration"]
    playable = total > 0 and info["width"] > 0
    add("plays", "The film plays", playable, "" if playable else "The file has no picture.")
    if not playable:
        return {"passed": False, "items": items}
    add("shape", "Right shape", _orientation_ok(aspect_ratio, info["width"], info["height"]),
        f"{info['width']}x{info['height']} for {aspect_ratio}.")
    planned = sum(p["duration"] for p in plan)
    add("length", "Complete length", total >= planned * MIN_LENGTH_SHARE,
        f"{total:.1f}s of {planned:.1f}s planned.")
    speaks = any(p["speaks"] for p in plan)
    add("sound", "Has sound", info["has_audio"] or not speaks,
        "" if info["has_audio"] or not speaks else "Characters speak but the film has no sound.")
    if info["has_audio"]:
        drift = abs(info["video_seconds"] - info["audio_seconds"]) if info["video_seconds"] and info["audio_seconds"] else 0
        add("sync", "Sound and picture stay together", drift <= SYNC_TOLERANCE, f"{drift:.2f}s apart.")
    spans = _spans(plan, total)

    black = ad_qc.black_intervals(path, min_seconds=1.0) if black is None else black
    head = (plan[0]["fade_in"] if plan else 0) + 0.5
    tail = (plan[-1]["fade_out"] if plan else 0) + 0.5
    # Black that the film asked for (its opening fade-in, its closing fade-out) isn't a fault: a gap is black
    # that lasts BLACK_FAIL_SECONDS or more once those fades are taken off.
    long_black = []
    for a, b in black:
        a2, b2 = max(a, head), min(b, total - tail)
        if b2 - a2 >= BLACK_FAIL_SECONDS:
            long_black.append((a2, b2))
    if long_black:
        a, b = long_black[0]
        add("picture", "No black gaps", False, f"{b - a:.1f}s of black picture in scene {_scene_at(spans, a)}.")
    else:
        add("picture", "No black gaps", True)

    if info["has_audio"] and speaks:
        silent = ad_qc.silent_intervals(path, min_seconds=0.8) if silent is None else silent
        mute = []
        for n, ((a, b), p) in enumerate(zip(spans, plan), start=1):
            if not p["speaks"] or b - a < 1.0:
                continue
            quiet = sum(max(0.0, min(b, e) - max(a, s)) for s, e in silent)
            if quiet >= (b - a) * VOICE_SILENCE_SHARE:
                mute.append(n)
        add("voices", "Voices can be heard", not mute,
            f"Scene {mute[0]} has dialogue but no sound." if mute else "")
    return {"passed": all(i["status"] == "passed" for i in items), "items": items}


def explain(report: dict) -> str:
    bad = [i for i in report["items"] if i["status"] == "failed"]
    if not bad:
        return ""
    what = "; ".join(f"{i['label'].lower()}: {i['note']}" if i["note"] else i["label"].lower() for i in bad)
    return (f"The finished film didn't pass the final check ({what}). It wasn't saved and your credits for this "
            "render were returned. Remake the scene it names, then render again.")
