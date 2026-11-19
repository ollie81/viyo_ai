"""Quality control on the FINISHED ad file, after FFmpeg has produced it.

FFmpeg finishing is not success. inspect() looks at the rendered video itself:

  format       real width/height (and so aspect ratio) of the file, and that it has a sound track
  audio        no voiceover scene that came out silent
  black        no black or empty stretch (a blank phone screen filling the frame, a failed image)
  screens      no scene where most of the picture is one flat dark block (a black device screen)
  text edges   no caption touching the left/right edge of the frame (cut-off text), read from the frames
  layout       every caption inside its band, none overlapping the logo, asset or end card items
               (checked on the boxes the renderer actually drew)
  website      text read off the frames (OCR, Tesseract) shows no web address other than the advertiser's,
               and the end card shows the advertiser's real address
  end card     logo, brand name and call to action actually visible in the final seconds
  assets       every asset the plan relied on was found and readable

Each item is passed / failed / warning / unchecked (never "passed" when it couldn't be checked, e.g. no
OCR installed), and failed items name the scene so the pipeline can correct just that scene.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Optional

import numpy as np
from PIL import Image

import ad_brand
from ad_compose import Box, frame_layout


def probe(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                       capture_output=True, text=True, timeout=60)
    data = json.loads(r.stdout or "{}")
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    return {"width": int(v.get("width") or 0), "height": int(v.get("height") or 0),
            "duration": float((data.get("format") or {}).get("duration") or 0), "has_audio": a is not None,
            "video_seconds": float(v.get("duration") or 0), "audio_seconds": float((a or {}).get("duration") or 0)}


def _intervals(cmd: list[str], start_key: str, end_key: str) -> list[tuple[float, float]]:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out, start = [], None
    for line in r.stderr.splitlines():
        m = re.search(rf"{start_key}[:=]\s*([\d.]+)", line)
        if m:
            start = float(m.group(1))
        m2 = re.search(rf"{end_key}[:=]\s*([\d.]+)", line)
        if m2 and start is not None:
            out.append((start, float(m2.group(1))))
            start = None
    return out


def black_intervals(path: str, min_seconds: float = 0.5) -> list[tuple[float, float]]:
    return _intervals(["ffmpeg", "-v", "info", "-i", path, "-vf", f"blackdetect=d={min_seconds}:pix_th=0.10",
                       "-an", "-f", "null", "-"], "black_start", "black_end")


def silent_intervals(path: str, min_seconds: float = 0.8) -> list[tuple[float, float]]:
    return _intervals(["ffmpeg", "-v", "info", "-i", path, "-af", f"silencedetect=noise=-45dB:d={min_seconds}",
                       "-vn", "-f", "null", "-"], "silence_start", "silence_end")


def frame(path: str, t: float, out: str) -> Optional[str]:
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", path, "-frames:v", "1", out],
                       capture_output=True, timeout=60)
    return out if r.returncode == 0 and os.path.exists(out) else None


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr(image: str) -> Optional[str]:
    """Text read off a frame, or None when Tesseract isn't installed. Upscaled and read twice (as-is and
    inverted) so both light-on-dark and dark-on-light text are caught."""
    if not ocr_available():
        return None
    im = Image.open(image).convert("L")
    if im.width < 1400:
        im = im.resize((im.width * 2, im.height * 2))
    texts, failed = [], 0
    # As-is and inverted (light text on dark and dark on light), each read as scattered text (psm 11)
    # and as a block (psm 3): together they catch signs, captions and end-card text.
    for vi, variant in enumerate((im, Image.eval(im, lambda p: 255 - p))):
        tmp = f"{image}.ocr{vi}.png"
        variant.save(tmp)
        for psm in ("11", "3"):
            try:
                # One thread: Tesseract's own threading is much slower on a busy server, never faster.
                r = subprocess.run(["tesseract", tmp, "-", "--psm", psm], capture_output=True, text=True, timeout=60,
                                   env={**os.environ, "OMP_THREAD_LIMIT": "1"})
                texts.append(r.stdout)
            except (subprocess.TimeoutExpired, OSError):
                failed += 1
        os.remove(tmp)
    # A text reader that can't finish is "not checked", never a crashed ad.
    return None if failed == 4 else "\n".join(texts)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", (s or "").lower())


def edge_text(image: str, rows_box: Box, strip_frac: float = 0.012) -> bool:
    """True if caption-like text (bright white strokes with a dark outline) touches the left or right edge
    of the frame in the rows where captions were drawn: text cut off by the frame edge."""
    a = np.asarray(Image.open(image).convert("L"), dtype=np.int16)
    h, w = a.shape
    s = max(3, int(w * strip_frac))
    rows = slice(max(0, rows_box.y), min(h, rows_box.y2))
    for cols in (slice(0, s), slice(w - s, w)):
        region = a[rows, cols]
        white = region > 240
        dark = region < 40
        if white.sum() > 12 and dark.sum() > 12:
            # white next to dark in the same small area = an outlined glyph, not a bright sky or wall
            near = (np.roll(white, 2, axis=0) & dark) | (np.roll(white, 2, axis=1) & dark)
            if near.sum() > 6:
                return True
    return False


def flat_dark_block(image: str, min_frac: float = 0.18) -> bool:
    """True if a big part of the frame is one flat, near-black block (e.g. a phone whose screen is black).
    Looks for the largest connected dark, texture-free area on a downscaled copy."""
    im = Image.open(image).convert("L").resize((96, 96))
    a = np.asarray(im, dtype=np.int16)
    gy = np.abs(np.diff(a, axis=0, prepend=a[:1]))
    gx = np.abs(np.diff(a, axis=1, prepend=a[:, :1]))
    mask = (a < 28) & (gx < 4) & (gy < 4)
    if mask.mean() < min_frac:
        return False
    try:
        import cv2
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    except Exception:
        return False
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area / mask.size < min_frac:
            continue
        # A blank device screen sits inside the picture as a solid, roughly rectangular block. A dark sky,
        # room or background runs from one side of the frame to the other, and isn't a blank screen.
        spans_width = x <= 1 and x + bw >= 95
        solid = area / float(bw * bh) >= 0.8
        if solid and not spans_width:
            return True
    return False


def logo_visible(image: str, box: dict, logo: str, centered: bool = False, max_diff: float = 38.0) -> bool:
    """True if the advertiser's logo file is really visible at [box] in the frame: the logo is scaled the way
    the renderer placed it and its opaque pixels are compared with the frame (mean colour difference).
    Catches a missing, covered or wrong logo, not just an empty area."""
    try:
        frame_im = Image.open(image).convert("RGB")
        lg = Image.open(logo).convert("RGBA")
    except Exception:
        return False
    scale = min(box["w"] / lg.width, box["h"] / lg.height)
    lw, lh = max(1, int(lg.width * scale)), max(1, int(lg.height * scale))
    lg = lg.resize((lw, lh))
    x = box["x"] + ((box["w"] - lw) // 2 if centered else 0)
    y = box["y"] + ((box["h"] - lh) // 2 if centered else 0)
    crop = np.asarray(frame_im.crop((x, y, x + lw, y + lh)), dtype=np.float32)
    ref = np.asarray(lg, dtype=np.float32)
    mask = ref[..., 3] > 200
    if mask.sum() < 20 or crop.shape[:2] != mask.shape:
        return False
    return float(np.abs(crop[mask] - ref[..., :3][mask]).mean()) <= max_diff


def region_has_content(image: str, box: dict, min_std: float = 12.0) -> bool:
    """True if a region isn't flat (something is drawn there): used to confirm a logo/text is visible."""
    a = np.asarray(Image.open(image).convert("L"), dtype=np.float32)
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    crop = a[max(0, y):y + h, max(0, x):x + w]
    return crop.size > 0 and float(crop.std()) >= min_std


def inspect(path: str, plan: dict, facts: ad_brand.BrandFacts, w: int, h: int, work: str) -> dict:
    """[plan]: {"scenes": [{"index", "start", "end", "kind", "has_voice", "drawn": {...}}], "end_card":
    {"start", "end", "drawn": {...}, "has_logo"}, "asset_issues": [...]}. Returns
    {"passed", "items": [{"key", "label", "status", "note", "scene"}], "failing_scenes": {i: [keys]}}."""
    items, failing = [], {}

    def add(key, label, status, note="", scene=None):
        items.append({"key": key, "label": label, "status": status, "note": note, "scene": scene})
        if status == "failed" and scene is not None:
            failing.setdefault(scene, []).append(key)

    lay = frame_layout(w, h)
    info = probe(path)
    add("format", "Right size and shape", "passed" if (info["width"], info["height"]) == (w, h) else "failed",
        f"File is {info['width']}x{info['height']}, expected {w}x{h}.")
    add("audio_track", "Has sound", "passed" if info["has_audio"] else "failed",
        "" if info["has_audio"] else "The video has no audio track.")
    if info["has_audio"] and info["video_seconds"] and info["audio_seconds"]:
        gap = abs(info["video_seconds"] - info["audio_seconds"])
        add("av_sync", "Sound and picture stay together", "passed" if gap <= 0.35 else "failed",
            f"Sound and picture differ by {gap:.2f}s." if gap > 0.35 else "")

    scenes = plan.get("scenes", [])
    expected = plan.get("expected_seconds")
    if expected:
        drift = abs(info["duration"] - expected) / expected
        add("duration", "Length as planned", "passed" if drift <= 0.25 else "warning" if drift <= 0.6 else "failed",
            f"{info['duration']:.1f}s (planned {expected:.0f}s).")
    if info["duration"] < 2.0:
        add("duration_min", "Long enough to watch", "failed", f"Only {info['duration']:.1f}s long.")

    def scene_at(t: float) -> Optional[int]:
        for s in scenes:
            if s["start"] - 0.05 <= t < s["end"]:
                return s["index"]
        ec = plan.get("end_card") or {}
        return "end" if ec and ec["start"] - 0.05 <= t <= ec["end"] + 0.05 else None

    # Silent voiceover scenes.
    silences = silent_intervals(path) if info["has_audio"] else []
    for s in scenes:
        if not s.get("has_voice"):
            continue
        span = max(0.01, s["end"] - s["start"])
        covered = sum(max(0.0, min(b, s["end"]) - max(a, s["start"])) for a, b in silences)
        if covered / span > 0.9:
            add("voice", f"Voiceover in scene {s['index'] + 1}", "failed", "This scene should have a voiceover but is silent.",
                s["index"])
    if not any(i["key"] == "voice" for i in items):
        add("voice", "No unexpected silence", "passed")

    # Black stretches.
    blacks = black_intervals(path)
    for a, b in blacks:
        sc = scene_at((a + b) / 2)
        add("black", "No black or empty picture", "failed", f"Black picture from {a:.1f}s to {b:.1f}s.",
            sc if isinstance(sc, int) else None)
    if not blacks:
        add("black", "No black or empty picture", "passed")

    # Per-scene frame checks.
    texts_on_screen = []
    for s in scenes:
        mid = (s["start"] + s["end"]) / 2
        img = frame(path, mid, os.path.join(work, f"qc_scene_{s['index']}.png"))
        if not img:
            add("frame", f"Scene {s['index'] + 1} readable", "failed", "Couldn't read a frame from this scene.", s["index"])
            continue
        if s.get("kind") in ("generated", "video") and flat_dark_block(img):
            add("screen", f"Scene {s['index'] + 1}: no blank dark screen", "failed",
                "A large flat black area (like a blank phone screen) fills part of the picture.", s["index"])
        caps = s.get("drawn", {}).get("captions") or []
        if caps:
            y1, y2 = min(c["y"] for c in caps), max(c["y"] + c["h"] for c in caps)
        if caps and edge_text(img, Box(0, y1, w, y2 - y1)):
            add("clipping", f"Scene {s['index'] + 1}: captions inside the frame", "failed",
                "Caption text touches the edge of the frame.", s["index"])
        drawn = s.get("drawn", {})
        if s.get("product_photo") and drawn.get("product"):
            ok = logo_visible(img, drawn["product"], s["product_photo"], max_diff=42.0)
            add("product", f"Scene {s['index'] + 1}: the real product is shown", "passed" if ok else "failed",
                "" if ok else "The product photo isn't clearly visible in this scene.", None if ok else s["index"])
        if s.get("character_ref") is False:
            add("character", f"Scene {s['index'] + 1}: same character", "failed",
                "This scene shows the character but wasn't made from the character's reference portrait.", s["index"])
        drawn = s.get("drawn", {})
        if drawn.get("brand_mark") and plan.get("logo_path") and not logo_visible(img, drawn["brand_mark"], plan["logo_path"]):
            add("brand_mark", f"Scene {s['index'] + 1}: logo visible", "failed", "The logo in the corner isn't visible.",
                s["index"])
        for c in drawn.get("captions", []):
            cb = Box(**c)
            if not cb.inside(Box(0, 0, w, h)) or not cb.inside(lay["safe"]):
                add("safe", f"Scene {s['index'] + 1}: captions in the safe area", "failed",
                    "A caption sits outside the safe margins.", s["index"])
            for other in ("brand_mark",):
                if drawn.get(other) and cb.intersects(Box(**drawn[other])):
                    add("overlap", f"Scene {s['index'] + 1}: captions clear of the logo", "failed",
                        "A caption overlaps the logo.", s["index"])
            if s.get("kind") == "asset" and drawn.get("content") and cb.intersects(Box(**drawn["content"])):
                add("overlap", f"Scene {s['index'] + 1}: captions clear of the product", "failed",
                    "A caption covers the product image.", s["index"])
        text = ocr(img)
        if text is not None:
            texts_on_screen.append((s["index"], text))

    # Websites seen on screen (generated images can contain text despite instructions).
    if ocr_available():
        bad = [(i, d) for i, t in texts_on_screen for d in ad_brand.ocr_foreign_domains(t, facts)]
        for i, d in bad:
            add("website", f"Scene {i + 1}: only the advertiser's website", "failed",
                f'Shows "{d}", which is not the campaign\'s website.', i)
        if not bad:
            add("website", "Only the advertiser's website appears", "passed")
    else:
        add("website", "Only the advertiser's website appears", "unchecked", "Text reading (OCR) isn't installed here.")

    # Burned-in text: everything the renderer drew must be free of foreign addresses.
    burned = [t for s in scenes for t in s.get("drawn", {}).get("texts", [])] + \
        list((plan.get("end_card") or {}).get("drawn", {}).get("texts", []))
    stray = [d for t in burned for d in ad_brand.foreign_domains(t, facts)]
    add("burned_text", "Captions and end card use only campaign facts", "failed" if stray else "passed",
        f"Drawn text contains {', '.join(sorted(set(stray)))}." if stray else "")

    # End card.
    ec = plan.get("end_card")
    if not ec:
        add("end_card", "Ends on the brand", "failed", "There is no end card.", "end")
    else:
        img = frame(path, ec["start"] + (ec["end"] - ec["start"]) * 0.6, os.path.join(work, "qc_end.png"))
        drawn = ec.get("drawn", {})
        if not img:
            add("end_card", "Ends on the brand", "failed", "Couldn't read the end card.", "end")
        else:
            boxes = {it["kind"]: it["box"] for it in drawn.get("items", [])}
            if ec.get("has_logo"):
                ok = "logo" in boxes and (logo_visible(img, boxes["logo"], plan["logo_path"], centered=True)
                                          if plan.get("logo_path") else region_has_content(img, boxes["logo"]))
                add("logo", "Logo visible at the end", "passed" if ok else "failed",
                    "" if ok else "The logo isn't visible on the end card.", None if ok else "end")
            if "cta" in boxes:
                ok = region_has_content(img, boxes["cta"], 8)
                add("cta", "Call to action visible", "passed" if ok else "failed", "" if ok else "The call to action isn't visible.",
                    None if ok else "end")
            if not drawn.get("fits", True):
                add("end_layout", "End card fits the frame", "failed", "The end card items don't fit the safe area.", "end")
            # Final scene composition: every fact the advertiser gave is on it, each in its own area, long
            # enough to read.
            missing = [k for k, want in (("logo", ec.get("has_logo")), ("name", bool(facts.name)),
                                         ("url", bool(facts.url)), ("cta", bool(facts.cta))) if want and k not in boxes]
            seconds = ec["end"] - ec["start"]
            if missing or seconds < 2.5:
                add("final_scene", "Final scene shows the brand", "failed",
                    (f"Missing from the end card: {', '.join(missing)}. " if missing else "") +
                    (f"Shown for only {seconds:.1f}s." if seconds < 2.5 else ""), "end")
            else:
                add("final_scene", "Final scene shows the brand", "passed", f"{seconds:.1f}s with " +
                    ", ".join(k for k in ("logo", "name", "url", "cta") if k in boxes) + ".")
            text = ocr(img)
            if facts.url:
                if text is None:
                    add("url_visible", "Website shown at the end", "unchecked", "Text reading (OCR) isn't installed here.")
                else:
                    ok = _norm(facts.display_url).replace("www.", "") in _norm(text).replace("www.", "")
                    add("url_visible", "Website shown at the end", "passed" if ok else "failed",
                        f'Reads as "{facts.display_url}".' if ok else f'"{facts.display_url}" couldn\'t be read on the end card.',
                        None if ok else "end")
            if facts.name and text is not None:
                ok = _norm(facts.name) in _norm(text)
                add("name_visible", "Brand name shown at the end", "passed" if ok else "warning",
                    "" if ok else "The brand name couldn't be read clearly (it may be in the logo).")

    char_scenes = [s for s in scenes if s.get("character_ref") is not None]
    if char_scenes and not any(i["key"] == "character" and i["status"] == "failed" for i in items):
        add("character", "Same character throughout", "passed",
            f"{len(char_scenes)} scenes made from one reference portrait, one voice.")
    notes = plan.get("story_notes") or []
    add("continuity", "Complete story, in order", "warning" if notes else "passed", " ".join(notes))

    for issue in plan.get("asset_issues", []):
        add("assets", "Campaign assets", "failed", issue)
    if not plan.get("asset_issues"):
        add("assets", "Campaign assets", "passed")

    passed = not any(i["status"] == "failed" for i in items)
    return {"passed": passed, "items": items, "failing_scenes": {str(k): v for k, v in failing.items()},
            "ocr": ocr_available()}


def explain(report: dict) -> str:
    bad = [i for i in report.get("items", []) if i["status"] == "failed"]
    if not bad:
        return ""
    return "Needs correction: " + " ".join(f"{i['label']}: {i['note']}".strip() for i in bad)
