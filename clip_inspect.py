"""Final inspection of a RENDERED clip, after the technical quality gate
(clip_qa) has passed. It asks what an editor asks when watching the export:

  starts right    - does the clip begin at the start of a sentence?
  ends naturally  - does it end on a finished sentence?
  captions        - large enough, short lines, never more than two lines?
  subject visible - is the person or subject in frame, heads not cut off?
  render defects  - black bars, a squashed picture, garbled frames, captions off-screen or over a face?
  too dark        - is the picture so dark it's hard to watch?

Programmatic checks come first (words and their timing, the caption file, frame
brightness). Then, with the Decisions client, ONE small request per clip asks
the judgement questions about three frames grabbed from the finished video
(captions burned in) plus text evidence (the opening and closing words and what
was said just outside the clip). Decisions never "watches" the video: it sees
those frames and that text.

The verdict drives a bounded correction: "reframe" re-renders once with the
whole frame kept (when the crop lost the subject), "reject" drops the clip (a
spare takes its place), "flag" keeps it with a note, "ok" passes. Every item is
reported; anything not checked is "unchecked", never "passed".
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
from typing import Optional

import decisions as dc

TERMINAL = re.compile(r"[.?!…]['\")\]]*$")
SENTENCE_GAP = 0.6           # a pause this long counts as a sentence boundary even without punctuation
MIN_CAPTION_SIZE = 48        # ASS font size at 1080 px wide
MAX_LINE_CHARS = 28
DARK_LUMA = 35               # mean luma (0-255) below which a frame is too dark
FRAME_HEIGHT = 360

SUBJECT_LEVELS = [("Poor", "The main person or subject is missing or cut off (a head or face is cropped out)."),
                  ("Average", "Visible but awkwardly framed: off to one side, top of the head cut, or very small."),
                  ("Good", "The subject is clearly in frame."),
                  ("Excellent", "The subject is clearly and comfortably framed for a vertical screen.")]


def _txt(w: dict) -> str:
    return str(w.get("text") or w.get("word") or "")


def sentence_edges(words: list, blocks: list) -> dict:
    """Does the main clip start at a sentence start and end on a sentence end? (`words`: punctuated, source time.)"""
    if not words or not blocks:
        return {"start_complete": None, "end_complete": None}
    mid = lambda w: (float(w["start"]) + float(w["end"])) / 2           # noqa: E731
    kept = [k for k, w in enumerate(words) if any(a <= mid(w) < b for a, b in blocks)]
    if not kept:
        return {"start_complete": None, "end_complete": None}
    f, l = kept[0], kept[-1]
    prev = words[f - 1] if f > 0 else None
    start_ok = prev is None or bool(TERMINAL.search(_txt(prev).strip())) or \
        float(words[f]["start"]) - float(prev["end"]) >= SENTENCE_GAP
    nxt = words[l + 1] if l + 1 < len(words) else None
    end_ok = bool(TERMINAL.search(_txt(words[l]).strip())) or nxt is None or \
        float(nxt["start"]) - float(words[l]["end"]) >= SENTENCE_GAP
    before = " ".join(_txt(w) for w in words[max(0, f - 8):f])
    after = " ".join(_txt(w) for w in words[l + 1:l + 9])
    return {"start_complete": start_ok, "end_complete": end_ok,
            "opening": " ".join(_txt(words[k]) for k in kept[:14]), "closing": " ".join(_txt(words[k]) for k in kept[-14:]),
            "before": before, "after": after}


def caption_readability(ass_text: Optional[str]) -> dict:
    """Font size, line length and line count of the burned captions."""
    if not ass_text:
        return {"readable": None}
    size = None
    for ln in ass_text.splitlines():
        if ln.startswith("Style: Caption,"):
            try:
                size = int(ln.split(",")[2])
            except (IndexError, ValueError):
                pass
    longest, most_lines = 0, 0
    for ln in ass_text.splitlines():
        if ln.startswith("Dialogue:") and ",Caption," in ln:
            text = re.sub(r"\{[^}]*\}", "", ln.split(",", 9)[9])
            rows = text.split("\\N")
            most_lines = max(most_lines, len(rows))
            longest = max(longest, max(len(r.strip()) for r in rows))
    readable = (size or 0) >= MIN_CAPTION_SIZE and longest <= MAX_LINE_CHARS and most_lines <= 2
    return {"readable": readable, "font_size": size, "longest_line": longest, "max_lines": most_lines}


def grab_frames(path: str, duration: float, work_dir: str) -> list:
    """Three frames from the finished clip (captions included), as JPEG data URLs, plus their mean brightness."""
    out = []
    for k, t in enumerate((min(0.6, duration / 3), duration / 2, max(0.0, duration - 1.2))):
        f = os.path.join(work_dir, f"insp_{k}.jpg")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                            "-vf", f"scale=-2:{FRAME_HEIGHT}", "-q:v", "5", f], capture_output=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(f):
            continue
        luma = None
        try:
            import cv2
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            luma = float(img.mean()) if img is not None else None
        except Exception:
            pass
        with open(f, "rb") as fh:
            out.append({"url": "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode("ascii"),
                        "luma": luma, "at": round(t, 2)})
        os.remove(f)
    return out


def inspect(path: str, duration: float, words: list, blocks: list, ass_text: Optional[str], render_facts: dict,
            work_dir: str, client: Optional[dc.Client] = None, mode: str = "speech") -> dict:
    """The final inspection. Returns {"action": ok|flag|reframe|reject, "items": [...], "cost_usd", ...}."""
    edges = sentence_edges(words, blocks) if mode == "speech" else {"start_complete": None, "end_complete": None}
    caps = caption_readability(ass_text) if mode == "speech" else {"readable": None}
    frames = grab_frames(path, duration, work_dir)
    lumas = [f["luma"] for f in frames if f.get("luma") is not None]
    dark = (sum(lumas) / len(lumas) < DARK_LUMA) if lumas else None
    ai, cost, usage, ai_error = {}, 0.0, {}, None
    if client is not None and frames:
        evidence = {"opening_words": edges.get("opening"), "said_just_before (not in the clip)": edges.get("before"),
                    "closing_words": edges.get("closing"), "said_just_after (not in the clip)": edges.get("after"),
                    "crop": render_facts.get("crop_mode")}
        imgs = [dc.image(f["url"]) for f in frames]
        qs = [dc.score("subject", [dc.text("Frames from the start, middle and end of a finished vertical short. "
                                           "How well is the main person or subject framed?")] + imgs, SUBJECT_LEVELS),
              dc.predicate("defects", [dc.text("Do these frames show a rendering problem: black bars, a stretched or "
                                               "squashed picture, garbled or frozen image, captions cut off at the "
                                               "edge, or captions covering a face?")] + imgs)]
        if mode == "speech":
            qs.append(dc.predicate("captions", [dc.text("Would the captions in these frames be easy to read on a "
                                                        "phone (large, high contrast, short lines)?")] + imgs))
            qs.append(dc.predicate("starts", "Given the opening words and what was said just before, does the clip "
                                             "start at the beginning of a sentence or thought (not mid-sentence)?"))
            qs.append(dc.predicate("ends", "Given the closing words and what is said just after, does the clip end on a "
                                           "finished sentence or thought (not cut off mid-sentence)?"))
        try:
            res = client.decide([{"role": "user", "content": [dc.json_part(evidence)]}], qs)
            ai = {k: v for k, v in res.answers.items() if not v.refused}
            cost, usage = res.cost_usd, res.usage
        except dc.DecisionsUnavailable as e:
            ai_error = str(e)[:160]
    items = []

    def add(key, label, status, note=""):
        items.append({"key": key, "label": label, "status": status, "note": note})

    def pred(name):
        a = ai.get(name)
        return None if a is None else a.probability

    # Start and end: the words decide; the AI's view is recorded alongside.
    for key, label, prog, p in (("starts_right", "Starts at the start of a sentence", edges["start_complete"], pred("starts")),
                                ("ends_naturally", "Ends on a finished sentence", edges["end_complete"], pred("ends"))):
        if prog is None and p is None:
            add(key, label, "unchecked")
        elif prog is False and (p is None or p < 0.5):
            add(key, label, "failed", "The words show it starts or ends mid-sentence.")
        elif prog is False or (p is not None and p < 0.5):
            add(key, label, "warning", "The words and the AI's view disagree; worth a look.")
        else:
            add(key, label, "passed")
    pc = pred("captions")
    if caps["readable"] is None and pc is None:
        add("captions_readable", "Captions readable on a phone", "unchecked")
    elif caps["readable"] is False:
        add("captions_readable", "Captions readable on a phone", "warning",
            f"Font {caps.get('font_size')}, longest line {caps.get('longest_line')} characters, "
            f"up to {caps.get('max_lines')} lines.")
    else:
        add("captions_readable", "Captions readable on a phone", "passed" if (pc is None or pc >= 0.5) else "warning")
    subj = ai.get("subject")
    if subj is None:
        add("subject_visible", "Main subject visible", "unchecked")
    else:
        add("subject_visible", "Main subject visible",
            "passed" if subj.score >= 1.5 else "warning" if subj.score >= 1.0 else "failed",
            f"{subj.level or ''}".strip())
    pd = pred("defects")
    add("render_defects", "No rendering problems", "unchecked" if pd is None else
        "failed" if pd >= 0.7 else "warning" if pd >= 0.4 else "passed")
    add("lighting", "Bright enough to watch", "unchecked" if dark is None else "warning" if dark else "passed",
        "The picture is very dark." if dark else "")
    by = {i["key"]: i["status"] for i in items}
    if by["subject_visible"] == "failed" and render_facts.get("crop_mode") in ("track", "center", "stack"):
        action = "reframe"
    elif by["render_defects"] == "failed" or by["starts_right"] == "failed" or by["ends_naturally"] == "failed":
        action = "reject"
    elif "warning" in by.values() or "failed" in by.values():
        action = "flag"
    else:
        action = "ok"
    return {"action": action, "items": items, "cost_usd": cost, "usage": usage, "ai_error": ai_error,
            "evidence": {k: edges.get(k) for k in ("start_complete", "end_complete")} | {"captions": caps,
                                                                                          "mean_luma": lumas}}
