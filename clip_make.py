"""One clip, from its edit decisions to the finished files.

  decide()  - every editorial decision for a clip, as SOURCE timestamps:
              the parts it is made of (and the parts without a join), the
              sentences the retention editor cuts, the cold-open line, the
              hook, plus the words and segments of that stretch and a slice
              of the visual analysis. This is the clip's edit decision list
              (EDL); it is saved next to the clip, so the creator can switch
              any edit off (or back on) and the clip is made again exactly the
              same way, without paying for any analysis twice.
  make()    - renders an EDL with the creator's choices (`toggles`): blocks,
              captions, crop and zoom, the caption-free master and the
              captioned clip, then quality control.

Used by the worker (repurpose_pipeline) and by the "apply my edit choices"
endpoint (repurpose.clip_edits). No AI calls here.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np

import captions
import clip_edit
import clip_qa
import clip_render
import clip_retention
import video_analysis as va

EDIT_KEYS = ("join", "cold_open", "retention_cuts", "tighten", "zoom")
DEFAULT_TOGGLES = {k: True for k in EDIT_KEYS}


class Rejected(Exception):
    """The rendered clip failed quality control."""

    def __init__(self, failures: list, report: dict):
        super().__init__(", ".join(failures))
        self.failures, self.report = failures, report


# ------------------------------------------------------------------ the visual analysis, sliced and saved

def timeline_to_json(tl: Optional[va.VisualTimeline], spans: list, pad: float = 2.0) -> Optional[dict]:
    """The samples of the visual analysis that fall inside the clip's source spans (for re-rendering it)."""
    if tl is None or not tl.usable:
        return None
    keep = np.zeros(len(tl.times), dtype=bool)
    for a, b in spans:
        keep |= (tl.times >= a - pad) & (tl.times <= b + pad)
    idx = np.nonzero(keep)[0]
    if len(idx) < 3:
        return None
    remap = {int(k): n for n, k in enumerate(idx)}
    shots = []
    for sh in tl.shots:
        inside = [remap[k] for k in range(sh.first, sh.last + 1) if k in remap]
        if inside:
            shots.append({"start": sh.start, "end": sh.end, "first": inside[0], "last": inside[-1], "cluster": sh.cluster})
    return {"times": tl.times[idx].round(3).tolist(), "sharpness": tl.sharpness[idx].round(2).tolist(),
            "brightness": tl.brightness[idx].round(2).tolist(),
            "faces": [[asdict(f) for f in tl.faces[k]] for k in idx],
            "cut_before": [bool(x) for x in tl.cut_before[idx]], "shots": shots, "frame_w": tl.frame_w,
            "frame_h": tl.frame_h, "bars": tl.bars, "detector": tl.detector}


def timeline_from_json(d: Optional[dict]) -> Optional[va.VisualTimeline]:
    if not d:
        return None
    return va.VisualTimeline(
        times=np.array(d["times"], dtype=float), sharpness=np.array(d["sharpness"], dtype=float),
        brightness=np.array(d["brightness"], dtype=float),
        faces=[[va.Face(**f) for f in row] for row in d["faces"]], hist=np.zeros((len(d["times"]), 48)),
        cut_before=np.array(d["cut_before"], dtype=bool), shots=[va.Shot(**s) for s in d["shots"]],
        frame_w=d["frame_w"], frame_h=d["frame_h"], bars=d.get("bars") or {}, detector=d.get("detector", "none"))


# ------------------------------------------------------------------ decisions

def _span_ids(sentences: list, a: float, b: float) -> list:
    return [s.id for s in sentences if a - 0.3 <= s.start and s.end <= b + 0.3]


def decide(clip: dict, punct: list, raw_words: list, segments: list, sentences: list, *, mode: str = "speech",
           retention: bool = True, cold_open: bool = True, timeline: Optional[va.VisualTimeline] = None) -> dict:
    """The clip's edit decision list (see the module notes)."""
    st = clip.get("structure") or {}
    parts = [list(p) for p in (clip.get("parts") or [])] or [[clip["start"], clip["end"]]]
    unjoined = (st.get("unjoined") or {}).get("parts") if st.get("joined") else None
    lo, hi = min(p[0] for p in parts) - 1.0, max(p[1] for p in parts) + 1.0
    in_span = lambda w: any(a - 1.0 <= float(w["start"]) <= b + 1.0 for a, b in parts)    # noqa: E731
    edl = {
        "version": 1, "mode": mode, "parts": parts, "parts_unjoined": unjoined,
        "joined_at": st.get("joined_at"),
        "join": {k: st.get(k) for k in ("relation", "why", "link")} if st.get("joined") else None,
        "overlay": clip.get("overlay_hook") or "",
        "words": [{"text": w.get("text") or w.get("word"), "start": float(w["start"]), "end": float(w["end"])}
                  for w in punct if in_span(w)],
        "raw_words": [{"word": w.get("word") or w.get("text"), "start": float(w["start"]), "end": float(w["end"])}
                      for w in raw_words if in_span(w)],
        "segments": [{"start": float(s["start"]), "end": float(s["end"])} for s in segments
                     if s.get("end", 0) > lo and s.get("start", 0) < hi],
        "retention": None, "retention_cuts": [], "teaser": None,
        "timeline": timeline_to_json(timeline, parts),
    }
    if mode != "speech" or not sentences:
        return edl
    # Retention analysis over the clip's sentences, part by part.
    ranges = []
    for a, b in parts:
        ids = _span_ids(sentences, a, b)
        if ids:
            ranges.append((ids[0], ids[-1]))
    dropped = set(st.get("dropped") or [])
    if ranges:
        report = clip_retention.analyze(ranges, sentences, punct, parts[0][0])
        report["sentences"] = [r for r in report["sentences"] if r["sentence"] not in dropped]
        edl["retention"] = {k: v for k, v in report.items() if k != "cuts"}
        if retention:
            edl["retention_cuts"] = [[round(sentences[i].start - 0.02, 3),
                                      round((sentences[i + 1].start - 0.02) if i + 1 < len(sentences)
                                            else sentences[i].end + 0.1, 3), sentences[i].text]
                                     for i in report["cuts"] if i not in dropped]
    # Cold open: from the main moment only (never from a joined part).
    if cold_open:
        main = st.get("main") or [clip.get("start_sentence"), clip.get("end_sentence")]
        if main[0] is not None:
            joined = st.get("joined") or [-1, -1]
            ids = [i for i in range(int(main[0]), int(main[1]) + 1) if i not in dropped
                   and not (joined[0] <= i <= joined[1])]
            if ids:
                edl["teaser"] = clip_edit.choose_teaser(sentences, ids, punct, parts[0][0])
    return edl


# ------------------------------------------------------------------ making

def blocks_for(edl: dict, toggles: dict, speaker_at=None, timeline=None, source: Optional[str] = None) -> tuple:
    """The kept source intervals for the EDL and the creator's choices. Returns (blocks, facts)."""
    import repurpose as r
    parts = edl["parts"] if (toggles.get("join", True) or not edl.get("parts_unjoined")) else edl["parts_unjoined"]
    blocks, dead_air = [], 0.0
    for a, b in parts:
        pb = r._find_speech_blocks(edl["segments"], a, b)
        dead_air += r._dead_air_removed_seconds(pb)
        blocks += pb
    before = sum(e - s for s, e in blocks)
    blocks, filler_count = r._remove_filler_words(blocks, edl["raw_words"])
    facts = {"dead_air_removed_seconds": round(dead_air, 2),
             "filler_words_removed_seconds": round(before - sum(e - s for s, e in blocks), 2),
             "filler_words_removed_count": filler_count, "parts": parts,
             "joined": parts is edl["parts"] and bool(edl.get("joined_at"))}
    cuts = []
    if toggles.get("retention_cuts", True):
        for a, b, text in edl.get("retention_cuts") or []:
            if any(p0 <= a < p1 for p0, p1 in parts):
                blocks = clip_edit._cut(blocks, a, b)
                cuts.append(text)
    facts["retention_cuts"] = cuts
    tight = {"pauses_tightened": 0, "pause_seconds_removed": 0.0, "repeats_removed": []}
    if toggles.get("tighten", True) and edl["mode"] == "speech":
        blocks, tight = clip_edit.tighten(blocks, edl["words"], speaker_at)
    facts.update(tight)
    blocks, moved = clip_render.snap_to_cuts(blocks, timeline, edl["words"], source)
    facts["edges_moved_to_camera_cuts"] = moved
    return blocks, facts


def make(source: str, edl: dict, toggles: dict, work_dir: str, tag: str, *, dims: dict, fps: Optional[float],
         has_audio: bool, caption_style: str, timeline: Optional[va.VisualTimeline] = None, speaker_at=None,
         qa: bool = True, zoom: bool = True, transcript: str = "") -> dict:
    """Renders the EDL: returns {"out", "master", "captions_for", "render", "captions", "qa", "edits", "planned"}.
    Raises Rejected when quality control fails after its bounded corrections."""
    toggles = {**DEFAULT_TOGGLES, **(toggles or {})}
    blocks, bfacts = blocks_for(edl, toggles, speaker_at, timeline, source)
    joined_at = edl.get("joined_at") if bfacts["joined"] else None
    markers = [(captions.build_remap(blocks)(float(joined_at)), "LATER IN THE VIDEO")] if joined_at else []
    teaser = edl.get("teaser") if toggles.get("cold_open") else None
    overlay = edl.get("overlay") or ""
    words = edl["words"]
    planned_main = sum(e - s for s, e in blocks)
    out = os.path.join(work_dir, f"{tag}.mp4")
    master = os.path.join(work_dir, f"{tag}_master.mp4")
    plan = clip_render.plan_crop(timeline, blocks, dims["width"], dims["height"], source)
    remap = captions.build_remap(blocks)
    sentence_starts = tuple(remap(float(w["start"])) for k, w in enumerate(words)
                            if k == 0 or str(words[k - 1]["text"]).rstrip().endswith((".", "?", "!")))
    state = {"teaser_dur": 0.0, "zoom": {}}

    def render_master() -> dict:
        windows, zf = clip_render.zoom_windows(blocks, timeline, plan, (float(joined_at),) if joined_at else (),
                                               sentence_starts) if (zoom and toggles.get("zoom")) else ([], {})
        state["zoom"] = zf
        main = os.path.join(work_dir, f"{tag}_main.mp4") if teaser else master
        facts = clip_render.render(source, main, blocks, plan, None, fps, has_audio=has_audio, zoom=windows,
                                   crf=clip_render.MASTER_CRF, preset=clip_render.MASTER_PRESET)
        if teaser:
            tb = [(teaser["start"], teaser["end"])]
            tplan = clip_render.plan_crop(timeline, tb, dims["width"], dims["height"], source)
            if plan.mode == "fit":
                tplan = clip_render.CropPlan(mode="fit", src_w=dims["width"], src_h=dims["height"], bars=tplan.bars,
                                             crop_w=tplan.crop_w, crop_h=tplan.crop_h,
                                             pieces=[(0.0, 0.0, tb[0][1] - tb[0][0], 0.0)], facts=tplan.facts)
            tpath = os.path.join(work_dir, f"{tag}_teaser.mp4")
            clip_render.render(source, tpath, tb, tplan, None, fps, has_audio=has_audio,
                               crf=clip_render.MASTER_CRF, preset=clip_render.MASTER_PRESET)
            state["teaser_dur"] = _duration(tpath) or (tb[0][1] - tb[0][0])
            clip_render.join_clips(tpath, main, master)
            for f in (tpath, main):
                os.remove(f)
        return facts

    def captions_for(layout: Optional[dict], shift: float = 0.0) -> tuple:
        main_ass, facts = captions.build_ass(words, blocks, hook="" if teaser else overlay, style=caption_style,
                                             speaker_at=speaker_at, shift=shift, layout=layout, markers=markers)
        if teaser:
            t_ass, _f = captions.build_ass(words, [(teaser["start"], teaser["end"])], hook=overlay,
                                           style=caption_style, speaker_at=speaker_at, shift=shift, layout=layout)
            return captions.join_ass(t_ass, main_ass, state["teaser_dur"]), facts
        return main_ass, facts

    def write(text: Optional[str], name: str) -> Optional[str]:
        if not text:
            return None
        path = os.path.join(work_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    qa_dict: dict = {}
    render_facts, shift, attempts = None, 0.0, 0
    while True:
        attempts += 1
        if render_facts is None:
            render_facts = render_master()
        ass_text, cap_facts = captions_for(None, shift)
        if shift:
            cap_facts["timing_shift_s"] = shift
        clip_render.burn_captions(master, out, write(ass_text, f"{tag}.ass"))
        planned = planned_main + state["teaser_dur"]
        if not qa:
            break
        c_words = ([w for w in words if teaser["start"] <= (w["start"] + w["end"]) / 2 < teaser["end"]] if teaser else []) + \
            [w for w in words if any(s <= (w["start"] + w["end"]) / 2 < e for s, e in blocks)]
        report = clip_qa.run_qa(out, planned, render_facts, ass_text, overlay or None, transcript,
                                [str(w["text"]) for w in c_words], work_dir, mode=edl["mode"], has_audio=has_audio)
        qa_dict = report.to_dict()
        if report.passed:
            break
        cs_fail = [f for f in report.failures if f.name == "caption_sync"]
        if (attempts == 1 and ass_text and len(report.failures) == 1 and cs_fail
                and 0.12 < abs(cs_fail[0].detail.get("offset_s", 0)) <= 0.7):
            shift = float(cs_fail[0].detail["offset_s"])      # re-burn the captions only; the master is reused
            continue
        if attempts <= 2 and report.failures and all(f.name == "framing" for f in report.failures):
            plan = clip_render.CropPlan(mode="fit", src_w=dims["width"], src_h=dims["height"], bars=plan.bars,
                                        crop_w=plan.crop_w, crop_h=plan.crop_h,
                                        pieces=[(0.0, 0.0, planned_main, 0.0)], facts={**plan.facts, "retry": "fit_after_framing"})
            render_facts = None
            continue
        for f in (out, master):
            if os.path.exists(f):
                os.remove(f)
        raise Rejected([f.name for f in report.failures], qa_dict)

    teaser_info = {"text": teaser["text"], "source_time": teaser["start"], "seconds": round(state["teaser_dur"], 2)} \
        if teaser else None
    edits = {
        "parts": [[round(a, 2), round(b, 2)] for a, b in bfacts["parts"]],
        "joined": bfacts["joined"], "joined_at": joined_at, "join": edl.get("join") if bfacts["joined"] else None,
        "can_unjoin": bool(edl.get("parts_unjoined")),
        "cold_open": teaser_info, "can_cold_open": bool(edl.get("teaser")),
        "retention_cuts": bfacts["retention_cuts"],
        "pauses_tightened": bfacts["pauses_tightened"], "pause_seconds_removed": bfacts["pause_seconds_removed"],
        "repeats_removed": bfacts["repeats_removed"],
        "zoom_cuts": (state["zoom"] or {}).get("jump_cuts", 0),
        "pacing_reframes": (state["zoom"] or {}).get("pacing_reframes", 0),
        "edges_moved_to_camera_cuts": bfacts["edges_moved_to_camera_cuts"],
        "toggles": toggles,
    }
    return {"out": out, "master": master, "captions_for": captions_for, "write": write, "render": render_facts,
            "captions": cap_facts, "qa": qa_dict, "edits": edits, "planned": planned,
            "trim": {k: bfacts[k] for k in ("dead_air_removed_seconds", "filler_words_removed_seconds",
                                            "filler_words_removed_count")}}


def _duration(path: str) -> Optional[float]:
    import subprocess
    try:
        return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     path], capture_output=True, text=True, timeout=30).stdout.strip())
    except (ValueError, Exception):
        return None


def save_json(storage, bucket: str, path: str, data: dict, work_dir: str) -> None:
    local = os.path.join(work_dir, os.path.basename(path))
    with open(local, "w", encoding="utf-8") as f:
        json.dump(data, f)
    storage.upload(bucket, path, local, "application/json")
