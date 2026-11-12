"""Phase 2 of the AI Shorts editor: story threads across the whole video,
retention analysis with bounded cuts, platform profiles (hook timing and
placement, export settings), visual decisions from the footage (cuts aligned
to camera changes, reframes on long static shots, the B-roll the video really
has), and the creator's control over each edit (an edit decision list, and
re-making the clip with edits switched off).

Scripted transcriber and LLM, local storage, real FFmpeg and Postgres. These
prove the decisions, the timestamps and the files. They do NOT prove that a
real model finds good threads, or that viewers watch longer: that needs real
videos.
"""
import json
import os
import subprocess

import numpy as np
import pytest

import captions
import clip_make
import clip_render
import clip_retention
import clip_selection as cs
import clip_story
import platforms as pf
import video_analysis as va
import video_map
from test_repurpose_worker import (balance, env, job_row, local_path, probe, source_video, submit)  # noqa: F401
from test_shorts_editor import LONG, sentences_from
from conftest import db  # noqa: F401


# ------------------------------------------------------------------ threads in the whole-video map

def test_the_video_map_keeps_only_valid_threads():
    data = {"summary": "s", "format": "talk", "topics": [], "avoid": [], "threads": [
        {"relation": "setup_payoff", "parts": [{"start_sentence": 0, "end_sentence": 2},
                                               {"start_sentence": 18, "end_sentence": 19}], "why": "the rent story ends"},
        {"relation": "setup_payoff", "parts": [{"start_sentence": 18, "end_sentence": 19},
                                               {"start_sentence": 0, "end_sentence": 2}]},          # out of order
        {"relation": "gossip", "parts": [{"start_sentence": 0, "end_sentence": 1},
                                         {"start_sentence": 5, "end_sentence": 6}]},               # unknown relation
        {"relation": "callback", "parts": [{"start_sentence": 0, "end_sentence": 15},
                                           {"start_sentence": 18, "end_sentence": 19}]},           # not a short moment
        {"relation": "callback", "parts": [{"start_sentence": 0, "end_sentence": 1},
                                           {"start_sentence": 99, "end_sentence": 100}]}]}         # not in the video
    vm = video_map._validate(data, 21, None)
    assert vm["threads"] == [{"relation": "setup_payoff", "parts": [[0, 2], [18, 19]], "why": "the rent story ends"}]


# ------------------------------------------------------------------ linking

def _cand(sents, words, s0=0, s1=5, **raw):
    base = {"start_sentence": s0, "end_sentence": s1, "category": "story", "scores": {k: 8 for k in cs.RUBRIC_KEYS},
            "title": "The call that paid my rent", "reason": "x", "overlay_hook": "", "caption": "c",
            "hashtags": ["rent"], "topic": "rent"}
    c, why = cs.evaluate_candidate({**base, **raw}, sents, words, 400.0)
    assert why is None, why
    return c


def _map(*threads, avoid=()):
    return {"summary": "s", "topics": [], "avoid": list(avoid),
            "threads": [{"relation": r, "parts": [list(a), list(b)], "why": "w"} for r, a, b in threads]}


def test_a_payoff_from_much_later_in_the_video_is_joined_with_exact_source_times():
    words, sents = sentences_from(LONG)
    c = _cand(sents, words)
    before = (c.parts, c.transcript)
    facts = clip_story.link([c], sents, words, _map(("setup_payoff", (0, 3), (18, 19))), 400.0)
    assert facts["linked"] == 1
    assert c.structure["joined"] == [18, 19] and c.structure["needs_confirmation"]
    # Timestamps: every part starts and ends on its real words.
    assert c.parts[0][0] == pytest.approx(max(0.0, sents[0].start - 0.05), abs=0.01)
    assert c.parts[-1][0] == pytest.approx(sents[18].start - 0.05, abs=0.01)
    assert c.parts[-1][1] == pytest.approx(sents[19].end + cs.END_TAIL_SECONDS, abs=0.01)
    assert c.structure["joined_at"] == c.parts[-1][0]
    assert c.transcript.endswith("which changed my whole career.") and "hired me full time" in c.transcript
    # The review can't confirm it (no review ran): the join is undone, exactly.
    clip_story.settle([c])
    assert (c.parts, c.transcript) == before and c.structure.get("join_undone")


def test_a_setup_from_earlier_comes_first_and_the_viewer_is_told_where_it_jumps():
    words, sents = sentences_from(LONG)
    c = _cand(sents, words, 18, 19)
    clip_story.link([c], sents, words, _map(("setup_payoff", (3, 5), (18, 19))), 400.0)
    assert c.structure["ranges"] == [[3, 5], [18, 20]] and c.start_sentence == 3, "main moment met the 12 s minimum"
    assert c.structure["joined_at"] == c.parts[1][0], "the marker sits where the video jumps forward"
    c.review = {"join_faithful": True}
    assert clip_story.settle([c]) == {"confirmed": 1, "undone": 0}
    assert "needs_confirmation" not in c.structure


@pytest.mark.parametrize("thread, other, avoid, reason", [
    (("question_answer", (0, 3), (19, 20)), None, (), "second_part_leans_back"),       # "And the rate stayed..."
    (("setup_payoff", (0, 3), (18, 19)), (17, 19), (), "partner_in_another_clip"),
    (("setup_payoff", (0, 3), (18, 19)), None, ({"start_sentence": 18, "end_sentence": 20, "kind": "outro"},),
     "partner_is_boilerplate"),
    (("setup_payoff", (0, 3), (6, 7)), None, (), "too_close"),
])
def test_unsafe_or_misleading_combinations_are_refused(thread, other, avoid, reason):
    words, sents = sentences_from(LONG)
    c = _cand(sents, words)
    pool = [c] + ([_cand(sents, words, *other)] if other else [])
    facts = clip_story.link(pool, sents, words, _map(thread, avoid=avoid), 400.0)
    assert facts["linked"] == 0 and set(facts["refused"]) == {reason}
    assert len(c.parts) == 1


def test_a_join_the_review_finds_misleading_is_undone_and_the_review_sees_both_parts():
    import clip_review
    words, sents = sentences_from(LONG)
    c = _cand(sents, words)
    clip_story.link([c], sents, words, _map(("claim_example", (0, 3), (18, 19))), 400.0)
    block = clip_review._clip_block(c, 0, sents, None)
    assert '[ON SCREEN: "LATER IN THE VIDEO"] Here is how the rent story ended' in block
    assert "linked as claim example" in block
    c.review = {"join_faithful": False}
    assert clip_story.settle([c]) == {"confirmed": 0, "undone": 1} and len(c.parts) == 1


# ------------------------------------------------------------------ retention

RETAIN = [
    "Most people price their work by the hour and that is the mistake.",          # 0
    "You should price the result your client gets, not your time.",               # 1
    "In other words, price the result your client gets instead of your time.",    # 2 restates 1
    "My first client paid 4000 dollars for a website that took a week.",          # 3
    "Price the result your client gets and not your time.",                       # 4 repeats 1, but it's the last line
]


def test_a_restated_point_is_cut_but_a_new_fact_or_the_last_line_never_is():
    words, sents = sentences_from(RETAIN)
    rep = clip_retention.analyze([(0, 4)], sents, words, 0.0)
    assert rep["cuts"] == [2], "the restatement goes; the 4000-dollar fact and the closing line stay"
    flags = {r["sentence"]: r["flags"] for r in rep["sentences"]}
    assert "restates" in flags[2] and "repeats" in flags[4]
    assert any(s["kind"] == "repeats" for s in rep["suggestions"]), "the kept repeat is explained, not hidden"
    blocks = clip_retention.cut_blocks([(0.0, sents[4].end)], sents, rep["cuts"])
    assert blocks == [(0.0, pytest.approx(sents[2].start - 0.02)), (pytest.approx(sents[3].start - 0.02), sents[4].end)]


def test_slow_stretches_and_warm_up_openings_are_suggestions_not_cuts():
    slow_words, slow = sentences_from(["So, okay, today I want to talk about money.", "Prices.",
                                       "Here is the thing about money and time.", "That is the end."],
                                      start=1.5, wps=1.2)
    rep = clip_retention.analyze([(0, 3)], slow, slow_words, 0.0)
    assert rep["warm_up_opening"] and rep["cuts"] == [] and rep["risk"] > 40
    assert {s["kind"] for s in rep["suggestions"]} >= {"opening", "slow"}


# ------------------------------------------------------------------ platforms

def test_each_platform_has_its_own_hook_timing_placement_and_export_settings(tmp_path):
    words, _ = sentences_from(["Nobody tells you the truth about rent."])
    a = {n: captions.build_ass(words, [(0.0, 3.0)], hook="The truth about rent", layout=pf.layout(n))[0]
         for n in pf.PLATFORMS}
    hook = lambda t: next(ln for ln in t.splitlines() if ",Hook,," in ln)            # noqa: E731
    style = lambda t: next(ln for ln in t.splitlines() if ln.startswith("Style: Hook"))  # noqa: E731
    assert "0:00:02.20" in hook(a["tiktok"]) and "0:00:02.60" in hook(a["shorts"]) and "0:00:02.40" in hook(a["reels"])
    assert style(a["reels"]).endswith(",270,1") and style(a["tiktok"]).endswith(",250,1")
    master, out = str(tmp_path / "m.mp4"), str(tmp_path / "o.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=d=2:sample_rate=44100", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", master], check=True)
    clip_render.burn_captions(master, out, None, encode=pf.PLATFORMS["shorts"]["export"])
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                      "stream=codec_type,sample_rate,avg_frame_rate", "-of", "json", out],
                                     capture_output=True, text=True).stdout)["streams"]
    assert next(s for s in info if s["codec_type"] == "audio")["sample_rate"] == "48000"
    assert next(s for s in info if s["codec_type"] == "video")["avg_frame_rate"] == "30/1"


# ------------------------------------------------------------------ visual decisions

class _TL:
    usable = True

    def __init__(self, cuts=()):
        self._cuts = list(cuts)

    def cuts_between(self, a, b):
        return [c for c in self._cuts if a < c <= b]


def test_a_long_static_shot_gets_a_reframe_at_a_sentence_and_a_camera_cut_resets_it():
    plan = clip_render.CropPlan(mode="track", facts={"face_ratio": 0.9})
    blocks = [(0.0, 20.0)]
    w, f = clip_render.zoom_windows(blocks, _TL(), plan, sentence_starts=(3.0, 8.0, 12.0, 16.5))
    assert w == [(8.0, 16.5)] and f["pacing_reframes"] == 2, "in after 7 s of one framing, out 8.5 s later"
    w, f = clip_render.zoom_windows(blocks, _TL([9.0]), plan, sentence_starts=(3.0, 8.0, 12.0, 16.5))
    assert w == [(16.5, 20.0)] and f["pacing_reframes"] == 1, \
        "no reframe right before a camera cut (the cut changes the picture anyway); 7 s after it instead"
    assert clip_render.zoom_windows(blocks, _TL(), clip_render.CropPlan(mode="fit", facts={"face_ratio": 0.9}),
                                    sentence_starts=(8.0,))[0] == []


def test_cut_points_move_onto_a_nearby_camera_cut_only_when_no_word_is_lost():
    words = [{"text": "hi", "start": 10.4, "end": 10.7}, {"text": "bye", "start": 19.0, "end": 19.4}]
    out, moved = clip_render.snap_to_cuts([(10.0, 20.0)], _TL([10.25, 19.8]), words)
    assert out == [(10.25, 19.8)] and moved == 2, "no flash of the other shot at either end"
    words.append({"text": "uh", "start": 10.05, "end": 10.15})
    out, moved = clip_render.snap_to_cuts([(10.0, 20.0)], _TL([10.25]), words)
    assert out == [(10.0, 20.0)] and moved == 0, "a spoken word is never cut to tidy a frame"


def test_only_b_roll_the_video_really_contains_is_reported():
    times = np.arange(0, 20, 0.5)
    faces = [[va.Face(0.5, 0.4, 0.2, 0.3, 0.9)] if t < 10 else [] for t in times]
    tl = va.VisualTimeline(times=times, sharpness=np.full(len(times), 50.0), brightness=np.full(len(times), 0.5),
                           faces=faces, shots=[va.Shot(0, 10, 0, 19), va.Shot(10, 20, 20, 39)])
    assert clip_render.cutaways(tl) == [(10.0, 20.0)]
    assert clip_render.cutaways(None) == []


def test_the_saved_visual_analysis_rebuilds_the_same_framing():
    times = np.arange(0, 30, 0.5)
    faces = [[va.Face(0.3 + 0.01 * (k % 5), 0.4, 0.15, 0.25, 0.9)] for k in range(len(times))]
    cut = np.zeros(len(times), dtype=bool)
    cut[30] = True
    tl = va.VisualTimeline(times=times, sharpness=np.full(len(times), 40.0), brightness=np.full(len(times), 0.5),
                           faces=faces, cut_before=cut, shots=[va.Shot(0, 15, 0, 29), va.Shot(15, 30, 30, 59)],
                           frame_w=1920, frame_h=1080)
    back = clip_make.timeline_from_json(json.loads(json.dumps(clip_make.timeline_to_json(tl, [[5.0, 25.0]]))))
    blocks = [(5.0, 12.0), (12.5, 25.0)]
    a = clip_render.plan_crop(tl, blocks, 1920, 1080, refine=False)
    b = clip_render.plan_crop(back, blocks, 1920, 1080, refine=False)
    assert (a.mode, a.crop_w, a.pieces) == (b.mode, b.crop_w, b.pieces)


# ------------------------------------------------------------------ end to end: a thread, then the creator's choices

def test_a_job_joins_a_confirmed_thread_and_the_creator_can_switch_edits_off(env, monkeypatch):
    s = env["scripted"]
    real_llm = s.llm
    # Two later lines carry the story's payoff (same word counts, so the test video's sound still lines up).
    import test_repurpose_worker as trw
    said = list(trw.ALL_SENTENCES)
    said[20] = "Six months later the rent story ended with a full time job there."
    said[21] = "My old boss later said hiring me back was his best decision ever."
    w2, seg2, _end = trw.script_words(said)
    monkeypatch.setattr(s, "transcribe", lambda path, prompt: {"text": " ".join(said), "language": "english",
                                                               "segments": seg2, "words": w2})
    env["worker"].deps.transcribe_chunk = s.transcribe
    env["db"].execute("delete from public.repurpose_transcripts")     # the same test video was transcribed before

    def llm(messages):
        prompt = messages[-1]["content"] if isinstance(messages[-1]["content"], str) else messages[-1]["content"][0]["text"]
        out = real_llm(messages)
        if "analysing a long video" in prompt:
            data = json.loads(out)
            # The story (S0-S7) gets its payoff from 16 sentences later (two "ordinary" lines stand in for it).
            data["threads"] = [{"relation": "setup_payoff", "parts": [{"start_sentence": 4, "end_sentence": 7},
                                                                      {"start_sentence": 20, "end_sentence": 21}],
                                "why": "what happened after the call"}]
            return json.dumps(data)
        if "strict senior editor" in prompt:
            data = json.loads(out)
            for rv in data["reviews"]:
                rv["join_faithful"] = True if "LATER IN THE VIDEO" in prompt else None
            return json.dumps(data)
        return out
    s.llm = llm
    env["worker"].deps.llm = llm
    env["worker"].deps.analysis = env["worker"].deps.review = env["worker"].deps.qa = True
    uid, job_id = submit(env)
    assert env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded", job.get("error_detail")
    sel = job["result"]["selection"]["analysis"]
    assert sel["story_links"]["linked"] == 1 and sel["story_links"]["confirmed"] == 1, sel.get("story_links")
    story = job["result"]["clips"][0]
    ed = story["edits"]
    assert ed["joined"] and len(ed["parts"]) == 2 and ed["join"]["relation"] == "setup_payoff"
    assert ed["parts"][1][0] > ed["parts"][0][1] + 8, "the second part really comes from later in the video"
    assert story["retention"]["risk"] is not None
    folder = os.path.dirname(local_path(story["processed_video_url"]))
    edl = json.load(open(os.path.join(folder, "clip_1_edl.json")))
    assert edl["parts_unjoined"] and edl["timeline"] is not None and edl["words"]
    joined_len = float(probe(local_path(story["processed_video_url"]))["format"]["duration"])

    # The creator turns the join off: the clip is made again from the original, without the later part.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import repurpose as r
    monkeypatch.setattr(r, "job_store", env["store"])
    monkeypatch.setattr(r, "storage", env["storage"])
    app = FastAPI()
    app.include_router(r.router)
    app.dependency_overrides[r._get_current_user_id] = lambda: str(uid)
    client = TestClient(app)
    calls_before = s.llm_calls
    res = client.post(f"/api/v1/repurpose/{job_id}/clips/1/edits", json={"join": False})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["version"] == "e01111" and body["edits"]["joined"] is False and len(body["edits"]["parts"]) == 1
    single_len = float(probe(local_path(body["url"]))["format"]["duration"])
    assert single_len < joined_len - 5
    assert body["quality"]["items"] and s.llm_calls == calls_before, "no new AI calls"
    again = client.post(f"/api/v1/repurpose/{job_id}/clips/1/edits", json={"join": False}).json()
    assert again["url"] == body["url"] and again["edits"]["joined"] is False, "made once, then reused"
    same = client.post(f"/api/v1/repurpose/{job_id}/clips/1/edits", json={}).json()
    assert same["version"] == "original" and same["url"] == story["processed_video_url"]
    # That version can be exported for a platform too.
    tk = client.post(f"/api/v1/repurpose/{job_id}/clips/1/platform/tiktok?version=e01111")
    assert tk.status_code == 200 and tk.json()["url"].endswith("clip_1_e01111_tiktok.mp4")
    # Once the original is deleted, edits can't be re-made (and the creator is told why).
    env["db"].execute("update repurpose_jobs set source_deleted_at = now() where id = %s", (job_id,))
    gone = client.post(f"/api/v1/repurpose/{job_id}/clips/1/edits", json={"zoom": False})
    assert gone.status_code == 409 and "no longer stored" in gone.json()["detail"]
    app.dependency_overrides[r._get_current_user_id] = lambda: "someone-else"
    assert client.post(f"/api/v1/repurpose/{job_id}/clips/1/edits", json={"join": False}).status_code == 403
