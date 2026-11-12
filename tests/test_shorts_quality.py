"""AI Shorts quality upgrade: the final inspection of rendered clips
(programmatic checks plus a mocked Decisions request on frames from the
finished video), creator-chosen moments, two-speaker and head-safe framing,
visual-only videos without transcription, and learning from what each creator
keeps or skips.

All AI responses are mocked; renders are real FFmpeg on synthetic footage
(test patterns, not people). They prove the decisions and the files, not how
real footage or a real model behaves: that needs real videos.
"""
import json
import os
import subprocess

import numpy as np
import pytest

import captions
import clip_inspect
import clip_render
import clip_selection as cs
import creator_insights as ci
import video_analysis as va
from test_decisions import Fake
from test_repurpose_worker import (balance, env, job_row, local_path, probe, source_video, submit)  # noqa: F401
from test_shorts_editor import sentences_from
from conftest import db  # noqa: F401


def clip(path, size="1080x1920", seconds=4, dark=False, color=None):
    src = f"color=c={color}:s={size}:d={seconds}" if color else f"testsrc2=size={size}:rate=30:duration={seconds}"
    vf = ["-vf", "eq=brightness=-0.85"] if dark else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-f", "lavfi", "-i",
                    f"sine=d={seconds}", *vf, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", path], check=True)
    return path


# ------------------------------------------------------------------ final inspection

def test_a_clip_that_starts_or_ends_mid_sentence_is_found_from_the_words():
    words, _ = sentences_from(["Nobody tells you the truth about rent.", "It costs more than you think.",
                               "And then it gets worse."])
    whole = [(words[0]["start"], words[-1]["end"] + 0.1)]
    e = clip_inspect.sentence_edges(words, whole)
    assert e["start_complete"] and e["end_complete"]
    mid = [(words[2]["start"], words[9]["end"])]                   # "the truth ... more than"
    e = clip_inspect.sentence_edges(words, mid)
    assert e["start_complete"] is False and e["end_complete"] is False
    assert e["before"] == "Nobody tells" and e["after"].startswith("than you think.")


def test_caption_readability_is_measured_on_the_burned_caption_file():
    words, _ = sentences_from(["Nobody tells you the truth about rent."])
    ass, _f = captions.build_ass(words, [(0.0, 4.0)])
    assert clip_inspect.caption_readability(ass)["readable"]
    tiny = ass.replace("Style: Caption,DejaVu Sans,62", "Style: Caption,DejaVu Sans,30")
    r = clip_inspect.caption_readability(tiny)
    assert r["readable"] is False and r["font_size"] == 30


def test_the_inspection_asks_decisions_about_frames_of_the_finished_clip(tmp_path):
    path = clip(str(tmp_path / "c.mp4"))
    words, _ = sentences_from(["Nobody tells you the truth about rent.", "It costs more than you think."])
    blocks = [(0.0, words[-1]["end"] + 0.1)]
    fake = Fake(overrides={"subject": {"s": 0.3}})
    res = clip_inspect.inspect(path, 4.0, words, blocks, None, {"crop_mode": "track"}, str(tmp_path), fake.client())
    q = fake.requests[0]["json"]
    assert {x["name"] for x in q["questions"]} == {"subject", "defects", "captions", "starts", "ends"}
    sub = next(x for x in q["questions"] if x["name"] == "subject")
    assert sum(p["type"] == "input_image" for p in sub["instructions"]) == 3, "three frames from the final video"
    assert res["action"] == "reframe" and res["cost_usd"] > 0, "subject cut off by the crop: re-render once, whole frame"
    by = {i["key"]: i["status"] for i in res["items"]}
    assert by["subject_visible"] == "failed" and by["starts_right"] == "passed"


def test_rendering_defects_reject_and_dark_footage_is_flagged(tmp_path):
    words, _ = sentences_from(["Nobody tells you the truth about rent."])
    blocks = [(0.0, words[-1]["end"] + 0.1)]
    res = clip_inspect.inspect(clip(str(tmp_path / "c.mp4")), 4.0, words, blocks, None, {"crop_mode": "fit"},
                               str(tmp_path), Fake(overrides={"defects": {"p": 0.9}}).client())
    assert res["action"] == "reject"
    dark = clip_inspect.inspect(clip(str(tmp_path / "d.mp4"), dark=True), 4.0, words, blocks, None,
                                {"crop_mode": "fit"}, str(tmp_path), Fake(overrides={"defects": {"p": 0.05}}).client())
    by = {i["key"]: i["status"] for i in dark["items"]}
    assert by["lighting"] == "warning" and dark["action"] == "flag"


def test_without_decisions_the_ai_items_are_unchecked_never_passed(tmp_path):
    words, _ = sentences_from(["Nobody tells you the truth about rent."])
    res = clip_inspect.inspect(clip(str(tmp_path / "c.mp4")), 4.0, words, [(0.0, words[-1]["end"] + 0.1)], None,
                               {"crop_mode": "fit"}, str(tmp_path), None)
    by = {i["key"]: i["status"] for i in res["items"]}
    assert by["subject_visible"] == by["render_defects"] == "unchecked" and by["starts_right"] == "passed"


# ------------------------------------------------------------------ the creator's request

def test_the_creators_request_shapes_selection_and_off_request_moments_are_left_out():
    req = {"focus": "funny", "text": "the bits where they argue about pizza"}
    words, sents = sentences_from(["Nobody tells you the truth about rent.", "It costs more than you think.",
                                   "Pineapple on pizza is a crime, he said.", "And they argued for an hour."] * 3)
    prompt = cs._proposal_prompt(sents, 0, len(sents) - 1, 120.0, 5, request=req)
    assert 'THE CREATOR WANTS: funny moments' in prompt and '"the bits where they argue about pizza"' in prompt
    assert '"matches_request": int' in prompt and "Do not include a moment just to reach a count" in prompt
    base = {"start_sentence": 0, "end_sentence": 5, "category": "story", "scores": {k: 8 for k in cs.RUBRIC_KEYS},
            "title": "t", "reason": "r", "overlay_hook": "", "caption": "c", "hashtags": ["x"], "topic": "x"}
    ctx = cs.AnalysisContext(request=req)
    off, why = cs.evaluate_candidate({**base, "matches_request": 2}, sents, words, 120.0, ctx)
    assert off is None and why == "off_request"
    on, why = cs.evaluate_candidate({**base, "matches_request": 8}, sents, words, 120.0, ctx)
    assert why is None and on.request_match == 8
    assert cs.request_text({"focus": "best", "text": ""}) == "", "best overall: the normal judgement"


def test_decisions_also_checks_each_clip_against_the_request():
    import clip_decisions as cd
    words, sents = sentences_from(["Nobody tells you the truth about rent.", "It costs more than you think."] * 4)
    base = {"start_sentence": 0, "end_sentence": 5, "category": "story", "scores": {k: 8 for k in cs.RUBRIC_KEYS},
            "title": "t", "reason": "r", "overlay_hook": "", "caption": "c", "hashtags": ["x"], "topic": "x"}
    c, _ = cs.evaluate_candidate(base, sents, words, 120.0)
    reviews, _f = cd.review([c], sents, None, None, None, Fake(overrides={"c0_matches": {"p": 0.1}}).client(),
                            request="THE CREATOR WANTS: funny moments.")
    assert reviews[0]["verdict"] == "reject" and "Not the kind of moment you asked for" in reviews[0]["issues"][-1]


# ------------------------------------------------------------------ framing

def _timeline(faces_per_sample, cuts=(), n=40, w=1920, h=1080):
    times = np.arange(n) * 0.5
    cut = np.zeros(n, dtype=bool)
    for c in cuts:
        cut[int(c / 0.5)] = True
    return va.VisualTimeline(times=times, sharpness=np.full(n, 40.0), brightness=np.full(n, 0.5),
                             faces=[faces_per_sample(k) for k in range(n)], cut_before=cut,
                             shots=[va.Shot(0, n * 0.5, 0, n - 1)], frame_w=w, frame_h=h)


TWO_APART = lambda k: [va.Face(0.2, 0.4, 0.08, 0.16, 0.9), va.Face(0.8, 0.42, 0.08, 0.16, 0.9)]   # noqa: E731
ONE = lambda k: [va.Face(0.5, 0.4, 0.1, 0.2, 0.9)]                                                 # noqa: E731


def test_two_people_too_far_apart_are_stacked_with_their_heads_in_frame(tmp_path):
    plan = clip_render.plan_crop(_timeline(TWO_APART), [(1.0, 9.0), (9.6, 18.0)], 1920, 1080, refine=False)
    assert plan.mode == "stack" and len(plan.stack) == 2, "a pause cut doesn't stop it; it's one continuous shot"
    for (x, y, w, h), face in zip(plan.stack, TWO_APART(0)):
        assert x <= face.cx * 1920 - face.w * 1920 / 2 and face.cx * 1920 + face.w * 1920 / 2 <= x + w
        assert y <= (face.cy - face.h / 2) * 1080, "the top of the head is inside the window"
        assert abs(w / h - 1080 / 960) < 0.02 and x + w <= 1920 and y + h <= 1080
    out = str(tmp_path / "s.mp4")
    clip_render.render(clip(str(tmp_path / "src.mp4"), size="1920x1080", seconds=8), out, [(0.0, 6.0)], plan)
    v = next(s for s in probe(out)["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)


def test_one_speaker_a_camera_cut_or_no_people_never_stack():
    assert clip_render.plan_crop(_timeline(ONE), [(1.0, 18.0)], 1920, 1080, refine=False).mode == "track"
    assert clip_render.plan_crop(_timeline(TWO_APART, cuts=(8.0,)), [(1.0, 18.0)], 1920, 1080, refine=False).mode != "stack"
    assert clip_render.plan_crop(_timeline(lambda k: []), [(1.0, 18.0)], 1920, 1080, refine=False).mode == "fit"
    portrait = clip_render.plan_crop(_timeline(ONE, w=1080, h=1920), [(1.0, 18.0)], 1080, 1920, refine=False)
    assert portrait.mode == "center", "portrait footage is used whole, never stretched"


def test_zooms_keep_heads_in_and_skip_close_ups():
    f = clip_render._zoom_filter([(1.0, 3.0)], face_top=0.01)
    assert "y='if(between(t,1.000,2.999),0,0)'" in f, "a head at the very top: no crop from the top"
    assert "if(between(t,1.000,2.999),53,0)" in clip_render._zoom_filter([(1.0, 3.0)], face_top=0.3)
    plan = clip_render.CropPlan(mode="track", facts={"face_ratio": 0.9, "face_height": 0.5})
    w, facts = clip_render.zoom_windows([(0, 5), (5.5, 12)], None, plan, sentence_starts=(8.0,))
    assert w == [] and facts["skipped"] == "close_up"


# ------------------------------------------------------------------ learning from the creator

def test_the_creators_own_choices_nudge_their_next_selection_only_a_little():
    ev = [{"event": "downloaded", "category": "humor", "duration": 22}] * 6 + \
         [{"event": "not_useful", "category": "advice", "duration": 50}] * 3
    prior = ci.build_feedback_prior(ev)

    class C:
        def __init__(self, category, duration):
            self.category, self.duration = category, duration
    up, _ = prior(C("humor", 20))
    down, _ = prior(C("advice", 55))
    assert 0 < up <= 3 and -3 <= down < 0
    assert ci.build_feedback_prior(ev[:3]) is None, "too little to act on"
    assert ci.preferred_length([{"event": "downloaded", "duration": 20}] * 8) == "short"
    assert ci.preferred_length([{"event": "downloaded", "duration": 20}] * 4 + [{"event": "downloaded", "duration": 50}] * 4) is None


# ------------------------------------------------------------------ in real jobs

def _api(env, monkeypatch, uid):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import repurpose as r
    monkeypatch.setattr(r, "job_store", env["store"])
    monkeypatch.setattr(r, "storage", env["storage"])
    app = FastAPI()
    app.include_router(r.router)
    app.dependency_overrides[r._get_current_user_id] = lambda: str(uid)
    return TestClient(app), app, r


def test_a_rendered_clip_is_inspected_reframed_once_and_the_checklist_and_cost_are_recorded(env, monkeypatch):
    import decisions as dc
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler_overrides():
        calls["n"] += 1
        return {"subject": {"s": 0.2}, "defects": {"p": 0.05}} if calls["n"] == 1 else {"defects": {"p": 0.05}}
    fake = Fake()
    real_call = fake.__call__

    class Flaky(Fake):
        def __call__(self, request):
            data = json.loads(request.content)
            if any(q["name"] == "subject" for q in data["questions"]):
                self.overrides = handler_overrides()
            else:
                self.overrides = {"defects": {"p": 0.05}}
            return super().__call__(request)
    flaky = Flaky()
    env["worker"].deps.qa = True
    env["worker"].deps.decisions = flaky.client()
    uid, job_id = submit(env)
    assert env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded", job.get("error_detail")
    first = job["clips"][0]["render_meta"]
    assert first["render"]["retry"] == "fit_after_inspection" and first["render"]["crop_mode"] == "fit"
    q = {i["key"]: i["status"] for i in job["result"]["clips"][0]["quality"]["items"]}
    assert {"starts_right", "ends_naturally", "subject_visible", "render_defects", "lighting"} <= set(q)
    assert job["result"]["selection"]["final_inspection_usd"] > 0
    del real_call


def test_a_clip_cut_mid_sentence_is_rejected_after_rendering_and_a_spare_takes_its_place(env, monkeypatch):
    import clip_inspect as insp
    real = insp.sentence_edges
    state = {"n": 0}

    def edges(words, blocks):
        e = real(words, blocks)
        state["n"] += 1
        if state["n"] == 1:
            e = {**e, "start_complete": False}            # the first clip rendered starts mid-sentence
        return e
    monkeypatch.setattr(insp, "sentence_edges", edges)
    env["worker"].deps.qa = True
    env["worker"].deps.decisions = Fake(overrides={"starts": {"p": 0.1}, "defects": {"p": 0.05}}).client()
    uid, job_id = submit(env)
    assert env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded"
    rejected = job["result"]["selection"].get("quality_rejected") or []
    assert rejected and rejected[0]["failed"] == ["final_inspection"]


def test_visual_moments_skip_transcription(env):
    uid, job_id = submit(env)
    assert env["store"].set_moment_focus(str(job_id), str(uid), "visual")
    env["worker"].run_once()
    job = job_row(env, job_id)
    assert env["scripted"].transcribe_calls == 0, "no speech-to-text was paid for"
    assert job["status"] in ("succeeded", "failed")
    if job["status"] == "succeeded":
        assert job["result"]["selection"]["mode"] == "visual"


def test_feedback_is_recorded_on_own_clips_only_and_shapes_nothing_for_others(env, monkeypatch):
    uid, job_id = submit(env)
    assert env["worker"].run_once()
    client, app, r = _api(env, monkeypatch, uid)
    for ev in ("downloaded", "downloaded", "not_useful"):
        assert client.post(f"/api/v1/repurpose/{job_id}/clips/1/feedback", json={"event": ev}).json()["ok"]
    events = env["store"].feedback_events(str(uid))
    assert sorted(e["event"] for e in events) == ["downloaded", "not_useful"], "one of each per clip per day"
    assert events[0]["category"] and events[0]["duration"] > 0
    app.dependency_overrides[r._get_current_user_id] = lambda: "11111111-1111-1111-1111-111111111111"
    assert client.post(f"/api/v1/repurpose/{job_id}/clips/1/feedback", json={"event": "kept"}).status_code == 404
    assert env["store"].feedback_events("11111111-1111-1111-1111-111111111111") == []
    assert client.post(f"/api/v1/repurpose/{job_id}/clips/1/feedback", json={"event": "viral"}).status_code == 422
