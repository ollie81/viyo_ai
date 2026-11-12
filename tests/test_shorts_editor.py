"""The AI Shorts editor's new passes: clip structure (digressions dropped, a
later part of the same topic joined, weak sign-offs trimmed), retention
tightening (long pauses, abandoned restarts), the cold open, jump-cut zooms,
platform caption layouts, packaging that is checked against the transcript,
the A/V sync check and the creator's quality checklist; and real renders of a
job that uses them.

Scripted transcriber and LLM, local storage, real FFmpeg and Postgres. These
prove the editing decisions and the files produced. They do NOT prove how the
clips perform with real viewers, or how well a real model follows the new
prompt fields: that needs real-video testing.
"""
import json
import os
import re
import subprocess

import pytest

import captions
import clip_edit
import clip_package
import clip_qa
import clip_render
import clip_selection as cs
import platforms as pf
import video_analysis as va
from test_repurpose_worker import (balance, env, job_row, local_path, probe, source_video, submit)  # noqa: F401
from conftest import db  # noqa: F401


def sentences_from(texts, start=0.0, wps=2.6, pause=0.5, pauses=None):
    """Words + sentences for a scripted transcript; `pauses` {sentence index: seconds before it}."""
    words, t = [], start
    for i, s in enumerate(texts):
        t += (pauses or {}).get(i, 0.0)
        for tok in s.split():
            words.append({"text": tok, "word": re.sub(r"[^\w']", "", tok), "start": round(t, 3), "end": round(t + 0.3, 3)})
            t += 1 / wps
        t += pause
    return words, cs.build_sentences(words)


# ------------------------------------------------------------------ structure

TALK = [
    "Nobody tells you that quitting your job is the easy part.",           # 0
    "The hard part starts three weeks later when the money stops.",        # 1
    "By the way my cat just walked across the keyboard again.",            # 2  digression
    "I had eleven hundred dollars left and a landlord waiting.",           # 3
    "So I called my old boss and asked for double my old rate.",           # 4
    "That one phone call paid my rent for six months.",                    # 5
    "So yeah.",                                                            # 6  sign-off
]


def test_a_digression_is_dropped_but_never_when_the_next_line_leans_on_it():
    _w, sents = sentences_from(TALK)
    st = clip_edit.plan_structure({"drop_sentences": [2]}, sents, 0, 5)
    assert st["ranges"] == [(0, 1), (3, 5)] and st["dropped"] == [2] and not st["refused"]
    # Dropping 4 would leave "That one phone call..." pointing at nothing.
    st = clip_edit.plan_structure({"drop_sentences": [4]}, sents, 0, 5)
    assert st["ranges"] == [(0, 5)] and st["refused"] == ["drop_breaks_meaning"]
    st = clip_edit.plan_structure({"drop_sentences": [1, 2, 3]}, sents, 0, 5)
    assert st["refused"] == ["drop_too_much"]
    st = clip_edit.plan_structure({"drop_sentences": [0, 5]}, sents, 0, 5)
    assert st["ranges"] == [(0, 5)], "never the first or last sentence"


def test_a_weak_sign_off_is_trimmed_when_the_thought_already_ended():
    _w, sents = sentences_from(TALK)
    st = clip_edit.plan_structure({}, sents, 0, 6)
    assert st["ranges"] == [(0, 5)] and st["trimmed_ending"] == [6]


LONG = TALK[:6] + [f"Ordinary talk number {i} about the weather and traffic today." for i in range(12)] + [
    "Here is how the rent story ended: my old boss hired me full time.",     # 18
    "And the rate stayed double, which changed my whole career.",            # 19
    "But the real question is why landlords never care.",                    # 20
]


def test_a_later_part_of_the_same_story_is_joined_in_order_and_unrelated_ones_are_not():
    _w, sents = sentences_from(LONG)
    st = clip_edit.plan_structure({"continues": {"start_sentence": 18, "end_sentence": 19}}, sents, 0, 5)
    assert st["joined"] == (18, 19) and st["ranges"] == [(0, 5), (18, 19)]
    st = clip_edit.plan_structure({"continues": {"start_sentence": 19, "end_sentence": 20}}, sents, 0, 5)
    assert st["joined"] is None and st["refused"] == ["join_opens_as_reply"]
    st = clip_edit.plan_structure({"continues": {"start_sentence": 6, "end_sentence": 8}}, sents, 0, 5)
    assert st["refused"] == ["join_too_close"]
    st = clip_edit.plan_structure({"continues": {"start_sentence": 2, "end_sentence": 3}}, sents, 4, 5)
    assert st["refused"] == ["join_out_of_order"]
    # With the whole-video topic map, only the same topic may be joined.
    topic = lambda i: 0 if i <= 5 else 1                            # noqa: E731
    st = clip_edit.plan_structure({"continues": {"start_sentence": 18, "end_sentence": 19}}, sents, 0, 5, topic)
    assert st["refused"] == ["join_other_topic"]


def test_selection_turns_a_join_into_source_parts_and_scores_only_the_main_moment():
    words, sents = sentences_from(LONG)
    raw = {"start_sentence": 0, "end_sentence": 5, "category": "story",
           "scores": {k: 8 for k in cs.RUBRIC_KEYS}, "title": "The call that paid my rent", "reason": "x",
           "overlay_hook": "", "caption": "c", "hashtags": ["rent"], "topic": "rent",
           "drop_sentences": [2], "continues": {"start_sentence": 18, "end_sentence": 19}}
    plain, _ = cs.evaluate_candidate({**raw, "drop_sentences": [], "continues": None}, sents, words, 300.0)
    cand, why = cs.evaluate_candidate(raw, sents, words, 300.0)
    assert why is None and len(cand.parts) == 3
    assert cand.parts[0][0] == plain.start and cand.parts[-1][0] > cand.parts[0][1]
    assert "cat just walked" not in cand.transcript and "hired me full time" in cand.transcript
    assert cand.structure["joined_at"] == cand.parts[-1][0] and cand.score == plain.score
    assert cs._conflicts(plain, [cand]) == "overlaps_better_clip"


# ------------------------------------------------------------------ pacing

def test_long_pauses_and_abandoned_restarts_are_cut_but_emphasis_is_kept():
    words = []
    t = 0.0
    for tok, gap in [("I", 0.1), ("I", 0.1), ("think", 0.1), ("this", 1.6), ("is", 0.1), ("very", 0.1),
                     ("very", 0.1), ("we", 0.1), ("went", 0.1), ("we", 0.1), ("went", 0.1), ("home.", 0.1)]:
        words.append({"text": tok, "start": round(t, 2), "end": round(t + 0.3, 2)})
        t += 0.3 + gap
    blocks = [(0.0, t)]
    out, facts = clip_edit.tighten(blocks, words)
    assert facts["pauses_tightened"] == 1 and facts["pause_seconds_removed"] == pytest.approx(1.3, abs=0.01)
    assert facts["repeats_removed"] == ["I", "we went"], "'very very' is emphasis, kept"
    kept = [w["text"] for w in words if any(s <= (w["start"] + w["end"]) / 2 < e for s, e in out)]
    assert kept == ["I", "think", "this", "is", "very", "very", "we", "went", "home."]
    # A repeat across two speakers is a conversation, not a restart.
    _out, facts = clip_edit.tighten(blocks, words, speaker_at=lambda x: "A" if x < 0.4 else "B")
    assert "I" not in facts["repeats_removed"]


def test_a_slow_opening_gets_a_verbatim_preview_of_its_strongest_later_line():
    texts = ["So, okay, today I want to talk about rent.", "I was broke and scared about it.",
             "Then my landlord called me at midnight.", "Nobody ever pays double rent, never!",
             "And that was the end of it."]
    words, sents = sentences_from(texts, start=1.2)
    tz = clip_edit.choose_teaser(sents, [0, 1, 2, 3, 4], words, 1.0)
    assert tz["sentence"] == 3 and tz["text"] == "Nobody ever pays double rent, never!"
    assert tz["start"] < sents[3].start < sents[3].end < tz["end"]
    strong = ["Nobody ever pays double rent, never!"] + texts[1:3]
    words, sents = sentences_from(strong, start=1.0)
    assert clip_edit.choose_teaser(sents, [0, 1, 2], words, 1.0) is None, "a strong opening is left alone"


# ------------------------------------------------------------------ visuals and captions

def test_jump_cuts_inside_one_shot_alternate_a_gentle_zoom_and_reset_at_camera_cuts_and_joins():
    plan = clip_render.CropPlan(mode="track", facts={"face_ratio": 0.9})

    class TL:
        usable = True

        def cuts_between(self, a, b):
            return [15.0] if a <= 15.0 <= b else []

    blocks = [(0, 3), (3.5, 6), (6.4, 9), (19.9, 23), (30, 33)]
    z = clip_render.jump_zoom_windows(blocks, TL(), plan, joins=(30,))
    assert z == [(3.0, 5.5)], "zoom on the 1st jump, back on the 2nd; camera cut and join reset it"
    assert clip_render.jump_zoom_windows(blocks, TL(), clip_render.CropPlan(mode="fit", facts={"face_ratio": 0.9})) == []
    assert clip_render.jump_zoom_windows(blocks, TL(), clip_render.CropPlan(mode="track", facts={"face_ratio": 0.1})) == []


def test_captions_are_placed_for_each_platform_and_a_join_is_marked():
    words, _ = sentences_from(["Nobody tells you the truth about rent.", "Here is how it ended."])
    blocks = [(0.0, 3.0), (3.2, 6.0)]
    default, _ = captions.build_ass(words, blocks)
    tiktok, facts = captions.build_ass(words, blocks, layout=pf.layout("tiktok"), markers=[(3.0, "LATER IN THE VIDEO")])
    shorts, _ = captions.build_ass(words, blocks, layout=pf.layout("shorts"))
    style = lambda a: next(ln for ln in a.splitlines() if ln.startswith("Style: Caption"))   # noqa: E731
    assert style(tiktok).endswith(",70,160,430,1") and style(shorts).endswith(",70,140,360,1")
    assert style(default) != style(tiktok) and facts["layout"] == "tiktok" and facts["markers"] == 1
    assert "Marker,,0,0,0,,LATER IN THE VIDEO" in tiktok and "0:00:03.00,0:00:04.40" in tiktok
    both = captions.join_ass(default, default, 6.2)
    times = [ln.split(",")[1] for ln in both.splitlines() if ln.startswith("Dialogue:")]
    assert times[0] == "0:00:00.00" and "0:00:06.20" in times, "second piece moved by the first's length"


# ------------------------------------------------------------------ packaging

CLIP = {"title": "Stop competing on price", "overlay_hook": None, "hashtags": ["pricing", "business"],
        "caption": "Raise your prices.",
        "transcript": "Charge what feels slightly uncomfortable and raise it after ten sales. "
                      "Every time I doubled my price the complaints went down. Stop competing on price."}


def test_packaging_keeps_only_lines_the_clip_supports_and_fits_each_platform():
    def llm(messages):
        return json.dumps({"clips": [{"id": 0,
            "titles": ["Why I doubled my price", "This will go viral", "Raise prices after 50 sales",
                       "Stop competing on price"],
            "hooks": ["The complaints went down", "Earn a million dollars"],
            "description": "Why doubling my price meant fewer complaints.",
            "thumbnail": [{"moment": "Speaker says complaints went down", "text": "Doubled my price"}],
            "tiktok": {"caption": "Every time I doubled my price the complaints went down.",
                       "hashtags": ["pricing", "fyp", "business", "sales", "money", "growth", "extra"]},
            "shorts": {"title": "Stop competing on price " * 8, "description": "Raise it after ten sales.",
                       "hashtags": ["pricing", "sales", "business", "money"]},
            "reels": {"caption": "Charge what feels slightly uncomfortable.", "hashtags": ["pricing"]}}]})
    (p,), facts = clip_package.package([CLIP], llm)
    assert p["source"] == "ai" and facts["written"] == 1
    assert p["titles"] == ["Why I doubled my price", "Stop competing on price"], "no virality promise, no 50 sales"
    assert p["hooks"] == ["The complaints went down"]
    tk, sh, rl = p["platforms"]["tiktok"], p["platforms"]["shorts"], p["platforms"]["reels"]
    assert "fyp" not in tk["hashtags"] and len(tk["hashtags"]) == 5 and len(sh["hashtags"]) == 3
    assert sh["title"] == "Why I doubled my price", "the over-long Shorts title was replaced by a vetted one"
    assert rl["hashtags"] == ["pricing", "business"], "too few tags are topped up from the clip's own, never invented"
    assert p["thumbnail"][0]["text"] == "Doubled my price"


def test_packaging_falls_back_to_what_selection_wrote_when_the_call_fails():
    def broken(messages):
        raise RuntimeError("model down")
    (p,), facts = clip_package.package([CLIP], broken)
    assert p["source"] == "selection" and p["titles"] == ["Stop competing on price"] and facts["fallback"] == 1
    assert p["platforms"]["tiktok"]["hashtags"] == ["pricing", "business"]


# ------------------------------------------------------------------ quality

def test_the_checklist_never_shows_an_unchecked_item_as_passed(tmp_path):
    qa = {"failed": [], "checks": {"format": {"ok": True}, "framing": {"ok": True}, "silence": {"ok": False},
                                   "av_sync": {"ok": True}, "caption_sync": {"ok": True}, "loudness": {"ok": True}}}
    s = clip_qa.summary(qa, {"status": "unreviewed"}, {}, {"audio": 7})
    by = {i["key"]: i["status"] for i in s["items"]}
    assert by["stands_alone"] == "unchecked" and by["opening"] == "unchecked" and by["pauses"] == "warning"
    assert s["status"] == "warning"
    s = clip_qa.summary({"failed": [], "checks": {}}, None, {}, None)
    assert s["status"] == "partly_checked" and all(i["status"] == "unchecked" for i in s["items"])
    good = clip_qa.summary({**qa, "checks": {**qa["checks"], "silence": {"ok": True}}},
                           {"status": "reviewed", "standalone": True, "opening": 8, "ending": 7},
                           {"cold_open": {"text": "Nobody pays double."}}, {"audio": 8})
    assert good["status"] == "passed" and "preview of a line from later" in good["items"][1]["note"]


def test_sound_and_picture_sync_is_measured_on_the_file(tmp_path):
    ok, bad = str(tmp_path / "ok.mp4"), str(tmp_path / "bad.mp4")
    base = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4"]
    subprocess.run(base + ["-f", "lavfi", "-i", "sine=d=4", "-c:v", "libx264", "-c:a", "aac", "-shortest", ok], check=True)
    subprocess.run(base + ["-f", "lavfi", "-i", "sine=d=2", "-c:v", "libx264", "-c:a", "aac", bad], check=True)
    assert clip_qa.check_av_sync(ok).ok
    c = clip_qa.check_av_sync(bad)
    assert not c.ok and c.severity == "fail" and c.detail["length_difference_s"] > 1.5


# ------------------------------------------------------------------ real renders

def test_a_job_delivers_edited_clips_with_a_checklist_platform_files_and_packaging(env, monkeypatch):
    real = clip_edit.choose_teaser
    picked = {}

    def teaser(sentences, ids, words, start):
        # The scripted video opens strongly, so force the preview path on the story clip to render it for real.
        t = real(sentences, ids, words, start)
        if t is None and sentences[ids[0]].text.startswith("Nobody tells you"):
            s = sentences[ids[6]]
            t = {"sentence": ids[6], "start": round(s.start - 0.04, 3), "end": round(s.end + 0.12, 3), "text": s.text}
        picked.setdefault("t", t)
        return t
    monkeypatch.setattr(clip_edit, "choose_teaser", teaser)
    env["worker"].deps.qa = True
    uid, job_id = submit(env)
    assert env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded", job.get("error_detail")
    story = job["result"]["clips"][0]
    ed = story["edits"]
    assert ed["cold_open"]["text"] == "That one phone call paid my rent for six months."
    info = probe(local_path(story["processed_video_url"]))
    h = story["highlight"]
    # The preview plays first, then the whole moment (minus removed pauses).
    assert float(info["format"]["duration"]) == pytest.approx(
        h["end_time"] - h["start_time"] + ed["cold_open"]["seconds"], abs=2.5)
    q = {i["key"]: i["status"] for i in story["quality"]["items"]}
    assert q["opening"] == "passed" and q["technical"] == "passed" and q["sync"] == "passed"
    assert q["stands_alone"] == "unchecked", "the review was switched off in this test, so it isn't 'passed'"
    assert story["platforms"] == ["tiktok", "shorts", "reels"]
    folder = os.path.dirname(local_path(story["processed_video_url"]))
    assert os.path.exists(os.path.join(folder, "clip_1_master.mp4"))
    tk = open(os.path.join(folder, "clip_1_tiktok.ass")).read()
    assert ",70,160,430,1" in tk
    pk = story["packaging"]
    assert pk["source"] == "ai" and "Go viral with pricing" not in pk["titles"]
    assert job["result"]["selection"]["packaging"]["written"] == 2
    assert env["scripted"].package_calls == 1, "one packaging call for the whole job"

    # The creator exports a TikTok version: captions placed for TikTok, burned onto the master, with its packaging.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import repurpose as r
    monkeypatch.setattr(r, "job_store", env["store"])
    monkeypatch.setattr(r, "storage", env["storage"])
    app = FastAPI()
    app.include_router(r.router)
    app.dependency_overrides[r._get_current_user_id] = lambda: str(uid)
    client = TestClient(app)
    res = client.post(f"/api/v1/repurpose/{job_id}/clips/1/platform/tiktok")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["packaging"]["hashtags"] and body["specs"]["label"] == "TikTok"
    out = local_path(body["url"])
    assert out.endswith("clip_1_tiktok.mp4") and os.path.exists(out)
    assert float(probe(out)["format"]["duration"]) == pytest.approx(float(info["format"]["duration"]), abs=0.3)
    assert client.post(f"/api/v1/repurpose/{job_id}/clips/1/platform/myspace").status_code == 404
    app.dependency_overrides[r._get_current_user_id] = lambda: "someone-else"
    assert client.post(f"/api/v1/repurpose/{job_id}/clips/1/platform/reels").status_code == 403


def test_the_editorial_review_sees_a_joined_clip_as_two_parts():
    import clip_review
    words, sents = sentences_from(LONG)
    raw = {"start_sentence": 0, "end_sentence": 5, "category": "story", "scores": {k: 8 for k in cs.RUBRIC_KEYS},
           "title": "The call that paid my rent", "reason": "x", "overlay_hook": "", "caption": "c",
           "hashtags": ["rent"], "topic": "rent", "drop_sentences": [2],
           "continues": {"start_sentence": 18, "end_sentence": 19}}
    cand, _ = cs.evaluate_candidate(raw, sents, words, 300.0)
    block = clip_review._clip_block(cand, 0, sents, None)
    assert '[ON SCREEN: "LATER IN THE VIDEO"] Here is how the rent story ended' in block
    assert "cat just walked" not in block.split("CLIP:")[1].split("AFTER:")[0]
    assert "AFTER: But the real question" in block
