"""Unit tests for captions, creator insights, video map, editorial review, crop planning, QA."""
import json
import types

import numpy as np
import pytest

import captions as cp
import clip_qa
import clip_render as cr
import clip_review
import clip_selection as cs
import creator_insights as ci
import video_analysis as va
import video_map as vm


def W(text, s, e, sp=None):
    return {"text": text, "start": s, "end": e}


SAMPLE = [W("Nobody", 0.0, 0.4), W("tells", 0.45, 0.8), W("you", 0.85, 1.0), W("this.", 1.05, 1.5),
          W("Quitting", 2.6, 3.0), W("is", 3.05, 3.2), W("easy,", 3.25, 3.7), W("staying", 3.75, 4.2),
          W("is", 4.25, 4.4), W("hard.", 4.45, 5.0)]


# ------------------------------------------------------------------ captions
def test_captions_use_punctuated_text_and_break_at_sentence_ends():
    ass, facts = cp.build_ass(SAMPLE, [(0.0, 5.2)], hook="Nobody tells you this", style="clean")
    ev = cp.caption_timeline(ass)
    texts = [t for _, _, t in ev]
    assert any("this." in t for t in texts) and any("easy," in t for t in texts)
    phrases = cp.group_phrases(cp.words_in_clip(SAMPLE, [(0.0, 5.2)]))
    assert [p[-1].text for p in phrases][0] == "this."          # a phrase never spans a sentence end
    assert facts["hook_shown"] is True


def test_cut_words_are_not_captioned_and_times_are_remapped():
    blocks = [(0.0, 1.6), (2.5, 5.2)]                           # the pause between is removed
    ev = cp.caption_timeline(cp.build_ass(SAMPLE, blocks)[0])
    first_after_cut = next(s for s, _, t in ev if "QUITTING" in t.upper())
    assert first_after_cut == pytest.approx(1.6 + 0.1, abs=0.05)
    only_first = cp.build_ass(SAMPLE, [(0.0, 1.6)])[0]
    assert "QUITTING" not in only_first.upper()


def test_captions_stay_inside_safe_zone_and_hold_never_overlaps_next_phrase():
    ass, _ = cp.build_ass(SAMPLE, [(0.0, 5.2)])
    style = next(l for l in ass.splitlines() if l.startswith("Style: Caption"))
    margin_v = int(style.split(",")[-2])
    assert margin_v >= 350
    ev = cp.caption_timeline(ass)
    for (s1, e1, _), (s2, e2, _) in zip(ev, ev[1:]):
        assert e1 <= s2 + 0.011 or e1 - s2 < 0.011 or s2 >= s1


def test_no_words_means_no_captions():
    assert cp.build_ass([], [(0.0, 3.0)])[0] is None


# ------------------------------------------------------------------ creator insights
def rows(n, category="story", completion=0.7, views=1000, duration=30):
    return [{"views": views, "completion_rate": completion, "rewatch_rate": 0.1, "shares": 40,
             "saves": 30, "comments": 10, "category": category, "duration": duration} for _ in range(n)]


def test_prior_is_inert_without_enough_data():
    assert ci.build_prior(rows(3)) is None
    assert ci.build_prior(rows(20, views=50)) is None            # too few views each


def test_prior_is_bounded_and_prefers_what_worked():
    data = rows(10, "story", 0.9) + rows(10, "humor", 0.2)
    prior = ci.build_prior(data)
    story = types.SimpleNamespace(category="story", duration=30)
    humor = types.SimpleNamespace(category="humor", duration=30)
    ps, info = prior(story)
    ph, _ = prior(humor)
    assert ps > 0 > ph
    assert abs(ps) <= ci.MAX_POINTS and abs(ph) <= ci.MAX_POINTS
    assert info["status"] == "applied"


# ------------------------------------------------------------------ video map
def sentences(n=8):
    return [cs.Sentence(i, i * 5.0, i * 5.0 + 4, i * 10, i * 10 + 9, f"Sentence number {i} here.", True, 1.0)
            for i in range(n)]


def test_video_map_validates_ranges_and_kinds():
    good = {"summary": "A talk.", "format": "talk", "speakers": [{"label": "A", "role": "host", "description": ""},
                                                                  {"label": "Z", "role": "x", "description": ""}],
            "topics": [{"start_sentence": 0, "end_sentence": 3, "kind": "story", "title": "t"},
                       {"start_sentence": 4, "end_sentence": 99, "kind": "story", "title": "bad range"}],
            "avoid": [{"start_sentence": 6, "end_sentence": 7, "kind": "outro"},
                      {"start_sentence": 1, "end_sentence": 2, "kind": "story"}]}
    out = vm.build_video_map(sentences(), ["A"] * 8, lambda m: json.dumps(good), speakers_reliable=True)
    assert len(out["topics"]) == 1 and len(out["avoid"]) == 1
    assert [s["label"] for s in out["speakers"]] == ["A"]        # "Z" was never detected


def test_video_map_failure_is_none_not_invented():
    assert vm.build_video_map(sentences(), None, lambda m: "not json") is None
    assert vm.build_video_map(sentences(), None, lambda m: json.dumps({"summary": ""})) is None


# ------------------------------------------------------------------ review
def cand(title, start_s, end_s, score=80):
    return types.SimpleNamespace(title=title, start_sentence=start_s, end_sentence=end_s, transcript="x", overlay_hook="Hook",
                                 overlay_status="ok", score=score, review={}, start=0, end=10)


def test_reject_reasons():
    assert clip_review.reject_reason({"verdict": "publish", "standalone": True, "opening": 8, "ending": 8}) is None
    assert clip_review.reject_reason({"verdict": "reject"}) == "review_rejected"
    assert clip_review.reject_reason({"verdict": "publish", "standalone": False}) is None
    assert clip_review.reject_reason({"verdict": "fix", "standalone": False}) == "review_not_standalone"
    assert clip_review.reject_reason({"verdict": "publish", "standalone": False,
                                      "issues": ["refers to a study never introduced"]}) == "review_not_standalone"
    assert clip_review.reject_reason({"verdict": "publish", "opening": 3}) == "review_weak_opening"
    assert clip_review.reject_reason({"verdict": "publish", "ending": 2}) == "review_weak_ending"
    assert clip_review.reject_reason({"verdict": "publish", "visual_ok": False}) == "review_bad_visual"


def test_rejected_clip_is_replaced_by_spare_and_inaccurate_hook_removed():
    a, b, spare = cand("a", 0, 1), cand("b", 2, 3), cand("spare", 4, 5)
    result = cs.SelectionResult([a, b], 2, 62, 100.0, [], "", [spare])
    reviews = {0: {"verdict": "reject", "standalone": True, "opening": 8, "ending": 8, "hook_accurate": None},
               1: {"verdict": "publish", "standalone": True, "opening": 8, "ending": 8, "hook_accurate": False},
               2: {"verdict": "publish", "standalone": True, "opening": 7, "ending": 7, "hook_accurate": True}}
    clip_review.apply_reviews(result, reviews, [a, b, spare])
    assert [c.title for c in result.clips] == ["b", "spare"]
    assert b.overlay_hook is None and b.overlay_status == "removed_by_review"
    assert result.rejected[-1]["reason"] == "review_rejected"


def test_missing_review_is_recorded_as_unreviewed_not_passed():
    a = cand("a", 0, 1)
    result = cs.SelectionResult([a], 1, 62, 100.0, [], "", [])
    clip_review.apply_reviews(result, {}, [a])
    assert a.review == {"status": "unreviewed"}


# ------------------------------------------------------------------ crop planning
def timeline(face_cx, n=40, cuts=(), bars=None):
    times = np.arange(n, dtype=float)
    faces = [[va.Face(face_cx(t), 0.4, 0.15, 0.25, 0.9)] for t in times]
    cut = np.zeros(n, dtype=bool)
    for c in cuts:
        cut[c] = True
    tl = va.VisualTimeline(times, np.full(n, 200.0), np.full(n, 120.0), faces, np.zeros((n, 48)), cut,
                           [], 320, 180, bars or {})
    return tl


def test_plan_follows_face_and_never_cuts_it_off():
    tl = timeline(lambda t: 0.3 if t < 20 else 0.7)
    plan = cr.plan_crop(tl, [(0.0, 39.0)], 1920, 1080, None, refine=False)
    assert plan.mode == "track"
    assert plan.facts["faces_cut_off_share"] <= 0.1
    assert plan.x_at(5) < plan.x_at(35)            # the crop moved right with the speaker


def test_plan_recentres_instantly_at_a_camera_cut():
    tl = timeline(lambda t: 0.25 if t < 10 else 0.75, cuts=(10,))
    plan = cr.plan_crop(tl, [(0.0, 39.0)], 1920, 1080, None, refine=False)
    assert plan.x_at(9.9) < plan.x_at(10.1) - 300     # a jump, not a slow glide across the cut
    assert plan.facts["faces_cut_off_share"] == 0.0


def test_plan_without_faces_keeps_the_whole_frame():
    n = 30
    tl = va.VisualTimeline(np.arange(n, dtype=float), np.full(n, 200.0), np.full(n, 120.0), [[] for _ in range(n)],
                           np.zeros((n, 48)), np.zeros(n, dtype=bool), [], 320, 180, {})
    assert cr.plan_crop(tl, [(0.0, 29.0)], 1920, 1080, None, refine=False).mode == "fit"


def test_plan_without_analysis_falls_back_to_centre():
    assert cr.plan_crop(None, [(0.0, 20.0)], 1920, 1080, None).mode == "center"


# ------------------------------------------------------------------ QA caption sync
def fake_features(speech_ranges, dur):
    import audio_features as af
    hop = af._HOP_SECONDS
    t = np.arange(int(dur / hop)) * hop
    sp = np.zeros(len(t), dtype=bool)
    for a, b in speech_ranges:
        sp[(t >= a) & (t < b)] = True
    return af.FrameFeatures(times=t, rms_db=np.where(sp, -20.0, -60.0), f0_hz=np.full(len(t), np.nan), speech=sp)


def ass_for(ranges):
    lines = [f"Dialogue: 0,{cp.ts(a)},{cp.ts(b)},Caption,,0,0,0,,WORD" for a, b in ranges]
    return "\n".join(lines)


def test_caption_sync_passes_when_aligned_and_fails_when_late():
    speech = [(0.0, 3.0), (3.5, 8.0)]
    ff = fake_features(speech, 8.5)
    assert clip_qa.check_caption_sync(ass_for(speech), ff, 8.5).ok
    late = [(a + 0.6, b + 0.6) for a, b in speech]
    rep = clip_qa.check_caption_sync(ass_for(late), ff, 8.5)
    assert not rep.ok and rep.detail["offset_s"] != 0


def test_qa_report_fails_only_on_fail_severity():
    r = clip_qa.QAReport([clip_qa.Check("a", True), clip_qa.Check("b", False, "warn"), clip_qa.Check("c", False)])
    assert not r.passed and [c.name for c in r.failures] == ["c"] and [c.name for c in r.warnings] == ["b"]


def test_measured_caption_offset_corrects_when_applied_as_a_shift():
    speech = [(0.0, 4.0), (4.5, 9.0)]
    ff = fake_features(speech, 9.5)
    words, t = [], 0.0
    for a, b in speech:                       # words 0.3 s EARLY relative to the audio
        t = a - 0.3
        while t + 0.4 < b - 0.3 + 0.01:
            words.append(W("word", max(0.0, t), max(0.0, t) + 0.35))
            t += 0.4
    ass, _ = cp.build_ass(words, [(0.0, 9.5)])
    first = clip_qa.check_caption_sync(ass, ff, 9.5)
    off = first.detail["offset_s"]
    assert abs(off) > 0.12
    fixed, _ = cp.build_ass(words, [(0.0, 9.5)], shift=off)
    after = clip_qa.check_caption_sync(fixed, ff, 9.5)
    assert abs(after.detail["offset_s"]) < abs(off) and abs(after.detail["offset_s"]) <= 0.1
