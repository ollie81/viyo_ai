"""No-talking mode: window building, measurement, review and scoring."""
import json

import numpy as np
import pytest

import clip_selection as cs
import video_analysis as va
import visual_moments as vm


def timeline(n=120, cut_every=15, dark=(), busy=()):
    times = np.arange(n, dtype=float)
    rng = np.random.default_rng(0)
    hist = np.zeros((n, 48))
    base = rng.random(48)
    cut = np.zeros(n, dtype=bool)
    for k in range(n):
        if k and k % cut_every == 0:
            cut[k] = True
            base = rng.random(48)
        jitter = 0.4 if any(a <= k < b for a, b in busy) else 0.01
        hist[k] = np.abs(base + rng.normal(0, jitter, 48))
        hist[k] /= hist[k].sum()
    bright = np.array([5.0 if any(a <= k < b for a, b in dark) else 120.0 for k in range(n)])
    tl = va.VisualTimeline(times, np.full(n, 250.0), bright, [[] for _ in range(n)], hist, cut, [], 320, 180, {})
    tl.shots = va.build_shots(tl)
    return tl


def test_windows_follow_shots_and_respect_length_limits():
    tl = timeline()
    wins = vm.build_windows(tl, 120.0)
    assert wins
    shot_starts = {s.start for s in tl.shots}
    for w in wins:
        assert vm.MIN_SECONDS <= w.duration <= vm.MAX_SECONDS
        assert w.start in shot_starts or any(s.end - s.start > vm.MAX_SECONDS for s in tl.shots)


def test_one_long_shot_is_scanned_not_skipped():
    tl = timeline(n=200, cut_every=10_000)
    wins = vm.build_windows(tl, 200.0)
    assert len(wins) > 5 and max(w.end for w in wins) > 150


def test_dark_windows_are_never_shortlisted_and_busy_ones_rank_first():
    tl = timeline(n=120, dark=[(0, 45)], busy=[(60, 90)])
    wins = vm.build_windows(tl, 120.0)
    vm.measure(wins, tl)
    short = vm.shortlist(wins)
    assert short and all(w.start >= 45 - 1e-6 for w in short)
    best = max(short, key=vm._measured)
    assert 45 <= best.start < 90


def fake_llm(appeal=8, verdict="publish", title="Golden crust forming in the pan"):
    def llm(messages):
        text = messages[0]["content"][0]["text"]
        ids = [int(x) for x in __import__("re").findall(r"--- CANDIDATE (\d+)", text)]
        return json.dumps({"reviews": [{"id": i, "appeal": appeal, "complete": 7, "what": "Butter melts in a pan.",
                                        "title": title, "caption": "Butter time", "hashtags": ["cooking"],
                                        "verdict": verdict} for i in ids]})
    return llm


def test_select_returns_titled_visual_clips_with_no_transcript():
    tl = timeline(n=240, busy=[(60, 120)])
    res = vm.select(tl, 240.0, fake_llm(), lambda w: [])
    assert res.clips and len(res.clips) <= vm.target_count(240.0)
    c = res.clips[0]
    assert c["category"] == "visual" and c["transcript"] == "" and c["overlay_hook"]
    assert len(c["title"].split()) <= vm.MAX_TITLE_WORDS
    assert 0 <= c["score"] <= 100


def test_rejected_or_weak_moments_give_no_clips_never_a_fake_one():
    tl = timeline(n=240)
    assert vm.select(tl, 240.0, fake_llm(verdict="reject"), lambda w: []).clips == []
    assert vm.select(tl, 240.0, fake_llm(appeal=1), lambda w: []).clips == []


def test_review_failure_is_retryable_not_a_made_up_result():
    tl = timeline(n=240)
    with pytest.raises(cs.SelectionUnavailable):
        vm.select(tl, 240.0, lambda m: "not json", lambda w: [])


def test_long_titles_are_cut_to_the_limit():
    tl = timeline(n=240, busy=[(60, 120)])
    res = vm.select(tl, 240.0, fake_llm(title="one two three four five six seven eight nine ten"), lambda w: [])
    assert len(res.clips[0]["title"].split()) == vm.MAX_TITLE_WORDS
