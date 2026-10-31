"""
Deterministic tests for clip_selection. The LLM is faked here, so these
check the code's own guarantees — boundaries on real sentences, honest
counts, no overlap, no invented hook text, no fabricated fallback — not
the quality of any model's judgement (see eval/ for that).
"""
import json
import re

import pytest

import clip_selection as cs


def make_transcript(sentences, words_per_second=2.5, pause=0.4, start=0.0):
    """Builds whisper-1-shaped words (no punctuation) and segments
    (punctuated text) from a list of sentence strings."""
    words, segments = [], []
    t = start
    for sentence in sentences:
        seg_start = t
        for token in sentence.split():
            dur = 1 / words_per_second
            words.append({"word": re.sub(r"[^\w']", "", token), "start": round(t, 3),
                          "end": round(t + dur * 0.85, 3)})
            t += dur
        segments.append({"start": round(seg_start, 3), "end": round(t, 3), "text": " " + sentence})
        t += pause
    return words, segments


STORY = [
    "Nobody tells you that quitting your job is the easy part.",
    "Um so the hard part starts three weeks later when the money stops.",
    "I had eleven hundred dollars left and a landlord who did not care about my dreams.",
    "So I did the one thing every business book tells you not to do.",
    "I called my old boss and asked for freelance work at double my old rate.",
    "He laughed, and then he said yes, because nobody else knew the system.",
    "That one phone call paid my rent for six months.",
    "The lesson is simple: your leverage is whatever only you can do.",
]


def filler(n):
    return [f"This is filler sentence number {i} about the weather and other things." for i in range(n)]


# ---------------------------------------------------------------- sentences

def test_punctuation_is_recovered_from_segments_and_sentences_split_on_it():
    words, segments = make_transcript(STORY)
    punct = cs.attach_punctuation(words, segments)
    sentences = cs.build_sentences(punct)
    assert len(sentences) == len(STORY)
    assert sentences[0].text.endswith("part.")
    assert all(s.terminal for s in sentences)
    # Sentence times are the real word times, not estimates.
    assert sentences[1].start == words[sentences[1].first_word]["start"]
    assert sentences[-1].end == words[-1]["end"]


def test_long_pause_breaks_a_sentence_without_marking_it_terminal():
    words, segments = make_transcript(["and then we waited", "for a very long time"], pause=2.0)
    segments = [{"start": 0, "end": 10, "text": "and then we waited for a very long time"}]
    sentences = cs.build_sentences(cs.attach_punctuation(words, segments))
    assert len(sentences) == 2
    assert sentences[0].terminal is False


def test_run_on_speech_is_split_at_pauses_and_clause_commas():
    run_on = ("and then I went to the station and I worked there for a while and it was great "
              "and we did all kinds of shows")
    words, segments = make_transcript([run_on, run_on + ",", run_on], pause=0.8)
    sentences = cs.build_sentences(cs.attach_punctuation(words, segments))
    assert len(sentences) == 3
    assert all(not s.terminal for s in sentences)
    assert sentences[0].pause_after == pytest.approx(0.8 + 0.4 * 0.15, abs=0.05)
    long_run = " ".join(["word"] * 90)
    words, segments = make_transcript([long_run])
    assert all(len(s.text.split()) <= cs.MAX_SENTENCE_WORDS
               for s in cs.build_sentences(cs.attach_punctuation(words, segments)))


def test_ending_at_a_pause_is_penalised_less_than_ending_mid_flow(story):
    words, sentences = story
    sentences[7].terminal = False
    sentences[7].pause_after = 1.0
    at_pause, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    sentences[7].pause_after = 0.1
    mid_flow, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    assert at_pause.breakdown["structure_penalties"] == {"soft_sentence_end": 10}
    assert at_pause.score > mid_flow.score


def test_abbreviations_do_not_end_sentences():
    words, segments = make_transcript(["I met Dr. Smith at the clinic yesterday."])
    sentences = cs.build_sentences(cs.attach_punctuation(words, segments))
    assert len(sentences) == 1


def test_target_count_scales_with_speech_and_caps_at_six():
    assert cs.target_clip_count(60) == 1
    assert cs.target_clip_count(4 * 60) == 3
    assert cs.target_clip_count(30 * 60) == 6


def test_windows_cover_every_sentence_and_overlap():
    words, segments = make_transcript(filler(400))
    sentences = cs.build_sentences(cs.attach_punctuation(words, segments))
    windows = cs._windows(sentences)
    assert len(windows) > 1
    covered = set()
    for lo, hi in windows:
        covered.update(range(lo, hi + 1))
        assert sentences[hi].end - sentences[lo].start <= cs.WINDOW_SECONDS + 10
    assert covered == set(range(len(sentences)))
    assert windows[1][0] <= windows[0][1], "consecutive windows overlap"


# ---------------------------------------------------------------- evaluation

def raw(s0, s1, **kw):
    base = {"start_sentence": s0, "end_sentence": s1, "category": "story",
            "scores": {"hook": 8, "completeness": 8, "standalone": 8, "payoff": 8, "emotional_pull": 7},
            "context_dependent": False, "title": "Quit your job", "reason": "Real stakes.",
            "hook_line": "Nobody tells you that quitting your job", "caption": "c",
            "hashtags": ["#Career", "career", "money!"], "topic": "quitting"}
    base.update(kw)
    return base


@pytest.fixture
def story():
    words, segments = make_transcript(STORY)
    punct = cs.attach_punctuation(words, segments)
    return punct, cs.build_sentences(punct)


def test_candidate_boundaries_are_real_word_times(story):
    words, sentences = story
    c, why = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    assert why is None
    assert c.start == pytest.approx(words[0]["start"], abs=0.06)
    assert c.end <= words[-1]["end"] + cs.END_TAIL_SECONDS + 1e-6
    assert c.ends_on_sentence
    assert c.hashtags == ["career", "money"]


def test_leading_filler_is_trimmed_from_the_clip_start(story):
    words, sentences = story
    c, why = cs.evaluate_candidate(raw(1, 7, hook_line="the hard part starts"), sentences, words, 200.0)
    assert why is None
    first = words[sentences[1].first_word: sentences[1].first_word + 3]
    assert [w["text"].lower().strip(",.") for w in first[:2]] == ["um", "so"]
    assert c.start == pytest.approx(first[2]["start"] - 0.05, abs=1e-6)
    assert c.transcript.lower().startswith("the hard part")


def test_hook_line_that_was_not_said_is_replaced_with_real_words(story):
    words, sentences = story
    c, _ = cs.evaluate_candidate(raw(0, 7, hook_line="You won't believe this secret"), sentences, words, 200.0)
    assert c.hook_line_verified is False
    assert c.hook_line.lower().startswith("nobody tells you")


@pytest.mark.parametrize("ids,reason", [((9, 12), "sentence_ids_out_of_range"),
                                        (("x", 2), "invalid_sentence_ids")])
def test_invalid_candidates_are_rejected_not_repaired(story, ids, reason):
    words, sentences = story
    c, why = cs.evaluate_candidate(raw(*ids), sentences, words, 200.0)
    assert c is None and why == reason


def test_too_long_moment_keeps_its_hook_and_ends_at_the_latest_full_stop_that_fits():
    words, segments = make_transcript(filler(30))
    punct = cs.attach_punctuation(words, segments)
    sentences = cs.build_sentences(punct)
    c, why = cs.evaluate_candidate(raw(0, 29, hook_line=""), sentences, punct, 999.0)
    assert why is None and c.length_adjustment == "trimmed_to_fit"
    assert c.start_sentence == 0 and c.start == pytest.approx(punct[0]["start"], abs=0.06)
    assert cs.MIN_CLIP_SECONDS <= c.speech_duration <= cs.MAX_CLIP_SECONDS
    assert c.ends_on_sentence
    # It is the LATEST full stop that fits: one more sentence would not.
    one_more = cs.speech_seconds(punct[: sentences[c.end_sentence + 1].last_word + 1])
    assert one_more > cs.MAX_CLIP_SECONDS
    assert c.breakdown["length_penalty"] == cs.LENGTH_ADJUSTMENT_PENALTY["trimmed_to_fit"]


def test_trimming_prefers_a_full_stop_over_a_later_pause():
    words, segments = make_transcript(filler(30))
    punct = cs.attach_punctuation(words, segments)
    sentences = cs.build_sentences(punct)
    plain, _ = cs.evaluate_candidate(raw(0, 29, hook_line=""), sentences, punct, 999.0)
    k = plain.end_sentence
    # Make the chosen ending non-terminal with no pause: the next-best
    # full stop earlier wins over it.
    sentences[k].terminal = False
    sentences[k].pause_after = 0.1
    moved, _ = cs.evaluate_candidate(raw(0, 29, hook_line=""), sentences, punct, 999.0)
    assert moved.end_sentence < k and moved.ends_on_sentence


def test_slightly_short_moment_is_extended_but_never_across_a_long_silence(story):
    words, sentences = story
    c, why = cs.evaluate_candidate(raw(0, 1), sentences, words, 200.0)
    assert why is None and c.length_adjustment == "extended_to_fit"
    assert c.speech_duration >= cs.MIN_CLIP_SECONDS
    assert c.breakdown["length_penalty"] == cs.LENGTH_ADJUSTMENT_PENALTY["extended_to_fit"]

    w2, seg2 = make_transcript(["A short line here.", "Another short one."], pause=3.0)
    p2 = cs.attach_punctuation(w2, seg2)
    s2 = cs.build_sentences(p2)
    c2, why2 = cs.evaluate_candidate(raw(0, 0), s2, p2, 60.0)
    assert c2 is None and why2 == "too_short"


def test_when_nothing_fits_the_note_says_so_instead_of_blaming_quality():
    result = cs.select_clips([], target=5, min_score=60,
                             rejected=[{"title": "a", "score": None, "reason": "too_short"}],
                             total_speech=400)
    assert "usable length" in result.note and "strong enough" not in result.note


def test_score_rewards_content_far_more_than_delivery(story):
    words, sentences = story
    strong_quiet, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0,
                                            delivery=lambda s, e, n: (20, {}))
    weak_loud, _ = cs.evaluate_candidate(
        raw(0, 7, scores={k: 4 for k in cs.RUBRIC_KEYS}), sentences, words, 200.0,
        delivery=lambda s, e, n: (100, {}))
    assert strong_quiet.score > weak_loud.score
    rejected = []
    result = cs.select_clips([weak_loud], 6, cs.DEFAULT_MIN_SCORE, rejected, 600)
    assert result.clips == [], "great delivery must not rescue weak content"
    assert rejected[0]["reason"] in {"below_quality_floor", "weak_content"}


def test_context_dependence_and_missing_sentence_end_are_penalised(story):
    words, sentences = story
    base, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    dependent, _ = cs.evaluate_candidate(raw(0, 7, context_dependent=True), sentences, words, 200.0)
    assert dependent.score == base.score - 15
    sentences[7].terminal = False
    sentences[7].pause_after = 0.1
    cut_off, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    assert cut_off.score < base.score
    assert cut_off.breakdown["structure_penalties"]["no_sentence_end"] == 35


# ---------------------------------------------------------------- selection

def cand(start, end, score, text, title="t", content=80):
    return cs.Candidate(content_score=content,
        start_sentence=0, end_sentence=0, start=start, end=end, duration=end - start,
        speech_duration=end - start, dead_air_ratio=0, category="insight", rubric={},
        context_dependent=False, title=title, reason="", hook_line="", hook_line_verified=True,
        caption="", hashtags=[], topic="", transcript=text, ends_on_sentence=True, end_pause=1.0,
        score=score)


def test_selection_is_best_first_without_overlap_or_duplicates():
    cands = [
        cand(0, 30, 90, "pricing strategy for freelancers explained clearly", "a"),
        cand(20, 50, 85, "completely different overlapping story about dogs", "b"),
        cand(60, 90, 80, "pricing strategy for freelancers explained clearly again", "c"),
        cand(100, 130, 70, "how morning routines changed my energy levels", "d"),
        cand(140, 170, 40, "weak moment about nothing much happening here", "e"),
    ]
    rejected = []
    result = cs.select_clips(cands, target=6, min_score=55, rejected=rejected, total_speech=600)
    assert [c.title for c in result.clips] == ["a", "d"]
    reasons = {r["title"]: r["reason"] for r in rejected}
    assert reasons == {"b": "overlaps_better_clip", "c": "duplicate_topic", "e": "below_quality_floor"}
    assert "Found 2 of up to 6" in result.note


def test_selection_never_exceeds_target():
    topics = ["pricing freelancers invoices", "morning routines energy", "camera lighting setup",
              "negotiating salary offers", "learning languages quickly", "marathon training plan",
              "startup fundraising mistakes", "cooking cheap meals", "sleep schedule science",
              "public speaking nerves"]
    cands = [cand(i * 40, i * 40 + 30, 90 - i, topics[i]) for i in range(10)]
    result = cs.select_clips(cands, target=3, min_score=55, rejected=[], total_speech=240)
    assert len(result.clips) == 3 and result.note == ""


# ---------------------------------------------------------------- end to end with a fake LLM

def fake_llm(responses):
    calls = []

    def llm(messages):
        calls.append(messages)
        r = responses[min(len(calls) - 1, len(responses) - 1)]
        return r(messages[-1]["content"]) if callable(r) else r
    llm.calls = calls
    return llm


def ids_in(prompt):
    return [int(x) for x in re.findall(r"^S(\d+) ", prompt, re.M)]


def test_run_selection_returns_strong_clips_from_real_sentence_ranges():
    words, segments = make_transcript(STORY + filler(40) + STORY)

    def respond(prompt):
        ids = ids_in(prompt)
        return json.dumps({"candidates": [raw(ids[0], ids[0] + 7)]})

    result = cs.run_selection(words, segments, 400.0, fake_llm([respond]))
    assert len(result.clips) == 1
    c = result.clips[0]
    assert c.score >= cs.DEFAULT_MIN_SCORE
    assert c.start == pytest.approx(words[0]["start"], abs=0.06)


def test_garbage_llm_output_is_retried_once_then_raises():
    words, segments = make_transcript(STORY + filler(20))
    llm = fake_llm(["not json at all"])
    with pytest.raises(cs.SelectionUnavailable):
        cs.run_selection(words, segments, 300.0, llm)
    assert len(llm.calls) == 2


def test_recovers_when_the_retry_is_valid():
    words, segments = make_transcript(STORY + filler(20))
    llm = fake_llm(["oops", lambda p: json.dumps({"candidates": [raw(ids_in(p)[0], ids_in(p)[0] + 7)]})])
    result = cs.run_selection(words, segments, 300.0, llm)
    assert len(result.clips) == 1


def test_no_good_moments_gives_zero_clips_and_an_honest_note():
    words, segments = make_transcript(filler(60))
    weak = {k: 3 for k in cs.RUBRIC_KEYS}
    llm = fake_llm([lambda p: json.dumps({"candidates": [
        raw(ids_in(p)[0], ids_in(p)[0] + 5, scores=weak, hook_line="")]})])
    result = cs.run_selection(words, segments, 300.0, llm)
    assert result.clips == []
    assert "No moment" in result.note


def test_too_little_speech_never_calls_the_llm():
    words, segments = make_transcript(["Hello there."])
    llm = fake_llm(["{}"])
    result = cs.run_selection(words, segments, 30.0, llm)
    assert result.clips == [] and llm.calls == []
