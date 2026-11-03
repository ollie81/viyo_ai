"""
Deterministic tests for clip_selection. The LLM is faked here, so these
check the code's own guarantees — boundaries on real sentences, honest
counts, no overlap, no invented hook text, hard quality gates, no
fabricated fallback — not the quality of any model's judgement (see
eval/ and the live acceptance run for that).
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


GOOD = {"hook": 8, "curiosity": 7, "emotional": 7, "completeness": 8, "standalone": 8, "information": 7}


def raw(s0, s1, **kw):
    base = {"start_sentence": s0, "end_sentence": s1, "category": "story",
            "scores": dict(GOOD), "opens_mid_conversation": False, "title": "Quit your job",
            "reason": "Real stakes.", "overlay_hook": "", "caption": "c",
            "hashtags": ["#Career", "career", "money!"], "topic": "quitting"}
    base.update(kw)
    return base


@pytest.fixture
def story():
    words, segments = make_transcript(STORY)
    punct = cs.attach_punctuation(words, segments)
    return punct, cs.build_sentences(punct)


# ---------------------------------------------------------------- sentences

def test_punctuation_is_recovered_from_segments_and_sentences_split_on_it():
    words, segments = make_transcript(STORY)
    punct = cs.attach_punctuation(words, segments)
    sentences = cs.build_sentences(punct)
    assert len(sentences) == len(STORY)
    assert sentences[0].text.endswith("part.")
    assert all(s.terminal for s in sentences)
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
    c, why = cs.evaluate_candidate(raw(1, 7), sentences, words, 200.0)
    assert why is None
    first = words[sentences[1].first_word: sentences[1].first_word + 3]
    assert [w["text"].lower().strip(",.") for w in first[:2]] == ["um", "so"]
    assert c.start == pytest.approx(first[2]["start"] - 0.05, abs=1e-6)
    assert c.transcript.lower().startswith("the hard part")


def test_opening_line_is_the_clips_real_first_words(story):
    words, sentences = story
    c, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    assert c.hook_line.lower().startswith("nobody tells you that quitting")


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
    c, why = cs.evaluate_candidate(raw(0, 29), sentences, punct, 999.0)
    assert why is None and c.length_adjustment == "trimmed_to_fit"
    assert c.start_sentence == 0 and c.start == pytest.approx(punct[0]["start"], abs=0.06)
    assert cs.MIN_CLIP_SECONDS <= c.speech_duration <= cs.MAX_CLIP_SECONDS
    assert c.ends_on_sentence
    one_more = cs.speech_seconds(punct[: sentences[c.end_sentence + 1].last_word + 1])
    assert one_more > cs.MAX_CLIP_SECONDS
    assert c.breakdown["penalties"]["length_trimmed_to_fit"] == 8


def test_trimming_prefers_a_full_stop_over_a_later_pause():
    words, segments = make_transcript(filler(30))
    punct = cs.attach_punctuation(words, segments)
    sentences = cs.build_sentences(punct)
    plain, _ = cs.evaluate_candidate(raw(0, 29), sentences, punct, 999.0)
    k = plain.end_sentence
    sentences[k].terminal = False
    sentences[k].pause_after = 0.1
    moved, _ = cs.evaluate_candidate(raw(0, 29), sentences, punct, 999.0)
    assert moved.end_sentence < k and moved.ends_on_sentence


def test_slightly_short_moment_is_extended_but_never_across_a_long_silence(story):
    words, sentences = story
    c, why = cs.evaluate_candidate(raw(0, 1), sentences, words, 200.0)
    assert why is None and c.length_adjustment == "extended_to_fit"
    assert c.speech_duration >= cs.MIN_CLIP_SECONDS
    assert c.breakdown["penalties"]["length_extended_to_fit"] == 5

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


# ---------------------------------------------------------------- scoring

def ctx_with(delivery=None, audio=None, visual=None, **kw):
    return cs.AnalysisContext(
        delivery=(lambda s, e, n: (delivery, {})) if delivery is not None else None,
        audio_quality=(lambda s, e: (audio, {"available": audio is not None})) if audio is not None else None,
        visual_quality=(lambda s, e: (visual, {"available": visual is not None})) if visual is not None else None,
        **kw)


def test_all_nine_components_are_reported_and_weights_sum_to_one(story):
    words, sentences = story
    assert sum(cs.COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    c, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0, ctx_with(delivery=60, audio=8, visual=7))
    assert set(c.components) == {"hook", "curiosity", "emotional", "completeness", "standalone",
                                 "information", "pacing", "visual", "audio"}
    assert all(v is not None for v in c.components.values())
    assert set(c.breakdown["weights"]) == set(c.components)


def test_unmeasurable_components_are_left_out_not_scored_as_good_or_bad(story):
    words, sentences = story
    none, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    assert none.components["visual"] is None and none.components["audio"] is None
    assert "visual" not in none.breakdown["weights"]
    assert sum(none.breakdown["weights"].values()) < 1.0
    up, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0, ctx_with(audio=10, visual=10))
    down, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0, ctx_with(audio=6, visual=1))
    assert up.score > none.score > down.score


def test_loudness_alone_cannot_rescue_weak_content(story):
    words, sentences = story
    weak = {k: 4 for k in cs.RUBRIC_KEYS}
    strong_quiet, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0, ctx_with(delivery=20))
    weak_loud, why = cs.evaluate_candidate(raw(0, 7, scores=weak), sentences, words, 200.0, ctx_with(delivery=100))
    assert why == "weak_core" and weak_loud is None
    middling = {"hook": 5, "curiosity": 5, "emotional": 5, "completeness": 5, "standalone": 6, "information": 5}
    m, _ = cs.evaluate_candidate(raw(0, 7, scores=middling), sentences, words, 200.0, ctx_with(delivery=100))
    result = cs.select_clips([m], 6, cs.DEFAULT_MIN_SCORE, [], 600)
    assert result.clips == [], "great delivery must not rescue ordinary content"
    assert strong_quiet.score > m.score


@pytest.mark.parametrize("scores", [
    {**GOOD, "hook": 4},
    {**GOOD, "completeness": 3},
    {**GOOD, "standalone": 4},
])
def test_a_weak_core_component_rejects_the_moment_outright(story, scores):
    words, sentences = story
    c, why = cs.evaluate_candidate(raw(0, 7, scores=scores), sentences, words, 200.0)
    assert c is None and why == "weak_core"


def test_poor_audio_rejects_the_moment(story):
    words, sentences = story
    c, why = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0, ctx_with(audio=2.0))
    assert c is None and why == "poor_audio"


def test_context_dependence_and_missing_sentence_end_are_penalised(story):
    words, sentences = story
    base, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    dependent, _ = cs.evaluate_candidate(raw(0, 7, opens_mid_conversation=True), sentences, words, 200.0)
    assert dependent.score == base.score - 12
    sentences[7].terminal = False
    sentences[7].pause_after = 0.1
    cut_off, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    assert cut_off.score < base.score
    assert cut_off.breakdown["penalties"]["no_sentence_end"] == 8


def test_ending_at_a_pause_is_penalised_less_than_ending_mid_flow(story):
    words, sentences = story
    sentences[7].terminal = False
    sentences[7].pause_after = 1.0
    at_pause, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    sentences[7].pause_after = 0.1
    mid_flow, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    assert at_pause.breakdown["penalties"] == {"soft_end": 2}
    assert at_pause.score > mid_flow.score


def test_creator_history_prior_is_bounded_and_reported(story):
    words, sentences = story
    ctx = cs.AnalysisContext(creator_prior=lambda c: (4.0, {"used": True}))
    base, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    boosted, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0, ctx)
    assert boosted.score == base.score + 4 and boosted.breakdown["creator_prior"] == {"used": True}
    assert base.breakdown["creator_prior"]["used"] is False


# ---------------------------------------------------------------- hooks and openings

def test_overlay_hook_must_be_grounded_in_what_is_said(story):
    words, sentences = story
    good, _ = cs.evaluate_candidate(
        raw(0, 7, overlay_hook="The hard part starts when the money stops"), sentences, words, 200.0)
    assert good.overlay_hook == "The hard part starts when the money stops" and good.overlay_status == "ok"
    invented, _ = cs.evaluate_candidate(
        raw(0, 7, overlay_hook="He made a million dollars overnight"), sentences, words, 200.0)
    assert invented.overlay_hook is None and invented.overlay_status == "not_grounded"
    repeat, _ = cs.evaluate_candidate(
        raw(0, 7, overlay_hook="Nobody tells you quitting is the easy part"), sentences, words, 200.0)
    assert repeat.overlay_hook is None and repeat.overlay_status == "repeats_opening"
    none, _ = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0)
    assert none.overlay_hook is None and none.overlay_status == "none_proposed"


QA = [
    "So what is the one mistake you see every new founder make?",
    "Absolutely, and I think the biggest mistake is hiring too fast before you have any customers.",
    "Every new hire adds cost long before they add revenue, and most startups never recover from that.",
    "The fix is boring: stay small, talk to customers every single day, and only hire when the pain is obvious.",
]


def test_a_clip_that_opens_as_an_answer_gets_the_question_prepended():
    words, segments = make_transcript(QA, words_per_second=2.2)
    punct = cs.attach_punctuation(words, segments)
    sentences = cs.build_sentences(punct)
    ctx = cs.AnalysisContext(speaker_labels=["A", "B", "B", "B"])
    c, why = cs.evaluate_candidate(raw(1, 3), sentences, punct, 100.0, ctx)
    assert why is None and c.prepended_question and c.start_sentence == 0
    assert c.transcript.startswith("So what is the one mistake")
    assert c.speakers == ["A", "B"] and not c.opens_as_reply


def test_the_same_person_asking_and_answering_is_not_treated_as_a_question_to_prepend():
    words, segments = make_transcript(QA, words_per_second=2.2)
    punct = cs.attach_punctuation(words, segments)
    sentences = cs.build_sentences(punct)
    ctx = cs.AnalysisContext(speaker_labels=["A", "A", "A", "A"])
    c, _ = cs.evaluate_candidate(raw(1, 3), sentences, punct, 100.0, ctx)
    assert not c.prepended_question and c.opens_as_reply
    assert c.breakdown["penalties"]["opens_as_reply"] == 10


def test_boilerplate_ranges_from_the_video_map_are_rejected(story):
    words, sentences = story
    ctx = cs.AnalysisContext(video_map={"avoid": [{"start_sentence": 0, "end_sentence": 7, "kind": "sponsor"}]})
    c, why = cs.evaluate_candidate(raw(0, 7), sentences, words, 200.0, ctx)
    assert c is None and why == "boilerplate"


# ---------------------------------------------------------------- selection

def cand(start, end, score, text, title="t", content=80):
    return cs.Candidate(
        start_sentence=0, end_sentence=0, start=start, end=end, duration=end - start,
        speech_duration=end - start, dead_air_ratio=0, category="insight",
        rubric={}, context_dependent=False, title=title, reason="", hook_line="",
        hook_line_verified=True, caption="", hashtags=[], topic="", transcript=text,
        ends_on_sentence=True, end_pause=1.0, content_score=content, score=score)


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


TOPICS = ["pricing freelancers invoices", "morning routines energy", "camera lighting setup",
          "negotiating salary offers", "learning languages quickly", "marathon training plan",
          "startup fundraising mistakes", "cooking cheap meals", "sleep schedule science",
          "public speaking nerves"]


def test_selection_never_exceeds_target_and_keeps_a_few_spares():
    cands = [cand(i * 40, i * 40 + 30, 90 - i, TOPICS[i]) for i in range(10)]
    result = cs.select_clips(cands, target=3, min_score=55, rejected=[], total_speech=240)
    assert len(result.clips) == 3 and result.note == ""
    assert len(result.spares) == cs.SPARE_CANDIDATES
    assert min(c.score for c in result.clips) >= max(c.score for c in result.spares)


def test_spares_still_have_to_clear_every_gate():
    cands = [cand(0, 30, 90, TOPICS[0]), cand(40, 70, 50, TOPICS[1])]
    result = cs.select_clips(cands, target=1, min_score=60, rejected=[], total_speech=100)
    assert result.spares == []


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


def test_the_prompt_carries_the_whole_video_context_and_speaker_labels():
    words, segments = make_transcript(STORY + filler(40))
    n = len(cs.build_sentences(cs.attach_punctuation(words, segments)))
    seen = []

    def respond(prompt):
        seen.append(prompt)
        return json.dumps({"candidates": []})

    ctx = cs.AnalysisContext(
        speaker_labels=["A" if i % 2 else "B" for i in range(n)],
        video_map={"summary": "A founder tells how quitting nearly broke him.", "format": "talk",
                   "speakers": [{"label": "A", "role": "host", "description": "asks questions"}],
                   "topics": [{"start_sentence": 0, "end_sentence": 7, "kind": "story", "title": "Quitting"}]})
    cs.run_selection(words, segments, 400.0, fake_llm([respond]), ctx)
    p = seen[0]
    assert "VIDEO CONTEXT" in p and "nearly broke him" in p and "Speaker A: host" in p
    assert re.search(r"^S0 [AB] \[", p, re.M), "sentences carry speaker letters"
    assert "overlay_hook" in p and "9-10 = you would share this" in p


def test_stale_speaker_labels_for_a_different_sentence_split_are_ignored():
    words, segments = make_transcript(STORY + filler(40))
    seen = []

    def respond(prompt):
        seen.append(prompt)
        return json.dumps({"candidates": []})

    cs.run_selection(words, segments, 400.0, fake_llm([respond]), cs.AnalysisContext(speaker_labels=["A", "B"]))
    assert not re.search(r"^S0 [AB] \[", seen[0], re.M)


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
    llm = fake_llm([lambda p: json.dumps({"candidates": [raw(ids_in(p)[0], ids_in(p)[0] + 5, scores=weak)]})])
    result = cs.run_selection(words, segments, 300.0, llm)
    assert result.clips == []
    assert "No moment" in result.note


def test_too_little_speech_never_calls_the_llm():
    words, segments = make_transcript(["Hello there."])
    llm = fake_llm(["{}"])
    result = cs.run_selection(words, segments, 30.0, llm)
    assert result.clips == [] and llm.calls == []


def test_list_marker_opening_is_dropped_but_real_first_words_are_kept():
    def w(texts, t0=0.0):
        out, t = [], t0
        for x in texts:
            out.append({"word": x.strip(",."), "text": x, "start": t, "end": t + 0.3})
            t += 0.35
        return out
    a = w(["Second,", "make", "sure", "your", "breath", "is", "fresh."])
    assert cs._trim_leading_fillers(a, 0, len(a) - 1) == 1
    b = w(["First", "impressions", "matter", "more", "than", "you", "think."])
    assert cs._trim_leading_fillers(b, 0, len(b) - 1) == 0
    c = w(["So,", "next,", "you", "want", "to", "relax."])
    assert cs._trim_leading_fillers(c, 0, len(c) - 1) == 2


def test_preferred_length_breaks_near_ties_but_never_beats_a_clearly_better_clip():
    long_good = cand(0, 55, 72, "a long story with a real payoff at the very end", "long")
    short_close = cand(100, 125, 70, "a short sharp insight about pricing work", "short")
    pick = cs.select_clips([long_good, short_close], target=1, min_score=55, rejected=[],
                           total_speech=600, length_pref="short")
    assert pick.clips[0].title == "short", "about as good and the length asked for"
    long_better = cand(0, 55, 85, "a long story with a real payoff at the very end", "long")
    pick = cs.select_clips([long_better, short_close], target=1, min_score=55, rejected=[],
                           total_speech=600, length_pref="short")
    assert pick.clips[0].title == "long", "a clearly stronger moment keeps its length"
    pick = cs.select_clips([long_good, short_close], target=1, min_score=55, rejected=[],
                           total_speech=600)
    assert pick.clips[0].title == "long", "no preference: best score wins"


def test_length_preference_is_a_target_in_the_prompt_not_a_cut_off():
    rule = cs._length_rule("short")
    assert "15-30 seconds" in rule and "Never end before the payoff" in rule
    assert cs._length_rule(None) == cs._length_rule("auto") == "- Target 15-55 seconds of speech.\n"


def test_short_choice_keeps_long_clips_only_for_standout_moments():
    long_ok = cand(0, 52, 75, "a good but long explanation of pricing work", "long_ok")
    long_star = cand(100, 155, 85, "an exceptional story with a twist at the end", "star")
    tight = cand(200, 222, 66, "a tight point about saying no to clients", "tight")
    rejected = []
    pick = cs.select_clips([long_ok, long_star, tight], target=3, min_score=55, rejected=rejected,
                           total_speech=600, length_pref="short")
    titles = {c.title for c in pick.clips}
    assert titles == {"star", "tight"}, "52 s at 75 is too long for Short; 55 s at 85 is a standout"
    assert {"title": "long_ok", "score": 75, "reason": "longer_than_chosen_length"} in rejected
    medium = cs.select_clips([long_ok, tight], target=2, min_score=55, rejected=[], total_speech=600,
                             length_pref="medium")
    assert {c.title for c in medium.clips} == {"long_ok", "tight"}, "52 s fits Medium's 55 s limit"
    auto = cs.select_clips([long_ok, tight], target=2, min_score=55, rejected=[], total_speech=600)
    assert len(auto.clips) == 2


def test_short_choice_with_only_long_moments_says_so():
    only_long = cand(0, 50, 75, "a good but long explanation of pricing work", "long")
    pick = cs.select_clips([only_long], target=1, min_score=55, rejected=[], total_speech=300, length_pref="short")
    assert not pick.clips and "Try Medium, Long or Auto" in pick.note


def test_short_prompt_allows_15_second_clips_and_names_the_limit():
    rule = cs._length_rule("short")
    assert "as short as 15 seconds" in rule and "Stay under 40 seconds" in rule
    assert "Stay under 55 seconds" in cs._length_rule("medium")
    assert "Stay under" not in cs._length_rule("long")
