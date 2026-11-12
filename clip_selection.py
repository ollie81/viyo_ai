"""
Clip selection for the AI Repurposer.

Pipeline:
  1. attach_punctuation / build_sentences — whisper-1's word list has no
     punctuation, its segment text does; aligning the two gives sentences
     with exact word-level start/end times.
  2. propose_candidates — the LLM reads numbered sentences (in windows
     for long videos, each given a whole-video summary, the speaker
     labels and topic map) and proposes moments as sentence RANGES plus a
     rubric. It never types a timestamp, so it can't invent one.
  3. evaluate_candidate — every proposal is checked against the
     transcript and scored on nine components:
        hook, curiosity, emotional engagement, story completeness,
        standalone context, information value  (LLM, harshly anchored)
        pacing, visual quality, audio quality   (MEASURED from the media)
     Emotional engagement blends the LLM's read with vocal delivery
     measured against the speaker's own baseline (loudness alone never
     counts). Components that can't be measured are left out and the
     weights renormalised — unknown is never scored as good or bad.
  4. select_clips — best-first, no time overlap, no near-duplicate
     topics, hard gates on the core components, and a count cap scaled to
     how much speech there is. Fewer than asked is a valid result;
     padding with weak clips is not. A few spares are kept so a clip that
     later fails review or rendering QC can be replaced.

Nothing here falls back to a made-up clip. If the LLM can't be reached
or keeps returning garbage, SelectionUnavailable is raised and the job
layer retries or fails (and refunds) the job.
"""
from __future__ import annotations

import difflib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import clip_edit
import clip_text

MIN_CLIP_SECONDS = 12.0
MAX_CLIP_SECONDS = 60.0
# The creator's preferred length (seconds of speech). A target only: the
# hard limits above still apply, and a moment that needs longer to land its
# hook and payoff keeps it. Between two moments of about the same quality,
# the one inside the preferred range ranks first.
LENGTH_PREFS = {"short": (15, 30), "medium": (30, 45), "long": (45, 60)}
LENGTH_FIT_BONUS = 4
# With Short or Medium chosen, a clip longer than this is only kept when it's
# one of the strongest moments; otherwise a tighter moment is used instead.
LENGTH_SOFT_CAP = {"short": 40.0, "medium": 55.0}
LENGTH_CAP_EXEMPT_SCORE = 80


def over_length_choice(c: "Candidate", length_pref: Optional[str]) -> bool:
    cap = LENGTH_SOFT_CAP.get(length_pref or "")
    return bool(cap) and c.duration > cap and c.score < LENGTH_CAP_EXEMPT_SCORE
TARGET_MAX_CLIPS = 6
MAX_REQUESTED_CLIPS = 10      # the most a creator can ask for in one job

# What creators can ask for. "best" (or nothing) keeps the normal judgement; "visual" is handled before selection
# (no transcription; moments are chosen from the picture).
FOCUS = {
    "best": "the best moments overall",
    "funny": "funny moments: jokes, banter, comic timing, absurd or unexpected reactions",
    "emotional": "emotional moments: vulnerability, conviction, inspiration, tension, heartfelt stories",
    "educational": "educational moments: clear explanations, useful advice, surprising facts, how-tos",
    "conversation": "interesting conversations: lively exchanges, disagreements, sharp questions and answers",
    "story": "storytelling: complete stories with a setup, a turn and a payoff",
    "product": "product moments: where a product, service or offer is shown, explained or recommended",
    "highlights": "the most important highlights: the key points, decisions or turning points of the video",
    "visual": "visual moments chosen from the picture",
}
MIN_REQUEST_MATCH = 4         # a moment rated below this for the creator's request is left out (never padded in)
# One clip per this much speech, at most. Keeps a 4-minute video from
# being chopped into six thin slices just because six were asked for.
SPEECH_SECONDS_PER_CLIP = 75.0
# Rubric anchors: 7-8 = good/publishable, 5-6 = ordinary. The floors sit at
# the top of "ordinary": the model's scores for the same video wander by
# about +-10 between runs, so a higher floor made a decent tutorial pass or
# fail on luck. The editor review and the render checks still gate every clip.
DEFAULT_MIN_SCORE = 56
# Floor on the LLM-judged components alone (0-100): measured components
# (pacing, visual, audio) must never lift a moment that's weak on
# substance over the overall floor.
MIN_CONTENT_SCORE = 56
# A core component this low rejects the moment outright, whatever else
# is good about it.
CORE_GATE = 5
MIN_AUDIO_QUALITY = 3.0
SPARE_CANDIDATES = 3
MAX_QUESTION_PREPEND_SECONDS = 14.0

# A word gap longer than this inside a clip is dead air the renderer
# will cut; counted here so the duration limits apply to what the
# viewer will actually watch.
DEAD_AIR_GAP_SECONDS = 0.7
# Conversational speech is mostly run-on: few full stops, clauses joined
# by "and", "so", pauses. Units are therefore also split at natural
# pauses and clause commas once they're long enough, so clip boundaries
# can land at real breaks instead of 15-second blocks.
SOFT_SENTENCE_BREAK_GAP_SECONDS = 1.5
PAUSE_BREAK_SECONDS = 0.6
PAUSE_BREAK_MIN_WORDS = 15
CLAUSE_BREAK_MIN_WORDS = 25
MAX_SENTENCE_WORDS = 40
END_TAIL_SECONDS = 0.25
MAX_LEADING_FILLER_SKIP_SECONDS = 1.5

WINDOW_SECONDS = 600.0
WINDOW_OVERLAP_SECONDS = 60.0
MAX_PARALLEL_WINDOWS = 3

_TERMINAL = re.compile(r"[.?!…]+[\"')\]]*$")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "eg", "ie", "jr", "sr", "prof"}
# Removed only at the very start of a clip, never mid-clip — these open
# a sentence without carrying it ("So I quit my job" -> "I quit my job").
_LEADING_FILLERS = {
    "um", "umm", "uh", "uhh", "erm", "er", "ah", "hmm", "mm", "mhm",
    "so", "and", "but", "okay", "ok", "yeah", "well", "anyway", "anyways", "basically",
}
_STOPWORDS = clip_text.STOPWORDS

CATEGORIES = {"story", "humor", "insight", "emotional", "hot_take", "advice", "hook", "other"}
# Judged by the LLM against explicit anchors.
RUBRIC_KEYS = ("hook", "curiosity", "emotional", "completeness", "standalone", "information")
# Measured from the media (not asked of the LLM).
MEASURED_KEYS = ("pacing", "visual", "audio")
COMPONENT_WEIGHTS = {
    "hook": 0.18, "curiosity": 0.10, "emotional": 0.10, "completeness": 0.14,
    "standalone": 0.14, "information": 0.10, "pacing": 0.08, "visual": 0.08, "audio": 0.08,
}
# Share of "emotional engagement" taken from measured vocal delivery.
DELIVERY_BLEND = 0.25

MAX_TITLE_CHARS = 80
MAX_CAPTION_CHARS = 300
MAX_HASHTAGS = 6
MAX_HOOK_LINE_CHARS = 70


class SelectionUnavailable(Exception):
    """The LLM step couldn't produce a usable answer. Retryable."""


# ---------------------------------------------------------------- transcript structure

def _key(token: str) -> str:
    return re.sub(r"[^\w]", "", token.lower())


def attach_punctuation(words: list[dict], segments: list[dict]) -> list[dict]:
    """
    Returns copies of `words` with a `text` field carrying the
    punctuation from the segment transcript ("world" -> "world.").
    Unmatched words keep their bare text.
    """
    out = [dict(w, text=str(w.get("word", "")).strip()) for w in words]
    seg_tokens: list[str] = []
    for seg in segments:
        seg_tokens.extend(str(seg.get("text", "")).split())
    if not out or not seg_tokens:
        return out

    word_keys = [_key(w["text"]) for w in out]
    seg_keys = [_key(t) for t in seg_tokens]
    matcher = difflib.SequenceMatcher(None, word_keys, seg_keys, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            wi, ti = block.a + k, block.b + k
            if word_keys[wi]:
                out[wi]["text"] = seg_tokens[ti]
    return out


@dataclass
class Sentence:
    id: int
    start: float
    end: float
    first_word: int   # index into the words list
    last_word: int    # inclusive
    text: str
    terminal: bool    # ends on . ? ! rather than a pause/length split
    pause_after: float = 0.0  # silence before the next word (inf at the end)


def _ends_sentence(token: str) -> bool:
    if not _TERMINAL.search(token):
        return False
    return _key(token) not in _ABBREVIATIONS


def build_sentences(words: list[dict]) -> list[Sentence]:
    """`words` must carry `text` (see attach_punctuation), start, end."""
    sentences: list[Sentence] = []
    start_idx = 0

    def close(end_idx: int, terminal: bool, gap: float):
        nonlocal start_idx
        chunk = words[start_idx: end_idx + 1]
        if chunk:
            sentences.append(Sentence(
                id=len(sentences),
                start=float(chunk[0]["start"]),
                end=float(chunk[-1]["end"]),
                first_word=start_idx,
                last_word=end_idx,
                text=" ".join(w["text"] for w in chunk).strip(),
                terminal=terminal,
                pause_after=gap,
            ))
        start_idx = end_idx + 1

    for i, w in enumerate(words):
        n = i - start_idx + 1
        token = w["text"]
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = float(nxt["start"]) - float(w["end"]) if nxt else math.inf
        if _ends_sentence(token):
            close(i, True, gap)
        elif nxt is None or gap > SOFT_SENTENCE_BREAK_GAP_SECONDS:
            close(i, False, gap)
        elif n >= PAUSE_BREAK_MIN_WORDS and gap >= PAUSE_BREAK_SECONDS:
            close(i, False, gap)
        elif n >= CLAUSE_BREAK_MIN_WORDS and re.search(r"[,;:—–]$", token):
            close(i, False, gap)
        elif n >= MAX_SENTENCE_WORDS:
            close(i, False, gap)
    return sentences


def speech_seconds(words: list[dict], gap: float = DEAD_AIR_GAP_SECONDS) -> float:
    """Total time covered by speech, with gaps longer than `gap` excluded."""
    total = 0.0
    block_start = block_end = None
    for w in words:
        s, e = float(w["start"]), float(w["end"])
        if block_start is None:
            block_start, block_end = s, e
        elif s - block_end > gap:
            total += block_end - block_start
            block_start, block_end = s, e
        else:
            block_end = max(block_end, e)
    if block_start is not None:
        total += block_end - block_start
    return total


def target_clip_count(total_speech_seconds: float) -> int:
    return max(1, min(TARGET_MAX_CLIPS, int(total_speech_seconds // SPEECH_SECONDS_PER_CLIP)))


# ---------------------------------------------------------------- LLM proposal

def _fmt_time(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def _windows(sentences: list[Sentence]) -> list[tuple[int, int]]:
    """Inclusive sentence-index ranges, ~WINDOW_SECONDS long, overlapping."""
    if not sentences:
        return []
    windows = []
    i = 0
    while i < len(sentences):
        start_t = sentences[i].start
        j = i
        while j + 1 < len(sentences) and sentences[j + 1].end - start_t <= WINDOW_SECONDS:
            j += 1
        windows.append((i, j))
        if j >= len(sentences) - 1:
            break
        overlap_start = sentences[j].end - WINDOW_OVERLAP_SECONDS
        nxt = j + 1
        while nxt - 1 > i and sentences[nxt - 1].start >= overlap_start:
            nxt -= 1
        i = max(nxt, i + 1)
    return windows


SYSTEM_PROMPT = (
    "You are a senior short-form video editor and story editor. You pick moments from long-form "
    "videos (podcasts, talks, interviews, vlogs, tutorials) that work as standalone vertical shorts "
    "on TikTok, YouTube Shorts and Instagram Reels. You are strict and you calibrate: most moments "
    "in most videos are ordinary, and you say so with ordinary scores. Reserve 8+ for moments you "
    "would genuinely stop scrolling for. You never invent content that is not in the transcript, "
    "and you never put words in a speaker's mouth."
)


def _speaker_prefix(label: Optional[str]) -> str:
    return f"{label} " if label else ""


def _length_rule(length_pref: Optional[str]) -> str:
    band = LENGTH_PREFS.get(length_pref or "")
    if not band:
        return "- Target 15-55 seconds of speech.\n"
    lo, hi = band
    cap = LENGTH_SOFT_CAP.get(length_pref or "")
    if cap:
        return (f"- The creator chose {length_pref.upper()} clips: about {lo}-{hi} seconds of speech. A clip "
                "as short as 15 seconds is fine when the point lands in that time. Stay under "
                f"{int(cap)} seconds: only an exceptional moment may run longer. When a good moment runs "
                "long, find the tightest stretch of it that still opens on its hook and ends on its "
                "payoff, and propose that instead. Never end before the payoff.\n")
    return (f"- The creator prefers clips of about {lo}-{hi} seconds of speech. Aim for that, but a "
            "strong hook and a complete payoff matter more: if a moment needs longer to land, keep it "
            f"(up to {int(MAX_CLIP_SECONDS)} seconds), and never cut a moment before its payoff just to "
            "make it shorter.\n")


def request_text(request: Optional[dict]) -> str:
    """The creator's request as one plain sentence, or "" for the normal judgement."""
    if not request:
        return ""
    focus = request.get("focus") if request.get("focus") in FOCUS and request.get("focus") not in ("best", "visual") else None
    words = (request.get("text") or "").strip()[:400]
    if not focus and not words:
        return ""
    return "THE CREATOR WANTS: " + "; ".join(x for x in (FOCUS[focus] if focus else "", f'"{words}"' if words else "") if x) + "."


def _proposal_prompt(sentences: list[Sentence], lo: int, hi: int, video_seconds: float,
                     max_candidates: int, speaker_labels: Optional[list] = None,
                     video_map: Optional[dict] = None, length_pref: Optional[str] = None,
                     request: Optional[dict] = None) -> str:
    lines = [
        f"S{s.id} {_speaker_prefix(speaker_labels[s.id] if speaker_labels else None)}"
        f"[{_fmt_time(s.start)}-{_fmt_time(s.end)}] {s.text}"
        for s in sentences[lo: hi + 1]
    ]
    context = ""
    if video_map:
        parts = []
        if video_map.get("summary"):
            parts.append(f"Whole-video summary: {video_map['summary']}")
        if video_map.get("format"):
            parts.append(f"Format: {video_map['format']}")
        for sp in video_map.get("speakers") or []:
            parts.append(f"Speaker {sp.get('label')}: {sp.get('role', 'unknown')} - {sp.get('description', '')}")
        topics = video_map.get("topics") or []
        if topics:
            parts.append("Topic outline (sentence ranges): " + "; ".join(
                f"S{t['start_sentence']}-S{t['end_sentence']} {t.get('kind', '')}: {t.get('title', '')}"
                for t in topics[:30]))
        if parts:
            context = "VIDEO CONTEXT (from analysing the whole video):\n" + "\n".join(parts) + "\n\n"
    labelled = ("Letters before a time range (A, B...) are speakers identified from their voices; "
                "a missing letter means unknown.\n" if speaker_labels and any(speaker_labels) else "")
    want = request_text(request)
    if want:
        context += (want + " Prefer moments that match this request. Rate each moment's match in "
                    "\"matches_request\" (0-10: 0 = unrelated, 5 = partly, 10 = exactly what was asked). Do not "
                    "include a moment just to reach a count: fewer, matching moments are better.\n\n")
    return (
        f"The video is {video_seconds / 60:.1f} minutes long. {context}"
        f"Below is the transcript for {_fmt_time(sentences[lo].start)}-{_fmt_time(sentences[hi].end)} "
        f"as numbered sentences with their real start/end times. {labelled}\n"
        f"List the {max_candidates} best candidate moments in this part, best first. Include "
        "weaker ones too and score them honestly: low scores are expected and get filtered out "
        "later, so never inflate a score to justify including a moment. Return fewer only if "
        "this part doesn't contain that many distinct, complete moments.\n\n"
        "WHAT MAKES A GOOD MOMENT\n"
        "Surprising or counter-intuitive statements, funny exchanges, emotional or vulnerable "
        "stories, controversial opinions with a reason, genuinely useful advice, an interesting "
        "discovery, an unexpected reaction, a vivid story with tension, a clear explanation of "
        "something people wonder about.\n\n"
        "RULES\n"
        "- A moment is a contiguous range of sentences: start_sentence..end_sentence "
        "(inclusive), using the S numbers below. Never use numbers outside this list.\n"
        "- It needs a beginning, a middle and an end: it opens on its hook, builds, and stops when "
        "the thought completes (the punchline, the answer, the lesson, the turn). Never end "
        "before the payoff, and never start mid-thought.\n"
        + _length_rule(length_pref) +
        "- It must make sense to someone who has not seen the rest of the video. If the first "
        "sentence is an answer to a question another speaker asked, or leans on something said "
        "earlier ('that', 'he', 'like I said'), set opens_mid_conversation true, unless the "
        "sentence restates what it's answering.\n"
        "- Do NOT choose intros, outros, sponsor reads, housekeeping, small talk, or moments that "
        "just repeat what was said earlier.\n"
        "- Moments must not overlap each other.\n"
        "- Loud or excited is not the same as interesting. Judge what is said.\n\n"
        "RUBRIC - integers 0-10. Anchors: 9-10 = you would share this; 7-8 = good, publishable; "
        "5-6 = ordinary, fine but forgettable; 3-4 = weak; 0-2 = unusable. At most about one in five "
        "of your candidates should score 8 or more on any component.\n"
        "- hook: would the first 2 seconds of THIS moment stop a scrolling viewer?\n"
        "- curiosity: does it make the viewer need to hear what comes next?\n"
        "- emotional: humor, surprise, tension, inspiration, vulnerability, conviction.\n"
        "- completeness: clear beginning, middle and end; a satisfying close.\n"
        "- standalone: understandable with zero prior context.\n"
        "- information: does the viewer learn or take away something real?\n\n"
        "For each moment also write, in the SAME LANGUAGE as the transcript:\n"
        "- category: one of story, humor, insight, emotional, hot_take, advice, hook, other\n"
        "- title: short and specific, under 60 characters\n"
        "- reason: 1-2 sentences on why it works, citing what is actually said\n"
        "- overlay_hook: optional on-screen text for the first 2 seconds, at most 9 words. It "
        "must be a faithful teaser or the speaker's own strongest line, built ONLY from things the "
        "speaker actually says in this moment: no new facts, no numbers they don't say, no claims "
        "they don't make, no fabricated quotes. It must add something, not repeat the first "
        "caption. Use an empty string if nothing faithful and compelling fits.\n"
        "- caption: a ready-to-post caption, 1-2 short lines, no hashtags, no quotes\n"
        "- hashtags: 3-6 specific lowercase tags without #\n"
        "- topic: 2-5 word label of what it is about\n"
        "- drop_sentences: optional S numbers INSIDE the moment (never its first or last) that are a "
        "digression or a repeat and can be cut without changing what the speaker means; [] if none\n"
        "- continues: optional {\"start_sentence\": int, \"end_sentence\": int} - ONE later range from this "
        "list on the SAME topic that completes this moment (e.g. the answer to its question, the payoff of "
        "its setup). Only when the two parts truly belong together and the later part makes sense right "
        "after the first; the viewer is shown that it comes from later. null if none.\n\n"
        'Respond with ONLY a JSON object: {"candidates": [{"start_sentence": int, '
        '"end_sentence": int, "category": str, "scores": {"hook": int, "curiosity": int, '
        '"emotional": int, "completeness": int, "standalone": int, "information": int}, '
        '"opens_mid_conversation": bool, "title": str, "reason": str, "overlay_hook": str, '
        '"caption": str, "hashtags": [str], "topic": str, "drop_sentences": [int], "continues": null'
        + (', "matches_request": int' if want else '') + '}]}\n\n'
        "TRANSCRIPT\n" + "\n".join(lines)
    )


def parse_json_object(raw: str):
    """Tolerates code fences and surrounding prose; raises ValueError."""
    if raw is None:
        raise ValueError("empty response")
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in response")
    return json.loads(match.group(0))


LLMFn = Callable[[list[dict]], str]


def propose_candidates(sentences: list[Sentence], video_seconds: float, target: int,
                       llm: LLMFn, total_speech: float, speaker_labels: Optional[list] = None,
                       video_map: Optional[dict] = None, length_pref: Optional[str] = None,
                       request: Optional[dict] = None) -> list[dict]:
    """
    Raw candidate dicts (sentence ranges + rubric + packaging) from every
    window. Raises SelectionUnavailable if any window can't be read after
    one retry — a partial read would silently skip part of the video.
    """
    windows = _windows(sentences)
    if not windows:
        return []

    def window_quota(lo: int, hi: int) -> int:
        span = sentences[hi].end - sentences[lo].start
        share = span / max(video_seconds, 1.0)
        return min(10, max(4, math.ceil(target * 2.0 * share) + 2))

    def run(window: tuple[int, int]) -> list[dict]:
        lo, hi = window
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _proposal_prompt(
                sentences, lo, hi, video_seconds, window_quota(lo, hi), speaker_labels, video_map,
                length_pref, request)},
        ]
        last_error: Optional[Exception] = None
        for _ in range(2):
            try:
                data = parse_json_object(llm(messages))
                items = data.get("candidates") if isinstance(data, dict) else data
                if not isinstance(items, list):
                    raise ValueError("candidates is not a list")
                for item in items:
                    if isinstance(item, dict):
                        item["_window"] = [lo, hi]
                return [i for i in items if isinstance(i, dict)]
            except SelectionUnavailable:
                raise
            except Exception as e:  # malformed output: retry once
                last_error = e
        raise SelectionUnavailable(f"unusable LLM output for window {lo}-{hi}: {last_error}")

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WINDOWS, len(windows))) as pool:
        results = list(pool.map(run, windows))
    return [c for batch in results for c in batch]


# ---------------------------------------------------------------- evaluation

@dataclass
class AnalysisContext:
    """What was measured about the media, as callables over a time window,
    so selection stays testable and optional signals can be absent."""
    # (start, end, n_words) -> (0-100 delivery score, breakdown)
    delivery: Optional[Callable[[float, float, int], tuple]] = None
    # (start, end) -> (0-10 or None, facts)
    audio_quality: Optional[Callable[[float, float], tuple]] = None
    visual_quality: Optional[Callable[[float, float], tuple]] = None
    # per-sentence speaker letters ("A"/"B"/None) when voices could be told apart
    speaker_labels: Optional[list] = None
    # whole-video understanding (summary, topics, speakers); see video_map.py
    video_map: Optional[dict] = None
    # (candidate) -> (points added to the score, info); from creator history
    creator_prior: Optional[Callable] = None
    # "short" / "medium" / "long" (see LENGTH_PREFS); None lets the moment decide
    length_pref: Optional[str] = None
    # How many shorts the creator asked for (1..MAX_REQUESTED_CLIPS); None lets the video's length decide. Fewer
    # are delivered when the video doesn't have that many strong moments.
    clip_count: Optional[int] = None
    # What the creator asked for: {"focus": one of FOCUS, "text": their own words} or None (best overall).
    request: Optional[dict] = None


@dataclass
class Candidate:
    start_sentence: int
    end_sentence: int
    start: float
    end: float
    duration: float
    speech_duration: float
    dead_air_ratio: float
    category: str
    rubric: dict
    context_dependent: bool
    title: str
    reason: str
    hook_line: str                 # the clip's actual opening words, verbatim
    hook_line_verified: bool
    caption: str
    hashtags: list
    topic: str
    transcript: str
    ends_on_sentence: bool
    end_pause: float
    # "" when the LLM's range fit as proposed; "trimmed_to_fit" /
    # "extended_to_fit" when the ending was moved to meet the length limits.
    length_adjustment: str = ""
    overlay_hook: Optional[str] = None      # vetted on-screen hook, or None
    overlay_status: str = "none_proposed"
    opens_as_reply: bool = False
    prepended_question: bool = False
    speakers: list = field(default_factory=list)
    components: dict = field(default_factory=dict)       # nine 0-10 scores (None = not measurable)
    content_score: int = 0
    delivery_score: int = 50
    structure_score: int = 0
    score: int = 0
    breakdown: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)            # filled by clip_review
    # Source time ranges the clip is made of, in order (one range unless sentences were dropped or a later
    # part of the same topic was joined), and what the structure edit did (see clip_edit.plan_structure).
    parts: list = field(default_factory=list)
    structure: dict = field(default_factory=dict)
    request_match: Optional[int] = None    # 0-10, how well it matches what the creator asked for (None: no request)

    def to_dict(self) -> dict:
        return asdict(self)


def clean_hashtags(tags) -> list:
    cleaned = []
    for tag in tags or []:
        tag = re.sub(r"[^\w]", "", str(tag).lower())
        if tag and tag not in cleaned:
            cleaned.append(tag)
        if len(cleaned) >= MAX_HASHTAGS:
            break
    return cleaned


def _int_score(v) -> int:
    try:
        return max(0, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        return 0


# Numbered-list markers ("Second, make sure...") point back at an earlier
# item the viewer never sees. Dropped at the start of a clip only when
# followed by a comma, so "First impressions matter" is left alone.
_LEADING_LIST_MARKERS = {
    "first", "firstly", "second", "secondly", "third", "thirdly", "fourth", "fifth",
    "next", "lastly", "finally", "also", "additionally",
}


def _is_leading_skip(token: str) -> bool:
    k = _key(token)
    return k in _LEADING_FILLERS or (k in _LEADING_LIST_MARKERS and token.rstrip().endswith(","))


def _trim_leading_fillers(words: list[dict], first: int, last: int) -> int:
    i = first
    limit = float(words[first]["start"]) + MAX_LEADING_FILLER_SKIP_SECONDS
    while i < last and _is_leading_skip(words[i]["text"]) and float(words[i + 1]["start"]) <= limit:
        i += 1
    return i


# Penalties for a range the code moved to meet the length limits: the
# LLM scored the moment it proposed, not exactly this one. Trimming can
# cut into the payoff, so it costs more than adding a little context.
LENGTH_ADJUSTMENT_PENALTY = {"trimmed_to_fit": 8, "extended_to_fit": 5}
MAX_EXTENSION_SENTENCES = 3


def _fit_length(words: list[dict], sentences: list["Sentence"], first_word: int,
                s0: int, s1: int, last_allowed: int) -> tuple[Optional[int], Optional[str]]:
    """
    Returns (end_sentence, adjustment) so the clip's speech lands within
    [MIN_CLIP_SECONDS, MAX_CLIP_SECONDS], or (None, reason) if it can't.

    Too long: keep the LLM's opening (the hook) and move the ending back
    to the latest sentence that fits, preferring one that ends on a full
    stop, then one followed by a natural pause. Too short: extend by up
    to MAX_EXTENSION_SENTENCES following sentences, never across a long
    silence (that's where a thought usually ends).
    """
    def speech_to(e: int) -> float:
        return speech_seconds(words[first_word: sentences[e].last_word + 1])

    speech = speech_to(s1)
    if MIN_CLIP_SECONDS <= speech <= MAX_CLIP_SECONDS:
        return s1, None

    if speech > MAX_CLIP_SECONDS:
        best: Optional[tuple[int, int]] = None
        for e in range(s1 - 1, s0 - 1, -1):
            sp = speech_to(e)
            if sp > MAX_CLIP_SECONDS:
                continue
            if sp < MIN_CLIP_SECONDS:
                break
            rank = 2 if sentences[e].terminal else (1 if sentences[e].pause_after >= PAUSE_BREAK_SECONDS else 0)
            if best is None or rank > best[1]:
                best = (e, rank)
            if rank == 2:
                break
        return (best[0], "trimmed_to_fit") if best else (None, "too_long")

    e = s1
    while e < min(last_allowed, len(sentences) - 1) and e - s1 < MAX_EXTENSION_SENTENCES:
        e += 1
        if sentences[e].start - sentences[e - 1].end > SOFT_SENTENCE_BREAK_GAP_SECONDS:
            break
        sp = speech_to(e)
        if sp > MAX_CLIP_SECONDS:
            break
        if sp >= MIN_CLIP_SECONDS:
            return e, "extended_to_fit"
    return None, "too_short"


def _hook_is_verbatim(hook: str, clip_words: list[dict]) -> bool:
    hook_keys = [k for k in (_key(t) for t in hook.split()) if k]
    if not hook_keys:
        return False
    keys = [_key(w["text"]) for w in clip_words[:40]]
    n = len(hook_keys)
    return any(keys[i:i + n] == hook_keys for i in range(0, max(1, len(keys) - n + 1)))


def _prepend_question(words: list[dict], sentences: list[Sentence], s0: int, s1: int,
                      labels: Optional[list]) -> tuple[int, bool]:
    """
    A clip that opens as an answer ("Absolutely, and I think that...")
    is unintelligible without the question. When the previous sentence IS
    the other speaker's short question, start the clip there instead of
    dropping the clip — the standard Q&A edit.
    """
    if s0 == 0:
        return s0, False
    prev = sentences[s0 - 1]
    if not clip_text.is_question(prev.text):
        return s0, False
    if labels and labels[s0 - 1] and labels[s0] and labels[s0 - 1] == labels[s0]:
        return s0, False   # the same person asked and answered: it's rhetorical
    q_speech = speech_seconds(words[prev.first_word: prev.last_word + 1])
    if q_speech > MAX_QUESTION_PREPEND_SECONDS or sentences[s0].start - prev.end > SOFT_SENTENCE_BREAK_GAP_SECONDS * 2:
        return s0, False
    if speech_seconds(words[prev.first_word: sentences[s1].last_word + 1]) > MAX_CLIP_SECONDS:
        return s0, False
    return s0 - 1, True


def evaluate_candidate(raw: dict, sentences: list[Sentence], words: list[dict],
                       video_seconds: float, ctx: Optional[AnalysisContext] = None,
                       ) -> tuple[Optional[Candidate], Optional[str]]:
    """(candidate, None) if usable, else (None, rejection reason)."""
    ctx = ctx or AnalysisContext()
    labels = ctx.speaker_labels
    try:
        s0 = int(raw.get("start_sentence"))
        s1 = int(raw.get("end_sentence"))
    except (TypeError, ValueError):
        return None, "invalid_sentence_ids"
    window = raw.get("_window") or [0, len(sentences) - 1]
    if not (0 <= s0 <= s1 < len(sentences)) or s0 < window[0] or s1 > window[1]:
        return None, "sentence_ids_out_of_range"

    # Boilerplate (intro / outro / sponsor / housekeeping) from the video map.
    if ctx.video_map and _mostly_boilerplate(ctx.video_map, s0, s1):
        return None, "boilerplate"
    # The creator asked for a kind of moment: one that clearly isn't it is left out, not used to fill the count.
    match = None
    if request_text(ctx.request):
        match = _int_score(raw.get("matches_request")) if raw.get("matches_request") is not None else None
        if match is not None and match < MIN_REQUEST_MATCH:
            return None, "off_request"

    first_text = sentences[s0].text
    reply_opening = bool(raw.get("opens_mid_conversation")) or clip_text.opens_as_reply(first_text)
    prepended = False
    if reply_opening:
        s0, prepended = _prepend_question(words, sentences, s0, s1, labels)

    first = _trim_leading_fillers(words, sentences[s0].first_word, sentences[s1].last_word)
    if prepended:
        first = sentences[s0].first_word
    s1, adjustment = _fit_length(words, sentences, first, s0, s1, window[1])
    if s1 is None:
        return None, adjustment
    # The editor's structure pass: digressions dropped, a weak sign-off trimmed, one later part joined.
    structure = clip_edit.plan_structure(raw, sentences, s0, s1, _topic_lookup(ctx.video_map), video_seconds,
                                         MAX_CLIP_SECONDS, window[1])
    main = [r for r in structure["ranges"] if r != structure["joined"]]
    s1 = main[-1][1]
    last = sentences[s1].last_word
    clip_words = words[first: last + 1]
    start = max(0.0, float(clip_words[0]["start"]) - 0.05)
    end = float(clip_words[-1]["end"]) + END_TAIL_SECONDS
    if last + 1 < len(words):
        end = min(end, float(words[last + 1]["start"]) - 0.05)
    end = min(end, video_seconds)
    duration = end - start
    speech = speech_seconds(clip_words)

    rubric = {k: _int_score((raw.get("scores") or {}).get(k)) for k in RUBRIC_KEYS}
    category = str(raw.get("category", "other")).strip().lower()
    if category not in CATEGORIES:
        category = "other"

    transcript = " ".join(w["text"] for w in clip_words)
    opening = " ".join(w["text"] for w in clip_words[:12])[:MAX_HOOK_LINE_CHARS]
    overlay, overlay_status = clip_text.vet_overlay_hook(
        str(raw.get("overlay_hook", "")), transcript, [w["text"] for w in clip_words])

    # Still opening as a reply with no question to anchor it?
    still_dependent = clip_text.opens_as_reply(sentences[s0].text) and not prepended
    clip_speakers = sorted({labels[i] for i in range(s0, s1 + 1) if labels and labels[i]}) if labels else []

    cand = Candidate(
        start_sentence=s0, end_sentence=s1, start=round(start, 3), end=round(end, 3),
        duration=round(duration, 3), speech_duration=round(speech, 3),
        dead_air_ratio=round(max(0.0, 1 - speech / duration), 3) if duration > 0 else 0.0,
        category=category, rubric=rubric,
        context_dependent=bool(raw.get("opens_mid_conversation", False)) and not prepended,
        title=str(raw.get("title", "")).strip()[:MAX_TITLE_CHARS],
        reason=str(raw.get("reason", "")).strip()[:600],
        hook_line=opening, hook_line_verified=True,
        caption=str(raw.get("caption", "")).strip()[:MAX_CAPTION_CHARS],
        hashtags=clean_hashtags(raw.get("hashtags")),
        topic=str(raw.get("topic", "")).strip()[:60],
        transcript=transcript,
        ends_on_sentence=sentences[s1].terminal,
        end_pause=min(sentences[s1].pause_after, 99.0),
        length_adjustment=adjustment or "",
        overlay_hook=overlay, overlay_status=overlay_status,
        opens_as_reply=still_dependent, prepended_question=prepended, speakers=clip_speakers,
    )
    if not cand.title:
        return None, "missing_title"

    score_candidate(cand, len(clip_words), ctx)
    if cand.breakdown.get("gate"):
        return None, cand.breakdown["gate"]
    _apply_structure(cand, structure, sentences, words, first, video_seconds)
    cand.request_match = match
    return cand, None


def _topic_lookup(video_map: Optional[dict]):
    """sentence id -> topic index from the whole-video map (None when there's no map)."""
    topics = (video_map or {}).get("topics") or []
    if not topics:
        return None

    def topic_of(i: int):
        for k, t in enumerate(topics):
            if t["start_sentence"] <= i <= t["end_sentence"]:
                return k
        return None
    return topic_of


def _range_span(words: list, sentences: list, a: int, b: int, first_word: Optional[int], video_seconds: float):
    fw = first_word if first_word is not None else sentences[a].first_word
    lw = sentences[b].last_word
    t0 = max(0.0, float(words[fw]["start"]) - 0.05)
    t1 = float(words[lw]["end"]) + END_TAIL_SECONDS
    if lw + 1 < len(words):
        t1 = min(t1, float(words[lw + 1]["start"]) - 0.05)
    return round(t0, 3), round(min(t1, video_seconds), 3)


def _apply_structure(cand: Candidate, st: dict, sentences: list, words: list, first: int, video_seconds: float):
    """Turns the structure plan into the clip's source time ranges and its spoken text. Scoring already ran on
    the main moment, so a joined part never inflates the score."""
    ranges = st["ranges"]
    cand.structure = {k: v for k, v in st.items() if k != "ranges" and v}
    main = [r for r in ranges if r != st.get("joined")]
    cand.structure["main"] = [main[0][0], main[-1][1]]
    if len(ranges) == 1:
        cand.parts = [[cand.start, cand.end]]
        cand.structure["ranges"] = [list(r) for r in ranges]
        return
    if st.get("joined"):
        cand.structure["unjoined"] = snapshot(cand)
    set_ranges(cand, ranges, sentences, words, video_seconds, first)
    if st.get("joined"):
        cand.structure["joined_at"] = cand.parts[[tuple(r) for r in ranges].index(tuple(st["joined"]))][0]


def snapshot(cand: Candidate) -> dict:
    """The clip as it was before a join, so the join can be undone exactly."""
    return {k: getattr(cand, k) for k in ("parts", "transcript", "duration", "speech_duration", "start", "end",
                                          "start_sentence", "end_sentence")} | {"parts": [list(p) for p in cand.parts]
                                                                                  or [[cand.start, cand.end]]}


def set_ranges(cand: Candidate, ranges: list, sentences: list, words: list, video_seconds: float,
               first: Optional[int] = None) -> None:
    """Sets the clip's parts, text and length from sentence ranges (in order). `first` is the main moment's
    first word when it was trimmed of leading fillers."""
    ranges = [tuple(r) for r in sorted(ranges)]
    main_start = (cand.structure or {}).get("main", [ranges[0][0]])[0]
    parts, ws = [], []
    for a, b in ranges:
        fw = first if (first is not None and a == main_start) else sentences[a].first_word
        parts.append(list(_range_span(words, sentences, a, b, fw, video_seconds)))
        ws += words[fw: sentences[b].last_word + 1]
    cand.parts = parts
    cand.transcript = " ".join(w["text"] for w in ws)
    cand.duration = round(sum(b - a for a, b in parts), 3)
    cand.speech_duration = round(speech_seconds(ws), 3)
    cand.start, cand.end = parts[0][0], parts[-1][1]
    cand.start_sentence, cand.end_sentence = ranges[0][0], ranges[-1][1]
    cand.structure["ranges"] = [list(r) for r in ranges]


def unjoin(cand: Candidate) -> bool:
    """Undoes a join (the review found it changes the meaning, or couldn't confirm it). True if undone."""
    st = cand.structure or {}
    snap = st.get("unjoined")
    if not st.get("joined") or not snap:
        return False
    for k, v in snap.items():
        setattr(cand, k, v)
    joined = tuple(st["joined"])
    cand.structure = {k: v for k, v in st.items() if k not in ("joined", "joined_at", "unjoined", "relation", "why",
                                                             "link", "needs_confirmation")}
    cand.structure["ranges"] = [r for r in st.get("ranges") or [] if tuple(r) != joined]
    cand.structure["join_undone"] = True
    return True


def _mostly_boilerplate(video_map: dict, s0: int, s1: int) -> bool:
    n = s1 - s0 + 1
    covered = 0
    for r in video_map.get("avoid") or []:
        lo, hi = int(r.get("start_sentence", -1)), int(r.get("end_sentence", -2))
        covered += max(0, min(hi, s1) - max(lo, s0) + 1)
    return n > 0 and covered / n >= 0.5


def score_candidate(c: Candidate, n_words: int, ctx: Optional[AnalysisContext] = None) -> None:
    """
    Fills c.components / c.score / c.breakdown. Components that couldn't be
    measured are None and are dropped with the weights renormalised.
    """
    import audio_features
    ctx = ctx or AnalysisContext()

    delivery_score, delivery_info = (50, {"unavailable": True})
    if ctx.delivery is not None:
        delivery_score, delivery_info = ctx.delivery(c.start, c.end, n_words)
    c.delivery_score = delivery_score

    emotional = c.rubric["emotional"]
    if ctx.delivery is not None and not delivery_info.get("unavailable"):
        emotional = (1 - DELIVERY_BLEND) * emotional + DELIVERY_BLEND * (delivery_score / 10.0)

    pacing, pacing_info = audio_features.pacing_score(n_words, c.duration, c.speech_duration)
    audio, audio_info = (None, {"available": False})
    if ctx.audio_quality is not None:
        audio, audio_info = ctx.audio_quality(c.start, c.end)
    visual, visual_info = (None, {"available": False})
    if ctx.visual_quality is not None:
        visual, visual_info = ctx.visual_quality(c.start, c.end)

    c.components = {
        "hook": c.rubric["hook"], "curiosity": c.rubric["curiosity"],
        "emotional": round(emotional, 2), "completeness": c.rubric["completeness"],
        "standalone": c.rubric["standalone"], "information": c.rubric["information"],
        "pacing": pacing, "visual": visual, "audio": audio,
    }

    gate = None
    if min(c.rubric["hook"], c.rubric["completeness"], c.rubric["standalone"]) < CORE_GATE:
        gate = "weak_core"
    elif audio is not None and audio < MIN_AUDIO_QUALITY:
        gate = "poor_audio"

    present = {k: v for k, v in c.components.items() if v is not None}
    wsum = sum(COMPONENT_WEIGHTS[k] for k in present)
    weighted = sum(COMPONENT_WEIGHTS[k] * v for k, v in present.items()) / wsum   # 0-10
    llm_keys = [k for k in COMPONENT_WEIGHTS if k in RUBRIC_KEYS]
    llm_w = sum(COMPONENT_WEIGHTS[k] for k in llm_keys)
    c.content_score = int(round(10 * sum(COMPONENT_WEIGHTS[k] * c.components[k] for k in llm_keys) / llm_w))

    penalties: dict = {}
    if c.context_dependent:
        penalties["needs_earlier_context"] = 12
    if c.opens_as_reply:
        penalties["opens_as_reply"] = 10
    if c.length_adjustment:
        penalties["length_" + c.length_adjustment] = {"trimmed_to_fit": 8, "extended_to_fit": 5}[c.length_adjustment]
    if not c.ends_on_sentence:
        penalties["soft_end" if c.end_pause >= PAUSE_BREAK_SECONDS else "no_sentence_end"] = (
            2 if c.end_pause >= PAUSE_BREAK_SECONDS else 8)
    if c.dead_air_ratio > 0.25:
        penalties["dead_air"] = 5

    prior_points, prior_info = (0.0, {"used": False, "reason": "no creator history supplied"})
    if ctx.creator_prior is not None:
        prior_points, prior_info = ctx.creator_prior(c)

    c.structure_score = max(0, 100 - 5 * sum(penalties.values()))
    c.score = int(round(max(0.0, min(100.0, 10 * weighted - sum(penalties.values()) + prior_points))))
    c.breakdown = {
        "components": dict(c.components),
        "weights": {k: COMPONENT_WEIGHTS[k] for k in present},
        "content": c.content_score, "delivery": delivery_score,
        "penalties": penalties, "creator_prior": prior_info,
        "pacing": pacing_info, "audio": audio_info, "visual": visual_info,
        "delivery_detail": delivery_info,
        "length_adjustment": c.length_adjustment or None,
        "rubric": dict(c.rubric),
        "gate": gate,
    }


# ---------------------------------------------------------------- selection

def _content_words(text: str) -> set:
    return {k for k in (_key(t) for t in text.split()) if k and k not in _STOPWORDS and len(k) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


DUPLICATE_TOPIC_SIMILARITY = 0.5
OVERLAP_TOLERANCE_SECONDS = 0.5


@dataclass
class SelectionResult:
    clips: list[Candidate]
    target: int
    min_score: int
    total_speech_seconds: float
    rejected: list[dict]
    note: str
    spares: list = field(default_factory=list)
    analysis: dict = field(default_factory=dict)
    fixable: list = field(default_factory=list)   # reviewed 'fix': needs setup context

    def summary(self) -> dict:
        return {
            "target": self.target,
            "selected": len(self.clips),
            "spares_kept": len(self.spares),
            "min_score": self.min_score,
            "speech_minutes": round(self.total_speech_seconds / 60, 1),
            "candidates_rejected": len(self.rejected),
            "rejected_reasons": _count_reasons(self.rejected),
            "note": self.note,
            "analysis": self.analysis,
        }


def _count_reasons(rejected: list[dict]) -> dict:
    counts: dict = {}
    for r in rejected:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    return counts


def _spans(c: Candidate) -> list:
    return [tuple(p) for p in c.parts] if c.parts else [(c.start, c.end)]


def _conflicts(c: Candidate, others: list[Candidate]) -> Optional[str]:
    if any(min(x1, y1) - max(x0, y0) > OVERLAP_TOLERANCE_SECONDS
           for a in others for x0, x1 in _spans(c) for y0, y1 in _spans(a)):
        return "overlaps_better_clip"
    words = _content_words(c.transcript)
    if any(_jaccard(words, _content_words(a.transcript)) > DUPLICATE_TOPIC_SIMILARITY for a in others):
        return "duplicate_topic"
    return None


def select_clips(candidates: list[Candidate], target: int, min_score: int,
                 rejected: list[dict], total_speech: float,
                 spares: int = SPARE_CANDIDATES, length_pref: Optional[str] = None) -> SelectionResult:
    """Best-first: the first `target` that clear every gate, then up to
    `spares` more that also clear them (to replace a clip that later fails
    review or rendering QC)."""
    accepted: list[Candidate] = []
    reserve: list[Candidate] = []
    band = LENGTH_PREFS.get(length_pref or "")

    def rank(c: Candidate) -> float:
        # Ordering only: the quality floors below still use the real score.
        if band and band[0] <= c.speech_duration <= band[1]:
            return c.score + LENGTH_FIT_BONUS
        return c.score

    for c in sorted(candidates, key=rank, reverse=True):
        if over_length_choice(c, length_pref):
            rejected.append({"title": c.title, "score": c.score, "reason": "longer_than_chosen_length"})
            continue
        if c.score < min_score:
            rejected.append({"title": c.title, "score": c.score, "reason": "below_quality_floor"})
            continue
        if c.content_score < MIN_CONTENT_SCORE:
            rejected.append({"title": c.title, "score": c.score, "reason": "weak_content"})
            continue
        why = _conflicts(c, accepted + reserve)
        if why:
            rejected.append({"title": c.title, "score": c.score, "reason": why})
            continue
        if len(accepted) < target:
            accepted.append(c)
        elif len(reserve) < spares:
            reserve.append(c)
        else:
            rejected.append({"title": c.title, "score": c.score, "reason": "over_target"})
    return SelectionResult(accepted, target, min_score, total_speech, rejected,
                           _selection_note(accepted, target, candidates, rejected), reserve)


LENGTH_REASONS = {"too_short", "too_long"}


def _selection_note(accepted: list, target: int, candidates: list, rejected: list) -> str:
    if len(accepted) >= target:
        return ""
    if not accepted and not candidates and rejected and all(r["reason"] in LENGTH_REASONS for r in rejected):
        return ("We couldn't cut any moment from this video into a clip of a usable length "
                f"({int(MIN_CLIP_SECONDS)}-{int(MAX_CLIP_SECONDS)} seconds of speech).")
    if not accepted and any(r["reason"] == "longer_than_chosen_length" for r in rejected):
        return ("The good moments in this video need more time than the clip length you chose. "
                "Try Medium, Long or Auto.")
    if not accepted:
        return ("No moment in this video was strong enough to work as a standalone short "
                "without the rest of the video.")
    return (f"Found {len(accepted)} of up to {target} possible clips — the other moments "
            "weren't strong enough to stand on their own.")


def run_selection(words: list[dict], segments: list[dict], video_seconds: float, llm: LLMFn,
                  ctx: Optional[AnalysisContext] = None,
                  min_score: int = DEFAULT_MIN_SCORE) -> SelectionResult:
    ctx = ctx or AnalysisContext()
    punctuated = attach_punctuation(words, segments)
    sentences = build_sentences(punctuated)
    total_speech = speech_seconds(punctuated)
    target = target_clip_count(total_speech)
    if ctx.clip_count:
        # The creator's choice, within what the speech can hold (about one clip per 30 s of speech at most).
        target = max(1, min(int(ctx.clip_count), MAX_REQUESTED_CLIPS, max(1, int(total_speech // 30))))
    if total_speech < MIN_CLIP_SECONDS or not sentences:
        return SelectionResult([], target, min_score, total_speech, [],
                               "There isn't enough speech in this video to make a clip.")
    if ctx.speaker_labels is not None and len(ctx.speaker_labels) != len(sentences):
        ctx.speaker_labels = None   # stale labels for a different sentence split: don't use

    raw = propose_candidates(sentences, video_seconds, target, llm, total_speech,
                             ctx.speaker_labels, ctx.video_map, ctx.length_pref, ctx.request)
    rejected: list[dict] = []
    candidates: list[Candidate] = []
    seen_ranges: set = set()
    for item in raw:
        c, why = evaluate_candidate(item, sentences, punctuated, video_seconds, ctx)
        if c is None:
            rejected.append({"title": str(item.get("title", ""))[:80], "score": None, "reason": why})
            continue
        # Overlapping windows can propose the same range twice.
        span = (c.start_sentence, c.end_sentence)
        if span in seen_ranges:
            continue
        seen_ranges.add(span)
        candidates.append(c)
    result = select_clips(candidates, target, min_score, rejected, total_speech,
                          length_pref=ctx.length_pref)
    result.analysis = {"candidates_evaluated": len(raw),
                       "candidates_passing_checks": len(candidates),
                       "sentences": len(sentences),
                       "speaker_labels_used": bool(ctx.speaker_labels and any(ctx.speaker_labels)),
                       "video_map_used": bool(ctx.video_map)}
    return result
