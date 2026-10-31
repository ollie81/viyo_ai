"""
Phase 1 clip selection for the AI Repurposer.

Pipeline:
  1. attach_punctuation / build_sentences — whisper-1's word list has no
     punctuation, its segment text does; aligning the two gives sentences
     with exact word-level start/end times.
  2. propose_candidates — the LLM reads numbered sentences (in windows
     for long videos) and proposes moments as sentence RANGES plus a
     rubric score. It never types a timestamp, so it can't invent one:
     boundaries always land on a real sentence start/end.
  3. evaluate_candidate — every proposal is checked against the
     transcript (length, dead air, verbatim hook) and scored:
       75% content rubric (LLM), 15% vocal delivery relative to the
       speaker's own baseline (audio_features), 10% structure,
       minus a penalty if it needs earlier context.
  4. select_clips — best-first, no time overlap, no near-duplicate
     topics, a quality floor, and a count cap that scales with how much
     speech the video actually has. Fewer than asked is a valid result;
     padding with weak clips is not.

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

MIN_CLIP_SECONDS = 12.0
MAX_CLIP_SECONDS = 60.0
TARGET_MAX_CLIPS = 6
# One clip per this much speech, at most. Keeps a 4-minute video from
# being chopped into six thin slices just because six were asked for.
SPEECH_SECONDS_PER_CLIP = 75.0
DEFAULT_MIN_SCORE = 60
# Separate floor on the LLM content score alone (0-100): delivery and
# structure points must never lift a moment that's weak on substance
# over the overall floor. 55 ~ a rubric average just above "ordinary".
MIN_CONTENT_SCORE = 55

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
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "so", "to", "of", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "that", "this", "i", "you", "we", "they",
    "he", "she", "my", "your", "our", "their", "me", "us", "them", "do", "did", "does", "have",
    "has", "had", "not", "no", "yes", "just", "like", "really", "very", "what", "when", "how",
    "if", "then", "there", "here", "about", "from", "as", "by", "all", "can", "will", "would",
}

CATEGORIES = {"story", "humor", "insight", "emotional", "hot_take", "advice", "hook", "other"}
RUBRIC_KEYS = ("hook", "completeness", "standalone", "payoff", "emotional_pull")
_RUBRIC_WEIGHTS = {"hook": 0.30, "completeness": 0.20, "standalone": 0.20, "payoff": 0.20,
                   "emotional_pull": 0.10}

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
    "You are a senior short-form video editor. You pick moments from long-form videos "
    "(podcasts, talks, vlogs, interviews) that work as standalone vertical shorts on TikTok, "
    "YouTube Shorts and Instagram Reels. You are strict: most moments in most videos are not "
    "good enough, and you say so with low scores instead of inflating them. You never invent "
    "content that is not in the transcript."
)


def _proposal_prompt(sentences: list[Sentence], lo: int, hi: int, video_seconds: float,
                     max_candidates: int) -> str:
    lines = [
        f"S{s.id} [{_fmt_time(s.start)}-{_fmt_time(s.end)}] {s.text}"
        for s in sentences[lo: hi + 1]
    ]
    return (
        f"The video is {video_seconds / 60:.1f} minutes long. Below is the transcript for "
        f"{_fmt_time(sentences[lo].start)}-{_fmt_time(sentences[hi].end)} as numbered sentences "
        "with their real start/end times.\n\n"
        f"List the {max_candidates} best candidate moments in this part, best first. Include "
        "weaker ones too and score them honestly: low scores are expected and get filtered out "
        "later, so never inflate a score to justify including a moment. Return fewer only if "
        "this part doesn't contain that many distinct, complete moments.\n\n"
        "RULES\n"
        "- A moment is a contiguous range of sentences: start_sentence..end_sentence "
        "(inclusive), using the S numbers below. Never use numbers outside this list.\n"
        "- Target 15-55 seconds of speech. Shorter is fine when the moment is complete.\n"
        "- It must open on a hook in its first sentence: a bold claim, a question, a "
        "surprising fact, a conflict, a strong emotion, or the start of a story with "
        "tension. Not a greeting, not a recap, not 'so as I was saying'.\n"
        "- It must end when the thought completes: the punchline, the answer, the lesson, "
        "the turn. Never end before the payoff or in the middle of a story.\n"
        "- It must make sense to someone who never saw the rest of the video. If it relies "
        "on something said earlier (unexplained 'he', 'that thing', 'like I said'), set "
        "context_dependent to true.\n"
        "- Moments must not overlap each other.\n"
        "- Loud or excited is not the same as interesting. Judge what is said.\n\n"
        "RUBRIC (integers 0-10, be harsh; 5 = ordinary, 8+ = genuinely strong):\n"
        "- hook: would the first 2 seconds stop a scrolling viewer?\n"
        "- completeness: does it have a clear beginning, middle and end?\n"
        "- standalone: does it work with zero prior context?\n"
        "- payoff: is there a satisfying punchline, insight, answer or emotional beat?\n"
        "- emotional_pull: humor, surprise, tension, inspiration, vulnerability.\n\n"
        "For each moment also write, in the SAME LANGUAGE as the transcript:\n"
        "- category: one of story, humor, insight, emotional, hot_take, advice, hook, other\n"
        "- title: short and specific, under 60 characters\n"
        "- reason: 1-2 sentences on why it works, citing what is actually said\n"
        "- hook_line: the exact opening words of the moment, copied verbatim, max 12 words\n"
        "- caption: a ready-to-post caption, 1-2 short lines, no hashtags, no quotes\n"
        "- hashtags: 3-6 specific lowercase tags without #\n"
        "- topic: 2-5 word label of what it is about\n\n"
        'Respond with ONLY a JSON object: {"candidates": [{"start_sentence": int, '
        '"end_sentence": int, "category": str, "scores": {"hook": int, "completeness": int, '
        '"standalone": int, "payoff": int, "emotional_pull": int}, "context_dependent": bool, '
        '"title": str, "reason": str, "hook_line": str, "caption": str, "hashtags": [str], '
        '"topic": str}]}\n\n'
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
                       llm: LLMFn, total_speech: float) -> list[dict]:
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
        return min(8, max(3, math.ceil(target * 1.5 * share) + 1))

    def run(window: tuple[int, int]) -> list[dict]:
        lo, hi = window
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _proposal_prompt(
                sentences, lo, hi, video_seconds, window_quota(lo, hi))},
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
    hook_line: str
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
    content_score: int = 0
    delivery_score: int = 50
    structure_score: int = 0
    score: int = 0
    breakdown: dict = field(default_factory=dict)

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


def _trim_leading_fillers(words: list[dict], first: int, last: int) -> int:
    i = first
    limit = float(words[first]["start"]) + MAX_LEADING_FILLER_SKIP_SECONDS
    while i < last and _key(words[i]["text"]) in _LEADING_FILLERS and float(words[i + 1]["start"]) <= limit:
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


def evaluate_candidate(raw: dict, sentences: list[Sentence], words: list[dict],
                       video_seconds: float,
                       delivery: Optional[Callable[[float, float, int], tuple[int, dict]]] = None,
                       ) -> tuple[Optional[Candidate], Optional[str]]:
    """(candidate, None) if usable, else (None, rejection reason)."""
    try:
        s0 = int(raw.get("start_sentence"))
        s1 = int(raw.get("end_sentence"))
    except (TypeError, ValueError):
        return None, "invalid_sentence_ids"
    window = raw.get("_window") or [0, len(sentences) - 1]
    if not (0 <= s0 <= s1 < len(sentences)) or s0 < window[0] or s1 > window[1]:
        return None, "sentence_ids_out_of_range"

    first = _trim_leading_fillers(words, sentences[s0].first_word, sentences[s1].last_word)
    s1, adjustment = _fit_length(words, sentences, first, s0, s1, window[1])
    if s1 is None:
        return None, adjustment
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

    hook_line = str(raw.get("hook_line", "")).strip()
    verified = _hook_is_verbatim(hook_line, clip_words)
    if not verified:
        # Never show words on screen that weren't said: fall back to the
        # clip's real opening words.
        hook_line = " ".join(w["text"] for w in clip_words[:10])
    hook_line = hook_line[:MAX_HOOK_LINE_CHARS]

    cand = Candidate(
        start_sentence=s0, end_sentence=s1, start=round(start, 3), end=round(end, 3),
        duration=round(duration, 3), speech_duration=round(speech, 3),
        dead_air_ratio=round(max(0.0, 1 - speech / duration), 3) if duration > 0 else 0.0,
        category=category, rubric=rubric,
        context_dependent=bool(raw.get("context_dependent", False)),
        title=str(raw.get("title", "")).strip()[:MAX_TITLE_CHARS],
        reason=str(raw.get("reason", "")).strip()[:600],
        hook_line=hook_line, hook_line_verified=verified,
        caption=str(raw.get("caption", "")).strip()[:MAX_CAPTION_CHARS],
        hashtags=clean_hashtags(raw.get("hashtags")),
        topic=str(raw.get("topic", "")).strip()[:60],
        transcript=" ".join(w["text"] for w in clip_words),
        ends_on_sentence=sentences[s1].terminal,
        end_pause=min(sentences[s1].pause_after, 99.0),
        length_adjustment=adjustment or "",
    )
    if not cand.title:
        return None, "missing_title"

    if delivery is not None:
        cand.delivery_score, delivery_breakdown = delivery(start, end, len(clip_words))
    else:
        delivery_breakdown = {"unavailable": True}

    score_candidate(cand)
    cand.breakdown["delivery"] = delivery_breakdown
    return cand, None


def score_candidate(c: Candidate) -> None:
    c.content_score = int(round(10 * sum(_RUBRIC_WEIGHTS[k] * c.rubric[k] for k in RUBRIC_KEYS)))
    structure = 100
    penalties = {}
    if not c.ends_on_sentence:
        # Ending at a real pause usually sounds finished even without a
        # full stop in the transcript; ending mid-flow does not.
        if c.end_pause >= PAUSE_BREAK_SECONDS:
            penalties["soft_sentence_end"] = 10
        else:
            penalties["no_sentence_end"] = 35
    if c.dead_air_ratio > 0.25:
        penalties["dead_air"] = 20
    if c.speech_duration < 15:
        penalties["very_short"] = 10
    structure -= sum(penalties.values())
    c.structure_score = max(0, structure)

    context_penalty = 15 if c.context_dependent else 0
    length_penalty = LENGTH_ADJUSTMENT_PENALTY.get(c.length_adjustment, 0)
    final = (0.75 * c.content_score + 0.15 * c.delivery_score + 0.10 * c.structure_score
             - context_penalty - length_penalty)
    c.score = max(0, min(100, int(round(final))))
    c.breakdown = {
        "content": c.content_score, "delivery": c.delivery_score,
        "structure": c.structure_score, "structure_penalties": penalties,
        "context_penalty": context_penalty, "length_adjustment": c.length_adjustment or None,
        "length_penalty": length_penalty, "rubric": dict(c.rubric),
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

    def summary(self) -> dict:
        return {
            "target": self.target,
            "selected": len(self.clips),
            "min_score": self.min_score,
            "speech_minutes": round(self.total_speech_seconds / 60, 1),
            "candidates_rejected": len(self.rejected),
            "rejected_reasons": _count_reasons(self.rejected),
            "note": self.note,
        }


def _count_reasons(rejected: list[dict]) -> dict:
    counts: dict = {}
    for r in rejected:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    return counts


def select_clips(candidates: list[Candidate], target: int, min_score: int,
                 rejected: list[dict], total_speech: float) -> SelectionResult:
    accepted: list[Candidate] = []
    for c in sorted(candidates, key=lambda c: c.score, reverse=True):
        if len(accepted) >= target:
            rejected.append({"title": c.title, "score": c.score, "reason": "over_target"})
            continue
        if c.score < min_score:
            rejected.append({"title": c.title, "score": c.score, "reason": "below_quality_floor"})
            continue
        if c.content_score < MIN_CONTENT_SCORE:
            rejected.append({"title": c.title, "score": c.score, "reason": "weak_content"})
            continue
        if any(min(c.end, a.end) - max(c.start, a.start) > OVERLAP_TOLERANCE_SECONDS for a in accepted):
            rejected.append({"title": c.title, "score": c.score, "reason": "overlaps_better_clip"})
            continue
        words = _content_words(c.transcript)
        if any(_jaccard(words, _content_words(a.transcript)) > DUPLICATE_TOPIC_SIMILARITY
               for a in accepted):
            rejected.append({"title": c.title, "score": c.score, "reason": "duplicate_topic"})
            continue
        accepted.append(c)

    if len(accepted) >= target:
        note = ""
    elif not accepted and not candidates and rejected:
        note = ("We couldn't cut any moment from this video into a clip of a usable length "
                f"({int(MIN_CLIP_SECONDS)}-{int(MAX_CLIP_SECONDS)} seconds of speech).")
    elif not accepted:
        note = ("No moment in this video was strong enough to work as a standalone short "
                "without the rest of the video.")
    else:
        note = (f"Found {len(accepted)} of up to {target} possible clips — the other moments "
                "weren't strong enough to stand on their own.")
    return SelectionResult(accepted, target, min_score, total_speech, rejected, note)


def run_selection(words: list[dict], segments: list[dict], video_seconds: float, llm: LLMFn,
                  delivery: Optional[Callable[[float, float, int], tuple[int, dict]]] = None,
                  min_score: int = DEFAULT_MIN_SCORE) -> SelectionResult:
    punctuated = attach_punctuation(words, segments)
    sentences = build_sentences(punctuated)
    total_speech = speech_seconds(punctuated)
    target = target_clip_count(total_speech)
    if total_speech < MIN_CLIP_SECONDS or not sentences:
        return SelectionResult([], target, min_score, total_speech, [],
                               "There isn't enough speech in this video to make a clip.")

    raw = propose_candidates(sentences, video_seconds, target, llm, total_speech)
    rejected: list[dict] = []
    candidates: list[Candidate] = []
    seen_ranges: set = set()
    for item in raw:
        c, why = evaluate_candidate(item, sentences, punctuated, video_seconds, delivery)
        if c is None:
            rejected.append({"title": str(item.get("title", ""))[:80], "score": None, "reason": why})
            continue
        # Overlapping windows can propose the same range twice.
        span = (c.start_sentence, c.end_sentence)
        if span in seen_ranges:
            continue
        seen_ranges.add(span)
        candidates.append(c)
    return select_clips(candidates, target, min_score, rejected, total_speech)
