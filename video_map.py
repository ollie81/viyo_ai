"""
Whole-video understanding: one LLM pass over the ENTIRE transcript before
any clip is proposed, so each selection window knows what the video is
about, who is speaking, how the topics flow, and which stretches are
intro / outro / sponsor / housekeeping that should never become a clip.

Everything the model returns is validated against the real sentence list
(ids in range, ranges ordered, labels only from the detected speakers).
A failure is non-fatal by design: selection simply runs without the map,
and the job records that it did — never a fabricated one.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

import clip_selection as cs

# ~3 tokens/word: a 33-minute talk (about 6,000 words) is well inside any
# modern context window. Beyond this we summarise in compact form instead.
MAX_FULL_TRANSCRIPT_WORDS = 14000
KINDS = {"story", "argument", "explanation", "qa", "banter", "intro", "outro", "ad", "housekeeping", "other"}
AVOID_KINDS = {"intro", "outro", "ad", "housekeeping"}
RELATIONS = {"setup_payoff", "question_answer", "claim_example", "problem_solution", "callback"}
MAX_THREADS = 8
FORMATS = {"podcast", "interview", "talk", "panel", "vlog", "tutorial", "other"}

PROMPT = """You are analysing a long video so that a short-form editor can later pick its best moments.
Read the whole numbered transcript. {speaker_note}

Return ONLY a JSON object:
{{
  "summary": "2-3 sentences: what this video is and what happens in it",
  "format": one of podcast, interview, talk, panel, vlog, tutorial, other,
  "speakers": [{{"label": "A", "role": "host|guest|presenter|narrator|unknown", "description": "who they are, only if the transcript says so"}}],
  "topics": [{{"start_sentence": int, "end_sentence": int, "kind": one of story, argument, explanation, qa, banter, intro, outro, ad, housekeeping, other, "title": "short title"}}],
  "avoid": [{{"start_sentence": int, "end_sentence": int, "kind": "intro|outro|ad|housekeeping"}}],
  "threads": [{{"relation": one of setup_payoff, question_answer, claim_example, problem_solution, callback,
               "parts": [{{"start_sentence": int, "end_sentence": int}}, {{"start_sentence": int, "end_sentence": int}}],
               "why": "one sentence: how the parts belong together"}}]
}}

Rules:
- Topics cover the video in order, without overlapping; use real S numbers from the list.
- "avoid" lists ONLY stretches that are greetings/introductions, sign-offs, sponsor reads or ads,
  channel housekeeping ("subscribe", "link in the description"), or technical set-up chatter.
  Do not list ordinary content, even if it is dull.
- "threads" (at most 8) link two moments in DIFFERENT parts of the video that tell one story together: a
  setup and its payoff, a question and its answer, a claim and the example that proves it, a problem and its
  solution, or a callback. Each part is a few complete sentences (about 5-25 seconds), in the order they happen.
  Only when the second part, heard right after the first, keeps exactly what the speaker meant. [] if none.
- Only state facts the transcript supports. Use "unknown" when you can't tell.

TRANSCRIPT
{lines}
"""


def _compact_lines(sentences: list, labels: Optional[list]) -> str:
    out = []
    for s in sentences:
        who = f"{labels[s.id]} " if labels and labels[s.id] else ""
        out.append(f"S{s.id} {who}{s.text}")
    return "\n".join(out)


def build_video_map(sentences: list, speaker_labels: Optional[list], llm: Callable[[list], str],
                    speakers_reliable: bool = False) -> Optional[dict]:
    """Returns a validated map, or None if the pass failed (non-fatal)."""
    if len(sentences) < 4:
        return None
    n_words = sum(len(s.text.split()) for s in sentences)
    if n_words > MAX_FULL_TRANSCRIPT_WORDS:
        # Keep each sentence's opening so the whole video still fits.
        sentences = [type(s)(**{**s.__dict__, "text": " ".join(s.text.split()[:18])}) for s in sentences]
    note = ("Letters before a sentence are speakers identified from their voices (the same letter is the "
            "same person)." if speakers_reliable and speaker_labels and any(speaker_labels)
            else "Speaker identities are not known.")
    messages = [
        {"role": "system", "content": "You are an expert video editor and story analyst. You report only what the transcript supports."},
        {"role": "user", "content": PROMPT.format(speaker_note=note, lines=_compact_lines(sentences, speaker_labels))},
    ]
    try:
        data = cs.parse_json_object(llm(messages))
        return _validate(data, len(sentences), speaker_labels if speakers_reliable else None)
    except Exception as e:
        print(f"[WARN] video map unavailable: {e}")
        return None


def _validate(data: dict, n: int, labels: Optional[list]) -> dict:
    if not isinstance(data, dict):
        raise ValueError("not an object")

    def rng(item) -> Optional[tuple[int, int]]:
        try:
            a, b = int(item["start_sentence"]), int(item["end_sentence"])
        except (KeyError, TypeError, ValueError):
            return None
        return (a, b) if 0 <= a <= b < n else None

    allowed = {l for l in (labels or []) if l}
    topics, avoid = [], []
    for t in data.get("topics") or []:
        r = rng(t) if isinstance(t, dict) else None
        if r:
            kind = str(t.get("kind", "other")).lower()
            topics.append({"start_sentence": r[0], "end_sentence": r[1],
                           "kind": kind if kind in KINDS else "other", "title": str(t.get("title", ""))[:80]})
    topics.sort(key=lambda t: t["start_sentence"])
    for a in data.get("avoid") or []:
        r = rng(a) if isinstance(a, dict) else None
        kind = str(a.get("kind", "")).lower() if isinstance(a, dict) else ""
        if r and kind in AVOID_KINDS:
            avoid.append({"start_sentence": r[0], "end_sentence": r[1], "kind": kind})
    speakers = []
    for sp in data.get("speakers") or []:
        if isinstance(sp, dict) and (not allowed or str(sp.get("label")) in allowed):
            speakers.append({"label": str(sp.get("label"))[:2], "role": str(sp.get("role", "unknown"))[:20],
                             "description": str(sp.get("description", ""))[:120]})
    threads = []
    for th in (data.get("threads") or [])[:MAX_THREADS]:
        if not isinstance(th, dict):
            continue
        parts = [rng(p) for p in (th.get("parts") or []) if isinstance(p, dict)]
        rel = str(th.get("relation", "")).lower()
        if len(parts) != 2 or None in parts or rel not in RELATIONS:
            continue
        (a0, a1), (b0, b1) = parts
        if b0 <= a1 or (a1 - a0) > 12 or (b1 - b0) > 12:      # in order, apart, and each a short moment
            continue
        threads.append({"relation": rel, "parts": [[a0, a1], [b0, b1]], "why": str(th.get("why", ""))[:200]})
    fmt = str(data.get("format", "other")).lower()
    summary = str(data.get("summary", "")).strip()[:500]
    if not summary:
        raise ValueError("empty summary")
    return {"summary": summary, "format": fmt if fmt in FORMATS else "other", "speakers": speakers,
            "topics": topics, "avoid": avoid, "threads": threads}
