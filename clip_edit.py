"""The editor's pass over a chosen moment: its structure, its pacing and its
opening. Deterministic, no AI, nothing invented.

Structure (applied when the moment is chosen, see clip_selection):
  * drop    — the selection model may mark whole sentences INSIDE a moment
              as a digression or a repeat. They are cut only when the moment
              still reads as one thought: never the first or last sentence,
              never more than a third of it, and never when the sentence
              after the cut leans on what was cut ("that", "he", "so"...).
  * join    — the model may add ONE later range from the same topic (a
              question and its answer minutes apart, a setup and its payoff).
              Joined only when both parts are whole sentences, in their real
              order, on the same topic, and the later part doesn't open as a
              reply to something the viewer never hears. The viewer sees a
              "Later" marker at the join; the creator sees both source times.
  * weak ending — a trailing sign-off ("so yeah", "anyway", "thanks for
              watching") is dropped when the thought already ended before it.

Pacing (applied when the clip is rendered):
  * long pauses INSIDE a sentence are shortened to a natural beat;
  * immediate restarts ("I I think", "we went, we went to") lose the first,
    abandoned attempt. Only exact repeats of the same words, back to back,
    from the same speaker: never a word that changes the meaning.

Opening (cold open):
  * when a clip opens slowly, its single strongest line, taken verbatim from
    LATER in the same clip, plays first as a short preview; then the clip
    plays in full, so the line is heard again in its real context. Only a
    complete sentence that doesn't lean on anything before it.
"""
from __future__ import annotations

import re
from typing import Optional

import clip_text

# ------------------------------------------------------------------ structure

MAX_DROP_SHARE = 1 / 3
MIN_PART_SECONDS = 4.0
MAX_JOIN_GAP_SECONDS = 20 * 60
MIN_JOIN_GAP_SECONDS = 8.0       # closer than this, it's one moment with a cut, not a join

_SIGN_OFF = re.compile(
    r"^(so yeah|yeah|anyway|anyways|so anyway|that's (it|all)|thank you|thanks( for watching| for listening)?|"
    r"(don't forget to |make sure to )?(like and )?subscribe|see you|bye|cool|okay|ok|alright|all right|right)\W*$",
    re.I)
_LEANS_BACK = {"that", "this", "those", "these", "it", "he", "she", "they", "him", "her", "them", "so", "and",
               "but", "because", "then", "also", "which", "there"}


def _first_word(text: str) -> str:
    w = [clip_text._norm(x) for x in text.split()[:1]]
    return w[0] if w else ""


def leans_back(text: str) -> bool:
    """A sentence that only makes sense after the one before it."""
    return clip_text.opens_as_reply(text) or _first_word(text) in _LEANS_BACK


def _ids(raw, lo: int, hi: int) -> list[int]:
    out = []
    for x in raw or []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if lo <= i <= hi and i not in out:
            out.append(i)
    return sorted(out)


def plan_structure(raw: dict, sentences: list, s0: int, s1: int, topic_of=None, video_seconds: float = 0.0,
                   max_seconds: float = 60.0, window_hi: Optional[int] = None) -> dict:
    """The sentence ranges a moment is made of, from the selection model's optional `drop_sentences` and
    `continues` fields, after the safety rules above. Returns {"ranges": [(a, b), ...] (sentence ids, inclusive,
    in order), "dropped": [...], "joined": (a, b) or None, "trimmed_ending": [...], "refused": [reasons]}."""
    refused: list[str] = []
    # Weak ending: drop trailing sign-offs when the sentence before them ended the thought.
    trimmed = []
    while s1 - s0 >= 2 and _SIGN_OFF.match(sentences[s1].text.strip()) and sentences[s1 - 1].terminal:
        trimmed.append(s1)
        s1 -= 1
    keep = list(range(s0, s1 + 1))
    # Digressions inside the moment.
    drop = _ids(raw.get("drop_sentences"), s0 + 1, s1 - 1)
    if drop:
        if len(drop) > len(keep) * MAX_DROP_SHARE:
            refused.append("drop_too_much")
            drop = []
        else:
            after = [i + 1 for i in drop if i + 1 not in drop]
            if any(leans_back(sentences[i].text) for i in after if i <= s1):
                refused.append("drop_breaks_meaning")
                drop = []
    kept = [i for i in keep if i not in drop]
    ranges = _runs(kept)
    joined = None
    cont = raw.get("continues")
    if isinstance(cont, dict):
        try:
            a, b = int(cont.get("start_sentence")), int(cont.get("end_sentence"))
        except (TypeError, ValueError):
            a = b = -1
        hi = window_hi if window_hi is not None else len(sentences) - 1
        if not (s1 < a <= b <= hi):
            refused.append("join_out_of_order")
        elif sentences[a].start - sentences[s1].end < MIN_JOIN_GAP_SECONDS:
            refused.append("join_too_close")
        elif sentences[a].start - sentences[s1].end > MAX_JOIN_GAP_SECONDS:
            refused.append("join_too_far")
        elif sentences[b].end - sentences[a].start < MIN_PART_SECONDS:
            refused.append("join_too_short")
        elif leans_back(sentences[a].text):
            refused.append("join_opens_as_reply")
        elif not sentences[b].terminal:
            refused.append("join_ends_mid_sentence")
        elif topic_of is not None and topic_of(s0) is not None and topic_of(a) != topic_of(s0):
            refused.append("join_other_topic")
        elif topic_of is None and _overlap(sentences, kept, list(range(a, b + 1))) < 0.08:
            refused.append("join_unrelated")
        else:
            total = sum(sentences[y].end - sentences[x].start for x, y in ranges) + sentences[b].end - sentences[a].start
            if total > max_seconds:
                refused.append("join_too_long")
            else:
                joined = (a, b)
                ranges.append((a, b))
    return {"ranges": ranges, "dropped": drop, "joined": joined, "trimmed_ending": trimmed, "refused": refused}


def _runs(ids: list[int]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for i in ids:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out]


def _overlap(sentences, a_ids, b_ids) -> float:
    a = set(clip_text.content_tokens(" ".join(sentences[i].text for i in a_ids)))
    b = set(clip_text.content_tokens(" ".join(sentences[i].text for i in b_ids)))
    return len(a & b) / max(1, min(len(a), len(b)))


# ------------------------------------------------------------------ pacing

PAUSE_KEEP = 0.30          # a long pause inside speech becomes a natural beat of this length
PAUSE_MIN = 0.60           # pauses shorter than this are left alone (that's how people talk)
MAX_REPEAT_WORDS = 4


def _tok(w: dict) -> str:
    return clip_text._norm(str(w.get("text") or w.get("word") or ""))


def _cut(blocks: list, a: float, b: float) -> list:
    out = []
    for s, e in blocks:
        if b <= s or a >= e:
            out.append((s, e))
            continue
        if a > s:
            out.append((s, a))
        if b < e:
            out.append((b, e))
    return [(s, e) for s, e in out if e - s >= 0.12]


def tighten(blocks: list, words: list, speaker_at=None, pause_keep: float = PAUSE_KEEP,
            pause_min: float = PAUSE_MIN) -> tuple[list, dict]:
    """Shorter long pauses inside speech and no abandoned restarts. `words`: punctuated words with start/end.
    Returns (new blocks, facts). Never removes a word that isn't an exact back-to-back repeat."""
    if not blocks or not words:
        return blocks, {"pauses_tightened": 0, "pause_seconds_removed": 0.0, "repeats_removed": []}
    lo, hi = blocks[0][0], blocks[-1][1]
    inside = [w for w in words if lo <= (float(w["start"]) + float(w["end"])) / 2 < hi
              and any(s <= (float(w["start"]) + float(w["end"])) / 2 < e for s, e in blocks)]
    out = list(blocks)
    removed_pause = 0.0
    n_pauses = 0
    for w0, w1 in zip(inside, inside[1:]):
        a, b = float(w0["end"]), float(w1["start"])
        gap = b - a
        if gap < pause_min:
            continue
        # Only a pause that the kept blocks still contain (dead air between blocks is already gone).
        if not any(s <= a and b <= e for s, e in out):
            continue
        cut_a, cut_b = a + pause_keep / 2, b - pause_keep / 2
        out = _cut(out, cut_a, cut_b)
        removed_pause += cut_b - cut_a
        n_pauses += 1
    # Restarts: the same 1-4 words said twice in a row, by the same speaker, within a breath.
    repeats = []
    toks = [_tok(w) for w in inside]
    i = 0
    while i < len(inside):
        done = False
        for n in range(MAX_REPEAT_WORDS, 0, -1):
            if i + 2 * n > len(inside):
                continue
            first, second = toks[i:i + n], toks[i + n:i + 2 * n]
            if not all(first) or first != second:
                continue
            if n == 1 and first[0] in _KEEP_DOUBLE:
                continue                         # "very very", "no no": emphasis, not a restart
            a, b = float(inside[i]["start"]), float(inside[i + n]["start"])
            if b - a > 2.5:
                continue
            if speaker_at and speaker_at(a) != speaker_at(b):
                continue
            out = _cut(out, a, b - 0.02)
            repeats.append(" ".join(str(w.get("text") or w.get("word")) for w in inside[i:i + n]))
            i += n
            done = True
            break
        if not done:
            i += 1
    return out, {"pauses_tightened": n_pauses, "pause_seconds_removed": round(removed_pause, 2),
                 "repeats_removed": repeats}


_KEEP_DOUBLE = {"very", "no", "yes", "yeah", "so", "really", "much", "more", "bye", "ha", "go", "okay", "ok",
                "wow", "oh", "hey", "now", "far", "long", "many", "too", "again"}


# ------------------------------------------------------------------ opening

_SLOW_OPEN = re.compile(
    r"^(so|okay|ok|alright|all right|well|hi|hello|hey|welcome|today|basically|um|uh|let me|i want to|"
    r"i'm going to|we're going to|in this video|first of all|you know)\b", re.I)
TEASER_MIN, TEASER_MAX = 1.4, 4.5


def slow_opening(words: list, start: float) -> bool:
    """The clip's first seconds are a warm-up, not a hook: a silent start, or a throat-clearing first phrase."""
    first = [w for w in words if float(w["start"]) >= start - 0.05][:6]
    if not first:
        return False
    if float(first[0]["start"]) - start > 0.9:
        return True
    text = " ".join(str(w.get("text") or w.get("word")) for w in first)
    return bool(_SLOW_OPEN.match(text.strip()))


def choose_teaser(sentences: list, s_ids: list[int], words: list, clip_start: float) -> Optional[dict]:
    """The strongest complete line from later in the clip (not its first 30%), 1.4-4.5 s long, standing on its
    own. Returns {"sentence": id, "start": t, "end": t, "text": str} or None."""
    if not s_ids or not slow_opening(words, clip_start):
        return None
    span = sentences[s_ids[-1]].end - sentences[s_ids[0]].start
    best, best_score = None, 0.0
    for i in s_ids[1:]:
        s = sentences[i]
        if s.start - sentences[s_ids[0]].start < 0.3 * span or not s.terminal:
            continue
        d = s.end - s.start
        if not (TEASER_MIN <= d <= TEASER_MAX) or leans_back(s.text) or _SIGN_OFF.match(s.text.strip()):
            continue
        toks = [clip_text._norm(w) for w in s.text.split()]
        score = (2.0 * sum(t in clip_text.EMPHASIS_STRONG for t in toks) + 1.5 * any(re.search(r"\d", t) for t in toks)
                 + 1.0 * sum(t in ("you", "your", "never", "always", "nobody", "everyone") for t in toks)
                 + 1.0 * s.text.rstrip().endswith(("!", "?")) + 0.5)
        if score > best_score:
            best, best_score = {"sentence": i, "start": round(s.start - 0.04, 3), "end": round(s.end + 0.12, 3),
                                "text": s.text}, score
    return best if best_score >= 2.0 else None
