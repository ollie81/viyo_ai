"""
Text-level editorial checks for the AI Repurposer — deterministic, no AI.

These are the guardrails around what the language model proposes:
  * an on-screen hook must be GROUNDED in what the speaker actually says
    (no invented claims, no fabricated quotes);
  * a hook must add something, not repeat the first caption;
  * a clip must not open as a reply to something the viewer never hears;
  * which words deserve emphasis in captions.
"""
from __future__ import annotations

import re
from typing import Optional

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "so", "to", "of", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "that", "this", "i", "you", "we", "they",
    "he", "she", "my", "your", "our", "their", "me", "us", "them", "do", "did", "does", "have",
    "has", "had", "not", "no", "yes", "just", "like", "really", "very", "what", "when", "how",
    "if", "then", "there", "here", "about", "from", "as", "by", "all", "can", "will", "would",
    "its", "into", "than", "too", "out", "up", "who", "why", "which", "his", "her", "one",
}

# First words that make a sentence a reply/continuation, not an opening.
REPLY_OPENERS = {
    "absolutely", "exactly", "right", "yeah", "yes", "yep", "sure", "totally", "definitely",
    "correct", "indeed", "agreed", "true", "okay", "ok", "well", "so", "and", "but", "because",
    "which", "that's", "thats", "that", "also", "plus", "anyway", "likewise", "same", "mhm",
    "interesting", "great", "thanks", "thank",
}
# Pronouns that point at something said earlier when they open a clip.
BACKREFERENCE_OPENERS = {"he", "she", "they", "it", "this", "these", "those", "him", "her", "them"}

EMPHASIS_STRONG = {
    "never", "always", "nobody", "everyone", "everything", "nothing", "wrong", "worst", "best",
    "secret", "mistake", "biggest", "only", "impossible", "dangerous", "incredible", "shocking",
    "free", "million", "billion", "thousand", "hate", "love", "afraid", "angry", "dead", "money",
    "truth", "lie", "lies", "stop", "must", "crazy", "insane", "huge", "tiny", "zero", "first", "last",
}
_NEGATIONS = {"not", "no", "never", "nobody", "nothing", "none", "cannot", "can't", "don't", "won't", "isn't"}


def _norm(token: str) -> str:
    return re.sub(r"[^\w']", "", token.lower()).strip("'")


_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
              "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
              "nineteen", "twenty"]


def content_tokens(text: str) -> list[str]:
    """Meaningful words, with small digits spelled out so "3" and "three"
    count as the same word."""
    out = []
    for raw in text.split():
        t = _norm(raw)
        if not t:
            continue
        if t.isdigit() and int(t) < len(_NUM_WORDS):
            t = _NUM_WORDS[int(t)]
        if t.isdigit() or (t not in STOPWORDS and len(t) > 2):
            out.append(t)
    return out


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def grounding_score(hook: str, transcript: str) -> float:
    """Share of the hook's content words (and every number) found in the
    clip's transcript. 1.0 = every claim word is something the speaker said."""
    toks = content_tokens(hook)
    if not toks:
        return 0.0
    said = {_stem(t) for t in content_tokens(transcript)}
    hits = sum(1 for t in toks if _stem(t) in said)
    return hits / len(toks)


def numbers_grounded(hook: str, transcript: str) -> bool:
    nums = re.findall(r"\d[\d,.]*", hook)
    spoken = re.findall(r"\d[\d,.]*", transcript)
    spoken_words = transcript.lower()
    for n in nums:
        if n in spoken or n.replace(",", "") in [s.replace(",", "") for s in spoken]:
            continue
        # spoken as words ("forty", "ninety")?  Accept if a number-word is present.
        if any(w in spoken_words for w in ("one", "two", "three", "four", "five", "six", "seven", "eight",
                                          "nine", "ten", "twenty", "thirty", "forty", "fifty", "hundred",
                                          "thousand", "million", "billion")):
            continue
        return False
    return True


def duplicates_opening(hook: str, clip_words: list[str], first_n: int = 14) -> bool:
    """True when the hook just restates the clip's first spoken words."""
    a = set(_stem(t) for t in content_tokens(hook))
    b = set(_stem(t) for t in content_tokens(" ".join(clip_words[:first_n])))
    if not a or not b:
        return False
    return len(a & b) / len(a) >= 0.7


MAX_OVERLAY_WORDS = 9
MIN_GROUNDING = 0.7


def vet_overlay_hook(hook: str, transcript: str, clip_words: list[str]) -> tuple[Optional[str], str]:
    """(text, reason). text is None when the hook can't be shown, and the
    reason says why — a hook that fails is dropped, never 'repaired' by
    inventing a replacement."""
    hook = re.sub(r"\s+", " ", (hook or "").strip().strip('"“”'))
    if not hook:
        return None, "none_proposed"
    if len(hook.split()) > MAX_OVERLAY_WORDS:
        return None, "too_long"
    if grounding_score(hook, transcript) < MIN_GROUNDING or not numbers_grounded(hook, transcript):
        return None, "not_grounded"
    if duplicates_opening(hook, clip_words):
        return None, "repeats_opening"
    return hook, "ok"


def opens_as_reply(first_sentence: str) -> bool:
    words = [_norm(w) for w in first_sentence.split()[:3]]
    words = [w for w in words if w]
    if not words:
        return False
    if words[0] in REPLY_OPENERS:
        return True
    return words[0] in BACKREFERENCE_OPENERS


def is_question(sentence_text: str) -> bool:
    return sentence_text.rstrip().endswith("?")


def emphasis_indices(tokens: list[str], hook: str = "", max_share: float = 0.14) -> set[int]:
    """Indices of words worth visually emphasising in captions: numbers,
    strong/emotional words, negations, proper nouns mid-sentence, and
    words from the hook — capped to a share of the words so emphasis stays
    meaningful and never spans adjacent words."""
    hook_set = {_stem(t) for t in content_tokens(hook)}
    scored: list[tuple[float, int]] = []
    for i, raw in enumerate(tokens):
        t = _norm(raw)
        if not t or t in STOPWORDS and t not in _NEGATIONS:
            continue
        s = 0.0
        if re.search(r"\d", t):
            s += 3
        if t in EMPHASIS_STRONG:
            s += 2.5
        if t in _NEGATIONS:
            s += 1.5
        if _stem(t) in hook_set:
            s += 2
        if raw[:1].isupper() and i > 0 and not re.search(r"[.?!]$", tokens[i - 1]):
            s += 1.5
        if len(t) >= 9:
            s += 0.8
        if s >= 1.5:
            scored.append((s, i))
    scored.sort(reverse=True)
    limit = max(1, int(len(tokens) * max_share))
    chosen: set[int] = set()
    for _, i in scored:
        if len(chosen) >= limit:
            break
        if i - 1 in chosen or i + 1 in chosen:
            continue
        chosen.add(i)
    return chosen
