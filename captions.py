"""
Captions for rendered clips: word-synced, punctuated, readable, and kept
inside the area the short-video apps leave free.

Design rules
  * Only words that survive in the edited clip are captioned; times are
    remapped through the same kept-block timeline the video uses, so a
    caption can never describe audio that was cut.
  * Text is the real transcript with its punctuation (not bare ASR tokens).
  * A phrase breaks at sentence ends, pauses, and clause punctuation first,
    and only then by length — never mid-thought if a natural break exists.
  * One phrase stays on screen; the word being spoken is highlighted.
  * Words picked by clip_text.emphasis_indices get the accent colour; with
    several voices, each speaker gets their own colour.
  * Safe zone: TikTok / Reels / Shorts put their own UI over the bottom
    ~20% and right ~12% of the frame, and the title bar at the top. Captions
    sit above that, and the hook sits below the top bar.
Styles: "bold" (uppercase, gold highlight), "clean" (sentence case, cyan
highlight), "boxed" (sentence case on a dark plate).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import clip_text

FONT = "DejaVu Sans"
PLAY_W, PLAY_H = 1080, 1920

# Platform overlays, in output pixels: nothing important goes in these margins.
SAFE_TOP = 250
SAFE_BOTTOM = 400
SAFE_SIDE = 90
SAFE_RIGHT = 140

MAX_LINE_CHARS = 22        # at 1080px wide and the font sizes below
MAX_PHRASE_WORDS = 7
MIN_PHRASE_WORDS = 2
PHRASE_GAP_SECONDS = 0.45
MIN_WORD_SECONDS = 0.08
TAIL_HOLD_SECONDS = 0.25
HOOK_SECONDS = 2.6

# &HAABBGGRR
WHITE = "&H00FFFFFF"
GOLD = "&H0000E5FF"
CYAN = "&H00FFE500"
SPEAKER_COLOURS = ["&H00FFFFFF", "&H0000E5FF", "&H00FFE500", "&H0090FF90"]

STYLES = {
    "bold": {"size": 62, "upper": True, "active": GOLD, "emph": CYAN, "border": 1, "outline": 5, "back": "&H00000000"},
    "clean": {"size": 56, "upper": False, "active": CYAN, "emph": GOLD, "border": 1, "outline": 4, "back": "&H00000000"},
    "boxed": {"size": 54, "upper": False, "active": GOLD, "emph": CYAN, "border": 3, "outline": 14, "back": "&HA0000000"},
}
DEFAULT_STYLE = "bold"


@dataclass
class CaptionWord:
    text: str
    start: float          # original-timeline seconds
    end: float
    speaker: Optional[str] = None


def build_remap(blocks: list):
    """original time -> time in the edited clip. A time inside a removed gap
    maps to the next kept boundary."""
    cumulative, total = [], 0.0
    for b0, b1 in blocks:
        cumulative.append(total)
        total += b1 - b0

    def remap(t: float) -> float:
        for i, (b0, b1) in enumerate(blocks):
            if t < b0:
                return cumulative[i]
            if t <= b1:
                return cumulative[i] + (t - b0)
        return total

    return remap


def clip_total(blocks: list) -> float:
    return sum(b1 - b0 for b0, b1 in blocks)


def words_in_clip(words: list[dict], blocks: list, speaker_at=None) -> list[CaptionWord]:
    """Punctuated words that survive the cut. `words` need `text` (or `word`),
    start, end — see clip_selection.attach_punctuation."""
    out: list[CaptionWord] = []
    for w in words:
        s, e = float(w["start"]), float(w["end"])
        mid = (s + e) / 2
        # A word counts as kept when its middle is in a kept block — a word
        # that only grazes a block edge was cut and must not be captioned.
        if not any(b0 <= mid < b1 for b0, b1 in blocks):
            continue
        text = escape(str(w.get("text") or w.get("word") or "")).strip()
        if not text:
            continue
        out.append(CaptionWord(text, s, e, speaker_at(mid) if speaker_at else None))
    return out


def escape(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\\", "").replace("\n", " ")


def _ends_clause(text: str) -> bool:
    return bool(re.search(r"[.?!…]+[\"')\]]*$", text))


def _soft_break(text: str) -> bool:
    return bool(re.search(r"[,;:—–]$", text))


def group_phrases(words: list[CaptionWord]) -> list[list[CaptionWord]]:
    """Break at sentence ends and long pauses; otherwise at a comma once a
    phrase is long enough; otherwise by length."""
    phrases: list[list[CaptionWord]] = []
    cur: list[CaptionWord] = []
    chars = 0
    for i, w in enumerate(words):
        cur.append(w)
        chars += len(w.text) + 1
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None:
            break
        gap = nxt.start - w.end
        hard = _ends_clause(w.text) or gap > PHRASE_GAP_SECONDS or (nxt.speaker != w.speaker)
        soft = _soft_break(w.text) and len(cur) >= 3
        full = len(cur) >= MAX_PHRASE_WORDS or chars + len(nxt.text) > MAX_LINE_CHARS * 2 - 2
        if hard or soft or full:
            phrases.append(cur)
            cur, chars = [], 0
    if cur:
        phrases.append(cur)
    # Merge one-word orphans into a neighbour when they fit.
    merged: list[list[CaptionWord]] = []
    for p in phrases:
        if (merged and len(p) < MIN_PHRASE_WORDS and len(merged[-1]) + len(p) <= MAX_PHRASE_WORDS
                and p[0].start - merged[-1][-1].end <= PHRASE_GAP_SECONDS
                and p[0].speaker == merged[-1][-1].speaker and not _ends_clause(merged[-1][-1].text)):
            merged[-1].extend(p)
        else:
            merged.append(p)
    return merged


def _wrap(texts: list[str], max_chars: int = MAX_LINE_CHARS) -> list[int]:
    """Indices at which to break a phrase into at most two balanced lines."""
    total = sum(len(t) for t in texts) + len(texts) - 1
    if total <= max_chars or len(texts) < 2:
        return []
    best, best_diff = None, 10 ** 9
    for k in range(1, len(texts)):
        a = sum(len(t) for t in texts[:k]) + k - 1
        b = sum(len(t) for t in texts[k:]) + (len(texts) - k) - 1
        diff = abs(a - b)
        if diff < best_diff:
            best, best_diff = k, diff
    return [best] if best else []


def ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs_total = int(round(seconds * 100))
    h, rem = divmod(cs_total, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def hook_dialogue(hook: str, upper: bool, style: str = "Hook") -> Optional[str]:
    text = escape(hook).strip()
    if not text:
        return None
    text = text.upper() if upper else text
    parts = _wrap(text.split(), 24)
    words = text.split()
    wrapped = " ".join(words) if not parts else " ".join(words[:parts[0]]) + "\\N" + " ".join(words[parts[0]:])
    return f"Dialogue: 1,{ts(0)},{ts(HOOK_SECONDS)},{style},,0,0,0,,{{\\fad(150,200)}}{wrapped}"


def build_ass(words: list[dict], blocks: list, hook: str = "", style: str = DEFAULT_STYLE,
              speaker_at=None, emphasis: bool = True, shift: float = 0.0) -> tuple[Optional[str], dict]:
    """Returns (ASS text or None, facts). None means nothing to caption."""
    st = STYLES.get(style) or STYLES[DEFAULT_STYLE]
    remap = build_remap(blocks)
    total = clip_total(blocks)
    cw = words_in_clip(words, blocks, speaker_at)
    speakers = sorted({w.speaker for w in cw if w.speaker})
    colour_for = {s: SPEAKER_COLOURS[i % len(SPEAKER_COLOURS)] for i, s in enumerate(speakers)}
    multi = len(speakers) >= 2

    header = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {PLAY_W}", f"PlayResY: {PLAY_H}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,{FONT},{st['size']},{WHITE},{WHITE},&H00000000,{st['back']},1,0,0,0,100,100,0,0,"
        f"{st['border']},{st['outline']},1,2,{SAFE_SIDE},{SAFE_RIGHT},{SAFE_BOTTOM},1",
        f"Style: Hook,{FONT},56,{GOLD},{GOLD},&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,5,1,8,"
        f"{SAFE_SIDE},{SAFE_RIGHT},{SAFE_TOP},1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events: list[str] = []
    hook_line = hook_dialogue(hook, st["upper"]) if hook else None
    if hook_line:
        events.append(hook_line)

    n_phrases = 0
    phrases = group_phrases(cw)
    for pi, phrase in enumerate(phrases):
        raw = [w.text for w in phrase]
        shown = [t.upper() if st["upper"] else t for t in raw]
        emph = clip_text.emphasis_indices(raw, hook) if emphasis else set()
        brk = _wrap(shown)
        next_start = remap(phrases[pi + 1][0].start) if pi + 1 < len(phrases) else total
        for i, w in enumerate(phrase):
            start = max(0.0, remap(w.start) + shift)
            if i + 1 < len(phrase):
                end = max(0.0, remap(phrase[i + 1].start) + shift)
            else:
                # Hold the last word a beat, but never into the next phrase.
                end = min(remap(w.end) + shift + TAIL_HOLD_SECONDS, max(next_start + shift, remap(w.end) + shift))
            end = max(start + MIN_WORD_SECONDS, end)
            base = colour_for.get(w.speaker, WHITE) if multi else WHITE
            parts = []
            for j, t in enumerate(shown):
                if j == i:
                    tag = f"{{\\c{st['active']}\\fscx110\\fscy110}}{t}{{\\r}}"
                elif j in emph:
                    tag = f"{{\\c{st['emph']}}}{t}{{\\r}}"
                elif base != WHITE:
                    tag = f"{{\\c{base}}}{t}{{\\r}}"
                else:
                    tag = t
                parts.append(tag)
            if brk:
                text = " ".join(parts[:brk[0]]) + "\\N" + " ".join(parts[brk[0]:])
            else:
                text = " ".join(parts)
            events.append(f"Dialogue: 0,{ts(start)},{ts(end)},Caption,,0,0,0,,{text}")
        n_phrases += 1
    if not events:
        return None, {"words": 0, "phrases": 0}
    facts = {"words": len(cw), "phrases": n_phrases, "style": style if style in STYLES else DEFAULT_STYLE,
             "speaker_colours": multi, "hook_shown": bool(hook_line)}
    return "\n".join(header + events) + "\n", facts


def caption_timeline(ass_text: str) -> list[tuple[float, float, str]]:
    """(start, end, text) of each Caption event — used by QC to check sync."""
    out = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10 or parts[3] != "Caption":
            continue
        out.append((_parse_ts(parts[1]), _parse_ts(parts[2]), re.sub(r"\{[^}]*\}", "", parts[9])))
    return out


def _parse_ts(s: str) -> float:
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)
