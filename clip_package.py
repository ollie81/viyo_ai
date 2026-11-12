"""Ready-to-post packaging for every delivered clip, per platform.

ONE batched call to the (configurable, cheap) text model covers all clips of
a job: several title options, opening hook options, a caption and description,
hashtags, and thumbnail ideas, written separately for TikTok, YouTube Shorts
and Instagram Reels. Then every line is checked by code, not trusted:

  * titles and hooks must be grounded in what the speaker actually says
    (clip_text: content overlap, every number said in the clip, no invented
    quotes) - anything else is dropped;
  * no promises of virality or results ("viral", "guaranteed", "you won't
    believe"...), and no clickbait the clip doesn't pay off;
  * lengths and hashtag counts fit each platform (platforms.PLATFORMS).

When the call fails, each clip still gets packaging built from what selection
already wrote (its title, caption, hashtags and vetted hook), marked as such;
nothing is made up to fill the gap.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import clip_selection as cs
import clip_text
import platforms as pf

TITLE_OPTIONS = 3
HOOK_OPTIONS = 3
_PROMISES = re.compile(r"\b(viral|go viral|guarantee[ds]?|you won'?t believe|shocking truth|mind[- ]?blowing|"
                       r"100%|life[- ]?changing|will change your life|secret they don'?t want)\b", re.I)

PROMPT = """You package short vertical videos for TikTok, YouTube Shorts and Instagram Reels.
For EACH clip below, using ONLY what the speaker says in that clip (no facts, numbers, names or claims that
aren't in its transcript, no quotes they didn't say, no promises of virality or results), write in the SAME
LANGUAGE as the transcript:
- titles: {titles} different specific titles, under 70 characters, that the clip delivers on
- hooks: {hooks} opening lines for on-screen text, at most 9 words each, faithful to the clip
- description: 1-2 sentences saying what the viewer gets
- thumbnail: 2 ideas, each {{"moment": "what is on screen, from the transcript", "text": "at most 4 words"}}
- tiktok: {{"caption": first line that makes people watch (under 150 characters), "hashtags": 3-5}}
- shorts: {{"title": under 100 characters, "description": 1-3 lines, "hashtags": 2-3}}
- reels: {{"caption": under 125 characters before "more", "hashtags": 3-5}}
Hashtags: specific, lowercase, without #, about what the clip is actually about.

Respond with ONLY JSON: {{"clips": [{{"id": int, "titles": [str], "hooks": [str], "description": str,
"thumbnail": [{{"moment": str, "text": str}}], "tiktok": {{...}}, "shorts": {{...}}, "reels": {{...}}}}]}}

CLIPS
{clips}"""


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


def _spoken(n: int) -> Optional[str]:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n % 100 == 0 and n < 1000:
        return _ONES[n // 100] + " hundred"
    if n % 1000 == 0 and n < 20000:
        return _ONES[n // 1000] + " thousand"
    return None


def numbers_said(text: str, transcript: str) -> bool:
    """Every number in `text` is said in the clip, as digits or in words (stricter than clip_text's check)."""
    low = transcript.lower().replace("-", " ")
    digits = {d.replace(",", "") for d in re.findall(r"\d[\d,]*", transcript)}
    for n in re.findall(r"\d[\d,]*", text):
        n = n.replace(",", "")
        if n in digits:
            continue
        w = _spoken(int(n)) if n.isdigit() and len(n) < 7 else None
        if w and re.search(rf"\b{w}\b", low):
            continue
        return False
    return True


def _vet_line(text: str, transcript: str, max_chars: int) -> Optional[str]:
    t = re.sub(r"\s+", " ", str(text or "")).strip().strip('"').strip()
    if not t or len(t) > max_chars or _PROMISES.search(t):
        return None
    if not clip_text.numbers_grounded(t, transcript) or not numbers_said(t, transcript):
        return None
    if clip_text.grounding_score(t, transcript) < 0.34:
        return None
    return t


def _tags(raw, lo_hi: tuple, fallback: list) -> list:
    tags = cs.clean_hashtags(raw) if raw else []
    tags = [t for t in tags if not _PROMISES.search(t) and t not in ("viral", "fyp", "foryou", "foryoupage")]
    if len(tags) < lo_hi[0]:
        tags += [t for t in fallback if t not in tags]
    return tags[:lo_hi[1]]


def _fit(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= n:
        return text
    cut = text[:n - 1].rsplit(" ", 1)[0].rstrip(",;:-")
    return cut + "…"


def _fallback(c: dict) -> dict:
    title = c.get("title") or ""
    hook = c.get("overlay_hook") or ""
    tags = list(c.get("hashtags") or [])
    caption = c.get("caption") or title
    return {
        "source": "selection",
        "titles": [title] if title else [],
        "hooks": [hook] if hook else [],
        "description": caption,
        "thumbnail": [],
        "platforms": {
            "tiktok": {"caption": _fit(caption, pf.PLATFORMS["tiktok"]["caption_chars"]),
                       "hashtags": tags[:pf.PLATFORMS["tiktok"]["hashtags"][1]]},
            "shorts": {"title": _fit(title, pf.PLATFORMS["shorts"]["title_chars"]), "description": caption,
                       "hashtags": tags[:pf.PLATFORMS["shorts"]["hashtags"][1]]},
            "reels": {"caption": _fit(caption, pf.PLATFORMS["reels"]["caption_chars"]),
                      "hashtags": tags[:pf.PLATFORMS["reels"]["hashtags"][1]]},
        },
    }


def package(clips: list[dict], llm: Optional[Callable]) -> tuple[list[dict], dict]:
    """Packaging for each clip (same order). `clips`: dicts with title, transcript, caption, hashtags,
    overlay_hook. Returns (packages, facts)."""
    out = [_fallback(c) for c in clips]
    if not clips or llm is None:
        return out, {"written": 0, "fallback": len(clips)}
    blocks = "\n".join(f"--- CLIP {i} (working title: {c.get('title')})\nTRANSCRIPT: {c.get('transcript', '')[:2500]}"
                       for i, c in enumerate(clips))
    try:
        data = cs.parse_json_object(llm([{"role": "user", "content": PROMPT.format(
            titles=TITLE_OPTIONS, hooks=HOOK_OPTIONS, clips=blocks)}]))
    except Exception as e:
        print(f"[WARN] packaging unavailable: {e}")
        return out, {"written": 0, "fallback": len(clips), "error": str(e)[:120]}
    written, dropped = 0, 0
    for item in (data.get("clips") if isinstance(data, dict) else None) or []:
        try:
            i = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if not 0 <= i < len(clips):
            continue
        c, tr = clips[i], clips[i].get("transcript") or ""
        base = out[i]
        titles = [t for t in (_vet_line(x, tr, 70) for x in (item.get("titles") or [])[:TITLE_OPTIONS + 2]) if t]
        hooks = []
        opening = tr.split()[:14]
        for h in (item.get("hooks") or [])[:HOOK_OPTIONS + 2]:
            v, _why = clip_text.vet_overlay_hook(str(h), tr, opening)
            if v and not _PROMISES.search(v) and numbers_said(v, tr) and v not in hooks:
                hooks.append(v)
        dropped += len(item.get("titles") or []) - len(titles) + len(item.get("hooks") or []) - len(hooks)
        thumbs = []
        for t in (item.get("thumbnail") or [])[:2]:
            if isinstance(t, dict):
                text = _vet_line(t.get("text"), tr, 32)
                moment = re.sub(r"\s+", " ", str(t.get("moment") or "")).strip()[:160]
                if moment:
                    thumbs.append({"moment": moment, "text": text or ""})
        tags_fb = list(c.get("hashtags") or [])
        p = {k: item.get(k) if isinstance(item.get(k), dict) else {} for k in ("tiktok", "shorts", "reels")}
        tk = _vet_line(p["tiktok"].get("caption"), tr, 400) or base["platforms"]["tiktok"]["caption"]
        st = _vet_line(p["shorts"].get("title"), tr, 100) or (titles[0] if titles else base["platforms"]["shorts"]["title"])
        rl = _vet_line(p["reels"].get("caption"), tr, 400) or base["platforms"]["reels"]["caption"]
        desc = _vet_line(item.get("description"), tr, 400) or base["description"]
        out[i] = {
            "source": "ai",
            "titles": (titles + [t for t in base["titles"] if t not in titles])[:TITLE_OPTIONS],
            "hooks": (hooks + [h for h in base["hooks"] if h not in hooks])[:HOOK_OPTIONS],
            "description": desc,
            "thumbnail": thumbs,
            "platforms": {
                "tiktok": {"caption": _fit(tk, pf.PLATFORMS["tiktok"]["caption_chars"]),
                           "hashtags": _tags(p["tiktok"].get("hashtags"), pf.PLATFORMS["tiktok"]["hashtags"], tags_fb)},
                "shorts": {"title": _fit(st, pf.PLATFORMS["shorts"]["title_chars"]),
                           "description": _vet_line(p["shorts"].get("description"), tr, 500) or desc,
                           "hashtags": _tags(p["shorts"].get("hashtags"), pf.PLATFORMS["shorts"]["hashtags"], tags_fb)},
                "reels": {"caption": _fit(rl, pf.PLATFORMS["reels"]["caption_chars"]),
                          "hashtags": _tags(p["reels"].get("hashtags"), pf.PLATFORMS["reels"]["hashtags"], tags_fb)},
            },
        }
        written += 1
    return out, {"written": written, "fallback": len(clips) - written, "lines_dropped_by_checks": max(0, dropped)}
