"""Viyo's voice director: one Character Voice Bible per production (or series),
scene-by-scene performance direction, and a translation of both into what
each video or voice provider can actually take.

Why: the video models make the dialogue themselves, and they differ. None of
the video models Viyo uses accepts a voice reference or a persistent voice ID
through our integrations, so a character's voice is held together by the SAME
detailed description sent every time, adapted to each provider's limits. The
saved-voices path (Gemini TTS) is the one place with a persistent voice (a
named prebuilt voice), so it's the fallback when a clip's own audio fails.
Identical voices across providers are NOT guaranteed; the limits of each
provider are recorded with every clip.

After a talking clip is made, its audio is checked: technical checks with
ffmpeg (is there sound, is it silent, clipped) and an AI that listens (which
lines were said, by whom, in which language, how naturally, how clearly).
"""
from __future__ import annotations

import datetime
import json
import re
from typing import Optional

# What each provider can really take (from our integration code and the
# providers' public parameters as integrated; re-check when an integration changes).
CAPS = {
    "omni": {"label": "Google Gemini Omni", "native_audio": True, "multi_speaker": "prompt",
             "voice_reference": False, "persistent_voice": False, "emotion": "prompt", "lip_sync": "native",
             "timing": "line order only", "languages": None, "prompt_chars": 6000, "images": 6},
    "kling": {"label": "Kling 3 Pro (fal.ai)", "native_audio": True, "multi_speaker": "prompt",
              "voice_reference": False, "persistent_voice": False, "emotion": "prompt", "lip_sync": "native",
              "timing": "line order only", "languages": {"english", "chinese"}, "prompt_chars": 2500, "images": 1},
    "seedance": {"label": "Seedance 2.0 (fal.ai)", "native_audio": True, "multi_speaker": "prompt",
                 "voice_reference": False, "persistent_voice": False, "emotion": "prompt", "lip_sync": "native",
                 "timing": "line order only", "languages": None, "prompt_chars": 2500, "images": 1},
    "tts": {"label": "Gemini TTS (saved voices)", "native_audio": False, "multi_speaker": "one voice per line",
            "voice_reference": False, "persistent_voice": True, "emotion": "style prompt", "lip_sync": "none (needs lip sync)",
            "timing": "exact (Viyo joins the lines)", "languages": None, "prompt_chars": 1000, "images": 0},
}

_S = {"type": "string"}
BIBLE_SCHEMA = {
    "type": "object",
    "properties": {"voices": {"type": "array", "items": {"type": "object", "properties": {
        "character": _S,
        "identity": {"type": "string", "description": "One vivid sentence: the voice as a casting note, specific to this person."},
        "vocal_age": _S, "presentation": _S,
        "pitch": {"type": "string", "description": "e.g. low, low-middle, middle, high-middle, high"},
        "texture": {"type": "string", "description": "e.g. breathy, husky, smooth, gravelly, bright, thin, rich"},
        "resonance": {"type": "string", "description": "e.g. chest, head, nasal, throaty"},
        "accent": _S, "language": _S,
        "speed": {"type": "string", "description": "slow, measured, brisk, fast"},
        "rhythm": {"type": "string", "description": "pauses, hesitations, clipped phrases..."},
        "articulation": _S, "habits": {"type": "array", "items": _S},
        "personality": {"type": "string", "description": "How the personality is heard in the voice."},
        "emotional_range": {"type": "string", "description": "How this person sounds afraid, angry, tender; what they never do."},
        "samples": {"type": "array", "items": _S, "description": "2-3 of their lines from the script."},
    }, "required": ["character", "identity", "vocal_age", "pitch", "texture", "speed", "rhythm", "personality",
                    "emotional_range"]}}},
    "required": ["voices"],
}
_FIELDS = ("identity", "vocal_age", "presentation", "pitch", "texture", "resonance", "accent", "language", "speed",
           "rhythm", "articulation", "personality", "emotional_range")
_DISTINCT = ("pitch", "texture", "speed", "vocal_age", "presentation")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _t(x, n=300) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()[:n]


def key(name: str) -> str:
    """Profiles are kept by name, so a character keeps their voice in every episode of a series."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


# ------------------------------------------------------------------ the voice bible

def bible_prompt(project: dict, cast: list[dict], scenes: list[dict], story: dict, world_text: str,
                 keep: dict) -> str:
    lines_by = {}
    for s in scenes:
        for d in s.get("dialogue") or []:
            if (d.get("text") or "").strip():
                lines_by.setdefault(key(d.get("character") or ""), []).append(
                    f"\"{_t(d['text'], 140)}\"" + (f" ({d['emotion']})" if d.get("emotion") else ""))
    plan_chars = {key(c.get("name")): c for c in (story.get("characters") or []) + ((story.get("saga") or {}).get("characters") or [])
                  if isinstance(c, dict) and c.get("name")}
    rows = []
    for c in cast:
        if key(c["name"]) in keep:
            continue
        p = plan_chars.get(key(c["name"])) or {}
        bits = [f"{k}: {p[k]}" for k in ("age", "role", "want", "wants", "flaw", "secret", "hides", "speech", "fears_losing",
                                         "inner_conflict", "arc", "history") if p.get(k)]
        rows.append(f"- {c['name']}: {_t(c.get('description'), 300)}"
                    + (f" | {'; '.join(_t(b, 140) for b in bits)}" if bits else "")
                    + (f"\n  Lines: {' '.join(lines_by.get(key(c['name']), [])[:6])}" if lines_by.get(key(c["name"])) else ""))
    others = [f"- {v.get('character')}: {v.get('identity')}" for v in keep.values()]
    rels = "; ".join(f"{r.get('between')}: {r.get('kind') or r.get('state') or ''}" for r in
                     ((story.get("saga") or {}).get("relationships") or story.get("relationships") or [])[:10] if isinstance(r, dict))
    return f"""You are the voice director of a film. Cast a voice for every speaking character from WHO THEY ARE
(their story, personality, background, relationships, fears and how they change), not only their age and gender.

STORY: {_t(project.get('logline') or project.get('title'), 400)}
LANGUAGE OF THE FILM: {project.get('language') or 'English'}
{world_text}{('Relationships: ' + rels + chr(10)) if rels else ''}
CHARACTERS TO CAST:
{chr(10).join(rows) or '(none)'}
{('VOICES ALREADY CAST (keep them; make the new ones clearly different):' + chr(10) + chr(10).join(others) + chr(10)) if others else ''}
RULES:
- Each voice is recognisably different from every other: vary pitch, texture, speed and rhythm, not just gender.
- Accent and pronunciation belong to the character's world and background (never a default American accent unless
  the story calls for it).
- identity: one concrete casting sentence (e.g. "a low, unhurried, slightly husky voice that never rises, even in
  anger"). Describe the voice, never a real actor.
- emotional_range: how THIS person sounds afraid, angry or tender, and what they never do (some people go quiet
  when furious).
- samples: 2-3 of their own lines from above."""


def clean_bible(raw, cast: list[dict]) -> dict:
    """{key(name): profile} for characters of this production only."""
    names = {key(c["name"]): c["name"] for c in cast}
    out = {}
    for v in (raw or {}).get("voices") or [] if isinstance(raw, dict) else []:
        if not isinstance(v, dict) or key(v.get("character")) not in names:
            continue
        prof = {f: _t(v.get(f), 300) for f in _FIELDS if _t(v.get(f))}
        prof["character"] = names[key(v["character"])]
        prof["habits"] = [_t(h, 120) for h in (v.get("habits") or [])[:4] if _t(h)]
        prof["samples"] = [_t(x, 160) for x in (v.get("samples") or [])[:3] if _t(x)]
        prof["at"] = _now()
        if prof.get("identity"):
            out[key(v["character"])] = prof
    return out


def too_similar(voices: dict) -> list[tuple[str, str]]:
    """Pairs of characters whose voices would be hard to tell apart (same pitch, texture and speed)."""
    items = list(voices.values())
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            same = sum(1 for f in _DISTINCT if _t(a.get(f), 40).lower() and _t(a.get(f), 40).lower() == _t(b.get(f), 40).lower())
            if same >= 3:
                out.append((a["character"], b["character"]))
    return out


def merge(old: dict, new: dict) -> dict:
    """Established voices are never replaced by a rebuild; only missing ones are added."""
    out = dict(new)
    out.update(old or {})
    return out


def profile_text(p: dict, short: bool = False) -> str:
    """The voice identity as a prompt phrase."""
    if not p:
        return ""
    if short:
        return "; ".join(x for x in (p.get("identity"), p.get("accent") and f"{p['accent']} accent") if x)[:220]
    bits = [p.get("identity", "")]
    for f, label in (("vocal_age", "sounds"), ("pitch", "pitch"), ("texture", "texture"), ("resonance", "resonance"),
                     ("accent", "accent"), ("speed", "pace"), ("rhythm", "rhythm"), ("articulation", "articulation")):
        if p.get(f):
            bits.append(f"{label} {p[f]}")
    if p.get("habits"):
        bits.append("habits: " + ", ".join(p["habits"]))
    if p.get("casting") and p.get("casting") != p.get("identity"):
        bits.append("base: " + p["casting"].rstrip("."))
    return "; ".join(b for b in bits if b)[:700]


# ------------------------------------------------------------------ performance direction per line

_LOUD = re.compile(r"\b(shout|shouts|shouting|scream|screams|yell|yells|roar|bellow)", re.I)
_SOFT = re.compile(r"\b(whisper|whispers|whispering|murmur|under (his|her|their) breath|softly|quietly|hushed)", re.I)
_STRONG = re.compile(r"\b(hysterical|sobbing|screaming|furious|enraged|terrified|devastated|ecstatic)", re.I)


def direct_line(d: dict, profile: Optional[dict], hides: str = "") -> dict:
    """How one line is performed: the feeling, the volume, how strongly, and the subtext. Restrained unless the
    script asks for more: most real emotion is held back."""
    emotion = _t(d.get("emotion"), 40)
    delivery = _t(d.get("delivery"), 80)
    text = f"{emotion} {delivery} {d.get('text') or ''}"
    volume = "raised" if _LOUD.search(text) else "soft" if _SOFT.search(text) else "conversational"
    intensity = "strong" if _STRONG.search(f"{emotion} {delivery}") or volume == "raised" else \
        "subtle" if not emotion else "held back"
    out = {"emotion": emotion or "neutral", "volume": volume, "intensity": intensity}
    if delivery:
        out["delivery"] = delivery
    if hides:
        out["subtext"] = f"hiding that {hides}; the words cover it"
    return out


def line_phrase(dirn: dict) -> str:
    bits = [dirn["emotion"] if dirn.get("emotion") != "neutral" else "", dirn.get("delivery", ""),
            f"{dirn['volume']} volume" if dirn.get("volume") != "conversational" else "",
            {"held back": "felt but held back, not overplayed", "strong": "full intensity"}.get(dirn.get("intensity"), ""),
            dirn.get("subtext", "")]
    return ", ".join(b for b in bits if b)


# ------------------------------------------------------------------ what each provider gets

def for_provider(provider: str, lines: list[dict], voices: dict, language: str, hides: Optional[dict] = None,
                 narrator=None) -> dict:
    """The dialogue and voice direction in the form this provider can take. Returns {text, limitations, speakers}.
    Every line keeps its exact words and speaker; what can't be sent is listed, never pretended."""
    caps = CAPS[provider]
    hides = hides or {}
    narrator = narrator or (lambda who: who.strip().lower() in ("narrator", "voice-over", "v.o.", "voiceover"))
    short = caps["prompt_chars"] < 4000
    limits = []
    if not caps["persistent_voice"]:
        limits.append(f"{caps['label']} has no voice ID or voice reference: the voice is held by the same description "
                      "each time, so it can drift between clips.")
    if caps["languages"] is not None and (language or "english").lower() not in caps["languages"]:
        limits.append(f"{caps['label']} only speaks {', '.join(sorted(caps['languages']))}.")
    speakers, parts, cast_notes = [], [], {}
    for d in lines:
        who = (d.get("character") or "").strip()
        if not (d.get("text") or "").strip():
            continue
        if not who or narrator(who):
            parts.append(f'NARRATOR (off-screen voice-over; nobody on screen speaks it): "{d["text"].strip()}"')
            continue
        p = voices.get(key(who)) or {}
        if who not in speakers:
            speakers.append(who)
        dirn = direct_line(d, p, hides.get(key(who), ""))
        how = line_phrase(dirn)
        if short:
            cast_notes[who] = profile_text(p, short=True)
            parts.append(f'{who.upper()}{f" ({how})" if how else ""}: "{d["text"].strip()}"')
        else:
            parts.append(f'{who.upper()} (voice: {profile_text(p) or "natural, true to the character"}'
                         f'{"; performance: " + how if how else ""}): "{d["text"].strip()}"')
        if not p:
            limits.append(f"{who} has no voice profile yet: a generic description was used.")
    head = f"The characters speak on camera with natural lip movement, in {language or 'English'}, saying exactly these words, in this order:"
    if short:
        voices_line = " ".join(f"{w}'s voice: {n}." for w, n in cast_notes.items() if n)
        text = f"DIALOGUE in {language or 'English'} (exact words, in order): " + " ".join(parts) + \
            (f" VOICES: {voices_line}" if voices_line else "") + " Lips move with the words. Nobody else speaks."
    else:
        text = head + " " + " ".join(parts) + (" Each character keeps exactly the voice described; their voices are "
                                                "clearly different from each other. Nobody else speaks.")
    if len(speakers) > 1 and caps["multi_speaker"] == "prompt":
        limits.append(f"{caps['label']} is told who speaks each line in words only; speaker mix-ups are possible.")
    return {"provider": provider, "text": text, "limitations": list(dict.fromkeys(limits)), "speakers": speakers}


def video_prompt(provider: str, scene_text: str, dialogue_block: str) -> str:
    """The clip prompt for a provider: the dialogue always survives the provider's length limit (the scene text
    is cut first, never the lines)."""
    limit = CAPS[provider]["prompt_chars"]
    if len(scene_text) + len(dialogue_block) + 1 <= limit:
        return f"{scene_text} {dialogue_block}"
    room = max(200, limit - len(dialogue_block) - 1)
    return f"{scene_text[:room].rsplit(' ', 1)[0]} {dialogue_block}"[:limit]


def tts_style(base_style: str, d: dict, profile: Optional[dict], hides: str = "") -> str:
    """Saved voices: the named prebuilt voice IS the identity (persistent); the style only carries this line's
    performance, kept short (Google advises short styles for a steadier voice)."""
    dirn = direct_line(d, profile, hides)
    bits = [base_style.strip(), line_phrase(dirn)]
    return ", ".join(b for b in bits if b)[:160]


# ------------------------------------------------------------------ checking the audio of a clip

LISTEN_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "index": {"type": "integer"}, "heard": {"type": "boolean"},
            "heard_text": {"type": "string", "description": "What you actually hear for this line (empty if not said)."},
            "speaker_ok": {"type": "boolean", "description": "Does it sound like the voice described for this character?"},
            "emotion_ok": {"type": "boolean"}, "pronunciation_ok": {"type": "boolean"}},
            "required": ["index", "heard", "heard_text", "speaker_ok"]}},
        "language_ok": {"type": "boolean"}, "natural": {"type": "integer", "description": "1-10"},
        "clarity": {"type": "integer", "description": "1-10: can every word be understood over the background?"},
        "extra_speech": {"type": "boolean", "description": "Someone says words that are not in the script."},
        "voices_distinct": {"type": "boolean", "description": "Different characters sound like different people."},
        "notes": _S,
    },
    "required": ["lines", "language_ok", "natural", "clarity", "extra_speech"],
}


def listen_prompt(lines: list[dict], voices: dict, language: str) -> str:
    rows = "\n".join(f"{i}. {d.get('character')}: \"{d.get('text')}\" — expected voice: "
                     f"{profile_text(voices.get(key(d.get('character') or ''), {}), short=True) or 'any'}"
                     f"; feeling: {d.get('emotion') or 'neutral'}" for i, d in enumerate(lines))
    return f"""Listen to this clip's audio. It should contain exactly these lines in {language}, in order:
{rows}

For each line: was it said (heard), what you actually hear, whether it sounds like the expected voice, whether the
feeling fits, and whether the words are pronounced correctly. Then: is the language right, how natural (1-10) and
clear (1-10) is the speech, does anyone say words that aren't in the script, and do different characters sound
like different people. Only report what you can hear; if unsure, say so in notes."""


def clean_listen(raw, n_lines: int) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    lines = []
    for x in raw.get("lines") or []:
        if isinstance(x, dict) and isinstance(x.get("index"), int) and 0 <= x["index"] < n_lines:
            lines.append({"index": x["index"], "heard": bool(x.get("heard")), "heard_text": _t(x.get("heard_text"), 200),
                          "speaker_ok": x.get("speaker_ok") is not False, "emotion_ok": x.get("emotion_ok") is not False,
                          "pronunciation_ok": x.get("pronunciation_ok") is not False})
    num = lambda k: max(1, min(10, int(raw[k]))) if isinstance(raw.get(k), (int, float)) and not isinstance(raw.get(k), bool) else None
    return {"lines": lines, "language_ok": raw.get("language_ok") is not False, "natural": num("natural"),
            "clarity": num("clarity"), "extra_speech": bool(raw.get("extra_speech")),
            "voices_distinct": raw.get("voices_distinct") is not False, "notes": _t(raw.get("notes"), 300)}


def verdict(tech: dict, heard: Optional[dict], lines: list[dict]) -> dict:
    """The clip's voice check: passed / needs_work / failed / unchecked, with the exact lines that are wrong."""
    problems = []
    if not tech.get("has_audio"):
        problems.append({"kind": "no_audio", "problem": "The clip has no sound at all.", "lines": []})
    elif tech.get("silent"):
        problems.append({"kind": "silent", "problem": "The clip's sound is (almost) silent.", "lines": []})
    if tech.get("clipping"):
        problems.append({"kind": "clipping", "problem": "The sound is distorted (clipping).", "lines": []})
    if heard is None:
        status = "failed" if problems else "unchecked"
        return {"status": status, "problems": problems, "technical": tech, "listened": False}
    by = {x["index"]: x for x in heard["lines"]}
    missing = [i for i in range(len(lines)) if not by.get(i, {}).get("heard")]
    if missing:
        problems.append({"kind": "missing_lines", "problem": "Not said: " + "; ".join(
            f"{lines[i].get('character')}: \"{_t(lines[i].get('text'), 80)}\"" for i in missing[:4]), "lines": missing})
    wrong = [i for i, x in by.items() if x["heard"] and not x["speaker_ok"]]
    if wrong:
        problems.append({"kind": "wrong_voice", "problem": "Doesn't sound like the character: " + ", ".join(
            f"line {i + 1} ({lines[i].get('character')})" for i in wrong if i < len(lines)), "lines": wrong})
    if not heard["language_ok"]:
        problems.append({"kind": "language", "problem": "Spoken in the wrong language.", "lines": []})
    if heard["extra_speech"]:
        problems.append({"kind": "extra_speech", "problem": "Someone says words that aren't in the script.", "lines": []})
    if not heard.get("voices_distinct", True):
        problems.append({"kind": "same_voice", "problem": "Different characters sound like the same person.", "lines": []})
    if (heard.get("clarity") or 10) < 6:
        problems.append({"kind": "clarity", "problem": f"Hard to understand (clarity {heard['clarity']}/10).", "lines": []})
    if (heard.get("natural") or 10) < 5:
        problems.append({"kind": "unnatural", "problem": f"Sounds unnatural ({heard['natural']}/10).", "lines": []})
    soft = [i for i, x in by.items() if x["heard"] and (not x["emotion_ok"] or not x["pronunciation_ok"])]
    if soft:
        problems.append({"kind": "performance", "problem": "Feeling or pronunciation off on line(s) " +
                         ", ".join(str(i + 1) for i in soft), "lines": soft})
    hard = {"no_audio", "silent", "missing_lines", "wrong_voice", "language", "extra_speech"}
    status = "failed" if any(p["kind"] in hard for p in problems) else "needs_work" if problems else "passed"
    return {"status": status, "problems": problems, "technical": tech, "listened": True,
            "scores": {"natural": heard.get("natural"), "clarity": heard.get("clarity")}}


def summary_lines(voices: dict, providers: list[str]) -> list[str]:
    if not voices:
        return ["VOICE BIBLE: not made yet (it's made when the script is broken into scenes, or propose direct_voices)."]
    out = ["VOICE BIBLE (each character's permanent voice; performance changes per scene, identity doesn't):"]
    for v in list(voices.values())[:16]:
        out.append(f"- {v['character']}: {profile_text(v)}; personality in the voice: {v.get('personality', '')}; "
                   f"emotional range: {v.get('emotional_range', '')}")
    for pr in providers:
        c = CAPS.get(pr)
        if c:
            out.append(f"  {c['label']}: native audio {'yes' if c['native_audio'] else 'no'}; voice ID or reference "
                       f"{'yes' if c['persistent_voice'] or c['voice_reference'] else 'NO (description only)'}; emotion via "
                       f"{c['emotion']}; lip sync {c['lip_sync']}; languages {'any' if c['languages'] is None else ', '.join(sorted(c['languages']))}")
    return out
