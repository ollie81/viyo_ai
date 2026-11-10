"""
Creator Studio AI providers.

Every call returns its output plus a studio_pricing.Cost built from what
the provider actually reported (tokens, images, seconds), so the creator
pays real usage, never a guess. Providers sit behind one small interface
(StudioAI) so a model can be swapped without touching the rest; tests use
FakeStudioAI and spend nothing.

Errors:
  ProviderBusy  - try again later (rate limit, overload, timeout). Retried.
  PolicyBlocked - the provider refused the content. Not charged, not retried.
  ProviderError - anything else. Not charged.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import tempfile
import time
import wave
from typing import Optional, Protocol

import studio_pricing as pricing

THINKING_BUDGET = pricing.THINKING_BUDGET


def drama_safety():
    """Google's filters set for adult fiction: violence, crime, dark themes and
    adult romance go through; only high-severity content is blocked. Google's
    own policy still refuses explicit sexual content and sexual content
    involving minors whatever these settings say."""
    from google.genai import types
    return [types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH) for c in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT, types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT)]

VOICES = [  # Gemini TTS prebuilt voices
    ("Zephyr", "female", "Bright"), ("Puck", "male", "Upbeat"), ("Charon", "male", "Informative"),
    ("Kore", "female", "Firm"), ("Fenrir", "male", "Excitable"), ("Leda", "female", "Youthful"),
    ("Orus", "male", "Firm"), ("Aoede", "female", "Breezy"), ("Callirrhoe", "female", "Easy-going"),
    ("Autonoe", "female", "Bright"), ("Enceladus", "male", "Breathy"), ("Iapetus", "male", "Clear"),
    ("Umbriel", "male", "Easy-going"), ("Algieba", "male", "Smooth"), ("Despina", "female", "Smooth"),
    ("Erinome", "female", "Clear"), ("Algenib", "male", "Gravelly"), ("Rasalgethi", "male", "Informative"),
    ("Laomedeia", "female", "Upbeat"), ("Achernar", "female", "Soft"), ("Alnilam", "male", "Firm"),
    ("Schedar", "male", "Even"), ("Gacrux", "female", "Mature"), ("Pulcherrima", "female", "Forward"),
    ("Achird", "male", "Friendly"), ("Zubenelgenubi", "male", "Casual"), ("Vindemiatrix", "female", "Gentle"),
    ("Sadachbia", "male", "Lively"), ("Sadaltager", "male", "Knowledgeable"), ("Sulafat", "female", "Warm"),
]
VOICE_NAMES = {v[0] for v in VOICES}


class ProviderBusy(Exception):
    pass


class PolicyBlocked(Exception):
    pass


class ProviderError(Exception):
    pass


class StudioAI(Protocol):
    def write_script(self, brief: dict) -> tuple[str, pricing.Cost]: ...
    def breakdown(self, brief: dict, script: str) -> tuple[dict, pricing.Cost]: ...
    def image(self, prompt: str, references: list[bytes], aspect_ratio: str, size: str = "1K") -> tuple[bytes, str, pricing.Cost]: ...
    def voice(self, text: str, voice: str, style: str = "") -> tuple[bytes, float, pricing.Cost]: ...
    def video(self, prompt: str, first_frame: Optional[bytes], aspect_ratio: str, seconds: int = 8,
              references: Optional[list] = None) -> tuple[bytes, pricing.Cost]: ...


# ---------------------------------------------------------------- prompts

def script_prompt(b: dict) -> str:
    return f"""You are a professional screenwriter for short-form drama.
Write an original screenplay in {b.get('language') or 'English'}.

Title: {b.get('title')}
Idea: {b.get('logline')}
Genre: {b.get('genre') or 'drama'}
Format: {b.get('kind')} of about {b.get('target_seconds')} seconds, {'vertical 9:16 phone video' if b.get('aspect_ratio') == '9:16' else 'landscape 16:9 film'}.

Rules:
- Standard screenplay format: scene headings (INT./EXT. PLACE - TIME), short action lines, character names in caps before dialogue.
- A strong hook in the first 5 seconds, rising tension, and an ending that makes people want the next episode.
- Keep dialogue natural and short; roughly {max(3, int(b.get('target_seconds') or 90) // 6)} scenes.
- Fictional characters only. No real people, brands or copyrighted characters.
- Return only the screenplay text."""


BREAKDOWN_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "description": {"type": "string", "description": "Visual description: age range, build, hair, face, clothing. No real people."},
            "gender": {"type": "string", "enum": ["female", "male", "other"]},
            "age_category": {"type": "string", "enum": ["infant", "young_child", "older_child", "young_teen", "older_teen",
                                                        "young_adult", "middle_aged", "elderly", "unknown"],
                             "description": "From the story: infant 0-1, young_child 2-7, older_child 8-11, young_teen 12-14, "
                                            "older_teen 15-17, young_adult 18-34, middle_aged 35-59, elderly 60+. "
                                            "'unknown' if the story gives no clue. Never decide from the name alone."},
            "age_years": {"type": "integer", "description": "Only if the story states the age. Otherwise omit."},
            "age_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "personality": {"type": "string", "description": "A few words on personality and how they speak."},
            "role": {"type": "string", "enum": ["lead", "supporting", "minor"]},
        }, "required": ["name", "description", "gender", "age_category", "age_confidence"]}},
        "scenes": {"type": "array", "items": {"type": "object", "properties": {
            "heading": {"type": "string"},
            "visual": {"type": "string", "description": "One-shot image prompt: who is in frame, what they do, setting, lighting, camera framing."},
            "dialogue": {"type": "array", "items": {"type": "object", "properties": {
                "character": {"type": "string"}, "text": {"type": "string"},
                "emotion": {"type": "string", "description": "The feeling of this one line in 1-3 English words "
                            "(e.g. 'angry', 'sad', 'frightened'). Empty if plain."},
                "delivery": {"type": "string", "description": "Optional short performance note in English "
                             "(e.g. 'soft, hesitant'). Empty if not needed."}},
                "required": ["character", "text"]}},
            "duration_seconds": {"type": "number"},
        }, "required": ["heading", "visual", "dialogue", "duration_seconds"]}},
    },
    "required": ["characters", "scenes"],
}


def _plan_cast(b: dict) -> str:
    cast = (b.get("story_bible") or {}).get("characters") or []
    if not cast:
        return ""
    lines = ["The story plan's characters (use these ages and looks; the age decides their voice):"]
    for c in cast[:12]:
        lines.append(f"- {c.get('name')}: age {c.get('age', '?')}; looks: {c.get('look', '')}; speaks: {c.get('speech', '')}")
    return "\n".join(lines) + "\n"


def breakdown_prompt(b: dict, script: str) -> str:
    lang = b.get("language") or "English"
    return _plan_cast(b) + f"""Break this screenplay into a shot list for an AI-made {b.get('kind')} ({b.get('aspect_ratio')}).
Visual style: {b.get('visual_style') or 'cinematic, realistic'}.
The production language is {lang}.
For each scene give: the heading, ONE visual prompt describing a single strong shot (characters by name, action, setting, lighting, framing),
the spoken lines in order (character name exactly as in the character list; use "NARRATOR" for voice-over),
and a duration in seconds (2-12; long enough for its lines).
Describe every character visually so they can be drawn the same way every time.
For every character give their age group from what the story says (stated age, school year, relationships,
how others speak to them). If the story doesn't make it clear, say 'unknown' with low confidence rather than guessing.
For a line whose delivery matters, give a short emotion and, if useful, a short delivery note; leave them empty otherwise.

Language rules:
- Dialogue: copy every spoken line in {lang}, word for word as the screenplay has it. Never translate dialogue into English or any other language.
- Visual prompts and character descriptions: write them in English, because they are instructions for the image and video models. Keep character names exactly as written.

SCREENPLAY:
{script}"""


def image_prompt(visual: str, style: str, aspect_ratio: str, cast_notes: str) -> str:
    return (f"{visual}\n\nVisual style: {style or 'cinematic, realistic, film still'}. "
            f"{'Vertical 9:16 frame.' if aspect_ratio == '9:16' else 'Wide 16:9 frame.'} "
            f"Keep each character exactly as in their reference image. {cast_notes}\n"
            "No text, captions, logos or watermarks in the image.")


# ---------------------------------------------------------------- Google

def _pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    if len(pcm) % 2:
        pcm = pcm[:-1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def as_wav(data: bytes, mime_type: str = "") -> tuple[bytes, float]:
    """(wav bytes, seconds). Gemini 3.8 TTS returns a complete WAV file;
    older models returned raw 16-bit PCM. Never wrap a WAV in a second
    header."""
    import wave
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data)) as w:
            return data, w.getnframes() / float(w.getframerate() or 24000)
    m = re.search(r"rate=(\d+)", mime_type or "")
    rate = int(m.group(1)) if m else 24000
    return _pcm_to_wav(data, rate), len(data) / 2 / rate


def blocked_message(text: str) -> str:
    """Google's refusal in plain words, with what the creator can change."""
    text = text.lower()
    if "real people" in text or "likeness" in text or "real person" in text:
        return ("Google thought this shows a real person. This happens when a character has the same name as a famous "
                "person, or a reference picture looks like a real photo of someone. Try a different name for the "
                "character in Cast, then make the video again.")
    if "filtered" in text or "prohibited use" in text:
        return ("Google's safety check blocked the finished video. Usually it's very graphic violence or injuries shown "
                "up close (blood, wounds, someone tied up or gagged). Try showing it less graphically, for example "
                "the fear on a face instead of the injury.")
    return "Google's safety check refused this scene. Try rewording what we see."


def _classify(e: Exception) -> Exception:
    text = str(e).lower()
    if any(k in text for k in ("429", "resource_exhausted", "quota", "503", "unavailable", "overloaded", "timeout", "deadline")):
        return ProviderBusy(str(e)[:300])
    if any(k in text for k in ("safety", "blocked", "prohibited", "policy", "filtered", "likeness")):
        return PolicyBlocked(blocked_message(text))
    if any(k in text for k in ("404", "not_found", "not found", "is not supported", "invalid_argument")):
        return ProviderError("The AI model rejected the request (" + str(e)[:200] + "). Your credits were returned.")
    return ProviderError(str(e)[:300])


class GoogleStudioAI:
    def __init__(self, api_key: Optional[str] = None):
        from google import genai
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ProviderError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key)

    @staticmethod
    def _text_usage(resp) -> tuple[int, int]:
        u = getattr(resp, "usage_metadata", None)
        if not u:
            return 0, 0
        out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        return u.prompt_token_count or 0, out

    def _text(self, prompt: str, **cfg):
        from google.genai import types
        try:
            resp = self.client.models.generate_content(
                model=pricing.TEXT_MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
                    safety_settings=drama_safety(), **cfg))
        except Exception as e:
            raise _classify(e)
        if not (resp.text or "").strip():
            raise PolicyBlocked("The AI returned nothing for this request (it may have been blocked).")
        return resp

    def write_script(self, brief: dict):
        text, _bible, cost = self.develop_script(brief, [])
        return text, cost

    def develop_script(self, brief: dict, previous: list):
        """The writers' room (studio_story): plan -> draft -> script doctor.
        Returns (screenplay, story bible, cost)."""
        import studio_story

        def text_fn(prompt):
            resp = self._text(prompt)
            i, o = self._text_usage(resp)
            return (resp.text or ""), i, o

        def json_fn(prompt, schema):
            resp = self._text(prompt, response_mime_type="application/json", response_json_schema=schema)
            i, o = self._text_usage(resp)
            try:
                return json.loads(resp.text), i, o
            except (TypeError, ValueError):
                raise ProviderError("The AI returned an unreadable story plan.")

        try:
            text, bible, i, o = studio_story.develop(text_fn, json_fn, brief, previous)
        except ValueError as e:
            raise ProviderError(f"The story couldn't be written: {e}")
        return text, bible, pricing.text_cost(i, o)

    def identity(self, name: str, description: str, context: str = ""):
        """The character's permanent identity profile (studio_identity)."""
        import studio_identity
        resp = self._text(studio_identity.identity_prompt(name, description, context),
                          response_mime_type="application/json", response_json_schema=studio_identity.IDENTITY_SCHEMA)
        try:
            data = json.loads(resp.text)
        except (TypeError, ValueError):
            raise ProviderError("The AI returned an unreadable character profile.")
        i, o = self._text_usage(resp)
        return studio_identity.clean_identity(data), pricing.text_cost(i, o)

    def chat(self, prompt: str):
        """One assistant reply (studio_chat): {reply, proposals, decisions}."""
        import studio_chat
        resp = self._text(prompt, response_mime_type="application/json", response_json_schema=studio_chat.CHAT_SCHEMA)
        try:
            data = json.loads(resp.text)
        except (TypeError, ValueError):
            raise ProviderError("The assistant's answer couldn't be read.")
        if not (data.get("reply") or "").strip():
            raise ProviderError("The assistant returned an empty answer.")
        i, o = self._text_usage(resp)
        return data, pricing.text_cost(i, o)

    def summarize(self, prompt: str):
        """A short plain-text summary (the assistant's memory of older messages)."""
        resp = self._text(prompt)
        i, o = self._text_usage(resp)
        return (resp.text or "").strip(), pricing.text_cost(i, o)

    def extract_elements(self, prompt: str):
        """Recurring places and objects in the script (studio_elements)."""
        import studio_elements
        resp = self._text(prompt, response_mime_type="application/json",
                          response_json_schema=studio_elements.ELEMENTS_SCHEMA)
        try:
            data = json.loads(resp.text)
        except (TypeError, ValueError):
            raise ProviderError("The AI returned an unreadable list of places and objects.")
        i, o = self._text_usage(resp)
        return data, pricing.text_cost(i, o)

    def cover_concept(self, prompt: str):
        """The art director's poster concept (viyo_brand)."""
        import viyo_brand
        resp = self._text(prompt, response_mime_type="application/json",
                          response_json_schema=viyo_brand.COVER_CONCEPT_SCHEMA)
        try:
            data = json.loads(resp.text)
        except (TypeError, ValueError):
            raise ProviderError("The AI returned an unreadable poster concept.")
        i, o = self._text_usage(resp)
        return data, pricing.text_cost(i, o)

    def suggest_outfits(self, prompt: str):
        """Costume suggestions from the screenplay (studio_identity)."""
        import studio_identity
        resp = self._text(prompt, response_mime_type="application/json",
                          response_json_schema=studio_identity.OUTFIT_SUGGEST_SCHEMA)
        try:
            data = json.loads(resp.text)
        except (TypeError, ValueError):
            raise ProviderError("The AI returned unreadable outfit suggestions.")
        i, o = self._text_usage(resp)
        return data.get("outfits") or [], pricing.text_cost(i, o)

    def breakdown(self, brief: dict, script: str):
        resp = self._text(breakdown_prompt(brief, script), response_mime_type="application/json",
                          response_json_schema=BREAKDOWN_SCHEMA)
        try:
            data = json.loads(resp.text)
        except ValueError:
            raise ProviderError("The AI returned an unreadable shot list.")
        i, o = self._text_usage(resp)
        return data, pricing.text_cost(i, o)

    def image(self, prompt: str, references: list[bytes], aspect_ratio: str, size: str = "1K"):
        from google.genai import types
        parts = [types.Part.from_bytes(data=r, mime_type="image/png") for r in references[:4]]
        parts.append(types.Part.from_text(text=prompt))
        try:
            resp = self.client.models.generate_content(
                model=pricing.IMAGE_MODEL, contents=parts,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"],
                                                   safety_settings=drama_safety(),
                                                   image_config=types.ImageConfig(aspect_ratio=aspect_ratio,
                                                                                  image_size=size)))
        except Exception as e:
            raise _classify(e)
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content else []) or []:
                if part.inline_data is not None and part.inline_data.data:
                    u = getattr(resp, "usage_metadata", None)
                    return part.inline_data.data, part.inline_data.mime_type or "image/png", \
                        pricing.image_cost(1, (u.prompt_token_count or 0) if u else 0, size)
        raise PolicyBlocked("No image came back for this prompt. Try rewording the scene.")

    def voice(self, text: str, voice: str, style: str = ""):
        """One line in one prebuilt voice, Gemini 3.8 Flash TTS through the
        Interactions API (request shape from Google's speech docs, 6 Nov
        2026). `style` is one short delivery note; the transcript stays
        verbatim. Not yet verified with a live call."""
        part: dict = {"type": "text", "text": text}
        if style.strip():
            part["annotations"] = [{"type": "speech_metadata", "style": style.strip()[:120]}]
        try:
            inter = self.client.interactions.create(
                model=pricing.TTS_MODEL,
                input=[{"type": "user_input", "content": [part]}],
                response_format={"type": "audio"},
                generation_config={"speech_config": [{"voice": voice}]})
        except Exception as e:
            raise _classify(e)
        out = getattr(inter, "output_audio", None)
        data = getattr(out, "data", None) if out is not None else None
        if not data:
            raise PolicyBlocked("No audio came back for this line.")
        raw = base64.b64decode(data) if isinstance(data, str) else data
        wav, seconds = as_wav(raw, getattr(out, "mime_type", "") or "")
        return wav, seconds, pricing.voice_cost(seconds)

    def video(self, prompt: str, first_frame: Optional[bytes], aspect_ratio: str, seconds: int = 8,
              references: Optional[list] = None):
        """Gemini Omni 1.2 Flash (Interactions API), request shape from
        Google's Omni docs (5 Nov 2026). Not yet verified with a live call:
        it costs real money, so it's tested only with fakes until the
        approved test run."""
        seconds = pricing.video_seconds_for(seconds)
        inp: list = []
        if references:
            # Several images: Omni treats them as references. Google's docs say
            # to describe each image's role in plain words (max 6).
            roles = []
            for i, (label, data) in enumerate(references[:6], start=1):
                inp.append({"type": "image", "data": base64.b64encode(data).decode(), "mime_type": "image/png"})
                roles.append(f"Image {i}: {label}.")
            prompt = " ".join(roles) + "\n\n" + prompt
        elif first_frame:
            inp.append({"type": "image", "data": base64.b64encode(first_frame).decode(), "mime_type": "image/png"})
        inp.append({"type": "text", "text": prompt})
        try:
            inter = self.client.interactions.create(
                model=pricing.VIDEO_MODEL, input=inp,
                response_format={"type": "video", "aspect_ratio": aspect_ratio, "resolution": "720p",
                                 "duration": f"{seconds}s"})
        except Exception as e:
            raise _classify(e)
        out = getattr(inter, "output_video", None)
        data = getattr(out, "data", None) if out is not None else None
        if not data:
            raise PolicyBlocked("No video came back for this scene.")
        video = base64.b64decode(data) if isinstance(data, str) else data
        seconds = _probe_seconds(video) or float(seconds)
        usage = getattr(inter, "usage", None) or getattr(inter, "usage_metadata", None)
        out_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "candidates_token_count", None)
        in_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_token_count", None) or 0
        return video, pricing.video_cost(seconds, out_tokens, in_tokens)


def _probe_seconds(data: bytes) -> Optional[float]:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(data)
        f.flush()
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f.name],
                           capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------- fake (tests)

class FakeStudioAI:
    """Real files (PNG, WAV, MP4) with zero provider cost. For tests and
    for the free local preview; never charges anything real."""

    def __init__(self, fail: Optional[dict] = None):
        self.calls: list[tuple] = []
        self.fail = fail or {}

    def _maybe_fail(self, kind: str):
        err = self.fail.get(kind)
        if err:
            if isinstance(err, list):
                if err:
                    raise err.pop(0)
                return
            raise err

    def develop_script(self, brief, previous):
        self.last_story_previous = list(previous)
        text, cost = self.write_script(brief)
        bible = {"title": brief.get("title"), "logline": brief.get("logline") or "A story.",
                 "central_conflict": "Maya must face the brother she left behind.", "stakes": "Her family",
                 "characters": [{"name": "Maya", "age": "late 20s", "role": "lead", "want": "to be forgiven",
                                 "flaw": "runs from conflict", "speech": "short, guarded", "look": "short black hair, red jacket"},
                                {"name": "Daniel", "age": "30s", "role": "supporting", "want": "the truth",
                                 "flaw": "stubborn", "speech": "blunt", "look": "beard, grey hoodie"}],
                 "beats": [{"purpose": "hook", "what_happens": "Maya returns", "emotional_turn": "calm -> shock", "seconds": 5}],
                 "ending": "Daniel calls her name.", "scores": {"hook": 6}, "fixes": ["sharper hook"],
                 "episode_summary": f"Episode about {brief.get('title')}.", "open_threads": ["Why did Maya leave?"],
                 "rewrite_used": True}
        return text, bible, cost

    def write_script(self, brief):
        self.calls.append(("script", brief.get("title")))
        self._maybe_fail("script")
        text = (f"INT. ROOFTOP - NIGHT\n\nMAYA (20s) stares at the city.\n\nMAYA\nThey said I'd never come back.\n\n"
                f"EXT. STREET - NIGHT\n\nDANIEL runs toward her.\n\nDANIEL\nMaya! Wait!\n")
        return text, pricing.text_cost(900, 1500)

    def breakdown(self, brief, script):
        self.calls.append(("breakdown",))
        self.last_breakdown_brief = dict(brief)
        self.last_breakdown_prompt = breakdown_prompt(brief, script)
        if getattr(self, "breakdown_result", None) is not None:
            self._maybe_fail("breakdown")
            return self.breakdown_result, pricing.text_cost(1400, 1800)
        self._maybe_fail("breakdown")
        return {
            "characters": [{"name": "Maya", "description": "Woman in her 20s, short black hair, red jacket", "gender": "female"},
                           {"name": "Daniel", "description": "Man in his 20s, beard, grey hoodie", "gender": "male"}],
            "scenes": [
                {"heading": "INT. ROOFTOP - NIGHT", "visual": "Maya on a rooftop at night, city lights behind her",
                 "dialogue": [{"character": "Maya", "text": "They said I'd never come back."}], "duration_seconds": 4},
                {"heading": "EXT. STREET - NIGHT", "visual": "Daniel running down a wet street",
                 "dialogue": [{"character": "Daniel", "text": "Maya! Wait!"}], "duration_seconds": 3},
            ]}, pricing.text_cost(1400, 1800)

    def chat(self, prompt):
        self.calls.append(("chat",))
        self.chat_prompts = getattr(self, "chat_prompts", []) + [prompt]
        self._maybe_fail("chat")
        if getattr(self, "chat_result", None) is not None:
            r = self.chat_result
            return (r(prompt) if callable(r) else r), pricing.text_cost(3000, 400)
        return {"reply": "Love it. What if he finds out on the day he's evicted? One question: who raised him?",
                "proposals": [{"type": "update_story", "title": "Set the idea",
                               "summary": "Title 'Hidden Heir' and this logline.",
                               "data": {"title": "Hidden Heir", "logline": "An evicted courier learns he is a billionaire's son.",
                                        "genre": "Drama"}}],
                "decisions": []}, pricing.text_cost(3000, 400)

    def summarize(self, prompt):
        self.calls.append(("summarize",))
        self.summary_prompts = getattr(self, "summary_prompts", []) + [prompt]
        return "Summary: the creator is developing a rags-to-riches drama.", pricing.text_cost(1500, 120)

    def extract_elements(self, prompt):
        self.calls.append(("elements",))
        self.last_elements_prompt = prompt
        self._maybe_fail("elements")
        return getattr(self, "elements_result", None) or {
            "locations": [{"name": "Rooftop", "description": "flat concrete roof, rusty railing, city lights below",
                           "time_of_day": "night", "mood": "tense", "scenes": [1]}],
            "props": [{"name": "Maya's black leather bag", "description": "small black leather shoulder bag, gold clasp",
                       "owner": "Maya", "scenes": [1]}],
        }, pricing.text_cost(1600, 500)

    def cover_concept(self, prompt):
        self.calls.append(("cover_concept",))
        self.last_concept_prompt = prompt
        self._maybe_fail("cover_concept")
        return getattr(self, "concept_result", None) or {
            "focus": "characters", "characters": ["Maya", "Daniel"], "composition": "Maya in front, Daniel behind her",
            "setting": "a rooftop at night in the rain", "colour_and_light": "neon blue and red",
            "title_style": "sharp silver letters"}, pricing.text_cost(1200, 300)

    def suggest_outfits(self, prompt):
        self.calls.append(("outfits",))
        self.last_outfit_prompt = prompt
        self._maybe_fail("outfits")
        return getattr(self, "outfit_result", None) or [
            {"character": "Maya", "name": "Wedding", "clothing": "long ivory wedding dress", "shoes": "satin heels",
             "accessories": "pearl earrings", "makeup": "soft bridal makeup", "scenes": [2], "reason": "Her wedding day."},
            {"character": "Nobody", "name": "Ghost", "clothing": "sheet", "scenes": [1]},
        ], pricing.text_cost(1500, 700)

    def identity(self, name, description, context=""):
        self.calls.append(("identity", name))
        self._maybe_fail("identity")
        import studio_identity
        return studio_identity.clean_identity({
            "basic": {"age": "late 20s", "height_cm": 168, "build": "slim", "skin_tone": "warm medium brown",
                      "posture": "upright, guarded"},
            "face": {"face_shape": "oval", "eyes": "almond, dark brown, slightly wide-set", "nose": "straight, narrow",
                     "hair": "short black hair, straight", "marks": "small scar above the left eyebrow"},
            "body": {"proportions": "long legs", "shoulders": "narrow", "physique": "lean"},
            "default_outfit": "red jacket, black jeans, white sneakers",
            "guessed": ["basic.height_cm", "face.eyes", "body.shoulders"]}), pricing.text_cost(900, 600)

    def image(self, prompt, references, aspect_ratio, size="1K"):
        self.calls.append(("image", aspect_ratio, len(references)))
        self.image_log = getattr(self, "image_log", []) + [{"prompt": prompt, "refs": len(references)}]
        self.last_image_size = size
        self._maybe_fail("image")
        from PIL import Image
        w, h = (540, 960) if aspect_ratio == "9:16" else (960, 540)
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (40, 30, 90)).save(buf, "PNG")
        return buf.getvalue(), "image/png", pricing.image_cost(1, 300, size)

    def voice(self, text, voice, style=""):
        self.calls.append(("voice", voice))
        self.voice_log = getattr(self, "voice_log", []) + [{"text": text, "voice": voice, "style": style}]
        self._maybe_fail("voice")
        seconds = max(0.8, len(text.split()) / 2.4)
        rate = 24000
        pcm = b"\x00\x00" * int(rate * seconds)
        return _pcm_to_wav(pcm, rate), seconds, pricing.voice_cost(seconds)

    def video(self, prompt, first_frame, aspect_ratio, seconds=8, references=None):
        self.calls.append(("video", aspect_ratio, seconds))
        self.last_video_prompt = prompt
        self.last_video_model = pricing.VIDEO_MODEL
        self.last_video_refs = [label for label, _data in (references or [])]
        self._maybe_fail("video")
        size = "720x1280" if aspect_ratio == "9:16" else "1280x720"
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "v.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24:duration=3",
                            "-f", "lavfi", "-i", "sine=f=300:d=3", "-c:v", "libx264", "-preset", "ultrafast",
                            "-c:a", "aac", "-shortest", out], check=True)
            data = open(out, "rb").read()
        return data, pricing.video_cost(3.0, 3 * 5792, 1000)


# Viyo makes adult fiction: violence, crime, death, dark themes and adult
# romance are allowed. Only these are refused (OpenAI moderation categories).
BLOCKED_ALWAYS = ("sexual_minors", "self_harm_instructions")
BLOCKED_IF_CERTAIN = {"illicit_violent": 0.9}   # real how-to for weapons or violent crime


def story_allowed(result) -> bool:
    cats, scores = result.categories, result.category_scores
    if any(getattr(cats, c, False) for c in BLOCKED_ALWAYS):
        return False
    return not any((getattr(scores, c, 0) or 0) >= limit for c, limit in BLOCKED_IF_CERTAIN.items())


def moderate_text(text: str) -> bool:
    """True if the text is acceptable for a Viyo drama (see BLOCKED_ALWAYS).
    OpenAI's moderation endpoint is free; without a key (local runs) everything passes."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or key.startswith("test") or key == "x" or not text.strip():
        return True
    try:
        from openai import OpenAI
        r = OpenAI(api_key=key, timeout=20, max_retries=1).moderations.create(
            model="omni-moderation-latest", input=text[:20000])
        return story_allowed(r.results[0])
    except Exception as e:
        print(f"[WARN] moderation unavailable, allowing: {e}")
        return True


def omni_clip(prompt: str, image_url: Optional[str], aspect_ratio: str, seconds: int = 8) -> tuple[bytes, float, int]:
    """For the admin Studio and Ads Studio, which used the Veo 3.1 preview
    models Google shut down on 22 Oct 2026: one Gemini Omni 1.2 Flash clip.
    Returns (mp4 bytes, seconds, cost in US cents). Raises ProviderBusy /
    PolicyBlocked / ProviderError."""
    import requests
    frame = None
    if image_url:
        try:
            r = requests.get(image_url, timeout=30)
            r.raise_for_status()
            frame = r.content
        except Exception as e:
            raise ProviderError(f"could not fetch the starting image: {e}")
    data, cost = GoogleStudioAI().video(prompt, frame, "16:9" if aspect_ratio == "16:9" else "9:16", seconds)
    return data, float(cost.units.get("video_seconds") or 8.0), max(1, round(cost.provider_cost_usd * 100))
