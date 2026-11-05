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
    def voice(self, text: str, voice: str) -> tuple[bytes, float, pricing.Cost]: ...
    def video(self, prompt: str, first_frame: Optional[bytes], aspect_ratio: str, seconds: int = 8) -> tuple[bytes, pricing.Cost]: ...


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
        }, "required": ["name", "description", "gender"]}},
        "scenes": {"type": "array", "items": {"type": "object", "properties": {
            "heading": {"type": "string"},
            "visual": {"type": "string", "description": "One-shot image prompt: who is in frame, what they do, setting, lighting, camera framing."},
            "dialogue": {"type": "array", "items": {"type": "object", "properties": {
                "character": {"type": "string"}, "text": {"type": "string"}}, "required": ["character", "text"]}},
            "duration_seconds": {"type": "number"},
        }, "required": ["heading", "visual", "dialogue", "duration_seconds"]}},
    },
    "required": ["characters", "scenes"],
}


def breakdown_prompt(b: dict, script: str) -> str:
    lang = b.get("language") or "English"
    return f"""Break this screenplay into a shot list for an AI-made {b.get('kind')} ({b.get('aspect_ratio')}).
Visual style: {b.get('visual_style') or 'cinematic, realistic'}.
The production language is {lang}.
For each scene give: the heading, ONE visual prompt describing a single strong shot (characters by name, action, setting, lighting, framing),
the spoken lines in order (character name exactly as in the character list; use "NARRATOR" for voice-over),
and a duration in seconds (2-12; long enough for its lines).
Describe every character visually so they can be drawn the same way every time.

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


def _classify(e: Exception) -> Exception:
    text = str(e).lower()
    if any(k in text for k in ("429", "resource_exhausted", "quota", "503", "unavailable", "overloaded", "timeout", "deadline")):
        return ProviderBusy(str(e)[:300])
    if any(k in text for k in ("safety", "blocked", "prohibited", "policy")):
        return PolicyBlocked(str(e)[:300])
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
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET), **cfg))
        except Exception as e:
            raise _classify(e)
        if not (resp.text or "").strip():
            raise PolicyBlocked("The AI returned nothing for this request (it may have been blocked).")
        return resp

    def write_script(self, brief: dict):
        resp = self._text(script_prompt(brief))
        i, o = self._text_usage(resp)
        return resp.text.strip(), pricing.text_cost(i, o)

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

    def voice(self, text: str, voice: str):
        from google.genai import types
        try:
            resp = self.client.models.generate_content(
                model=pricing.TTS_MODEL, contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))))
        except Exception as e:
            raise _classify(e)
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content else []) or []:
                if part.inline_data is not None and part.inline_data.data:
                    m = re.search(r"rate=(\d+)", part.inline_data.mime_type or "")
                    rate = int(m.group(1)) if m else 24000
                    pcm = part.inline_data.data
                    seconds = len(pcm) / 2 / rate
                    return _pcm_to_wav(pcm, rate), seconds, pricing.voice_cost(seconds)
        raise PolicyBlocked("No audio came back for this line.")

    def video(self, prompt: str, first_frame: Optional[bytes], aspect_ratio: str, seconds: int = 8):
        """Gemini Omni 1.2 Flash (Interactions API), request shape from
        Google's Omni docs (5 Nov 2026). Not yet verified with a live call:
        it costs real money, so it's tested only with fakes until the
        approved test run."""
        seconds = pricing.video_seconds_for(seconds)
        inp: list = []
        if first_frame:
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

    def image(self, prompt, references, aspect_ratio, size="1K"):
        self.calls.append(("image", aspect_ratio, len(references)))
        self.last_image_size = size
        self._maybe_fail("image")
        from PIL import Image
        w, h = (540, 960) if aspect_ratio == "9:16" else (960, 540)
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (40, 30, 90)).save(buf, "PNG")
        return buf.getvalue(), "image/png", pricing.image_cost(1, 300, size)

    def voice(self, text, voice):
        self.calls.append(("voice", voice))
        self._maybe_fail("voice")
        seconds = max(0.8, len(text.split()) / 2.4)
        rate = 24000
        pcm = b"\x00\x00" * int(rate * seconds)
        return _pcm_to_wav(pcm, rate), seconds, pricing.voice_cost(seconds)

    def video(self, prompt, first_frame, aspect_ratio, seconds=8):
        self.calls.append(("video", aspect_ratio, seconds))
        self.last_video_model = pricing.VIDEO_MODEL
        self._maybe_fail("video")
        size = "720x1280" if aspect_ratio == "9:16" else "1280x720"
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "v.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24:duration=3",
                            "-f", "lavfi", "-i", "sine=f=300:d=3", "-c:v", "libx264", "-preset", "ultrafast",
                            "-c:a", "aac", "-shortest", out], check=True)
            data = open(out, "rb").read()
        return data, pricing.video_cost(3.0, 3 * 5792, 1000)


def moderate_text(text: str) -> bool:
    """True if the text is acceptable. OpenAI's moderation endpoint is free;
    without a key (local runs) everything passes."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or key.startswith("test") or key == "x" or not text.strip():
        return True
    try:
        from openai import OpenAI
        r = OpenAI(api_key=key, timeout=20, max_retries=1).moderations.create(
            model="omni-moderation-latest", input=text[:20000])
        return not r.results[0].flagged
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
