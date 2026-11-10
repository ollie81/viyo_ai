"""Original background scores for Viyo Creator Studio.

Uses Google's Lyria 3.5 Interactions API. The generated track is instrumental,
stored as a private project asset, and looped/ducked by studio_render.
"""
from __future__ import annotations

import base64
import os
from typing import Optional

import httpx

MODEL = "lyria-3.5"
COST_PER_SONG_USD = 0.08
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"
SONG_SECONDS = 120  # one Lyria song is about two minutes; longer films loop it


class MusicGenerationError(RuntimeError):
    def __init__(self, message: str, retry: bool = False):
        super().__init__(message)
        self.retry = retry


def wanted(project: dict) -> bool:
    """Make a score only when the creator has no track and hasn't turned it off."""
    return not project.get("music_path") and project.get("auto_music") is not False


def music_prompt(project: dict, film_seconds: float = 0) -> str:
    genre = (project.get("genre") or "cinematic drama").strip()[:80]
    logline = (project.get("logline") or project.get("title") or "an emotional story").strip()[:500]
    style = (project.get("visual_style") or "cinematic and emotionally expressive").strip()[:160]
    return (
        "Create an original, instrumental-only cinematic background score for a film. "
        "No vocals, no lyrics, no speech, and no sound effects. "
        f"Genre: {genre}. Story mood and context: {logline}. Visual atmosphere: {style}. "
        "Use an emotionally coherent instrumental arrangement with a memorable but subtle "
        "original motif and gentle development. "
        + ("End so the music flows smoothly back into its opening, because it will repeat under a longer film. "
           if film_seconds > SONG_SECONDS else "Give it a natural ending. ")
        + "Keep the music supportive "
        "rather than overpowering dialogue. Do not imitate any existing song, artist, film score, "
        "or recognizable melody. Create approximately two minutes of music."
    )


def _audio_from_response(payload: dict) -> bytes:
    """Accept the documented convenience output_audio shape and raw REST steps."""
    candidates = []
    output = payload.get("output_audio")
    if isinstance(output, dict):
        candidates.append(output.get("data"))
    for step in payload.get("steps") or []:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for block in step.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "audio":
                candidates.append(block.get("data"))
    for encoded in candidates:
        if isinstance(encoded, str) and encoded:
            try:
                audio = base64.b64decode(encoded, validate=True)
            except Exception:
                continue
            if len(audio) > 1024:
                return audio
    raise MusicGenerationError("The music service returned no usable audio.")


def generate(project: dict, film_seconds: float = 0, api_key: Optional[str] = None) -> bytes:
    """Generate one original instrumental MP3. Never silently substitute silence."""
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise MusicGenerationError("Background music is not configured yet.")
    try:
        response = httpx.post(
            API_URL,
            headers={"x-goog-api-key": key, "Content-Type": "application/json", "Api-Revision": API_REVISION},
            json={"model": MODEL, "input": music_prompt(project, film_seconds)},
            timeout=httpx.Timeout(180.0, connect=15.0),
        )
    except httpx.TimeoutException as exc:
        raise MusicGenerationError("The music service timed out.", retry=True) from exc
    except httpx.HTTPError as exc:
        raise MusicGenerationError("The music service could not be reached.", retry=True) from exc

    if response.status_code in (408, 429, 500, 502, 503, 504):
        raise MusicGenerationError("The music service is busy. Trying again shortly.", retry=True)
    if response.status_code >= 400:
        # Do not expose provider response bodies or API details to creators.
        raise MusicGenerationError("The music service rejected this request.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise MusicGenerationError("The music service returned an unreadable response.") from exc
    return _audio_from_response(payload)
