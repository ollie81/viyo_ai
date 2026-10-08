"""
Viyo Studio: Script -> Cast & Locations (Phase 1), Voices (Phase 2),
and Scenes (Phase 3).

Phase 1 turns a pasted episode/series script into a structured cast
list — characters and locations, each with a Gemini-generated
reference image — that gets saved onto a `series` row and reused by
every later episode. Phase 2 assigns each saved character a
text-to-speech voice. Phase 3 splits one specific episode's script
into scenes and sends those same Phase 1 reference images back to
Gemini when generating each scene's image, so faces and places stay
consistent across episodes, then generates each dialogue line's audio
using that character's Phase 2 voice.

There is no separate "episode" entity anywhere in this codebase — an
episode is just a `posts` row with `series_id` + `episode_number` set
(see episodes.py's own docstring). Phase 3 follows that same
convention rather than inventing an episode table: `series_scenes` is
keyed by `(series_id, episode_number)` directly, not by a foreign key
to anything episode-shaped.

Admin-only for now, gated the same way every other admin surface in
this app already is (analytics.py, moderation.py): a shared secret in
the X-Admin-Key header, not a per-user `profiles` role column — there
is no admin-role concept anywhere else in this codebase, and inventing
one here would mean a schema change this codebase has no
migration-runner access to make (see coins.py's own docstring on the
same constraint). `_require_admin` is deliberately the ONLY gate
between "nobody" and "full access" today; when creators get enabled
later, the intended shape is a second, narrower dependency
(`_require_studio_access`, checking a `profiles.studio_enabled` flag
instead) sitting next to this one, with `_spend_today_cents` already
accepting an optional `user_id` filter so a per-user cap is a one-line
change at that point, not a rewrite.

Every Gemini call (script analysis, character portraits, location
images) is logged to `studio_api_costs` with its actual or estimated
USD cost, and `_check_daily_cap` is called before every one of them —
once today's logged spend would cross STUDIO_DAILY_CAP_USD, every
further call 429s until the next UTC day, no exceptions. This is a
real-money guardrail on an API that bills per call, not a coin-gating
feature, so it fails closed (blocks generation) rather than open.

New tables this file depends on (`series_characters`, `series_locations`,
`studio_api_costs`, and Phase 3's `series_scenes` / `series_scene_lines`)
are NOT created by this codebase — same "no migration-runner access"
constraint as everywhere else here. The SQL to create them (with RLS
enabled and zero public policies, since every read/write here goes
through this file's own service-role Supabase client, never the
Flutter app's anon-key client directly) was handed over separately
and must be run by hand in the Supabase SQL editor before this router
will do anything but fail with a clear 500 on the first real table
access.
"""
import datetime
import io
import os
import re
import subprocess
import tempfile
import time
import uuid
import wave
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1/admin/studio", tags=["studio"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# $3/day by default — a single config value, not hardcoded in the
# Flutter app (the app only ever reads cap_usd_cents back from
# GET /spend-today). Override in Railway (STUDIO_DAILY_CAP_USD, in
# whole dollars) to change the limit without a new app release.
STUDIO_DAILY_CAP_USD_CENTS = int(float(os.environ.get("STUDIO_DAILY_CAP_USD", "3")) * 100)

# The admin's timezone (Africa/Kigali, UTC+2, no DST) — "today" for the
# spend cap resets at midnight here, not at UTC midnight. ZoneInfo reads
# from the `tzdata` package (requirements.txt) rather than assuming the
# container image ships its own IANA tz database, which slim Python
# Docker images often don't.
STUDIO_TIMEZONE = ZoneInfo("Africa/Kigali")

STUDIO_IMAGES_BUCKET = "studio-images"
STUDIO_AUDIO_BUCKET = "studio-audio"
STUDIO_VIDEOS_BUCKET = "studio-videos"

# Phase 4 pushes the assembled MP4 to Bunny Stream server-side, so it
# reuses bunny_stream.py's own env vars, config check, and deterministic
# playback-URL builder rather than redefining them here — one source of
# truth for how this backend talks to Bunny, even though Phase 4's
# upload mechanism (a direct PUT of bytes already on disk) differs from
# that file's own client-facing TUS credential flow (see
# _upload_finished_video_to_bunny's docstring below for why).
from bunny_stream import (
    BUNNY_STREAM_API_KEY,
    BUNNY_STREAM_LIBRARY_ID,
    _BUNNY_API_BASE,
    _configured as _bunny_configured,
    _playback_url as _bunny_playback_url,
)

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

_gemini_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_TEXT_MODEL = "gemini-3.8-flash"
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
# Gemini's flagship TTS model, per Google's own Gemini API docs
# (ai.google.dev/gemini-api/docs/models) — the "flash" variant
# specifically, since voice previews are short and cheap is what
# matters here, not the extra quality a pro-tier TTS model would add.
GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"

# The 2.5 Flash generation these constants originally used was
# retired ("no longer available to new users", surfaced as a live
# 404 from the API itself) — bumped to the 3.8/3.1 generation above
# accordingly. Pricing below is unverified against the new models;
# re-check https://ai.google.dev/gemini-api/docs/pricing before
# trusting it for real budgeting. This is best-effort cost *tracking*
# against the daily cap, not a billing-accurate invoice.
GEMINI_TEXT_INPUT_USD_PER_1M_TOKENS = 0.30
GEMINI_TEXT_OUTPUT_USD_PER_1M_TOKENS = 2.50

# Gemini's image model bills image output as a fixed ~1290-token chunk
# per image rather than exposing a separate per-image price, so this
# is a flat per-call estimate (~1290 tokens x the output rate above,
# rounded up) rather than computed from usage_metadata like the text
# calls below are.
GEMINI_IMAGE_COST_USD_CENTS = 4

# Same reasoning as the image cost above: Gemini's TTS pricing isn't
# broken out in official per-token docs in a way that's reliably
# computable from usage_metadata the way plain text is, and a voice
# preview is always a short one- or two-sentence sample — so this is a
# flat best-effort estimate per preview call, not a computed cost.
GEMINI_TTS_PREVIEW_COST_USD_CENTS = 1

# A dialogue line is the same kind of short TTS call as a voice
# preview — kept as its own constant (same value today) so Phase 3's
# per-line cost can be tuned independently of Phase 2's preview cost
# later without the two meanings colliding.
GEMINI_TTS_LINE_COST_USD_CENTS = 1

# Optional per-scene upgrade from the default Ken Burns (still image +
# zoom/pan) to a real Veo-generated video clip. Lite is the only tier
# that fits a whole episode inside a $3/day cap — Standard is $0.40/s
# (a single 8s clip alone blows the entire cap), Fast is $0.10/s.
# Verified against ai.google.dev/gemini-api/docs/pricing's published
# per-second rates (720p); re-check there before trusting this for
# real budgeting, same caveat as the text/image costs above.
VEO_MODEL = "veo-3.1-lite-generate-preview"
VEO_PRICE_PER_SEC_USD_CENTS = 5
# Veo only accepts 4, 6 or 8 second clips — no arbitrary duration.
VEO_ALLOWED_DURATIONS = (4, 6, 8)
# Generation is an async operation polled to completion rather than a
# normal request/response call — Google's own docs give an 11s-to-6min
# latency range. Polling inside the HTTP request (rather than a proper
# job queue Studio has no infrastructure for) risks hitting Railway's
# own request timeout on a slow render, so this caps how long a single
# call will wait before giving up — comfortably under typical platform
# timeouts, with the real operation still finishing server-side on
# Google's end either way.
VEO_MAX_POLL_SECONDS = 280
VEO_POLL_INTERVAL_SECONDS = 10

# Phase 4 assembly is pure ffmpeg/Bunny work — no Gemini calls, nothing
# to log against the daily cap or STUDIO_DAILY_CAP_USD.
_FFMPEG_FPS = 30
_FFMPEG_RESOLUTION = "1080x1920"
# How long a scene with no dialogue lines holds on screen — a plain
# establishing shot still needs *some* duration to not flash by instantly.
_SCENE_SILENCE_SECONDS = 3.0
# Installed via the Dockerfile's `apt-get install ... fonts-dejavu-core`
# line (same package repurpose.py's own burned-in text already depends
# on — see QUOTE_CARD_FONT_PATH there).
_CAPTION_FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Ken Burns zoom range for every still-image scene — 1.0 (no crop) up
# to this cap, never further and never reset back to 1.0 mid-scene, so
# a scene is always visibly, continuously moving (see _render_scene_clip's
# own docstring for why this is never skipped).
_ZOOM_MIN = 1.0
_ZOOM_MAX = 1.12

# Burned-in dialogue captions (see _build_caption_filters). Deliberately
# real pixel values against this pipeline's own known 1080x1920 output,
# not libass/ASS units — see _build_caption_filters' own docstring for
# why: an ASS MarginV computed from these same numbers silently rendered
# off-screen, confirmed empirically before switching to drawtext.
_CAPTION_FONT_SIZE = 52
_CAPTION_Y_FRACTION = 0.72  # "lower third" — about 72% down the frame
_CAPTION_LINE_HEIGHT = _CAPTION_FONT_SIZE + 18
_CAPTION_MAX_CHARS_PER_LINE = 26  # calibrated against this exact font/size — see _wrap_caption_lines

# Freeze-frame outro (replaces the old separate end card) — holds the
# final scene's last frame instead of cutting to a different, unrelated
# card. No bundled emoji in _OUTRO_TEXT: tried rendering 🔥 via drawtext
# against both DejaVu Sans Bold (no emoji glyphs — renders as a visible
# tofu/missing-glyph box) and the system's own NotoColorEmoji (a color
# bitmap font drawtext's FreeType-based text renderer can't load at all
# — "Error initializing filters") before dropping it from the actual
# burned-in video text rather than ship a broken-looking glyph.
_OUTRO_SECONDS = 1.5
_OUTRO_TEXT = "PART 2 ON VIYO"
_OUTRO_FONT_SIZE = 84

# Gemini's prebuilt TTS voice catalog, each tagged with a gender/age/
# description for the auto-assignment heuristic below. The tags are
# this codebase's own best-effort characterization of each voice (not
# pulled live from Google) — double check against
# https://ai.google.dev/gemini-api/docs/speech-generation before
# relying on them for anything beyond a reasonable starting guess; the
# admin can always override via POST /character/{id}/voice.
GEMINI_VOICES: list[dict] = [
    {"name": "Zephyr", "gender": "female", "age": "young", "description": "Bright"},
    {"name": "Puck", "gender": "male", "age": "young", "description": "Upbeat"},
    {"name": "Charon", "gender": "male", "age": "mature", "description": "Informative"},
    {"name": "Kore", "gender": "female", "age": "adult", "description": "Firm"},
    {"name": "Fenrir", "gender": "male", "age": "adult", "description": "Excitable"},
    {"name": "Leda", "gender": "female", "age": "young", "description": "Youthful"},
    {"name": "Orus", "gender": "male", "age": "adult", "description": "Firm"},
    {"name": "Aoede", "gender": "female", "age": "adult", "description": "Breezy"},
    {"name": "Callirrhoe", "gender": "female", "age": "adult", "description": "Easy-going"},
    {"name": "Autonoe", "gender": "female", "age": "young", "description": "Bright"},
    {"name": "Enceladus", "gender": "male", "age": "mature", "description": "Breathy"},
    {"name": "Iapetus", "gender": "male", "age": "mature", "description": "Clear"},
    {"name": "Umbriel", "gender": "male", "age": "adult", "description": "Easy-going"},
    {"name": "Algieba", "gender": "male", "age": "mature", "description": "Smooth"},
    {"name": "Despina", "gender": "female", "age": "adult", "description": "Smooth"},
    {"name": "Erinome", "gender": "female", "age": "adult", "description": "Clear"},
    {"name": "Algenib", "gender": "male", "age": "mature", "description": "Gravelly"},
    {"name": "Rasalgethi", "gender": "male", "age": "mature", "description": "Informative"},
    {"name": "Laomedeia", "gender": "female", "age": "young", "description": "Upbeat"},
    {"name": "Achernar", "gender": "female", "age": "young", "description": "Soft"},
    {"name": "Alnilam", "gender": "male", "age": "adult", "description": "Firm"},
    {"name": "Schedar", "gender": "male", "age": "mature", "description": "Even"},
    {"name": "Gacrux", "gender": "female", "age": "mature", "description": "Mature"},
    {"name": "Pulcherrima", "gender": "female", "age": "adult", "description": "Forward"},
    {"name": "Achird", "gender": "male", "age": "young", "description": "Friendly"},
    {"name": "Zubenelgenubi", "gender": "male", "age": "adult", "description": "Casual"},
    {"name": "Vindemiatrix", "gender": "female", "age": "mature", "description": "Gentle"},
    {"name": "Sadachbia", "gender": "male", "age": "young", "description": "Lively"},
    {"name": "Sadaltager", "gender": "male", "age": "mature", "description": "Knowledgeable"},
    {"name": "Sulafat", "gender": "female", "age": "adult", "description": "Warm"},
]


def _require_admin(x_admin_key: str = Header(None)) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured.")
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


def _require_configured() -> None:
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    if _gemini_client is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (GEMINI_API_KEY unset).")


def _today_start_utc() -> datetime.datetime:
    """Midnight in STUDIO_TIMEZONE (Africa/Kigali), expressed in UTC
    for comparison against `created_at` — "today" resets for the admin,
    not at UTC midnight, which for UTC+2 would otherwise cut the day
    over two hours early from their perspective."""
    now_local = datetime.datetime.now(STUDIO_TIMEZONE)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_local.astimezone(datetime.timezone.utc)


def _spend_today_cents(user_id: Optional[str] = None) -> int:
    """Total studio_api_costs logged since local midnight. `user_id` is
    unused today (every call is admin-gated, not tied to a specific
    Supabase user) but already plumbed through so a future per-user cap
    only needs to pass it in — see this file's own module docstring."""
    try:
        query = (
            supabase_admin.table("studio_api_costs")
            .select("cost_usd_cents")
            .gte("created_at", _today_start_utc().isoformat())
        )
        if user_id is not None:
            query = query.eq("user_id", user_id)
        rows = query.execute().data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check today's Studio spend: {e}")
    return sum(int(r.get("cost_usd_cents") or 0) for r in rows)


def _check_daily_cap(estimated_cost_cents: int) -> None:
    """Called before every billed Gemini call (text/image/voice) with
    that call's actual-or-estimated cost, so a call that would push
    today's spend over the cap never fires at all — not just once
    already over it."""
    spent = _spend_today_cents()
    if spent + estimated_cost_cents > STUDIO_DAILY_CAP_USD_CENTS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily limit reached (${STUDIO_DAILY_CAP_USD_CENTS / 100:.2f}). "
                "Try again tomorrow or raise the limit."
            ),
        )


def _estimate_text_cost_cents(input_text: str) -> int:
    """Pre-call estimate for a text/structured-output Gemini call —
    actual token counts aren't known until the response comes back, so
    _check_daily_cap can't just wait for the real cost the way image/
    voice calls do (those have a flat, known-ahead cost). Input tokens
    are estimated at ~4 characters/token (a standard rule of thumb);
    output is budgeted as a generous fixed upper bound for the
    structured JSON this file's text calls return (cast/scene lists).
    Deliberately conservative — overestimating blocks a call a few
    cents early; underestimating lets one slip past the cap."""
    estimated_input_tokens = max(1, len(input_text) // 4)
    estimated_output_tokens = 4000
    input_cost = estimated_input_tokens / 1_000_000 * GEMINI_TEXT_INPUT_USD_PER_1M_TOKENS
    output_cost = estimated_output_tokens / 1_000_000 * GEMINI_TEXT_OUTPUT_USD_PER_1M_TOKENS
    return max(1, round((input_cost + output_cost) * 100))


def _log_cost(series_id: Optional[str], call_type: str, cost_usd_cents: int) -> None:
    """Best-effort — a failed log write must never undo work (an image
    already generated and uploaded, a script already analyzed) that the
    caller is about to return to the admin."""
    try:
        supabase_admin.table("studio_api_costs").insert({
            "series_id": series_id,
            "user_id": None,
            "call_type": call_type,
            "provider": "gemini",
            "cost_usd_cents": cost_usd_cents,
        }).execute()
    except Exception as e:
        print(f"[WARN] Could not log Studio API cost ({call_type}, {cost_usd_cents}c): {e}")


def _text_cost_cents(usage: Optional[types.GenerateContentResponseUsageMetadata]) -> int:
    if usage is None:
        return 0
    input_cost = (usage.prompt_token_count or 0) / 1_000_000 * GEMINI_TEXT_INPUT_USD_PER_1M_TOKENS
    output_cost = (usage.candidates_token_count or 0) / 1_000_000 * GEMINI_TEXT_OUTPUT_USD_PER_1M_TOKENS
    return max(1, round((input_cost + output_cost) * 100))


def _upload_image(image_bytes: bytes, mime_type: str, path: str) -> str:
    """Uploads generated image bytes to Supabase Storage and returns
    its public URL — same two-call upload-then-get_public_url idiom
    repurpose.py already uses for thumbnails, same public-bucket
    convention (not Bunny, which is video-only)."""
    try:
        supabase_admin.storage.from_(STUDIO_IMAGES_BUCKET).upload(
            path, image_bytes, file_options={"content-type": mime_type}
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not upload generated image: {e}")
    return supabase_admin.storage.from_(STUDIO_IMAGES_BUCKET).get_public_url(path)


def _extract_image(response: types.GenerateContentResponse) -> tuple[bytes, str]:
    candidates = response.candidates or []
    if candidates and candidates[0].content and candidates[0].content.parts:
        for part in candidates[0].content.parts:
            if part.inline_data is not None and part.inline_data.data:
                return part.inline_data.data, part.inline_data.mime_type or "image/png"
    raise HTTPException(status_code=502, detail="Gemini did not return an image for this prompt.")


def _extract_audio_pcm(response: types.GenerateContentResponse) -> tuple[bytes, int]:
    """Gemini TTS returns raw 16-bit mono PCM, not a self-describing
    container — the sample rate comes back in the part's mime_type
    (e.g. "audio/L16;codec=pcm;rate=24000") rather than in the bytes
    themselves, so it has to be parsed out here and threaded through
    to _pcm_to_wav."""
    candidates = response.candidates or []
    if candidates and candidates[0].content and candidates[0].content.parts:
        for part in candidates[0].content.parts:
            if part.inline_data is not None and part.inline_data.data:
                rate_match = re.search(r"rate=(\d+)", part.inline_data.mime_type or "")
                sample_rate = int(rate_match.group(1)) if rate_match else 24000
                return part.inline_data.data, sample_rate
    raise HTTPException(status_code=502, detail="Gemini did not return audio for this voice.")


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buf.getvalue()


def _fetch_image_part(url: str) -> types.Part:
    """Downloads a previously-generated reference image (a character
    portrait or location image, already public in Supabase Storage)
    so its bytes can be sent to Gemini as conditioning input — Gemini
    takes inline image bytes, not a URL, so this is the bridge between
    "an image we already generated and saved" and "an image Gemini can
    look at again for the next generation."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch reference image: {e}")
    mime_type = resp.headers.get("content-type", "image/png").split(";")[0].strip() or "image/png"
    return types.Part.from_bytes(data=resp.content, mime_type=mime_type)


def _upload_audio(wav_bytes: bytes, path: str) -> str:
    try:
        supabase_admin.storage.from_(STUDIO_AUDIO_BUCKET).upload(
            path, wav_bytes, file_options={"content-type": "audio/wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not upload generated audio: {e}")
    return supabase_admin.storage.from_(STUDIO_AUDIO_BUCKET).get_public_url(path)


def _normalize_gender(raw: str) -> str:
    g = (raw or "").strip().lower()
    if g.startswith("f") or "woman" in g or "girl" in g:
        return "female"
    if g.startswith("m") or "man" in g or "boy" in g:
        return "male"
    return "other"


def _age_bucket(raw: str) -> str:
    """Parses a free-text age like "mid-20s" or "around 60" into a
    young/adult/mature bucket matching GEMINI_VOICES' own tags.
    Falls back to word hints, then to "adult", when no digits are
    present at all."""
    digits = re.findall(r"\d+", raw or "")
    if digits:
        age = int(digits[0])
        if age < 25:
            return "young"
        if age < 50:
            return "adult"
        return "mature"
    text = (raw or "").lower()
    if any(w in text for w in ("teen", "young", "kid", "child")):
        return "young"
    if any(w in text for w in ("old", "elder", "senior", "mature")):
        return "mature"
    return "adult"


def _auto_assign_voice(gender: str, age: str, used: set) -> str:
    """Free, local heuristic — no Gemini call, no cost logged. Prefers
    an unused voice matching both gender and age, then gender alone,
    then any unused voice, and only reuses a voice already given to
    another character in this series once the cast outgrows the
    30-voice catalog (a shared voice beats no voice at all)."""
    target_gender = _normalize_gender(gender)
    target_age = _age_bucket(age)

    def pick(predicate):
        for voice in GEMINI_VOICES:
            if voice["name"] not in used and predicate(voice):
                return voice["name"]
        return None

    choice = pick(lambda v: v["gender"] == target_gender and v["age"] == target_age)
    if choice is None and target_gender != "other":
        choice = pick(lambda v: v["gender"] == target_gender)
    if choice is None:
        choice = pick(lambda v: True)
    if choice is None:
        choice = GEMINI_VOICES[len(used) % len(GEMINI_VOICES)]["name"]
    return choice


def _get_character(character_id: str) -> dict:
    try:
        result = supabase_admin.table("series_characters").select("*").eq("id", character_id).limit(1).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load character: {e}")
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Character not found.")
    return rows[0]


# ---------------------------------------------------------------------------
# Script -> characters & locations
# ---------------------------------------------------------------------------
class CharacterOut(BaseModel):
    name: str
    age: str
    gender: str
    appearance: str
    clothing: str
    personality: str


class LocationOut(BaseModel):
    name: str
    description: str
    time_of_day: str
    mood: str


class ScriptAnalysis(BaseModel):
    characters: list[CharacterOut]
    locations: list[LocationOut]


class AnalyzeScriptRequest(BaseModel):
    script: str = Field(..., min_length=1, max_length=200_000)


class AnalyzeScriptResponse(BaseModel):
    characters: list[CharacterOut]
    locations: list[LocationOut]
    cost_usd_cents: int


_SCRIPT_ANALYSIS_PROMPT = """You are a casting and location-scouting assistant for a vertical short-drama production.

Read the following script and extract every distinct speaking character and every distinct location.

For each CHARACTER, infer (from dialogue, stage directions and context — make a reasonable judgment call if the script doesn't spell it out explicitly):
- name: their name as used in the script (or a short descriptive label like "Barista" if unnamed)
- age: an approximate age or age range (e.g. "mid-20s")
- gender: as implied by the script
- appearance: physical description — build, hair, face, distinguishing features
- clothing: what they'd plausibly wear in this story
- personality: 2-3 traits that would inform how they're voiced and directed

For each LOCATION, infer:
- name: a short label (e.g. "Maya's Apartment - Living Room")
- description: visual description of the space
- time_of_day: when scenes here take place (e.g. "Night", "Morning", "Varies")
- mood: the emotional tone of the space (e.g. "Tense", "Cozy")

Do not invent characters or locations that aren't in the script. Do not merge two distinct characters into one.

SCRIPT:
{script}
"""


@router.post("/analyze-script", response_model=AnalyzeScriptResponse, dependencies=[Depends(_require_admin)])
async def analyze_script(req: AnalyzeScriptRequest):
    _require_configured()
    _check_daily_cap(_estimate_text_cost_cents(req.script))

    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=_SCRIPT_ANALYSIS_PROMPT.format(script=req.script),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ScriptAnalysis,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini script analysis failed: {e}")

    parsed = response.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Gemini returned a response Viyo Studio couldn't parse.")

    cost_cents = _text_cost_cents(response.usage_metadata)
    _log_cost(None, "analyze_script", cost_cents)

    return AnalyzeScriptResponse(
        characters=parsed.characters,
        locations=parsed.locations,
        cost_usd_cents=cost_cents,
    )


# ---------------------------------------------------------------------------
# Reference images
# ---------------------------------------------------------------------------
class CharacterImageRequest(BaseModel):
    name: str
    age: str
    gender: str
    appearance: str
    clothing: str
    personality: str
    # A fixed, verbatim costume description (e.g. "black designer suit
    # jacket, white shirt, black tie") reused as-is in every image
    # prompt this character appears in — unlike [clothing] above (a
    # looser, script-inferred wardrobe guess), this is admin-written
    # specifically to pin one exact outfit across every scene, so a
    # character doesn't visibly change clothes scene to scene. Empty
    # by default — see CharacterIn's own comment.
    costume_lock: str = ""


class LocationImageRequest(BaseModel):
    name: str
    description: str
    time_of_day: str
    mood: str


class ImageResponse(BaseModel):
    image_url: str
    cost_usd_cents: int


def _generate_and_upload_image(prompt: str, path_prefix: str, call_type: str) -> ImageResponse:
    _check_daily_cap(GEMINI_IMAGE_COST_USD_CENTS)
    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini image generation failed: {e}")

    image_bytes, mime_type = _extract_image(response)
    ext = "png" if "png" in mime_type else "jpg"
    path = f"{path_prefix}/{uuid.uuid4().hex}.{ext}"
    image_url = _upload_image(image_bytes, mime_type, path)

    _log_cost(None, call_type, GEMINI_IMAGE_COST_USD_CENTS)
    return ImageResponse(image_url=image_url, cost_usd_cents=GEMINI_IMAGE_COST_USD_CENTS)


@router.post("/character/portrait", response_model=ImageResponse, dependencies=[Depends(_require_admin)])
async def generate_character_portrait(req: CharacterImageRequest):
    _require_configured()
    clothing_line = f"Clothing: {req.clothing}"
    if req.costume_lock.strip():
        # Reinforces, not just adds to, [clothing] — this is the exact
        # outfit every later scene image is also told to match (see
        # generate_scene_image), so the reference portrait itself
        # needs to start from it, not a looser general description.
        clothing_line = f"Clothing (exact, required): {req.costume_lock}"
    prompt = (
        "Front-facing portrait photo of a fictional character for a drama series, "
        "shoulders-up, looking directly at camera, neutral plain studio background, "
        "soft even lighting, photorealistic.\n\n"
        f"Name: {req.name}\nAge: {req.age}\nGender: {req.gender}\n"
        f"Appearance: {req.appearance}\n{clothing_line}\nPersonality: {req.personality}\n\n"
        "No text, no watermark, no other people in frame."
    )
    return _generate_and_upload_image(prompt, "characters", "character_portrait")


@router.post("/location/image", response_model=ImageResponse, dependencies=[Depends(_require_admin)])
async def generate_location_image(req: LocationImageRequest):
    _require_configured()
    prompt = (
        "Establishing shot of a location for a vertical short-drama series, "
        "9:16 vertical aspect ratio, cinematic lighting, no people, no text, no watermark.\n\n"
        f"Location: {req.name}\nDescription: {req.description}\n"
        f"Time of day: {req.time_of_day}\nMood: {req.mood}"
    )
    return _generate_and_upload_image(prompt, "locations", "location_image")


# ---------------------------------------------------------------------------
# Save / read a series' cast
# ---------------------------------------------------------------------------
class CharacterIn(BaseModel):
    name: str
    age: str
    gender: str
    appearance: str
    clothing: str
    personality: str
    portrait_url: Optional[str] = None
    voice_id: Optional[str] = None
    # See CharacterImageRequest's own comment — a fixed costume
    # description always injected verbatim into every scene image
    # prompt this character appears in (generate_scene_image), not
    # just the portrait. Empty/unset is a real, supported choice (no
    # lock — clothing can vary scene to scene same as before this
    # existed), not a default waiting to be filled in.
    costume_lock: str = ""


class LocationIn(BaseModel):
    name: str
    description: str
    time_of_day: str
    mood: str
    reference_image_url: Optional[str] = None


def _row_with_field_defaults(row: dict, model: type[BaseModel]) -> dict:
    """Builds the kwargs for model(**...) from a DB row, falling back to
    a field's own default instead of raising KeyError when the row
    predates that column — exactly what happened with costume_lock: it
    was added to CharacterIn (and, separately, handed over as a
    migration that adds the column to series_characters) after some
    character rows already existed. A row that query ran against
    before that migration landed has no costume_lock key in it at all,
    and the plain `{k: r[k] for k in Model.model_fields}` this replaces
    raised a bare KeyError on the very first such row — a crash neither
    get_cast/save_cast/assign_voices' own try/except caught (it happens
    after them, building the response), so it surfaced as a raw 500
    with no detail, and the Flutter Studio home screen's per-series
    error handling silently displayed that as "Cast: not started" —
    indistinguishable from a series that genuinely has no cast. A
    column that's still genuinely missing from the row AND has no
    default on the model (a required field with real data loss) still
    raises, correctly, instead of inventing a value for something that
    was never optional.
    """
    out = {}
    for key, field in model.model_fields.items():
        if key in row:
            out[key] = row[key]
        elif field.default is not PydanticUndefined:
            out[key] = field.default
        elif field.default_factory is not None:
            out[key] = field.default_factory()
        else:
            raise KeyError(key)
    return out


class CharacterSaveIn(CharacterIn):
    # Present when this item is an already-saved character being
    # edited/kept; absent for one newly added via Analyze Script. See
    # save_cast's own comment for why this is what makes the save a
    # real upsert instead of delete-then-insert.
    id: Optional[str] = None


class LocationSaveIn(LocationIn):
    id: Optional[str] = None


class SaveCastRequest(BaseModel):
    characters: list[CharacterSaveIn]
    locations: list[LocationSaveIn]


class SavedCharacter(CharacterIn):
    id: str


class SavedLocation(LocationIn):
    id: str


class CastResponse(BaseModel):
    characters: list[SavedCharacter]
    locations: list[SavedLocation]


def _get_series_owner(series_id: str) -> str:
    try:
        result = supabase_admin.table("series").select("id,user_id").eq("id", series_id).limit(1).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load series: {e}")
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Series not found.")
    return rows[0]["user_id"]


@router.post("/series/{series_id}/cast", response_model=CastResponse, dependencies=[Depends(_require_admin)])
async def save_cast(series_id: str, req: SaveCastRequest):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    _get_series_owner(series_id)  # 404s if the series doesn't exist

    # Real upsert, not delete-then-insert. A character/location's id is
    # what scenes and dialogue lines actually reference
    # (generate_scene_image reads character_id off a scene; editLine
    # stores one per line) — recreating every row under a fresh id on
    # every single save orphaned that reference silently, which read
    # in the app as a character the admin had plainly already cast
    # (e.g. "MR. OSEI") permanently flagged unmatched on every scene
    # that used them, with no save ever able to fix it since the next
    # save just orphaned the new id too. It also wiped voice_id every
    # time, since the old insert-only path never wrote it back at all.
    #
    # An item in the request with an id that belongs to this series
    # gets updated in place (same row, same id, every field including
    # voice_id refreshed). An item with no id is a brand new character/
    # location and gets inserted. An existing row whose id isn't in
    # this request at all was explicitly removed via the per-card
    # delete button in the UI, and only that row gets deleted — not
    # "everything that existed before this save."
    try:
        existing_char_ids = {
            r["id"]
            for r in (
                supabase_admin.table("series_characters").select("id").eq("series_id", series_id).execute()
            ).data
            or []
        }
        existing_loc_ids = {
            r["id"]
            for r in (
                supabase_admin.table("series_locations").select("id").eq("series_id", series_id).execute()
            ).data
            or []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load existing cast: {e}")

    removed_char_ids = existing_char_ids - {c.id for c in req.characters if c.id}
    removed_loc_ids = existing_loc_ids - {l.id for l in req.locations if l.id}

    try:
        if removed_char_ids:
            supabase_admin.table("series_characters").delete().in_("id", list(removed_char_ids)).execute()
        if removed_loc_ids:
            supabase_admin.table("series_locations").delete().in_("id", list(removed_loc_ids)).execute()

        char_rows = []
        for i, c in enumerate(req.characters):
            row = {
                "series_id": series_id,
                "name": c.name,
                "age": c.age,
                "gender": c.gender,
                "appearance": c.appearance,
                "clothing": c.clothing,
                "personality": c.personality,
                "portrait_url": c.portrait_url,
                "voice_id": c.voice_id,
                "costume_lock": c.costume_lock,
                "sort_order": i,
            }
            if c.id and c.id in existing_char_ids:
                result = supabase_admin.table("series_characters").update(row).eq("id", c.id).execute().data
            else:
                # An id supplied here that ISN'T in existing_char_ids
                # (as opposed to no id at all) is an explicit recreate
                # — recovering a row that's been deleted, with its
                # original id, so anything that already referenced it
                # (a scene's character_id) resolves again instead of
                # getting a new, unrelated row. Postgres accepts an
                # explicit UUID on insert same as any other column.
                if c.id:
                    row["id"] = c.id
                result = supabase_admin.table("series_characters").insert(row).execute().data
            char_rows.extend(result or [])

        loc_rows = []
        for i, l in enumerate(req.locations):
            row = {
                "series_id": series_id,
                "name": l.name,
                "description": l.description,
                "time_of_day": l.time_of_day,
                "mood": l.mood,
                "reference_image_url": l.reference_image_url,
                "sort_order": i,
            }
            if l.id and l.id in existing_loc_ids:
                result = supabase_admin.table("series_locations").update(row).eq("id", l.id).execute().data
            else:
                if l.id:
                    row["id"] = l.id
                result = supabase_admin.table("series_locations").insert(row).execute().data
            loc_rows.extend(result or [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save cast: {e}")

    return CastResponse(
        characters=[SavedCharacter(id=r["id"], **_row_with_field_defaults(r, CharacterIn)) for r in char_rows],
        locations=[SavedLocation(id=r["id"], **_row_with_field_defaults(r, LocationIn)) for r in loc_rows],
    )


@router.get("/series/{series_id}/cast", response_model=CastResponse, dependencies=[Depends(_require_admin)])
async def get_cast(series_id: str):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    try:
        char_rows = (
            supabase_admin.table("series_characters")
            .select("*")
            .eq("series_id", series_id)
            .order("sort_order")
            .execute()
        ).data or []
        loc_rows = (
            supabase_admin.table("series_locations")
            .select("*")
            .eq("series_id", series_id)
            .order("sort_order")
            .execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load cast: {e}")

    return CastResponse(
        characters=[SavedCharacter(id=r["id"], **_row_with_field_defaults(r, CharacterIn)) for r in char_rows],
        locations=[SavedLocation(id=r["id"], **_row_with_field_defaults(r, LocationIn)) for r in loc_rows],
    )


class UpdateSeriesDetailsRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    genre: Optional[str] = None
    # Reassigns which real account this drama — and every episode
    # already published from it — belongs to. Every Studio-generated
    # episode already gets a real owner at publish time (see
    # publish_episode's own owner_user_id, read from series.user_id),
    # but until now there was no way to change it after the fact, or
    # to pick anyone other than whichever account happened to be
    # signed in when the series was first created in Studio.
    user_id: Optional[str] = None


class UpdateSeriesDetailsResponse(BaseModel):
    id: str
    title: str
    description: str
    genre: str
    user_id: str


@router.post(
    "/series/{series_id}/details",
    response_model=UpdateSeriesDetailsResponse,
    dependencies=[Depends(_require_admin)],
)
async def update_series_details(series_id: str, req: UpdateSeriesDetailsRequest):
    """Renames/edits a drama's title, description or genre, and/or
    reassigns which account owns it.

    Routed through the service-role client, like every other write in
    this file, rather than a direct RLS-scoped client update: `series`'
    update policy is owner-only, but Viyo Studio manages every drama
    regardless of which account originally created it (often a
    different session than whoever's running Studio today) — a direct
    client write would silently match 0 rows for any series Studio
    itself didn't just create in the current session.
    """
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    updates = {k: v for k, v in req.model_dump(exclude_none=True).items()}
    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update.")

    if "user_id" in updates:
        try:
            profile_rows = (
                supabase_admin.table("profiles").select("id").eq("id", updates["user_id"]).limit(1).execute()
            ).data or []
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not verify creator account: {e}")
        if not profile_rows:
            raise HTTPException(status_code=400, detail="No account found with that id.")

    try:
        result = supabase_admin.table("series").update(updates).eq("id", series_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save drama details: {e}")
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Series not found.")
    row = rows[0]

    if "user_id" in updates:
        # Every episode already published from this series needs to
        # move with it — publish_episode only ever reads series.user_id
        # once, at publish time, so leaving already-published posts
        # behind would mean the drama's own page shows the new creator
        # while its episodes in the feed still show the old one.
        try:
            supabase_admin.table("posts").update({"user_id": updates["user_id"]}).eq(
                "series_id", series_id
            ).execute()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Saved the new owner on the drama, but could not update its already-published "
                f"episodes to match: {e}",
            )

    return UpdateSeriesDetailsResponse(
        id=row["id"],
        title=row.get("title") or "",
        description=row.get("description") or "",
        genre=row.get("genre") or "",
        user_id=row.get("user_id") or "",
    )


# ---------------------------------------------------------------------------
# Spend tracking
# ---------------------------------------------------------------------------
class SpendTodayResponse(BaseModel):
    spent_usd_cents: int
    cap_usd_cents: int


@router.get("/spend-today", response_model=SpendTodayResponse, dependencies=[Depends(_require_admin)])
async def spend_today():
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    return SpendTodayResponse(
        spent_usd_cents=_spend_today_cents(),
        cap_usd_cents=STUDIO_DAILY_CAP_USD_CENTS,
    )


# ---------------------------------------------------------------------------
# Phase 2: Voices
# ---------------------------------------------------------------------------
# Unlike Phase 1's analyze/portrait/location endpoints (which operate on
# draft data with no DB id yet, so the admin can freely edit before
# ever saving), voice assignment operates on already-saved
# series_characters rows — "the same voice in every episode" only
# means something once a character has a stable id to hang that voice
# off of. That's why every endpoint below takes a character_id rather
# than a full character payload.
class VoiceInfo(BaseModel):
    name: str
    gender: str
    age: str
    description: str


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]


@router.get("/voices", response_model=VoicesResponse, dependencies=[Depends(_require_admin)])
async def list_voices():
    return VoicesResponse(voices=[VoiceInfo(**v) for v in GEMINI_VOICES])


class VoicePreviewRequest(BaseModel):
    voice_name: str
    sample_text: Optional[str] = Field(None, max_length=500)


class VoicePreviewResponse(BaseModel):
    audio_url: str
    cost_usd_cents: int


@router.post(
    "/character/{character_id}/voice-preview",
    response_model=VoicePreviewResponse,
    dependencies=[Depends(_require_admin)],
)
async def preview_voice(character_id: str, req: VoicePreviewRequest):
    """Generates a short sample line in the given voice and uploads it
    — does NOT persist anything, so the admin can audition as many
    voices as they like before committing one via POST .../voice."""
    _require_configured()
    character = _get_character(character_id)
    _check_daily_cap(GEMINI_TTS_PREVIEW_COST_USD_CENTS)

    sample_text = req.sample_text or f"Hi, I'm {character['name']}."

    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=sample_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=req.voice_name)
                    )
                ),
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini voice preview failed: {e}")

    pcm_bytes, sample_rate = _extract_audio_pcm(response)
    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)
    path = f"previews/{uuid.uuid4().hex}.wav"
    audio_url = _upload_audio(wav_bytes, path)

    _log_cost(None, "voice_preview", GEMINI_TTS_PREVIEW_COST_USD_CENTS)
    return VoicePreviewResponse(audio_url=audio_url, cost_usd_cents=GEMINI_TTS_PREVIEW_COST_USD_CENTS)


class SetVoiceRequest(BaseModel):
    voice_name: str


class CharacterVoiceResponse(BaseModel):
    id: str
    voice_id: str


@router.post(
    "/character/{character_id}/voice",
    response_model=CharacterVoiceResponse,
    dependencies=[Depends(_require_admin)],
)
async def set_character_voice(character_id: str, req: SetVoiceRequest):
    """Persists the admin's chosen voice — this, not the preview call
    above, is what makes a character keep the same voice in every
    future episode (Phase 3's dialogue-audio generation will read this
    same voice_id back off series_characters)."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    _get_character(character_id)  # 404s if missing
    try:
        supabase_admin.table("series_characters").update({"voice_id": req.voice_name}).eq(
            "id", character_id
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save voice: {e}")
    return CharacterVoiceResponse(id=character_id, voice_id=req.voice_name)


class AssignVoicesResponse(BaseModel):
    characters: list[SavedCharacter]


@router.post(
    "/series/{series_id}/assign-voices",
    response_model=AssignVoicesResponse,
    dependencies=[Depends(_require_admin)],
)
async def assign_voices(series_id: str):
    """Free local-heuristic auto-assignment (see _auto_assign_voice) for
    every character in this series that doesn't already have a voice
    — no Gemini call, no cost logged. Characters that already have a
    voice_id are left untouched, so this is safe to call repeatedly
    (e.g. right after saving a new batch of characters from Phase 1)."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    try:
        rows = (
            supabase_admin.table("series_characters")
            .select("*")
            .eq("series_id", series_id)
            .order("sort_order")
            .execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load cast: {e}")

    used = {r["voice_id"] for r in rows if r.get("voice_id")}
    updated_rows = []
    for row in rows:
        if row.get("voice_id"):
            updated_rows.append(row)
            continue
        voice_name = _auto_assign_voice(row.get("gender", ""), row.get("age", ""), used)
        used.add(voice_name)
        try:
            supabase_admin.table("series_characters").update({"voice_id": voice_name}).eq(
                "id", row["id"]
            ).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save auto-assigned voice: {e}")
        row["voice_id"] = voice_name
        updated_rows.append(row)

    return AssignVoicesResponse(
        characters=[SavedCharacter(id=r["id"], **_row_with_field_defaults(r, CharacterIn)) for r in updated_rows]
    )


# ---------------------------------------------------------------------------
# Phase 3: Scenes
# ---------------------------------------------------------------------------
# Scenes belong to one specific episode of a series, not the series as
# a whole (unlike characters/locations/voices, which are reused across
# every episode) — hence the (series_id, episode_number) keying
# throughout this section rather than series_id alone.
def _load_characters(series_id: str) -> list[dict]:
    try:
        return (
            supabase_admin.table("series_characters").select("*").eq("series_id", series_id).execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load characters: {e}")


def _load_locations(series_id: str) -> list[dict]:
    try:
        return (
            supabase_admin.table("series_locations").select("*").eq("series_id", series_id).execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load locations: {e}")


def _match_by_name(name: str, rows: list[dict]) -> Optional[dict]:
    needle = (name or "").strip().lower()
    for row in rows:
        if (row.get("name") or "").strip().lower() == needle:
            return row
    return None


_NAME_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "mister", "doctor"}


def _normalize_name_tokens(name: str) -> list[str]:
    """Lowercases, strips punctuation, and drops honorifics, leaving the
    bare name tokens to compare — so "MR. OSEI", "Mr. Osei" and "Osei"
    all reduce to ["osei"]."""
    cleaned = re.sub(r"[^\w\s]", " ", (name or "").lower())
    return [t for t in cleaned.split() if t and t not in _NAME_TITLES]


def _match_character_by_name(name: str, characters: list[dict]) -> Optional[dict]:
    """Matches a script speaker name to a saved character: case-,
    punctuation- and title-insensitive, and tolerant of a bare first or
    last name standing in for a full name ("DANIEL" or "VANCE" both
    match "Daniel Vance"). Only resolves a bare first/last name when
    exactly one character could be meant — if two characters share a
    first or last name, that's left unmatched (same as no match) rather
    than silently guessing which one was meant.
    """
    needle_tokens = _normalize_name_tokens(name)
    if not needle_tokens:
        return None
    needle = " ".join(needle_tokens)

    full_matches = []
    partial_matches = []
    for row in characters:
        row_tokens = _normalize_name_tokens(row.get("name") or "")
        if not row_tokens:
            continue
        if " ".join(row_tokens) == needle:
            full_matches.append(row)
        elif len(needle_tokens) == 1 and needle_tokens[0] in row_tokens:
            partial_matches.append(row)

    if len(full_matches) == 1:
        return full_matches[0]
    if full_matches:
        return None  # same normalized full name on multiple characters

    if len(partial_matches) == 1:
        return partial_matches[0]
    return None  # no match, or a first/last name shared by several characters


class SceneLineOut(BaseModel):
    speaker: str
    text: str


class SceneOut(BaseModel):
    location: str
    camera_shot: str
    characters_present: list[str]
    visual_description: str
    lines: list[SceneLineOut]


class SceneSplitResult(BaseModel):
    scenes: list[SceneOut]


_SCENE_SPLIT_PROMPT = """You are a director breaking an episode script into individual scenes for a vertical short-drama production.

If the script below has explicit scene markers ("Scene 1", "Scene 2 — Kitchen", "SCENE 3:", etc.), they are hard boundaries:
- Every marked scene becomes at least one output scene. Never merge two differently-marked scenes into one, even if they feel short or continuous with each other.
- Only split a single marked scene into MORE than one output scene when either: it has more than 2 dialogue lines, or the action changes significantly partway through it (characters move somewhere else, a new character enters in a way that shifts what's happening, a clear beat change). A marked scene with 2 or fewer lines and no major action shift stays exactly one output scene — do not split it further and do not merge it into a neighboring marked scene either.
- Keep the marked scenes in their original order.

If the script has NO scene markers at all, fall back to splitting wherever the location changes or there's a significant time jump — don't split a single continuous conversation into multiple scenes just because several lines are spoken.

For each scene, identify:
- location: the location name, matching the script's own naming as closely as possible
- camera_shot: one of "wide", "medium", "close-up" — pick whichever best suits the scene's emotional beat
- characters_present: every character who appears or speaks in this scene, by the name used in the script
- visual_description: a short (1-2 sentence) description of what's visible on screen during this scene — the setting, who's where, what they're doing physically — not a summary of the dialogue
- lines: the dialogue for this scene, in order, each with the speaking character's name and their line (skip stage directions that aren't actually spoken)

SCRIPT:
{script}
"""


class SplitScenesRequest(BaseModel):
    script: str = Field(..., min_length=1, max_length=200_000)


class SavedSceneLine(BaseModel):
    id: str
    sort_order: int
    character_id: Optional[str]
    character_name: str
    text: str
    audio_url: Optional[str]


class SavedScene(BaseModel):
    id: str
    sort_order: int
    location_id: Optional[str]
    location_name: str
    camera_shot: str
    characters: list[dict]  # [{"character_id": str|None, "name": str}, ...]
    visual_description: str
    image_url: Optional[str]
    video_url: Optional[str]
    lines: list[SavedSceneLine]


class ScenesResponse(BaseModel):
    scenes: list[SavedScene]
    cost_usd_cents: int = 0


def _row_to_scene(scene_row: dict, line_rows: list[dict]) -> SavedScene:
    return SavedScene(
        id=scene_row["id"],
        sort_order=scene_row["sort_order"],
        location_id=scene_row.get("location_id"),
        location_name=scene_row.get("location_name") or "",
        camera_shot=scene_row.get("camera_shot") or "medium",
        characters=scene_row.get("characters") or [],
        visual_description=scene_row.get("visual_description") or "",
        image_url=scene_row.get("image_url"),
        video_url=scene_row.get("video_url"),
        lines=[
            SavedSceneLine(
                id=l["id"],
                sort_order=l["sort_order"],
                character_id=l.get("character_id"),
                character_name=l.get("character_name") or "",
                text=l.get("text") or "",
                audio_url=l.get("audio_url"),
            )
            for l in sorted(line_rows, key=lambda l: l["sort_order"])
        ],
    )


def _load_scenes(series_id: str, episode_number: int) -> list[SavedScene]:
    try:
        scene_rows = (
            supabase_admin.table("series_scenes")
            .select("*")
            .eq("series_id", series_id)
            .eq("episode_number", episode_number)
            .order("sort_order")
            .execute()
        ).data or []
        scene_ids = [s["id"] for s in scene_rows]
        line_rows = (
            supabase_admin.table("series_scene_lines").select("*").in_("scene_id", scene_ids).execute()
        ).data if scene_ids else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load scenes: {e}")

    lines_by_scene: dict[str, list[dict]] = {}
    for l in line_rows:
        lines_by_scene.setdefault(l["scene_id"], []).append(l)

    return [_row_to_scene(s, lines_by_scene.get(s["id"], [])) for s in scene_rows]


@router.post(
    "/series/{series_id}/episode/{episode_number}/split-scenes",
    response_model=ScenesResponse,
    dependencies=[Depends(_require_admin)],
)
async def split_scenes(series_id: str, episode_number: int, req: SplitScenesRequest):
    """Splits this episode's script into scenes and dialogue lines and
    saves them — unlike Phase 1's analyze-script, there's no separate
    draft-then-save step, since scene images and line audio (generated
    by the endpoints below) need a stable id to attach to right away.

    Replace-all for this (series_id, episode_number): re-running this
    on the same episode wipes any images/audio already generated for
    its previous scenes, same tradeoff save_cast already makes for the
    whole cast — see that endpoint's own comment.
    """
    _require_configured()
    _get_series_owner(series_id)  # 404s if the series doesn't exist
    _check_daily_cap(_estimate_text_cost_cents(req.script))

    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=_SCENE_SPLIT_PROMPT.format(script=req.script),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SceneSplitResult,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini scene split failed: {e}")

    parsed = response.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Gemini returned a response Viyo Studio couldn't parse.")
    cost_cents = _text_cost_cents(response.usage_metadata)
    _log_cost(series_id, "split_scenes", cost_cents)

    characters = _load_characters(series_id)
    locations = _load_locations(series_id)

    try:
        supabase_admin.table("series_scenes").delete().eq("series_id", series_id).eq(
            "episode_number", episode_number
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clear previous scenes: {e}")

    saved_scenes: list[SavedScene] = []
    for i, scene in enumerate(parsed.scenes):
        location_match = _match_by_name(scene.location, locations)
        scene_characters = []
        for name in scene.characters_present:
            match = _match_character_by_name(name, characters)
            scene_characters.append({"character_id": match["id"] if match else None, "name": name})

        try:
            scene_row = (
                supabase_admin.table("series_scenes")
                .insert(
                    {
                        "series_id": series_id,
                        "episode_number": episode_number,
                        "sort_order": i,
                        "location_id": location_match["id"] if location_match else None,
                        "location_name": scene.location,
                        "camera_shot": scene.camera_shot,
                        "characters": scene_characters,
                        "visual_description": scene.visual_description,
                    }
                )
                .execute()
                .data[0]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save scene: {e}")

        line_rows = []
        if scene.lines:
            try:
                line_rows = (
                    supabase_admin.table("series_scene_lines")
                    .insert(
                        [
                            {
                                "scene_id": scene_row["id"],
                                "sort_order": j,
                                "character_id": (_match_character_by_name(line.speaker, characters) or {}).get("id"),
                                "character_name": line.speaker,
                                "text": line.text,
                            }
                            for j, line in enumerate(scene.lines)
                        ]
                    )
                    .execute()
                    .data
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Could not save scene lines: {e}")

        saved_scenes.append(_row_to_scene(scene_row, line_rows))

    return ScenesResponse(scenes=saved_scenes, cost_usd_cents=cost_cents)


@router.get(
    "/series/{series_id}/episode/{episode_number}/scenes",
    response_model=ScenesResponse,
    dependencies=[Depends(_require_admin)],
)
async def get_scenes(series_id: str, episode_number: int):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    return ScenesResponse(scenes=_load_scenes(series_id, episode_number))


def _get_scene(scene_id: str) -> dict:
    try:
        result = supabase_admin.table("series_scenes").select("*").eq("id", scene_id).limit(1).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load scene: {e}")
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Scene not found.")
    return rows[0]


def _get_scene_line(line_id: str) -> dict:
    try:
        result = supabase_admin.table("series_scene_lines").select("*").eq("id", line_id).limit(1).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load line: {e}")
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Line not found.")
    return rows[0]


class EditSceneRequest(BaseModel):
    visual_description: Optional[str] = None
    camera_shot: Optional[str] = None
    location_id: Optional[str] = None
    location_name: Optional[str] = None
    # [{"character_id": str|None, "name": str, "costume_override": str},
    # ...] — lets the admin fix a scene-level characters_present entry
    # that split-scenes couldn't match at the time (e.g. one split
    # before the name matcher got smarter, or a name the matcher is
    # still ambiguous about). Scene-level entries are a split-time
    # snapshot, unlike dialogue lines' own character_id, so nothing
    # re-matches them automatically — this is how the admin corrects
    # one by hand. costume_override (optional, defaults to "" when
    # absent) swaps out that one character's locked costume for just
    # this one scene — e.g. pajamas for a home scene instead of their
    # usual suit — without touching the character's own costume_lock,
    # which every *other* scene they're in keeps using unchanged. See
    # generate_scene_image's own use of it.
    characters: Optional[list[dict]] = None


@router.post("/scene/{scene_id}/edit", response_model=SavedScene, dependencies=[Depends(_require_admin)])
async def edit_scene(scene_id: str, req: EditSceneRequest):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    scene_row = _get_scene(scene_id)
    updates = {k: v for k, v in req.model_dump(exclude_none=True).items()}
    if updates:
        try:
            scene_row = supabase_admin.table("series_scenes").update(updates).eq("id", scene_id).execute().data[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save scene edit: {e}")

    try:
        line_rows = (
            supabase_admin.table("series_scene_lines").select("*").eq("scene_id", scene_id).execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load scene lines: {e}")
    return _row_to_scene(scene_row, line_rows)


class EditLineRequest(BaseModel):
    text: Optional[str] = None
    character_id: Optional[str] = None
    character_name: Optional[str] = None


@router.post("/line/{line_id}/edit", response_model=SavedSceneLine, dependencies=[Depends(_require_admin)])
async def edit_line(line_id: str, req: EditLineRequest):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    row = _get_scene_line(line_id)  # 404s if missing
    updates = {k: v for k, v in req.model_dump(exclude_none=True).items()}
    if updates:
        try:
            row = supabase_admin.table("series_scene_lines").update(updates).eq("id", line_id).execute().data[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save line edit: {e}")
    return SavedSceneLine(
        id=row["id"],
        sort_order=row["sort_order"],
        character_id=row.get("character_id"),
        character_name=row.get("character_name") or "",
        text=row.get("text") or "",
        audio_url=row.get("audio_url"),
    )


class SceneImageResponse(BaseModel):
    image_url: str
    cost_usd_cents: int


@router.post(
    "/scene/{scene_id}/image", response_model=SceneImageResponse, dependencies=[Depends(_require_admin)]
)
async def generate_scene_image(scene_id: str):
    """Generates (or regenerates) this scene's 9:16 image, conditioned
    on the Phase 1 reference images of every character present and
    the scene's location — this, not a fresh unconditioned generation,
    is what keeps a character's face and a location's look consistent
    from scene to scene and episode to episode."""
    _require_configured()
    scene_row = _get_scene(scene_id)
    _check_daily_cap(GEMINI_IMAGE_COST_USD_CENTS)

    reference_parts: list[types.Part] = []
    character_names = []
    costume_lines = []
    for c in scene_row.get("characters") or []:
        character_id = c.get("character_id")
        if not character_id:
            continue
        try:
            rows = (
                supabase_admin.table("series_characters")
                .select("name,portrait_url,costume_lock")
                .eq("id", character_id)
                .execute()
            ).data or []
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load character reference: {e}")
        if rows and rows[0].get("portrait_url"):
            reference_parts.append(types.Part.from_text(text=f"Reference photo for character \"{rows[0]['name']}\":"))
            reference_parts.append(_fetch_image_part(rows[0]["portrait_url"]))
            character_names.append(rows[0]["name"])
        if rows:
            # A per-scene override (EditSceneRequest.characters' own
            # costume_override) wins over the character's own locked
            # default — this is the one place in the whole costume-
            # lock feature where a scene is allowed to show a
            # different outfit, used deliberately (e.g. pajamas for a
            # home scene) rather than the lock silently failing to
            # hold. The reference portrait above is unaffected either
            # way, so the face/likeness stays pinned regardless of
            # which costume line wins here.
            costume = (c.get("costume_override") or "").strip() or (rows[0].get("costume_lock") or "").strip()
            if costume:
                costume_lines.append(f"{rows[0]['name']}: {costume}")

    location_id = scene_row.get("location_id")
    if location_id:
        try:
            loc_rows = (
                supabase_admin.table("series_locations")
                .select("name,reference_image_url")
                .eq("id", location_id)
                .execute()
            ).data or []
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load location reference: {e}")
        if loc_rows and loc_rows[0].get("reference_image_url"):
            reference_parts.append(
                types.Part.from_text(text=f"Reference photo for location \"{loc_rows[0]['name']}\":")
            )
            reference_parts.append(_fetch_image_part(loc_rows[0]["reference_image_url"]))

    costume_block = ""
    if costume_lines:
        # Repeated verbatim in every scene this character appears in —
        # the reference portrait alone only pins likeness, and Gemini
        # has been observed drifting an outfit's details (a tie color,
        # a jacket vs. no jacket) across otherwise-consistent scene
        # images. Phrased as a hard constraint, not a style suggestion,
        # same register as the "No text, no watermark" line below.
        costume_block = (
            "\n\nCOSTUME LOCK — these characters must be wearing exactly this, with no variation "
            "from scene to scene:\n" + "\n".join(costume_lines)
        )

    instruction = (
        "Using the reference photos above for likeness (same faces, same location — keep clothing and "
        "setting consistent with them unless the description below says otherwise), generate a single "
        "cinematic scene image for a vertical short-drama series.\n\n"
        "9:16 vertical aspect ratio. "
        f"Camera shot: {scene_row.get('camera_shot') or 'medium'} shot. "
        f"Location: {scene_row.get('location_name') or 'unspecified'}. "
        f"Characters in frame: {', '.join(character_names) or 'none specified'}."
        f"{costume_block}\n\n"
        f"What's happening: {scene_row.get('visual_description') or ''}\n\n"
        "No text, no watermark, no speech bubbles or captions baked into the image."
    )

    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[*reference_parts, types.Part.from_text(text=instruction)],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini scene image generation failed: {e}")

    image_bytes, mime_type = _extract_image(response)
    ext = "png" if "png" in mime_type else "jpg"
    path = f"scenes/{uuid.uuid4().hex}.{ext}"
    image_url = _upload_image(image_bytes, mime_type, path)

    try:
        supabase_admin.table("series_scenes").update({"image_url": image_url}).eq("id", scene_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save scene image: {e}")

    _log_cost(scene_row.get("series_id"), "scene_image", GEMINI_IMAGE_COST_USD_CENTS)
    return SceneImageResponse(image_url=image_url, cost_usd_cents=GEMINI_IMAGE_COST_USD_CENTS)


def _fetch_veo_image(url: str) -> types.Image:
    """Same job as _fetch_image_part, but Veo's image-to-video input
    takes a types.Image (image_bytes + mime_type), not the types.Part
    shape generate_content's contents= list expects."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch scene image for Veo: {e}")
    mime_type = resp.headers.get("content-type", "image/png").split(";")[0].strip() or "image/png"
    return types.Image(image_bytes=resp.content, mime_type=mime_type)


class SceneVideoRequest(BaseModel):
    duration_seconds: int = 8


class SceneVideoResponse(BaseModel):
    video_url: str
    duration_seconds: int
    cost_usd_cents: int


@router.post(
    "/scene/{scene_id}/video", response_model=SceneVideoResponse, dependencies=[Depends(_require_admin)]
)
async def generate_scene_video(scene_id: str, req: SceneVideoRequest):
    """Generates a real Veo 3.1 Lite video clip for this one scene,
    animating its already-generated reference image — an opt-in,
    per-scene upgrade from the default Ken Burns zoom/pan, since Veo
    costs dramatically more than everything else Studio calls (see
    VEO_PRICE_PER_SEC_USD_CENTS's own comment). assemble_episode uses
    this clip instead of the still image for any scene that has one,
    so one episode can freely mix Veo scenes and Ken Burns scenes —
    nothing here requires every scene to match.

    Note for whoever reads this after the first real call: Veo's
    generate_videos is new to this codebase and untested against a
    live API key as of writing — if the google-genai SDK's exact
    param/class names here (types.Image, GenerateVideosConfig's
    fields, operations.get, files.download) don't match what 2.28.0
    actually exposes, this fails loudly with a 502 before anything is
    charged (Veo only bills on a successfully generated video) or
    saved, rather than silently costing money for a broken result.
    """
    _require_configured()
    scene_row = _get_scene(scene_id)
    if not scene_row.get("image_url"):
        raise HTTPException(
            status_code=400, detail="Generate this scene's image first — Veo needs it as a starting frame."
        )

    duration = req.duration_seconds if req.duration_seconds in VEO_ALLOWED_DURATIONS else 8
    estimated_cost_cents = duration * VEO_PRICE_PER_SEC_USD_CENTS
    _check_daily_cap(estimated_cost_cents)

    veo_image = _fetch_veo_image(scene_row["image_url"])
    prompt = (
        "Animate this image into a short cinematic video clip for a vertical short-drama series. "
        f"Camera shot: {scene_row.get('camera_shot') or 'medium'} shot. "
        f"What's happening: {scene_row.get('visual_description') or ''}\n\n"
        "Subtle, natural motion — keep the characters, location and framing consistent with the "
        "reference image. No text, no captions, no watermark."
    )

    try:
        operation = _gemini_client.models.generate_videos(
            model=VEO_MODEL,
            prompt=prompt,
            image=veo_image,
            config=types.GenerateVideosConfig(aspect_ratio="9:16", duration_seconds=duration),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Veo video generation failed to start: {e}")

    waited = 0
    while not operation.done:
        if waited >= VEO_MAX_POLL_SECONDS:
            raise HTTPException(
                status_code=504,
                detail=(
                    "Veo is still rendering after several minutes. It may still finish on Google's "
                    "end — wait a bit and check back rather than retrying right away."
                ),
            )
        time.sleep(VEO_POLL_INTERVAL_SECONDS)
        waited += VEO_POLL_INTERVAL_SECONDS
        try:
            operation = _gemini_client.operations.get(operation)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Could not check Veo generation status: {e}")

    if getattr(operation, "error", None):
        raise HTTPException(status_code=502, detail=f"Veo video generation failed: {operation.error}")

    generated_videos = (operation.response.generated_videos if operation.response else None) or []
    if not generated_videos or not generated_videos[0].video:
        raise HTTPException(status_code=502, detail="Veo did not return a video.")

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "veo_output.mp4")
        try:
            _gemini_client.files.download(file=generated_videos[0].video, destination=video_path)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Could not download Veo's generated video: {e}")
        with open(video_path, "rb") as f:
            video_bytes = f.read()

    path = f"scenes/{scene_id}/{uuid.uuid4().hex}.mp4"
    try:
        supabase_admin.storage.from_(STUDIO_VIDEOS_BUCKET).upload(
            path, video_bytes, file_options={"content-type": "video/mp4"}
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not upload Veo video: {e}")
    video_url = supabase_admin.storage.from_(STUDIO_VIDEOS_BUCKET).get_public_url(path)

    try:
        supabase_admin.table("series_scenes").update({"video_url": video_url}).eq("id", scene_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save scene video: {e}")

    _log_cost(scene_row.get("series_id"), "scene_video", estimated_cost_cents)
    return SceneVideoResponse(video_url=video_url, duration_seconds=duration, cost_usd_cents=estimated_cost_cents)


class LineAudioResponse(BaseModel):
    audio_url: str
    cost_usd_cents: int


@router.post("/line/{line_id}/audio", response_model=LineAudioResponse, dependencies=[Depends(_require_admin)])
async def generate_line_audio(line_id: str):
    """Generates (or regenerates) this dialogue line's audio using its
    speaking character's Phase 2 voice — the same voice_id every other
    line of theirs uses, in this episode and every other one."""
    _require_configured()
    line_row = _get_scene_line(line_id)
    scene_row = _get_scene(line_row["scene_id"])
    character_id = line_row.get("character_id")
    if not character_id:
        raise HTTPException(
            status_code=400,
            detail=f'"{line_row.get("character_name")}" isn\'t matched to a saved character, so there\'s no voice to use. Fix the speaker name or assign one in Edit.',
        )
    character = _get_character(character_id)
    voice_name = character.get("voice_id")
    if not voice_name:
        raise HTTPException(
            status_code=400,
            detail=f'{character["name"]} doesn\'t have a voice assigned yet — assign one in the Voices step first.',
        )

    _check_daily_cap(GEMINI_TTS_LINE_COST_USD_CENTS)
    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=line_row["text"],
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                ),
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini line audio generation failed: {e}")

    pcm_bytes, sample_rate = _extract_audio_pcm(response)
    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)
    path = f"lines/{uuid.uuid4().hex}.wav"
    audio_url = _upload_audio(wav_bytes, path)

    try:
        supabase_admin.table("series_scene_lines").update({"audio_url": audio_url}).eq("id", line_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save line audio: {e}")

    _log_cost(scene_row.get("series_id"), "line_audio", GEMINI_TTS_LINE_COST_USD_CENTS)
    return LineAudioResponse(audio_url=audio_url, cost_usd_cents=GEMINI_TTS_LINE_COST_USD_CENTS)


# ---------------------------------------------------------------------------
# Phase 4: Assemble & Publish
# ---------------------------------------------------------------------------
# Turns one episode's already-generated scene images + line audio into
# a real 9:16 MP4 (ffmpeg, subprocess-based — matching repurpose.py's
# own style; moviepy/ffmpeg-python are in requirements.txt but unused
# anywhere in this codebase, so this doesn't introduce a second way of
# doing the same thing) and publishes it exactly the way every other
# episode in this app gets published: a `posts` row with series_id +
# episode_number, video hosted on Bunny Stream. The one difference from
# the client's own publish flow (post_service.dart's createPost) is
# *how* the bytes reach Bunny — see _upload_finished_video_to_bunny.
#
# Two separate endpoints, matching the spec's own "admin previews and
# publishes" — assemble() renders the MP4 and uploads it to Supabase
# Storage (STUDIO_VIDEOS_BUCKET) for an in-app preview; publish() is a
# deliberate separate action that re-downloads that same preview and
# pushes it to Bunny + creates the real post. There's no server-side
# episode-draft state between the two calls (no new table for it) —
# the preview_video_url and duration_seconds assemble() returns are
# simply held in the Flutter screen's memory and sent back on publish,
# the same "draft held client-side until a deliberate save" pattern
# Phase 1's cast editing already uses.
def _download_to_file(url: str, path: str) -> None:
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not download {url}: {e}")
    if not resp.content:
        raise HTTPException(status_code=502, detail=f"Downloaded 0 bytes from {url} — the file may have expired or failed to generate.")
    with open(path, "wb") as f:
        f.write(resp.content)


def _validate_clip(path: str, label: str) -> None:
    """Checks a just-rendered scene/end-card clip actually has a real
    video stream and a real audio stream before it's handed to the
    final concat — concat's own failure mode for a broken input (an
    image that failed to download/decode, audio that came back empty)
    is a cryptic libx264/aac "-22 Invalid argument" with no indication
    of which of the N inputs was the problem, so this catches it one
    clip earlier with an error that actually names the scene."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
         "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    lines = [l for l in result.stdout.strip().splitlines() if l]
    has_video = any(l.startswith("video") for l in lines)
    has_audio = any(l.startswith("audio") for l in lines)
    if result.returncode != 0 or not lines or not has_video or not has_audio:
        raise HTTPException(
            status_code=500,
            detail=f"{label} failed to render properly (missing video or audio stream) — "
            "try regenerating its image/audio and assembling again.",
        )


def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        raise HTTPException(status_code=500, detail=f"Could not read duration of {os.path.basename(path)}.")


def _run_ffmpeg(cmd: list[str], step: str) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"{step} failed: {result.stderr[-500:]}")


def _scene_cues(texts: list[str], durations: list[float]) -> list[tuple[str, float, float]]:
    """Turns a scene's dialogue line texts + their (already-known, from
    each line's own TTS audio) durations into (text, start, end) cues,
    each timed back-to-back starting at 0 — the same cursor-accumulation
    _write_scene_srt used to do, just handed to _build_caption_filters
    instead of written out as an SRT file."""
    cursor = 0.0
    cues = []
    for text, duration in zip(texts, durations):
        cues.append((text, cursor, cursor + duration))
        cursor += duration
    return cues


def _wrap_caption_lines(
    text: str, max_chars: int = _CAPTION_MAX_CHARS_PER_LINE, max_lines: int = 2
) -> list[str]:
    """Greedy word-wrap into at most [max_lines] lines of up to
    [max_chars] each — the "max 2 lines" cap on burned-in captions.
    Any text left over past the last line is truncated with an
    ellipsis rather than silently dropped or left to overflow past the
    frame edge; a TTS dialogue line actually long enough to hit this
    is rare in practice."""
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    if len(lines) <= max_lines:
        return lines

    kept = lines[: max_lines - 1]
    rest = " ".join(lines[max_lines - 1 :])
    if len(rest) > max_chars:
        rest = rest[: max_chars - 1].rstrip() + "…"
    kept.append(rest)
    return kept


def _build_caption_filters(cues: list[tuple[str, float, float]], tmp_dir: str, prefix: str) -> str:
    """Builds a chained drawtext filter string burning in [cues] —
    (text, start_seconds, end_seconds) dialogue cues — as bold white,
    black-outlined text anchored in the lower third (~72% down the
    frame, see _CAPTION_Y_FRACTION), each cue capped at 2 lines.
    Returns "" (nothing to append to a -vf chain) if there are no cues.

    Deliberately drawtext, not the `subtitles` filter this used to use:
    an SRT run through `subtitles` falls back to libass's own default
    PlayResX/Y (384x288 — unrelated to this pipeline's real 1080x1920
    output) whenever the SRT itself carries no resolution of its own,
    so a MarginV written against this frame's real pixel height ends up
    scaled into that other coordinate space and renders off the top of
    the frame entirely — confirmed empirically (text present and
    correctly positioned at the filter's small built-in default margin,
    completely gone once that margin was raised to where "72% down"
    actually needed it) before switching to drawtext, whose x/y are
    plain output-frame pixels with no such hidden coordinate space.

    Each wrapped line is also its own separate drawtext instance (its
    own textfile=, individually centered via its own (w-text_w)/2)
    rather than one drawtext call with an embedded newline: a single
    multi-line drawtext centers the whole block on its widest line, not
    each line on its own, so a cue whose two lines are different
    lengths — the normal case — renders its shorter line partly or
    fully off-center. Confirmed the same way: a deliberately short
    first line against a long second line landed flush against the
    frame edge, invisible, until each line got its own instance.
    """
    if not cues:
        return ""
    filters = []
    for ci, (text, start, end) in enumerate(cues):
        for li, line in enumerate(_wrap_caption_lines(text)):
            text_path = os.path.join(tmp_dir, f"{prefix}_cap_{ci}_{li}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(line)
            # Same path-escaping precaution the old srt_path handling
            # took — ffmpeg's filter string syntax treats a bare ":" as
            # an option separator.
            safe_path = text_path.replace("\\", "/").replace(":", "\\:")
            y = f"h*{_CAPTION_Y_FRACTION}+{li * _CAPTION_LINE_HEIGHT}"
            filters.append(
                f"drawtext=fontfile={_CAPTION_FONT_BOLD_PATH}:textfile='{safe_path}':"
                f"fontcolor=white:fontsize={_CAPTION_FONT_SIZE}:borderw=3:bordercolor=black:"
                f"x=(w-text_w)/2:y={y}:enable='between(t\\,{start:.3f}\\,{end:.3f})'"
            )
    return "," + ",".join(filters)


def _concat_audio(input_paths: list[str], out_path: str) -> None:
    cmd = ["ffmpeg", "-y"]
    for p in input_paths:
        cmd += ["-i", p]
    inputs = "".join(f"[{i}:a]" for i in range(len(input_paths)))
    cmd += [
        "-filter_complex", f"{inputs}concat=n={len(input_paths)}:v=0:a=1[aout]",
        "-map", "[aout]", "-ar", "44100", "-ac", "1", out_path,
    ]
    _run_ffmpeg(cmd, "Scene audio concat")


def _render_silence(duration: float, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
        "-t", str(duration), "-ar", "44100", "-ac", "1", out_path,
    ]
    _run_ffmpeg(cmd, "Silence render")


def _render_scene_clip(
    image_path: str, audio_path: str, cues: list[tuple[str, float, float]], duration: float, pan: str, out_path: str
) -> None:
    """Ken Burns (slow zoom-in, or zoom-plus-pan-across, alternating by
    scene index for some visual variety — see assemble_episode's own
    i % 2) with the scene's dialogue burned in underneath. Always
    moving, start to finish: zoom climbs from _ZOOM_MIN to _ZOOM_MAX
    continuously for the scene's whole duration (never resets, never
    plateaus early), so a still image never just sits there frozen —
    the one thing this whole function exists to guarantee. The pan
    variant reuses the same zoom range rather than its own: a pure
    translation needs zoom > 1 to have any x-room to move across at
    all (at zoom=1 the crop window already fills the source exactly,
    leaving nothing to pan into — confirmed empirically), so this
    gets the pan "for free" out of the same subtle zoom already
    driving the plain zoom-in variant.

    Verified end to end (zoompan expression, caption position/timing,
    exact duration) against synthetic test assets before writing this
    — zoompan's expression syntax, and libass's own default coordinate
    space the previous subtitle-based caption approach silently fell
    into, are each easy to get subtly, invisibly wrong."""
    frames = max(1, int(round(duration * _FFMPEG_FPS)))
    zoom_step = (_ZOOM_MAX - _ZOOM_MIN) / frames
    if pan == "left_right":
        x_expr = f"(iw-iw/zoom)*(on/{frames})"
    else:
        x_expr = "iw/2-(iw/zoom/2)"
    vf = (
        f"scale=2160:3840,zoompan=z='min(zoom+{zoom_step:.8f},{_ZOOM_MAX})':x='{x_expr}':"
        f"y='ih/2-(ih/zoom/2)':d={frames}:s={_FFMPEG_RESOLUTION}:fps={_FFMPEG_FPS}"
    )
    vf += _build_caption_filters(cues, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0])
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path, "-i", audio_path,
        "-vf", vf, "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path,
    ]
    _run_ffmpeg(cmd, "Scene render")


def _render_scene_clip_from_video(
    video_path: str, audio_path: str, cues: list[tuple[str, float, float]], duration: float, out_path: str
) -> None:
    """Same job as _render_scene_clip, but for a scene that got the
    optional Veo upgrade: starts from a real generated video clip
    instead of a still image. When the scene has real dialogue, Veo's
    own audio (always on, can't be disabled via the API) is dropped
    entirely in favor of this scene's actual TTS voice — otherwise
    Veo's guessed audio would play under/over the real character
    voice. When the scene has NO dialogue (cues is empty — a pure
    visual/establishing beat), there's nothing for Veo's audio to
    compete with, so its own ambient sound is kept instead of
    replacing it with flat silence. Veo only returns fixed 4/6/8-
    second clips, so -stream_loop repeats it to cover a longer
    duration and -t/-shortest trims it to cover a shorter one —
    verified both directions against synthetic test clips before
    writing this, same discipline _render_scene_clip's own docstring
    describes."""
    # Explicit fps= matters here the same way it matters in
    # _render_scene_clip's zoompan filter: without it, this clip's
    # output framerate is whatever Veo itself generated at (observed:
    # 24fps) while every Ken Burns scene and the freeze outro render at
    # _FFMPEG_FPS (30) — a real, confirmed mismatch (via the per-clip
    # diagnostic _concat_clips attaches on failure) that's the likely
    # cause of "Episode concat failed" when an episode mixes a Veo
    # scene with Ken Burns scenes.
    vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps={_FFMPEG_FPS}"
    has_dialogue = bool(cues)
    vf += _build_caption_filters(cues, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0])

    if has_dialogue:
        cmd = [
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", video_path, "-i", audio_path,
            "-vf", vf, "-t", str(duration),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", video_path,
            "-vf", vf, "-t", str(duration),
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path,
        ]
    _run_ffmpeg(cmd, "Scene render (Veo)")


def _render_sound_hit(duration: float, out_path: str) -> None:
    """Synthesizes a cinematic "hit" — a decaying low-frequency boom
    layered with a short filtered-noise crack at the onset — entirely
    in ffmpeg. There's no bundled sound-effects library in this
    codebase (same constraint upload_music's own docstring notes for
    background music), and the freeze outro this backs needs to sound
    dramatic with zero admin setup, not wait on someone finding and
    uploading their own impact sample.

    Renders from real files on disk and mixes them in a second pass,
    rather than one ffmpeg call juggling two live `-f lavfi` sources
    and a filter_complex — same "real files, not live dual-lavfi"
    discipline _render_end_card (this function's predecessor) already
    adopted after bisecting a real concat failure to that exact
    pattern. Gain-staged by hand after measuring: the straightforward
    boom+crack mix peaks right at ~0dBFS (the edge of clipping) even
    with amix's own normalize — confirmed via ffmpeg's astats filter
    before picking the 0.5 final gain below, which lands around -2dB."""
    boom_path = os.path.join(os.path.dirname(out_path), "_sound_hit_boom.wav")
    crack_path = os.path.join(os.path.dirname(out_path), "_sound_hit_crack.wav")
    _run_ffmpeg(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"aevalsrc=0.8*exp(-6*t)*sin(2*PI*85*t):d={duration}:s=44100:c=mono",
            boom_path,
        ],
        "Sound hit (boom)",
    )
    _run_ffmpeg(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"anoisesrc=d={duration}:c=white:a=0.6:r=44100",
            "-af", f"highpass=f=1500,afade=t=out:st=0.05:d=0.12,atrim=0:{duration}",
            crack_path,
        ],
        "Sound hit (crack)",
    )
    _run_ffmpeg(
        [
            "ffmpeg", "-y", "-i", boom_path, "-i", crack_path,
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,volume=0.5[aout]",
            "-map", "[aout]", "-ac", "1", "-ar", "44100", out_path,
        ],
        "Sound hit (mix)",
    )


def _render_freeze_outro(last_frame_path: str, tmp_dir: str, out_path: str) -> None:
    """Freezes the final scene's last frame for _OUTRO_SECONDS with a
    synthesized dramatic sound hit and big "PART 2 ON VIYO" text over
    it — replaces the old separate end card (a different, unrelated
    solid-color card cutting in after the episode actually ends).
    Holding on the episode's own last frame instead keeps the cut
    inside the scene the viewer was just watching, which is what makes
    the "freeze" read as a deliberate dramatic beat rather than a
    jarring cut to a slide.

    Renders from a real image file + a real synthesized-but-on-disk
    WAV, same "real files on disk, not a live dual-source filter
    graph" discipline _render_sound_hit's own docstring explains —
    verified locally that this still produces a valid, concat-
    compatible clip (matching every other clip's exact spec) before
    shipping it."""
    sound_path = os.path.join(tmp_dir, "outro_sound_hit.wav")
    _render_sound_hit(_OUTRO_SECONDS, sound_path)

    text_path = os.path.join(tmp_dir, "outro_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(_OUTRO_TEXT)

    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", last_frame_path, "-i", sound_path,
        "-vf",
        f"fps={_FFMPEG_FPS},drawtext=textfile={text_path}:fontfile={_CAPTION_FONT_BOLD_PATH}:"
        f"fontcolor=white:fontsize={_OUTRO_FONT_SIZE}:borderw=5:bordercolor=black:"
        "x=(w-text_w)/2:y=(h-text_h)/2",
        "-t", str(_OUTRO_SECONDS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path,
    ]
    _run_ffmpeg(cmd, "Freeze outro render")


def _describe_clip(path: str) -> str:
    """One-line ffprobe summary of a clip's streams (codec, duration,
    sample rate/channels, dimensions/frame rate) plus its file size —
    used to enrich a concat failure with the real numbers from every
    input clip. _validate_clip already rules out a clip missing a
    stream entirely before concat runs; this catches the next layer
    down — a stream that exists but looks wrong in some way (an
    implausible duration, an odd sample rate) that only a concat
    re-encode trips over."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,duration,sample_rate,channels,width,height,r_frame_rate",
         "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    size = os.path.getsize(path) if os.path.exists(path) else -1
    lines = [l for l in result.stdout.strip().splitlines() if l]
    if not lines:
        return f"{os.path.basename(path)} ({size}B): ffprobe found no streams ({result.stderr.strip()[:200]})"
    return f"{os.path.basename(path)} ({size}B): " + " | ".join(lines)


def _concat_clips(clip_paths: list[str], out_path: str, labels: Optional[list[str]] = None) -> None:
    """Tries the concat *demuxer* with a lossless stream copy first,
    falling back to the concat *filter* (re-encoding) only if that
    fails. This order used to be the other way around: the demuxer
    used to silently drop ~15% of total duration when stream-copying
    heterogeneous clips (different codecs/frame rates between scene
    images and captions), so the filter was the only safe choice.
    That's no longer true — every clip this function receives is now
    rendered to the exact same spec (1080x1920, _FFMPEG_FPS, h264/
    yuv420p, aac/44100/mono) by _render_scene_clip,
    _render_scene_clip_from_video and _render_freeze_outro, so a stream
    copy is valid and sidesteps re-encoding (and whatever's opening
    the encoder successfully in local testing but failing in
    production with "Could not open encoder before EOF" — frame rate
    and live-lavfi-source fixes that were each independently confirmed
    real via bisection on an actual failure still didn't resolve it,
    and the likeliest remaining explanation is the ffmpeg *build*
    itself: Debian bookworm, what this app's own Dockerfile's
    python:3.11-slim base actually ships, packages ffmpeg 5.1.9 — a
    full major version behind the 6.1.1 this was tested against).

    On a demuxer failure, falls back to the filter-based re-encode,
    and on failure there, [labels] (parallel to clip_paths, e.g.
    "Scene 1", "End card") name which input is which in the attached
    per-clip diagnostic + bisection dump."""
    list_path = os.path.join(os.path.dirname(out_path), "_concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    copy_result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if copy_result.returncode == 0:
        return

    cmd = ["ffmpeg", "-y"]
    for p in clip_paths:
        cmd += ["-i", p]
    parts = "".join(f"[{i}:v][{i}:a]" for i in range(len(clip_paths)))
    cmd += [
        "-filter_complex", f"{parts}concat=n={len(clip_paths)}:v=1:a=1[vout][aout]",
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        dump = "\n".join(
            f"{labels[i] if labels and i < len(labels) else f'input {i}'}: {_describe_clip(p)}"
            for i, p in enumerate(clip_paths)
        )
        bisection = _bisect_concat_failure(clip_paths, labels, os.path.dirname(out_path))
        raise HTTPException(
            status_code=500,
            detail=(
                f"Episode concat failed (stream copy also failed: {copy_result.stderr[-200:]}): "
                f"{result.stderr[-400:]}\n\n{bisection}\n\nPer-clip diagnostic:\n{dump}"
            ),
        )


def _bisect_concat_failure(clip_paths: list[str], labels: Optional[list[str]], tmp_dir: str) -> str:
    """When the full concat fails, retries concatenating increasing
    prefixes of clip_paths (2 clips, then 3, then 4, ...) to find
    exactly which clip's addition first breaks it. The per-clip
    ffprobe dump shows every stream's codec/resolution/rate looking
    consistent, so whatever's actually wrong only surfaces when ffmpeg
    tries to combine streams in the filter graph — not from reading
    headers, which is all the static dump above can see."""
    if len(clip_paths) < 2:
        return "Bisection: only one clip — nothing to narrow down."
    for i in range(2, len(clip_paths) + 1):
        probe_path = os.path.join(tmp_dir, f"_bisect_{i}.mp4")
        cmd = ["ffmpeg", "-y"]
        for p in clip_paths[:i]:
            cmd += ["-i", p]
        parts = "".join(f"[{j}:v][{j}:a]" for j in range(i))
        cmd += [
            "-filter_complex", f"{parts}concat=n={i}:v=1:a=1[vout][aout]",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", probe_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            culprit = labels[i - 1] if labels and i - 1 < len(labels) else f"input {i - 1}"
            return (
                f"Bisection: concatenating the first {i} clips fails; adding \"{culprit}\" "
                f"is what breaks it. Its own error: {result.stderr[-300:]}"
            )
    return "Bisection: every prefix concatenated fine on retry (the failure may be intermittent)."


def _mix_background_music(video_path: str, music_path: str, duration: float, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", music_path,
        "-filter_complex",
        f"[1:a]volume=0.12,atrim=0:{duration}[music];"
        "[0:a][music]amix=inputs=2:duration=first:dropout_transition=0[aout]",
        "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", out_path,
    ]
    _run_ffmpeg(cmd, "Background music mix")


def _extract_thumbnail(video_path: str, out_path: str, at_seconds: float = 0.5) -> None:
    # Defaults to 0.5s in, matching PostService.generateAndUploadVideoThumbnail's
    # own timeMs: 500 — skips a possible black opening frame the same
    # way a normally-uploaded episode's thumbnail already does.
    # assemble_episode also calls this at other timestamps to offer
    # thumbnail candidates, since a single fixed frame is sometimes a
    # bad pick (mid-blink, a blank establishing shot, a transition).
    cmd = ["ffmpeg", "-y", "-ss", str(at_seconds), "-i", video_path, "-frames:v", "1", "-update", "1", out_path]
    _run_ffmpeg(cmd, "Thumbnail extraction")


def _upload_video_preview(video_bytes: bytes, path: str) -> str:
    try:
        supabase_admin.storage.from_(STUDIO_VIDEOS_BUCKET).upload(
            path, video_bytes, file_options={"content-type": "video/mp4"}
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not upload assembled video: {e}")
    return supabase_admin.storage.from_(STUDIO_VIDEOS_BUCKET).get_public_url(path)


class MusicUploadResponse(BaseModel):
    music_url: str


@router.post("/music", response_model=MusicUploadResponse, dependencies=[Depends(_require_admin)])
async def upload_music(file: UploadFile = File(...)):
    """Uploads a background-music file the admin already has the
    rights to use (see AssembleEpisodeRequest.music_url's own comment
    below — there's no bundled library) so it can be referenced by URL
    without the admin needing to find their own external hosting for
    it first."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    ext = os.path.splitext(file.filename or "")[1] or ".mp3"
    path = f"music/{uuid.uuid4().hex}{ext}"
    try:
        supabase_admin.storage.from_(STUDIO_AUDIO_BUCKET).upload(
            path, data, file_options={"content-type": file.content_type or "audio/mpeg"}
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not upload music: {e}")
    music_url = supabase_admin.storage.from_(STUDIO_AUDIO_BUCKET).get_public_url(path)
    return MusicUploadResponse(music_url=music_url)


class ThumbnailUploadResponse(BaseModel):
    thumbnail_url: str


@router.post("/thumbnail", response_model=ThumbnailUploadResponse, dependencies=[Depends(_require_admin)])
async def upload_thumbnail(file: UploadFile = File(...)):
    """Lets the admin use a thumbnail that isn't one of
    assemble_episode's auto-extracted candidates — a cover image made
    elsewhere — fed into publish_episode's thumbnail_url the same way
    a chosen candidate is."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    path = f"episode-thumbnails/{uuid.uuid4().hex}{ext}"
    thumbnail_url = _upload_image(data, file.content_type or "image/jpeg", path)
    return ThumbnailUploadResponse(thumbnail_url=thumbnail_url)


class AssembleEpisodeRequest(BaseModel):
    # Admin-supplied — this backend has no royalty-free music library
    # of its own (bundling real audio files here would mean vouching
    # for licensing this codebase has no way to verify), so background
    # music is opt-in: a URL to a track the admin already has the
    # rights to use, mixed in low under the dialogue (uploaded via
    # upload_music above, or any other URL the admin already has).
    # Omitted entirely when not given, not replaced with a placeholder.
    # Sound effects aren't implemented for the same reason plus the
    # lack of any way to pick which effect fits a given scene
    # automatically.
    music_url: Optional[str] = None


class AssembleEpisodeResponse(BaseModel):
    preview_video_url: str
    duration_seconds: int
    # A handful of candidate hero frames spread across the episode, so
    # the admin can pick a good one in publish_episode's caption/
    # thumbnail step instead of always getting the fixed 0.5s-in frame
    # that function falls back to when none is chosen.
    thumbnail_candidates: list[str] = []


@router.post(
    "/series/{series_id}/episode/{episode_number}/assemble",
    response_model=AssembleEpisodeResponse,
    dependencies=[Depends(_require_admin)],
)
async def assemble_episode(series_id: str, episode_number: int, req: AssembleEpisodeRequest):
    """Renders this episode's scenes into one 9:16 MP4 and uploads it
    to Storage for preview — does NOT touch Bunny or create a post;
    see this section's own header comment for why that's a separate
    deliberate publish() call instead of happening automatically here."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    _get_series_owner(series_id)
    scenes = _load_scenes(series_id, episode_number)
    if not scenes:
        raise HTTPException(status_code=400, detail="No scenes for this episode yet — split the script first.")

    missing = []
    for i, scene in enumerate(scenes):
        if not scene.image_url and not scene.video_url:
            missing.append(f"Scene {i + 1} has no image")
        for j, line in enumerate(scene.lines):
            if not line.audio_url:
                missing.append(f"Scene {i + 1}, line {j + 1} has no audio")
    if missing:
        detail = "Generate everything before assembling: " + "; ".join(missing[:5])
        if len(missing) > 5:
            detail += f" (+{len(missing) - 5} more)"
        raise HTTPException(status_code=400, detail=detail)

    with tempfile.TemporaryDirectory() as tmp:
        clip_paths = []
        for i, scene in enumerate(scenes):
            line_audio_paths = []
            for j, line in enumerate(scene.lines):
                p = os.path.join(tmp, f"scene_{i}_line_{j}.wav")
                _download_to_file(line.audio_url, p)
                line_audio_paths.append(p)

            scene_audio_path = os.path.join(tmp, f"scene_{i}_audio.wav")
            if line_audio_paths:
                _concat_audio(line_audio_paths, scene_audio_path)
            else:
                _render_silence(_SCENE_SILENCE_SECONDS, scene_audio_path)
            scene_duration = _ffprobe_duration(scene_audio_path)

            cues: list[tuple[str, float, float]] = []
            if scene.lines:
                line_durations = [_ffprobe_duration(p) for p in line_audio_paths]
                cues = _scene_cues([l.text for l in scene.lines], line_durations)

            clip_path = os.path.join(tmp, f"scene_{i}_clip.mp4")
            if scene.video_url:
                video_path = os.path.join(tmp, f"scene_{i}_veo.mp4")
                _download_to_file(scene.video_url, video_path)
                _render_scene_clip_from_video(video_path, scene_audio_path, cues, scene_duration, clip_path)
            else:
                image_path = os.path.join(tmp, f"scene_{i}.png")
                _download_to_file(scene.image_url, image_path)
                pan = "left_right" if i % 2 else "center"
                _render_scene_clip(image_path, scene_audio_path, cues, scene_duration, pan, clip_path)
            _validate_clip(clip_path, f"Scene {i + 1}")
            clip_paths.append(clip_path)

        # Freeze outro: holds the episode's own last frame (not a
        # separate, unrelated card) for a dramatic beat into the "Part
        # 2" hook — see _render_freeze_outro's own docstring. Reuses
        # the final iteration's clip_path/scene_duration above; Python
        # keeps a for loop's locals alive after it ends, and scenes is
        # already confirmed non-empty up front.
        last_frame_path = os.path.join(tmp, "outro_last_frame.png")
        # A hair before the true end, not exactly at it — seeking to a
        # clip's literal last timestamp sometimes lands past EOF with
        # no frame to extract (confirmed empirically).
        _extract_thumbnail(clip_path, last_frame_path, at_seconds=max(0.0, scene_duration - 0.1))
        outro_path = os.path.join(tmp, "outro.mp4")
        _render_freeze_outro(last_frame_path, tmp, outro_path)
        _validate_clip(outro_path, "Freeze outro")
        clip_paths.append(outro_path)

        assembled_path = os.path.join(tmp, "assembled.mp4")
        clip_labels = [f"Scene {i + 1}" for i in range(len(scenes))] + ["Freeze outro"]
        _concat_clips(clip_paths, assembled_path, labels=clip_labels)

        final_path = assembled_path
        if req.music_url:
            music_path = os.path.join(tmp, "music_src")
            _download_to_file(req.music_url, music_path)
            mixed_path = os.path.join(tmp, "with_music.mp4")
            _mix_background_music(assembled_path, music_path, _ffprobe_duration(assembled_path), mixed_path)
            final_path = mixed_path

        duration_seconds = int(round(_ffprobe_duration(final_path)))

        # A few candidate hero frames spread across the episode — a
        # single fixed-timestamp grab (what publish_episode falls back
        # to) is sometimes a bad pick: mid-blink, a blank establishing
        # shot, a scene transition. Letting the admin see and choose
        # from several matters for click-through in the feed, and
        # that's a judgment call only a human previewing the episode
        # can make well.
        thumbnail_candidates: list[str] = []
        candidate_offsets = sorted({
            round(min(t, max(duration_seconds - 0.3, 0.1)), 2)
            for t in (0.5, duration_seconds * 0.25, duration_seconds * 0.5, duration_seconds * 0.75)
        })
        for i, offset in enumerate(candidate_offsets):
            cand_path = os.path.join(tmp, f"thumb_candidate_{i}.jpg")
            try:
                _extract_thumbnail(final_path, cand_path, at_seconds=offset)
                with open(cand_path, "rb") as f:
                    cand_bytes = f.read()
                thumbnail_candidates.append(
                    _upload_image(cand_bytes, "image/jpeg", f"episode-thumbnails/{uuid.uuid4().hex}.jpg")
                )
            except Exception:
                # A bad candidate frame shouldn't block assembling the
                # episode itself — publish_episode still has its own
                # guaranteed 0.5s extraction as a fallback.
                continue

        with open(final_path, "rb") as f:
            video_bytes = f.read()
        preview_url = _upload_video_preview(video_bytes, f"{series_id}/{episode_number}/{uuid.uuid4().hex}.mp4")

    return AssembleEpisodeResponse(
        preview_video_url=preview_url,
        duration_seconds=duration_seconds,
        thumbnail_candidates=thumbnail_candidates,
    )


def _upload_finished_video_to_bunny(video_path: str, title: str) -> str:
    """Server-side push of a file already on disk — different from
    bunny_stream.py's own create_bunny_video, which hands the Flutter
    CLIENT a time-boxed TUS credential and never sees the bytes itself
    (see that file's module docstring). Studio already has the whole
    finished MP4 after ffmpeg assembly, so there's nothing for TUS's
    resumable-chunked-upload machinery to buy here — Bunny's simpler
    direct (non-TUS) upload API does the same job in two calls: create
    the video object, then PUT the bytes straight to it."""
    if not _bunny_configured():
        raise HTTPException(status_code=503, detail="Bunny Stream is not configured.")

    try:
        create_resp = requests.post(
            f"{_BUNNY_API_BASE}/library/{BUNNY_STREAM_LIBRARY_ID}/videos",
            json={"title": title[:200]},
            headers={"AccessKey": BUNNY_STREAM_API_KEY, "Content-Type": "application/json"},
            timeout=15,
        )
        create_resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not create Bunny video: {e}")

    video_id = create_resp.json().get("guid")
    if not video_id:
        raise HTTPException(status_code=502, detail="Bunny did not return a video id.")

    try:
        with open(video_path, "rb") as f:
            upload_resp = requests.put(
                f"{_BUNNY_API_BASE}/library/{BUNNY_STREAM_LIBRARY_ID}/videos/{video_id}",
                data=f,
                headers={"AccessKey": BUNNY_STREAM_API_KEY},
                timeout=600,
            )
        upload_resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not upload video to Bunny: {e}")

    return video_id


class PublishEpisodeRequest(BaseModel):
    preview_video_url: str
    duration_seconds: int
    caption: Optional[str] = None
    # One of assemble_episode's thumbnail_candidates, or a URL from
    # upload_thumbnail below — used as-is instead of the automatic
    # 0.5s-in extraction when given.
    thumbnail_url: Optional[str] = None


class PublishEpisodeResponse(BaseModel):
    post_id: str
    media_url: str
    video_status: str


@router.post(
    "/series/{series_id}/episode/{episode_number}/publish",
    response_model=PublishEpisodeResponse,
    dependencies=[Depends(_require_admin)],
)
async def publish_episode(series_id: str, episode_number: int, req: PublishEpisodeRequest):
    """Re-downloads the already-assembled preview (see assemble_episode
    above for why this doesn't just keep the ffmpeg working directory
    around), pushes it to Bunny, and inserts the same `posts` row shape
    post_service.dart's own createPost does — this IS the publish
    action (is_private is always False here), unlike that client flow,
    which can also create a scheduled/private draft row."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")
    owner_user_id = _get_series_owner(series_id)

    try:
        series_rows = supabase_admin.table("series").select("title").eq("id", series_id).limit(1).execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load series: {e}")
    series_title = series_rows[0]["title"] if series_rows else "Viyo"

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "episode.mp4")
        _download_to_file(req.preview_video_url, video_path)

        video_id = _upload_finished_video_to_bunny(video_path, f"{series_title} - Episode {episode_number}")

        if req.thumbnail_url:
            thumbnail_url = req.thumbnail_url
        else:
            thumb_path = os.path.join(tmp, "thumb.jpg")
            _extract_thumbnail(video_path, thumb_path)
            with open(thumb_path, "rb") as f:
                thumb_bytes = f.read()
            thumbnail_url = _upload_image(thumb_bytes, "image/jpeg", f"episode-thumbnails/{uuid.uuid4().hex}.jpg")

    row = {
        "user_id": owner_user_id,
        "post_type": "video",
        "caption": req.caption or f"{series_title} - Episode {episode_number}",
        "media_url": _bunny_playback_url(video_id),
        "thumbnail_url": thumbnail_url,
        "duration_seconds": req.duration_seconds,
        "is_private": False,
        "series_id": series_id,
        "episode_number": episode_number,
        "video_provider": "bunny",
        "bunny_video_id": video_id,
        "video_status": "processing",
    }
    try:
        inserted = supabase_admin.table("posts").insert(row).execute().data[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video uploaded to Bunny ({video_id}) but could not create the post: {e}",
        )

    # Same reward the creator gets for posting any other episode —
    # best-effort, matching _log_cost's own "never undo already-
    # finished work over a logging failure" reasoning.
    try:
        supabase_admin.rpc(
            "award_post_creation",
            {"p_user_id": owner_user_id, "p_post_id": inserted["id"], "p_post_type": "video"},
        ).execute()
    except Exception as e:
        print(f"[WARN] award_post_creation failed for Studio-published post {inserted['id']}: {e}")

    return PublishEpisodeResponse(post_id=inserted["id"], media_url=row["media_url"], video_status="processing")


class DeletePostResponse(BaseModel):
    deleted: bool


@router.delete("/post/{post_id}", response_model=DeletePostResponse, dependencies=[Depends(_require_admin)])
async def delete_studio_post(post_id: str):
    """Removes a wrongly-published episode — e.g. published under the
    wrong series/title, or a test run that was never meant to go
    live. Deletes the Bunny video first, best-effort (a failed Bunny
    delete shouldn't block removing the post the admin is actually
    trying to get rid of), then deletes the posts row itself via the
    service role — bypasses the owner-only RLS policy
    PostService.deletePost relies on client-side, since a Studio-
    published post's owner is the series' designated account, not
    necessarily whoever is holding the admin key."""
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")

    try:
        rows = (
            supabase_admin.table("posts")
            .select("video_provider, bunny_video_id")
            .eq("id", post_id)
            .limit(1)
            .execute()
        ).data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load post: {e}")
    if not rows:
        raise HTTPException(status_code=404, detail="Post not found.")

    post = rows[0]
    if post.get("video_provider") == "bunny" and post.get("bunny_video_id") and _bunny_configured():
        try:
            requests.delete(
                f"{_BUNNY_API_BASE}/library/{BUNNY_STREAM_LIBRARY_ID}/videos/{post['bunny_video_id']}",
                headers={"AccessKey": BUNNY_STREAM_API_KEY},
                timeout=15,
            )
        except requests.RequestException:
            pass

    try:
        supabase_admin.table("posts").delete().eq("id", post_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete post: {e}")

    return DeletePostResponse(deleted=True)


# ---------------------------------------------------------------------------
# Studio home screen support
# ---------------------------------------------------------------------------
# Not a new phase — just enough to let the Flutter home screen show
# "where did I leave off" per series without the admin having to
# re-paste a script to find out: which episodes have been split into
# scenes, whether those scenes are fully generated, and whether
# they've already been published.
class EpisodeStudioStatus(BaseModel):
    episode_number: int
    scene_count: int
    images_done: bool
    audio_done: bool
    published: bool
    post_id: Optional[str] = None


class EpisodesStatusResponse(BaseModel):
    episodes: list[EpisodeStudioStatus]


@router.get(
    "/series/{series_id}/episodes",
    response_model=EpisodesStatusResponse,
    dependencies=[Depends(_require_admin)],
)
async def list_episode_status(series_id: str):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Viyo Studio is not configured (Supabase).")

    try:
        scene_rows = (
            supabase_admin.table("series_scenes").select("episode_number").eq("series_id", series_id).execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load episodes: {e}")

    episode_numbers_with_scenes = {r["episode_number"] for r in scene_rows}

    # Queried independently of episode_numbers_with_scenes, not just to
    # check status for episode numbers scenes already told us about —
    # an episode published through the simpler direct-upload flow
    # (upload_ai_drama_screen.dart) never creates series_scenes rows at
    # all, so without this a published episode like that would be
    # completely invisible here even though it's live in the Dramas
    # feed: this function would report "Scenes: not started" for a
    # series that actually already has an episode out.
    try:
        published_rows = (
            supabase_admin.table("posts")
            .select("id, episode_number")
            .eq("series_id", series_id)
            .not_.is_("episode_number", "null")
            .execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check published episodes: {e}")
    # Last write wins if an episode number was somehow ever published
    # twice — fine here, this is only used to offer a delete action.
    post_id_by_episode = {r["episode_number"]: r["id"] for r in published_rows}

    episode_numbers = sorted(episode_numbers_with_scenes | set(post_id_by_episode.keys()))
    if not episode_numbers:
        return EpisodesStatusResponse(episodes=[])

    episodes = []
    for episode_number in episode_numbers:
        # No series_scenes rows for a published-but-scene-less episode
        # (the direct-upload case above) — _load_scenes correctly
        # returns [] for it, which reads as scene_count 0 / neither
        # images nor audio done, rather than this endpoint claiming
        # scene progress it has no actual record of.
        scenes = _load_scenes(series_id, episode_number)
        images_done = bool(scenes) and all(s.image_url for s in scenes)
        audio_done = bool(scenes) and all(all(l.audio_url for l in s.lines) for s in scenes)
        episodes.append(
            EpisodeStudioStatus(
                episode_number=episode_number,
                scene_count=len(scenes),
                images_done=images_done,
                audio_done=audio_done,
                published=episode_number in post_id_by_episode,
                post_id=post_id_by_episode.get(episode_number),
            )
        )

    return EpisodesStatusResponse(episodes=episodes)
