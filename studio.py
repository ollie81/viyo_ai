"""
Viyo Studio (Phase 1): Script -> Cast & Locations.

Turns a pasted episode/series script into a structured cast list —
characters and locations, each with a Gemini-generated reference
image — that gets saved onto a `series` row and reused by every later
episode (Phase 3's scene generation sends these same reference images
back to Gemini so faces and places stay consistent across episodes).

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
`studio_api_costs`) are NOT created by this codebase — same "no
migration-runner access" constraint as everywhere else here. The SQL to
create them (with RLS enabled and zero public policies, since every
read/write here goes through this file's own service-role Supabase
client, never the Flutter app's anon-key client directly) was handed
over separately and must be run by hand in the Supabase SQL editor
before this router will do anything but fail with a clear 500 on the
first real table access.
"""
import datetime
import io
import os
import re
import uuid
import wave
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1/admin/studio", tags=["studio"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# $20/day by default — override in Railway if that's too tight or too
# loose for how much casting work actually happens per day. Read as
# whole dollars (easier to set correctly in an env var than cents) and
# converted once here.
STUDIO_DAILY_CAP_USD_CENTS = int(float(os.environ.get("STUDIO_DAILY_CAP_USD", "20")) * 100)

STUDIO_IMAGES_BUCKET = "studio-images"
STUDIO_AUDIO_BUCKET = "studio-audio"

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

_gemini_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
# Gemini's TTS-capable model, per Google's own Gemini API docs — the
# "flash" (not "pro") preview variant specifically, since voice
# previews are short and cheap is what matters here, not the extra
# quality the pro TTS model charges more for.
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Approximate Gemini 2.5 Flash pricing (per 1M tokens) as of this
# writing — Google changes these; re-check
# https://ai.google.dev/gemini-api/docs/pricing before trusting this
# for real budgeting. This is best-effort cost *tracking* against the
# daily cap, not a billing-accurate invoice.
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
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _spend_today_cents(user_id: Optional[str] = None) -> int:
    """Total studio_api_costs logged since UTC midnight. `user_id` is
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
    spent = _spend_today_cents()
    if spent + estimated_cost_cents > STUDIO_DAILY_CAP_USD_CENTS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Today's Studio spending cap reached (${spent / 100:.2f} of "
                f"${STUDIO_DAILY_CAP_USD_CENTS / 100:.2f}). Try again after midnight UTC, "
                "or raise STUDIO_DAILY_CAP_USD."
            ),
        )


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
    _check_daily_cap(0)  # text cost isn't known until after the call; this just blocks when already over cap

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
    prompt = (
        "Front-facing portrait photo of a fictional character for a drama series, "
        "shoulders-up, looking directly at camera, neutral plain studio background, "
        "soft even lighting, photorealistic.\n\n"
        f"Name: {req.name}\nAge: {req.age}\nGender: {req.gender}\n"
        f"Appearance: {req.appearance}\nClothing: {req.clothing}\nPersonality: {req.personality}\n\n"
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


class LocationIn(BaseModel):
    name: str
    description: str
    time_of_day: str
    mood: str
    reference_image_url: Optional[str] = None


class SaveCastRequest(BaseModel):
    characters: list[CharacterIn]
    locations: list[LocationIn]


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

    # Replace-all semantics for Phase 1 — simplest correct behavior
    # while there's no scene data yet referencing individual character/
    # location rows by id. Once Phase 3 scenes reference these ids,
    # this will need to become a real upsert instead of delete+insert.
    # NOTE: this also means re-saving a cast wipes any voice_id already
    # assigned in Phase 2 (the Flutter cast objects in memory don't
    # carry it round-trip) — another reason this needs to become a
    # real upsert before Phase 3, not a new Phase 2 problem to solve
    # on its own.
    try:
        supabase_admin.table("series_characters").delete().eq("series_id", series_id).execute()
        supabase_admin.table("series_locations").delete().eq("series_id", series_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clear previous cast: {e}")

    try:
        char_rows = supabase_admin.table("series_characters").insert([
            {
                "series_id": series_id,
                "name": c.name,
                "age": c.age,
                "gender": c.gender,
                "appearance": c.appearance,
                "clothing": c.clothing,
                "personality": c.personality,
                "portrait_url": c.portrait_url,
                "sort_order": i,
            }
            for i, c in enumerate(req.characters)
        ]).execute().data if req.characters else []

        loc_rows = supabase_admin.table("series_locations").insert([
            {
                "series_id": series_id,
                "name": l.name,
                "description": l.description,
                "time_of_day": l.time_of_day,
                "mood": l.mood,
                "reference_image_url": l.reference_image_url,
                "sort_order": i,
            }
            for i, l in enumerate(req.locations)
        ]).execute().data if req.locations else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save cast: {e}")

    return CastResponse(
        characters=[SavedCharacter(id=r["id"], **{k: r[k] for k in CharacterIn.model_fields}) for r in char_rows],
        locations=[SavedLocation(id=r["id"], **{k: r[k] for k in LocationIn.model_fields}) for r in loc_rows],
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
        characters=[SavedCharacter(id=r["id"], **{k: r[k] for k in CharacterIn.model_fields}) for r in char_rows],
        locations=[SavedLocation(id=r["id"], **{k: r[k] for k in LocationIn.model_fields}) for r in loc_rows],
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
        characters=[SavedCharacter(id=r["id"], **{k: r[k] for k in CharacterIn.model_fields}) for r in updated_rows]
    )
