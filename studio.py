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
import os
import uuid
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

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

_gemini_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"

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
