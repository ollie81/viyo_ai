"""
Viyo AI Ads Studio: marketing-video generation for VIYO itself, Ollie AI,
or any other app/product/website — a second, complementary feature to
Viyo Studio (studio.py)'s AI Drama Maker, sharing its Gemini/Veo/ffmpeg/
Bunny pipeline rather than duplicating it.

Workflow: create a campaign (what to promote) -> upload up to 8 reference
images (or pull real VIYO screenshots from the shared library below) ->
pick a format (or let Gemini recommend one) -> generate 5 hook candidates,
each scored on a transparent, weighted rubric (HOOK_SCORE_WEIGHTS) -> pick
one (or auto-pick the top score) -> generate an editable scene-by-scene
script built around that hook -> generate the actual video (per-scene
image, optional Veo animation, voiceover, burned-in captions, optional
music) as a tracked background job -> preview/download.

Admin-only, gated by the exact same `_require_admin` (X-Admin-Key) this
whole codebase already uses for every other admin surface — this is an
internal marketing tool for the app owner, not a consumer-facing feature,
so it deliberately does NOT touch coins.py/subscriptions.py (those gate
AI features for end users; irrelevant here). Spends against the same
STUDIO_DAILY_CAP_USD budget pool as Viyo Studio (one admin, one real-money
guardrail), just tagged with its own `call_type` values in
`studio_api_costs` so the two features' spend stays distinguishable.

Reuses studio.py's Gemini client, model constants, image/audio/caption/
render/concat/Bunny helpers directly (imported below) rather than
redefining them — see each import's origin for what it does. Only the
ffmpeg scale/crop target is parameterized here (Studio's own
_render_scene_clip/_render_scene_clip_from_video are hardcoded to
9:16/1080x1920, since Drama Studio only ever produces 9:16 episodes) —
_render_ad_scene_clip/_render_ad_scene_clip_from_video below are the
same filters with width/height as real parameters instead.

New tables this file depends on (ad_campaigns, ad_assets,
ad_hook_candidates, ad_generation_jobs, ad_performance_metrics,
ad_viyo_asset_library) are NOT created by this codebase — same
no-migration-runner-access constraint studio.py's own module docstring
explains. The SQL was handed over separately; every table is RLS-enabled
with zero public policies (service-role only, same posture as every
Studio table). Reuses Studio's existing storage buckets (studio-images,
studio-audio, studio-videos) under an "ads/" prefix rather than asking
for new ones.

No background-job infrastructure (no Celery/Redis) exists anywhere in
this codebase — Studio's own Veo call blocks synchronously with internal
polling inside one HTTP request (studio.py, generate_scene_video), fine
for one scene but too long to hold a request open for a whole multi-
scene ad. POST /campaign/{id}/generate instead creates an
ad_generation_jobs row and hands the real work to FastAPI's own
BackgroundTasks — no new infra, genuinely async, genuinely trackable via
GET /campaign/{id}/job, which the Flutter client polls (same shape as
the Bunny-processing-status polling already proven in
video_feed_screen.dart). Calling /generate again on a failed or even a
ready campaign simply reruns it from the already-saved script — nothing
about retry needs a separate endpoint or any partial-stage checkpoint,
since assets/hooks/script all live on rows /generate never touches.
"""
import datetime
import os
import uuid
from typing import Optional

import requests
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from google.genai import types
from pydantic import BaseModel, Field
from supabase import create_client, Client

from studio import (
    _gemini_client,
    _require_admin,
    _check_daily_cap,
    _log_cost,
    _estimate_text_cost_cents,
    _text_cost_cents,
    _extract_image,
    _extract_audio_pcm,
    _pcm_to_wav,
    _fetch_image_part,
    _fetch_veo_image,
    _upload_image,
    _upload_audio,
    _upload_video_preview,
    _auto_assign_voice,
    _download_to_file,
    _validate_clip,
    _ffprobe_duration,
    _run_ffmpeg,
    _scene_cues,
    _concat_audio,
    _render_silence,
    _build_caption_filters,
    _concat_clips,
    _mix_background_music,
    _extract_thumbnail,
    _upload_finished_video_to_bunny,
    GEMINI_TEXT_MODEL,
    GEMINI_IMAGE_MODEL,
    GEMINI_TTS_MODEL,
    GEMINI_IMAGE_COST_USD_CENTS,
    GEMINI_TTS_LINE_COST_USD_CENTS,
    GEMINI_VOICES,
    VEO_ALLOWED_DURATIONS,
    VEO_MAX_POLL_SECONDS,
    VEO_POLL_INTERVAL_SECONDS,
    _FFMPEG_FPS,
    _ZOOM_MIN,
    _ZOOM_MAX,
)
from bunny_stream import _playback_url as _bunny_playback_url

router = APIRouter(prefix="/api/v1/admin/ads-studio", tags=["ads_studio"])

# Drama Studio only ever uses Lite (the one tier that fits inside its own
# $3/day cap). Ads Studio exposes all three real Veo 3.1 tiers so a
# campaign can trade cost for quality deliberately — model IDs confirmed
# against Google's Gemini API docs/model listing, not guessed:
# "veo-3.1-generate-preview" (Standard) is the exact ID the official
# ai.google.dev Veo docs use; "-lite-"/"-fast-" follow the same documented
# naming convention and "-lite-" is already proven working in studio.py.
# Per-second prices are the same best-effort estimates studio.py's own
# VEO_PRICE_PER_SEC_USD_CENTS carries — re-check ai.google.dev/gemini-api
# /docs/pricing before trusting these for real budgeting; the daily cap
# (_check_daily_cap) is the real backstop regardless of tier, not these
# numbers themselves. Standard is ~8x Lite's cost (an 8s clip alone is
# close to the whole default daily cap) — the Flutter picker warns before
# letting an admin select it.
VEO_TIERS: dict[str, dict] = {
    "lite": {"model": "veo-3.1-lite-generate-preview", "price_per_sec_cents": 5, "label": "Lite"},
    "fast": {"model": "veo-3.1-fast-generate-preview", "price_per_sec_cents": 10, "label": "Fast"},
    "standard": {"model": "veo-3.1-generate-preview", "price_per_sec_cents": 40, "label": "Standard"},
}
DEFAULT_VEO_TIER = "lite"

# Same two env vars every other file in this codebase reads independently
# to build its own service-role client (video_metadata.py, bunny_stream.py,
# studio.py each do this too, rather than importing one shared instance) —
# matching that established convention instead of relying on Python
# import-time binding of studio.py's own module-level supabase_admin.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def _require_configured() -> None:
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Ads Studio is not configured (Supabase).")
    if _gemini_client is None:
        raise HTTPException(status_code=503, detail="Ads Studio is not configured (GEMINI_API_KEY unset).")


MAX_AD_ASSET_BYTES = int(os.environ.get("MAX_AD_ASSET_MB", "15")) * 1024 * 1024
MAX_ASSETS_PER_CAMPAIGN = 8

AD_DURATIONS = (8, 15, 30, 45, 60)
AD_ASPECT_RATIOS = ("9:16", "16:9", "1:1")
AD_RESOLUTIONS = ("720p", "1080p")

# ---------------------------------------------------------------------------
# Ad formats — each entry's "style" is baked into the hook/script prompts
# below so the generated content actually reads differently per format,
# not just a label change over the same generic ad copy.
# ---------------------------------------------------------------------------
AD_FORMATS: dict[str, dict] = {
    "story_driven": {
        "label": "Story-Driven Ad",
        "style": "A compelling mini-story with a real beginning, turn, and payoff that happens to feature the "
        "product naturally near the end — never open with the product itself.",
    },
    "ugc": {
        "label": "UGC-Style Video",
        "style": "A natural, unpolished creator-style video — someone talking straight to camera the way a real "
        "person films a phone recommendation, not a scripted ad read. Casual language, not a pitch.",
    },
    "avatar": {
        "label": "AI Avatar Presenter",
        "style": "A virtual presenter demonstrating the product directly to camera — clear, friendly, "
        "straightforward delivery. This is a presenter character, never framed as a real customer's own "
        "testimonial or experience.",
    },
    "cinematic": {
        "label": "Cinematic Promo",
        "style": "A polished, visually striking promo — dramatic framing, confident pacing, more mood and "
        "visual craft than dialogue.",
    },
    "demo": {
        "label": "App Demonstration",
        "style": "A clear, real walkthrough of the actual product/app — what the screenshots show, step by "
        "step, framed as genuinely useful to watch, not just a feature list read aloud.",
    },
    "pov": {
        "label": "POV Video",
        "style": "Put the viewer directly into a relatable or intriguing situation, told from their own point "
        "of view (\"You're sitting on the bus when...\") that the product naturally resolves.",
    },
    "mystery": {
        "label": "Mystery or Curiosity",
        "style": "Open on an intriguing, unexplained event or detail that makes the viewer need to know what "
        "happened — the product is part of the explanation, revealed partway through, not upfront.",
    },
    "problem_solution": {
        "label": "Problem and Solution",
        "style": "Show a real, relatable problem the audience will recognize immediately, then demonstrate how "
        "the product actually solves it — concrete, not abstract.",
    },
}

# A real, defensible feature list for VIYO itself — used to pre-fill a
# "Promote VIYO" campaign so hook/script generation is grounded in actual
# functionality instead of inventing claims. Editable per-campaign via the
# normal target_features field once pre-filled (GET /promote-viyo-defaults
# just supplies a starting point, it doesn't lock the admin into it).
VIYO_REAL_FEATURES: list[str] = [
    "Bite-sized vertical short dramas you swipe through like a video feed",
    "Viyo AI Drama Studio: turn a script into a fully produced episode with AI-generated scenes, character "
    "voices, and burned-in captions",
    "Follow ongoing drama series and pick up new episodes as they're released",
    "Viyo Premium subscription unlocks every episode with no per-episode cost",
    "Viyo Coins let you unlock individual episodes without subscribing",
    "A Discover tab for finding trending dramas and creators",
]


class PromoteViyoDefaults(BaseModel):
    target_name: str
    target_description: str
    target_features: list[str]
    cta_text: str


@router.get("/promote-viyo-defaults", response_model=PromoteViyoDefaults, dependencies=[Depends(_require_admin)])
async def promote_viyo_defaults():
    return PromoteViyoDefaults(
        target_name="VIYO",
        target_description="A vertical short-drama app with an AI Drama Studio for creators.",
        target_features=VIYO_REAL_FEATURES,
        cta_text="Discover your next drama on VIYO",
    )


class AdFormatOut(BaseModel):
    key: str
    label: str


@router.get("/formats", response_model=list[AdFormatOut], dependencies=[Depends(_require_admin)])
async def list_formats():
    return [AdFormatOut(key=k, label=v["label"]) for k, v in AD_FORMATS.items()]


class VeoTierOut(BaseModel):
    key: str
    label: str
    price_per_sec_cents: int


@router.get("/veo-tiers", response_model=list[VeoTierOut], dependencies=[Depends(_require_admin)])
async def list_veo_tiers():
    return [VeoTierOut(key=k, label=v["label"], price_per_sec_cents=v["price_per_sec_cents"]) for k, v in VEO_TIERS.items()]


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------
class CreateCampaignRequest(BaseModel):
    user_id: str
    promote_target: str = Field(..., pattern="^(viyo|ollie_ai|other_app|website)$")
    target_name: str = ""
    target_description: str = ""
    target_features: list[str] = []
    target_audience: str = ""
    objective: str = ""
    destination_link: str = ""
    cta_text: str = ""
    format: Optional[str] = None
    duration_seconds: int = 15
    aspect_ratio: str = "9:16"
    resolution: str = "720p"
    use_veo: bool = False
    veo_tier: str = DEFAULT_VEO_TIER
    voice_gender_preference: Optional[str] = None


class CampaignOut(BaseModel):
    id: str
    user_id: str
    promote_target: str
    target_name: str
    target_description: str
    target_features: list[str]
    target_audience: str
    objective: str
    destination_link: str
    cta_text: str
    format: Optional[str] = None
    recommended_format: Optional[str] = None
    duration_seconds: int
    aspect_ratio: str
    resolution: str
    use_veo: bool
    veo_tier: str
    voice_gender_preference: Optional[str] = None
    voice_name: Optional[str] = None
    selected_hook_id: Optional[str] = None
    script: Optional[list[dict]] = None
    status: str
    video_url: Optional[str] = None
    bunny_video_id: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration_actual_seconds: Optional[int] = None
    cost_usd_cents: int
    error: Optional[str] = None
    created_at: str


def _row_to_campaign(row: dict) -> CampaignOut:
    return CampaignOut(
        id=row["id"],
        user_id=row["user_id"],
        promote_target=row["promote_target"],
        target_name=row.get("target_name") or "",
        target_description=row.get("target_description") or "",
        target_features=row.get("target_features") or [],
        target_audience=row.get("target_audience") or "",
        objective=row.get("objective") or "",
        destination_link=row.get("destination_link") or "",
        cta_text=row.get("cta_text") or "",
        format=row.get("format"),
        recommended_format=row.get("recommended_format"),
        duration_seconds=row.get("duration_seconds") or 15,
        aspect_ratio=row.get("aspect_ratio") or "9:16",
        resolution=row.get("resolution") or "720p",
        use_veo=bool(row.get("use_veo")),
        veo_tier=row.get("veo_tier") or DEFAULT_VEO_TIER,
        voice_gender_preference=row.get("voice_gender_preference"),
        voice_name=row.get("voice_name"),
        selected_hook_id=row.get("selected_hook_id"),
        script=row.get("script"),
        status=row.get("status") or "draft",
        video_url=row.get("video_url"),
        bunny_video_id=row.get("bunny_video_id"),
        thumbnail_url=row.get("thumbnail_url"),
        duration_actual_seconds=row.get("duration_actual_seconds"),
        cost_usd_cents=row.get("cost_usd_cents") or 0,
        error=row.get("error"),
        created_at=row.get("created_at") or "",
    )


def _get_campaign(campaign_id: str) -> dict:
    try:
        rows = supabase_admin.table("ad_campaigns").select("*").eq("id", campaign_id).limit(1).execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load campaign: {e}")
    if not rows:
        raise HTTPException(status_code=404, detail="Campaign not found.")
    return rows[0]


@router.post("/campaign", response_model=CampaignOut, dependencies=[Depends(_require_admin)])
async def create_campaign(req: CreateCampaignRequest):
    _require_configured()
    if req.duration_seconds not in AD_DURATIONS:
        raise HTTPException(status_code=400, detail=f"duration_seconds must be one of {AD_DURATIONS}.")
    if req.aspect_ratio not in AD_ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail=f"aspect_ratio must be one of {AD_ASPECT_RATIOS}.")
    if req.resolution not in AD_RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"resolution must be one of {AD_RESOLUTIONS}.")
    if req.format is not None and req.format not in AD_FORMATS:
        raise HTTPException(status_code=400, detail=f"format must be one of {list(AD_FORMATS)} or omitted.")
    if req.veo_tier not in VEO_TIERS:
        raise HTTPException(status_code=400, detail=f"veo_tier must be one of {list(VEO_TIERS)}.")
    if req.use_veo and req.aspect_ratio == "1:1":
        raise HTTPException(
            status_code=400,
            detail="Veo-animated scenes don't support a 1:1 square output (only 16:9/9:16) — "
            "turn off Veo animation for a square campaign, or pick 9:16/16:9.",
        )
    row = {
        "user_id": req.user_id,
        "promote_target": req.promote_target,
        "target_name": req.target_name,
        "target_description": req.target_description,
        "target_features": req.target_features,
        "target_audience": req.target_audience,
        "objective": req.objective,
        "destination_link": req.destination_link,
        "cta_text": req.cta_text,
        "format": req.format,
        "duration_seconds": req.duration_seconds,
        "aspect_ratio": req.aspect_ratio,
        "resolution": req.resolution,
        "use_veo": req.use_veo,
        "veo_tier": req.veo_tier,
        "voice_gender_preference": req.voice_gender_preference,
        "status": "draft",
        "cost_usd_cents": 0,
    }
    try:
        inserted = supabase_admin.table("ad_campaigns").insert(row).execute().data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not create campaign: {e}")
    return _row_to_campaign(inserted)


class UpdateCampaignRequest(BaseModel):
    target_name: Optional[str] = None
    target_description: Optional[str] = None
    target_features: Optional[list[str]] = None
    target_audience: Optional[str] = None
    objective: Optional[str] = None
    destination_link: Optional[str] = None
    cta_text: Optional[str] = None
    format: Optional[str] = None
    duration_seconds: Optional[int] = None
    aspect_ratio: Optional[str] = None
    resolution: Optional[str] = None
    use_veo: Optional[bool] = None
    veo_tier: Optional[str] = None
    voice_gender_preference: Optional[str] = None


@router.put("/campaign/{campaign_id}", response_model=CampaignOut, dependencies=[Depends(_require_admin)])
async def update_campaign(campaign_id: str, req: UpdateCampaignRequest):
    _require_configured()
    _get_campaign(campaign_id)
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    if "duration_seconds" in patch and patch["duration_seconds"] not in AD_DURATIONS:
        raise HTTPException(status_code=400, detail=f"duration_seconds must be one of {AD_DURATIONS}.")
    if "aspect_ratio" in patch and patch["aspect_ratio"] not in AD_ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail=f"aspect_ratio must be one of {AD_ASPECT_RATIOS}.")
    if "resolution" in patch and patch["resolution"] not in AD_RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"resolution must be one of {AD_RESOLUTIONS}.")
    if "veo_tier" in patch and patch["veo_tier"] not in VEO_TIERS:
        raise HTTPException(status_code=400, detail=f"veo_tier must be one of {list(VEO_TIERS)}.")
    if not patch:
        return _row_to_campaign(_get_campaign(campaign_id))
    try:
        updated = (
            supabase_admin.table("ad_campaigns").update(patch).eq("id", campaign_id).execute().data[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not update campaign: {e}")
    return _row_to_campaign(updated)


@router.get("/campaigns", response_model=list[CampaignOut], dependencies=[Depends(_require_admin)])
async def list_campaigns():
    _require_configured()
    try:
        rows = (
            supabase_admin.table("ad_campaigns")
            .select("*")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
            .data
        ) or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list campaigns: {e}")
    return [_row_to_campaign(r) for r in rows]


@router.get("/campaign/{campaign_id}", response_model=CampaignOut, dependencies=[Depends(_require_admin)])
async def get_campaign(campaign_id: str):
    _require_configured()
    return _row_to_campaign(_get_campaign(campaign_id))


@router.delete("/campaign/{campaign_id}", dependencies=[Depends(_require_admin)])
async def delete_campaign(campaign_id: str):
    """Best-effort Bunny delete (if a video was ever generated) then the
    row itself — same order/posture as studio.py's delete_studio_post.
    ad_assets/ad_hook_candidates/ad_generation_jobs/ad_performance_metrics
    cascade via each table's own `on delete cascade` foreign key."""
    _require_configured()
    campaign = _get_campaign(campaign_id)
    bunny_video_id = campaign.get("bunny_video_id")
    if bunny_video_id:
        try:
            from bunny_stream import BUNNY_STREAM_API_KEY, BUNNY_STREAM_LIBRARY_ID, _BUNNY_API_BASE, _configured

            if _configured():
                requests.delete(
                    f"{_BUNNY_API_BASE}/library/{BUNNY_STREAM_LIBRARY_ID}/videos/{bunny_video_id}",
                    headers={"AccessKey": BUNNY_STREAM_API_KEY},
                    timeout=15,
                )
        except Exception:
            pass
    try:
        supabase_admin.table("ad_campaigns").delete().eq("id", campaign_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete campaign: {e}")
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Assets — up to 8 per campaign, plus a reusable real-VIYO-screenshot
# library any campaign can attach from without re-uploading.
# ---------------------------------------------------------------------------
class AssetOut(BaseModel):
    id: str
    url: str
    asset_type: str
    source: str
    label: str


def _validate_and_store_image(data: bytes, content_type: Optional[str], filename: Optional[str], path_prefix: str) -> str:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_AD_ASSET_BYTES:
        raise HTTPException(
            status_code=400, detail=f"File too large — max {MAX_AD_ASSET_BYTES // (1024 * 1024)}MB."
        )
    content_type = content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported here.")
    ext = os.path.splitext(filename or "")[1] or ".jpg"
    path = f"{path_prefix}/{uuid.uuid4().hex}{ext}"
    return _upload_image(data, content_type, path)


@router.post("/campaign/{campaign_id}/assets", response_model=AssetOut, dependencies=[Depends(_require_admin)])
async def upload_campaign_asset(campaign_id: str, asset_type: str = "reference", file: UploadFile = File(...)):
    _require_configured()
    _get_campaign(campaign_id)
    try:
        count_result = (
            supabase_admin.table("ad_assets")
            .select("id", count="exact")
            .eq("campaign_id", campaign_id)
            .execute()
        )
        existing_count = count_result.count or 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check existing assets: {e}")
    if existing_count >= MAX_ASSETS_PER_CAMPAIGN:
        raise HTTPException(status_code=400, detail=f"Up to {MAX_ASSETS_PER_CAMPAIGN} assets per campaign.")

    data = await file.read()
    url = _validate_and_store_image(data, file.content_type, file.filename, f"ads/{campaign_id}")
    try:
        inserted = (
            supabase_admin.table("ad_assets")
            .insert({"campaign_id": campaign_id, "url": url, "asset_type": asset_type, "source": "uploaded"})
            .execute()
            .data[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save asset: {e}")
    return AssetOut(id=inserted["id"], url=url, asset_type=asset_type, source="uploaded", label="")


@router.get("/campaign/{campaign_id}/assets", response_model=list[AssetOut], dependencies=[Depends(_require_admin)])
async def list_campaign_assets(campaign_id: str):
    _require_configured()
    try:
        rows = (
            supabase_admin.table("ad_assets")
            .select("*")
            .eq("campaign_id", campaign_id)
            .order("created_at")
            .execute()
            .data
        ) or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list assets: {e}")
    return [AssetOut(id=r["id"], url=r["url"], asset_type=r["asset_type"], source=r["source"], label=r.get("label") or "") for r in rows]


@router.delete("/asset/{asset_id}", dependencies=[Depends(_require_admin)])
async def delete_campaign_asset(asset_id: str):
    _require_configured()
    try:
        supabase_admin.table("ad_assets").delete().eq("id", asset_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete asset: {e}")
    return {"deleted": True}


@router.get("/viyo-asset-library", response_model=list[AssetOut], dependencies=[Depends(_require_admin)])
async def list_viyo_asset_library():
    _require_configured()
    try:
        rows = (
            supabase_admin.table("ad_viyo_asset_library").select("*").order("created_at", desc=True).execute().data
        ) or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list the Viyo asset library: {e}")
    return [AssetOut(id=r["id"], url=r["url"], asset_type="screenshot", source="viyo_library", label=r.get("label") or "") for r in rows]


@router.post("/viyo-asset-library", response_model=AssetOut, dependencies=[Depends(_require_admin)])
async def upload_viyo_asset_library_item(label: str = "", file: UploadFile = File(...)):
    """A real screenshot of the live VIYO app, uploaded once and reusable
    across every future "Promote VIYO" campaign without re-uploading —
    this is the only source of truth "Promote VIYO" mode's image
    conditioning draws from; nothing here is auto-captured or invented."""
    _require_configured()
    data = await file.read()
    url = _validate_and_store_image(data, file.content_type, file.filename, "ads/viyo-library")
    try:
        inserted = (
            supabase_admin.table("ad_viyo_asset_library").insert({"url": url, "label": label}).execute().data[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save library asset: {e}")
    return AssetOut(id=inserted["id"], url=url, asset_type="screenshot", source="viyo_library", label=label)


@router.post(
    "/campaign/{campaign_id}/assets/from-library/{library_id}",
    response_model=AssetOut,
    dependencies=[Depends(_require_admin)],
)
async def attach_library_asset(campaign_id: str, library_id: str):
    _require_configured()
    _get_campaign(campaign_id)
    try:
        lib_rows = supabase_admin.table("ad_viyo_asset_library").select("*").eq("id", library_id).limit(1).execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load library asset: {e}")
    if not lib_rows:
        raise HTTPException(status_code=404, detail="Library asset not found.")
    lib = lib_rows[0]
    try:
        count_result = (
            supabase_admin.table("ad_assets").select("id", count="exact").eq("campaign_id", campaign_id).execute()
        )
        if (count_result.count or 0) >= MAX_ASSETS_PER_CAMPAIGN:
            raise HTTPException(status_code=400, detail=f"Up to {MAX_ASSETS_PER_CAMPAIGN} assets per campaign.")
        inserted = (
            supabase_admin.table("ad_assets")
            .insert({"campaign_id": campaign_id, "url": lib["url"], "asset_type": "screenshot", "source": "viyo_library"})
            .execute()
            .data[0]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not attach library asset: {e}")
    return AssetOut(id=inserted["id"], url=lib["url"], asset_type="screenshot", source="viyo_library", label=lib.get("label") or "")


def _load_campaign_assets(campaign_id: str) -> list[dict]:
    try:
        return (
            supabase_admin.table("ad_assets").select("*").eq("campaign_id", campaign_id).order("created_at").execute().data
        ) or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load campaign assets: {e}")


# ---------------------------------------------------------------------------
# Hook-First Engine
# ---------------------------------------------------------------------------
# Transparent, server-computed scoring: Gemini self-rates each candidate
# 1-10 on five separate dimensions (never a single opaque "score"), and
# the actual ranking total is a plain weighted sum computed here, not
# trusted from the model's own output — these weights are the whole
# "configurable scoring system" the feature asks for; change them and
# every future hook-generation call picks up the new weighting
# immediately, no prompt change needed.
HOOK_SCORE_WEIGHTS = {
    "attention": 0.30,
    "curiosity": 0.25,
    "emotional": 0.20,
    "relevance": 0.15,
    "transition": 0.10,
}


def _weighted_hook_score(attention: int, curiosity: int, emotional: int, relevance: int, transition: int) -> float:
    """Pure function, deliberately — this is the one piece of the Hook-
    First Engine that's cheap and meaningful to unit test without a live
    Gemini call. Scores are clamped to [1, 10] before weighting so a
    malformed/out-of-range model response can't skew a ranking."""
    a = max(1, min(10, attention))
    c = max(1, min(10, curiosity))
    e = max(1, min(10, emotional))
    r = max(1, min(10, relevance))
    t = max(1, min(10, transition))
    return round(
        a * HOOK_SCORE_WEIGHTS["attention"]
        + c * HOOK_SCORE_WEIGHTS["curiosity"]
        + e * HOOK_SCORE_WEIGHTS["emotional"]
        + r * HOOK_SCORE_WEIGHTS["relevance"]
        + t * HOOK_SCORE_WEIGHTS["transition"],
        2,
    )


class HookCandidateSchema(BaseModel):
    hook_text: str
    angle: str
    rationale: str
    score_attention: int
    score_curiosity: int
    score_emotional: int
    score_relevance: int
    score_transition: int


class HookGenerationSchema(BaseModel):
    recommended_format: str
    hooks: list[HookCandidateSchema]


_HOOK_PROMPT = """You are a short-form video creative strategist specializing in opening hooks — the single \
biggest lever on whether a viewer keeps watching past the first 1-3 seconds.

PRODUCT/APP TO PROMOTE: {target_name}
DESCRIPTION: {target_description}
REAL FEATURES (do not invent anything beyond these): {features}
TARGET AUDIENCE: {audience}
OBJECTIVE: {objective}
FORMAT: {format_label} — {format_style}

Generate exactly 5 distinct opening-hook candidates for a short marketing video in this format. Each hook is \
ONLY the opening 1-3 seconds — a visual, an action, a line of dialogue, a question, or an intriguing statement. \
NOT the whole script.

Hard rules for every hook:
- No logo animation, no brand introduction, no empty establishing shot — start with something actually \
interesting happening or being said.
- No generic creator openings like "Hey guys, check out this app."
- The video must feel like entertaining content, not an advertisement — a story, a relatable moment, an \
intriguing situation, or a genuine demonstration. The product can appear naturally later; it does not need to \
be in the hook itself.
- Each of the 5 hooks must take a genuinely different angle (vary: mystery/curiosity, bold statement, relatable \
problem, striking visual, direct question, surprising fact) — not five minor variations of the same idea.
- Never fabricate a real customer testimonial or claim a feature the product doesn't have.

For each hook, also self-score 1-10 (10 = strongest) on exactly these five dimensions, honestly and with real \
variation between candidates — not every hook should score the same:
- score_attention: would this stop a scroll in the first second?
- score_curiosity: does it create a real question the viewer needs answered?
- score_emotional: does it carry real emotional pull (tension, humor, surprise, relatability)?
- score_relevance: will THIS specific audience care about this hook?
- score_transition: how naturally can the rest of the video grow out of this opening?

Also recommend the single best-fitting format for this product/audience/objective from this exact list: {format_keys}.
"""


def _build_hook_prompt(campaign: dict, format_key: Optional[str]) -> str:
    if format_key and format_key in AD_FORMATS:
        format_label = AD_FORMATS[format_key]["label"]
        format_style = AD_FORMATS[format_key]["style"]
    else:
        format_label = "Not chosen yet — recommend one"
        format_style = "Open to any of the formats listed below; pick whichever fits best."
    features = "; ".join(campaign.get("target_features") or []) or "none specified"
    return _HOOK_PROMPT.format(
        target_name=campaign.get("target_name") or "this product",
        target_description=campaign.get("target_description") or "",
        features=features,
        audience=campaign.get("target_audience") or "general short-form video viewers",
        objective=campaign.get("objective") or "drive interest and engagement",
        format_label=format_label,
        format_style=format_style,
        format_keys=", ".join(AD_FORMATS.keys()),
    )


class HookOut(BaseModel):
    id: str
    hook_text: str
    angle: str
    rationale: str
    score_attention: int
    score_curiosity: int
    score_emotional: int
    score_relevance: int
    score_transition: int
    score_total: float
    rank: int
    selected: bool


class GenerateHooksResponse(BaseModel):
    hooks: list[HookOut]
    recommended_format: str
    cost_usd_cents: int


@router.post("/campaign/{campaign_id}/hooks", response_model=GenerateHooksResponse, dependencies=[Depends(_require_admin)])
async def generate_hooks(campaign_id: str):
    _require_configured()
    campaign = _get_campaign(campaign_id)
    prompt = _build_hook_prompt(campaign, campaign.get("format"))
    _check_daily_cap(_estimate_text_cost_cents(prompt))

    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HookGenerationSchema,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Hook generation failed: {e}")

    parsed = response.parsed
    if parsed is None or not parsed.hooks:
        raise HTTPException(status_code=502, detail="Gemini returned a response Ads Studio couldn't parse.")

    cost_cents = _text_cost_cents(response.usage_metadata)
    _log_cost(campaign_id, "ad_hooks", cost_cents)

    # Replace this campaign's previous hook set rather than piling up
    # duplicates on every regenerate — a campaign only ever has one
    # live set of 5 candidates at a time.
    try:
        supabase_admin.table("ad_hook_candidates").delete().eq("campaign_id", campaign_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clear previous hooks: {e}")

    scored = []
    for h in parsed.hooks:
        total = _weighted_hook_score(
            h.score_attention, h.score_curiosity, h.score_emotional, h.score_relevance, h.score_transition
        )
        scored.append((h, total))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    inserted_rows = []
    for rank, (h, total) in enumerate(scored, start=1):
        row = {
            "campaign_id": campaign_id,
            "hook_text": h.hook_text,
            "angle": h.angle,
            "rationale": h.rationale,
            "score_attention": max(1, min(10, h.score_attention)),
            "score_curiosity": max(1, min(10, h.score_curiosity)),
            "score_emotional": max(1, min(10, h.score_emotional)),
            "score_relevance": max(1, min(10, h.score_relevance)),
            "score_transition": max(1, min(10, h.score_transition)),
            "score_total": total,
            "rank": rank,
            "selected": False,
        }
        inserted_rows.append(row)
    try:
        inserted = supabase_admin.table("ad_hook_candidates").insert(inserted_rows).execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save hook candidates: {e}")

    recommended_format = parsed.recommended_format if parsed.recommended_format in AD_FORMATS else next(iter(AD_FORMATS))
    try:
        supabase_admin.table("ad_campaigns").update({
            "recommended_format": recommended_format,
            "cost_usd_cents": (campaign.get("cost_usd_cents") or 0) + cost_cents,
        }).eq("id", campaign_id).execute()
    except Exception as e:
        print(f"[WARN] Could not save recommended_format for campaign {campaign_id}: {e}")

    hooks_out = [
        HookOut(
            id=r["id"], hook_text=r["hook_text"], angle=r["angle"], rationale=r["rationale"],
            score_attention=r["score_attention"], score_curiosity=r["score_curiosity"],
            score_emotional=r["score_emotional"], score_relevance=r["score_relevance"],
            score_transition=r["score_transition"], score_total=r["score_total"], rank=r["rank"], selected=r["selected"],
        )
        for r in inserted
    ]
    return GenerateHooksResponse(hooks=hooks_out, recommended_format=recommended_format, cost_usd_cents=cost_cents)


@router.get("/campaign/{campaign_id}/hooks", response_model=list[HookOut], dependencies=[Depends(_require_admin)])
async def list_hooks(campaign_id: str):
    _require_configured()
    try:
        rows = (
            supabase_admin.table("ad_hook_candidates").select("*").eq("campaign_id", campaign_id).order("rank").execute().data
        ) or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list hooks: {e}")
    return [
        HookOut(
            id=r["id"], hook_text=r["hook_text"], angle=r["angle"], rationale=r["rationale"],
            score_attention=r["score_attention"], score_curiosity=r["score_curiosity"],
            score_emotional=r["score_emotional"], score_relevance=r["score_relevance"],
            score_transition=r["score_transition"], score_total=r["score_total"], rank=r["rank"], selected=r["selected"],
        )
        for r in rows
    ]


@router.post(
    "/campaign/{campaign_id}/hooks/{hook_id}/select", response_model=CampaignOut, dependencies=[Depends(_require_admin)]
)
async def select_hook(campaign_id: str, hook_id: str):
    """Picks one of the 5 candidates — or call with hook_id="auto" to
    select whichever already has rank=1 (the top-scored candidate),
    matching the "or let the system recommend one" requirement. The
    score is explicitly not proof of real performance; selecting here
    never claims otherwise — see the UI copy, not this endpoint."""
    _require_configured()
    _get_campaign(campaign_id)
    try:
        if hook_id == "auto":
            top = (
                supabase_admin.table("ad_hook_candidates")
                .select("id")
                .eq("campaign_id", campaign_id)
                .eq("rank", 1)
                .limit(1)
                .execute()
                .data
            )
            if not top:
                raise HTTPException(status_code=400, detail="No hooks generated yet for this campaign.")
            hook_id = top[0]["id"]
        else:
            hook_rows = supabase_admin.table("ad_hook_candidates").select("id").eq("id", hook_id).eq("campaign_id", campaign_id).execute().data
            if not hook_rows:
                raise HTTPException(status_code=404, detail="Hook not found for this campaign.")

        supabase_admin.table("ad_hook_candidates").update({"selected": False}).eq("campaign_id", campaign_id).execute()
        supabase_admin.table("ad_hook_candidates").update({"selected": True}).eq("id", hook_id).execute()
        updated = supabase_admin.table("ad_campaigns").update({"selected_hook_id": hook_id}).eq("id", campaign_id).execute().data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not select hook: {e}")
    return _row_to_campaign(updated)


# ---------------------------------------------------------------------------
# Script / scene-plan generation
# ---------------------------------------------------------------------------
def _scene_count_for_duration(duration_seconds: int) -> int:
    if duration_seconds <= 8:
        return 2
    if duration_seconds <= 15:
        return 3
    if duration_seconds <= 30:
        return 5
    if duration_seconds <= 45:
        return 6
    return 8


class AdSceneSchema(BaseModel):
    order: int
    seconds: float
    visual_description: str
    camera_shot: str
    dialogue_or_vo: str
    caption_text: str
    is_cta: bool


class AdScriptSchema(BaseModel):
    scenes: list[AdSceneSchema]


_SCRIPT_PROMPT = """You are writing the full scene-by-scene plan for a {duration}-second {format_label} marketing \
video, built around this already-chosen opening hook. Do not change the hook — expand it into a complete video.

CHOSEN HOOK (this must be scene 1's content): {hook_text}

PRODUCT/APP: {target_name} — {target_description}
REAL FEATURES (do not invent anything beyond these): {features}
AUDIENCE: {audience}
OBJECTIVE: {objective}
FORMAT STYLE: {format_style}
CALL TO ACTION (must be the final scene): {cta_text}

Write exactly {scene_count} scenes totaling {duration} seconds. Apply this pacing:
- Scene 1 (0-3s): the hook above, staged as an interesting visual/action/line — no logo, no intro, nothing wasted.
- Early-middle scenes: continue the story/demonstration naturally, raising curiosity or stakes.
- Later scenes: deliver the payoff and introduce the product where it fits naturally — never forced in before \
the content has earned it.
- Final scene: the call to action above, concise and relevant, not a hard sales pitch.

For each scene give:
- order: 1-based scene number
- seconds: this scene's target duration (all scenes must sum to {duration})
- visual_description: what's on screen
- camera_shot: e.g. "close-up", "medium shot", "wide establishing shot"
- dialogue_or_vo: the exact spoken line or voiceover for this scene (can be empty for a pure visual beat)
- caption_text: the on-screen caption burned into the video for this scene (usually the same as dialogue_or_vo, \
condensed if long)
- is_cta: true only for the final scene

Keep dialogue natural and in-character for the format — never a generic ad-read, never "Hey guys, check out this app."
"""


def _build_script_prompt(campaign: dict, hook_text: str) -> str:
    format_key = campaign.get("format") or campaign.get("recommended_format") or "story_driven"
    format_info = AD_FORMATS.get(format_key, AD_FORMATS["story_driven"])
    duration = campaign.get("duration_seconds") or 15
    features = "; ".join(campaign.get("target_features") or []) or "none specified"
    return _SCRIPT_PROMPT.format(
        duration=duration,
        format_label=format_info["label"],
        format_style=format_info["style"],
        hook_text=hook_text,
        target_name=campaign.get("target_name") or "this product",
        target_description=campaign.get("target_description") or "",
        features=features,
        audience=campaign.get("target_audience") or "general short-form video viewers",
        objective=campaign.get("objective") or "drive interest and engagement",
        cta_text=campaign.get("cta_text") or "Try it today",
        scene_count=_scene_count_for_duration(duration),
    )


def _normalize_scene_durations(scenes: list[dict], target_total: float) -> list[dict]:
    total = sum(max(0.1, s["seconds"]) for s in scenes) or 1.0
    scale = target_total / total
    for s in scenes:
        s["seconds"] = round(max(0.5, s["seconds"] * scale), 1)
    return scenes


class GenerateScriptResponse(BaseModel):
    scenes: list[dict]
    cost_usd_cents: int


@router.post("/campaign/{campaign_id}/script", response_model=GenerateScriptResponse, dependencies=[Depends(_require_admin)])
async def generate_script(campaign_id: str):
    _require_configured()
    campaign = _get_campaign(campaign_id)
    hook_id = campaign.get("selected_hook_id")
    if not hook_id:
        raise HTTPException(status_code=400, detail="Select a hook first.")
    hook_rows = supabase_admin.table("ad_hook_candidates").select("hook_text").eq("id", hook_id).execute().data
    if not hook_rows:
        raise HTTPException(status_code=400, detail="Selected hook no longer exists — pick another.")
    hook_text = hook_rows[0]["hook_text"]

    prompt = _build_script_prompt(campaign, hook_text)
    _check_daily_cap(_estimate_text_cost_cents(prompt))
    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=AdScriptSchema),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Script generation failed: {e}")

    parsed = response.parsed
    if parsed is None or not parsed.scenes:
        raise HTTPException(status_code=502, detail="Gemini returned a response Ads Studio couldn't parse.")

    cost_cents = _text_cost_cents(response.usage_metadata)
    _log_cost(campaign_id, "ad_script", cost_cents)

    scenes = sorted((s.model_dump() for s in parsed.scenes), key=lambda s: s["order"])
    scenes = _normalize_scene_durations(scenes, float(campaign.get("duration_seconds") or 15))
    if scenes:
        scenes[-1]["is_cta"] = True

    try:
        supabase_admin.table("ad_campaigns").update({
            "script": scenes,
            "cost_usd_cents": (campaign.get("cost_usd_cents") or 0) + cost_cents,
            "status": "script_ready",
        }).eq("id", campaign_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save script: {e}")

    return GenerateScriptResponse(scenes=scenes, cost_usd_cents=cost_cents)


class SaveScriptRequest(BaseModel):
    scenes: list[AdSceneSchema]


@router.put("/campaign/{campaign_id}/script", response_model=CampaignOut, dependencies=[Depends(_require_admin)])
async def save_script(campaign_id: str, req: SaveScriptRequest):
    """Persists admin edits to the generated scene plan before
    generation — the script stays fully editable right up until
    /generate actually renders it."""
    _require_configured()
    _get_campaign(campaign_id)
    if not req.scenes:
        raise HTTPException(status_code=400, detail="A script needs at least one scene.")
    for s in req.scenes:
        if s.seconds <= 0:
            raise HTTPException(status_code=400, detail=f"Scene {s.order} has a non-positive duration.")
    scenes = [s.model_dump() for s in sorted(req.scenes, key=lambda s: s.order)]
    try:
        updated = (
            supabase_admin.table("ad_campaigns")
            .update({"script": scenes, "status": "script_ready"})
            .eq("id", campaign_id)
            .execute()
            .data[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save script: {e}")
    return _row_to_campaign(updated)


# ---------------------------------------------------------------------------
# Video generation — parameterized ffmpeg (Studio's own render helpers are
# hardcoded to 9:16/1080x1920 since Drama episodes are always vertical;
# these two are the same filters with width/height as real parameters).
# ---------------------------------------------------------------------------
def _ad_target_dimensions(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    short = 1080 if resolution == "1080p" else 720
    long_edge = int(round(short * 16 / 9))
    if aspect_ratio == "9:16":
        return short, long_edge
    if aspect_ratio == "16:9":
        return long_edge, short
    return short, short  # 1:1


def _render_ad_scene_clip(
    image_path: str, audio_path: str, cues: list[tuple[str, float, float]], duration: float,
    pan: str, width: int, height: int, out_path: str,
) -> None:
    frames = max(1, int(round(duration * _FFMPEG_FPS)))
    zoom_step = (_ZOOM_MAX - _ZOOM_MIN) / frames
    upscale_w, upscale_h = width * 2, height * 2
    if pan == "left_right":
        x_expr = f"(iw-iw/zoom)*(on/{frames})"
    else:
        x_expr = "iw/2-(iw/zoom/2)"
    vf = (
        f"scale={upscale_w}:{upscale_h},zoompan=z='min(zoom+{zoom_step:.8f},{_ZOOM_MAX})':x='{x_expr}':"
        f"y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={_FFMPEG_FPS}"
    )
    vf += _build_caption_filters(cues, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0])
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path, "-i", audio_path,
        "-vf", vf, "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path,
    ]
    _run_ffmpeg(cmd, "Ad scene render")


def _render_ad_scene_clip_from_video(
    video_path: str, audio_path: str, cues: list[tuple[str, float, float]], duration: float,
    width: int, height: int, out_path: str,
) -> None:
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={_FFMPEG_FPS}"
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
    _run_ffmpeg(cmd, "Ad scene render (Veo)")


def _render_ad_scene_clip_veo_native_audio(
    video_path: str, cues: list[tuple[str, float, float]], width: int, height: int, out_path: str,
) -> None:
    """For a Veo-animated scene using Veo's OWN generated audio (see
    _generate_ad_scene_video's native_audio param) instead of a
    separately-generated TTS track layered on top — no second audio
    input, no -stream_loop/-t stretch-to-fit, since the whole point is
    to keep Veo's video and the speech it generated together exactly as
    it rendered them. The scene's clip length is whatever Veo actually
    returned (passed back as the scene's duration by the caller), not
    the script's originally planned seconds — looping or trimming
    generated dialogue would chop or repeat actual spoken words, which
    is worse than a campaign's total runtime drifting slightly from the
    requested duration."""
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={_FFMPEG_FPS}"
    vf += _build_caption_filters(cues, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0])
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out_path,
    ]
    _run_ffmpeg(cmd, "Ad scene render (Veo native audio)")


def _generate_ad_scene_image(prompt: str, reference_urls: list[str], campaign_id: str) -> str:
    _check_daily_cap(GEMINI_IMAGE_COST_USD_CENTS)
    reference_parts = [_fetch_image_part(u) for u in reference_urls[:4]]
    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[*reference_parts, types.Part.from_text(text=prompt)] if reference_parts else prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ad scene image generation failed: {e}")
    image_bytes, mime_type = _extract_image(response)
    ext = "png" if "png" in mime_type else "jpg"
    url = _upload_image(image_bytes, mime_type, f"ads/{campaign_id}/scenes/{uuid.uuid4().hex}.{ext}")
    _log_cost(campaign_id, "ad_scene_image", GEMINI_IMAGE_COST_USD_CENTS)
    return url


def _generate_ad_scene_video(
    image_url: str, visual_description: str, camera_shot: str, aspect_ratio: str, campaign_id: str,
    veo_tier: str = DEFAULT_VEO_TIER, dialogue: Optional[str] = None,
) -> tuple[str, int]:
    """Reuses studio.py's _fetch_veo_image unmodified. Veo only accepts
    "16:9"/"9:16" (never called for a 1:1 campaign — create_campaign/
    update_campaign already reject use_veo+1:1 together).

    When [dialogue] is given, the prompt asks Veo to speak that exact
    line itself rather than animating silently — real lip-adjacent
    sync, since Veo renders the video and its own audio together, at
    the cost of not controlling the exact voice or guaranteeing word-
    for-word delivery the way a separate Gemini TTS pass does. The
    caller (_run_ad_generation) keeps this native audio track instead
    of layering TTS on top when dialogue is set.

    Does NOT pass generate_audio in GenerateVideosConfig — confirmed
    live that it 502s immediately ("generate_audio parameter is only
    supported in Gemini Enterprise Agent Platform mode, not in Gemini
    Developer API mode"), since this codebase authenticates with a
    plain API key (genai.Client(api_key=...)), not Vertex AI/Enterprise.
    Per studio.py's own generate_scene_video (proven working, also
    never sets this field), Veo's audio generation is always on
    regardless on the Developer API tier — the field exists only to
    explicitly toggle it on the Enterprise tier, which isn't this
    project's access level. Returns (video_url, actual_duration_seconds)
    — the duration matters because the caller must NOT loop/trim a clip
    that has real generated speech in it (see
    _render_ad_scene_clip_veo_native_audio's own docstring)."""
    tier = VEO_TIERS.get(veo_tier, VEO_TIERS[DEFAULT_VEO_TIER])
    veo_aspect = "16:9" if aspect_ratio == "16:9" else "9:16"
    duration = VEO_ALLOWED_DURATIONS[-1] if dialogue else VEO_ALLOWED_DURATIONS[0]
    cost_cents = duration * tier["price_per_sec_cents"]
    _check_daily_cap(cost_cents)

    veo_image = _fetch_veo_image(image_url)
    if dialogue:
        prompt = (
            "Animate this image into a short video clip for a marketing video, with the subject speaking "
            f"this exact line out loud, naturally: \"{dialogue}\"\n\n"
            f"Camera shot: {camera_shot or 'medium'} shot. What's happening: {visual_description}\n\n"
            "Natural lip movement and delivery matching the speech. No text, no captions, no watermark."
        )
    else:
        prompt = (
            "Animate this image into a short video clip for a marketing video. "
            f"Camera shot: {camera_shot or 'medium'} shot. What's happening: {visual_description}\n\n"
            "Subtle, natural motion — keep framing and subject consistent with the reference image. "
            "No text, no captions, no watermark."
        )
    try:
        operation = _gemini_client.models.generate_videos(
            model=tier["model"],
            prompt=prompt,
            image=veo_image,
            config=types.GenerateVideosConfig(aspect_ratio=veo_aspect, duration_seconds=duration),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Veo video generation failed to start: {e}")

    waited = 0
    while not operation.done:
        if waited >= VEO_MAX_POLL_SECONDS:
            raise HTTPException(status_code=504, detail="Veo is still rendering after several minutes.")
        import time as _time
        _time.sleep(VEO_POLL_INTERVAL_SECONDS)
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

    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "veo_output.mp4")
        try:
            _gemini_client.files.download(file=generated_videos[0].video, destination=video_path)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Could not download Veo's generated video: {e}")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
    url = _upload_video_preview(video_bytes, f"ads/{campaign_id}/scenes/{uuid.uuid4().hex}_veo.mp4")
    _log_cost(campaign_id, "ad_scene_video", cost_cents)
    return url, duration


def _generate_ad_line_audio(text: str, voice_name: str, campaign_id: str) -> str:
    _check_daily_cap(GEMINI_TTS_LINE_COST_USD_CENTS)
    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name))
                ),
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ad line audio generation failed: {e}")
    pcm_bytes, sample_rate = _extract_audio_pcm(response)
    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)
    url = _upload_audio(wav_bytes, f"ads/{campaign_id}/lines/{uuid.uuid4().hex}.wav")
    _log_cost(campaign_id, "ad_line_audio", GEMINI_TTS_LINE_COST_USD_CENTS)
    return url


def _update_job(job_id: str, **fields) -> None:
    fields["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        supabase_admin.table("ad_generation_jobs").update(fields).eq("id", job_id).execute()
    except Exception as e:
        print(f"[WARN] Could not update ad_generation_jobs {job_id}: {e}")


def _run_ad_generation(campaign_id: str, job_id: str) -> None:
    """The actual generation pipeline — runs in FastAPI's BackgroundTasks
    thread (this is a plain `def`, not `async def`, specifically so
    FastAPI dispatches it off the event loop into a worker thread; it's
    full of blocking subprocess/requests calls that would otherwise stall
    every other in-flight request). Any exception anywhere in here is
    caught at the bottom and recorded on both the job and campaign rows
    rather than propagating — there's no HTTP response left to raise
    into once a background task is running."""
    try:
        campaign = _get_campaign(campaign_id)
        scenes = campaign.get("script") or []
        if not scenes:
            raise HTTPException(status_code=400, detail="No script to generate from.")
        assets = _load_campaign_assets(campaign_id)
        reference_urls = [a["url"] for a in assets]
        aspect_ratio = campaign.get("aspect_ratio") or "9:16"
        resolution = campaign.get("resolution") or "720p"
        use_veo = bool(campaign.get("use_veo"))
        veo_tier = campaign.get("veo_tier") or DEFAULT_VEO_TIER
        width, height = _ad_target_dimensions(aspect_ratio, resolution)

        voice_name = campaign.get("voice_name")
        if not voice_name:
            voice_name = _auto_assign_voice(campaign.get("voice_gender_preference") or "", "adult", set())
            supabase_admin.table("ad_campaigns").update({"voice_name": voice_name}).eq("id", campaign_id).execute()

        _update_job(job_id, status="running", progress_pct=1)

        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory() as tmp:
            clip_paths = []
            total = len(scenes)
            for i, scene in enumerate(scenes):
                dialogue = (scene.get("dialogue_or_vo") or "").strip()
                planned_seconds = float(scene.get("seconds") or 3.0)
                caption = (scene.get("caption_text") or dialogue).strip()

                image_prompt = (
                    f"Cinematic marketing video scene. {aspect_ratio} aspect ratio. "
                    f"Camera shot: {scene.get('camera_shot') or 'medium'} shot. "
                    f"What's happening: {scene.get('visual_description') or ''}\n\n"
                    "No text, no watermark, no captions baked into the image."
                )
                image_url = _generate_ad_scene_image(image_prompt, reference_urls, campaign_id)

                clip_path = os.path.join(tmp, f"scene_{i}_clip.mp4")

                if use_veo and dialogue:
                    # Veo generates the video AND voices the dialogue
                    # itself — kept as-is (not layered with a separate
                    # TTS track) for real lip-adjacent sync. The clip's
                    # length follows Veo's own fixed duration, not the
                    # script's planned seconds — see
                    # _render_ad_scene_clip_veo_native_audio's docstring
                    # for why looping/trimming generated speech would be
                    # worse than a little total-runtime drift.
                    video_url, veo_duration = _generate_ad_scene_video(
                        image_url, scene.get("visual_description") or "", scene.get("camera_shot") or "",
                        aspect_ratio, campaign_id, veo_tier=veo_tier, dialogue=dialogue,
                    )
                    video_path = os.path.join(tmp, f"scene_{i}_veo.mp4")
                    _download_to_file(video_url, video_path)
                    scene_duration = float(veo_duration)
                    cues = _scene_cues([caption], [scene_duration]) if caption else []
                    _render_ad_scene_clip_veo_native_audio(video_path, cues, width, height, clip_path)
                else:
                    audio_path = os.path.join(tmp, f"scene_{i}_audio.wav")
                    if dialogue:
                        line_url = _generate_ad_line_audio(dialogue, voice_name, campaign_id)
                        raw_path = os.path.join(tmp, f"scene_{i}_raw.wav")
                        _download_to_file(line_url, raw_path)
                        audio_duration = _ffprobe_duration(raw_path)
                        if audio_duration < planned_seconds:
                            pad_path = os.path.join(tmp, f"scene_{i}_pad.wav")
                            _render_silence(planned_seconds - audio_duration, pad_path)
                            _concat_audio([raw_path, pad_path], audio_path)
                            scene_duration = planned_seconds
                        else:
                            audio_path = raw_path
                            scene_duration = audio_duration
                        cues = _scene_cues([caption], [scene_duration]) if caption else []
                    else:
                        _render_silence(planned_seconds, audio_path)
                        scene_duration = planned_seconds
                        cues = _scene_cues([caption], [scene_duration]) if caption else []

                    if use_veo:
                        # No dialogue — a pure visual/establishing beat.
                        # Veo's own ambient audio is kept (no speech to
                        # protect from looping), so the silence track
                        # above is unused here; it still gets rendered
                        # for simplicity/symmetry with the Ken Burns path.
                        video_url, _ = _generate_ad_scene_video(
                            image_url, scene.get("visual_description") or "", scene.get("camera_shot") or "",
                            aspect_ratio, campaign_id, veo_tier=veo_tier,
                        )
                        video_path = os.path.join(tmp, f"scene_{i}_veo.mp4")
                        _download_to_file(video_url, video_path)
                        _render_ad_scene_clip_from_video(video_path, audio_path, cues, scene_duration, width, height, clip_path)
                    else:
                        image_path = os.path.join(tmp, f"scene_{i}.png")
                        _download_to_file(image_url, image_path)
                        pan = "left_right" if i % 2 else "center"
                        _render_ad_scene_clip(image_path, audio_path, cues, scene_duration, pan, width, height, clip_path)

                _validate_clip(clip_path, f"Scene {i + 1}")
                clip_paths.append(clip_path)
                _update_job(job_id, progress_pct=int((i + 1) / total * 85))

            assembled_path = os.path.join(tmp, "assembled.mp4")
            labels = [f"Scene {i + 1}" for i in range(len(scenes))]
            _concat_clips(clip_paths, assembled_path, labels=labels)

            final_path = assembled_path
            music_url = campaign.get("music_url")
            if music_url:
                music_path = os.path.join(tmp, "music_src")
                _download_to_file(music_url, music_path)
                mixed_path = os.path.join(tmp, "with_music.mp4")
                _mix_background_music(assembled_path, music_path, _ffprobe_duration(assembled_path), mixed_path)
                final_path = mixed_path
            _update_job(job_id, progress_pct=90)

            duration_actual = int(round(_ffprobe_duration(final_path)))
            thumb_path = os.path.join(tmp, "thumb.jpg")
            _extract_thumbnail(final_path, thumb_path)
            with open(thumb_path, "rb") as f:
                thumb_bytes = f.read()
            thumbnail_url = _upload_image(thumb_bytes, "image/jpeg", f"ads/{campaign_id}/thumb_{uuid.uuid4().hex}.jpg")

            video_id = _upload_finished_video_to_bunny(final_path, campaign.get("target_name") or "Viyo Ad")
            video_url = _bunny_playback_url(video_id)
            _update_job(job_id, progress_pct=97)

        supabase_admin.table("ad_campaigns").update({
            "video_url": video_url,
            "bunny_video_id": video_id,
            "thumbnail_url": thumbnail_url,
            "duration_actual_seconds": duration_actual,
            "status": "ready",
            "error": None,
        }).eq("id", campaign_id).execute()
        _update_job(job_id, status="succeeded", progress_pct=100)
    except HTTPException as e:
        supabase_admin.table("ad_campaigns").update({"status": "failed", "error": e.detail}).eq("id", campaign_id).execute()
        _update_job(job_id, status="failed", error=str(e.detail))
    except Exception as e:
        supabase_admin.table("ad_campaigns").update({"status": "failed", "error": str(e)}).eq("id", campaign_id).execute()
        _update_job(job_id, status="failed", error=str(e))


class JobOut(BaseModel):
    id: str
    campaign_id: str
    stage: str
    status: str
    progress_pct: int
    error: Optional[str] = None


@router.post("/campaign/{campaign_id}/generate", response_model=JobOut, dependencies=[Depends(_require_admin)])
async def generate_video(campaign_id: str, background_tasks: BackgroundTasks):
    """Kicks off the real generation pipeline as a background task and
    returns immediately with a job id to poll — see this module's own
    docstring for why (no job-queue infra, Veo alone can take minutes
    per scene). Calling this again on any campaign that already has a
    script (ready, failed, or mid-generation) is how retry works —
    nothing here depends on a previous run's partial state, since
    assets/hooks/script all live on rows this endpoint never touches."""
    _require_configured()
    campaign = _get_campaign(campaign_id)
    if not campaign.get("script"):
        raise HTTPException(status_code=400, detail="Generate a script first.")
    aspect_ratio = campaign.get("aspect_ratio") or "9:16"
    resolution = campaign.get("resolution") or "720p"
    if aspect_ratio not in AD_ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail=f"aspect_ratio must be one of {AD_ASPECT_RATIOS}.")
    if resolution not in AD_RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"resolution must be one of {AD_RESOLUTIONS}.")
    if campaign.get("use_veo") and aspect_ratio == "1:1":
        raise HTTPException(status_code=400, detail="Veo doesn't support 1:1 — turn off Veo animation or change aspect ratio.")

    try:
        supabase_admin.table("ad_campaigns").update({"status": "generating", "error": None}).eq("id", campaign_id).execute()
        job = (
            supabase_admin.table("ad_generation_jobs")
            .insert({"campaign_id": campaign_id, "stage": "video", "status": "pending", "progress_pct": 0})
            .execute()
            .data[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not start generation: {e}")

    background_tasks.add_task(_run_ad_generation, campaign_id, job["id"])
    return JobOut(id=job["id"], campaign_id=campaign_id, stage="video", status="pending", progress_pct=0)


@router.get("/campaign/{campaign_id}/job", response_model=Optional[JobOut], dependencies=[Depends(_require_admin)])
async def get_latest_job(campaign_id: str):
    _require_configured()
    try:
        rows = (
            supabase_admin.table("ad_generation_jobs")
            .select("*")
            .eq("campaign_id", campaign_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
        ) or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check job status: {e}")
    if not rows:
        return None
    r = rows[0]
    return JobOut(id=r["id"], campaign_id=r["campaign_id"], stage=r["stage"], status=r["status"], progress_pct=r["progress_pct"], error=r.get("error"))


# ---------------------------------------------------------------------------
# Performance learning — manual/uploaded metrics only. There is no live
# TikTok/Instagram/YouTube integration anywhere in this codebase, so this
# never claims to see real platform analytics on its own; every row here
# is either typed in by the admin or comes from a file they uploaded
# elsewhere, recorded for later comparison across hooks/campaigns.
# ---------------------------------------------------------------------------
class RecordPerformanceRequest(BaseModel):
    hook_id: Optional[str] = None
    retention_1s: Optional[float] = None
    retention_3s: Optional[float] = None
    retention_5s: Optional[float] = None
    avg_watch_seconds: Optional[float] = None
    completion_rate: Optional[float] = None
    ctr: Optional[float] = None
    installs: Optional[int] = None
    notes: str = ""


class PerformanceOut(BaseModel):
    id: str
    campaign_id: str
    hook_id: Optional[str] = None
    retention_1s: Optional[float] = None
    retention_3s: Optional[float] = None
    retention_5s: Optional[float] = None
    avg_watch_seconds: Optional[float] = None
    completion_rate: Optional[float] = None
    ctr: Optional[float] = None
    installs: Optional[int] = None
    source: str
    notes: str
    recorded_at: str


@router.post(
    "/campaign/{campaign_id}/performance", response_model=PerformanceOut, dependencies=[Depends(_require_admin)]
)
async def record_performance(campaign_id: str, req: RecordPerformanceRequest):
    _require_configured()
    _get_campaign(campaign_id)
    row = {"campaign_id": campaign_id, "source": "manual", **req.model_dump()}
    try:
        inserted = supabase_admin.table("ad_performance_metrics").insert(row).execute().data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not record performance: {e}")
    return PerformanceOut(**inserted)


@router.get(
    "/campaign/{campaign_id}/performance", response_model=list[PerformanceOut], dependencies=[Depends(_require_admin)]
)
async def list_performance(campaign_id: str):
    _require_configured()
    try:
        rows = (
            supabase_admin.table("ad_performance_metrics")
            .select("*")
            .eq("campaign_id", campaign_id)
            .order("recorded_at", desc=True)
            .execute()
            .data
        ) or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list performance records: {e}")
    return [PerformanceOut(**r) for r in rows]
