"""
Viyo AI Ads Studio: marketing-video generation for VIYO itself, Ollie AI,
or any other app/product/website — a second, complementary feature to
Viyo Studio (studio.py)'s AI Drama Maker, sharing its Gemini/Veo/ffmpeg/
Bunny pipeline rather than duplicating it.

Production rules (see ad_brand.py, ad_compose.py, ad_qc.py): the advertiser's brand name, website and call
to action are the only ones that ever appear (invented ones are corrected before rendering); captions, real
assets, the brand mark and the end card each have their own area of the frame; and the finished file is
checked (format, sound, black/blank pictures, cut-off text, overlaps, the website actually shown, the end
card) before an ad is delivered. One that can't be fixed automatically is "needs_correction", with reasons.

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
import copy
import datetime
import os
import re
import shutil
import uuid
from typing import Optional

import ad_brand
import ad_claims
import ad_compose
import ad_qc

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
from bunny_stream import _thumbnail_url as _bunny_thumbnail_url

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
    # Google shut down the Veo 3.1 preview models on 22 Oct 2026; every
    # tier now uses its replacement, Gemini Omni 1.2 Flash (720p, about
    # $0.07/s). Tier keys are kept so saved campaigns stay valid.
    "lite": {"model": "gemini-omni-1.2-flash", "price_per_sec_cents": 7, "label": "Omni Flash"},
    "fast": {"model": "gemini-omni-1.2-flash", "price_per_sec_cents": 7, "label": "Omni Flash"},
    "standard": {"model": "gemini-omni-1.2-flash", "price_per_sec_cents": 7, "label": "Omni Flash"},
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


# ---------------------------------------------------------------------------
# Who pays for a run. Admin runs: the shared real-money daily cap (studio.py's _check_daily_cap), as
# before. Creator runs (ads_maker.py): paid with Studio credits held up front; every provider call is
# counted here, and a run can never spend more than the credits it held.
# ---------------------------------------------------------------------------
import contextvars as _contextvars

_creator_run: "_contextvars.ContextVar[Optional[dict]]" = _contextvars.ContextVar("ads_creator_run", default=None)


class BudgetExceeded(HTTPException):
    def __init__(self):
        super().__init__(status_code=402, detail="This step would cost more than the credits held for it. "
                                                 "Nothing more was spent; unused credits are returned.")


def _cap(cents) -> None:
    run = _creator_run.get()
    if run is None:
        _check_daily_cap(cents)
    elif run["spent_cents"] + float(cents) > run["budget_cents"]:
        raise BudgetExceeded()


def _spent(campaign_id: str, call_type: str, cents) -> None:
    _log_cost(campaign_id, call_type, cents)
    run = _creator_run.get()
    if run is not None:
        run["spent_cents"] += float(cents)


def _objective_text(campaign: dict) -> str:
    """The objective line of the hook and script prompts, plus the creative concept the advertiser chose (from a
    daily idea), when there is one. A concept is direction only: it never adds claims (the facts guard still
    applies to everything written)."""
    text = campaign.get("objective") or "drive interest and engagement"
    brief = (campaign.get("creative_brief") or "").strip()
    if brief:
        text += ("\nCREATIVE CONCEPT (chosen by the advertiser; follow it, but it adds no facts or claims): "
                 + brief[:1200])
    return text


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

# What an uploaded asset actually IS, for any campaign/product — not
# specific to any one brand. This is the real distinction section 2 of
# the Ads Studio spec asks for: a "product asset" (screenshot/product
# photo/logo) is real source material that must render exactly as
# uploaded, never silently reinterpreted by a generative model; a
# "creative" asset (plain "other") is fair game as loose visual/style
# conditioning for AI-generated scenes. "reference" is kept accepted
# (not rejected) purely for backward compatibility with assets uploaded
# before this distinction existed — those stay treated as creative-only
# (the safe default when the asset's real nature isn't known), never
# promoted to exact-reproduction automatically.
# What is being advertised. Only changes how the prompts describe it; nothing in the pipeline is tied to
# one advertiser. (viyo / ollie_ai are kept so existing campaigns stay valid; both are apps.)
PRODUCT_KINDS: dict[str, str] = {
    "viyo": "a mobile app", "ollie_ai": "a mobile app", "other_app": "a mobile app", "website": "a website",
    "digital_product": "a digital product (software or an online platform)",
    "physical_product": "a physical product", "service": "a service", "business": "a business",
    "personal_brand": "a personal brand", "film": "a film or show", "music": "a music project",
    "event": "an event", "creator_project": "a creator's project",
}
_KIND_PATTERN = "^(" + "|".join(PRODUCT_KINDS) + ")$"
# Common kinds of thing within each category, offered in the app (any short text is accepted).
PRODUCT_SUBTYPES: dict[str, list[str]] = {
    "physical_product": ["drink", "food", "clothing", "shoes", "electronics", "furniture", "cosmetics", "accessories"],
    "service": ["restaurant", "hotel", "education", "cleaning", "transport", "consulting", "health", "beauty salon"],
    "digital_product": ["software", "online platform", "online course", "game"],
    "event": ["concert", "conference", "festival", "workshop"],
    "music": ["single", "album", "tour"],
}
# Products whose look (packaging, label, logo) must be shown exactly: real product photos are composited
# into the ad, never redrawn by an image model.
PHYSICAL_KINDS = {"physical_product"}
CHARACTER_MODES = ("auto", "custom", "reuse", "none")
AD_STYLES = {
    "auto": "Pick the style that suits the product best.",
    "product_cinematic": "Product-led cinematic shots: no people needed; the product is the hero.",
    "character": "A character-led commercial: one consistent character carries the story.",
    "lifestyle": "Lifestyle: the product in real everyday moments of the people it's for.",
    "demo": "Demonstration/explainer: show exactly how it works or what it does.",
    "social": "Native social-media ad: fast, direct, feels like a post, not a TV spot.",
}
_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


def _clean_profile(p: dict) -> dict:
    """Validates the product profile fields of a campaign (create or update) and returns them cleaned.
    Facts are the advertiser's word and are kept exactly; only shape and size are checked."""
    out = {}
    if p.get("product_subtype") is not None:
        out["product_subtype"] = str(p["product_subtype"]).strip()[:40]
    if p.get("facts") is not None:
        facts = {}
        for key in ad_claims.FACT_FIELDS:
            items = [str(x).strip()[:300] for x in (p["facts"].get(key) or []) if str(x).strip()]
            if len(items) > 12:
                raise HTTPException(status_code=400, detail=f"Up to 12 {key.replace('_', ' ')} please.")
            facts[key] = items
        out["facts"] = facts
    if p.get("brand_colors") is not None:
        colors = [c.strip() for c in p["brand_colors"] if c and c.strip()]
        if len(colors) > 4 or any(not _HEX.match(c) for c in colors):
            raise HTTPException(status_code=400, detail="Brand colours: up to 4, as hex like #FF6A3D.")
        out["brand_colors"] = ["#" + c.lstrip("#").upper() for c in colors]
    if p.get("social_links") is not None:
        links = [str(x).strip()[:120] for x in p["social_links"] if str(x).strip()]
        if len(links) > 5:
            raise HTTPException(status_code=400, detail="Up to 5 social accounts.")
        out["social_links"] = links
    if p.get("contact") is not None:
        c = {k: str(v).strip()[:120] for k, v in (p["contact"] or {}).items() if k in ("phone", "email", "address") and v}
        if c.get("email") and not _EMAIL_RE.match(c["email"]):
            raise HTTPException(status_code=400, detail=f'"{c["email"]}" isn\'t a valid email address.')
        if c.get("phone") and len(re.sub(r"\D", "", c["phone"])) < 7:
            raise HTTPException(status_code=400, detail=f'"{c["phone"]}" isn\'t a valid phone number.')
        out["contact"] = c
    if p.get("character_mode") is not None:
        if p["character_mode"] not in CHARACTER_MODES:
            raise HTTPException(status_code=400, detail=f"character_mode must be one of {CHARACTER_MODES}.")
        out["character_mode"] = p["character_mode"]
    if p.get("character") is not None:
        out["character"] = {k: str(v).strip()[:300] for k, v in (p["character"] or {}).items()
                            if k in CHARACTER_FIELDS and v is not None}
    if p.get("ad_style") is not None:
        if p["ad_style"] not in AD_STYLES:
            raise HTTPException(status_code=400, detail=f"ad_style must be one of {list(AD_STYLES)}.")
        out["ad_style"] = p["ad_style"]
    return out


CHARACTER_FIELDS = ("name", "role", "appearance", "clothing", "personality", "voice_style", "gender", "age",
                    "portrait", "voice_name", "saved_id")
AD_ASSET_TYPES = ("screenshot", "product_photo", "logo", "other")
EXACT_ASSET_TYPES = {"screenshot", "product_photo", "logo"}

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
    promote_target: str = Field(..., pattern=_KIND_PATTERN)
    target_name: str = ""
    brand_name: str = ""          # when the brand and the product have different names
    product_subtype: str = ""
    facts: Optional[dict] = None              # {features, benefits, prices, selling_points, specs}: lists
    brand_colors: Optional[list[str]] = None
    social_links: Optional[list[str]] = None
    contact: Optional[dict] = None            # {phone, email, address}
    character_mode: str = "auto"
    character: Optional[dict] = None
    ad_style: str = "auto"
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
    brand_name: str = ""
    brand_id: Optional[str] = None     # the saved brand (My Brands) this campaign belongs to
    track: str = ""                     # Campaign Universe: which kind of ad in the brand's ongoing campaign
    creative_brief: str = ""            # the concept chosen from a daily idea, if any
    product_subtype: str = ""
    facts: dict = {}
    brand_colors: list[str] = []
    social_links: list[str] = []
    contact: dict = {}
    character_mode: str = "auto"
    character: dict = {}
    ad_style: str = "auto"
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
    # Post-render quality check of the finished file (ad_qc.inspect) and, when it needs correction, the
    # video it produced (never presented as finished: video_url stays empty until the checks pass).
    quality_report: Optional[dict] = None
    preview_video_url: Optional[str] = None


def _row_to_campaign(row: dict) -> CampaignOut:
    return CampaignOut(
        id=row["id"],
        user_id=row["user_id"],
        promote_target=row["promote_target"],
        target_name=row.get("target_name") or "",
        target_description=row.get("target_description") or "",
        target_features=row.get("target_features") or [],
        brand_name=row.get("brand_name") or "",
        brand_id=row.get("brand_id"),
        track=row.get("track") or "",
        creative_brief=row.get("creative_brief") or "",
        product_subtype=row.get("product_subtype") or "",
        facts=row.get("facts") or {},
        brand_colors=row.get("brand_colors") or [],
        social_links=row.get("social_links") or [],
        contact=row.get("contact") or {},
        character_mode=row.get("character_mode") or "auto",
        character=row.get("character") or {},
        ad_style=row.get("ad_style") or "auto",
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
        quality_report=row.get("quality_report"),
        preview_video_url=row.get("preview_video_url"),
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
    if req.destination_link.strip() and not ad_brand.parse_url(req.destination_link)[0]:
        raise HTTPException(status_code=400, detail=f'"{req.destination_link}" isn\'t a valid website address.')
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
        **({"brand_name": req.brand_name} if req.brand_name else {}),
        **_clean_profile({"product_subtype": req.product_subtype or None, "facts": req.facts,
                          "brand_colors": req.brand_colors, "social_links": req.social_links, "contact": req.contact,
                          "character_mode": req.character_mode if req.character_mode != "auto" else None,
                          "character": req.character, "ad_style": req.ad_style if req.ad_style != "auto" else None}),
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
    promote_target: Optional[str] = Field(None, pattern=_KIND_PATTERN)
    target_name: Optional[str] = None
    brand_name: Optional[str] = None
    product_subtype: Optional[str] = None
    facts: Optional[dict] = None
    brand_colors: Optional[list[str]] = None
    social_links: Optional[list[str]] = None
    contact: Optional[dict] = None
    character_mode: Optional[str] = None
    character: Optional[dict] = None
    ad_style: Optional[str] = None
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
    if (patch.get("destination_link") or "").strip() and not ad_brand.parse_url(patch["destination_link"])[0]:
        raise HTTPException(status_code=400, detail=f'"{patch["destination_link"]}" isn\'t a valid website address.')
    patch.update(_clean_profile({k: patch.get(k) for k in ("product_subtype", "facts", "brand_colors", "social_links",
                                                           "contact", "character_mode", "character", "ad_style")}))
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
async def upload_campaign_asset(campaign_id: str, asset_type: str = "other", file: UploadFile = File(...)):
    _require_configured()
    _get_campaign(campaign_id)
    if asset_type not in AD_ASSET_TYPES:
        asset_type = "other"
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

WHAT IS ADVERTISED: {target_name} ({product_kind})
{facts}DESCRIPTION: {target_description}
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
        product_kind=PRODUCT_KINDS.get(campaign.get("promote_target") or "", "a product"),
        facts=ad_brand.prompt_facts(ad_brand.BrandFacts.from_campaign(campaign)),
        target_description=campaign.get("target_description") or "",
        features=features,
        audience=campaign.get("target_audience") or "general short-form video viewers",
        objective=_objective_text(campaign),
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
    _cap(_estimate_text_cost_cents(prompt))

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
    _spent(campaign_id, "ad_hooks", cost_cents)

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
    # Empty string (never omitted — Gemini structured output needs a
    # concrete value) unless this scene should show one of the real
    # uploaded assets AS-IS, in which case it's that asset's token
    # (e.g. "A2") from the ASSET LIST block in the prompt below. See
    # _run_ad_generation: a scene with this set renders via
    # deterministic compositing on the real asset file, never through
    # an image/video model, so it must be a genuine match. Defaulted so
    # an older client's save-script request (sent before this field
    # existed) still validates instead of 422ing.
    featured_asset_id: str = ""
    # What the scene is for in the story (SCENE_PURPOSES), how it leads into the next one, which verified facts
    # its words rely on, and whether the product / the ad's character are on screen. Defaulted so older saved
    # scripts and clients still validate.
    purpose: str = ""
    transition: str = ""
    fact_ids: list[str] = []
    features_product: bool = False
    features_character: bool = False


SCENE_PURPOSES = ("hook", "story", "demo", "product_intro", "selling_point", "cta")


class AdCharacterSchema(BaseModel):
    """A fictional advertising character (never a real person)."""
    name: str = ""
    role: str = ""            # e.g. "a student who discovers the drink"
    appearance: str = ""      # face, hair, build: specific, so every scene shows the same person
    clothing: str = ""
    personality: str = ""
    voice_style: str = ""
    gender: str = ""          # "female" / "male" / "" (for the voice)
    age: str = ""             # "child" / "teen" / "adult" / "senior"


class AdScriptSchema(BaseModel):
    uses_character: bool = False
    character: AdCharacterSchema = AdCharacterSchema()
    scenes: list[AdSceneSchema]


_SCRIPT_PROMPT = """You are writing the full scene-by-scene plan for a {duration}-second {format_label} marketing \
video, built around this already-chosen opening hook. Do not change the hook — expand it into a complete video.

CHOSEN HOOK (this must be scene 1's content): {hook_text}

WHAT IS ADVERTISED: {target_name} ({product_kind}) — {target_description}
{facts}REAL FEATURES (do not invent anything beyond these): {features}
AUDIENCE: {audience}
OBJECTIVE: {objective}
FORMAT STYLE: {format_style}
{asset_block}
Write exactly {scene_count} scenes totaling {duration} seconds. Apply this pacing:
- Scene 1 (0-3s): the hook above, staged as an interesting visual/action/line — no logo, no intro, nothing wasted.
- Early-middle scenes: continue the story/demonstration naturally, raising curiosity or stakes.
- Later scenes: deliver the payoff and introduce the product where it fits naturally — never forced in before \
the content has earned it.
- Final scene: the closing card. On screen it shows the logo, the exact brand name, the exact website and the \
exact call to action given above (added automatically — don't describe text for it). Its voiceover is one short \
line that says the brand name and, only if a website is given above, that exact website.
- Every scene must be about this product: what it does, who it helps, the real problem it solves. No unrelated \
stock scenes, no scenery that doesn't move the message forward.
- Any scene that shows the app, the website or a screen must use one of the real screenshots (featured_asset_id) \
when there is one; never describe an invented interface or a blank/black screen.

Every scene needs a clear purpose and must lead visually into the next. Use these purposes, in an order that \
makes a complete ad: hook (scene 1), then story and/or demo, product_intro (where the product is introduced), \
one or more selling_point scenes (each built on verified facts), and cta (the final scene).
{style}{character}
For each scene give:
- order: 1-based scene number
- seconds: this scene's target duration (all scenes must sum to {duration})
- visual_description: what's on screen
- camera_shot: e.g. "close-up", "medium shot", "wide establishing shot"
- dialogue_or_vo: the exact spoken line or voiceover for this scene (can be empty for a pure visual beat)
- caption_text: the on-screen caption for this scene: at most 8 words, the key phrase of the line (never a website)
- is_cta: true only for the final scene
- featured_asset_id: {asset_field_instruction}
- purpose: one of hook, story, demo, product_intro, selling_point, cta
- transition: how this scene's last image leads into the next scene (a matching action, object, colour or move)
- fact_ids: ids of the verified facts this scene's words rely on (empty if it states no facts)
- features_product: true if the product itself is on screen
- features_character: true if the ad's character is on screen

Keep dialogue natural and in-character for the format — never a generic ad-read, never "Hey guys, check out this app."
"""

_ASSET_BLOCK_TEMPLATE = """
REAL UPLOADED ASSETS AVAILABLE (shown above as images, in this exact order) — these are actual product \
screenshots/photos/logos the user supplied, not something to reinterpret:
{asset_list}

Any scene that should show one of these AS-IS — the real app/website screenshot being demonstrated, the real \
product photo, or the real logo — must set featured_asset_id to that asset's token (e.g. "A2"). A scene with \
featured_asset_id set renders the literal uploaded image directly with no AI reinterpretation, so only use a \
token when that specific asset is actually right for that scene (the screenshot that matches what's being \
described, the product actually being shown, etc.) — never reference an asset that doesn't fit. Leave it "" for \
any scene that's a generated/creative visual (atmosphere, a situation, storytelling with no specific real asset \
on screen). Not every scene needs a real asset, and some campaigns call for mostly generated scenes with the \
real asset shown only once or twice where it actually matters (e.g. a screenshot during an app demo beat, a \
product photo at the reveal, a logo at the very end) — use real assets where showing the literal real thing is \
what the scene calls for, not everywhere just because they're available.
"""


def _build_asset_block(assets: list[dict]) -> tuple[str, str]:
    """Returns (prompt block describing the assets by token, instruction
    for the featured_asset_id field) — empty strings when there are no
    assets, so a campaign with nothing uploaded gets the exact same
    prompt as before this feature existed."""
    if not assets:
        return "", '"" — this campaign has no uploaded assets, so every scene is generated/creative.'
    lines = []
    for i, a in enumerate(assets):
        token = f"A{i + 1}"
        label = (a.get("label") or "").strip() or "untitled"
        lines.append(f"{token}: type={a.get('asset_type') or 'other'}, label={label}")
    asset_list = "\n".join(lines)
    block = _ASSET_BLOCK_TEMPLATE.format(asset_list=asset_list)
    instruction = (
        'one of the asset tokens above (e.g. "A1") if this scene should show that real asset as-is, or "" for a '
        "generated/creative scene"
    )
    return block, instruction


def _build_script_prompt(campaign: dict, hook_text: str, assets: list[dict]) -> str:
    format_key = campaign.get("format") or campaign.get("recommended_format") or "story_driven"
    format_info = AD_FORMATS.get(format_key, AD_FORMATS["story_driven"])
    duration = campaign.get("duration_seconds") or 15
    features = "; ".join(campaign.get("target_features") or []) or "none specified"
    asset_block, asset_field_instruction = _build_asset_block(assets)
    return _SCRIPT_PROMPT.format(
        duration=duration,
        format_label=format_info["label"],
        format_style=format_info["style"],
        hook_text=hook_text,
        target_name=campaign.get("target_name") or "this product",
        product_kind=PRODUCT_KINDS.get(campaign.get("promote_target") or "", "a product"),
        facts=ad_brand.prompt_facts(ad_brand.BrandFacts.from_campaign(campaign))
        + ad_claims.FactSheet.from_campaign(campaign).prompt_block(),
        target_description=campaign.get("target_description") or "",
        features=features,
        audience=campaign.get("target_audience") or "general short-form video viewers",
        objective=_objective_text(campaign),
        style=_style_block(campaign),
        character=_character_block(campaign),
        scene_count=_scene_count_for_duration(duration),
        asset_block=asset_block,
        asset_field_instruction=asset_field_instruction,
    )


def _style_block(campaign: dict) -> str:
    style = campaign.get("ad_style") or "auto"
    kind = campaign.get("promote_target") or ""
    extra = ""
    if kind in PHYSICAL_KINDS:
        extra = (" The product's real photos are composited into product scenes exactly as uploaded (packaging, label "
                 "and logo unchanged), so describe the setting around the product, never a redesigned product.")
    sub = (campaign.get("product_subtype") or "").strip()
    return f"PRODUCTION STYLE: {AD_STYLES.get(style, AD_STYLES['auto'])}{f' The product is a {sub}.' if sub else ''}{extra}\n"


def _character_block(campaign: dict) -> str:
    mode = campaign.get("character_mode") or "auto"
    ch = campaign.get("character") or {}
    if mode == "none":
        return ("CHARACTER: none. Make a product-focused ad with no recurring character (hands or passers-by are fine "
                "only where needed). Set uses_character to false.\n")
    if mode in ("custom", "reuse") and (ch.get("appearance") or ch.get("name")):
        desc = "; ".join(f"{k}: {ch[k]}" for k in ("name", "role", "appearance", "clothing", "personality", "voice_style")
                         if ch.get(k))
        return (f"CHARACTER (use exactly this fictional character, the same person in every scene they appear in): {desc}. "
                "Set uses_character to true and copy these details into character.\n")
    return ("CHARACTER: decide whether this ad is better with one fictional character (a relatable person the audience "
            "sees themselves in) or without (product-led). Don't force a character in: a product-only cinematic ad is "
            "often better for physical products. If you use one, set uses_character to true and describe them in "
            "character: specific appearance (face, hair, build), clothing, personality, voice style, gender and age, "
            "so every scene can show the same person. Never a real or famous person.\n")


def _normalize_scene_durations(scenes: list[dict], target_total: float) -> list[dict]:
    total = sum(max(0.1, s["seconds"]) for s in scenes) or 1.0
    scale = target_total / total
    for s in scenes:
        s["seconds"] = round(max(0.5, s["seconds"] * scale), 1)
    return scenes


class GenerateScriptResponse(BaseModel):
    scenes: list[dict]
    cost_usd_cents: int
    notes: list[str] = []          # what was corrected (invented brand facts/claims) and story structure warnings


def story_notes(scenes: list[dict]) -> list[str]:
    """Story structure check: a hook first, the product introduced, at least one selling point when facts
    exist, a call to action last, and every scene with a purpose and a lead into the next."""
    notes = []
    purposes = [(s.get("purpose") or "").strip() for s in scenes]
    if not scenes:
        return ["The script has no scenes."]
    if purposes[0] not in ("hook", ""):
        notes.append("Scene 1 isn't a hook.")
    if purposes and any(purposes) and "product_intro" not in purposes and not any(s.get("features_product") for s in scenes):
        notes.append("The product is never introduced.")
    if purposes and purposes[-1] not in ("cta", ""):
        notes.append("The last scene isn't the call to action.")
    for i, s in enumerate(scenes[:-1]):
        if s.get("purpose") and not (s.get("transition") or "").strip():
            notes.append(f"Scene {i + 1} has no lead into scene {i + 2}.")
    return notes


def _character_from_script(campaign: dict, parsed) -> dict:
    """In auto mode, the model's character (if it chose one) becomes the campaign's character, so the scenes,
    the reference portrait and the voice all use the same one. Custom and reused characters are never
    overwritten; "none" stays none."""
    mode = campaign.get("character_mode") or "auto"
    if mode != "auto":
        return {}
    if not getattr(parsed, "uses_character", False):
        return {"character": {}}
    ch = parsed.character.model_dump()
    return {"character": {k: str(v)[:300] for k, v in ch.items() if v}}


def _resolve_featured_asset_tokens(scenes: list[dict], assets: list[dict]) -> list[dict]:
    """Maps each scene's featured_asset_id (a short token like "A2" —
    see _build_asset_block) back to the real ad_assets row id, so
    everything downstream (_run_ad_generation) only ever deals in real
    ids and never has to know the token scheme existed. An
    unrecognized/hallucinated token (or one from a stale/edited asset
    list) safely falls back to "" — a generated/creative scene — rather
    than erroring the whole script, since a wrong guess here is a
    missed opportunity, not a correctness bug worth failing over."""
    token_to_id = {f"A{i + 1}": a["id"] for i, a in enumerate(assets)}
    for s in scenes:
        token = (s.get("featured_asset_id") or "").strip()
        s["featured_asset_id"] = token_to_id.get(token, "")
    return scenes


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
    assets = _load_campaign_assets(campaign_id)

    prompt = _build_script_prompt(campaign, hook_text, assets)
    # A small flat per-image buffer on top of the text-only estimate —
    # _estimate_text_cost_cents only counts prompt characters, not the
    # vision tokens each uploaded asset adds once shown to the model
    # below. This is only a pre-flight budget-cap check; the real spend
    # logged after the call (_text_cost_cents) always reads the actual
    # usage_metadata regardless.
    _cap(_estimate_text_cost_cents(prompt) + len(assets))
    # Gemini actually SEES each uploaded asset (not just a text label)
    # so it can judge which real screenshot/photo genuinely fits which
    # scene — "understand the selected product and use the supplied
    # assets appropriately" only works if the model looks at them.
    # Interleaving "Asset A{n}:" immediately before each image is what
    # lets it reliably echo back the right token.
    contents: list = []
    for i, a in enumerate(assets):
        contents.append(types.Part.from_text(text=f"Asset A{i + 1} ({a.get('asset_type') or 'other'}):"))
        try:
            contents.append(_fetch_image_part(_resolve(a["url"])))
        except HTTPException:
            contents.pop()  # text label with no image behind it is worse than omitting the asset entirely
    contents.append(types.Part.from_text(text=prompt))
    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=AdScriptSchema),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Script generation failed: {e}")

    parsed = response.parsed
    if parsed is None or not parsed.scenes:
        raise HTTPException(status_code=502, detail="Gemini returned a response Ads Studio couldn't parse.")

    cost_cents = _text_cost_cents(response.usage_metadata)
    _spent(campaign_id, "ad_script", cost_cents)

    scenes = sorted((s.model_dump() for s in parsed.scenes), key=lambda s: s["order"])
    character_update = _character_from_script(campaign, parsed)
    scenes = _normalize_scene_durations(scenes, float(campaign.get("duration_seconds") or 15))
    scenes = _resolve_featured_asset_tokens(scenes, assets)
    # Brand facts are the advertiser's, never the model's: any invented website or brand-name variant in
    # the captions, voiceover or visual descriptions is corrected now, before anyone reviews the script.
    scenes, _brand_changes = ad_brand.sanitize_script(scenes, ad_brand.BrandFacts.from_campaign(campaign))
    # Claims: only what the advertiser supplied. Unsupported lines are dropped now, before anyone reviews it.
    scenes, _claim_changes = ad_claims.enforce_script(scenes, ad_claims.FactSheet.from_campaign(campaign))
    notes = _brand_changes + _claim_changes + story_notes(scenes)

    try:
        supabase_admin.table("ad_campaigns").update({
            **character_update,
            "script": scenes,
            "cost_usd_cents": (campaign.get("cost_usd_cents") or 0) + cost_cents,
            "status": "script_ready",
        }).eq("id", campaign_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save script: {e}")

    return GenerateScriptResponse(scenes=scenes, cost_usd_cents=cost_cents, notes=notes)


class SaveScriptRequest(BaseModel):
    scenes: list[AdSceneSchema]


@router.put("/campaign/{campaign_id}/script", response_model=CampaignOut, dependencies=[Depends(_require_admin)])
async def save_script(campaign_id: str, req: SaveScriptRequest):
    """Persists admin edits to the generated scene plan before
    generation — the script stays fully editable right up until
    /generate actually renders it."""
    _require_configured()
    campaign = _get_campaign(campaign_id)
    if not req.scenes:
        raise HTTPException(status_code=400, detail="A script needs at least one scene.")
    for s in req.scenes:
        if s.seconds <= 0:
            raise HTTPException(status_code=400, detail=f"Scene {s.order} has a non-positive duration.")
    scenes = [s.model_dump() for s in sorted(req.scenes, key=lambda s: s.order)]
    scenes, _brand_changes = ad_brand.sanitize_script(scenes, ad_brand.BrandFacts.from_campaign(campaign))
    scenes, _claim_changes = ad_claims.enforce_script(scenes, ad_claims.FactSheet.from_campaign(campaign))
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
    vf += _build_caption_filters(cues, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0], canvas_width=width)
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
    vf += _build_caption_filters(cues, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0], canvas_width=width)
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
    vf += _build_caption_filters(cues, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0], canvas_width=width)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out_path,
    ]
    _run_ffmpeg(cmd, "Ad scene render (Veo native audio)")


def _reference_part(ref: str):
    """A reference image for the image model: a local file (e.g. the character portrait made in this run) or a
    link (private references are signed first)."""
    if os.path.exists(ref):
        with open(ref, "rb") as f:
            data = f.read()
        mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return types.Part.from_bytes(data=data, mime_type=mime)
    return _fetch_image_part(_resolve(ref) or ref)


def _generate_ad_scene_image(prompt: str, reference_urls: list[str], campaign_id: str) -> str:
    _cap(GEMINI_IMAGE_COST_USD_CENTS)
    reference_parts = [_reference_part(u) for u in reference_urls[:4] if u]
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
    _spent(campaign_id, "ad_scene_image", GEMINI_IMAGE_COST_USD_CENTS)
    return url


def _veo_duration_for_scene(target_seconds: float, has_dialogue: bool) -> int:
    """Picks which of Veo's fixed clip lengths (VEO_ALLOWED_DURATIONS,
    e.g. 4/6/8s) to render. A dialogue scene's clip length becomes the
    final scene length as-is (its real generated speech can't be looped
    or trimmed to fit afterward — see _render_ad_scene_clip_veo_native_
    audio), so it picks whichever allowed duration is closest to what
    the script actually planned for this scene. Previously this always
    returned the longest allowed duration regardless of the plan, so a
    short campaign split into several short dialogue scenes (e.g. an 8s
    ad as two ~4s scenes) rendered every scene at a flat 8s and silently
    doubled the assembled video's real length. A silent scene has
    nothing to protect from looping, so it always takes the shortest
    (cheapest) tier — the renderer stretches/loops it to the planned
    length afterward."""
    if not has_dialogue:
        return VEO_ALLOWED_DURATIONS[0]
    return min(VEO_ALLOWED_DURATIONS, key=lambda d: abs(d - target_seconds))


def _generate_ad_scene_video(
    image_url: str, visual_description: str, camera_shot: str, aspect_ratio: str, campaign_id: str,
    veo_tier: str = DEFAULT_VEO_TIER, dialogue: Optional[str] = None, target_seconds: float = 4.0,
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
    _render_ad_scene_clip_veo_native_audio's own docstring).

    [target_seconds] is the script's planned length for this scene —
    see _veo_duration_for_scene for how that maps to one of Veo's fixed
    clip lengths."""
    tier = VEO_TIERS.get(veo_tier, VEO_TIERS[DEFAULT_VEO_TIER])
    veo_aspect = "16:9" if aspect_ratio == "16:9" else "9:16"
    duration = _veo_duration_for_scene(target_seconds, bool(dialogue))
    cost_cents = duration * tier["price_per_sec_cents"]
    _cap(cost_cents)

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
    import studio_ai as _studio_ai
    try:
        video_bytes, _omni_seconds, _omni_cents = _studio_ai.omni_clip(prompt, image_url, veo_aspect, duration)
    except _studio_ai.ProviderBusy as e:
        raise HTTPException(status_code=503, detail=f"The video model is busy. Try again in a minute. ({e})")
    except (_studio_ai.PolicyBlocked, _studio_ai.ProviderError) as e:
        raise HTTPException(status_code=502, detail=f"Ad scene video failed: {e}")
    duration = int(round(_omni_seconds))
    cost_cents = _omni_cents
    url = _upload_video_preview(video_bytes, f"ads/{campaign_id}/scenes/{uuid.uuid4().hex}_veo.mp4")
    _spent(campaign_id, "ad_scene_video", cost_cents)
    return url, duration


def _generate_ad_line_audio(text: str, voice_name: str, campaign_id: str) -> str:
    _cap(GEMINI_TTS_LINE_COST_USD_CENTS)
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
    _spent(campaign_id, "ad_line_audio", GEMINI_TTS_LINE_COST_USD_CENTS)
    return url


def _build_scene_audio_track(
    dialogue: str, planned_seconds: float, caption: str, voice_name: str, campaign_id: str, tmp: str, i: int,
) -> tuple[str, float, list[tuple[str, float, float]]]:
    """Builds a scene's standalone audio track — TTS dialogue (padded
    with silence up to the planned length if the line runs short, or
    left at its real spoken length if longer, never truncated mid-
    word) or plain silence for a pure-visual beat — plus its caption
    cues. Shared by both the real-asset compositing path and the
    generated-image Ken Burns path below; neither involves Veo's own
    native audio, so both need this exact same track-building logic."""
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
    return audio_path, scene_duration, cues


def _update_job(job_id: str, **fields) -> None:
    fields["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        supabase_admin.table("ad_generation_jobs").update(fields).eq("id", job_id).execute()
    except Exception as e:
        print(f"[WARN] Could not update ad_generation_jobs {job_id}: {e}")


# ---------------------------------------------------------------------------
# Production pipeline: plan -> render -> check the finished file -> correct -> deliver.
# ---------------------------------------------------------------------------
_SCREEN_WORDS = re.compile(r"(?i)\b(app|apps|screen|phone|smartphone|website|web page|site|dashboard|interface|ui|"
                           r"laptop|tablet|browser|scrolls?|taps?|swipes?|notification|inbox|chat)\b")
MAX_IMAGE_REGENERATIONS = 2     # paid corrections per run (one image each)
MIN_END_CARD_SECONDS = 3.0


PRIVATE_BUCKET = "creator-studio"     # creators' ad files: private, served as one-hour signed links
PRIVATE_PREFIX = "private:"
SIGN_SECONDS = 3600


def _is_creator(campaign: dict) -> bool:
    return (campaign.get("created_via") or "admin") == "creator"


def _store_file(campaign: dict, data: bytes, mime: str, name: str) -> str:
    """Stores a file for a campaign. Creator campaigns: the private bucket under the creator's own folder
    (returns a "private:" reference, never a public link). Admin campaigns: the existing public buckets."""
    if _is_creator(campaign):
        path = f"ads/{campaign['user_id']}/{campaign['id']}/{name}"
        supabase_admin.storage.from_(PRIVATE_BUCKET).upload(path, data, {"content-type": mime, "upsert": "true"})
        return f"{PRIVATE_PREFIX}{PRIVATE_BUCKET}/{path}"
    if mime.startswith("video/"):
        return _upload_video_preview(data, f"ads/{campaign['id']}/{name}")
    return _upload_image(data, mime, f"ads/{campaign['id']}/{name}")


def _store_character_portrait(campaign: dict, data: bytes) -> str:
    """A character's reference portrait. Creators: in their own characters folder, not the campaign's, so a
    saved character keeps working after the campaign it came from is deleted."""
    name = f"character_{uuid.uuid4().hex}.png"
    if _is_creator(campaign):
        path = f"ads/{campaign['user_id']}/characters/{name}"
        supabase_admin.storage.from_(PRIVATE_BUCKET).upload(path, data, {"content-type": "image/png", "upsert": "true"})
        return f"{PRIVATE_PREFIX}{PRIVATE_BUCKET}/{path}"
    return _store_file(campaign, data, "image/png", name)


def _resolve(ref: Optional[str]) -> Optional[str]:
    """A link the app (or the renderer) can open: private references become one-hour signed links."""
    if not ref or not ref.startswith(PRIVATE_PREFIX):
        return ref
    bucket, _, path = ref[len(PRIVATE_PREFIX):].partition("/")
    try:
        res = supabase_admin.storage.from_(bucket).create_signed_url(path, SIGN_SECONDS)
        return (res or {}).get("signedURL") or (res or {}).get("signed_url")
    except Exception:
        return None


def _preflight(campaign: dict) -> None:
    """Checks that cost nothing, before any paid step: a website that was given must be valid (it is never
    guessed or 'fixed'), and there must be a script."""
    link = (campaign.get("destination_link") or "").strip()
    if link and not ad_brand.parse_url(link)[0]:
        raise HTTPException(status_code=400, detail=f'The website "{link}" isn\'t a valid web address. Fix it in the '
                                                    "campaign before generating; nothing was charged.")
    if not campaign.get("script"):
        raise HTTPException(status_code=400, detail="Generate a script first.")


def _fetch_local(url: str, path: str) -> None:
    """Downloads a URL (or copies a local file, used by tests and local renders)."""
    if url.startswith(PRIVATE_PREFIX):
        url = _resolve(url) or ""
        if not url:
            raise RuntimeError("private file not found")
    if url.startswith("file://") or os.path.exists(url):
        shutil.copyfile(url.removeprefix("file://"), path)
    else:
        _download_to_file(url, path)


def _prepare_assets(assets: list[dict], tmp: str) -> tuple[dict, list[str]]:
    """Downloads every campaign asset and checks it is a real, readable image. Returns ({asset id: local
    path}, problems). A broken asset is reported and never shown, rather than failing silently."""
    from PIL import Image
    paths, issues = {}, []
    for a in assets:
        label = (a.get("label") or a.get("asset_type") or "asset").strip()
        ext = os.path.splitext(a["url"].split("?")[0])[1] or ".png"
        path = os.path.join(tmp, f"asset_{a['id']}{ext}")
        try:
            _fetch_local(a["url"], path)
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                if im.width < 64 or im.height < 64:
                    raise ValueError("too small")
            paths[a["id"]] = path
        except Exception as e:
            issues.append(f'The {a.get("asset_type") or "asset"} "{label}" couldn\'t be loaded ({str(e)[:80]}).')
    return paths, issues


def _assign_product_photos(scenes: list[dict], assets: list[dict], local: dict, kind: str) -> list[dict]:
    """For a physical product, every scene that shows the product (or introduces it / sells a point about it)
    composites a REAL product photo into a generated setting (a hero shot), so packaging, label and logo are
    exactly the advertiser's. Photos are used in turn."""
    if kind not in PHYSICAL_KINDS:
        return scenes
    photos = [a["id"] for a in assets if a.get("asset_type") == "product_photo" and a["id"] in local]
    if not photos:
        return scenes
    k = 0
    for s in scenes[:-1]:
        if s.get("featured_asset_id"):
            continue
        if s.get("features_product") or (s.get("purpose") in ("product_intro", "selling_point")):
            s["product_asset_id"] = photos[k % len(photos)]
            k += 1
    return scenes


def _assign_real_assets(scenes: list[dict], assets: list[dict], local: dict) -> list[dict]:
    """Scenes about the app/website/screen show a real screenshot when the advertiser supplied one, instead
    of an AI-drawn interface (or a blank phone). Screenshots are used in turn so they don't repeat."""
    shots = [a["id"] for a in assets if a.get("asset_type") == "screenshot" and a["id"] in local]
    if not shots:
        return scenes
    used = {s.get("featured_asset_id") for s in scenes}
    queue = [a for a in shots if a not in used] + [a for a in shots if a in used]
    k = 0
    for s in scenes[:-1]:
        if (s.get("featured_asset_id") or "") or not _SCREEN_WORDS.search(s.get("visual_description") or ""):
            continue
        s["featured_asset_id"] = queue[k % len(queue)]
        s["asset_auto_assigned"] = True
        k += 1
    return scenes


def _image_prompt(scene: dict, campaign: dict, aspect_ratio: str, stricter: bool = False,
                  character: Optional[dict] = None, for_product: bool = False) -> str:
    kind = PRODUCT_KINDS.get(campaign.get("promote_target") or "", "a product")
    prompt = (
        f"Photorealistic scene for a short marketing video about {kind}. {aspect_ratio} framing. "
        f"Camera shot: {scene.get('camera_shot') or 'medium'} shot. What's happening: "
        f"{scene.get('visual_description') or ''}\n\n"
        "Absolutely no text of any kind in the image: no letters, numbers, logos, brand names, signs, labels, "
        "website addresses, captions or watermarks. If a phone, laptop or other screen is in the picture, show it "
        "from the side or back or with the screen out of view: never a blank or black screen, never an invented "
        "app interface."
    )
    if character and scene.get("features_character"):
        prompt += (f" The person in this scene is the SAME person as in the reference portrait: {character.get('appearance', '')}; "
                   f"wearing {character.get('clothing', 'the same clothes')}. Keep the face, hair and clothes identical. "
                   "A fictional person, not a real or famous one.")
    if for_product:
        prompt += (" Leave the centre of the picture clear and uncluttered: the real product photo is placed there "
                   "afterwards. Do not draw the product, any packaging, bottle, can, box or label yourself.")
    if stricter:
        prompt += " Keep every device screen completely out of view, and keep the scene bright and clearly lit."
    return prompt


def _contact_line(campaign: dict) -> str:
    """The advertiser's own social account and phone/email for the end card, exactly as supplied."""
    sheet = ad_claims.FactSheet.from_campaign(campaign)
    bits = sheet.handles[:1] + (sheet.phones[:1] or sheet.emails[:1])
    return "  ·  ".join(bits)


def _age_bucket(age: str) -> str:
    a = (age or "").lower()
    if any(w in a for w in ("child", "kid")):
        return "child"
    if "teen" in a:
        return "teen"
    if any(w in a for w in ("senior", "elder", "old")):
        return "senior"
    return "adult"


def _prepare_character(campaign: dict, scenes: list[dict], tmp: str) -> tuple[Optional[dict], Optional[str], Optional[str]]:
    """The ad's one character, if its scenes use one: (character, local reference portrait, voice). The
    portrait is made once (or reused from a saved character) and given to every scene the character is in,
    and the same voice speaks every line, so the person looks and sounds the same throughout."""
    ch = dict(campaign.get("character") or {})
    if (campaign.get("character_mode") or "auto") == "none" or not ch or not any(s.get("features_character") for s in scenes):
        return None, None, None
    portrait_ref = ch.get("portrait")
    path = os.path.join(tmp, "character_ref.png")
    if portrait_ref:
        _fetch_local(portrait_ref, path)          # a saved (reused) character's portrait
    else:
        prompt = ("Character reference portrait for a video ad: a single fictional person, head and shoulders, facing the "
                  f"camera, soft even light, plain background. {ch.get('appearance', '')}. Wearing {ch.get('clothing', '')}. "
                  f"Expression: {ch.get('personality', 'friendly')}. No text, no logos. Not a real or famous person.")
        url = _generate_ad_scene_image(prompt, [], campaign["id"])
        _fetch_local(url, path)
        with open(path, "rb") as f:
            portrait_ref = _store_character_portrait(campaign, f.read())
        ch["portrait"] = portrait_ref
    voice = ch.get("voice_name") or _auto_assign_voice(ch.get("gender") or campaign.get("voice_gender_preference") or "",
                                                       _age_bucket(ch.get("age", "")), set())
    ch["voice_name"] = voice
    try:
        supabase_admin.table("ad_campaigns").update({"character": ch}).eq("id", campaign["id"]).execute()
        if not ch.get("saved_id"):
            saved = supabase_admin.table("ad_characters").insert({
                "owner_id": campaign["user_id"], "name": ch.get("name") or "Character",
                "spec": {k: v for k, v in ch.items() if k not in ("portrait", "voice_name", "saved_id")},
                "portrait": portrait_ref, "voice_name": voice,
                **({"brand_id": campaign["brand_id"]} if campaign.get("brand_id") else {})}).execute().data
            if saved:
                ch["saved_id"] = saved[0]["id"]
                supabase_admin.table("ad_campaigns").update({"character": ch}).eq("id", campaign["id"]).execute()
    except Exception as e:      # saving for reuse is a convenience; the ad itself still uses the character
        print(f"[WARN] could not save ad character: {e}")
    return ch, path, voice


def _clip_seconds(path: str) -> float:
    return float(ad_qc.probe(path)["duration"])


def _render_spec(spec: dict, w: int, h: int, tmp: str, scale: float = 1.0) -> None:
    """(Re)renders one scene from its spec into spec["clip"], recording what was drawn where."""
    pages = ad_compose.layout_caption(spec["caption"], w, h, scale)
    timed = ad_compose.time_pages(pages, 0.0, spec["duration"])
    out = os.path.join(tmp, f"scene_{spec['index']}_{spec.get('version', 0)}.mp4")
    prefix = f"s{spec['index']}v{spec.get('version', 0)}"
    if spec["kind"] == "hero":
        spec["drawn"] = ad_compose.render_hero_scene(spec["image"], spec["product"], w, h, spec["duration"], spec["audio"],
                                                     timed, out, tmp, prefix, logo=spec.get("logo"))
    elif spec["kind"] == "video":
        spec["drawn"] = ad_compose.render_video_scene(
            spec["video"], w, h, timed, out, tmp, prefix,
            duration=None if spec.get("native_audio") else spec["duration"],
            audio=None if spec.get("native_audio") else spec["audio"])
    else:
        spec["drawn"] = ad_compose.render_image_scene(
            spec["image"], "fit" if spec["kind"] == "asset" else "cover", w, h, spec["duration"], spec["audio"],
            timed, out, tmp, prefix, logo=spec.get("logo"), pan=spec.get("pan", "center"))
    spec["clip"] = out
    spec["version"] = spec.get("version", 0) + 1


def _voice_track(text: str, planned: float, voice: str, campaign_id: str, tmp: str, name: str,
                 minimum: float = 0.0) -> tuple[str, float]:
    """A scene's sound: the voiceover (padded with silence to the planned length, never cut mid-word) or
    silence. Returns (path, seconds)."""
    target = max(planned, minimum)
    path = os.path.join(tmp, f"{name}.wav")
    if not text:
        ad_compose.silence(target, path)
        return path, target
    raw = os.path.join(tmp, f"{name}_raw.wav")
    _fetch_local(_generate_ad_line_audio(text, voice, campaign_id), raw)
    spoken = _ffprobe_duration(raw)
    if spoken >= target:
        return raw, spoken
    pad = os.path.join(tmp, f"{name}_pad.wav")
    _render_silence(target - spoken, pad)
    _concat_audio([raw, pad], path)
    return path, target


def _plan_timeline(specs: list[dict], end_card: dict) -> dict:
    t, scenes = 0.0, []
    for sp in specs:
        d = _clip_seconds(sp["clip"])
        scenes.append({"index": sp["index"], "start": t, "end": t + d, "kind": sp["kind"],
                       "has_voice": bool(sp.get("dialogue")), "drawn": sp.get("drawn", {}),
                       "product_photo": sp.get("product") if sp["kind"] == "hero" else None,
                       "character_ref": sp.get("character_ref")})
        t += d
    d = _clip_seconds(end_card["clip"])
    return {"scenes": scenes, "end_card": {"start": t, "end": t + d, "drawn": end_card.get("drawn", {}),
                                           "has_logo": bool(end_card.get("logo"))}}


def _save_campaign(campaign_id: str, fields: dict) -> None:
    """Updates the campaign; if the quality columns don't exist yet (ads_studio_quality.sql not run), saves
    the rest and folds the report summary into the error text instead of losing it."""
    try:
        supabase_admin.table("ad_campaigns").update(fields).eq("id", campaign_id).execute()
    except Exception as e:
        if "quality_report" not in str(e) and "preview_video_url" not in str(e):
            raise
        rest = {k: v for k, v in fields.items() if k not in ("quality_report", "preview_video_url")}
        if fields.get("preview_video_url"):
            rest["error"] = ((rest.get("error") or "") + f" Preview: {fields['preview_video_url']}").strip()
        supabase_admin.table("ad_campaigns").update(rest).eq("id", campaign_id).execute()


def _run_ad_generation(campaign_id: str, job_id: str) -> None:
    """Runs in a BackgroundTasks worker thread (plain def: it blocks on ffmpeg and network).

    1. Facts: the brand name, website and call to action come only from the campaign; every caption,
       voiceover line and visual description is checked against them (ad_brand) before rendering.
    2. Assets: every uploaded asset is downloaded and verified; scenes about the app/site use real
       screenshots; a broken asset is reported, never shown.
    3. Render: each scene with the layout engine (ad_compose): captions only in their band, real assets
       whole and above it, the brand mark in its corner; then an end card with the logo, exact name,
       exact website and exact call to action.
    4. Check the finished file (ad_qc) and correct what can be corrected automatically (smaller captions,
       a real screenshot or a regenerated image instead of a black/blank or text-bearing scene, a plain end
       card if the text isn't readable, a redone voiceover), then check again.
    5. Deliver only if every check passes; otherwise the campaign is "needs_correction" with the reasons
       and a preview, never presented as a finished ad."""
    try:
        campaign = _get_campaign(campaign_id)
        script = campaign.get("script") or []
        if not script:
            raise HTTPException(status_code=400, detail="No script to generate from.")
        facts = ad_brand.BrandFacts.from_campaign(campaign)
        scenes, brand_changes = ad_brand.sanitize_script(copy.deepcopy(script), facts)
        scenes, claim_changes = ad_claims.enforce_script(scenes, ad_claims.FactSheet.from_campaign(campaign))
        assets = _load_campaign_assets(campaign_id)
        assets_by_id = {a["id"]: a for a in assets}
        aspect_ratio = campaign.get("aspect_ratio") or "9:16"
        resolution = campaign.get("resolution") or "720p"
        use_veo = bool(campaign.get("use_veo"))
        veo_tier = campaign.get("veo_tier") or DEFAULT_VEO_TIER
        w, h = _ad_target_dimensions(aspect_ratio, resolution)

        voice_name = campaign.get("voice_name")
        if not voice_name:
            voice_name = _auto_assign_voice(campaign.get("voice_gender_preference") or "", "adult", set())
            supabase_admin.table("ad_campaigns").update({"voice_name": voice_name}).eq("id", campaign_id).execute()
        _update_job(job_id, status="running", progress_pct=1)

        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory() as tmp:
            local, asset_issues = _prepare_assets(assets, tmp)
            scenes = _assign_real_assets(scenes, assets, local)
            logo_id = next((a["id"] for a in assets if a.get("asset_type") == "logo" and a["id"] in local), None)
            logo_path = local.get(logo_id) if logo_id else None
            # Generated scenes may borrow a product photo/logo as style grounding; never a screenshot.
            creative_refs = [_resolve(a["url"]) for a in assets if a.get("asset_type") != "screenshot" and a["id"] in local]

            kind = campaign.get("promote_target") or ""
            scenes = _assign_product_photos(scenes, assets, local, kind)
            body, closing = scenes[:-1], scenes[-1]
            character, character_path, character_voice = _prepare_character(campaign, body, tmp)
            if character_voice:
                voice_name = character_voice          # one voice for the whole ad: the character's
            specs, total = [], len(scenes) + 1
            for i, scene in enumerate(body):
                dialogue = (scene.get("dialogue_or_vo") or "").strip()
                planned = float(scene.get("seconds") or 3.0)
                spec = {"index": i, "dialogue": dialogue, "caption": (scene.get("caption_text") or dialogue).strip(),
                        "pan": "left_right" if i % 2 else "center", "scene": scene}
                featured = assets_by_id.get((scene.get("featured_asset_id") or "").strip())
                if featured and featured["id"] in local and featured.get("asset_type") in EXACT_ASSET_TYPES:
                    spec.update(kind="asset", image=local[featured["id"]], asset_id=featured["id"])
                    spec["audio"], spec["duration"] = _voice_track(dialogue, planned, voice_name, campaign_id, tmp, f"a{i}")
                    # The brand mark stays off the hook and off a scene already showing the logo.
                    spec["logo"] = logo_path if (i > 0 and featured["id"] != logo_id) else None
                elif scene.get("product_asset_id") in local:
                    # Physical product: generated setting + the real product photo composited on top.
                    pid = scene["product_asset_id"]
                    refs = ([character_path] if character and scene.get("features_character") else [])
                    bg_url = _generate_ad_scene_image(
                        _image_prompt(scene, campaign, aspect_ratio, character=character, for_product=True), refs, campaign_id)
                    bg_path = os.path.join(tmp, f"gen_{i}.png")
                    _fetch_local(bg_url, bg_path)
                    spec.update(kind="hero", image=bg_path, product=local[pid], asset_id=pid,
                                logo=logo_path if i > 0 else None,
                                character_ref=bool(refs) if (character and scene.get("features_character")) else None)
                    spec["audio"], spec["duration"] = _voice_track(dialogue, planned, voice_name, campaign_id, tmp, f"a{i}")
                else:
                    with_character = bool(character) and bool(scene.get("features_character"))
                    refs = ([character_path] if with_character else []) + creative_refs
                    image_url = _generate_ad_scene_image(
                        _image_prompt(scene, campaign, aspect_ratio, character=character), refs[:4], campaign_id)
                    image_path = os.path.join(tmp, f"gen_{i}.png")
                    _fetch_local(image_url, image_path)
                    spec.update(kind="generated", image=image_path, logo=logo_path if i > 0 else None,
                                character_ref=with_character if (character and scene.get("features_character")) else None)
                    if use_veo:
                        video_url, veo_seconds = _generate_ad_scene_video(
                            image_url, scene.get("visual_description") or "", scene.get("camera_shot") or "",
                            aspect_ratio, campaign_id, veo_tier=veo_tier, dialogue=dialogue or None, target_seconds=planned)
                        spec["video"] = os.path.join(tmp, f"veo_{i}.mp4")
                        _fetch_local(video_url, spec["video"])
                        spec["kind"] = "video"
                        if dialogue:
                            spec["native_audio"], spec["duration"] = True, float(veo_seconds)
                            spec["audio"] = None
                        else:
                            spec["audio"], spec["duration"] = _voice_track("", planned, voice_name, campaign_id, tmp, f"a{i}")
                    else:
                        spec["audio"], spec["duration"] = _voice_track(dialogue, planned, voice_name, campaign_id, tmp, f"a{i}")
                _render_spec(spec, w, h, tmp)
                specs.append(spec)
                _update_job(job_id, progress_pct=int((i + 1) / total * 80))

            # Closing card: only the advertiser's own facts.
            closing_line = (closing.get("dialogue_or_vo") or "").strip()
            audio, seconds = _voice_track(closing_line, float(closing.get("seconds") or 3.0), voice_name, campaign_id,
                                          tmp, "end", minimum=MIN_END_CARD_SECONDS)
            background = next((sp["image"] for sp in reversed(specs) if sp["kind"] in ("generated", "video")), None)
            end_card = {"audio": audio, "duration": seconds, "logo": logo_path, "background": background,
                        "dialogue": closing_line, "plain": False}

            def render_end(version: int) -> None:
                end_card["clip"] = os.path.join(tmp, f"end_{version}.mp4")
                end_card["drawn"] = ad_compose.render_end_card(
                    facts.name, facts.display_url, facts.cta, w, h, end_card["duration"], end_card["audio"],
                    end_card["clip"], tmp, logo=end_card["logo"], background=end_card["background"], plain=end_card["plain"],
                    product=facts.product, contact=_contact_line(campaign),
                    brand_colors=campaign.get("brand_colors") or None)

            render_end(0)
            _update_job(job_id, progress_pct=84)

            def assemble(tag: str) -> str:
                assembled = os.path.join(tmp, f"assembled_{tag}.mp4")
                ad_compose.concat([sp["clip"] for sp in specs] + [end_card["clip"]], assembled)
                music_url = campaign.get("music_url")
                if not music_url:
                    return assembled
                music = os.path.join(tmp, "music_src")
                if not os.path.exists(music):
                    _fetch_local(music_url, music)
                mixed = os.path.join(tmp, f"with_music_{tag}.mp4")
                _mix_background_music(assembled, music, _ffprobe_duration(assembled), mixed)
                return mixed

            final_path = assemble("0")
            # A missing name/website is simply left out; a website that was given but isn't valid must be fixed.
            blocking = asset_issues + ([i for i in facts.issues if "valid web address" in i])
            expected = float(sum(float(s.get("seconds") or 0) for s in scenes)) or None
            continuity = story_notes(scenes)
            plan = _plan_timeline(specs, end_card)
            plan["asset_issues"], plan["expected_seconds"] = blocking, expected
            plan["logo_path"] = logo_path
            plan["story_notes"] = continuity
            report = ad_qc.inspect(final_path, plan, facts, w, h, tmp)
            corrections: list[str] = []
            if not report["passed"]:
                corrections = _auto_correct(report, specs, end_card, render_end, assets, local, campaign,
                                            campaign_id, aspect_ratio, voice_name, w, h, tmp)
                if corrections:
                    final_path = assemble("1")
                    plan = _plan_timeline(specs, end_card)
                    plan["asset_issues"], plan["expected_seconds"] = blocking, expected
                    plan["logo_path"] = logo_path
                    plan["story_notes"] = continuity
                    report = ad_qc.inspect(final_path, plan, facts, w, h, tmp)
            report["corrections"] = corrections
            report["brand_corrections"] = brand_changes
            report["claim_corrections"] = claim_changes
            report["notes"] = list(facts.issues)
            _update_job(job_id, progress_pct=92)

            duration_actual = int(round(_ffprobe_duration(final_path)))
            thumb_path = os.path.join(tmp, "thumb.jpg")
            _extract_thumbnail(final_path, thumb_path)
            with open(thumb_path, "rb") as f:
                thumbnail_url = _store_file(campaign, f.read(), "image/jpeg", f"thumb_{uuid.uuid4().hex}.jpg")
            if report["passed"]:
                # Only an ad that passed every check goes to Bunny as a finished video.
                video_id = _upload_finished_video_to_bunny(final_path, campaign.get("target_name") or "Viyo Ad")
                playback, preview = _bunny_playback_url(video_id), None
            else:
                # A failed ad is kept privately as a preview to look at, never published.
                with open(final_path, "rb") as f:
                    preview = _store_file(campaign, f.read(), "video/mp4", f"preview_{uuid.uuid4().hex}.mp4")
                video_id, playback = None, None
            _update_job(job_id, progress_pct=97)

        if report["passed"]:
            _save_campaign(campaign_id, {
                "video_url": playback, "bunny_video_id": video_id, "thumbnail_url": thumbnail_url,
                "duration_actual_seconds": duration_actual, "status": "ready", "error": None,
                "quality_report": report, "preview_video_url": None})
            _update_job(job_id, status="succeeded", progress_pct=100)
        else:
            reason = ad_qc.explain(report)
            _save_campaign(campaign_id, {
                "video_url": None, "bunny_video_id": None, "thumbnail_url": thumbnail_url,
                "duration_actual_seconds": duration_actual, "status": "needs_correction", "error": reason,
                "quality_report": report, "preview_video_url": preview})
            _update_job(job_id, status="needs_correction", progress_pct=100, error=reason[:1000])
    except HTTPException as e:
        supabase_admin.table("ad_campaigns").update({"status": "failed", "error": e.detail}).eq("id", campaign_id).execute()
        _update_job(job_id, status="failed", error=str(e.detail))
    except Exception as e:
        supabase_admin.table("ad_campaigns").update({"status": "failed", "error": str(e)}).eq("id", campaign_id).execute()
        _update_job(job_id, status="failed", error=str(e))


def _auto_correct(report: dict, specs: list[dict], end_card: dict, render_end, assets: list[dict], local: dict,
                  campaign: dict, campaign_id: str, aspect_ratio: str, voice_name: str, w: int, h: int,
                  tmp: str) -> list[str]:
    """One round of fixes for what the check found, scene by scene. Returns what was done."""
    done, regenerations = [], 0
    photos = [a["id"] for a in assets if a.get("asset_type") in ("screenshot", "product_photo") and a["id"] in local]
    used = {sp.get("asset_id") for sp in specs}
    spare = [a for a in photos if a not in used] + [a for a in photos if a in used]
    by_index = {sp["index"]: sp for sp in specs}
    for key, problems in report.get("failing_scenes", {}).items():
        if key == "end":
            end_card["plain"] = True
            render_end(1)
            done.append("End card re-rendered on a plain background so the logo, website and call to action read clearly.")
            continue
        sp = by_index.get(int(key))
        if not sp:
            continue
        n = sp["index"] + 1
        if any(p in problems for p in ("black", "screen", "website")) and sp["kind"] == "hero":
            # Keep the real product; only the generated setting behind it is redone.
            if regenerations >= MAX_IMAGE_REGENERATIONS:
                continue
            regenerations += 1
            url = _generate_ad_scene_image(_image_prompt(sp["scene"], campaign, aspect_ratio, stricter=True,
                                                         for_product=True), [], campaign_id)
            sp["image"] = os.path.join(tmp, f"gen_{sp['index']}_fix.png")
            _fetch_local(url, sp["image"])
            _render_spec(sp, w, h, tmp)
            done.append(f"Scene {n}: generated a new setting behind the product.")
        elif any(p in problems for p in ("black", "screen", "website")) and sp["kind"] in ("generated", "video"):
            if spare:
                aid = spare.pop(0)
                sp.update(kind="asset", image=local[aid], asset_id=aid, native_audio=False)
                if sp.get("audio") is None:
                    sp["audio"], sp["duration"] = _voice_track(sp.get("dialogue", ""), sp["duration"], voice_name,
                                                               campaign_id, tmp, f"a{sp['index']}_fix")
                done.append(f"Scene {n}: replaced the generated picture with a real product image.")
            elif sp["kind"] == "video":
                sp["kind"] = "generated"
                if sp.get("audio") is None:
                    sp["audio"], sp["duration"] = _voice_track(sp.get("dialogue", ""), sp["duration"], voice_name,
                                                               campaign_id, tmp, f"a{sp['index']}_fix")
                done.append(f"Scene {n}: used the scene's still image instead of the animated clip.")
            elif regenerations < MAX_IMAGE_REGENERATIONS:
                regenerations += 1
                url = _generate_ad_scene_image(_image_prompt(sp["scene"], campaign, aspect_ratio, stricter=True), [], campaign_id)
                sp["image"] = os.path.join(tmp, f"gen_{sp['index']}_fix.png")
                _fetch_local(url, sp["image"])
                done.append(f"Scene {n}: generated a new picture (the first had a blank screen, black frames or stray text).")
            else:
                continue
            _render_spec(sp, w, h, tmp)
        elif "voice" in problems and sp.get("dialogue") and not sp.get("native_audio"):
            sp["audio"], sp["duration"] = _voice_track(sp["dialogue"], sp["duration"], voice_name, campaign_id, tmp,
                                                       f"a{sp['index']}_retry")
            _render_spec(sp, w, h, tmp)
            done.append(f"Scene {n}: redid the voiceover.")
        elif any(p in problems for p in ("clipping", "safe", "overlap")):
            _render_spec(sp, w, h, tmp, scale=0.85)
            done.append(f"Scene {n}: re-rendered with smaller captions.")
    return done


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
    job = _start_generation(campaign)
    background_tasks.add_task(_run_ad_generation, campaign_id, job["id"])
    return JobOut(id=job["id"], campaign_id=campaign_id, stage="video", status="pending", progress_pct=0)


def _start_generation(campaign: dict) -> dict:
    """Free checks, then the job row. Shared by the admin and creator routes."""
    aspect_ratio = campaign.get("aspect_ratio") or "9:16"
    resolution = campaign.get("resolution") or "720p"
    if aspect_ratio not in AD_ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail=f"aspect_ratio must be one of {AD_ASPECT_RATIOS}.")
    if resolution not in AD_RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"resolution must be one of {AD_RESOLUTIONS}.")
    if campaign.get("use_veo") and aspect_ratio == "1:1":
        raise HTTPException(status_code=400, detail="Veo doesn't support 1:1 — turn off Veo animation or change aspect ratio.")
    _preflight(campaign)
    try:
        supabase_admin.table("ad_campaigns").update({"status": "generating", "error": None}).eq("id", campaign["id"]).execute()
        return (
            supabase_admin.table("ad_generation_jobs")
            .insert({"campaign_id": campaign["id"], "stage": "video", "status": "pending", "progress_pct": 0})
            .execute()
            .data[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not start generation: {e}")


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
