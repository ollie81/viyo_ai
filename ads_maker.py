"""Ads Maker for creators: the same production pipeline as the admin Ads Studio (ads_studio.py), for any
signed-in creator, under /api/v1/ads-maker.

Security
  * Every route needs a signed-in, non-guest account (creator_studio.creator: the same check Creator
    Studio uses, including its invite-only switch).
  * A creator only ever sees and changes their own campaigns: every campaign, asset, hook and job route
    loads the campaign and checks its user_id; anyone else's answers 404 (not 403), so ids can't be probed.
  * Creators' files go in the private bucket under their own folder and come back as one-hour signed links;
    finished ads that pass the quality check go to Bunny like any published video.
  * The admin routes (X-Admin-Key) are unchanged and not reachable from here: no VIYO asset library, no
    "Promote VIYO" defaults, no performance records, and creators can't advertise as VIYO.

Payment
  * Paid steps (hooks, script, video) are priced with Creator Studio's pricing (studio_pricing: real provider
    cost plus the target margin), shown first (/estimate), then held from the creator's Studio credits
    before anything runs. The run counts every provider call and can't spend more than it held
    (ads_studio._cap). Only the real cost is kept; the rest goes back. A video that fails or needs
    correction keeps nothing (Viyo absorbs it).
"""
from __future__ import annotations

import io
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import ad_brand
import ads_studio as core
import studio_pricing as pricing
from creator_studio import creator

router = APIRouter(prefix="/api/v1/ads-maker", tags=["ads_maker"])

CREATOR_KINDS = ("other_app", "website", "physical_product", "business", "film", "creator_project")
MAX_RUNS_PER_HOUR = 6
_runs: dict[str, deque] = defaultdict(deque)


def _db():
    if core.supabase_admin is None:
        raise HTTPException(status_code=503, detail="Ads Maker is not configured.")
    return core.supabase_admin


def _uuid(value: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found.")


def owned_campaign(campaign_id: str, user_id: str) -> dict:
    """The campaign, if it belongs to [user_id]. Anything else (missing, someone else's, malformed id) is a
    plain 404."""
    rows = _db().table("ad_campaigns").select("*").eq("id", _uuid(campaign_id)).limit(1).execute().data or []
    if not rows or rows[0].get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    return rows[0]


def _out(row: dict) -> dict:
    """A campaign for its owner, with private files as signed links."""
    out = core._row_to_campaign(row).model_dump()
    out["thumbnail_url"] = core._resolve(out.get("thumbnail_url"))
    out["preview_video_url"] = core._resolve(out.get("preview_video_url"))
    return out


# ---------------------------------------------------------------- credits

def _balance(user_id: str) -> dict:
    try:
        data = _db().rpc("studio_credit_balance", {"p_user_id": user_id}).execute().data
    except Exception:
        return {"available": 0}
    data = data[0] if isinstance(data, list) else data
    return data or {"available": 0}


def _hold(user_id: str, credits: int) -> str:
    try:
        r = _db().rpc("ads_credit_hold", {"p_user_id": user_id, "p_credits": credits}).execute().data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Credits aren't available right now: {e}")
    r = (r[0] if isinstance(r, list) else r) or {}
    if not r.get("ok"):
        available = int(((r.get("balance") or {}).get("available")) or 0)
        raise HTTPException(status_code=402, detail={"error": "insufficient_credits", "needed": credits,
                                                     "available": available,
                                                     "message": f"This step needs {credits} Studio credits; you have {available}."})
    return r["hold_id"]


def _settle(hold_id: str, used: int) -> None:
    try:
        _db().rpc("ads_credit_settle", {"p_hold_id": hold_id, "p_used": max(0, int(used))}).execute()
    except Exception as e:
        print(f"[WARN] ads credit settle failed for {hold_id}: {e}")


def _budget_cents(credits: int) -> float:
    """What a hold of [credits] may really spend: their value minus the margin (a run never spends more)."""
    return credits * pricing.USD_PER_CREDIT * (1 - pricing.target_margin()) * 100


def _used_credits(spent_cents: float, held: int) -> int:
    return 0 if spent_cents <= 0 else min(held, pricing.credits_for(spent_cents / 100))


@contextmanager
def paid(user_id: str, credits: int):
    """Holds [credits], runs the block with every provider call counted against them, keeps only the real
    cost (nothing if the block fails)."""
    hold = _hold(user_id, credits)
    run = {"spent_cents": 0.0, "budget_cents": _budget_cents(credits)}
    token = core._creator_run.set(run)
    ok = False
    try:
        yield run
        ok = True
    finally:
        core._creator_run.reset(token)
        _settle(hold, _used_credits(run["spent_cents"], credits) if ok else 0)


# ---------------------------------------------------------------- estimates

def estimate(campaign: dict, assets: Optional[list] = None) -> dict:
    """Credits each paid step holds. Built from the same per-call prices the pipeline counts (studio.py's
    cents per image/voice line/text call and the video tier's price per second), so a run always has the
    budget it needs; credits_for() adds Creator Studio's margin. Video: a picture for every scene that may
    not use a real asset, the two corrections a run may need, a voice line per scene (and one redo each),
    animation at its longest clip length when it's on, and the render."""
    assets = assets if assets is not None else core._load_campaign_assets(campaign["id"])
    seconds = int(campaign.get("duration_seconds") or 15)
    scenes = core._scene_count_for_duration(seconds)
    prompt = core._build_hook_prompt(campaign, campaign.get("format"))
    hooks_cents = max(2, 2 * core._estimate_text_cost_cents(prompt))
    script_cents = max(3, 2 * core._estimate_text_cost_cents(prompt + " " * 3000)) + len(assets)
    video_cents = ((scenes - 1 + core.MAX_IMAGE_REGENERATIONS) * core.GEMINI_IMAGE_COST_USD_CENTS
                   + scenes * 2 * core.GEMINI_TTS_LINE_COST_USD_CENTS + 2)
    if campaign.get("use_veo"):
        tier = core.VEO_TIERS.get(campaign.get("veo_tier") or core.DEFAULT_VEO_TIER, core.VEO_TIERS[core.DEFAULT_VEO_TIER])
        video_cents += (scenes - 1) * max(core.VEO_ALLOWED_DURATIONS) * tier["price_per_sec_cents"]
    to_credits = lambda cents: pricing.credits_for(cents / 100)   # noqa: E731
    return {"hooks": to_credits(hooks_cents), "script": to_credits(script_cents), "video": to_credits(video_cents)}


# ---------------------------------------------------------------- campaigns

class CampaignIn(BaseModel):
    promote_target: str = Field("other_app", pattern="^(" + "|".join(CREATOR_KINDS) + ")$")
    target_name: str = ""
    brand_name: str = ""
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
    voice_gender_preference: Optional[str] = None


class CampaignPatch(BaseModel):
    promote_target: Optional[str] = Field(None, pattern="^(" + "|".join(CREATOR_KINDS) + ")$")
    target_name: Optional[str] = None
    brand_name: Optional[str] = None
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
    voice_gender_preference: Optional[str] = None


@router.get("/options")
async def options(user_id: str = Depends(creator)):
    return {"kinds": {k: core.PRODUCT_KINDS[k] for k in CREATOR_KINDS},
            "formats": [{"key": k, "label": v["label"]} for k, v in core.AD_FORMATS.items()],
            "durations": core.AD_DURATIONS, "aspect_ratios": core.AD_ASPECT_RATIOS, "resolutions": core.AD_RESOLUTIONS,
            "asset_types": core.AD_ASSET_TYPES, "credits": _balance(user_id)}


@router.get("/formats")
async def formats(user_id: str = Depends(creator)):
    return await core.list_formats()


@router.get("/veo-tiers")
async def veo_tiers(user_id: str = Depends(creator)):
    return await core.list_veo_tiers()


@router.get("/campaigns")
async def list_campaigns(user_id: str = Depends(creator)):
    rows = (_db().table("ad_campaigns").select("*").eq("user_id", user_id).order("created_at", desc=True)
            .execute().data) or []
    return [_out(r) for r in rows]


@router.post("/campaigns")
async def create_campaign(body: CampaignIn, user_id: str = Depends(creator)):
    req = core.CreateCampaignRequest(user_id=user_id, veo_tier=core.DEFAULT_VEO_TIER, **body.model_dump())
    created = await core.create_campaign(req)                 # same validation as the admin route
    _db().table("ad_campaigns").update({"created_via": "creator"}).eq("id", created.id).execute()
    return _out(owned_campaign(created.id, user_id))


@router.get("/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    return {**_out(c), "estimate": estimate(c)}


@router.put("/campaigns/{campaign_id}")
async def update_campaign(campaign_id: str, body: CampaignPatch, user_id: str = Depends(creator)):
    owned_campaign(campaign_id, user_id)
    await core.update_campaign(campaign_id, core.UpdateCampaignRequest(**body.model_dump()))
    return _out(owned_campaign(campaign_id, user_id))


@router.delete("/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    try:   # the creator's private files for this campaign
        folder = f"ads/{user_id}/{c['id']}"
        names = [f["name"] for f in (_db().storage.from_(core.PRIVATE_BUCKET).list(folder) or []) if f.get("name")]
        if names:
            _db().storage.from_(core.PRIVATE_BUCKET).remove([f"{folder}/{n}" for n in names])
    except Exception:
        pass
    return await core.delete_campaign(c["id"])


# ---------------------------------------------------------------- assets

def _asset_out(a: dict) -> dict:
    return {"id": a["id"], "url": core._resolve(a["url"]), "asset_type": a["asset_type"],
            "source": a.get("source") or "uploaded", "label": a.get("label") or ""}


@router.post("/campaigns/{campaign_id}/assets")
async def upload_asset(campaign_id: str, asset_type: str = "other", label: str = "", file: UploadFile = File(...),
                       user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    if asset_type not in core.AD_ASSET_TYPES:
        asset_type = "other"
    existing = core._load_campaign_assets(c["id"])
    if len(existing) >= core.MAX_ASSETS_PER_CAMPAIGN:
        raise HTTPException(status_code=400, detail=f"Up to {core.MAX_ASSETS_PER_CAMPAIGN} assets per campaign.")
    data = await file.read()
    if not data or len(data) > core.MAX_AD_ASSET_BYTES:
        raise HTTPException(status_code=400, detail=f"Images up to {core.MAX_AD_ASSET_BYTES // (1024 * 1024)} MB.")
    try:    # a real, readable image of a usable size, whatever the file claims to be
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            fmt, size = (im.format or "").lower(), im.size
    except Exception:
        raise HTTPException(status_code=400, detail="That file isn't an image Viyo can read (PNG, JPEG or WebP).")
    if fmt not in ("png", "jpeg", "webp") or min(size) < 64:
        raise HTTPException(status_code=400, detail="Use a PNG, JPEG or WebP image at least 64 pixels on each side.")
    mime = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}[fmt]
    ref = core._store_file(c, data, mime, f"asset_{uuid.uuid4().hex}.{'jpg' if fmt == 'jpeg' else fmt}")
    row = _db().table("ad_assets").insert({"campaign_id": c["id"], "url": ref, "asset_type": asset_type,
                                           "source": "uploaded", "label": label[:80]}).execute().data[0]
    return _asset_out(row)


@router.get("/campaigns/{campaign_id}/assets")
async def list_assets(campaign_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    return [_asset_out(a) for a in core._load_campaign_assets(c["id"])]


@router.delete("/campaigns/{campaign_id}/assets/{asset_id}")
async def delete_asset(campaign_id: str, asset_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    rows = (_db().table("ad_assets").select("*").eq("id", _uuid(asset_id)).eq("campaign_id", c["id"])
            .limit(1).execute().data) or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found.")
    ref = rows[0]["url"]
    if ref.startswith(core.PRIVATE_PREFIX):
        bucket, _, path = ref[len(core.PRIVATE_PREFIX):].partition("/")
        try:
            _db().storage.from_(bucket).remove([path])
        except Exception:
            pass
    _db().table("ad_assets").delete().eq("id", rows[0]["id"]).execute()
    return {"deleted": True}


# ---------------------------------------------------------------- paid steps

@router.post("/campaigns/{campaign_id}/hooks")
async def generate_hooks(campaign_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    with paid(user_id, estimate(c)["hooks"]):
        return await core.generate_hooks(c["id"])


@router.get("/campaigns/{campaign_id}/hooks")
async def list_hooks(campaign_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    return await core.list_hooks(c["id"])


@router.post("/campaigns/{campaign_id}/hooks/{hook_id}/select")
async def select_hook(campaign_id: str, hook_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    if hook_id != "auto":
        hook_id = _uuid(hook_id)
    await core.select_hook(c["id"], hook_id)            # checks the hook belongs to this campaign
    return _out(owned_campaign(c["id"], user_id))


@router.post("/campaigns/{campaign_id}/script")
async def generate_script(campaign_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    with paid(user_id, estimate(c)["script"]):
        return await core.generate_script(c["id"])


@router.put("/campaigns/{campaign_id}/script")
async def save_script(campaign_id: str, body: core.SaveScriptRequest, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    await core.save_script(c["id"], body)
    return _out(owned_campaign(c["id"], user_id))


def _rate_limit(user_id: str) -> None:
    q, now = _runs[user_id], time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= MAX_RUNS_PER_HOUR:
        raise HTTPException(status_code=429, detail="That's a lot of videos in one hour. Try again a little later.")
    q.append(now)


def _run_paid(campaign_id: str, job_id: str, user_id: str, hold_id: str, credits: int) -> None:
    """The background run, with its spending counted against the hold; only a finished (passing) ad keeps
    credits, and only its real cost."""
    run = {"spent_cents": 0.0, "budget_cents": _budget_cents(credits)}
    token = core._creator_run.set(run)
    try:
        core._run_ad_generation(campaign_id, job_id)
    finally:
        core._creator_run.reset(token)
        try:
            status = (_db().table("ad_campaigns").select("status").eq("id", campaign_id).limit(1).execute().data
                      or [{}])[0].get("status")
        except Exception:
            status = None
        _settle(hold_id, _used_credits(run["spent_cents"], credits) if status == "ready" else 0)


@router.post("/campaigns/{campaign_id}/generate")
async def generate_video(campaign_id: str, background_tasks: BackgroundTasks, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    core._preflight(c)                       # free checks first: nothing is held for an invalid campaign
    _rate_limit(user_id)
    credits = estimate(c)["video"]
    hold = _hold(user_id, credits)
    try:
        job = core._start_generation(c)
    except Exception:
        _settle(hold, 0)
        raise
    background_tasks.add_task(_run_paid, c["id"], job["id"], user_id, hold, credits)
    return {"id": job["id"], "campaign_id": c["id"], "stage": "video", "status": "pending", "progress_pct": 0,
            "credits_held": credits}


@router.get("/campaigns/{campaign_id}/job")
async def latest_job(campaign_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    return await core.get_latest_job(c["id"])


@router.get("/campaigns/{campaign_id}/estimate")
async def get_estimate(campaign_id: str, user_id: str = Depends(creator)):
    c = owned_campaign(campaign_id, user_id)
    return {**estimate(c), "available": int(_balance(user_id).get("available") or 0)}
