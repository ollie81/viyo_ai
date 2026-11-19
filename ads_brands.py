"""My Brands: a creator saves a brand once (name, description, logo, colours, product photos, verified
facts, website, social accounts, contact details, its advertising character) and every new ad starts from it.

Rules this file keeps:
  * A brand, its pictures and its characters belong to one creator. Every route checks the owner; anyone
    else gets 404, so ids can't be probed.
  * Brand pictures are private files in the creator's own folder (one-hour signed links), like campaign files.
  * A new ad made from a brand gets its own copy of the brand's pictures, so deleting that ad (which removes
    its folder) never touches the brand, and editing the brand later never changes an ad already made.
  * The brand's character is reused as-is (same portrait, same voice): no new paid portrait.
  * Facts are the creator's own word and are validated exactly like a campaign's (ads_studio._clean_profile).
  * Archiving a brand hides it from the list; nothing is deleted, and its ads and characters stay.
"""
from __future__ import annotations

import io
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import ad_brand
import ads_studio as core
from creator_studio import creator

router = APIRouter(prefix="/api/v1/ads-maker/brands", tags=["ads_maker"])

MAX_BRANDS = 30
MAX_BRAND_ASSETS = 12
BRAND_ASSET_TYPES = ("logo", "product_photo", "screenshot", "other")
_PROFILE_FIELDS = ("product_subtype", "facts", "brand_colors", "social_links", "contact")


def _db():
    if core.supabase_admin is None:
        raise HTTPException(status_code=503, detail="Ads Maker is not configured.")
    return core.supabase_admin


def _uuid(value: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found.")


def owned_brand(brand_id: str, user_id: str) -> dict:
    rows = _db().table("ad_brands").select("*").eq("id", _uuid(brand_id)).limit(1).execute().data or []
    if not rows or rows[0].get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    return rows[0]


class BrandIn(BaseModel):
    name: str = Field("", max_length=120)
    description: str = Field("", max_length=2000)
    promote_target: str = "physical_product"
    product_subtype: str = ""
    website: str = ""
    cta_text: str = Field("", max_length=80)
    target_audience: str = Field("", max_length=300)
    facts: Optional[dict] = None
    brand_colors: Optional[list[str]] = None
    social_links: Optional[list[str]] = None
    contact: Optional[dict] = None


class BrandPatch(BaseModel):
    name: Optional[str] = Field(None, max_length=120)
    description: Optional[str] = Field(None, max_length=2000)
    promote_target: Optional[str] = None
    product_subtype: Optional[str] = None
    website: Optional[str] = None
    cta_text: Optional[str] = Field(None, max_length=80)
    target_audience: Optional[str] = Field(None, max_length=300)
    facts: Optional[dict] = None
    brand_colors: Optional[list[str]] = None
    social_links: Optional[list[str]] = None
    contact: Optional[dict] = None
    archived: Optional[bool] = None


def _clean(fields: dict) -> dict:
    """Same checks as a campaign: a real web address, a known kind, valid colours/contact/facts."""
    out = {k: v for k, v in fields.items() if v is not None}
    if "promote_target" in out and out["promote_target"] not in core.PRODUCT_KINDS:
        raise HTTPException(status_code=400, detail="Pick what kind of thing this brand advertises.")
    if out.get("promote_target") in ("viyo", "ollie_ai"):
        raise HTTPException(status_code=400, detail="Pick what kind of thing this brand advertises.")
    if out.get("website"):
        if not ad_brand.parse_url(out["website"])[0]:
            raise HTTPException(status_code=400, detail=f'"{out["website"]}" isn\'t a valid web address.')
        out["website"] = out["website"].strip()
    profile = {k: out.pop(k) for k in list(out) if k in _PROFILE_FIELDS}
    if profile:
        if "product_subtype" in profile and out.get("promote_target"):
            profile["promote_target"] = out["promote_target"]
        cleaned = core._clean_profile(profile)
        cleaned.pop("promote_target", None)
        out.update(cleaned)
    if "name" in out:
        out["name"] = out["name"].strip()
    return out


def _asset_out(a: dict) -> dict:
    return {"id": a["id"], "url": core._resolve(a["url"]), "asset_type": a["asset_type"], "label": a.get("label") or ""}


def _brand_out(b: dict, assets: Optional[list] = None, characters: Optional[list] = None,
               campaigns: Optional[list] = None) -> dict:
    out = {k: b.get(k) for k in ("id", "name", "description", "promote_target", "product_subtype", "website",
                                 "cta_text", "target_audience", "created_at")}
    out.update({"facts": b.get("facts") or {}, "brand_colors": b.get("brand_colors") or [],
                "social_links": b.get("social_links") or [], "contact": b.get("contact") or {},
                "archived": bool(b.get("archived")), "character_id": b.get("character_id"),
                "logo_url": core._resolve(b.get("logo"))})
    if assets is not None:
        out["assets"] = [_asset_out(a) for a in assets]
    if characters is not None:
        out["characters"] = [{"id": r["id"], "name": r.get("name") or "", "spec": r.get("spec") or {},
                              "portrait_url": core._resolve(r.get("portrait")), "voice_name": r.get("voice_name")}
                             for r in characters]
    if campaigns is not None:
        out["campaigns"] = [{"id": c["id"], "status": c.get("status") or "draft", "ad_style": c.get("ad_style") or "auto",
                             "track": c.get("track") or "",
                             "aspect_ratio": c.get("aspect_ratio"), "thumbnail_url": core._resolve(c.get("thumbnail_url")),
                             "published_post_id": c.get("published_post_id"), "created_at": c.get("created_at")}
                            for c in campaigns]
    return out


def _brand_assets(brand_id: str) -> list[dict]:
    return _db().table("ad_brand_assets").select("*").eq("brand_id", brand_id).order("created_at").execute().data or []


def _read_private(ref: str) -> Optional[bytes]:
    if not ref or not ref.startswith(core.PRIVATE_PREFIX):
        return None
    bucket, _, path = ref[len(core.PRIVATE_PREFIX):].partition("/")
    try:
        return _db().storage.from_(bucket).download(path)
    except Exception:
        return None


def _store_brand_file(user_id: str, brand_id: str, data: bytes, mime: str, ext: str) -> str:
    path = f"ads/{user_id}/brands/{brand_id}/{uuid.uuid4().hex}.{ext}"
    _db().storage.from_(core.PRIVATE_BUCKET).upload(path, data, {"content-type": mime, "upsert": "true"})
    return f"{core.PRIVATE_PREFIX}{core.PRIVATE_BUCKET}/{path}"


# ---------------------------------------------------------------- brands

@router.get("")
async def list_brands(include_archived: bool = False, user_id: str = Depends(creator)):
    rows = (_db().table("ad_brands").select("*").eq("owner_id", user_id).order("created_at", desc=True)
            .execute().data) or []
    return [_brand_out(b) for b in rows if include_archived or not b.get("archived")]


@router.post("")
async def create_brand(body: BrandIn, user_id: str = Depends(creator)):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Give the brand a name.")
    mine = _db().table("ad_brands").select("*").eq("owner_id", user_id).execute().data or []
    if len([b for b in mine if not b.get("archived")]) >= MAX_BRANDS:
        raise HTTPException(status_code=400, detail=f"Up to {MAX_BRANDS} brands. Archive one first.")
    row = _db().table("ad_brands").insert({"owner_id": user_id, **_clean(body.model_dump())}).execute().data[0]
    return _brand_out(row, assets=[], characters=[], campaigns=[])


@router.get("/{brand_id}")
async def get_brand(brand_id: str, user_id: str = Depends(creator)):
    b = owned_brand(brand_id, user_id)
    chars = [r for r in (_db().table("ad_characters").select("*").eq("brand_id", b["id"]).execute().data or [])
             if r.get("owner_id") == user_id]
    camps = [c for c in (_db().table("ad_campaigns").select("*").eq("brand_id", b["id"]).order("created_at", desc=True)
                         .execute().data or []) if c.get("user_id") == user_id]
    return _brand_out(b, assets=_brand_assets(b["id"]), characters=chars, campaigns=camps)


@router.put("/{brand_id}")
async def update_brand(brand_id: str, body: BrandPatch, user_id: str = Depends(creator)):
    b = owned_brand(brand_id, user_id)
    patch = _clean(body.model_dump())
    if patch.get("name") == "":
        raise HTTPException(status_code=400, detail="Give the brand a name.")
    if patch:
        _db().table("ad_brands").update(patch).eq("id", b["id"]).execute()
    return await get_brand(b["id"], user_id)


@router.post("/{brand_id}/assets")
async def upload_brand_asset(brand_id: str, asset_type: str = "product_photo", label: str = "",
                             file: UploadFile = File(...), user_id: str = Depends(creator)):
    """A logo or product photo for the brand (private). The first logo becomes the brand's logo."""
    b = owned_brand(brand_id, user_id)
    if asset_type not in BRAND_ASSET_TYPES:
        asset_type = "other"
    if len(_brand_assets(b["id"])) >= MAX_BRAND_ASSETS:
        raise HTTPException(status_code=400, detail=f"Up to {MAX_BRAND_ASSETS} pictures per brand.")
    data = await file.read()
    if not data or len(data) > core.MAX_AD_ASSET_BYTES:
        raise HTTPException(status_code=400, detail=f"Images up to {core.MAX_AD_ASSET_BYTES // (1024 * 1024)} MB.")
    try:
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
    ref = _store_brand_file(user_id, b["id"], data, mime, "jpg" if fmt == "jpeg" else fmt)
    row = _db().table("ad_brand_assets").insert({"brand_id": b["id"], "owner_id": user_id, "url": ref,
                                                 "asset_type": asset_type, "label": label[:80]}).execute().data[0]
    if asset_type == "logo" and not b.get("logo"):
        _db().table("ad_brands").update({"logo": ref}).eq("id", b["id"]).execute()
    return _asset_out(row)


@router.delete("/{brand_id}/assets/{asset_id}")
async def delete_brand_asset(brand_id: str, asset_id: str, user_id: str = Depends(creator)):
    """Removes a picture from the brand. Ads already made from it keep their own copy."""
    b = owned_brand(brand_id, user_id)
    rows = (_db().table("ad_brand_assets").select("*").eq("id", _uuid(asset_id)).eq("brand_id", b["id"])
            .limit(1).execute().data) or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found.")
    ref = rows[0]["url"]
    _db().table("ad_brand_assets").delete().eq("id", rows[0]["id"]).execute()
    if ref.startswith(core.PRIVATE_PREFIX):
        bucket, _, path = ref[len(core.PRIVATE_PREFIX):].partition("/")
        try:
            _db().storage.from_(bucket).remove([path])
        except Exception:
            pass
    if b.get("logo") == ref:
        _db().table("ad_brands").update({"logo": None}).eq("id", b["id"]).execute()
    return {"deleted": True}


@router.put("/{brand_id}/character/{character_id}")
async def set_brand_character(brand_id: str, character_id: str, user_id: str = Depends(creator)):
    """The brand's advertising character: used in every new ad from this brand (same face, same voice)."""
    b = owned_brand(brand_id, user_id)
    rows = _db().table("ad_characters").select("*").eq("id", _uuid(character_id)).limit(1).execute().data or []
    if not rows or rows[0].get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    _db().table("ad_characters").update({"brand_id": b["id"]}).eq("id", rows[0]["id"]).execute()
    _db().table("ad_brands").update({"character_id": rows[0]["id"]}).eq("id", b["id"]).execute()
    return await get_brand(b["id"], user_id)


@router.delete("/{brand_id}/character")
async def clear_brand_character(brand_id: str, user_id: str = Depends(creator)):
    """New ads from this brand go back to letting the AI decide. The character itself is kept."""
    b = owned_brand(brand_id, user_id)
    _db().table("ad_brands").update({"character_id": None}).eq("id", b["id"]).execute()
    return await get_brand(b["id"], user_id)


# ---------------------------------------------------------------- brands <-> ads

@router.post("/from-campaign/{campaign_id}")
async def save_campaign_as_brand(campaign_id: str, user_id: str = Depends(creator)):
    """Saves an ad's product details, pictures and character as a new brand, so the next ad starts there."""
    from ads_maker import owned_campaign
    c = owned_campaign(campaign_id, user_id)
    if c.get("brand_id"):
        try:
            return await get_brand(c["brand_id"], user_id)
        except HTTPException:
            pass
    name = (c.get("brand_name") or c.get("target_name") or "").strip() or "My brand"
    kind = c.get("promote_target") if c.get("promote_target") not in ("viyo", "ollie_ai") else "other_app"
    brand = await create_brand(BrandIn(
        name=name[:120], description=(c.get("target_description") or "")[:2000], promote_target=kind,
        product_subtype=c.get("product_subtype") or "", website=c.get("destination_link") or "",
        cta_text=(c.get("cta_text") or "")[:80], target_audience=(c.get("target_audience") or "")[:300],
        facts=c.get("facts") or ({"features": c["target_features"]} if c.get("target_features") else None),
        brand_colors=c.get("brand_colors") or None, social_links=c.get("social_links") or None,
        contact=c.get("contact") or None), user_id)
    bid = brand["id"]
    for a in core._load_campaign_assets(c["id"])[:MAX_BRAND_ASSETS]:
        data = _read_private(a.get("url") or "")
        if data is None:
            continue
        ext = (a["url"].rsplit(".", 1)[-1] or "png").lower()
        ref = _store_brand_file(user_id, bid, data, {"jpg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png"), ext)
        _db().table("ad_brand_assets").insert({"brand_id": bid, "owner_id": user_id, "url": ref,
                                               "asset_type": a.get("asset_type") if a.get("asset_type") in BRAND_ASSET_TYPES else "other",
                                               "label": (a.get("label") or "")[:80]}).execute()
        if a.get("asset_type") == "logo":
            _db().table("ad_brands").update({"logo": ref}).eq("id", bid).execute()
    saved = (c.get("character") or {}).get("saved_id")
    if saved:
        rows = _db().table("ad_characters").select("*").eq("id", saved).limit(1).execute().data or []
        if rows and rows[0].get("owner_id") == user_id:
            _db().table("ad_characters").update({"brand_id": bid}).eq("id", saved).execute()
            _db().table("ad_brands").update({"character_id": saved}).eq("id", bid).execute()
    _db().table("ad_campaigns").update({"brand_id": bid}).eq("id", c["id"]).execute()
    return await get_brand(bid, user_id)


class NewAdIn(BaseModel):
    ad_style: str = "auto"
    duration_seconds: int = 15
    aspect_ratio: str = "9:16"
    objective: str = Field("", max_length=300)
    use_character: bool = True        # false: a product-only ad even if the brand has a character
    track: str = ""                   # Campaign Universe track (ads_ideas.TRACKS); empty: not set


@router.post("/{brand_id}/campaigns")
async def new_ad_from_brand(brand_id: str, body: NewAdIn, user_id: str = Depends(creator)):
    """A new ad campaign filled in from the brand: details, facts, colours, links, its own copy of the brand's
    pictures, and the brand's character (reused, no new portrait). The creator can still change anything."""
    import ads_maker
    b = owned_brand(brand_id, user_id)
    if b.get("archived"):
        raise HTTPException(status_code=400, detail="This brand is archived. Restore it first.")
    character, mode = None, "auto"
    if not body.use_character:
        mode = "none"
    elif b.get("character_id"):
        rows = _db().table("ad_characters").select("*").eq("id", b["character_id"]).limit(1).execute().data or []
        if rows and rows[0].get("owner_id") == user_id:
            r = rows[0]
            character = {**(r.get("spec") or {}), "name": r.get("name") or "", "portrait": r.get("portrait"),
                         "voice_name": r.get("voice_name"), "saved_id": r["id"]}
            mode = "reuse"
    created = await ads_maker.create_campaign(ads_maker.CampaignIn(
        promote_target=b.get("promote_target") or "physical_product", target_name=b.get("name") or "",
        brand_name=b.get("name") or "", product_subtype=b.get("product_subtype") or "",
        facts=b.get("facts") or None, brand_colors=b.get("brand_colors") or None,
        social_links=b.get("social_links") or None, contact=b.get("contact") or None,
        character_mode=mode, ad_style=body.ad_style if body.ad_style in core.AD_STYLES else "auto",
        target_description=b.get("description") or "", target_features=(b.get("facts") or {}).get("features") or [],
        target_audience=b.get("target_audience") or "", objective=body.objective,
        destination_link=b.get("website") or "", cta_text=b.get("cta_text") or "",
        duration_seconds=body.duration_seconds, aspect_ratio=body.aspect_ratio), user_id)
    cid = created["id"]
    patch = {"brand_id": b["id"]}
    from ads_ideas import TRACKS
    if body.track in TRACKS:
        patch["track"] = body.track
    if character:
        patch["character"] = character
    _db().table("ad_campaigns").update(patch).eq("id", cid).execute()
    campaign = ads_maker.owned_campaign(cid, user_id)
    for a in _brand_assets(b["id"])[:core.MAX_ASSETS_PER_CAMPAIGN]:
        data = _read_private(a["url"])
        if data is None:
            continue
        ext = (a["url"].rsplit(".", 1)[-1] or "png").lower()
        mime = {"jpg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")
        ref = core._store_file(campaign, data, mime, f"asset_{uuid.uuid4().hex}.{ext}")
        _db().table("ad_assets").insert({"campaign_id": cid, "url": ref, "asset_type": a["asset_type"],
                                         "source": "brand", "label": a.get("label") or ""}).execute()
    return ads_maker._out(ads_maker.owned_campaign(cid, user_id))
