"""Publishing a finished ad to VIYO.

A finished ad (status "ready": it passed the quality check and is already in Bunny Stream) becomes one
ordinary `posts` row, the same kind every VIYO video is, marked post_kind = "ad". Nothing is uploaded again
and no second copy is made: the post plays the ad's existing Bunny video. So likes, comments, views,
reports, the creator's profile and creator downloads all work on it exactly as on any other post.

Rules this file keeps:
  * Only the campaign's owner can publish, change or take down its post (anyone else: 404).
  * Only a ready ad can be published: never a draft, a failed ad or one that needs correction.
  * Publishing twice updates the same post (caption, link, visibility...) rather than making a duplicate.
  * Taking it down archives the post (hidden everywhere, kept for the owner); nothing is ever deleted, and
    the creator keeps their video and can still download it. Publishing on VIYO is never exclusive.
  * Private files (a creator's own pictures) become public only when the creator picks one as the post's
    thumbnail, and only that one picture is copied.
  * The link must be a real web address; it is stored exactly as given.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import ad_brand
import ads_studio as core

AD_CATEGORIES = {
    "food_drink": "Food & drink",
    "fashion_beauty": "Fashion & beauty",
    "apps_tech": "Apps & tech",
    "home_living": "Home & living",
    "services": "Services",
    "entertainment": "Film, music & events",
    "education": "Education",
    "health_fitness": "Health & fitness",
    "travel_places": "Travel & places",
    "business": "Business",
    "other": "Other",
}
_KIND_CATEGORY = {"viyo": "apps_tech", "ollie_ai": "apps_tech", "other_app": "apps_tech", "website": "apps_tech",
                  "digital_product": "apps_tech", "service": "services", "business": "business",
                  "personal_brand": "other", "film": "entertainment", "music": "entertainment",
                  "event": "entertainment", "creator_project": "entertainment"}
_SUBTYPE_CATEGORY = {"drink": "food_drink", "food": "food_drink", "restaurant": "food_drink",
                     "clothing": "fashion_beauty", "shoes": "fashion_beauty", "cosmetics": "fashion_beauty",
                     "accessories": "fashion_beauty", "beauty salon": "fashion_beauty", "furniture": "home_living",
                     "cleaning": "home_living", "electronics": "apps_tech", "hotel": "travel_places",
                     "transport": "travel_places", "education": "education", "online course": "education",
                     "health": "health_fitness"}
MAX_CAPTION = 2200
MAX_INFO = 600


class PublishIn(BaseModel):
    caption: str = Field("", max_length=MAX_CAPTION)
    category: str = "other"
    link: str = ""                        # website or contact link shown on the post ("Visit")
    product_info: str = Field("", max_length=MAX_INFO)
    thumbnail: str = "video"              # "video" | "cover" | "asset:<asset id>"
    public: bool = True


def suggested_category(c: dict) -> str:
    return _SUBTYPE_CATEGORY.get(c.get("product_subtype") or "") or _KIND_CATEGORY.get(c.get("promote_target") or "") \
        or ("food_drink" if c.get("promote_target") == "physical_product" else "other")


def _product_info(c: dict) -> str:
    """Only the advertiser's own facts: prices first, then features and selling points."""
    facts = c.get("facts") or {}
    items = list(facts.get("prices") or []) + list(facts.get("features") or c.get("target_features") or []) \
        + list(facts.get("selling_points") or [])
    return " · ".join(str(i).strip() for i in items if str(i).strip())[:MAX_INFO]


def _post(post_id: Optional[str]) -> Optional[dict]:
    if not post_id:
        return None
    rows = core.supabase_admin.table("posts").select("*").eq("id", post_id).limit(1).execute().data or []
    return rows[0] if rows else None


def _thumbnail_choices(c: dict, assets: list[dict]) -> list[dict]:
    out = []
    if c.get("bunny_video_id"):
        out.append({"key": "video", "label": "From the video", "url": core._bunny_thumbnail_url(c["bunny_video_id"])})
    if c.get("thumbnail_url"):
        out.append({"key": "cover", "label": "Ad cover", "url": core._resolve(c["thumbnail_url"])})
    for a in assets:
        if a.get("asset_type") in ("product_photo", "logo", "screenshot", "other") and a.get("url"):
            out.append({"key": f"asset:{a['id']}", "label": a.get("label") or a["asset_type"].replace("_", " "),
                        "url": core._resolve(a["url"])})
    return out


def publish_status(c: dict) -> Optional[dict]:
    p = _post(c.get("published_post_id"))
    if not p or p.get("user_id") != c.get("user_id"):
        return None
    return {"post_id": p["id"], "public": not p.get("is_private") and not p.get("is_archived"),
            "archived": bool(p.get("is_archived")), "caption": p.get("caption") or "",
            "category": p.get("ad_category") or "other", "link": p.get("ad_link") or "",
            "product_info": p.get("ad_info") or "", "thumbnail_url": p.get("thumbnail_url")}


def options(c: dict, assets: list[dict]) -> dict:
    """What the publish screen shows: categories, suggestions from the campaign, thumbnail choices, and the
    current post if the ad was published before."""
    name = (c.get("brand_name") or c.get("target_name") or "").strip()
    desc = (c.get("target_description") or "").strip()
    return {
        "can_publish": c.get("status") == "ready" and bool(c.get("bunny_video_id")),
        "categories": [{"key": k, "label": v} for k, v in AD_CATEGORIES.items()],
        "suggested": {"caption": (f"{name} — {desc}" if name and desc else name or desc)[:MAX_CAPTION],
                      "category": suggested_category(c), "link": (c.get("destination_link") or "").strip(),
                      "product_info": _product_info(c)},
        "thumbnails": _thumbnail_choices(c, assets),
        "published": publish_status(c),
    }


def _clean_link(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith(("tel:", "mailto:", "https://wa.me/")):
        return raw[:300]
    url = ad_brand.parse_url(raw)[0]
    if not url:
        raise HTTPException(status_code=400, detail=f'"{raw}" isn\'t a valid web address.')
    return url if "://" in url else "https://" + url


def _public_thumbnail(c: dict, choice: str, assets: list[dict]) -> Optional[str]:
    """A public picture for the post. A private file is copied to public storage only because the creator
    chose it as the thumbnail of a post they're publishing."""
    if choice == "video" or not choice:
        return core._bunny_thumbnail_url(c["bunny_video_id"])
    if choice == "cover":
        ref = c.get("thumbnail_url")
    elif choice.startswith("asset:"):
        aid = choice.split(":", 1)[1]
        ref = next((a.get("url") for a in assets if a.get("id") == aid), None)
    else:
        ref = None
    if not ref:
        raise HTTPException(status_code=400, detail="That thumbnail isn't available. Pick another one.")
    if not ref.startswith(core.PRIVATE_PREFIX):
        return ref
    bucket, _, path = ref[len(core.PRIVATE_PREFIX):].partition("/")
    try:
        data = core.supabase_admin.storage.from_(bucket).download(path)
    except Exception:
        raise HTTPException(status_code=502, detail="The thumbnail couldn't be read. Try again or pick another one.")
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else "jpg"
    mime = {"png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    return core._upload_image(data, mime, f"ad-posts/{c['user_id']}/{uuid.uuid4().hex}.{ext}")


def publish(c: dict, body: PublishIn, assets: list[dict]) -> dict:
    if c.get("status") != "ready" or not c.get("bunny_video_id") or not c.get("video_url"):
        raise HTTPException(status_code=409, detail="Only a finished ad that passed the quality check can be "
                                                    "published.")
    if body.category not in AD_CATEGORIES:
        raise HTTPException(status_code=400, detail="Pick a category.")
    link = _clean_link(body.link)
    thumb = _public_thumbnail(c, body.thumbnail, assets)
    w, h = core._ad_target_dimensions(c.get("aspect_ratio") or "9:16", c.get("resolution") or "720p")
    fields = {
        "caption": body.caption.strip() or (c.get("brand_name") or c.get("target_name") or "")[:MAX_CAPTION],
        "thumbnail_url": thumb, "is_private": not body.public, "is_archived": False,
        "ad_category": body.category, "ad_link": link or None, "ad_info": body.product_info.strip() or None,
    }
    existing = _post(c.get("published_post_id"))
    if existing and existing.get("user_id") == c["user_id"]:
        core.supabase_admin.table("posts").update(fields).eq("id", existing["id"]).execute()
        post_id = existing["id"]
    else:
        row = {"user_id": c["user_id"], "post_type": "video", "media_url": c["video_url"],
               "duration_seconds": int(c.get("duration_actual_seconds") or c.get("duration_seconds") or 0) or None,
               "width": w, "height": h, "video_provider": "bunny", "bunny_video_id": c["bunny_video_id"],
               "video_status": "ready", "post_kind": "ad", "ad_campaign_id": c["id"], **fields}
        post_id = core.supabase_admin.table("posts").insert(row).execute().data[0]["id"]
        core.supabase_admin.table("ad_campaigns").update({"published_post_id": post_id}).eq("id", c["id"]).execute()
    c = {**c, "published_post_id": post_id}
    return publish_status(c)


def unpublish(c: dict) -> dict:
    """Takes the post off VIYO (archived: hidden from everyone else, kept, never deleted)."""
    p = _post(c.get("published_post_id"))
    if not p or p.get("user_id") != c.get("user_id"):
        raise HTTPException(status_code=404, detail="This ad isn't published.")
    core.supabase_admin.table("posts").update({"is_archived": True}).eq("id", p["id"]).execute()
    return publish_status(c)


# ---------------------------------------------------------------- admin (Ads Studio), same rules

admin_router = APIRouter(prefix="/api/v1/admin/ads-studio", tags=["ads_studio"])


def _admin_campaign(campaign_id: str) -> dict:
    rows = core.supabase_admin.table("ad_campaigns").select("*").eq("id", campaign_id).limit(1).execute().data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found.")
    return rows[0]


@admin_router.get("/campaign/{campaign_id}/publish", dependencies=[Depends(core._require_admin)])
async def admin_publish_options(campaign_id: str):
    return options(_admin_campaign(campaign_id), core._load_campaign_assets(campaign_id))


@admin_router.post("/campaign/{campaign_id}/publish", dependencies=[Depends(core._require_admin)])
async def admin_publish(campaign_id: str, body: PublishIn):
    return publish(_admin_campaign(campaign_id), body, core._load_campaign_assets(campaign_id))


@admin_router.post("/campaign/{campaign_id}/unpublish", dependencies=[Depends(core._require_admin)])
async def admin_unpublish(campaign_id: str):
    return unpublish(_admin_campaign(campaign_id))
