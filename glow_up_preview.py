"""
Viyo Glow Up — Stage 2: AI hairstyle previews.

  * The selfie is sent with each request (the app keeps it in memory) and
    is never stored. Only a generated preview is stored, and only when the
    user taps Save (private bucket, short-lived signed links, deletable).
  * Every preview is an AI simulation and is stamped "AI preview" in the
    corner, so a saved or shared image can't pass as a real photo.
  * The edit is a fixed, narrow instruction: change the hair only, keep the
    same person, face, skin, age, body, clothes, background and light. No
    beautifying, slimming or makeup. The style text is length- and
    character-limited, so it can't carry other instructions.
  * Checked before charging: 18+, the photo (one clear face), moderation,
    the user's daily limit and the platform's daily limit.
  * Checked after generating: the preview still has exactly one face, in
    about the same place and size (same person, same framing), and passes
    moderation. If not, coins are refunded and nothing is kept.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import os
import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import glow_up as g

router = APIRouter(prefix="/api/v1/glow-up", tags=["glow_up"])

FEATURE = "glow_up_preview"
BUCKET = "glow-up-previews"
IMAGE_MODEL = os.environ.get("GLOW_UP_IMAGE_MODEL", "gemini-3.1-flash-image")
USER_DAILY_LIMIT = int(os.environ.get("GLOW_UP_PREVIEWS_PER_USER_PER_DAY", "10"))
PLATFORM_DAILY_LIMIT = int(os.environ.get("GLOW_UP_MAX_PREVIEWS_PER_DAY", "150"))
SIGNED_URL_SECONDS = 3600
MAX_STYLE_CHARS = 60
KINDS = {"hair": "hairstyle", "hair_color": "hair colour"}
# Framing must survive the edit: same face, roughly same place and size.
MAX_CENTER_SHIFT = 0.12
SIZE_RATIO_RANGE = (0.7, 1.45)

_STYLE_OK = re.compile(r"^[\w\s\-'’,&()/.]+$", re.UNICODE)

_genai_client = None


def _gemini():
    global _genai_client
    if _genai_client is None:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            return None
        from google import genai
        _genai_client = genai.Client(api_key=key)
    return _genai_client


def clean_style(style: str) -> str:
    s = re.sub(r"\s+", " ", (style or "").strip())
    if not s or len(s) > MAX_STYLE_CHARS or not _STYLE_OK.match(s):
        raise g.PhotoProblem("Pick a style from the list, or describe one in a few words.")
    if re.search(r"\b(ignore|instruction|prompt|system|nude|naked|body|face|skin|younger|older|slim|thin)\b", s, re.I):
        raise g.PhotoProblem("Describe just the hairstyle, for example \"shoulder-length layers\".")
    return s


def edit_prompt(kind: str, style: str) -> str:
    what = KINDS[kind]
    change = (f"Change ONLY the hairstyle to: {style}." if kind == "hair"
              else f"Change ONLY the hair colour to: {style}. Keep the same haircut and length.")
    return (
        f"Edit this photo of a real person to preview a new {what}. {change} "
        "Keep everything else exactly the same: the same person and identity, the same face, facial "
        "features, face shape, skin tone and texture, expression, age, body, clothing, background, "
        "lighting and camera angle. Photorealistic, natural result, as if it were a real photo. "
        "Do not beautify, retouch, slim, add makeup, change the eyes or smooth the skin. "
        "No text, no watermark, no borders.")


def _today_count(admin, user_id: Optional[str]) -> int:
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)).isoformat()
    q = admin.table("transactions").select("id", count="exact").eq("type", FEATURE).gte("created_at", since)
    if user_id:
        q = q.eq("user_id", user_id)
    try:
        return int(q.limit(1).execute().count or 0)
    except Exception:
        return 0


def generate(jpeg: bytes, prompt: str) -> bytes:
    """One Gemini image edit. Raises RuntimeError on any failure."""
    client = _gemini()
    if client is None:
        raise RuntimeError("image generation is not configured")
    from google.genai import types
    resp = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else None) or []:
            if part.inline_data is not None and part.inline_data.data:
                return part.inline_data.data
    raise RuntimeError("no image came back (the request may have been declined)")


def same_framing(before: dict, after: dict) -> bool:
    shift = ((before["cx"] - after["cx"]) ** 2 + (before["cy"] - after["cy"]) ** 2) ** 0.5
    ratio = after["face_width"] / before["face_width"] if before["face_width"] else 0
    return shift <= MAX_CENTER_SHIFT and SIZE_RATIO_RANGE[0] <= ratio <= SIZE_RATIO_RANGE[1]


def stamp(jpeg: bytes) -> bytes:
    """Burns a small "AI preview" label into the corner."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = img.size
    size = max(14, w // 32)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    text = "AI preview"
    draw = ImageDraw.Draw(img, "RGBA")
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    pad = size // 2
    x, y = pad, h - th - pad * 3
    draw.rounded_rectangle((x, y, x + tw + pad * 2, y + th + pad * 2), radius=pad, fill=(11, 11, 26, 190))
    draw.text((x + pad, y + pad - 2), text, font=font, fill=(0, 229, 255, 255))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


class PreviewRequest(BaseModel):
    image_base64: str = Field(..., max_length=12_000_000)
    age_confirmed: bool = False
    kind: str = "hair"
    style: str = Field(..., max_length=200)
    save: bool = False


class PreviewResponse(BaseModel):
    image_base64: str
    saved: bool
    preview_id: Optional[str] = None
    image_url: Optional[str] = None
    coins_charged: int
    note: str = "AI simulation — real results depend on your hair and your stylist."


@router.post("/preview", response_model=PreviewResponse)
async def preview(req: PreviewRequest, user_id: str = Depends(g._user_no_guest)):
    if not req.age_confirmed:
        raise HTTPException(status_code=403, detail="Glow Up is for people aged 18 and over.")
    if req.kind not in KINDS:
        raise HTTPException(status_code=400, detail="Unknown preview type.")
    g._rate_limit(user_id)
    try:
        style = clean_style(req.style)
        jpeg = g.normalise_image(g._decode_b64(req.image_base64))
        before = g.check_single_face(jpeg)
    except g.PhotoProblem as e:
        raise HTTPException(status_code=400, detail=e.message)

    admin = g._get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Glow Up is not configured.")
    if _today_count(admin, user_id) >= USER_DAILY_LIMIT:
        raise HTTPException(status_code=429, detail=f"You've made {USER_DAILY_LIMIT} previews today. Try again tomorrow.")
    if _today_count(admin, None) >= PLATFORM_DAILY_LIMIT:
        raise HTTPException(status_code=503, detail="Previews are busy today. Please try again tomorrow. Nothing was charged.")
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    try:
        if not g.moderate(data_url):
            raise HTTPException(status_code=400, detail="This photo can't be used. Please use a regular selfie.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Previews are temporarily unavailable. Nothing was charged. ({e})")

    import coins
    charged = coins.spend_on_feature(admin, user_id, FEATURE)
    try:
        out = g.normalise_image(generate(jpeg, edit_prompt(req.kind, style)))
        try:
            after = g.check_single_face(out)
        except g.PhotoProblem:
            raise RuntimeError("the preview didn't keep a clear face")
        if not same_framing(before, after):
            raise RuntimeError("the preview changed the photo too much")
        if not g.moderate("data:image/jpeg;base64," + base64.b64encode(out).decode("ascii")):
            raise RuntimeError("the preview didn't pass the safety check")
        out = stamp(out)
    except Exception as e:
        coins.refund_feature(admin, user_id, FEATURE, charged)
        raise HTTPException(status_code=502,
                            detail=f"We couldn't make a realistic preview this time. Your coins were refunded. ({e})")
    finally:
        del jpeg, data_url            # the selfie is never kept

    saved, preview_id, url = False, None, None
    if req.save:
        try:
            preview_id = str(uuid.uuid4())
            path = f"{user_id}/{preview_id}.jpg"
            admin.storage.from_(BUCKET).upload(path, out, file_options={"content-type": "image/jpeg"})
            admin.table("glow_up_previews").insert({
                "id": preview_id, "user_id": user_id, "kind": req.kind, "style": style, "image_path": path,
            }).execute()
            url = _signed(admin, path)
            saved = True
        except Exception as e:
            print(f"[WARN] glow up preview: could not save: {e}")
            preview_id = None
    return PreviewResponse(image_base64=base64.b64encode(out).decode("ascii"), saved=saved,
                           preview_id=preview_id, image_url=url, coins_charged=charged)


def _signed(admin, path: str) -> Optional[str]:
    try:
        r = admin.storage.from_(BUCKET).create_signed_url(path, SIGNED_URL_SECONDS)
        return r.get("signedURL") or r.get("signedUrl")
    except Exception:
        return None


@router.get("/previews")
async def list_previews(user_id: str = Depends(g._user)):
    admin = g._get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Glow Up is not configured.")
    try:
        rows = (admin.table("glow_up_previews").select("id, created_at, kind, style, image_path")
                .eq("user_id", user_id).order("created_at", desc=True).limit(60).execute()).data or []
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not load your previews: {e}")
    return {"previews": [{**{k: r[k] for k in ("id", "created_at", "kind", "style")},
                          "image_url": _signed(admin, r["image_path"])} for r in rows]}


def delete_previews(admin, user_id: str, preview_id: Optional[str] = None) -> int:
    q = admin.table("glow_up_previews").select("id, image_path").eq("user_id", user_id)
    if preview_id:
        q = q.eq("id", preview_id)
    rows = q.execute().data or []
    if rows:
        admin.storage.from_(BUCKET).remove([r["image_path"] for r in rows])
        d = admin.table("glow_up_previews").delete().eq("user_id", user_id)
        if preview_id:
            d = d.eq("id", preview_id)
        d.execute()
    return len(rows)


@router.delete("/previews/{preview_id}")
async def delete_preview(preview_id: str, user_id: str = Depends(g._user)):
    admin = g._get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Glow Up is not configured.")
    try:
        n = delete_previews(admin, user_id, preview_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not delete: {e}")
    return {"deleted": n}


@router.delete("/previews")
async def delete_all_previews(user_id: str = Depends(g._user)):
    admin = g._get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Glow Up is not configured.")
    try:
        n = delete_previews(admin, user_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not delete: {e}")
    return {"deleted": n}
