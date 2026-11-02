"""
Viyo Glow Up — Stage 1: style suggestions from a selfie.

Privacy by design:
  * The selfie is NOT stored unless the user explicitly opts in
    (save_photo=true, off by default). Without consent it is dropped when the
    request ends. With consent it goes to the PRIVATE glow-up-photos bucket at
    <user_id>/<session_id>.jpg, re-encoded without metadata, readable only
    through 1-hour signed links issued to its owner, and deleted with its
    session, on its own ("remove photo"), or with the account.
  * The text suggestions are saved (save=true, the default) with the chosen
    look, style direction and date, so they can be reopened later.
  * 18+ only: every request must carry age_confirmed=true.
  * Saved suggestions can be deleted one by one or all at once, and are
    removed with the account (coach.py's account deletion).

What it checks before charging anything:
  1. the image decodes, isn't huge, and is re-encoded to a clean JPEG
     (EXIF / GPS stripped, max 1024 px);
  2. exactly one clear face is in it (local face detector, no API cost);
  3. OpenAI's moderation check passes.
Then it charges 10 coins, asks the vision model for structured suggestions,
and screens the answer: no attractiveness ratings, no comments on weight,
ethnicity, age, health or personality. A bad answer is retried once; if it
still fails, or anything else fails, the coins are refunded.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Callable, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, ValidationError

router = APIRouter(prefix="/api/v1/glow-up", tags=["glow_up"])

FEATURE = "glow_up_analysis"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_SIDE = 1024
MIN_FACE_WIDTH = 0.08          # share of image width; smaller and the advice would be guesswork
SECOND_FACE_RATIO = 0.5        # another face at least this big = more than one person
GLOW_UP_MODEL = os.environ.get("GLOW_UP_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
# "high" lets the model see hair texture, colour and length properly; "low"
# downsizes to 512 px and produced generic advice in testing (missed grey hair).
IMAGE_DETAIL = os.environ.get("GLOW_UP_IMAGE_DETAIL", "high")
PHOTOS_BUCKET = "glow-up-photos"          # private; only with explicit consent
SIGNED_URL_SECONDS = 3600
STYLE_GOALS = {"everyday", "professional", "bold", "minimal", "event"}
PRESENTATIONS = {"feminine", "masculine", "neutral", "any"}

_admin = None
_client = None


def _get_admin():
    global _admin
    if _admin is None:
        url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if url and key:
            from supabase import create_client
            _admin = create_client(url, key)
    return _admin


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=90, max_retries=1)
    return _client


async def _user_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest
    return await get_current_user_id_no_guest(authorization)


async def _user(authorization: str = Header(None)) -> str:
    from main import get_current_user_id
    return await get_current_user_id(authorization)


def _rate_limit(user_id: str) -> None:
    try:
        from main import _check_rate_limit
    except Exception:
        return
    _check_rate_limit(user_id)


# ------------------------------------------------------------------ image checks

class PhotoProblem(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def normalise_image(raw: bytes) -> bytes:
    """Clean JPEG, upright, max MAX_SIDE px, no metadata (EXIF/GPS dropped)."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    if len(raw) > MAX_UPLOAD_BYTES:
        raise PhotoProblem("That photo is too large. Please use one under 8 MB.")
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise PhotoProblem("We couldn't open that photo. Please use a JPG or PNG.")
    if min(img.size) < 200:
        raise PhotoProblem("That photo is too small. Please use a clearer, closer selfie.")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88)       # a fresh file: no EXIF/GPS carried over
    return out.getvalue()


def check_single_face(jpeg: bytes) -> dict:
    """Exactly one clear, close-enough face. Uses the local detector (free)."""
    import cv2
    import numpy as np
    import video_analysis as va
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise PhotoProblem("We couldn't read that photo.")
    faces, detector = va.detect_faces(frame)
    if not faces:
        raise PhotoProblem("We couldn't find a face. Use a front-facing selfie in good light, with your face uncovered.")
    faces.sort(key=lambda f: f.w * f.h, reverse=True)
    main = faces[0]
    if len(faces) > 1 and faces[1].w * faces[1].h >= SECOND_FACE_RATIO * main.w * main.h:
        raise PhotoProblem("Please use a photo with only you in it.")
    if main.w < MIN_FACE_WIDTH:
        raise PhotoProblem("Your face is too small in that photo. Move closer to the camera.")
    return {"faces": len(faces), "face_width": round(main.w, 3), "face_height": round(main.h, 3),
            "cx": round(main.cx, 3), "cy": round(main.cy, 3), "detector": detector}


def moderate(data_url: str) -> bool:
    """True when the image is acceptable. Raises on service failure."""
    res = _get_client().moderations.create(
        model="omni-moderation-latest",
        input=[{"type": "image_url", "image_url": {"url": data_url}}])
    return not any(r.flagged for r in res.results)


# ------------------------------------------------------------------ suggestions

class HairIdea(BaseModel):
    name: str = Field(..., max_length=80)
    why: str = Field(..., max_length=300)
    how_to_ask: str = Field("", max_length=300)


class StyleIdea(BaseModel):
    item: str = Field(..., max_length=100)
    why: str = Field(..., max_length=300)


class Observed(BaseModel):
    """What the model says it can actually see. Shown to the user, so advice
    is visibly grounded in their photo rather than generic."""
    hair_length: str = Field(..., max_length=60)
    hair_texture: str = Field(..., max_length=60)
    hair_color: str = Field(..., max_length=60)
    face_shape: str = Field(..., max_length=60)
    visible_style: str = Field(..., max_length=160)


class GlowUpResult(BaseModel):
    # Optional only so sessions saved before this field existed still load.
    observed: Optional[Observed] = None
    summary: str = Field(..., max_length=500)
    hair: list[HairIdea] = Field(..., min_length=2, max_length=4)
    grooming: list[str] = Field(..., min_length=2, max_length=6)
    style: list[StyleIdea] = Field(..., min_length=2, max_length=6)
    colors: list[str] = Field(..., min_length=2, max_length=6)
    next_step: str = Field(..., max_length=300)
    photo_note: Optional[str] = Field(None, max_length=200)


SYSTEM = (
    "You are a warm, practical personal stylist and barber/hairdresser. You give specific, constructive, "
    "respectful suggestions based only on what is visible in the photo: current hair (length, texture, how "
    "it falls), face shape, visible clothing and colours. You never judge the person.")

RULES = """Hard rules — breaking any of them makes the answer unusable:
- Never rate, score, rank or compare how attractive anyone is. Never say "attractive", "ugly", "hot", "sexy".
- Never comment on body weight or size, ethnicity or race, age, health, skin conditions, or personality.
- Don't frame anything as a flaw to fix. Frame every suggestion as an option to try.
- Don't assume gender. Follow the style direction below if one is given; otherwise keep suggestions open.
- Be specific (named cuts, lengths, products by type, cuts of clothing), not generic.
- Describe only what you can see. If something isn't visible (e.g. hair hidden by a hat), say "not visible"
  and don't guess.
- Each hair, style and colour idea must fit what you observed; mention the observed detail in "why".
- If there isn't one clear, front-facing face in the photo, return only {"status": "no_clear_face"}."""

FORMAT = """First look carefully and describe only what is visible. Then base every suggestion on it.
Return ONLY JSON:
{"observed": {"hair_length": "e.g. short / chin-length / long", "hair_texture": "straight / wavy / curly / coily",
              "hair_color": "as visible", "face_shape": "oval / round / square / heart / long / diamond",
              "visible_style": "what they are wearing in the photo, briefly"},
 "summary": "1-2 encouraging sentences on a style direction that suits them",
 "hair": [{"name": "named cut or style", "why": "why it suits what you see", "how_to_ask": "what to tell the barber/stylist"}],  (3 ideas)
 "grooming": ["practical tip"],  (3-5 tips: hair care, styling products, brows/beard if relevant, skincare basics)
 "style": [{"item": "clothing cut or piece", "why": "why it works"}],  (3-5 ideas)
 "colors": ["colour name"],  (3-5 colours to try near the face)
 "next_step": "one easy thing to try this week",
 "photo_note": "only if the photo limited the advice (dim light, hair hidden), else null"}"""

_BANNED = re.compile(
    r"\b(attractive(ness)?|unattractive|ugly|sexy|hotter|hottest|beauty score|out of (10|ten)|\d+\s*/\s*10|"
    r"rating|overweight|underweight|lose weight|weight loss|fat|obese|ethnic(ity)?|racial|race|acne|"
    r"wrinkles?|flaws?|imperfections?|insecure)\b", re.IGNORECASE)


def _prompt(style_goal: Optional[str], presentation: Optional[str]) -> str:
    direction = []
    if style_goal:
        direction.append(f"The look they want: {style_goal}.")
    if presentation and presentation != "any":
        direction.append(f"Style direction they chose: {presentation}.")
    return "\n\n".join([RULES, " ".join(direction) or "No style direction given: keep suggestions open.", FORMAT])


def respectful(result: GlowUpResult) -> bool:
    return not _BANNED.search(result.model_dump_json())


def ask_model(data_url: str, style_goal: Optional[str], presentation: Optional[str],
              reminder: str = "") -> str:
    resp = _get_client().chat.completions.create(
        model=GLOW_UP_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": [
                      {"type": "text", "text": _prompt(style_goal, presentation) + reminder},
                      {"type": "image_url", "image_url": {"url": data_url, "detail": IMAGE_DETAIL}}]}],
        response_format={"type": "json_object"},
        temperature=0.5,
        max_completion_tokens=1200,
    )
    return resp.choices[0].message.content or ""


def get_suggestions(data_url: str, style_goal: Optional[str], presentation: Optional[str],
                    llm: Optional[Callable] = None) -> GlowUpResult:
    """Two attempts at a valid, respectful answer. Raises PhotoProblem when
    the model sees no clear face, RuntimeError otherwise."""
    llm = llm or ask_model
    reminder = ""
    for _ in range(2):
        raw = llm(data_url, style_goal, presentation, reminder)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            reminder = "\n\nReturn valid JSON only."
            continue
        if isinstance(data, dict) and data.get("status") == "no_clear_face":
            raise PhotoProblem("We couldn't see your face clearly. Try a front-facing selfie in good light.")
        try:
            result = GlowUpResult(**data)
        except (ValidationError, TypeError):
            reminder = "\n\nYour last answer didn't match the JSON format exactly. Follow it precisely."
            continue
        if result.observed is None:
            reminder = "\n\nInclude the \"observed\" section describing what is visible."
            continue
        if respectful(result):
            return result
        reminder = ("\n\nYour last answer broke the hard rules (it judged looks, weight, age, ethnicity, "
                    "health or similar). Rewrite it following every rule.")
    raise RuntimeError("no valid, respectful answer after two attempts")


# ------------------------------------------------------------------ endpoints

class AnalyzeRequest(BaseModel):
    image_base64: str = Field(..., max_length=12_000_000)
    age_confirmed: bool = False
    style_goal: Optional[str] = None
    presentation: Optional[str] = None
    save: bool = True
    # Off unless the user explicitly turns it on. Implies save=True.
    save_photo: bool = False


class AnalyzeResponse(BaseModel):
    result: GlowUpResult
    saved: bool
    session_id: Optional[str] = None
    coins_charged: int
    photo_saved: bool = False
    photo_url: Optional[str] = None       # short-lived private link, only when photo_saved


def _decode_b64(s: str) -> bytes:
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail="We couldn't read that photo.")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, user_id: str = Depends(_user_no_guest)):
    if not req.age_confirmed:
        raise HTTPException(status_code=403, detail="Glow Up is for people aged 18 and over.")
    style_goal = req.style_goal if req.style_goal in STYLE_GOALS else None
    presentation = req.presentation if req.presentation in PRESENTATIONS else None
    _rate_limit(user_id)

    # Free checks first: nothing is charged for a photo we can't use.
    try:
        jpeg = normalise_image(_decode_b64(req.image_base64))
        check_single_face(jpeg)
    except PhotoProblem as e:
        raise HTTPException(status_code=400, detail=e.message)
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    try:
        if not moderate(data_url):
            raise HTTPException(status_code=400, detail="This photo can't be used. Please use a regular selfie.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail="Glow Up is temporarily unavailable. Nothing was charged.")

    import coins
    admin = _get_admin()
    charged = coins.spend_on_feature(admin, user_id, FEATURE)
    try:
        result = get_suggestions(data_url, style_goal, presentation)
    except PhotoProblem as e:
        coins.refund_feature(admin, user_id, FEATURE, charged)
        raise HTTPException(status_code=400, detail=e.message + " Your coins were refunded.")
    except Exception as e:
        coins.refund_feature(admin, user_id, FEATURE, charged)
        print(f"[WARN] glow up analysis failed for {user_id}: {str(e)[:500]}")
        raise HTTPException(status_code=502,
                            detail="We couldn't create your suggestions right now. Your coins were refunded.")
    finally:
        # The photo is kept past this point only with explicit consent.
        keep_photo = jpeg if req.save_photo else None
        del jpeg, data_url

    saved, session_id = False, None
    if (req.save or req.save_photo) and admin is not None:
        try:
            row = admin.table("glow_up_sessions").insert({
                "user_id": user_id, "recommendations": result.model_dump(),
                "style_goal": style_goal, "presentation": presentation,
            }).execute()
            session_id = (row.data or [{}])[0].get("id")
            saved = session_id is not None
        except Exception as e:
            print(f"[WARN] glow up: could not save suggestions: {str(e)[:300]}")
    photo_saved, photo_url = False, None
    if keep_photo is not None and saved:
        photo_saved, photo_url = _store_photo(admin, user_id, session_id, keep_photo)
    keep_photo = None
    return AnalyzeResponse(result=result, saved=saved, session_id=session_id, coins_charged=charged,
                           photo_saved=photo_saved, photo_url=photo_url)


def _signed(admin, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        r = admin.storage.from_(PHOTOS_BUCKET).create_signed_url(path, SIGNED_URL_SECONDS)
        return r.get("signedURL") or r.get("signedUrl")
    except Exception:
        return None


def _store_photo(admin, user_id: str, session_id: str, jpeg: bytes) -> tuple[bool, Optional[str]]:
    """Private bucket, path tied to the owner and session. If attaching it to
    the session fails, the file is removed again so nothing is orphaned."""
    path = f"{user_id}/{session_id}.jpg"
    try:
        # cache-control 0: the CDN must never keep a copy, so a deleted photo
        # stops loading at once, even through a link issued earlier.
        admin.storage.from_(PHOTOS_BUCKET).upload(
            path, jpeg, file_options={"content-type": "image/jpeg", "cache-control": "0"})
    except Exception as e:
        print(f"[WARN] glow up: could not store photo: {str(e)[:300]}")
        return False, None
    try:
        admin.table("glow_up_sessions").update({"photo_path": path}).eq("id", session_id).eq("user_id", user_id).execute()
    except Exception as e:
        print(f"[WARN] glow up: could not attach photo to session: {str(e)[:300]}")
        try:
            admin.storage.from_(PHOTOS_BUCKET).remove([path])
        except Exception:
            pass
        return False, None
    return True, _signed(admin, path)


def _remove_photos(admin, paths: list) -> None:
    """Deletes the files and confirms storage actually removed each one
    (storage answers with the objects it deleted). Raises if any remain, so
    nobody is ever told a photo was deleted when it wasn't."""
    paths = [p for p in paths if p]
    if not paths:
        return
    removed = admin.storage.from_(PHOTOS_BUCKET).remove(paths) or []
    gone = {r.get("name") for r in removed if isinstance(r, dict)}
    missing = [p for p in paths if p not in gone]
    if missing:
        raise RuntimeError(f"storage did not delete {len(missing)} photo(s)")


def _session_out(admin, row: dict) -> dict:
    path = row.get("photo_path")
    return {"id": row["id"], "created_at": row.get("created_at"), "style_goal": row.get("style_goal"),
            "presentation": row.get("presentation"), "recommendations": row.get("recommendations"),
            "has_photo": bool(path), "photo_url": _signed(admin, path)}


_SESSION_COLS = "id, created_at, style_goal, presentation, recommendations, photo_path"
_SESSION_COLS_OLD = "id, created_at, style_goal, presentation, recommendations"


def _select_sessions(admin, user_id: str, session_id: Optional[str] = None) -> list:
    """Owner's rows only. Works before and after the photo_path column exists."""
    for cols in (_SESSION_COLS, _SESSION_COLS_OLD):
        try:
            q = admin.table("glow_up_sessions").select(cols).eq("user_id", user_id)
            if session_id:
                q = q.eq("id", session_id)
            return q.order("created_at", desc=True).limit(50).execute().data or []
        except Exception as e:
            last = e
            if "photo_path" not in str(e):
                break
    raise last


def _admin_or_503():
    admin = _get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Glow Up is not configured.")
    return admin


@router.get("/sessions")
async def list_sessions(user_id: str = Depends(_user)):
    admin = _admin_or_503()
    try:
        rows = _select_sessions(admin, user_id)
    except Exception as e:
        print(f"[WARN] glow up: list failed: {str(e)[:300]}")
        raise HTTPException(status_code=502, detail="Could not load your Glow Ups. Please try again.")
    return {"sessions": [_session_out(admin, r) for r in rows]}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user_id: str = Depends(_user)):
    admin = _admin_or_503()
    try:
        rows = _select_sessions(admin, user_id, session_id)
    except Exception as e:
        print(f"[WARN] glow up: get failed: {str(e)[:300]}")
        raise HTTPException(status_code=502, detail="Could not load this Glow Up. Please try again.")
    if not rows:
        raise HTTPException(status_code=404, detail="Glow Up not found.")
    return _session_out(admin, rows[0])


@router.delete("/sessions/{session_id}/photo")
async def delete_session_photo(session_id: str, user_id: str = Depends(_user)):
    """Removes only the saved photo; the ideas stay."""
    admin = _admin_or_503()
    rows = _select_sessions(admin, user_id, session_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Glow Up not found.")
    path = rows[0].get("photo_path")
    try:
        _remove_photos(admin, [path])
        if path:
            admin.table("glow_up_sessions").update({"photo_path": None}).eq("id", session_id).eq("user_id", user_id).execute()
    except Exception as e:
        print(f"[WARN] glow up: photo delete failed: {str(e)[:300]}")
        raise HTTPException(status_code=502, detail="Could not delete the photo. Please try again.")
    return {"photo_deleted": bool(path)}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(_user)):
    """Deletes the session and its saved photo (if any)."""
    admin = _admin_or_503()
    try:
        rows = _select_sessions(admin, user_id, session_id)
        if not rows:
            raise HTTPException(status_code=404, detail="Glow Up not found.")
        path = rows[0].get("photo_path")
        _remove_photos(admin, [path])          # photo first: never leave a file without its session
        admin.table("glow_up_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
    except HTTPException:
        raise
    except Exception as e:
        print(f"[WARN] glow up: delete failed: {str(e)[:300]}")
        raise HTTPException(status_code=502, detail="Could not delete. Please try again.")
    return {"deleted": True, "photo_deleted": bool(path)}


@router.delete("/sessions")
async def delete_all_sessions(user_id: str = Depends(_user)):
    admin = _admin_or_503()
    try:
        rows = _select_sessions(admin, user_id)
        _remove_photos(admin, [r.get("photo_path") for r in rows])
        # Anything else under this user's folder (e.g. left by a failed save) goes too.
        try:
            leftovers = admin.storage.from_(PHOTOS_BUCKET).list(user_id) or []
            _remove_photos(admin, [f"{user_id}/{f['name']}" for f in leftovers if f.get("name")])
        except Exception:
            pass
        admin.table("glow_up_sessions").delete().eq("user_id", user_id).execute()
    except Exception as e:
        print(f"[WARN] glow up: delete all failed: {str(e)[:300]}")
        raise HTTPException(status_code=502, detail="Could not delete. Please try again.")
    return {"deleted": True}
