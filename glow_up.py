"""
Viyo Glow Up — Stage 1: style suggestions from a selfie.

Privacy by design:
  * The selfie is NEVER stored. It arrives in the request, is checked and
    sent to the vision model, and is dropped when the request ends. Only the
    text suggestions are saved, and only if the user asks (save=true).
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
IMAGE_DETAIL = os.environ.get("GLOW_UP_IMAGE_DETAIL", "low")
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
    return {"faces": len(faces), "face_width": round(main.w, 3), "detector": detector}


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


class GlowUpResult(BaseModel):
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
- If there isn't one clear, front-facing face in the photo, return only {"status": "no_clear_face"}."""

FORMAT = """Return ONLY JSON:
{"summary": "1-2 encouraging sentences on a style direction that suits them",
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


class AnalyzeResponse(BaseModel):
    result: GlowUpResult
    saved: bool
    session_id: Optional[str] = None
    coins_charged: int


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
        raise HTTPException(status_code=503, detail=f"Glow Up is temporarily unavailable. Nothing was charged. ({e})")

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
        raise HTTPException(status_code=502,
                            detail=f"We couldn't create your suggestions right now. Your coins were refunded. ({e})")
    finally:
        del jpeg, data_url    # the photo is never kept

    saved, session_id = False, None
    if req.save and admin is not None:
        try:
            row = admin.table("glow_up_sessions").insert({
                "user_id": user_id, "recommendations": result.model_dump(),
                "style_goal": style_goal, "presentation": presentation,
            }).execute()
            session_id = (row.data or [{}])[0].get("id")
            saved = session_id is not None
        except Exception as e:
            print(f"[WARN] glow up: could not save suggestions: {e}")
    return AnalyzeResponse(result=result, saved=saved, session_id=session_id, coins_charged=charged)


@router.get("/sessions")
async def list_sessions(user_id: str = Depends(_user)):
    admin = _get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Glow Up is not configured.")
    try:
        rows = (admin.table("glow_up_sessions").select("id, created_at, style_goal, presentation, recommendations")
                .eq("user_id", user_id).order("created_at", desc=True).limit(50).execute())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not load your Glow Ups: {e}")
    return {"sessions": rows.data or []}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(_user)):
    admin = _get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Glow Up is not configured.")
    try:
        admin.table("glow_up_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not delete: {e}")
    return {"deleted": True}


@router.delete("/sessions")
async def delete_all_sessions(user_id: str = Depends(_user)):
    admin = _get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Glow Up is not configured.")
    try:
        admin.table("glow_up_sessions").delete().eq("user_id", user_id).execute()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not delete: {e}")
    return {"deleted": True}
