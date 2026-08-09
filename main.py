"""
Viyo AI Backend — FastAPI service providing:
  POST /content-ideas    -> 5 content ideas based on user's niche
  POST /improve-caption   -> improves a caption
  POST /post-feedback     -> short positive feedback + 1 improvement tip
  POST /analyze-post      -> AI Creator Coach with vision capabilities
  POST /repurpose         -> Video repurposing (if repurpose.py is available)

Auth: expects a Supabase JWT in the Authorization header. Verified against
Supabase's JWKS endpoint so this service never needs the Supabase service
role key just to identify the caller.

Run locally:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000

Environment variables required (see .env.example):
  OPENAI_API_KEY
  SUPABASE_URL          (e.g. https://xyzxyz.supabase.co)
  ALLOW_INSECURE_AUTH   set to "true" for local dev only (skips JWKS verification)
"""

import json
import os
import time
import urllib.request
from collections import defaultdict, deque
from typing import Optional

import jwt
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="Viyo AI Backend", version="1.0.0")

# Video repurposing lives in its own file (repurpose.py) so a bug there
# can't take down the working endpoints above. If this import fails
# (e.g. missing SUPABASE_SERVICE_ROLE_KEY dependency not installed yet),
# main.py still runs — /repurpose just won't exist until it's fixed.
try:
   from repurpose import router as repurpose_router
from coach import router as coach_router

app.include_router(repurpose_router)
app.include_router(coach_router)
except Exception as _repurpose_import_error:
    print(f"[WARN] Video repurposing endpoint not loaded: {_repurpose_import_error}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your app's domain(s) in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _call_openai_with_retry(**kwargs):
    """One retry on transient OpenAI errors (timeouts, momentary 5xx) before
    giving up — cuts down on "AI service unavailable" for blips that would
    have succeeded a second later."""
    last_err = None
    for attempt in range(2):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(0.5)
    raise last_err

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Simple in-memory per-user rate limiter (swap for Redis in production —
# this resets on every restart and doesn't work across multiple instances).
# ---------------------------------------------------------------------------
RATE_LIMIT = 20          # max requests
RATE_WINDOW = 60 * 60    # per hour, in seconds
_user_requests: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(user_id: str):
    now = time.time()
    q = _user_requests[user_id]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    q.append(now)


# ---------------------------------------------------------------------------
# Auth: verify Supabase JWT via JWKS (ES256 / P-256)
# ---------------------------------------------------------------------------
# Supabase projects using ECC signing publish their public keys at:
#   {SUPABASE_URL}/auth/v1/.well-known/jwks.json
# We fetch those keys once, cache them for 1 hour, and verify tokens
# locally using PyJWT — no JWT secret needed on this server at all.
#
# The only way to disable verification is ALLOW_INSECURE_AUTH=true, which
# is intended for local dev only. A missing SUPABASE_URL in production
# fails loudly at startup instead of silently accepting forged tokens.

ALLOW_INSECURE_AUTH = os.environ.get("ALLOW_INSECURE_AUTH", "").lower() == "true"

if not SUPABASE_URL and not ALLOW_INSECURE_AUTH:
    raise RuntimeError(
        "SUPABASE_URL is not set. Refusing to start — cannot fetch JWKS for "
        "token verification. Set SUPABASE_URL in your Railway environment "
        "variables (e.g. https://xyzxyz.supabase.co). For local dev without "
        "a Supabase project, set ALLOW_INSECURE_AUTH=true explicitly."
    )

# JWKS cache — avoids a network round-trip on every request.
# Refreshed automatically after TTL, or immediately on unknown kid.
_jwks_cache: Optional[dict] = None
_jwks_fetched_at: float = 0.0
_JWKS_TTL = 3600  # seconds


def _fetch_jwks(force: bool = False) -> dict:
    """Return the cached JWKS, refreshing if stale or forced."""
    global _jwks_cache, _jwks_fetched_at
    now = time.time()
    if not force and _jwks_cache and now - _jwks_fetched_at < _JWKS_TTL:
        return _jwks_cache
    url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            _jwks_cache = json.loads(resp.read())
            _jwks_fetched_at = now
            return _jwks_cache
    except Exception as exc:
        if _jwks_cache:
            # Serve stale cache on transient network errors — better than
            # rejecting every user because of a momentary DNS blip.
            return _jwks_cache
        raise HTTPException(
            status_code=503,
            detail=f"Auth service unavailable — could not fetch JWKS: {exc}",
        )


def _public_key_for_token(token: str):
    """
    Extract the kid from the token header, look it up in the JWKS, and
    return a cryptography public-key object that PyJWT can verify against.

    On an unknown kid we force-refresh the JWKS once (Supabase may have
    rotated keys) before giving up.
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Malformed token header: {exc}")

    kid = header.get("kid")
    alg = header.get("alg", "ES256")

    def _find_key(jwks: dict, kid: Optional[str]):
        keys = jwks.get("keys", [])
        if kid:
            matches = [k for k in keys if k.get("kid") == kid]
        else:
            # No kid in header — fall back to the first EC key.
            matches = [k for k in keys if k.get("kty") == "EC"]
        return matches[0] if matches else None

    jwks = _fetch_jwks()
    key_data = _find_key(jwks, kid)

    if key_data is None and kid:
        # kid not in cache — try a forced refresh (key rotation).
        jwks = _fetch_jwks(force=True)
        key_data = _find_key(jwks, kid)

    if key_data is None:
        raise HTTPException(
            status_code=401,
            detail=f"Token signed with unknown key (kid={kid!r}). "
                   "If you just rotated Supabase keys, wait up to 1 hour or restart the server.",
        )

    try:
        return jwt.algorithms.ECAlgorithm.from_jwk(json.dumps(key_data))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load public key: {exc}")


async def get_current_user_id(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()

    if not SUPABASE_URL:
        # Local dev fallback — only reachable when ALLOW_INSECURE_AUTH=true,
        # enforced by the startup check above. NEVER in production.
        # algorithms= required by PyJWT 2.x even with verify_signature=False.
        try:
            payload = jwt.decode(
                token,
                options={"verify_signature": False},
                algorithms=["ES256"],
            )
        except Exception as exc:
            raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")
    else:
        public_key = _public_key_for_token(token)
        try:
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["ES256"],
                audience="authenticated",
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token expired — log out and back in on the app, then try again.",
            )
        except jwt.InvalidSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token signature invalid — the token may have been tampered with.",
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail=f"Token verification failed: {exc}")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")
    return user_id


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class ContentIdeasRequest(BaseModel):
    niche: str = Field(..., min_length=1, max_length=100)


class ContentIdeasResponse(BaseModel):
    ideas: list[str]


class ImproveCaptionRequest(BaseModel):
    caption: str = Field(..., min_length=1, max_length=2000)


class ImproveCaptionResponse(BaseModel):
    improved_caption: str


class PostFeedbackRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)


class PostFeedbackResponse(BaseModel):
    feedback: str
    improvement_tip: str


class AnalyzePostRequest(BaseModel):
    post_type: str = Field(..., pattern="^(text|photo|video)$")
    caption: str = Field(default="", max_length=2000)
    niche: str = Field(default="", max_length=100)
    # Public URL of the photo, or a video thumbnail/frame for video posts.
    # Optional — if omitted, feedback falls back to caption-only analysis
    # (e.g. text posts have no image to look at).
    image_url: str | None = Field(default=None, max_length=2000)


class AnalyzePostResponse(BaseModel):
    what_worked: str
    what_to_improve: str
    content_ideas: list[str]
    engagement_tip: str
    visual_analysis: bool  # true if this feedback actually looked at the image, not just the caption
# ---------------------------------------------------------------------------
from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "traceback": traceback.format_exc()
        }
    )

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/content-ideas", response_model=ContentIdeasResponse)
async def content_ideas(
    req: ContentIdeasRequest,
    user_id: str = Depends(get_current_user_id),
):
    _check_rate_limit(user_id)

    prompt = (
        f"Give 5 short, punchy content ideas for a creator whose niche is "
        f"'{req.niche}'. Each idea should be one sentence, specific, and "
        f"actionable (not generic). Return them as a numbered list, nothing else."
    )

    try:
        completion = _call_openai_with_retry(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=300,
        )
        text = completion.choices[0].message.content or ""
    except Exception:
        raise HTTPException(status_code=502, detail="AI service unavailable")

    ideas = _parse_numbered_list(text, limit=5)
    if not ideas:
        raise HTTPException(status_code=502, detail="AI returned an unexpected format")

    return ContentIdeasResponse(ideas=ideas)


@app.post("/improve-caption", response_model=ImproveCaptionResponse)
async def improve_caption(
    req: ImproveCaptionRequest,
    user_id: str = Depends(get_current_user_id),
):
    _check_rate_limit(user_id)

    prompt = (
        "Improve this social media caption to be more engaging, keep the "
        "original meaning and tone, keep it under 220 characters, and do not "
        "add hashtags unless the original had them. Return only the improved "
        f"caption, nothing else.\n\nOriginal caption: {req.caption}"
    )

    try:
        completion = _call_openai_with_retry(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        improved = (completion.choices[0].message.content or "").strip().strip('"')
    except Exception:
        raise HTTPException(status_code=502, detail="AI service unavailable")

    return ImproveCaptionResponse(improved_caption=improved or req.caption)


@app.post("/post-feedback", response_model=PostFeedbackResponse)
async def post_feedback(
    req: PostFeedbackRequest,
    user_id: str = Depends(get_current_user_id),
):
    _check_rate_limit(user_id)

    prompt = (
        "You are a supportive creator coach. Given this post content, respond "
        "with exactly two lines:\n"
        "LINE 1: One short, genuinely positive sentence about it.\n"
        "LINE 2: One short, specific, actionable improvement tip.\n"
        "No preamble, no extra lines, no markdown.\n\n"
        f"Post content: {req.content}"
    )

    try:
        completion = _call_openai_with_retry(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception:
        raise HTTPException(status_code=502, detail="AI service unavailable")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    feedback = lines[0] if len(lines) > 0 else "Nice work getting this posted!"
    tip = lines[1] if len(lines) > 1 else "Try adding a clear call-to-action at the end."

    return PostFeedbackResponse(feedback=feedback, improvement_tip=tip)


@app.post("/analyze-post", response_model=AnalyzePostResponse)
async def analyze_post(
    req: AnalyzePostRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    The AI Creator Coach. Runs right after a creator posts, and gives
    structured feedback framed as coaching, not grading — the goal is
    "help this person get better," not "score this post."

    When `image_url` is provided (a photo, or a video thumbnail/frame),
    the coach actually looks at the image via GPT-4o's vision input —
    composition, lighting, framing, what's in the shot — instead of
    only reacting to the caption text. This is what lets feedback say
    something like "the subject is off-center and half-cropped" rather
    than generic caption advice.
    """
    _check_rate_limit(user_id)

    niche_context = f" Their content niche is '{req.niche}'." if req.niche else ""
    has_image = bool(req.image_url)

    instruction = (
        "You are a warm, encouraging creator coach — think a supportive mentor, "
        "not a critic. A creator just posted the following content.\n\n"
        f"Post type: {req.post_type}\n"
        f"Caption: {req.caption or '(no caption)'}\n"
        f"{niche_context}\n\n"
    )

    if has_image:
        instruction += (
            "An image is attached below — this is either the actual photo posted, "
            "or a representative frame from the video posted. Look at it directly: "
            "composition, framing, lighting, color, what's in focus, what's happening "
            "in the shot. Ground your feedback in what you actually see, not generic "
            "advice that could apply to any post.\n\n"
        )

    instruction += (
        "Respond with exactly 4 sections, each on its own line, in this exact format "
        "(no markdown, no extra commentary):\n"
        "WORKED: <one encouraging sentence about something genuinely good here"
        + (", grounded in what's visible in the image" if has_image else "") + ">\n"
        "IMPROVE: <one specific, actionable thing they could do better next time"
        + (" — reference the actual framing/lighting/composition if relevant" if has_image else "")
        + ">\n"
        "IDEAS: <three short content ideas separated by ' | ', related to this post/niche>\n"
        "ENGAGEMENT: <one specific tip to get more engagement on posts like this>"
    )

    user_content: list[dict] = [{"type": "text", "text": instruction}]
    if has_image:
        user_content.append({"type": "image_url", "image_url": {"url": req.image_url}})

    try:
        completion = _call_openai_with_retry(
            model=MODEL,  # Using gpt-4o-mini for cost savings
            messages=[{"role": "user", "content": user_content}],
            temperature=0.8,
            max_tokens=400,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception:
        raise HTTPException(status_code=502, detail="AI service unavailable")

    parsed = _parse_coach_response(text)
    if parsed is None:
        raise HTTPException(status_code=502, detail="AI returned an unexpected format")

    return AnalyzePostResponse(**parsed, visual_analysis=has_image)


def _parse_coach_response(text: str) -> dict | None:
    lines = {}
    for line in text.splitlines():
        line = line.strip()
        for key in ("WORKED", "IMPROVE", "IDEAS", "ENGAGEMENT"):
            prefix = f"{key}:"
            if line.upper().startswith(prefix):
                lines[key] = line[len(prefix):].strip()
                break

    if not all(k in lines for k in ("WORKED", "IMPROVE", "IDEAS", "ENGAGEMENT")):
        return None

    ideas = [i.strip() for i in lines["IDEAS"].split("|") if i.strip()]
    if not ideas:
        ideas = [lines["IDEAS"]]

    return {
        "what_worked": lines["WORKED"],
        "what_to_improve": lines["IMPROVE"],
        "content_ideas": ideas[:3],
        "engagement_tip": lines["ENGAGEMENT"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_numbered_list(text: str, limit: int) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    ideas = []
    for line in lines:
        # strip leading "1.", "1)", "-", etc.
        cleaned = line
        for prefix_len in range(1, 4):
            if len(cleaned) > prefix_len and cleaned[:prefix_len].rstrip(".)- ").isdigit():
                cleaned = cleaned[prefix_len:].lstrip(".)- ").strip()
                break
        cleaned = cleaned.lstrip("-• ").strip()
        if cleaned:
            ideas.append(cleaned)
    return ideas[:limit]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "API is running! Visit /docs for Swagger UI"}


# ========================================================================
# SERVER STARTUP BLOCK - REQUIRED FOR RAILWAY DEPLOYMENT
# ========================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
