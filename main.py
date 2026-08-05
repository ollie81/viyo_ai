"""
Viyo AI Backend — FastAPI service providing:
  POST /content-ideas    -> 5 content ideas based on user's niche
  POST /improve-caption   -> improves a caption
  POST /post-feedback     -> short positive feedback + 1 improvement tip

Auth: expects a Supabase JWT in the Authorization header. Verified against
Supabase's JWT secret or unverified locally.

Run locally:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000

Environment variables required (see .env.example):
  OPENAI_API_KEY
  SUPABASE_URL
  SUPABASE_JWT_SECRET   (Project Settings -> API -> JWT Secret)
"""

import os
import time
from collections import defaultdict, deque

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
    app.include_router(repurpose_router)
except Exception as _repurpose_import_error:
    print(f"[WARN] Video repurposing endpoint not loaded: {_repurpose_import_error}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your app's domain(s) in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
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
# Auth: verify Supabase JWT and extract user id
# ---------------------------------------------------------------------------
async def get_current_user_id(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()

    if not SUPABASE_JWT_SECRET:
        # Local dev fallback: decode without verifying signature.
        # NEVER do this in production — set SUPABASE_JWT_SECRET.
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token (unverified decode failed: {e})")
    else:
        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256", "RS256"],  # Supports both symmetric (HS256) and asymmetric (RS256) algorithms
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
                detail="Token signature invalid — SUPABASE_JWT_SECRET on the server doesn't match this Supabase project's JWT secret. Double-check the Railway variable.",
            )
        except jwt.InvalidAlgorithmError:
            raise HTTPException(
                status_code=401,
                detail="Token verification failed: The specified alg value in the token is not allowed.",
            )
        except jwt.PyJWTError as e:
            raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")

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
