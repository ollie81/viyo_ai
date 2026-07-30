"""
Viyo AI Backend — FastAPI service providing:
  POST /content-ideas    -> 5 content ideas based on user's niche
  POST /improve-caption   -> improves a caption
  POST /post-feedback     -> short positive feedback + 1 improvement tip

Auth: expects a Supabase JWT in the Authorization header. Verified against
Supabase's JWKS endpoint so this service never needs the Supabase service
role key just to identify the caller.

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
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
    else:
        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
            )
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

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
        completion = client.chat.completions.create(
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
        completion = client.chat.completions.create(
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
        completion = client.chat.completions.create(
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
        completion = client.chat.completions.create(
            model=MODEL,
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


# ========================================================================
# SERVER STARTUP BLOCK - REQUIRED FOR RAILWAY DEPLOYMENT
# ========================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


@app.get("/")
def root():
    return {"message": "API is running! Visit /docs for Swagger UI"}

