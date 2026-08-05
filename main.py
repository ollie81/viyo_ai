"""
Viyo AI Backend — FastAPI service providing AI-powered creator tools
"""

import os
import time
import logging
from collections import defaultdict, deque
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import jwt
from fastapi import FastAPI, HTTPException, Header, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, validator
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration & Environment Validation
# ============================================================================
class Config:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    ALLOW_INSECURE_AUTH = os.environ.get("ALLOW_INSECURE_AUTH", "").lower() == "true"
    RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "20"))
    RATE_WINDOW = int(os.environ.get("RATE_WINDOW", "3600"))  # seconds
    REDIS_URL = os.environ.get("REDIS_URL", None)
    ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
    
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required")
        if not cls.SUPABASE_JWT_SECRET and not cls.ALLOW_INSECURE_AUTH:
            raise RuntimeError(
                "SUPABASE_JWT_SECRET is not set. Set ALLOW_INSECURE_AUTH=true "
                "for local development only."
            )

Config.validate()

# ============================================================================
# Redis client for production rate limiting (optional)
# ============================================================================
redis_client = None
if Config.REDIS_URL:
    try:
        redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
        logger.info("Redis rate limiting enabled")
    except Exception as e:
        logger.warning(f"Redis connection failed, falling back to in-memory: {e}")

# ============================================================================
# Lifespan context manager for startup/shutdown
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting Viyo AI Backend in {Config.ENVIRONMENT} mode")
    yield
    # Shutdown
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")

app = FastAPI(
    title="Viyo AI Backend",
    version="2.0.0",
    description="AI-powered creator tools including content ideas, caption improvement, and post analysis",
    lifespan=lifespan
)

# ============================================================================
# CORS Configuration
# ============================================================================
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# ============================================================================
# OpenAI Client with retry configuration
# ============================================================================
client = OpenAI(api_key=Config.OPENAI_API_KEY)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def _call_openai_with_retry(**kwargs):
    """Enhanced retry with exponential backoff"""
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

# ============================================================================
# Rate Limiter (supports both Redis and in-memory)
# ============================================================================
class RateLimiter:
    def __init__(self):
        self._requests: Dict[str, deque] = defaultdict(deque)
    
    async def check(self, user_id: str) -> bool:
        if redis_client:
            return await self._check_redis(user_id)
        return await self._check_memory(user_id)
    
    async def _check_redis(self, user_id: str) -> bool:
        """Redis-based rate limiting for production"""
        key = f"rate_limit:{user_id}"
        now = int(time.time())
        window_start = now - Config.RATE_WINDOW
        
        # Remove old requests and count current
        await redis_client.zremrangebyscore(key, 0, window_start)
        count = await redis_client.zcard(key)
        
        if count >= Config.RATE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request
        await redis_client.zadd(key, {str(now): now})
        await redis_client.expire(key, Config.RATE_WINDOW)
        return True
    
    async def _check_memory(self, user_id: str) -> bool:
        """In-memory rate limiting (development only)"""
        now = time.time()
        q = self._requests[user_id]
        
        # Clean old requests
        while q and now - q[0] > Config.RATE_WINDOW:
            q.popleft()
        
        if len(q) >= Config.RATE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        q.append(now)
        return True

rate_limiter = RateLimiter()

# ============================================================================
# Auth: Supabase JWT verification
# ============================================================================
async def get_current_user_id(authorization: str = Header(None)) -> str:
    """Extract and verify user ID from Supabase JWT"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token"
        )
    
    token = authorization.removeprefix("Bearer ").strip()
    
    if not Config.SUPABASE_JWT_SECRET:
        # Local dev fallback
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
        except Exception as e:
            logger.warning(f"Local auth decode failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format"
            )
    else:
        try:
            payload = jwt.decode(
                token,
                Config.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"require": ["exp", "aud", "sub"]}
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired. Please log in again."
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token. Please log in again."
            )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user identifier"
        )
    return user_id

# ============================================================================
# Request/Response Models with enhanced validation
# ============================================================================
class ContentIdeasRequest(BaseModel):
    niche: str = Field(..., min_length=2, max_length=100, description="Content niche/topic")
    
    @validator('niche')
    def validate_niche(cls, v):
        if len(v.split()) < 2 and len(v) < 10:
            raise ValueError("Niche should be at least 2 words or 10 characters")
        return v.strip()

class ContentIdeasResponse(BaseModel):
    ideas: List[str] = Field(..., max_items=5)
    generated_at: float = Field(default_factory=time.time)

class ImproveCaptionRequest(BaseModel):
    caption: str = Field(..., min_length=3, max_length=2000)
    tone: Optional[str] = Field(None, description="Optional tone: casual, professional, witty, etc.")
    target_length: Optional[int] = Field(None, ge=50, le=500, description="Target character count")

class ImproveCaptionResponse(BaseModel):
    improved_caption: str
    original_length: int
    improved_length: int

class PostFeedbackRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    post_type: Optional[str] = Field(None, pattern="^(text|photo|video)$")

class PostFeedbackResponse(BaseModel):
    feedback: str
    improvement_tip: str

class AnalyzePostRequest(BaseModel):
    post_type: str = Field(..., pattern="^(text|photo|video)$")
    caption: str = Field(default="", max_length=2000)
    niche: str = Field(default="", max_length=100)
    image_url: Optional[HttpUrl] = Field(None, description="Public URL of the image")
    
    @validator('caption')
    def validate_caption_or_image(cls, v, values):
        if not v and not values.get('image_url'):
            raise ValueError("Either caption or image_url must be provided")
        return v

class AnalyzePostResponse(BaseModel):
    what_worked: str
    what_to_improve: str
    content_ideas: List[str]
    engagement_tip: str
    visual_analysis: bool
    analyzed_at: float = Field(default_factory=time.time)

# ============================================================================
# Try to import repurpose router
# ============================================================================
try:
    from repurpose import router as repurpose_router
    app.include_router(repurpose_router, prefix="/api")
    logger.info("Video repurposing endpoint loaded")
except ImportError as e:
    logger.warning(f"Video repurposing endpoint not loaded: {e}")

# ============================================================================
# Endpoint: Content Ideas
# ============================================================================
@app.post("/api/content-ideas", response_model=ContentIdeasResponse)
async def content_ideas(
    req: ContentIdeasRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Generate 5 content ideas based on user's niche"""
    await rate_limiter.check(user_id)
    
    prompt = f"""Generate 5 specific, actionable content ideas for a creator in the niche: "{req.niche}".

Requirements:
- Each idea must be ONE sentence, specific and actionable
- Focus on what would actually perform well on social media
- Include variety: educational, entertaining, engaging formats
- Return as a numbered list only, no additional text

Format:
1. [Idea]
2. [Idea]
3. [Idea]
4. [Idea]
5. [Idea]"""

    try:
        completion = _call_openai_with_retry(
            model=Config.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=300,
        )
        text = completion.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Content ideas generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service temporarily unavailable"
        )
    
    ideas = _parse_numbered_list(text, limit=5)
    if not ideas:
        # Fallback ideas
        ideas = [
            f"Create a 'day in the life' style video showing your {req.niche} process",
            f"Share your top 5 {req.niche} tips that helped you grow",
            f"Answer the most common questions you get about {req.niche}",
            f"Create a behind-the-scenes look at your {req.niche} workflow",
            f"Share a mistake you made in {req.niche} and what you learned"
        ]
    
    return ContentIdeasResponse(ideas=ideas[:5])

# ============================================================================
# Endpoint: Improve Caption
# ============================================================================
@app.post("/api/improve-caption", response_model=ImproveCaptionResponse)
async def improve_caption(
    req: ImproveCaptionRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Improve a social media caption for better engagement"""
    await rate_limiter.check(user_id)
    
    tone_instruction = f" Use a {req.tone} tone." if req.tone else ""
    length_instruction = f" Keep it under {req.target_length} characters." if req.target_length else " Keep it under 220 characters."
    
    prompt = f"""Improve this social media caption to be more engaging.

Requirements:
- Keep the original meaning and core message{tone_instruction}
- {length_instruction}
- Don't add hashtags unless the original had them
- Make it conversational and compelling
- Return ONLY the improved caption, nothing else

Original caption: {req.caption}"""

    try:
        completion = _call_openai_with_retry(
            model=Config.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        improved = (completion.choices[0].message.content or "").strip().strip('"')
    except Exception as e:
        logger.error(f"Caption improvement failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service temporarily unavailable"
        )
    
    if not improved:
        improved = req.caption
    
    return ImproveCaptionResponse(
        improved_caption=improved,
        original_length=len(req.caption),
        improved_length=len(improved)
    )

# ============================================================================
# Endpoint: Post Feedback
# ============================================================================
@app.post("/api/post-feedback", response_model=PostFeedbackResponse)
async def post_feedback(
    req: PostFeedbackRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Get supportive feedback and improvement tips for a post"""
    await rate_limiter.check(user_id)
    
    prompt = f"""You are a supportive creator coach. Analyze this post and provide feedback.

Requirements:
- LINE 1: One short, genuinely positive sentence about what works well
- LINE 2: One short, specific, actionable improvement tip
- No preamble, no extra lines, no markdown
- Focus on helping the creator grow, not criticizing

Post content: {req.content}"""

    try:
        completion = _call_openai_with_retry(
            model=Config.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Post feedback generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service temporarily unavailable"
        )
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    feedback = lines[0] if len(lines) > 0 else "Great effort putting this content out there!"
    tip = lines[1] if len(lines) > 1 else "Consider adding a clear call-to-action to boost engagement."
    
    return PostFeedbackResponse(feedback=feedback, improvement_tip=tip)

# ============================================================================
# Endpoint: Analyze Post (AI Creator Coach with Vision)
# ============================================================================
@app.post("/api/analyze-post", response_model=AnalyzePostResponse)
async def analyze_post(
    req: AnalyzePostRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    AI Creator Coach with vision capabilities. Analyzes both the caption
    and optional image (photo or video thumbnail) to provide structured feedback.
    """
    await rate_limiter.check(user_id)
    
    has_image = bool(req.image_url)
    niche_context = f" The creator's niche is '{req.niche}'." if req.niche else ""
    
    instruction = f"""You are a warm, encouraging creator coach. A creator just posted the following content.

Post type: {req.post_type}
Caption: {req.caption or '(no caption)'}
{niche_context}

"""

    if has_image:
        instruction += """An image is attached - this is the actual photo (or a frame from a video). 
Look at it carefully: composition, framing, lighting, colors, focus, what's happening in the shot.
Ground your feedback in what you actually see, not generic advice.
"""

    instruction += """Respond with exactly 4 sections, each on its own line:

WORKED: <One encouraging sentence about something genuinely good>
IMPROVE: <One specific, actionable thing to improve>
IDEAS: <Three content ideas separated by ' | '>
ENGAGEMENT: <One specific engagement tip>

No markdown, no extra text, just these 4 lines."""

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
    if has_image:
        user_content.append({"type": "image_url", "image_url": {"url": str(req.image_url)}})

    try:
        completion = _call_openai_with_retry(
            model=Config.MODEL,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.8,
            max_tokens=400,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Post analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service temporarily unavailable"
        )
    
    parsed = _parse_coach_response(text)
    if parsed is None:
        # Fallback response
        parsed = {
            "what_worked": "You're consistently creating content and building your presence!",
            "what_to_improve": "Try adding more specific value or unique perspective to stand out.",
            "content_ideas": ["Share a personal story related to this", "Create a tutorial on this topic", "Ask your audience a question"],
            "engagement_tip": "End with a specific question to encourage comments."
        }
    
    return AnalyzePostResponse(**parsed, visual_analysis=has_image)

# ============================================================================
# Helper Functions
# ============================================================================
def _parse_numbered_list(text: str, limit: int) -> List[str]:
    """Parse numbered list from AI response"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    ideas = []
    
    for line in lines:
        # Remove numbering (1., 2., etc.)
        cleaned = line
        for prefix_len in range(1, 4):
            if len(cleaned) > prefix_len:
                prefix = cleaned[:prefix_len]
                if prefix.rstrip(".)- ").isdigit():
                    cleaned = cleaned[prefix_len:].lstrip(".)- ").strip()
                    break
        
        # Remove bullet points
        cleaned = cleaned.lstrip("-•* ").strip()
        
        if cleaned and len(cleaned) > 5:  # Minimum length to filter out noise
            ideas.append(cleaned)
        
        if len(ideas) >= limit:
            break
    
    return ideas

def _parse_coach_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse structured response from AI coach"""
    lines_dict = {}
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        for key in ("WORKED", "IMPROVE", "IDEAS", "ENGAGEMENT"):
            prefix = f"{key}:"
            if line.upper().startswith(prefix):
                lines_dict[key] = line[len(prefix):].strip()
                break
    
    if not all(k in lines_dict for k in ("WORKED", "IMPROVE", "IDEAS", "ENGAGEMENT")):
        return None
    
    ideas = [i.strip() for i in lines_dict["IDEAS"].split("|") if i.strip()]
    if not ideas:
        ideas = [lines_dict["IDEAS"]]
    
    return {
        "what_worked": lines_dict["WORKED"],
        "what_to_improve": lines_dict["IMPROVE"],
        "content_ideas": ideas[:3],
        "engagement_tip": lines_dict["ENGAGEMENT"],
    }

# ============================================================================
# Exception Handlers
# ============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with better logging"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected error occurred. Please try again later."}
    )

# ============================================================================
# Health & Root Endpoints
# ============================================================================
@app.get("/health")
async def health():
    """Health check endpoint for monitoring"""
    status_data = {
        "status": "healthy",
        "environment": Config.ENVIRONMENT,
        "model": Config.MODEL,
        "auth_configured": bool(Config.SUPABASE_JWT_SECRET),
        "redis_available": bool(redis_client),
        "version": "2.0.0"
    }
    return status_data

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Viyo AI Backend API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": [
            {"path": "/api/content-ideas", "method": "POST"},
            {"path": "/api/improve-caption", "method": "POST"},
            {"path": "/api/post-feedback", "method": "POST"},
            {"path": "/api/analyze-post", "method": "POST"},
            {"path": "/health", "method": "GET"},
        ]
    }

# ============================================================================
# Server Startup
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info")
    )
