import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from openai import OpenAI
from supabase import create_client, Client


router = APIRouter(
    prefix="/api/v1",
    tags=["coach"],
)

ai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY", ""
)

supabase_admin: Optional[Client] = None

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
    )


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------

async def _get_current_user_id(
    authorization: str = Header(None),
) -> str:

    from main import get_current_user_id

    return await get_current_user_id(authorization)


# ---------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------

class CoachMessageRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=4000)

    # Optional information about the current video version.
    video_version: int = Field(default=1, ge=1)

    # Optional score if this message is reporting a new analysis.
    score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
    )


class CoachMessageResponse(BaseModel):
    video_id: str
    response: str
    video_version: int
    score: Optional[int] = None


# ---------------------------------------------------------
# Save a Coach message
# ---------------------------------------------------------

def _save_message(
    user_id: str,
    video_id: str,
    role: str,
    message: str,
    video_version: int,
    score: Optional[int] = None,
):

    if supabase_admin is None:
        raise HTTPException(
            status_code=503,
            detail="Coach database is not configured.",
        )

    try:
        result = (
            supabase_admin
            .table("video_coach_messages")
            .insert({
                "user_id": user_id,
                "video_id": video_id,
                "role": role,
                "message": message,
                "video_version": video_version,
                "score": score,
            })
            .execute()
        )

        return result.data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save Coach message: {e}",
        )


# ---------------------------------------------------------
# Get Coach history for one video
# ---------------------------------------------------------

@router.get("/coach/{video_id}")
async def get_coach_history(
    video_id: str,
    user_id: str = Depends(_get_current_user_id),
):

    if supabase_admin is None:
        raise HTTPException(
            status_code=503,
            detail="Coach database is not configured.",
        )

    try:
        result = (
            supabase_admin
            .table("video_coach_messages")
            .select(
                "id,video_id,role,message,video_version,score,created_at"
            )
            .eq("user_id", user_id)
            .eq("video_id", video_id)
            .order("created_at", desc=False)
            .execute()
        )

        return {
            "video_id": video_id,
            "messages": result.data or [],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load Coach history: {e}",
        )


# ---------------------------------------------------------
# AI Coach
# ---------------------------------------------------------

@router.post(
    "/coach/message",
    response_model=CoachMessageResponse,
)
async def coach_message(
    req: CoachMessageRequest,
    user_id: str = Depends(_get_current_user_id),
):

    if supabase_admin is None:
        raise HTTPException(
            status_code=503,
            detail="Coach database is not configured.",
        )

    # -----------------------------------------------------
    # Load previous conversation for THIS video only
    # -----------------------------------------------------

    try:
        history_result = (
            supabase_admin
            .table("video_coach_messages")
            .select(
                "role,message,video_version,score,created_at"
            )
            .eq("user_id", user_id)
            .eq("video_id", req.video_id)
            .order("created_at", desc=False)
            .limit(30)
            .execute()
        )

        history = history_result.data or []

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load Coach history: {e}",
        )

    # -----------------------------------------------------
    # Save creator's new message first
    # -----------------------------------------------------

    _save_message(
        user_id=user_id,
        video_id=req.video_id,
        role="user",
        message=req.message,
        video_version=req.video_version,
        score=req.score,
    )

    # -----------------------------------------------------
    # Build conversation
    # -----------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": """
You are Viyo Coach, a professional AI coach for content creators.

Your job is NOT simply to compliment the creator.

You should help the creator make better videos.

Analyze and discuss:

- Hook strength
- First few seconds
- Viewer retention
- Clarity
- Story structure
- Pacing
- Unnecessary pauses
- Repetition
- Emotional impact
- Value to the viewer
- Ending
- Captions
- Titles
- Content quality
- Audience fit
- Short-form potential

IMPORTANT RULES:

1. Be honest.
2. Give specific and practical advice.
3. Never invent something the creator did not say or show.
4. If you do not have enough information, say so.
5. Remember the conversation history for this specific video.
6. Do not mix this video's history with another video.
7. If the creator says they changed something, compare the new version
   with the previous feedback when possible.
8. Explain what improved.
9. Explain what still needs improvement.
10. Give the creator clear next steps.
11. Preserve the creator's spoken language when discussing captions.
12. Do not assume the creator speaks English.
13. Do not automatically translate their content unless they request it.
14. Be encouraging but honest.

When appropriate, give a score from 0 to 100.

A score should reflect the current version of the video, not the creator
as a person.

You are a coach, not a judge.
"""
        }
    ]

    # Add previous history
    for item in history:

        role = item.get("role")

        if role not in ("user", "coach"):
            continue

        messages.append({
            "role": "user" if role == "user" else "assistant",
            "content": item.get("message", ""),
        })

    # Add current message
    messages.append({
        "role": "user",
        "content": req.message,
    })

    # -----------------------------------------------------
    # Ask OpenAI
    # -----------------------------------------------------

    try:

        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
        )

        coach_response = (
            response.choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=f"Coach AI request failed: {e}",
        )

    # -----------------------------------------------------
    # Save Coach response
    # -----------------------------------------------------

    _save_message(
        user_id=user_id,
        video_id=req.video_id,
        role="coach",
        message=coach_response,
        video_version=req.video_version,
        score=req.score,
    )

    return CoachMessageResponse(
        video_id=req.video_id,
        response=coach_response,
        video_version=req.video_version,
        score=req.score,
    )
