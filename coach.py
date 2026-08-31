import datetime
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
# Outcome follow-up: close the loop between coaching and what
# actually happened to the post.
#
# The Coach previously only ever gave an opinion (a score + advice)
# with nothing to check it against. This adds a lazy, on-demand check
# — triggered by the creator reopening a coached video's chat, since
# this deploy has no background worker/cron to run it proactively —
# that compares the post's actual in-app engagement (likes + comments,
# the only performance data Viyo has without a TikTok/Instagram/YouTube
# integration) against the creator's own recent average, and appends a
# coach message closing the loop.
# ---------------------------------------------------------

_OUTCOME_PREFIX = "\U0001F4CA How this one did:"
_OUTCOME_DELAY_HOURS = 24
_BASELINE_POST_COUNT = 10
_MIN_BASELINE_POSTS = 3


def _has_outcome_followup(history: list[dict], video_version: int) -> bool:
    return any(
        item.get("video_version") == video_version
        and str(item.get("message", "")).startswith(_OUTCOME_PREFIX)
        for item in history
    )


def _parse_timestamp(value: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _maybe_add_outcome_followup(
    user_id: str,
    video_id: str,
    history: list[dict],
) -> list[dict]:
    """
    If the creator got scored coaching on this video at least
    _OUTCOME_DELAY_HOURS ago and hasn't seen an outcome follow-up yet,
    append a coach message comparing the post's actual engagement to
    the creator's own baseline. Returns the newly saved message(s) as a
    list (empty if nothing was added), so callers can just concatenate.

    Deliberately best-effort: any failure here (missing data, a query
    error, not enough post history yet) just skips the follow-up rather
    than breaking the coach history the creator is trying to load.
    """
    if supabase_admin is None:
        return []

    scored = [item for item in history if item.get("score") is not None]
    if not scored:
        return []

    latest = scored[-1]
    video_version = latest.get("video_version", 1)
    if _has_outcome_followup(history, video_version):
        return []

    scored_at = _parse_timestamp(latest.get("created_at", ""))
    if scored_at is None:
        return []

    now = datetime.datetime.now(datetime.timezone.utc)
    if now - scored_at < datetime.timedelta(hours=_OUTCOME_DELAY_HOURS):
        return []

    try:
        post_result = (
            supabase_admin
            .table("posts")
            .select("id,like_count,comment_count")
            .eq("id", video_id)
            .limit(1)
            .execute()
        )
        posts = post_result.data or []
        if not posts:
            return []
        post = posts[0]

        recent_result = (
            supabase_admin
            .table("posts")
            .select("like_count,comment_count")
            .eq("user_id", user_id)
            .neq("id", video_id)
            .order("created_at", desc=True)
            .limit(_BASELINE_POST_COUNT)
            .execute()
        )
        recent = recent_result.data or []
    except Exception:
        return []

    if len(recent) < _MIN_BASELINE_POSTS:
        return []

    baseline = sum(
        (p.get("like_count") or 0) + (p.get("comment_count") or 0) for p in recent
    ) / len(recent)
    actual = (post.get("like_count") or 0) + (post.get("comment_count") or 0)

    if baseline <= 0:
        comparison = f"it picked up {actual} likes and comments combined"
        takeaway = "Not enough history yet to compare that against — keep posting."
    else:
        delta_pct = round((actual - baseline) / baseline * 100)
        if delta_pct >= 10:
            comparison = (
                f"it's running {delta_pct}% above your last {len(recent)} posts' "
                f"average ({actual} vs. an average of {baseline:.0f})"
            )
            takeaway = "That lines up with the feedback above — keep doing what worked here."
        elif delta_pct <= -10:
            comparison = (
                f"it's running {abs(delta_pct)}% below your last {len(recent)} posts' "
                f"average ({actual} vs. an average of {baseline:.0f})"
            )
            takeaway = "Worth comparing this one against the feedback above to see what to change next time."
        else:
            comparison = (
                f"it's about in line with your last {len(recent)} posts' average "
                f"({actual} vs. an average of {baseline:.0f})"
            )
            takeaway = "Consistent is fine, but the feedback above still has ideas for pushing it higher."

    score = latest.get("score")
    score_line = f"you scored {score}/100 on this one, and " if score is not None else ""
    message_text = f"{_OUTCOME_PREFIX} {score_line}{comparison}. {takeaway}"

    try:
        saved = _save_message(
            user_id=user_id,
            video_id=video_id,
            role="coach",
            message=message_text,
            video_version=video_version,
            score=None,
        )
        return saved or []
    except HTTPException:
        return []


# ---------------------------------------------------------
# Account deletion
#
# Nothing in this codebase could previously delete a creator's account
# or the content they uploaded — the Flutter app has no service-role
# key (by design, see main.py's auth comments), so this has to live on
# the backend, which already holds one for the Coach's Supabase admin
# client.
#
# This is a best-effort cascade across the tables this backend knows
# about (coach history, posts, profile), followed by the Supabase auth
# user itself — not a guarantee that every row everywhere is gone.
# There are no migration files in this repo to confirm foreign-key
# cascade behavior for other tables (coins/transactions, missions,
# follows, likes, comments), so verify that directly in Supabase before
# relying on this alone for a compliance/GDPR "right to erasure" claim.
# ---------------------------------------------------------

_DELETE_TABLES = {
    "video_coach_messages": "user_id",
    "posts": "user_id",
    "profiles": "id",
}


@router.delete("/account")
async def delete_account(
    user_id: str = Depends(_get_current_user_id),
):
    if supabase_admin is None:
        raise HTTPException(
            status_code=503,
            detail="Account service is not configured.",
        )

    cleanup_errors = []
    for table, column in _DELETE_TABLES.items():
        try:
            supabase_admin.table(table).delete().eq(column, user_id).execute()
        except Exception as e:
            cleanup_errors.append(f"{table}: {e}")

    try:
        supabase_admin.auth.admin.delete_user(user_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not delete account: {e}. "
                f"Partial cleanup errors before this: {cleanup_errors}"
                if cleanup_errors
                else f"Could not delete account: {e}"
            ),
        )

    return {"deleted": True, "partial_cleanup_errors": cleanup_errors}


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

        messages = result.data or []
        messages += _maybe_add_outcome_followup(user_id, video_id, messages)

        return {
            "video_id": video_id,
            "messages": messages,
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
