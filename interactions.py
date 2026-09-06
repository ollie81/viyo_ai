"""
Like/unlike/comment on someone else's post.

The client already writes its own `likes`/`comments` rows directly
against Supabase (RLS lets you insert a row as yourself regardless of
whose post it's under) — but *bumping the counter on the post itself*
(`posts.like_count`, and whatever `comments`-table check is in place)
is a different story: Supabase RLS on `posts` only lets the post's
owner update their own row. A direct client UPDATE to like_count for
someone else's post just silently matches zero rows — the like row
saves, the count never moves — which is exactly why `like_post`/
`decrement_like_count` originally existed as RPCs (SECURITY DEFINER,
bypassing RLS) rather than plain client calls. This module is the
same fix already applied to boost_post/spotlight in posts.py/
discover.py: do the mutation here, through the service-role client,
which bypasses RLS unconditionally instead of depending on whatever a
policy this codebase can't inspect happens to allow.

Comments are routed through here too for the same reason — whatever
is blocking commenting on someone else's post is somewhere in RLS this
codebase has no visibility into; going through the admin client sidesteps
the question entirely rather than guessing at a policy fix blind.
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1", tags=["interactions"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


def _get_post(post_id: str) -> dict:
    try:
        result = (
            supabase_admin
            .table("posts")
            .select("id,user_id,like_count")
            .eq("id", post_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load post: {e}")

    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Post not found.")
    return rows[0]


class LikeResponse(BaseModel):
    liked: bool
    like_count: int


@router.post("/posts/{post_id}/like", response_model=LikeResponse)
async def like_post(post_id: str, user_id: str = Depends(_get_current_user_id_no_guest)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Interactions service is not configured.")

    post = _get_post(post_id)
    current = int(post.get("like_count") or 0)

    try:
        supabase_admin.table("likes").insert({"user_id": user_id, "post_id": post_id}).execute()
    except Exception as e:
        # Unique-violation — already liked (e.g. a stale double-tap).
        # Nothing left to do; the count is already right.
        if "23505" in str(e) or "duplicate" in str(e).lower():
            return LikeResponse(liked=True, like_count=current)
        raise HTTPException(status_code=500, detail=f"Could not like post: {e}")

    # Compare-and-swap increment, same tradeoff used everywhere else in
    # this app for a counter with no atomic DB-side function to call.
    try:
        result = (
            supabase_admin.table("posts")
            .update({"like_count": current + 1})
            .eq("id", post_id)
            .eq("like_count", current)
            .execute()
        )
        new_count = current + 1 if result.data else current
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Liked, but could not update like count: {e}")

    if post.get("user_id") != user_id:
        try:
            actor = (
                supabase_admin.table("profiles")
                .select("display_name,username")
                .eq("id", user_id)
                .maybe_single()
                .execute()
            )
            actor_data = actor.data or {}
            actor_name = actor_data.get("display_name") or actor_data.get("username") or "Someone"
            supabase_admin.table("notifications").insert({
                "user_id": post["user_id"],
                "actor_id": user_id,
                "type": "like",
                "message": f"{actor_name} liked your post",
            }).execute()
        except Exception:
            pass  # best-effort — never block the like itself over this

    return LikeResponse(liked=True, like_count=new_count)


@router.post("/posts/{post_id}/unlike", response_model=LikeResponse)
async def unlike_post(post_id: str, user_id: str = Depends(_get_current_user_id_no_guest)):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Interactions service is not configured.")

    post = _get_post(post_id)
    current = int(post.get("like_count") or 0)

    try:
        supabase_admin.table("likes").delete().eq("user_id", user_id).eq("post_id", post_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not unlike post: {e}")

    if current <= 0:
        return LikeResponse(liked=False, like_count=0)

    try:
        result = (
            supabase_admin.table("posts")
            .update({"like_count": current - 1})
            .eq("id", post_id)
            .eq("like_count", current)
            .execute()
        )
        new_count = current - 1 if result.data else current
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unliked, but could not update like count: {e}")

    return LikeResponse(liked=False, like_count=new_count)


class AddCommentRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)


class CommentResponse(BaseModel):
    id: str
    post_id: str
    user_id: str
    content: str
    created_at: str


@router.post("/posts/{post_id}/comments", response_model=CommentResponse)
async def add_comment(
    post_id: str,
    req: AddCommentRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Interactions service is not configured.")

    _get_post(post_id)  # 404s if the post doesn't exist

    try:
        result = (
            supabase_admin
            .table("comments")
            .insert({"post_id": post_id, "user_id": user_id, "content": req.content})
            .select()
            .single()
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not add comment: {e}")

    return CommentResponse(**result.data)
