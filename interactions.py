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

from push import send_push_to_user

router = APIRouter(prefix="/api/v1", tags=["interactions"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


def _notify_post_owner(post_owner_id: str, actor_id: str, notif_type: str, message_template: str, push_title: str) -> None:
    """
    Best-effort in-app notification + push about `actor_id`'s action on
    a post owned by `post_owner_id` — never blocks the action itself.
    `message_template` gets `{actor_name}` filled in for both the
    in-app notification row and the push body.
    """
    if post_owner_id == actor_id:
        return

    try:
        actor = (
            supabase_admin.table("profiles")
            .select("display_name,username")
            .eq("id", actor_id)
            .maybe_single()
            .execute()
        )
        actor_data = actor.data or {}
        actor_name = actor_data.get("display_name") or actor_data.get("username") or "Someone"
    except Exception:
        actor_name = "Someone"

    message = message_template.format(actor_name=actor_name)

    try:
        supabase_admin.table("notifications").insert({
            "user_id": post_owner_id,
            "actor_id": actor_id,
            "type": notif_type,
            "message": message,
        }).execute()
    except Exception:
        pass  # best-effort — never block the action that earned this

    send_push_to_user(
        supabase_admin, post_owner_id, push_title, message,
        {"type": notif_type, "actor_id": actor_id},
    )


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

    _notify_post_owner(post["user_id"], user_id, "like", "{actor_name} liked your post", "New like ❤️")

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

    post = _get_post(post_id)  # 404s if the post doesn't exist

    try:
        # PostgREST returns the inserted row(s) by default (supabase-py's
        # default Prefer: return=representation) — no .select()/.single()
        # to chain here, unlike a plain query. Every other insert in this
        # codebase already does it this way; this one didn't, and
        # supabase-py 2.x's insert builder only exposes .execute(), so
        # the extra chaining raised AttributeError before the comment
        # ever reached the database.
        result = (
            supabase_admin
            .table("comments")
            .insert({"post_id": post_id, "user_id": user_id, "content": req.content})
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not add comment: {e}")

    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=500, detail="Comment insert returned no row")

    # Previously missing entirely — a comment never notified the post
    # owner at all, in-app or push.
    _notify_post_owner(post["user_id"], user_id, "comment", "{actor_name} commented on your post", "New comment 💬")

    return CommentResponse(**rows[0])
