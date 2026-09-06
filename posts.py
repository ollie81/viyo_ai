"""
Post-level actions that spend Viyo Coins.

Boosting already existed as a Postgres RPC (`boost_post`) called
directly from Flutter, but that RPC takes a client-supplied cost with
no way to verify from this codebase whether it validates that amount
server-side before deducting it — a client-controlled coin cost is
exactly the kind of thing worth not trusting blindly. This routes the
same action through coins.py's spend_on_feature instead, which is
already the one audited, server-controlled place coin costs are
enforced everywhere else in this app.
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client

from coins import spend_on_feature, FEATURE_COSTS

router = APIRouter(prefix="/api/v1", tags=["posts"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


class BoostPostRequest(BaseModel):
    post_id: str = Field(..., min_length=1)


class BoostPostResponse(BaseModel):
    post_id: str
    is_boosted: bool
    cost: int


@router.post("/boost-post", response_model=BoostPostResponse)
async def boost_post(
    req: BoostPostRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    """
    Marks the requester's own post as boosted, after spending coins.
    Boost's actual effect (a ranking multiplier in the home feed,
    naturally fading as the post's own age/engagement decays — see
    PostService._hotScore on the Flutter side) needs no expiry
    timestamp here: there's no `boosted_at` column, and none is needed
    since the feed's own age-decay already keeps an old boosted post
    from dominating forever.
    """
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Boost service is not configured.")

    try:
        result = (
            supabase_admin
            .table("posts")
            .select("id,user_id,is_boosted")
            .eq("id", req.post_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load post: {e}")

    if not rows:
        raise HTTPException(status_code=404, detail="Post not found.")
    post = rows[0]
    if post.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="You can only boost your own posts.")
    if post.get("is_boosted"):
        raise HTTPException(status_code=400, detail="This post is already boosted.")

    spend_on_feature(supabase_admin, user_id, "boost_post")

    try:
        supabase_admin.table("posts").update({"is_boosted": True}).eq("id", req.post_id).execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Coins were spent but the post could not be marked boosted: {e}",
        )

    return BoostPostResponse(
        post_id=req.post_id,
        is_boosted=True,
        cost=FEATURE_COSTS["boost_post"]["cost"],
    )
