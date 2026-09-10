"""
Push notifications via Firebase Cloud Messaging.

Device tokens live in a new `device_tokens` table — each device
registers/removes only its own token directly against Supabase (RLS:
auth.uid() = user_id), the same self-scoped-write pattern as `likes`
or `blocks`. Actually *sending* a push means reading a token that
belongs to someone else (whoever should receive the notification), so
that has to happen here, through the service-role client, the same
reasoning as every other cross-user action in this app.

FCM itself needs a Firebase service account (FIREBASE_SERVICE_ACCOUNT_JSON,
the whole JSON key as one env var) — unset means send_push_to_user is a
silent no-op, same "off until configured" posture as Stripe/Sentry.
"""
import json
import os
from typing import Optional

import firebase_admin
from firebase_admin import credentials, messaging
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1", tags=["push"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
FIREBASE_SERVICE_ACCOUNT_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

_firebase_app = None
if FIREBASE_SERVICE_ACCOUNT_JSON:
    try:
        _cred = credentials.Certificate(json.loads(FIREBASE_SERVICE_ACCOUNT_JSON))
        _firebase_app = firebase_admin.initialize_app(_cred)
    except Exception as e:
        print(f"[WARN] Firebase push not configured: {e}")


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


def send_push_to_user(admin, user_id: str, title: str, body: str, data: Optional[dict] = None) -> None:
    """
    Best-effort — a push failing (Firebase not configured, no device
    on file, a stale/uninstalled token) must never fail the action
    that triggered it. By the time this runs, whatever earned the
    notification (a like, a comment, a gift) already succeeded.
    """
    if _firebase_app is None or admin is None:
        return

    try:
        result = admin.table("device_tokens").select("token").eq("user_id", user_id).execute()
        tokens = [r["token"] for r in (result.data or [])]
    except Exception:
        return
    if not tokens:
        return

    str_data = {k: str(v) for k, v in (data or {}).items()}
    for token in tokens:
        try:
            messaging.send(
                messaging.Message(
                    notification=messaging.Notification(title=title, body=body),
                    data=str_data,
                    token=token,
                ),
                app=_firebase_app,
            )
        except messaging.UnregisteredError:
            # The app was uninstalled or the token rotated — stop trying it.
            try:
                admin.table("device_tokens").delete().eq("token", token).execute()
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] Push to {user_id} failed: {e}")


class NotifyFollowRequest(BaseModel):
    followed_id: str = Field(..., min_length=1)


@router.post("/notify/follow")
async def notify_follow(
    req: NotifyFollowRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    """
    The in-app notification row for a follow is already created by the
    existing `follow_user` RPC (see the 'follow' icon case already
    handled in NotificationsScreen) — unlike likes/comments/gifts, a
    `follows` insert never needs to touch a row someone else owns, so
    there's no evidence that RPC has the same RLS-blocked-mutation bug
    found elsewhere. This endpoint only adds the push half, called by
    the client right after that RPC succeeds. Best-effort by design —
    a push that doesn't go out should never surface as a failed follow.
    """
    if supabase_admin is None:
        return {"sent": False}

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
    except Exception:
        actor_name = "Someone"

    send_push_to_user(
        supabase_admin, req.followed_id, "New follower",
        f"{actor_name} started following you",
        {"type": "follow", "user_id": user_id},
    )
    return {"sent": True}
