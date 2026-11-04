"""
One place to tell a user something: an in-app notification row (the
Notifications screen) plus a phone push when Firebase is configured.

Always best-effort: a notification that can't be saved or pushed never
fails the action that caused it (a like, an unlock, a finished job).
"""
from __future__ import annotations

import os
from typing import Optional

_client = None


def _admin():
    global _client
    if _client is None:
        url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not (url and key):
            return None
        from supabase import create_client
        _client = create_client(url, key)
    return _client


def display_name(user_id: Optional[str], admin=None) -> str:
    admin = admin or _admin()
    if not user_id or admin is None:
        return "Someone"
    try:
        row = (admin.table("profiles").select("display_name,username").eq("id", user_id)
               .limit(1).execute()).data or [{}]
        return row[0].get("display_name") or row[0].get("username") or "Someone"
    except Exception:
        return "Someone"


def notify(user_id: str, notif_type: str, message: str, *, actor_id: Optional[str] = None,
           post_id: Optional[str] = None, entity_id: Optional[str] = None,
           push_title: Optional[str] = None, admin=None) -> bool:
    """Saves the notification and pushes it. True if the row was saved."""
    if not user_id or (actor_id and actor_id == user_id):
        return False
    admin = admin or _admin()
    if admin is None:
        return False
    row = {"user_id": user_id, "type": notif_type, "message": message[:300]}
    if actor_id:
        row["actor_id"] = actor_id
    if post_id:
        row["post_id"] = post_id
    if entity_id:
        row["entity_id"] = entity_id
    saved = False
    try:
        admin.table("notifications").insert(row).execute()
        saved = True
    except Exception as e:
        print(f"[WARN] notification '{notif_type}' for {user_id} not saved: {e}")
    if push_title:
        try:
            from push import send_push_to_user
            data = {"type": notif_type}
            for k, v in (("post_id", post_id), ("entity_id", entity_id), ("actor_id", actor_id)):
                if v:
                    data[k] = str(v)
            send_push_to_user(admin, user_id, push_title, message, data)
        except Exception as e:
            print(f"[WARN] push '{notif_type}' for {user_id} not sent: {e}")
    return saved
