"""
Direct messaging — one-to-one conversations between users.

Every read and write here goes through the service-role client, never
a client-side Supabase query. That's a deliberate departure from how
`likes`/`follows`/`blocks` work (plain RLS-scoped client inserts) and
is instead the same pattern interactions.py already uses for
likes/comments: this app was bitten once by an RLS policy this
codebase couldn't inspect silently blocking a legitimate cross-user
write, and a DM inbox is exactly the kind of feature where "message
sent, recipient can't see it" would be a bad way to find that out
again. See messaging_tables.sql — `conversations`/`messages` are
RLS-enabled with zero policies, so only this backend can touch them.
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client

from push import send_push_to_user

router = APIRouter(prefix="/api/v1", tags=["messaging"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


def _require_admin_client() -> None:
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Messaging service is not configured.")


def _is_blocked(user_id: str, other_id: str) -> bool:
    """True if either has blocked the other, in either direction."""
    try:
        result = (
            supabase_admin
            .table("blocks")
            .select("blocker_id,blocked_id")
            .or_(
                f"and(blocker_id.eq.{user_id},blocked_id.eq.{other_id}),"
                f"and(blocker_id.eq.{other_id},blocked_id.eq.{user_id})"
            )
            .limit(1)
            .execute()
        )
        return bool(result.data)
    except Exception:
        # Fails closed on the side of NOT sending rather than silently
        # ignoring a block this couldn't verify.
        raise HTTPException(status_code=500, detail="Could not verify block status.")


def _profile_lookup(user_ids: list) -> dict:
    if not user_ids:
        return {}
    try:
        result = (
            supabase_admin.table("profiles")
            .select("id,username,display_name,avatar_url")
            .in_("id", list(set(user_ids)))
            .execute()
        )
        return {p["id"]: p for p in (result.data or [])}
    except Exception:
        return {}


def _get_conversation_or_404(conversation_id: str, user_id: str) -> dict:
    try:
        result = (
            supabase_admin
            .table("conversations")
            .select("id,user_a,user_b,last_message_at")
            .eq("id", conversation_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load conversation: {e}")

    if not rows or user_id not in (rows[0]["user_a"], rows[0]["user_b"]):
        # Same 404 whether it doesn't exist or the caller isn't a
        # participant — a 403 would confirm a conversation id exists
        # to someone who isn't part of it.
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return rows[0]


def _other_participant(conversation: dict, user_id: str) -> str:
    return conversation["user_b"] if conversation["user_a"] == user_id else conversation["user_a"]


# ---------------------------------------------------------
# Start (or resume) a conversation
# ---------------------------------------------------------

class StartConversationRequest(BaseModel):
    other_user_id: str = Field(..., min_length=1)


class ConversationSummary(BaseModel):
    id: str
    other_user_id: str
    other_username: Optional[str] = None
    other_display_name: Optional[str] = None
    other_avatar_url: Optional[str] = None
    last_message_at: str
    last_message_preview: Optional[str] = None
    last_message_is_mine: bool = False
    unread_count: int = 0


@router.post("/conversations", response_model=ConversationSummary)
async def start_conversation(
    req: StartConversationRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    _require_admin_client()

    if req.other_user_id == user_id:
        raise HTTPException(status_code=400, detail="Can't start a conversation with yourself.")

    if not _profile_lookup([req.other_user_id]):
        raise HTTPException(status_code=404, detail="User not found.")

    if _is_blocked(user_id, req.other_user_id):
        raise HTTPException(status_code=403, detail="You can't message this user.")

    # Look for an existing conversation in either direction before
    # creating one — (a, b) and (b, a) are the same conversation.
    try:
        existing = (
            supabase_admin
            .table("conversations")
            .select("id,user_a,user_b,last_message_at")
            .or_(
                f"and(user_a.eq.{user_id},user_b.eq.{req.other_user_id}),"
                f"and(user_a.eq.{req.other_user_id},user_b.eq.{user_id})"
            )
            .limit(1)
            .execute()
        )
        rows = existing.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not look up conversation: {e}")

    if rows:
        conversation = rows[0]
    else:
        try:
            created = (
                supabase_admin
                .table("conversations")
                .insert({"user_a": user_id, "user_b": req.other_user_id})
                .execute()
            )
            conversation = (created.data or [None])[0]
        except Exception as e:
            # A concurrent request creating the same pair at the same
            # moment would hit the unique index — treat that as "someone
            # else just created it, go find it" rather than a hard
            # failure, since the caller just wants a working conversation.
            if "23505" in str(e) or "duplicate" in str(e).lower():
                retry = (
                    supabase_admin
                    .table("conversations")
                    .select("id,user_a,user_b,last_message_at")
                    .or_(
                        f"and(user_a.eq.{user_id},user_b.eq.{req.other_user_id}),"
                        f"and(user_a.eq.{req.other_user_id},user_b.eq.{user_id})"
                    )
                    .limit(1)
                    .execute()
                )
                rows = retry.data or []
                if not rows:
                    raise HTTPException(status_code=500, detail="Could not create conversation.")
                conversation = rows[0]
            else:
                raise HTTPException(status_code=500, detail=f"Could not create conversation: {e}")

    other = _profile_lookup([req.other_user_id]).get(req.other_user_id, {})
    return ConversationSummary(
        id=conversation["id"],
        other_user_id=req.other_user_id,
        other_username=other.get("username"),
        other_display_name=other.get("display_name"),
        other_avatar_url=other.get("avatar_url"),
        last_message_at=conversation["last_message_at"],
    )


# ---------------------------------------------------------
# List conversations
# ---------------------------------------------------------

class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(user_id: str = Depends(_get_current_user_id_no_guest)):
    _require_admin_client()

    try:
        result = (
            supabase_admin
            .table("conversations")
            .select("id,user_a,user_b,last_message_at")
            .or_(f"user_a.eq.{user_id},user_b.eq.{user_id}")
            .order("last_message_at", desc=True)
            .limit(100)
            .execute()
        )
        conversations = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list conversations: {e}")

    if not conversations:
        return ConversationListResponse(conversations=[])

    conversation_ids = [c["id"] for c in conversations]
    other_ids = [_other_participant(c, user_id) for c in conversations]
    profiles = _profile_lookup(other_ids)

    try:
        recent_messages = (
            supabase_admin
            .table("messages")
            .select("conversation_id,sender_id,content,created_at,read_at")
            .in_("conversation_id", conversation_ids)
            .order("created_at", desc=True)
            .limit(1000)  # generous cap — enough recent history across every conversation to summarize
            .execute()
        )
        all_messages = recent_messages.data or []
    except Exception:
        all_messages = []

    last_by_conversation = {}
    unread_by_conversation = {}
    for m in all_messages:
        cid = m["conversation_id"]
        if cid not in last_by_conversation:
            last_by_conversation[cid] = m
        if m["sender_id"] != user_id and m.get("read_at") is None:
            unread_by_conversation[cid] = unread_by_conversation.get(cid, 0) + 1

    summaries = []
    for c in conversations:
        other_id = _other_participant(c, user_id)
        other = profiles.get(other_id, {})
        last = last_by_conversation.get(c["id"])
        summaries.append(ConversationSummary(
            id=c["id"],
            other_user_id=other_id,
            other_username=other.get("username"),
            other_display_name=other.get("display_name"),
            other_avatar_url=other.get("avatar_url"),
            last_message_at=c["last_message_at"],
            last_message_preview=last["content"] if last else None,
            last_message_is_mine=bool(last and last["sender_id"] == user_id),
            unread_count=unread_by_conversation.get(c["id"], 0),
        ))

    return ConversationListResponse(conversations=summaries)


# ---------------------------------------------------------
# Messages within a conversation
# ---------------------------------------------------------

class MessageItem(BaseModel):
    id: str
    conversation_id: str
    sender_id: str
    content: str
    created_at: str
    read_at: Optional[str] = None
    is_mine: bool


class MessageListResponse(BaseModel):
    messages: list[MessageItem]


@router.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def list_messages(
    conversation_id: str,
    before: Optional[str] = None,
    limit: int = 50,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    _require_admin_client()
    _get_conversation_or_404(conversation_id, user_id)

    try:
        query = (
            supabase_admin
            .table("messages")
            .select("id,conversation_id,sender_id,content,created_at,read_at")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=True)
            .limit(min(limit, 100))
        )
        if before:
            query = query.lt("created_at", before)
        result = query.execute()
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load messages: {e}")

    # Returned oldest-first within the page — a chat screen appends
    # older pages above what's already rendered, which only works if
    # each page is already in reading order.
    rows.reverse()

    return MessageListResponse(messages=[
        MessageItem(
            id=r["id"], conversation_id=r["conversation_id"], sender_id=r["sender_id"],
            content=r["content"], created_at=r["created_at"], read_at=r.get("read_at"),
            is_mine=(r["sender_id"] == user_id),
        )
        for r in rows
    ])


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)


@router.post("/conversations/{conversation_id}/messages", response_model=MessageItem)
async def send_message(
    conversation_id: str,
    req: SendMessageRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    _require_admin_client()
    conversation = _get_conversation_or_404(conversation_id, user_id)
    other_id = _other_participant(conversation, user_id)

    if _is_blocked(user_id, other_id):
        raise HTTPException(status_code=403, detail="You can't message this user.")

    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message can't be empty.")

    try:
        inserted = (
            supabase_admin
            .table("messages")
            .insert({"conversation_id": conversation_id, "sender_id": user_id, "content": content})
            .execute()
        )
        rows = inserted.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not send message: {e}")

    if not rows:
        raise HTTPException(status_code=500, detail="Message insert returned no row.")

    row = rows[0]

    try:
        supabase_admin.table("conversations").update(
            {"last_message_at": row["created_at"]}
        ).eq("id", conversation_id).execute()
    except Exception:
        pass  # the message itself sent — a stale sort position is cosmetic, not worth failing over

    sender = _profile_lookup([user_id]).get(user_id, {})
    sender_name = sender.get("display_name") or sender.get("username") or "Someone"
    preview = content if len(content) <= 80 else content[:77] + "..."
    send_push_to_user(
        supabase_admin, other_id, f"New message from {sender_name}", preview,
        {"type": "message", "conversation_id": conversation_id, "sender_id": user_id},
    )

    return MessageItem(
        id=row["id"], conversation_id=conversation_id, sender_id=user_id,
        content=content, created_at=row["created_at"], read_at=None, is_mine=True,
    )


@router.post("/conversations/{conversation_id}/read")
async def mark_read(
    conversation_id: str,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    _require_admin_client()
    _get_conversation_or_404(conversation_id, user_id)

    try:
        import datetime
        supabase_admin.table("messages").update(
            {"read_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        ).eq("conversation_id", conversation_id).neq("sender_id", user_id).is_("read_at", "null").execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not mark messages read: {e}")

    return {"ok": True}
