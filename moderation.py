"""
Admin moderation review — the other half of the reporting feature.

Reporting a post/user already writes a row to `reports` (see
Viyo-/lib/services/moderation_service.dart and moderation_tables.sql).
That table's own RLS deliberately blocks every read back through the
app — nobody, reporter included, can list reports via a normal
Supabase query, so a reported user can never learn who reported them.

But that also means there was no way to act on a report short of
opening the Supabase dashboard and reading raw rows by hand. This is
that missing half: list open reports with real context (what was
reported, not just an opaque UUID) and take an actual action — dismiss,
remove the reported post, or ban the reported user — from one place.

Gated by the same shared X-Admin-Key header as analytics.py, for the
same reason: there's no per-user admin role anywhere else in this app,
and a solo operator doesn't need one built just for this.
"""
import os
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1/admin", tags=["moderation"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# A permanent-enough ban. Supabase's admin API takes a duration, not a
# boolean — there's no "banned forever" value, so this is the
# convention (100 years) most Supabase deployments use to mean it.
_BAN_DURATION = "876000h"


def _require_admin(x_admin_key: str = Header(None)) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Moderation endpoint is not configured.")
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


class ReportContext(BaseModel):
    """
    What's actually being reported, resolved from target_id — a report
    row alone is a reporter id and an opaque target id, which isn't
    enough to review anything without this.
    """
    exists: bool = True
    caption: Optional[str] = None       # target_type == "post"
    media_url: Optional[str] = None     # target_type == "post"
    username: Optional[str] = None      # the post's author, or the reported user
    display_name: Optional[str] = None
    owner_id: Optional[str] = None      # target_type == "post" — who to ban, if it comes to that


class ReportItem(BaseModel):
    id: str
    reporter_id: str
    reporter_username: Optional[str] = None
    target_type: str
    target_id: str
    reason: str
    details: Optional[str] = None
    status: str
    created_at: str
    context: ReportContext


class ReportListResponse(BaseModel):
    reports: list[ReportItem]


def _profile_lookup(user_ids: list[str]) -> dict:
    if not user_ids:
        return {}
    try:
        result = (
            supabase_admin.table("profiles")
            .select("id,username,display_name")
            .in_("id", list(set(user_ids)))
            .execute()
        )
        return {p["id"]: p for p in (result.data or [])}
    except Exception:
        return {}


def _resolve_context(target_type: str, target_id: str) -> ReportContext:
    if target_type == "post":
        try:
            result = (
                supabase_admin
                .table("posts")
                .select("caption,media_url,user_id")
                .eq("id", target_id)
                .limit(1)
                .execute()
            )
            rows = result.data or []
        except Exception:
            rows = []

        if not rows:
            return ReportContext(exists=False)

        post = rows[0]
        author = _profile_lookup([post.get("user_id")]).get(post.get("user_id"), {})
        return ReportContext(
            caption=post.get("caption"),
            media_url=post.get("media_url"),
            owner_id=post.get("user_id"),
            username=author.get("username"),
            display_name=author.get("display_name"),
        )

    if target_type == "user":
        profile = _profile_lookup([target_id]).get(target_id)
        if not profile:
            return ReportContext(exists=False)
        return ReportContext(
            owner_id=target_id,
            username=profile.get("username"),
            display_name=profile.get("display_name"),
        )

    return ReportContext(exists=False)


@router.get("/reports", response_model=ReportListResponse)
async def list_reports(
    status: str = "pending",
    limit: int = 50,
    x_admin_key: str = Header(None),
):
    _require_admin(x_admin_key)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Moderation service is not configured.")

    try:
        result = (
            supabase_admin
            .table("reports")
            .select("id,reporter_id,target_type,target_id,reason,details,status,created_at")
            .eq("status", status)
            .order("created_at", desc=False)  # oldest first — first reported, first reviewed
            .limit(min(limit, 200))
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load reports: {e}")

    reporters = _profile_lookup([r["reporter_id"] for r in rows])

    items = []
    for r in rows:
        reporter = reporters.get(r["reporter_id"], {})
        items.append(ReportItem(
            id=r["id"],
            reporter_id=r["reporter_id"],
            reporter_username=reporter.get("username"),
            target_type=r["target_type"],
            target_id=r["target_id"],
            reason=r["reason"],
            details=r.get("details"),
            status=r["status"],
            created_at=r["created_at"],
            context=_resolve_context(r["target_type"], r["target_id"]),
        ))

    return ReportListResponse(reports=items)


class ResolveReportRequest(BaseModel):
    action: Literal["dismiss", "remove_post", "ban_user"]


class ResolveReportResponse(BaseModel):
    report_id: str
    status: str
    action_taken: str
    warning: Optional[str] = None


def _delete_post(post_id: str) -> None:
    """
    Admin-side post removal — deletes the media from Storage first (best
    effort; a missing/already-gone file shouldn't block removing the
    row), then the row itself. Separate from PostService.deletePost in
    the Flutter app, which is scoped to the post's own owner acting
    through their own session; this runs as the service role so it
    works on ANY post, which is the whole point of a moderation action.
    """
    try:
        result = supabase_admin.table("posts").select("media_url,thumbnail_url").eq("id", post_id).limit(1).execute()
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not look up post: {e}")

    if not rows:
        raise HTTPException(status_code=404, detail="Post not found (already deleted?).")

    post = rows[0]
    for url in (post.get("media_url"), post.get("thumbnail_url")):
        if not url:
            continue
        try:
            marker = "posts-media/"
            idx = url.find(marker)
            if idx != -1:
                supabase_admin.storage.from_("posts-media").remove([url[idx + len(marker):]])
        except Exception:
            pass  # best-effort — a stray file left in storage is not worth failing the removal over

    try:
        supabase_admin.table("posts").delete().eq("id", post_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete post: {e}")


def _ban_user(user_id: str) -> None:
    try:
        supabase_admin.auth.admin.update_user_by_id(user_id, {"ban_duration": _BAN_DURATION})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not ban user: {e}")


@router.post("/reports/{report_id}/resolve", response_model=ResolveReportResponse)
async def resolve_report(
    report_id: str,
    req: ResolveReportRequest,
    x_admin_key: str = Header(None),
):
    _require_admin(x_admin_key)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Moderation service is not configured.")

    try:
        result = (
            supabase_admin
            .table("reports")
            .select("id,target_type,target_id")
            .eq("id", report_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load report: {e}")

    if not rows:
        raise HTTPException(status_code=404, detail="Report not found.")

    report = rows[0]
    warning = None

    if req.action == "remove_post":
        if report["target_type"] != "post":
            raise HTTPException(
                status_code=400,
                detail=f"Can't remove a post — this report is about a {report['target_type']}.",
            )
        _delete_post(report["target_id"])

    elif req.action == "ban_user":
        # A user report already names the account. A post report names
        # the post — resolve to its author rather than require the
        # caller to look that up separately first.
        if report["target_type"] == "user":
            user_id = report["target_id"]
        else:
            context = _resolve_context("post", report["target_id"])
            if not context.owner_id:
                raise HTTPException(
                    status_code=404,
                    detail="Could not resolve this post to an owner to ban (post may already be deleted).",
                )
            user_id = context.owner_id
        _ban_user(user_id)

    new_status = "dismissed" if req.action == "dismiss" else "reviewed"

    try:
        supabase_admin.table("reports").update({"status": new_status}).eq("id", report_id).execute()
    except Exception as e:
        # The action itself (post removed / user banned) already
        # happened and is real — losing the status update afterward is
        # a cosmetic problem, not a reason to make the caller think the
        # action failed and possibly retry a ban/deletion that already
        # went through.
        warning = f"Action succeeded but report status could not be updated: {e}"

    return ResolveReportResponse(
        report_id=report_id,
        status=new_status,
        action_taken=req.action,
        warning=warning,
    )
