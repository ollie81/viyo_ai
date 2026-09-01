import datetime
import json
import os
import re
import time
from collections import defaultdict, deque
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
# Personalized caption/title variants
#
# The existing /content-ideas endpoint (main.py) suggests WHAT to post
# — this suggests HOW to caption something the creator already has in
# mind, grounded in what has actually gotten this specific creator
# above-their-own-average engagement before, using the same
# "likes + comments" signal as the outcome follow-up above. A creator
# with no post history yet just gets a solid generic set of variants
# instead of a personalization step that has nothing to work from.
# ---------------------------------------------------------

_CAPTION_HISTORY_LOOKBACK = 30
_MIN_CAPTIONS_FOR_PERSONALIZATION = 3
_MAX_EXAMPLE_CAPTIONS = 5

_CAPTION_VARIANTS_RATE_LIMIT = 20
_CAPTION_VARIANTS_RATE_WINDOW = 60 * 60 * 24  # per day
_caption_variants_requests: dict = defaultdict(deque)


def _check_caption_variants_rate_limit(user_id: str):
    now = time.time()
    q = _caption_variants_requests[user_id]
    while q and now - q[0] > _CAPTION_VARIANTS_RATE_WINDOW:
        q.popleft()
    if len(q) >= _CAPTION_VARIANTS_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Limit reached: {_CAPTION_VARIANTS_RATE_LIMIT} caption generations per day. Try again tomorrow.",
        )
    q.append(now)


class CaptionVariantsRequest(BaseModel):
    draft: str = Field(..., min_length=1, max_length=500)
    niche: str = Field(default="", max_length=100)


class CaptionVariantsResponse(BaseModel):
    variants: list[str]
    # False when the creator doesn't have enough post history yet for
    # personalization to mean anything — lets the UI say so honestly
    # instead of implying every set of variants is tailored to them.
    personalized: bool


def _get_top_performing_captions(user_id: str) -> list[str]:
    """
    Pulls captions from this creator's own past posts that performed
    above their own average engagement, so new variants can be grounded
    in a voice that has actually worked for THEM rather than a generic
    tone. Never raises — any failure or lack of history just means the
    caller falls back to non-personalized generation.
    """
    if supabase_admin is None:
        return []

    try:
        result = (
            supabase_admin
            .table("posts")
            .select("caption,like_count,comment_count")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(_CAPTION_HISTORY_LOOKBACK)
            .execute()
        )
        posts = result.data or []
    except Exception:
        return []

    scored = [
        (p, (p.get("like_count") or 0) + (p.get("comment_count") or 0))
        for p in posts
        if (p.get("caption") or "").strip()
    ]
    if len(scored) < _MIN_CAPTIONS_FOR_PERSONALIZATION:
        return []

    baseline = sum(engagement for _, engagement in scored) / len(scored)
    above_baseline = [p for p, engagement in scored if engagement > baseline]
    if len(above_baseline) < _MIN_CAPTIONS_FOR_PERSONALIZATION:
        # Too few posts clearly beat their own baseline to treat that as
        # a real signal — fall back to plain top-performers-by-engagement
        # instead of forcing a comparison that isn't meaningful yet.
        above_baseline = [p for p, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

    return [p["caption"].strip() for p in above_baseline[:_MAX_EXAMPLE_CAPTIONS]]


def _parse_variant_list(raw: str) -> list[str]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(v).strip() for v in parsed if str(v).strip()][:6]


@router.post("/caption-variants", response_model=CaptionVariantsResponse)
async def caption_variants(
    req: CaptionVariantsRequest,
    user_id: str = Depends(_get_current_user_id),
):
    _check_caption_variants_rate_limit(user_id)

    top_captions = _get_top_performing_captions(user_id)
    personalized = bool(top_captions)

    niche_context = f" in the {req.niche} niche" if req.niche else ""

    if personalized:
        examples = "\n".join(f'- "{c}"' for c in top_captions)
        instruction = (
            f"A content creator{niche_context} has this rough idea for their next post:\n"
            f'"{req.draft}"\n\n'
            "Here are captions from THIS creator's own past posts that performed above "
            "their usual engagement — study their voice, tone, length, emoji/hashtag use:\n"
            f"{examples}\n\n"
            "Write 4 new caption/title variants for the idea above that sound like they "
            "came from this same creator — borrow whatever made those captions work, don't "
            'copy them verbatim. Respond with ONLY a JSON array of 4 strings, like '
            '["variant one", "variant two", "variant three", "variant four"] — no other text.'
        )
    else:
        instruction = (
            f"A content creator{niche_context} has this rough idea for their next post:\n"
            f'"{req.draft}"\n\n'
            "Write 4 punchy, scroll-stopping caption/title variants for a short-form video "
            'post. Respond with ONLY a JSON array of 4 strings, like '
            '["variant one", "variant two", "variant three", "variant four"] — no other text.'
        )

    try:
        completion = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": instruction}],
            temperature=0.9,
            max_tokens=400,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception:
        raise HTTPException(status_code=502, detail="AI service unavailable")

    variants = _parse_variant_list(raw)
    if not variants:
        raise HTTPException(status_code=502, detail="AI returned an unexpected format")

    return CaptionVariantsResponse(variants=variants, personalized=personalized)


# ---------------------------------------------------------
# Weekly report card
#
# Coach feedback and the outcome follow-up above both live at the
# per-post level — nothing previously rolled them up into "how is this
# week going overall," which is the shape creators actually think in
# ("am I improving?", not "how did Tuesday's post do?"). This computes
# real stats first (never invented by the model) and only uses
# GPT-4o-mini to phrase them as a short, encouraging note — the same
# coaching voice as the rest of the app, not a numbers dashboard.
# ---------------------------------------------------------

_REPORT_WINDOW_DAYS = 7
_REPORT_LOOKBACK_ROWS = 200  # generous cap so two weeks of an active creator's data always fits


class WeeklyReportResponse(BaseModel):
    posts_this_week: int
    avg_score_this_week: Optional[float] = None
    avg_score_last_week: Optional[float] = None
    # 'up' / 'down' / 'flat' — only set when both weeks have a score to
    # compare; None means there isn't enough history for a trend yet.
    score_trend: Optional[str] = None
    total_likes: int
    total_comments: int
    best_post_caption: Optional[str] = None
    summary: str


def _week_boundaries() -> tuple[datetime.datetime, datetime.datetime, datetime.datetime]:
    now = datetime.datetime.now(datetime.timezone.utc)
    this_week_start = now - datetime.timedelta(days=_REPORT_WINDOW_DAYS)
    last_week_start = now - datetime.timedelta(days=_REPORT_WINDOW_DAYS * 2)
    return last_week_start, this_week_start, now


@router.get("/weekly-report", response_model=WeeklyReportResponse)
async def weekly_report(
    user_id: str = Depends(_get_current_user_id),
):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Report service is not configured.")

    last_week_start, this_week_start, now = _week_boundaries()

    try:
        scored_result = (
            supabase_admin
            .table("video_coach_messages")
            .select("score,created_at")
            .eq("user_id", user_id)
            .not_.is_("score", "null")
            .order("created_at", desc=True)
            .limit(_REPORT_LOOKBACK_ROWS)
            .execute()
        )
        scored_messages = scored_result.data or []

        posts_result = (
            supabase_admin
            .table("posts")
            .select("caption,like_count,comment_count,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(_REPORT_LOOKBACK_ROWS)
            .execute()
        )
        posts = posts_result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load report data: {e}")

    def _in_range(created_at: str, start: datetime.datetime, end: datetime.datetime) -> bool:
        ts = _parse_timestamp(created_at or "")
        return ts is not None and start <= ts < end

    scores_this_week = [
        m["score"] for m in scored_messages if _in_range(m.get("created_at", ""), this_week_start, now)
    ]
    scores_last_week = [
        m["score"] for m in scored_messages
        if _in_range(m.get("created_at", ""), last_week_start, this_week_start)
    ]
    posts_this_week = [
        p for p in posts if _in_range(p.get("created_at", ""), this_week_start, now)
    ]

    avg_this_week = round(sum(scores_this_week) / len(scores_this_week), 1) if scores_this_week else None
    avg_last_week = round(sum(scores_last_week) / len(scores_last_week), 1) if scores_last_week else None

    score_trend = None
    if avg_this_week is not None and avg_last_week is not None:
        if avg_this_week - avg_last_week >= 2:
            score_trend = "up"
        elif avg_last_week - avg_this_week >= 2:
            score_trend = "down"
        else:
            score_trend = "flat"

    total_likes = sum(p.get("like_count") or 0 for p in posts_this_week)
    total_comments = sum(p.get("comment_count") or 0 for p in posts_this_week)

    best_post_caption = None
    if posts_this_week:
        best_post = max(
            posts_this_week, key=lambda p: (p.get("like_count") or 0) + (p.get("comment_count") or 0)
        )
        caption = (best_post.get("caption") or "").strip()
        best_post_caption = caption or None

    if not posts_this_week and avg_this_week is None:
        summary = "No activity yet this week — post something and your coach will start tracking it here."
    else:
        stats_lines = [f"Posts this week: {len(posts_this_week)}"]
        if avg_this_week is not None:
            stats_lines.append(f"Average Coach score this week: {avg_this_week}/100")
        if avg_last_week is not None:
            stats_lines.append(f"Average Coach score last week: {avg_last_week}/100")
        stats_lines.append(f"Total likes + comments this week: {total_likes + total_comments}")
        if best_post_caption:
            stats_lines.append(f"Best-performing post's caption: \"{best_post_caption}\"")

        instruction = (
            "You are a warm, encouraging creator coach writing a short weekly report card "
            "for a content creator. Here are their real stats for the week — use ONLY these "
            "numbers, don't invent anything:\n" + "\n".join(stats_lines) + "\n\n"
            "Write 2-3 sentences: acknowledge the effort, call out the trend if there's a "
            "clear one (improving, dipping, or steady), and end with one specific, encouraging "
            "nudge for next week. No markdown, no headers, just the sentences."
        )
        try:
            completion = ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": instruction}],
                temperature=0.8,
                max_tokens=200,
            )
            summary = (completion.choices[0].message.content or "").strip()
        except Exception:
            summary = (
                f"You posted {len(posts_this_week)} time(s) this week"
                + (f" and averaged {avg_this_week}/100 on Coach feedback" if avg_this_week is not None else "")
                + ". Keep it up."
            )
        if not summary:
            summary = "Keep posting — your coach will have more to say once there's more to look at."

    return WeeklyReportResponse(
        posts_this_week=len(posts_this_week),
        avg_score_this_week=avg_this_week,
        avg_score_last_week=avg_last_week,
        score_trend=score_trend,
        total_likes=total_likes,
        total_comments=total_comments,
        best_post_caption=best_post_caption,
        summary=summary,
    )


# ---------------------------------------------------------
# Voice/style consistency check
#
# Every other AI feature in this file looks at ONE post in isolation.
# This is the first one that looks across a creator's post history to
# answer a different question: not "is this caption good," but "does
# this sound like ME" — catching a draft that reads noticeably more
# formal, more sarcastic, or otherwise off-brand versus everything
# else they've posted, before it goes out under their name.
# ---------------------------------------------------------

_VOICE_PROFILE_LOOKBACK = 20
_MIN_CAPTIONS_FOR_VOICE_PROFILE = 5
_MAX_VOICE_EXAMPLE_CAPTIONS = 8

_VOICE_CHECK_RATE_LIMIT = 20
_VOICE_CHECK_RATE_WINDOW = 60 * 60 * 24  # per day
_voice_check_requests: dict = defaultdict(deque)


def _check_voice_check_rate_limit(user_id: str):
    now = time.time()
    q = _voice_check_requests[user_id]
    while q and now - q[0] > _VOICE_CHECK_RATE_WINDOW:
        q.popleft()
    if len(q) >= _VOICE_CHECK_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Limit reached: {_VOICE_CHECK_RATE_LIMIT} voice checks per day. Try again tomorrow.",
        )
    q.append(now)


class VoiceCheckRequest(BaseModel):
    draft: str = Field(..., min_length=1, max_length=500)


class VoiceCheckResponse(BaseModel):
    # False when there isn't enough caption history yet to know what
    # this creator's voice even is — everything below is None in that case.
    has_voice_profile: bool
    consistent: Optional[bool] = None
    reason: Optional[str] = None
    # Only set when consistent is False — a rewrite in their established voice.
    suggested_rewrite: Optional[str] = None


def _get_recent_captions(user_id: str) -> list[str]:
    """
    Pulls this creator's most recent captions to build a voice profile —
    deliberately NOT filtered/ranked by engagement like
    _get_top_performing_captions: voice consistency is about how they
    usually write, not what happened to perform best.
    """
    if supabase_admin is None:
        return []
    try:
        result = (
            supabase_admin
            .table("posts")
            .select("caption,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(_VOICE_PROFILE_LOOKBACK)
            .execute()
        )
        posts = result.data or []
    except Exception:
        return []
    return [c for p in posts if (c := (p.get("caption") or "").strip())]


def _parse_voice_response(raw: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    if "consistent" not in parsed:
        return None

    consistent = bool(parsed.get("consistent"))
    reason = str(parsed.get("reason") or "").strip() or None
    suggested_rewrite = parsed.get("suggested_rewrite")
    suggested_rewrite = str(suggested_rewrite).strip() if suggested_rewrite else None
    if consistent:
        # A rewrite only makes sense when something was flagged as off-brand.
        suggested_rewrite = None

    return {"consistent": consistent, "reason": reason, "suggested_rewrite": suggested_rewrite}


@router.post("/voice-check", response_model=VoiceCheckResponse)
async def voice_check(
    req: VoiceCheckRequest,
    user_id: str = Depends(_get_current_user_id),
):
    _check_voice_check_rate_limit(user_id)

    captions = _get_recent_captions(user_id)
    if len(captions) < _MIN_CAPTIONS_FOR_VOICE_PROFILE:
        return VoiceCheckResponse(has_voice_profile=False)

    examples = "\n".join(f'- "{c}"' for c in captions[:_MAX_VOICE_EXAMPLE_CAPTIONS])
    instruction = (
        "A content creator has an established voice, shown by their past captions "
        "below. Study the tone, formality, length, emoji/hashtag habits, and typical "
        "phrasing:\n"
        f"{examples}\n\n"
        f'Here is a NEW draft caption they\'re considering posting:\n"{req.draft}"\n\n'
        "Does this draft sound consistent with their established voice, or does it read "
        "noticeably off-brand (a different tone, much more/less formal, out of character)? "
        "A different topic is fine on its own — focus on VOICE, not subject matter.\n\n"
        'Respond with ONLY a JSON object like {"consistent": true, "reason": "one short '
        'sentence", "suggested_rewrite": null} — suggested_rewrite should be a version '
        "rewritten in their established voice ONLY when consistent is false, otherwise "
        "null. No other text."
    )

    try:
        completion = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": instruction}],
            temperature=0.3,
            max_tokens=250,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception:
        raise HTTPException(status_code=502, detail="AI service unavailable")

    parsed = _parse_voice_response(raw)
    if parsed is None:
        raise HTTPException(status_code=502, detail="AI returned an unexpected format")

    return VoiceCheckResponse(has_voice_profile=True, **parsed)


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
# about (coach history, posts, profile) and the Storage buckets those
# tables reference (uploaded posts/videos, avatars, repurposed clips),
# followed by the Supabase auth user itself — not a guarantee that
# every row everywhere is gone. There are no migration files in this
# repo to confirm foreign-key cascade behavior for other tables
# (coins/transactions, missions, follows, likes, comments), so verify
# that directly in Supabase before relying on this alone for a
# compliance/GDPR "right to erasure" claim.
# ---------------------------------------------------------

_DELETE_TABLES = {
    "video_coach_messages": "user_id",
    "posts": "user_id",
    "profiles": "id",
}

# Every Storage bucket the app uploads user content into, all keyed by
# a "{user_id}/..." path prefix (confirmed against the Flutter upload
# call sites: post_service.dart, edit_profile.dart, and repurpose.py's
# PROCESSED_BUCKET). This previously deleted DB rows referencing this
# content but left the actual files in Storage forever — a real gap in
# what the app's Privacy screen claims about deleting your data.
_STORAGE_BUCKETS_TO_PURGE = ["posts-media", "avatars", "processed-videos"]


def _purge_storage(user_id: str) -> list[str]:
    """Removes every file under this user's prefix in each bucket above.

    Best-effort per bucket: a failure on one doesn't stop the others,
    and the caller doesn't fail the whole deletion over a storage
    cleanup issue — errors are collected and returned instead.
    """
    errors = []
    for bucket in _STORAGE_BUCKETS_TO_PURGE:
        try:
            files = supabase_admin.storage.from_(bucket).list(user_id)
            paths = [f"{user_id}/{f['name']}" for f in files if f.get("name")]
            if paths:
                supabase_admin.storage.from_(bucket).remove(paths)
        except Exception as e:
            errors.append(f"storage:{bucket}: {e}")
    return errors


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

    cleanup_errors += _purge_storage(user_id)

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
