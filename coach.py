import datetime
import json
import os
import re
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Header, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from supabase import create_client, Client

from coins import spend_on_feature, refund_feature


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


async def _get_current_user_id_no_guest(
    authorization: str = Header(None),
) -> str:
    """
    Same as _get_current_user_id but rejects a guest (anonymous Supabase
    session). Used on every endpoint in this file that calls OpenAI —
    see main.get_current_user_id_no_guest for why.
    """
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


# ---------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------

class VideoContext(BaseModel):
    """
    What the Coach can actually see about the video being discussed.

    Until this existed the Coach was blind: it received a video_id and
    used it for exactly one thing — partitioning chat history — so every
    answer was generic content-coaching filler about a video it had
    never seen, charged at real coins per message.

    Two sources fill this in, and they are not equally trusted:

      * A posted video resolves server-side from the `posts` row
        (see _load_post_context) — authoritative, ownership-checked.
      * A clip that only exists in the AI repurposer has no post row
        yet, so the client passes what the repurposer returned. That is
        the creator describing their own video to their own coach, so
        there is nothing to escalate here; it is still clamped in
        length and clearly labelled in the prompt as creator-supplied.
    """

    # Straight from the repurposer's RepurposeResponse / VideoFeedback.
    # Generous, not tight: a 30-minute upload transcribes to roughly
    # 25,000 characters, and a coach that 422s on a long video is worse
    # than one that reads an abridged version of it. The prompt-side
    # clamp below is what actually keeps the token bill bounded.
    transcript: str = Field(default="", max_length=60000)
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    hook_line: str = Field(default="", max_length=500)
    caption: str = Field(default="", max_length=1000)
    hashtags: list[str] = Field(default_factory=list)
    verdict: str = Field(default="", max_length=1000)
    issues: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    footage_score: Optional[int] = Field(default=None, ge=0, le=100)
    words_per_minute: Optional[float] = Field(default=None, ge=0)
    silence_percent: Optional[float] = Field(default=None, ge=0, le=100)


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

    # What the client knows about the video. Ignored when the video is
    # a real post, since the database is the better source.
    video_context: Optional[VideoContext] = None


class CoachMessageResponse(BaseModel):
    video_id: str
    response: str
    video_version: int
    score: Optional[int] = None
    # Three short follow-ups the creator can tap instead of typing.
    # Empty if the model didn't emit a usable set.
    suggestions: list[str] = Field(default_factory=list)


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
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    _check_caption_variants_rate_limit(user_id)
    spend_on_feature(supabase_admin, user_id, "caption_variants")

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
    user_id: str = Depends(_get_current_user_id_no_guest),
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
# "Why This Worked" — post performance insight
#
# Grounded ONLY in this creator's own real numbers (this post's likes +
# comments vs. the average of their own other recent posts) and the
# post's own caption text — deliberately not framed as "the algorithm"
# or reach, since Viyo doesn't track view/impression counts at all
# today and claiming otherwise would just be inventing data.
# ---------------------------------------------------------

_POST_INSIGHT_BASELINE_LOOKBACK = 30
_MIN_POSTS_FOR_BASELINE = 3


class PostInsightResponse(BaseModel):
    post_id: str
    engagement: int
    baseline_avg_engagement: Optional[float] = None
    # 'above' / 'about' / 'below' this creator's own average — only set
    # once there's enough post history to compare against.
    performance: Optional[str] = None
    explanation: str


@router.get("/post-insight/{post_id}", response_model=PostInsightResponse)
async def post_insight(
    post_id: str,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Insight service is not configured.")

    try:
        post_result = (
            supabase_admin
            .table("posts")
            .select("id,user_id,caption,like_count,comment_count")
            .eq("id", post_id)
            .limit(1)
            .execute()
        )
        rows = post_result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load post: {e}")

    if not rows:
        raise HTTPException(status_code=404, detail="Post not found.")
    post = rows[0]
    if post.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="You can only see insights for your own posts.")

    spend_on_feature(supabase_admin, user_id, "post_insight")

    engagement = (post.get("like_count") or 0) + (post.get("comment_count") or 0)
    caption = (post.get("caption") or "").strip()

    try:
        others_result = (
            supabase_admin
            .table("posts")
            .select("like_count,comment_count")
            .eq("user_id", user_id)
            .neq("id", post_id)
            .order("created_at", desc=True)
            .limit(_POST_INSIGHT_BASELINE_LOOKBACK)
            .execute()
        )
        other_posts = others_result.data or []
    except Exception:
        other_posts = []

    baseline_avg = None
    performance = None
    if len(other_posts) >= _MIN_POSTS_FOR_BASELINE:
        others_engagement = [
            (p.get("like_count") or 0) + (p.get("comment_count") or 0) for p in other_posts
        ]
        baseline_avg = round(sum(others_engagement) / len(others_engagement), 1)
        if baseline_avg > 0:
            ratio = engagement / baseline_avg
            performance = "above" if ratio >= 1.2 else "below" if ratio <= 0.8 else "about"
        else:
            performance = "above" if engagement > 0 else "about"

    stats_lines = [f"This post's likes + comments: {engagement}"]
    if baseline_avg is not None:
        stats_lines.append(f"This creator's average likes + comments per post: {baseline_avg}")
        stats_lines.append(f"Performance vs. their own average: {performance}")
    if caption:
        stats_lines.append(f'Caption: "{caption}"')

    instruction = (
        "You are a creator coach explaining, in plain language, why one specific post "
        "performed the way it did — grounded ONLY in the numbers and caption text given "
        "below. Never invent metrics, view/reach counts, or claims about a recommendation "
        "algorithm that aren't in this data. If there isn't enough history to compare "
        "against, say so honestly instead of guessing.\n\n"
        + "\n".join(stats_lines) + "\n\n"
        "Write 2-3 sentences: name one or two concrete things about the caption itself "
        "(hook, length, question, call-to-action, tone) that plausibly helped or hurt, "
        "reference the real comparison number if there is one, and end with one specific "
        "thing to try next time. No markdown, no headers, just the sentences."
    )

    try:
        completion = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": instruction}],
            temperature=0.6,
            max_tokens=200,
        )
        explanation = (completion.choices[0].message.content or "").strip()
    except Exception:
        explanation = (
            f"This post got {engagement} likes and comments combined"
            + (f", vs. your usual average of {baseline_avg}." if baseline_avg is not None else ".")
        )
    if not explanation:
        explanation = "Not enough to go on yet — keep posting and this will get sharper."

    return PostInsightResponse(
        post_id=post_id,
        engagement=engagement,
        baseline_avg_engagement=baseline_avg,
        performance=performance,
        explanation=explanation,
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
    user_id: str = Depends(_get_current_user_id_no_guest),
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
# Trend matching against the app's own aggregated engagement data
#
# Every other feature in this file only ever looks at ONE creator's
# own history. This is the first one that looks ACROSS creators —
# there's no TikTok/Instagram/YouTube trends API integration here, so
# the only "what's trending" signal this app can honestly offer is its
# own: which recent posts, from other creators in the same niche, are
# actually getting engagement right now. Captions are already public
# content shown across the feed to every user regardless (see
# video_feed_screen.dart / post_card.dart) — this surfaces them
# anonymously (no user_id, no way to trace a caption back to who
# posted it) as inspiration rather than attributed examples.
# ---------------------------------------------------------

_TRENDING_WINDOW_DAYS = 7
_TRENDING_LOOKBACK_ROWS = 300
_MIN_POSTS_FOR_TREND = 5
_TOP_TREND_POSTS = 8

_TRENDING_RATE_LIMIT = 20
_TRENDING_RATE_WINDOW = 60 * 60 * 24  # per day
_trending_requests: dict = defaultdict(deque)


def _check_trending_rate_limit(user_id: str):
    now = time.time()
    q = _trending_requests[user_id]
    while q and now - q[0] > _TRENDING_RATE_WINDOW:
        q.popleft()
    if len(q) >= _TRENDING_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Limit reached: {_TRENDING_RATE_LIMIT} trend checks per day. Try again tomorrow.",
        )
    q.append(now)


class TrendingResponse(BaseModel):
    has_data: bool
    niche: str
    sample_size: int
    themes: list[str] = []
    # Real captions from the niche's top-performing recent posts —
    # anonymous (no user_id attached), shown as inspiration.
    example_captions: list[str] = []
    idea: str = ""


def _get_trending_posts(niche: str) -> list[dict]:
    """
    Pulls the top-performing recent posts from OTHER creators sharing
    this niche. Two-step lookup (profiles -> posts) since there's no
    niche column on posts itself — service-role client bypasses RLS
    here the same way every other admin query in this file does.
    Never raises: any failure or lack of data just means no trend.
    """
    if supabase_admin is None or not niche.strip():
        return []

    try:
        profiles_result = (
            supabase_admin
            .table("profiles")
            .select("id")
            .eq("niche", niche)
            .limit(500)
            .execute()
        )
        user_ids = [p["id"] for p in (profiles_result.data or []) if p.get("id")]
        if not user_ids:
            return []

        cutoff = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=_TRENDING_WINDOW_DAYS)
        ).isoformat()
        posts_result = (
            supabase_admin
            .table("posts")
            .select("caption,like_count,comment_count,created_at")
            .in_("user_id", user_ids)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(_TRENDING_LOOKBACK_ROWS)
            .execute()
        )
        posts = posts_result.data or []
    except Exception:
        return []

    scored = sorted(
        (p for p in posts if (p.get("caption") or "").strip()),
        key=lambda p: (p.get("like_count") or 0) + (p.get("comment_count") or 0),
        reverse=True,
    )
    return scored[:_TOP_TREND_POSTS]


def _parse_trending_response(raw: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None

    themes = parsed.get("themes")
    idea = parsed.get("idea")
    if not isinstance(themes, list) or not idea:
        return None

    themes = [str(t).strip() for t in themes if str(t).strip()][:4]
    if not themes:
        return None

    return {"themes": themes, "idea": str(idea).strip()}


@router.get("/trending", response_model=TrendingResponse)
async def trending(
    niche: str = Query(..., min_length=1, max_length=100),
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    _check_trending_rate_limit(user_id)

    top_posts = _get_trending_posts(niche)
    if len(top_posts) < _MIN_POSTS_FOR_TREND:
        return TrendingResponse(has_data=False, niche=niche, sample_size=len(top_posts))

    example_captions = [p["caption"].strip() for p in top_posts[:3]]
    all_captions = [p["caption"].strip() for p in top_posts]
    examples = "\n".join(f'- "{c}"' for c in all_captions)

    instruction = (
        f"Here are the top-performing recent post captions from creators in the "
        f"'{niche}' niche on this app, ranked by engagement:\n{examples}\n\n"
        "Identify 2-3 short common THEMES or angles across these (e.g. 'before/after "
        "transformations', 'quick daily tips', 'personal storytelling') and suggest ONE "
        f"concrete new post idea for a creator in the '{niche}' niche that rides one of "
        "these trends.\n\n"
        'Respond with ONLY a JSON object like {"themes": ["theme one", "theme two"], '
        '"idea": "one sentence post idea"} — no other text.'
    )

    try:
        completion = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": instruction}],
            temperature=0.7,
            max_tokens=250,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception:
        raise HTTPException(status_code=502, detail="AI service unavailable")

    parsed = _parse_trending_response(raw)
    if parsed is None:
        raise HTTPException(status_code=502, detail="AI returned an unexpected format")

    return TrendingResponse(
        has_data=True,
        niche=niche,
        sample_size=len(top_posts),
        themes=parsed["themes"],
        example_captions=example_captions,
        idea=parsed["idea"],
    )


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
# Giving the Coach eyes
#
# The Coach used to receive a video_id and use it for exactly one
# thing: partitioning chat history. It never loaded the caption, the
# stats, or a word of what was actually said — so it answered with
# generic advice about a video it had never seen, at real coins per
# message, while its own prompt told it to admit when it lacked
# information. These two helpers are what it looks at now.
# ---------------------------------------------------------

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_MAX_CONTEXT_TRANSCRIPT_CHARS = 6000
_MAX_CONTEXT_COMMENTS = 8


def _abridge_transcript(transcript: str) -> str:
    """
    Keeps a long transcript inside a sane token budget without throwing
    away the two parts the Coach is most often asked about.

    A naive head-truncation on a 30-minute upload would leave the Coach
    able to discuss the opening and literally nothing else — so it can't
    answer "does my ending land?", which is one of the things it is
    explicitly asked to critique. Keeping the head AND the tail costs
    the same tokens and preserves both the hook and the payoff.
    """
    if len(transcript) <= _MAX_CONTEXT_TRANSCRIPT_CHARS:
        return transcript

    head_chars = int(_MAX_CONTEXT_TRANSCRIPT_CHARS * 0.6)
    tail_chars = _MAX_CONTEXT_TRANSCRIPT_CHARS - head_chars
    skipped = len(transcript) - head_chars - tail_chars

    return (
        transcript[:head_chars]
        + f"\n\n[... roughly {skipped // 6} words from the middle omitted "
        f"— you have the opening and the ending, not the middle ...]\n\n"
        + transcript[-tail_chars:]
    )


def _load_post_context(user_id: str, video_id: str) -> Optional[dict]:
    """
    Resolves video_id as one of the creator's own posts.

    Returns None when video_id isn't a post at all — which is the normal
    case for a clip that only exists in the AI repurposer and hasn't
    been posted yet. Scoped to user_id so a creator can only ever pull
    context for their own video, and best-effort: a query failure means
    the Coach answers with less context, never that the chat breaks.
    """
    if supabase_admin is None or not _UUID_RE.match(video_id):
        return None

    try:
        result = (
            supabase_admin
            .table("posts")
            .select(
                "id,caption,duration_seconds,like_count,comment_count,"
                "post_type,is_boosted,created_at"
            )
            .eq("id", video_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
    except Exception:
        return None

    if not rows:
        return None

    post = rows[0]

    # Real viewer comments are the single most useful thing the Coach
    # can read: they are the audience reacting in their own words,
    # rather than the model guessing how an audience might react.
    try:
        comment_result = (
            supabase_admin
            .table("comments")
            .select("content,created_at")
            .eq("post_id", video_id)
            .order("created_at", desc=True)
            .limit(_MAX_CONTEXT_COMMENTS)
            .execute()
        )
        post["recent_comments"] = [
            (c.get("content") or "").strip()
            for c in (comment_result.data or [])
            if (c.get("content") or "").strip()
        ]
    except Exception:
        post["recent_comments"] = []

    return post


def _render_video_context(
    post: Optional[dict],
    client_ctx: Optional[VideoContext],
) -> Optional[str]:
    """
    Turns whatever context we managed to gather into one system message.

    Returns None when there is genuinely nothing to show, so the Coach
    keeps its old honest "I can't see this video" behaviour instead of
    being handed an empty form to hallucinate into.
    """
    lines: list[str] = []

    if post:
        lines.append("THIS VIDEO, AS POSTED ON VIYO (from the database — reliable):")
        caption = (post.get("caption") or "").strip()
        if caption:
            lines.append(f'- Caption: "{caption}"')
        if post.get("duration_seconds"):
            lines.append(f"- Length: {post['duration_seconds']} seconds")
        lines.append(
            f"- Engagement so far: {post.get('like_count') or 0} likes, "
            f"{post.get('comment_count') or 0} comments"
        )
        posted_at = _parse_timestamp(post.get("created_at") or "")
        if posted_at:
            age_hours = (
                datetime.datetime.now(datetime.timezone.utc) - posted_at
            ).total_seconds() / 3600
            if age_hours < 48:
                lines.append(f"- Posted about {max(1, round(age_hours))} hours ago")
            else:
                lines.append(f"- Posted about {round(age_hours / 24)} days ago")
        if post.get("is_boosted"):
            lines.append("- The creator paid coins to boost this post")

        comments = post.get("recent_comments") or []
        if comments:
            lines.append("- What viewers actually commented:")
            lines.extend(f'    * "{c}"' for c in comments)

    if client_ctx:
        ctx_lines: list[str] = []

        if client_ctx.duration_seconds:
            ctx_lines.append(f"- Clip length: {client_ctx.duration_seconds:.0f} seconds")
        if client_ctx.hook_line.strip():
            ctx_lines.append(f'- Opening hook on screen: "{client_ctx.hook_line.strip()}"')
        if client_ctx.caption.strip():
            ctx_lines.append(f'- Generated caption: "{client_ctx.caption.strip()}"')
        if client_ctx.hashtags:
            ctx_lines.append("- Hashtags: " + " ".join(client_ctx.hashtags[:12]))
        if client_ctx.footage_score is not None:
            ctx_lines.append(
                f"- Viyo's automated footage score: {client_ctx.footage_score}/100"
            )
        if client_ctx.verdict.strip():
            ctx_lines.append(f"- Automated verdict: {client_ctx.verdict.strip()}")
        if client_ctx.issues:
            ctx_lines.append("- Problems the analyzer measured:")
            ctx_lines.extend(f"    * {i}" for i in client_ctx.issues[:8])
        if client_ctx.strengths:
            ctx_lines.append("- What the analyzer said worked:")
            ctx_lines.extend(f"    * {i}" for i in client_ctx.strengths[:8])
        if client_ctx.words_per_minute:
            ctx_lines.append(
                f"- Speaking pace: {client_ctx.words_per_minute:.0f} words per minute"
            )
        if client_ctx.silence_percent:
            ctx_lines.append(
                f"- Dead air: {client_ctx.silence_percent:.0f}% of the video is silence"
            )

        transcript = _abridge_transcript(client_ctx.transcript.strip())
        if transcript:
            ctx_lines.append(f"- Word-for-word transcript:\n\"\"\"{transcript}\"\"\"")

        if ctx_lines:
            if lines:
                lines.append("")
            lines.append(
                "WHAT VIYO'S VIDEO ANALYZER MEASURED ON THIS VIDEO "
                "(supplied by the creator's own app session):"
            )
            lines.extend(ctx_lines)

    if not lines:
        return None

    return (
        "You CAN see this video. Here is everything Viyo knows about it. "
        "Ground every piece of advice in these specifics — quote the "
        "creator's own words back to them, name the actual caption, cite "
        "the real numbers. Never invent a detail that is not listed here, "
        "and if something you need is missing (for example you have stats "
        "but no transcript), say which part you cannot see rather than "
        "guessing.\n\n" + "\n".join(lines)
    )


# ---------------------------------------------------------
# Tappable follow-ups
#
# The model is asked to end its reply with a machine-readable block so
# the app can offer three next questions as chips. It is stripped out
# of the text before anything is stored or shown — a creator should
# never see the plumbing, and the stored history stays clean for the
# next turn's context.
# ---------------------------------------------------------

_SUGGESTIONS_RE = re.compile(
    r"\[\[SUGGESTIONS:(.*?)\]\]", re.DOTALL | re.IGNORECASE
)

_SUGGESTIONS_INSTRUCTION = (
    "\n\nAfter your reply, on its own final line, list exactly three short "
    "follow-up questions the creator could ask you next, in their own "
    "voice, each under 45 characters, in this exact format and nothing "
    "after it:\n"
    "[[SUGGESTIONS: first question | second question | third question]]"
)


def _split_suggestions(raw: str) -> tuple[str, list[str]]:
    """
    Returns (clean reply, suggestions). A reply with no block — or a
    malformed one — just comes back untouched with no suggestions,
    since chips are a nicety and must never cost the creator the
    answer they paid coins for.
    """
    match = _SUGGESTIONS_RE.search(raw)
    if not match:
        return raw.strip(), []

    suggestions = [
        part.strip().strip('"').strip()
        for part in match.group(1).split("|")
    ]
    suggestions = [s for s in suggestions if s][:3]

    clean = _SUGGESTIONS_RE.sub("", raw).strip()
    # A model that emitted ONLY the block gave us nothing to show, so
    # keep the raw text rather than handing back an empty bubble.
    if not clean:
        return raw.strip(), []

    return clean, suggestions


# ---------------------------------------------------------
# AI Coach
# ---------------------------------------------------------

_COACH_SYSTEM_PROMPT = """
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

FORMATTING:

Write for a phone screen. Keep paragraphs to two or three lines. Use
**bold** for the thing that matters most, and `- ` bullets for lists of
fixes. Do not use headings larger than `## `. Do not write a wall of
text.

When appropriate, give a score from 0 to 100.

A score should reflect the current version of the video, not the creator
as a person.

You are a coach, not a judge.
"""

_COACH_HISTORY_LIMIT = 30


def _load_coach_history(user_id: str, video_id: str) -> list[dict]:
    try:
        history_result = (
            supabase_admin
            .table("video_coach_messages")
            .select(
                "role,message,video_version,score,created_at"
            )
            .eq("user_id", user_id)
            .eq("video_id", video_id)
            .order("created_at", desc=False)
            .limit(_COACH_HISTORY_LIMIT)
            .execute()
        )

        return history_result.data or []

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load Coach history: {e}",
        )


def _build_coach_messages(
    user_id: str,
    req: CoachMessageRequest,
    history: list[dict],
) -> list[dict]:
    """
    Assembles the full prompt: the coaching persona, then everything
    Viyo actually knows about this specific video, then the chat so far.

    The video-context block is rebuilt on every turn rather than stored,
    so a post's engagement numbers are always current — the creator who
    asks "how is it doing now?" gets now, not whatever was true when the
    conversation started.
    """
    messages: list[dict] = [
        {"role": "system", "content": _COACH_SYSTEM_PROMPT}
    ]

    video_context = _render_video_context(
        _load_post_context(user_id, req.video_id),
        req.video_context,
    )
    if video_context:
        messages.append({"role": "system", "content": video_context})

    for item in history:
        role = item.get("role")

        if role not in ("user", "coach"):
            continue

        messages.append({
            "role": "user" if role == "user" else "assistant",
            "content": item.get("message", ""),
        })

    messages.append({
        "role": "user",
        "content": req.message + _SUGGESTIONS_INSTRUCTION,
    })

    return messages


def _prepare_coach_turn(
    user_id: str, req: CoachMessageRequest
) -> tuple[list[dict], int]:
    """
    Everything both the blocking and the streaming endpoint do before a
    single token is generated: charge the coins, load the history, save
    the creator's message, build the prompt.

    Returns the prompt and how many coins were actually taken, so that
    a turn which then fails can hand them straight back.
    """
    if supabase_admin is None:
        raise HTTPException(
            status_code=503,
            detail="Coach database is not configured.",
        )

    charged = spend_on_feature(supabase_admin, user_id, "coach_message")

    history = _load_coach_history(user_id, req.video_id)

    _save_message(
        user_id=user_id,
        video_id=req.video_id,
        role="user",
        message=req.message,
        video_version=req.video_version,
        score=req.score,
    )

    return _build_coach_messages(user_id, req, history), charged


@router.post(
    "/coach/message",
    response_model=CoachMessageResponse,
)
async def coach_message(
    req: CoachMessageRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):

    messages, charged = _prepare_coach_turn(user_id, req)

    try:

        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
        )

        raw_response = (
            response.choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

        # The creator got no coaching, so they keep their coins.
        refund_feature(supabase_admin, user_id, "coach_message", charged)

        raise HTTPException(
            status_code=502,
            detail=f"Coach AI request failed: {e}",
        )

    coach_response, suggestions = _split_suggestions(raw_response)

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
        suggestions=suggestions,
    )


# ---------------------------------------------------------
# Streaming version of the same turn
#
# A coach that takes eight silent seconds and then dumps a paragraph
# feels broken; one that starts talking immediately feels alive. Same
# charge, same history, same saved result — only the delivery differs,
# so the blocking endpoint above stays as the fallback for any client
# that can't read a stream.
#
# Server-Sent Events, one JSON object per `data:` line:
#   {"delta": "..."}                      incremental text
#   {"done": true, "suggestions": [...]}  end of a successful turn
#   {"error": "..."}                      generation failed mid-stream
# ---------------------------------------------------------

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/coach/message/stream")
async def coach_message_stream(
    req: CoachMessageRequest,
    user_id: str = Depends(_get_current_user_id_no_guest),
):
    # Deliberately outside the generator: coins, history and validation
    # must fail as a normal HTTP error the client can show, not as an
    # error frame inside a 200 response that has already started.
    messages, charged = _prepare_coach_turn(user_id, req)

    def generate():
        collected: list[str] = []

        # The suggestions block arrives at the very end, one token at a
        # time. Holding back a small tail means the creator never sees
        # "[[SUGGES" flicker onto the screen before it gets stripped.
        pending = ""
        emitted = ""

        try:
            stream = ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                collected.append(delta)
                pending += delta

                # Never emit anything from the marker onward — and drop
                # the blank line the model puts before it, so the text
                # already on screen matches the text that gets stored
                # and no reply ends in a needless full-bubble redraw.
                marker_at = pending.find("[[")
                if marker_at == -1:
                    # Hold back two characters in case "[[" itself is
                    # split across chunk boundaries.
                    safe_until = len(pending) - 2
                else:
                    safe_until = len(pending[:marker_at].rstrip())

                if safe_until > len(emitted):
                    delta_out = pending[len(emitted):safe_until]
                    emitted += delta_out
                    yield _sse({"delta": delta_out})

        except Exception as e:
            refund_feature(supabase_admin, user_id, "coach_message", charged)
            yield _sse({"error": f"Coach AI request failed: {e}"})
            return

        raw_response = "".join(collected)
        if not raw_response.strip():
            refund_feature(supabase_admin, user_id, "coach_message", charged)
            yield _sse({"error": "Coach returned an empty response."})
            return

        coach_response, suggestions = _split_suggestions(raw_response)

        # Flush whatever the tail-holding above kept back, so the client
        # ends up with exactly the text that gets stored. If stripping
        # moved the text out from under what was already sent (a model
        # that opened with whitespace), replace the bubble outright
        # rather than leave the creator with a mismatched answer.
        if coach_response.startswith(emitted):
            remainder = coach_response[len(emitted):]
            if remainder:
                yield _sse({"delta": remainder})
        else:
            yield _sse({"replace": coach_response})

        try:
            _save_message(
                user_id=user_id,
                video_id=req.video_id,
                role="coach",
                message=coach_response,
                video_version=req.video_version,
                score=req.score,
            )
        except HTTPException as e:
            # The creator already paid and already read the answer, so
            # the turn is not a failure — but they need to know it won't
            # be there when they come back.
            yield _sse({
                "done": True,
                "suggestions": suggestions,
                "warning": f"This reply could not be saved: {e.detail}",
            })
            return

        yield _sse({"done": True, "suggestions": suggestions})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops nginx-style proxies buffering the stream
        },
    )
