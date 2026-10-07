"""
Bunny Stream integration — video FILES (the actual .mp4 bytes) live in
Bunny Stream instead of Supabase Storage; everything else (auth, the
posts row, likes/comments/coins) stays exactly as it already was.
Supabase Storage keeps serving every video posted before this existed
— see Post.videoProvider in the Flutter app, which is null/'supabase'
for those rows and 'bunny' only for a post created through this flow.

Two endpoints, both auth'd (no guest):

- POST /api/v1/videos/bunny/create — creates the Bunny video object
  server-side (needs the real API key) and hands the client back only
  a short-lived, video-specific TUS upload credential, plus the two
  URLs (playback/thumbnail) that are deterministic from the video id.
  The client then uploads the actual bytes straight to Bunny over TUS
  — this backend never sees or proxies the video file itself, and
  never sends the API key or pull-zone hostname to the client as a
  standing secret (the signature is single-video, time-boxed).

- GET /api/v1/videos/bunny/{video_id}/status — proxies Bunny's own
  get-video call (needs the API key again) so the client can poll
  whether processing has finished, without the key.

MP4 Fallback, not HLS: Bunny's HLS (.m3u8) output needs hls.js on web
(Chrome has no native HLS support), which the existing video_player-
based player here doesn't include. Bunny's per-video MP4 Fallback URL
(https://pull_zone/video_id/play_720p.mp4) is a plain network video
file — every existing VideoPlayerController.networkUrl call site needs
zero changes to play it, which is the entire point of "keep the
existing player." Needs "MP4 Fallback" turned on for the library in
Bunny's dashboard (Stream > this library > General) — see the config
notes handed back with this change.
"""
import hashlib
import os
import time
from typing import Optional

import requests
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/api/v1", tags=["bunny_stream"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# Service-role client used only to self-heal a post's stored media_url
# below — bypasses RLS on purpose. The posts.update RLS policy is
# owner-only, so a plain client-side write from a non-owner viewer
# (anyone except the uploader) silently fails; get_bunny_video_status
# is polled by EVERY viewer, not just the owner, so the fix has to be
# able to write regardless of who's asking.
supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

BUNNY_STREAM_API_KEY = os.environ.get("BUNNY_STREAM_API_KEY", "")
BUNNY_STREAM_LIBRARY_ID = os.environ.get("BUNNY_STREAM_LIBRARY_ID", "")
# The CDN hostname Bunny assigns this library's pull zone (Stream > the
# library > General > "Pull Zone Hostname", e.g. vz-xxxxx.b-cdn.net) —
# not secret, but kept server-side so it only needs updating in one
# place if it ever changes, rather than baked into an app build.
BUNNY_STREAM_PULL_ZONE = os.environ.get("BUNNY_STREAM_PULL_ZONE", "")
# Which MP4 Fallback rung to play when Bunny hasn't reported its
# actually-generated resolutions yet (create_bunny_video, right before
# upload even starts) — see _pick_resolution below for the real,
# per-video choice used everywhere else. Bunny only generates rungs at
# or below the source video's own resolution (a short, phone-shot or
# AI-generated drama episode is very often under 720p), so asking for
# a fixed 720p unconditionally 404s for any video Bunny didn't also
# generate a 720p rendition for.
BUNNY_STREAM_FALLBACK_RESOLUTION = os.environ.get("BUNNY_STREAM_FALLBACK_RESOLUTION", "720")

# Highest to lowest — the first of these actually present in a video's
# own availableResolutions wins. 240 is Bunny's baseline rung, normally
# generated for every finished video, so it's the final fallback.
_RESOLUTION_PRIORITY = ["1080", "720", "480", "360", "240"]


def _pick_resolution(available_resolutions: str) -> str:
    """
    Bunny's video object reports availableResolutions as a comma-separated
    string like "240p,360p,480p" — only the rungs it actually generated for
    this specific video, which depends on the source's own resolution.
    Picks the highest one actually present instead of assuming a fixed
    rung exists, which is what let play_720p.mp4 404 forever for any
    video Bunny encoded below 720p.
    """
    present = {r.strip().rstrip("p") for r in available_resolutions.split(",") if r.strip()}
    for res in _RESOLUTION_PRIORITY:
        if res in present:
            return res
    return BUNNY_STREAM_FALLBACK_RESOLUTION

_BUNNY_API_BASE = "https://video.bunnycdn.com"
_TUS_UPLOAD_ENDPOINT = "https://video.bunnycdn.com/tusupload"
# How long the client has to finish the TUS upload before the signed
# credential expires — generous for a large drama episode on a slow
# connection; a fresh credential just needs a new create call if this
# is somehow exceeded.
_UPLOAD_WINDOW_SECONDS = 6 * 60 * 60

# Bunny's numeric video.status values (GET /library/{id}/videos/{id}):
# 0 Created, 1 Uploaded, 2 Processing, 3 Transcoding, 4 Finished,
# 5 Error, 6 UploadFailed, 7 JitSegmenting, 8 JitPlaylistsCreated.
_STATUS_FINISHED = {4, 8}
_STATUS_FAILED = {5, 6}


def _configured() -> bool:
    return bool(BUNNY_STREAM_API_KEY and BUNNY_STREAM_LIBRARY_ID and BUNNY_STREAM_PULL_ZONE)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


async def _get_current_user_id(authorization: str = Header(None)) -> str:
    from main import get_current_user_id

    return await get_current_user_id(authorization)


def _playback_url(video_id: str, resolution: Optional[str] = None) -> str:
    return f"https://{BUNNY_STREAM_PULL_ZONE}/{video_id}/play_{resolution or BUNNY_STREAM_FALLBACK_RESOLUTION}p.mp4"


def _thumbnail_url(video_id: str) -> str:
    return f"https://{BUNNY_STREAM_PULL_ZONE}/{video_id}/thumbnail.jpg"


class CreateBunnyVideoRequest(BaseModel):
    title: str = "Untitled"


class CreateBunnyVideoResponse(BaseModel):
    video_id: str
    library_id: str
    upload_endpoint: str
    authorization_signature: str
    authorization_expire: int
    playback_url: str
    thumbnail_url: str


@router.post("/videos/bunny/create", response_model=CreateBunnyVideoResponse)
async def create_bunny_video(
    req: CreateBunnyVideoRequest, user_id: str = Depends(_get_current_user_id_no_guest)
):
    """
    Called right before the client starts a video upload — creates the
    Bunny-side video object and returns a single-video, time-boxed TUS
    credential for it. 503 (not a 500) when Bunny isn't configured at
    all, so the Flutter client can tell "not set up yet, fall back to
    Supabase Storage" apart from "Bunny itself is having a problem."
    """
    if not _configured():
        raise HTTPException(status_code=503, detail="Bunny Stream is not configured.")

    try:
        resp = requests.post(
            f"{_BUNNY_API_BASE}/library/{BUNNY_STREAM_LIBRARY_ID}/videos",
            json={"title": (req.title or "Untitled")[:200]},
            headers={"AccessKey": BUNNY_STREAM_API_KEY, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not create Bunny video: {e}")

    video_id = resp.json().get("guid")
    if not video_id:
        raise HTTPException(status_code=502, detail="Bunny did not return a video id.")

    expire = int(time.time()) + _UPLOAD_WINDOW_SECONDS
    signature = hashlib.sha256(
        f"{BUNNY_STREAM_LIBRARY_ID}{BUNNY_STREAM_API_KEY}{expire}{video_id}".encode()
    ).hexdigest()

    return CreateBunnyVideoResponse(
        video_id=video_id,
        library_id=BUNNY_STREAM_LIBRARY_ID,
        upload_endpoint=_TUS_UPLOAD_ENDPOINT,
        authorization_signature=signature,
        authorization_expire=expire,
        playback_url=_playback_url(video_id),
        thumbnail_url=_thumbnail_url(video_id),
    )


class BunnyVideoStatusResponse(BaseModel):
    video_id: str
    ready: bool
    failed: bool
    raw_status: int
    playback_url: str
    thumbnail_url: str
    duration_seconds: Optional[int] = None


@router.get("/videos/bunny/{video_id}/status", response_model=BunnyVideoStatusResponse)
async def get_bunny_video_status(video_id: str, user_id: str = Depends(_get_current_user_id)):
    """
    Ownership isn't checked against a posts row here on purpose: this
    is polled right after the TUS upload finishes, before the posts
    row necessarily exists yet, and the only thing a guessed video_id
    could leak is processing progress on a video that isn't attached
    to any post/profile — not its bytes, not who owns it. user_id is
    still required so only a signed-in (or guest) session can call it
    at all.

    Guest-inclusive on purpose, unlike create_bunny_video above: this
    is read-only and polled by every viewer of the Dramas feed, not
    just the account that uploaded a video — Studio publishes every
    drama episode through Bunny (video_status="processing" by
    default), and the ONLY thing that ever flips that status to
    "ready" is a guest or signed-in viewer's own client successfully
    polling this endpoint (see video_feed_screen.dart's
    _pollBunnyStatus). Using the no-guest variant here meant every
    guest viewer's poll got a 403, failed silently, and left every
    single episode in the feed stuck showing "Processing..." forever
    for them — there's no server-side Bunny webhook backing this up.
    """
    if not _configured():
        raise HTTPException(status_code=503, detail="Bunny Stream is not configured.")

    try:
        resp = requests.get(
            f"{_BUNNY_API_BASE}/library/{BUNNY_STREAM_LIBRARY_ID}/videos/{video_id}",
            headers={"AccessKey": BUNNY_STREAM_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not check video status: {e}")

    data = resp.json()
    raw_status = int(data.get("status", 0))
    length = data.get("length")
    resolution = _pick_resolution(data.get("availableResolutions") or "")
    playback_url = _playback_url(video_id, resolution)

    _self_heal_post_media_url(
        video_id, raw_status, playback_url,
        width=data.get("width"), height=data.get("height"),
        duration_seconds=int(length) if length else None,
    )

    return BunnyVideoStatusResponse(
        video_id=video_id,
        ready=raw_status in _STATUS_FINISHED,
        failed=raw_status in _STATUS_FAILED,
        raw_status=raw_status,
        playback_url=playback_url,
        thumbnail_url=_thumbnail_url(video_id),
        duration_seconds=int(length) if length else None,
    )


def _self_heal_post_media_url(
    video_id: str,
    raw_status: int,
    playback_url: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    duration_seconds: Optional[int] = None,
) -> None:
    """
    Persists this poll's result onto every posts row for this video, via
    the service-role client — bypassing RLS on purpose.

    Without this, the only place the corrected URL/status ever got saved
    was the Flutter client's own follow-up write (PostService.
    updateVideoStatus), which goes through the normal Supabase client and
    is RLS-scoped: posts' update policy is owner-only, so that write
    silently no-ops for every viewer except the post's own uploader. A
    drama episode is watched overwhelmingly by people who aren't its
    uploader, so in practice the fix never stuck — the same broken
    play_<N>p.mp4 URL (or a stuck "processing" status) kept getting
    re-served to every other viewer forever, each one re-discovering the
    same failure. Doing the write here means the very first poll from
    ANYONE, owner or not, heals it for everyone after.

    width/height/duration_seconds ride along on the same write for the
    same reason: this is the one place that already has Bunny's own
    video object in hand, which reports a Bunny-hosted video's real
    encoded dimensions and length — there's no reason to ask a viewer's
    browser to decode the video client-side just to learn its shape
    when Bunny already told us. width/height only ever fill a
    currently-null value, same posture as media_url/video_status above;
    duration_seconds is always kept in sync with Bunny's own number
    when it differs, since Bunny's is authoritative and the client-side
    create-post flow only ever has a placeholder/guessed value to start
    with (never a real one — it hasn't finished uploading yet).
    """
    if supabase_admin is None:
        return
    new_status = "failed" if raw_status in _STATUS_FAILED else ("ready" if raw_status in _STATUS_FINISHED else None)
    try:
        rows = (
            supabase_admin.table("posts")
            .select("id, media_url, video_status, width, height, duration_seconds")
            .eq("bunny_video_id", video_id)
            .execute()
            .data
        )
        for row in rows:
            updates = {}
            if new_status is not None:
                if row.get("video_status") != new_status:
                    updates["video_status"] = new_status
                if new_status == "ready" and row.get("media_url") != playback_url:
                    updates["media_url"] = playback_url
            if width and height and not row.get("width") and not row.get("height"):
                updates["width"] = width
                updates["height"] = height
            if duration_seconds and row.get("duration_seconds") != duration_seconds:
                updates["duration_seconds"] = duration_seconds
            if updates:
                supabase_admin.table("posts").update(updates).eq("id", row["id"]).execute()
    except Exception as e:
        # Best-effort, same reasoning as every other self-heal path in
        # this codebase: worse case is the next poll (from anyone) tries
        # again, never that a transient DB hiccup breaks video playback.
        print(f"[WARN] self-heal write-back failed for bunny video {video_id}: {e}")
