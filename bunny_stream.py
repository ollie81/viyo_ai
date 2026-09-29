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

router = APIRouter(prefix="/api/v1", tags=["bunny_stream"])

BUNNY_STREAM_API_KEY = os.environ.get("BUNNY_STREAM_API_KEY", "")
BUNNY_STREAM_LIBRARY_ID = os.environ.get("BUNNY_STREAM_LIBRARY_ID", "")
# The CDN hostname Bunny assigns this library's pull zone (Stream > the
# library > General > "Pull Zone Hostname", e.g. vz-xxxxx.b-cdn.net) —
# not secret, but kept server-side so it only needs updating in one
# place if it ever changes, rather than baked into an app build.
BUNNY_STREAM_PULL_ZONE = os.environ.get("BUNNY_STREAM_PULL_ZONE", "")
# Which MP4 Fallback rung to play — must be a resolution the library
# actually generates (1080/720/480/360); asking for higher than the
# source's own resolution 404s, so this defaults conservatively.
BUNNY_STREAM_FALLBACK_RESOLUTION = os.environ.get("BUNNY_STREAM_FALLBACK_RESOLUTION", "720")

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


def _playback_url(video_id: str) -> str:
    return f"https://{BUNNY_STREAM_PULL_ZONE}/{video_id}/play_{BUNNY_STREAM_FALLBACK_RESOLUTION}p.mp4"


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
async def get_bunny_video_status(video_id: str, user_id: str = Depends(_get_current_user_id_no_guest)):
    """
    Ownership isn't checked against a posts row here on purpose: this
    is polled right after the TUS upload finishes, before the posts
    row necessarily exists yet, and the only thing a guessed video_id
    could leak is processing progress on a video that isn't attached
    to any post/profile — not its bytes, not who owns it. user_id is
    still required so only a signed-in account can call it at all.
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

    return BunnyVideoStatusResponse(
        video_id=video_id,
        ready=raw_status in _STATUS_FINISHED,
        failed=raw_status in _STATUS_FAILED,
        raw_status=raw_status,
        playback_url=_playback_url(video_id),
        thumbnail_url=_thumbnail_url(video_id),
        duration_seconds=int(length) if length else None,
    )
