"""
Creator downloads: a creator can download their own finished videos
(episodes, films, standalone videos) and thumbnails to post anywhere.

Rules this file keeps:
  * Read-only. Nothing here deletes, moves, re-uploads or edits a file
    or a post. Viyo's stored copy (Supabase Storage or Bunny Stream) stays
    exactly where it is; a download only hands out a link to read it.
  * Owner only, decided on the server from the signed-in user, never from
    an id the app sends. Someone else's post answers 404, the same as a
    post that doesn't exist, so private posts can't be probed for.
  * No video bytes pass through this server. Supabase files get a signed
    link that expires (LINK_SECONDS) and asks the browser to save the
    file; Bunny files get the best MP4 rendition Bunny made (up to 1080p).
  * Downloading never publishes. Nothing here touches is_private,
    is_archived or any other post field.
  * Each link handed out is logged in creator_download_events (user,
    post, kind, time — nothing else). Logging is best effort, so the
    feature works before that table exists.
"""
from __future__ import annotations

import os
import re
import time
import urllib.parse
from collections import defaultdict, deque
from typing import Callable, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/creator", tags=["creator_downloads"])

LINK_SECONDS = 60 * 60          # long enough for a big file on a slow phone connection
LINKS_PER_WINDOW = 60           # links one creator may request...
LINK_WINDOW_SECONDS = 10 * 60   # ...per 10 minutes (stops scripted bulk pulling)
_PUBLIC_PATH = re.compile(r"/storage/v1/object/(?:public|sign)/([^/]+)/(.+)$")

_admin = None
_link_log: dict[str, deque] = defaultdict(deque)


def _get_admin():
    global _admin
    if _admin is None:
        url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if url and key:
            from supabase import create_client
            _admin = create_client(url, key)
    return _admin


async def _creator(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest
    return await get_current_user_id_no_guest(authorization)


# ---------------------------------------------------------------- data access
# Small functions so tests can swap them for fakes.

def load_post(post_id: str) -> Optional[dict]:
    admin = _get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Downloads are not configured.")
    res = admin.table("posts").select("*").eq("id", post_id).limit(1).execute()
    return (res.data or [None])[0]


def load_owned(user_id: str) -> tuple[list[dict], list[dict]]:
    """(series, video posts) owned by user_id, newest first."""
    admin = _get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Downloads are not configured.")
    series = (admin.table("series").select("*").eq("user_id", user_id)
              .order("created_at", desc=True).execute().data or [])
    posts = (admin.table("posts").select("*").eq("user_id", user_id).eq("post_type", "video")
             .order("created_at", desc=True).limit(500).execute().data or [])
    return series, posts


def sign_storage(bucket: str, path: str, seconds: int, file_name: str) -> str:
    admin = _get_admin()
    if admin is None:
        raise HTTPException(status_code=503, detail="Downloads are not configured.")
    r = admin.storage.from_(bucket).create_signed_url(path, seconds, {"download": file_name})
    url = r.get("signedURL") or r.get("signedUrl")
    if not url:
        raise RuntimeError("storage returned no signed link")
    return url


def bunny_info(video_id: str) -> dict:
    import bunny_stream
    return bunny_stream.get_video_info(video_id)


def log_event(user_id: str, post_id: Optional[str], kind: str) -> None:
    admin = _get_admin()
    if admin is None:
        return
    try:
        admin.table("creator_download_events").insert(
            {"user_id": user_id, "post_id": post_id, "kind": kind}).execute()
    except Exception as e:  # table not created yet, or a blip: never block the download
        print(f"[WARN] download event not logged: {e}")


# ---------------------------------------------------------------- helpers

def _rate_limit(user_id: str) -> None:
    now = time.time()
    q = _link_log[user_id]
    while q and now - q[0] > LINK_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= LINKS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many downloads at once. Wait a few minutes and try again.")
    q.append(now)


def _owned_post(user_id: str, post_id: str) -> dict:
    post = load_post(post_id) if re.fullmatch(r"[0-9a-fA-F-]{36}", post_id or "") else None
    if not post or post.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Production not found.")
    return post


def storage_location(url: Optional[str]) -> Optional[tuple[str, str]]:
    """(bucket, path) for a Supabase Storage URL on this project, else None."""
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    base = urllib.parse.urlparse(os.environ.get("SUPABASE_URL", ""))
    if base.netloc and parsed.netloc != base.netloc:
        return None
    m = _PUBLIC_PATH.search(parsed.path)
    if not m:
        return None
    return m.group(1), urllib.parse.unquote(m.group(2))


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip()).strip("-").lower()[:60]
    return s or fallback


def file_name(post: dict, series_title: Optional[str], ext: str) -> str:
    if post.get("series_id") and post.get("episode_number") is not None:
        base = f"{_slug(series_title or '', 'viyo-series')}-ep{int(post['episode_number']):02d}"
    else:
        base = _slug((post.get("caption") or "").split("\n")[0], "viyo-video")
    # A short id keeps two videos with the same caption from colliding (the
    # phone resumes a partial download by file name).
    return f"{base}-{str(post.get('id', ''))[-6:]}.{ext}"


def _orientation(width: Optional[int], height: Optional[int]) -> Optional[str]:
    if not width or not height:
        return None
    if height > width:
        return "9:16 vertical"
    if width > height:
        return "16:9 landscape"
    return "1:1 square"


def _ext(url: str, default: str) -> str:
    m = re.search(r"\.([A-Za-z0-9]{2,4})$", urllib.parse.urlparse(url).path)
    return m.group(1).lower() if m else default


# ---------------------------------------------------------------- listing

class EpisodeOut(BaseModel):
    post_id: str
    episode_number: Optional[int] = None
    title: str
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    is_private: bool = False
    is_archived: bool = False
    watch_status: str              # public | private | archived
    downloadable: bool
    not_downloadable_reason: Optional[str] = None


class ProductionOut(BaseModel):
    id: str
    kind: str                      # series | video
    title: str
    content_type: Optional[str] = None
    orientation: Optional[str] = None
    status: Optional[str] = None
    thumbnail_url: Optional[str] = None
    episode_count: int
    public_episode_count: int
    episodes: list[EpisodeOut]


class ProductionsResponse(BaseModel):
    productions: list[ProductionOut]


def _watch_status(post: dict) -> str:
    if post.get("is_archived"):
        return "archived"
    return "private" if post.get("is_private") else "public"


def _download_state(post: dict) -> tuple[bool, Optional[str]]:
    if post.get("video_provider") == "bunny" and post.get("bunny_video_id"):
        # The stored status only flips to "ready" when someone views it, so
        # the download itself asks Bunny for the real state.
        if post.get("video_status") == "failed":
            return False, "Processing failed"
        return True, None
    if storage_location(post.get("media_url")):
        return True, None
    return False, "No stored video file"


def _episode(post: dict) -> EpisodeOut:
    ok, why = _download_state(post)
    title = (post.get("caption") or "").split("\n")[0].strip()
    if post.get("episode_number") is not None:
        title = title or f"Episode {post['episode_number']}"
    return EpisodeOut(
        post_id=post["id"], episode_number=post.get("episode_number"), title=title or "Untitled video",
        thumbnail_url=post.get("thumbnail_url"), duration_seconds=post.get("duration_seconds"),
        is_private=bool(post.get("is_private")), is_archived=bool(post.get("is_archived")),
        watch_status=_watch_status(post), downloadable=ok, not_downloadable_reason=why)


def build_productions(series_rows: list[dict], posts: list[dict]) -> list[ProductionOut]:
    by_series: dict[str, list[dict]] = defaultdict(list)
    standalone = []
    for p in posts:
        (by_series[p["series_id"]] if p.get("series_id") else standalone).append(p)
    out = []
    for s in series_rows:
        eps = sorted(by_series.get(s["id"], []), key=lambda p: (p.get("episode_number") is None, p.get("episode_number") or 0))
        items = [_episode(p) for p in eps]
        out.append(ProductionOut(
            id=s["id"], kind="series", title=s.get("title") or "Untitled", content_type=s.get("content_type"),
            orientation=s.get("orientation"), status=s.get("status"), thumbnail_url=s.get("cover_image_url"),
            episode_count=len(items), public_episode_count=sum(1 for e in items if e.watch_status == "public"),
            episodes=items))
    for p in standalone:
        e = _episode(p)
        out.append(ProductionOut(
            id=p["id"], kind="video", title=e.title, thumbnail_url=p.get("thumbnail_url"),
            episode_count=1, public_episode_count=1 if e.watch_status == "public" else 0, episodes=[e]))
    return out


@router.get("/productions", response_model=ProductionsResponse)
async def my_productions(user_id: str = Depends(_creator)):
    series_rows, posts = load_owned(user_id)
    return ProductionsResponse(productions=build_productions(series_rows, posts))


# ---------------------------------------------------------------- download links

class DownloadLink(BaseModel):
    url: str
    file_name: str
    kind: str                      # video | thumbnail
    source: str                    # supabase | bunny
    expires_at: Optional[int] = None   # unix time; None = Bunny CDN link (see docstring)
    quality: Optional[str] = None      # e.g. "1080p"
    orientation: Optional[str] = None
    size_bytes: Optional[int] = None
    note: Optional[str] = None


def _series_title(post: dict) -> Optional[str]:
    if not post.get("series_id"):
        return None
    admin = _get_admin()
    if admin is None:
        return None
    try:
        rows = admin.table("series").select("title").eq("id", post["series_id"]).limit(1).execute().data or []
        return rows[0]["title"] if rows else None
    except Exception:
        return None


def video_link(post: dict, series_title: Optional[str]) -> DownloadLink:
    if post.get("video_provider") == "bunny" and post.get("bunny_video_id"):
        import bunny_stream
        try:
            info = bunny_info(post["bunny_video_id"])
        except bunny_stream.BunnyVideoNotFound:
            raise HTTPException(status_code=410, detail="The stored video file is missing. Contact support.")
        except bunny_stream.BunnyError:
            raise HTTPException(status_code=502, detail="Video storage didn't answer. Try again in a minute.")
        if bunny_stream.is_failed(info):
            raise HTTPException(status_code=409, detail="This video failed to process, so there's no file to download.")
        if not bunny_stream.is_finished(info):
            raise HTTPException(status_code=409, detail="This video is still processing. Try again in a few minutes.")
        url = bunny_stream.source_download_url(post["bunny_video_id"], info)
        res = re.search(r"play_(\d+)p\.mp4", url)
        return DownloadLink(
            url=url, file_name=file_name(post, series_title, "mp4"), kind="video", source="bunny",
            quality=f"{res.group(1)}p" if res else None,
            orientation=_orientation(info.get("width"), info.get("height")),
            note="Best MP4 quality stored for this video.")
    loc = storage_location(post.get("media_url"))
    if not loc:
        raise HTTPException(status_code=409, detail="This post has no stored video file to download.")
    name = file_name(post, series_title, _ext(post["media_url"], "mp4"))
    try:
        url = sign_storage(loc[0], loc[1], LINK_SECONDS, name)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Couldn't prepare the download. Try again.")
    return DownloadLink(url=url, file_name=name, kind="video", source="supabase",
                        expires_at=int(time.time()) + LINK_SECONDS, quality="Original upload",
                        note="The file exactly as it was uploaded.")


def thumbnail_link(post: dict, series_title: Optional[str]) -> DownloadLink:
    thumb = post.get("thumbnail_url")
    if not thumb:
        raise HTTPException(status_code=404, detail="This video has no thumbnail.")
    name = file_name(post, series_title, _ext(thumb, "jpg")).rsplit(".", 1)
    name = f"{name[0]}-thumbnail.{name[1]}"
    loc = storage_location(thumb)
    if loc:
        try:
            url = sign_storage(loc[0], loc[1], LINK_SECONDS, name)
        except Exception:
            raise HTTPException(status_code=502, detail="Couldn't prepare the download. Try again.")
        return DownloadLink(url=url, file_name=name, kind="thumbnail", source="supabase",
                            expires_at=int(time.time()) + LINK_SECONDS)
    return DownloadLink(url=thumb, file_name=name, kind="thumbnail", source="bunny")


@router.post("/downloads/{post_id}", response_model=DownloadLink)
async def download_video(post_id: str, user_id: str = Depends(_creator)):
    _rate_limit(user_id)
    post = _owned_post(user_id, post_id)
    link = video_link(post, _series_title(post))
    log_event(user_id, post["id"], "video")
    return link


@router.post("/downloads/{post_id}/thumbnail", response_model=DownloadLink)
async def download_thumbnail(post_id: str, user_id: str = Depends(_creator)):
    _rate_limit(user_id)
    post = _owned_post(user_id, post_id)
    link = thumbnail_link(post, _series_title(post))
    log_event(user_id, post["id"], "thumbnail")
    return link
