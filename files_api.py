"""
Uploads from the app when files live on Bunny Storage.

On Supabase the app uploads photos straight to Storage with the user's own
session. Bunny has no per-user upload permission and its key must never
reach the app, so once FILE_STORE=bunny the app sends the file here and the
server stores it. GET /files/config tells the app which way to go, so
nothing changes in the app until the switch is flipped.
"""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

import viyo_files

router = APIRouter(prefix="/api/v1/files", tags=["files"])

IMAGE_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif",
               "image/heic": "heic"}
VIDEO_TYPES = {"video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
KINDS = {"post_media": "posts-media", "avatar": "avatars"}
DELETABLE_BUCKETS = {"posts-media", "avatars"}
_CHUNK = 1024 * 1024

_admin = None


def _client():
    """Supabase client for when files are still on Supabase (Bunny mode doesn't need it)."""
    global _admin
    if _admin is None:
        from main import supabase_admin
        _admin = supabase_admin
    return _admin


async def _user(authorization: str = Header(None)) -> str:
    from main import _decode_token
    uid = _decode_token(authorization).get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Token missing subject")
    return uid


@router.get("/config")
async def config():
    """Where the app should upload photos: "bunny" (through POST /files/upload) or "supabase" (directly)."""
    return {"store": "bunny" if viyo_files.using_bunny() else "supabase"}


async def _save_capped(file: UploadFile, max_bytes: int) -> str:
    fd, tmp = tempfile.mkstemp(prefix="viyo_upload_")
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413,
                                        detail=f"File too large. The limit is {max_bytes // (1024 * 1024)} MB.")
                out.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="The file is empty.")
        return tmp
    except BaseException:
        os.unlink(tmp)
        raise


@router.post("/upload")
async def upload(kind: str = Form(...), file: UploadFile = File(...), user_id: str = Depends(_user)):
    """Stores one photo (or, for a post, a video) under the user's own folder and returns its public URL."""
    if kind not in KINDS:
        raise HTTPException(status_code=400, detail="Unknown upload kind.")
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype in IMAGE_TYPES:
        ext, limit = IMAGE_TYPES[ctype], MAX_IMAGE_BYTES
    elif ctype in VIDEO_TYPES and kind == "post_media":
        ext, limit = VIDEO_TYPES[ctype], MAX_VIDEO_BYTES
    else:
        raise HTTPException(status_code=400, detail="That file type isn't supported. Use a JPG, PNG or WebP photo.")
    name = f"avatar_{int(time.time() * 1000)}" if kind == "avatar" else uuid.uuid4().hex
    path = f"{user_id}/{name}.{ext}"
    tmp = await _save_capped(file, limit)
    try:
        bucket = KINDS[kind]
        try:
            store = viyo_files.from_(_client(), bucket)
            store.upload(path, tmp, {"content-type": ctype, "upsert": "true"})
        except Exception:
            if kind != "avatar" or viyo_files.using_bunny():
                raise
            # Same fallback the app had: some projects never created an "avatars" bucket.
            bucket = "posts-media"
            store = viyo_files.from_(_client(), bucket)
            store.upload(path, tmp, {"content-type": ctype, "upsert": "true"})
        return {"url": store.get_public_url(path), "bucket": bucket, "path": path}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Couldn't save the file. Try again.")
    finally:
        os.unlink(tmp)


class DeleteIn(BaseModel):
    urls: list[str] = Field(..., max_length=20)


@router.post("/delete")
async def delete(body: DeleteIn, user_id: str = Depends(_user)):
    """Removes the user's own post or avatar files by URL. Anyone else's file, or another bucket, is skipped."""
    removed, skipped = 0, 0
    for url in body.urls:
        loc = viyo_files.parse_url(url)
        if not loc or loc[0] not in DELETABLE_BUCKETS or not loc[1].startswith(f"{user_id}/") or ".." in loc[1]:
            skipped += 1
            continue
        try:
            removed += len(viyo_files.from_(_client(), loc[0]).remove([loc[1]]) or [])
        except Exception:
            skipped += 1
    return {"removed": removed, "skipped": skipped}
