"""
Storage operations for the AI Repurposer, behind one small interface so
the worker can run against Supabase Storage in production and a local
directory in tests.

Source videos live in the PRIVATE `repurpose-sources` bucket. Creators
upload through a signed upload URL the backend issues for a path it
chose, and only the backend reads them back (through short-lived signed
download URLs). Generated clips go to the public `processed-videos`
bucket, as before, because the app plays and posts them by URL.
"""
from __future__ import annotations

import os
import shutil
import time
import urllib.parse
import urllib.request
from typing import Optional

SOURCE_BUCKET = "repurpose-sources"
CLIPS_BUCKET = "processed-videos"

DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 30
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_BUCKET_LIMIT_CACHE_SECONDS = 600


class SourceMissing(Exception):
    pass


class SourceTooLarge(Exception):
    pass


class StorageUnavailable(Exception):
    """Network/provider trouble — retryable."""


def stream_url_to_file(url: str, dest: str, max_bytes: int, allowed_host: Optional[str]) -> int:
    """
    Streams `url` to `dest`, enforcing `max_bytes` while downloading (not
    after), and refusing any URL — including a redirect target — that
    isn't on `allowed_host` over HTTPS. Returns bytes written.
    """
    def check(u: str) -> None:
        parsed = urllib.parse.urlparse(u)
        if allowed_host and (parsed.scheme != "https" or parsed.hostname != allowed_host):
            raise SourceMissing(f"refusing to download from {parsed.scheme}://{parsed.hostname}")

    check(url)
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_CONNECT_TIMEOUT_SECONDS) as resp:
            check(resp.geturl())
            length = resp.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise SourceTooLarge(f"{length} bytes")
            written = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise SourceTooLarge(f">{max_bytes} bytes")
                    out.write(chunk)
            return written
    except (SourceMissing, SourceTooLarge):
        raise
    except urllib.error.HTTPError as e:
        if e.code in (400, 403, 404):
            raise SourceMissing(f"HTTP {e.code}")
        raise StorageUnavailable(f"HTTP {e.code}")
    except Exception as e:
        raise StorageUnavailable(str(e))


class SupabaseStorage:
    def __init__(self, admin, supabase_url: str):
        self._admin = admin
        self._host = urllib.parse.urlparse(supabase_url).hostname
        self._limit_cache: dict[str, tuple[float, Optional[int]]] = {}

    def _bucket(self, bucket: str):
        return self._admin.storage.from_(bucket)

    def signed_upload(self, bucket: str, path: str) -> dict:
        data = self._bucket(bucket).create_signed_upload_url(path)
        return {"url": data["signed_url"], "token": data["token"]}

    def object_size(self, bucket: str, path: str) -> Optional[int]:
        """Size in bytes, or None if the object doesn't exist."""
        try:
            info = self._bucket(bucket).info(path)
        except Exception:
            return None
        if isinstance(info, list):
            info = info[0] if info else None
        if not isinstance(info, dict):
            return None
        size = info.get("size")
        if size is None and isinstance(info.get("metadata"), dict):
            size = info["metadata"].get("size")
        try:
            return int(size) if size is not None else -1
        except (TypeError, ValueError):
            return -1

    def bucket_file_size_limit(self, bucket: str) -> Optional[int]:
        cached = self._limit_cache.get(bucket)
        if cached and time.time() - cached[0] < _BUCKET_LIMIT_CACHE_SECONDS:
            return cached[1]
        try:
            limit = getattr(self._admin.storage.get_bucket(bucket), "file_size_limit", None)
            limit = int(limit) if limit else None
        except Exception:
            limit = None
        self._limit_cache[bucket] = (time.time(), limit)
        return limit

    def download(self, bucket: str, path: str, dest: str, max_bytes: int) -> int:
        try:
            signed = self._bucket(bucket).create_signed_url(path, 900)
        except Exception as e:
            message = str(e).lower()
            if "not found" in message or "404" in message:
                raise SourceMissing(str(e))
            raise StorageUnavailable(str(e))
        url = signed.get("signedURL") or signed.get("signedUrl")
        return stream_url_to_file(url, dest, max_bytes, self._host)

    def download_url(self, url: str, dest: str, max_bytes: int) -> int:
        return stream_url_to_file(url, dest, max_bytes, self._host)

    def upload(self, bucket: str, path: str, local_path: str, content_type: str) -> str:
        with open(local_path, "rb") as f:
            try:
                self._bucket(bucket).upload(
                    path, f, file_options={"content-type": content_type, "upsert": "true"})
            except Exception as e:
                raise StorageUnavailable(f"upload {path}: {e}")
        return self.public_url(bucket, path)

    def public_url(self, bucket: str, path: str) -> str:
        return self._bucket(bucket).get_public_url(path)

    def remove(self, bucket: str, paths: list[str]) -> None:
        if paths:
            self._bucket(bucket).remove(paths)


class LocalStorage:
    """Directory-backed stand-in with the same interface, for tests and
    local runs. Public URLs are file:// paths."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, bucket: str, path: str) -> str:
        full = os.path.normpath(os.path.join(self.root, bucket, path))
        if not full.startswith(os.path.normpath(self.root) + os.sep):
            raise ValueError("path escapes storage root")
        return full

    def put_source(self, bucket: str, path: str, local_path: str) -> None:
        dest = self._path(bucket, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(local_path, dest)

    def signed_upload(self, bucket: str, path: str) -> dict:
        return {"url": f"file://{self._path(bucket, path)}", "token": "local"}

    def object_size(self, bucket: str, path: str) -> Optional[int]:
        p = self._path(bucket, path)
        return os.path.getsize(p) if os.path.exists(p) else None

    def bucket_file_size_limit(self, bucket: str) -> Optional[int]:
        return None

    def download(self, bucket: str, path: str, dest: str, max_bytes: int) -> int:
        src = self._path(bucket, path)
        if not os.path.exists(src):
            raise SourceMissing(path)
        size = os.path.getsize(src)
        if size > max_bytes:
            raise SourceTooLarge(str(size))
        shutil.copyfile(src, dest)
        return size

    def download_url(self, url: str, dest: str, max_bytes: int) -> int:
        raise SourceMissing("LocalStorage has no URLs to download")

    def upload(self, bucket: str, path: str, local_path: str, content_type: str) -> str:
        dest = self._path(bucket, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(local_path, dest)
        return self.public_url(bucket, path)

    def public_url(self, bucket: str, path: str) -> str:
        return f"file://{self._path(bucket, path)}"

    def remove(self, bucket: str, paths: list[str]) -> None:
        for p in paths:
            full = self._path(bucket, p)
            if os.path.exists(full):
                os.remove(full)
