"""
Where VIYO keeps its files: Supabase Storage (default) or Bunny Storage.

Every place that used `<supabase client>.storage.from_(bucket)` now calls
`viyo_files.from_(<client>, bucket)`. With FILE_STORE=bunny (and the Bunny
settings below filled in) that returns a Bunny bucket with the same methods
the code already uses; otherwise it returns the Supabase bucket exactly as
before, so nothing changes until the switch is flipped.

Bunny layout: two storage zones, each with its own pull zone (CDN).
  * public zone: files anyone may see by URL (posts, avatars, covers, clips).
    URL: https://<BUNNY_PUBLIC_CDN>/<bucket>/<path>
  * private zone: creators' working files, selfies. Its pull zone has Token
    Authentication on, so a link only works with a token that expires.
    URL: https://<BUNNY_PRIVATE_CDN>/<bucket>/<path>?token=...&expires=...
The Supabase bucket name is kept as the first folder, so a file's path is the
same in both places and a Supabase public URL becomes a Bunny URL by swapping
the prefix (see bunny_files_rewrite.sql).

Settings (server only; the keys never reach the app):
  FILE_STORE                 "bunny" to switch; anything else = Supabase
  BUNNY_STORAGE_HOST         storage API host of the zones' main region,
                             e.g. jh.storage.bunnycdn.com (Johannesburg)
  BUNNY_PUBLIC_ZONE / BUNNY_PUBLIC_KEY / BUNNY_PUBLIC_CDN
  BUNNY_PRIVATE_ZONE / BUNNY_PRIVATE_KEY / BUNNY_PRIVATE_CDN
  BUNNY_PRIVATE_TOKEN_KEY    the private pull zone's token authentication key
  FILE_STORE_STRICT          "true": files go to Bunny or nowhere. Never Supabase:
                             if Bunny isn't switched on and complete, every file
                             read/write stops with StorageNotConfigured (naming the
                             missing settings, never their values) instead of
                             falling back.
Without FILE_STORE_STRICT, if FILE_STORE=bunny but a setting is missing, files keep
going to Supabase and a warning is logged.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import time
import urllib.parse
from typing import Any, Iterable, Optional

import requests

log = logging.getLogger("viyo_files")

PUBLIC_BUCKETS = frozenset({"posts-media", "avatars", "studio-images", "studio-audio", "studio-videos",
                            "processed-videos"})
# Everything else (creator-studio, repurpose-sources, glow-up-photos, glow-up-previews, and any bucket not
# listed) is private: an unknown bucket is never made public by accident.

CONNECT_TIMEOUT = 15
TRANSFER_TIMEOUT = 600
_SUPABASE_PATH = re.compile(r"/storage/v1/object/(?:public|sign|authenticated)/([^/]+)/(.+)$")


class FileNotFound(Exception):
    """The file isn't there (message says "not found", which older callers check for)."""


class FileStoreError(Exception):
    """Bunny refused or failed a request."""


# ---------------------------------------------------------------- settings

def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _host(value: str) -> str:
    """Accepts "x.b-cdn.net" or "https://x.b-cdn.net/" and returns the bare host."""
    value = value.strip()
    if "://" in value:
        value = urllib.parse.urlparse(value).netloc
    return value.strip("/")


class Zone:
    def __init__(self, name: str, key: str, cdn: str, token_key: str = ""):
        self.name, self.key, self.cdn, self.token_key = name, key, _host(cdn), token_key


_warned = False


NEEDED = ["BUNNY_PUBLIC_ZONE", "BUNNY_PUBLIC_KEY", "BUNNY_PUBLIC_CDN", "BUNNY_PRIVATE_ZONE",
          "BUNNY_PRIVATE_KEY", "BUNNY_PRIVATE_CDN", "BUNNY_PRIVATE_TOKEN_KEY"]


class StorageNotConfigured(RuntimeError):
    """Strict mode: Bunny isn't fully set up, so the file is refused rather than sent to Supabase."""


def strict() -> bool:
    return _env("FILE_STORE_STRICT").lower() in ("1", "true", "yes", "on")


def missing_settings() -> list[str]:
    """Names (never values) of what Bunny Storage still needs."""
    out = [] if _env("FILE_STORE").lower() == "bunny" else ["FILE_STORE=bunny"]
    return out + [n for n in NEEDED if not _env(n)]


def status() -> dict:
    """Where files go right now (for the admin check): no secrets, only names."""
    import bunny_stream
    files = "bunny" if settings() is not None else ("blocked" if strict() else "supabase")
    films = "bunny_stream" if bunny_stream._configured() else files
    return {"files": files, "films": films, "strict": strict(), "missing": missing_settings(),
            "films_missing": [] if bunny_stream._configured() else
            [n for n in ("BUNNY_STREAM_API_KEY", "BUNNY_STREAM_LIBRARY_ID", "BUNNY_STREAM_PULL_ZONE") if not _env(n)],
            "supabase_receives_media": files == "supabase" or films == "supabase"}


def settings(require_switch: bool = True) -> Optional[tuple[str, Zone, Zone]]:
    """(storage host, public zone, private zone) when Bunny is switched on and complete, else None.
    require_switch=False: the Bunny settings alone are enough (used to copy old files before switching)."""
    global _warned
    if require_switch and _env("FILE_STORE").lower() != "bunny":
        return None
    missing = [n for n in NEEDED if not _env(n)]
    if missing:
        if not _warned:
            log.warning("FILE_STORE=bunny but %s not set: files stay on Supabase", ", ".join(missing))
            _warned = True
        return None
    host = _host(_env("BUNNY_STORAGE_HOST") or "storage.bunnycdn.com")
    return (host,
            Zone(_env("BUNNY_PUBLIC_ZONE"), _env("BUNNY_PUBLIC_KEY"), _env("BUNNY_PUBLIC_CDN")),
            Zone(_env("BUNNY_PRIVATE_ZONE"), _env("BUNNY_PRIVATE_KEY"), _env("BUNNY_PRIVATE_CDN"),
                 _env("BUNNY_PRIVATE_TOKEN_KEY")))


def using_bunny() -> bool:
    return settings() is not None


def is_public(bucket: str) -> bool:
    return bucket in PUBLIC_BUCKETS


# ---------------------------------------------------------------- URLs

def _quote(path: str) -> str:
    return urllib.parse.quote(path.lstrip("/"), safe="/-_.~")


def sign_token(token_key: str, url_path: str, expires: int) -> str:
    """Bunny Token Authentication (SHA-256): base64url(sha256(key + path + expires)), no padding."""
    digest = hashlib.sha256(f"{token_key}{url_path}{expires}".encode()).digest()
    return base64.b64encode(digest).decode().replace("+", "-").replace("/", "_").replace("=", "")


def public_url(cdn: str, bucket: str, path: str) -> str:
    return f"https://{cdn}/{_quote(bucket + '/' + path)}"


def signed_url(zone: Zone, bucket: str, path: str, seconds: int, now: Optional[float] = None) -> str:
    url_path = "/" + _quote(bucket + "/" + path)
    expires = int(now if now is not None else time.time()) + int(seconds)
    return f"https://{zone.cdn}{url_path}?token={sign_token(zone.token_key, url_path, expires)}&expires={expires}"


def parse_url(url: Optional[str], supabase_url: Optional[str] = None) -> Optional[tuple[str, str]]:
    """(bucket, path) for a file URL of this project, on Supabase or Bunny; None for anything else."""
    if not url or not isinstance(url, str):
        return None
    parsed = urllib.parse.urlparse(url)
    base = urllib.parse.urlparse(supabase_url if supabase_url is not None else _env("SUPABASE_URL"))
    m = _SUPABASE_PATH.search(parsed.path)
    if m and (not base.netloc or parsed.netloc == base.netloc):
        return m.group(1), urllib.parse.unquote(m.group(2))
    cdns = {_host(_env("BUNNY_PUBLIC_CDN")), _host(_env("BUNNY_PRIVATE_CDN"))} - {""}
    if parsed.netloc in cdns:
        bucket, _, path = parsed.path.lstrip("/").partition("/")
        if bucket and path:
            return bucket, urllib.parse.unquote(path)
    return None


def file_hosts(supabase_url: Optional[str] = None) -> set[str]:
    """Hosts the backend may download its own files from."""
    hosts = {urllib.parse.urlparse(supabase_url if supabase_url is not None else _env("SUPABASE_URL")).hostname,
             _host(_env("BUNNY_PUBLIC_CDN")), _host(_env("BUNNY_PRIVATE_CDN"))}
    return {h for h in hosts if h}


# ---------------------------------------------------------------- Bunny bucket

def _content_type(options: Any) -> Optional[str]:
    if isinstance(options, dict):
        return options.get("content-type") or options.get("contentType") or options.get("content_type")
    return None


class BunnyBucket:
    """The subset of supabase-py's bucket API that VIYO uses, on Bunny Storage."""

    def __init__(self, bucket: str, host: str, public: Zone, private: Zone, http=None):
        self.bucket = bucket
        self.public = is_public(bucket)
        self.zone = public if self.public else private
        self.host = host
        self.http = http or requests

    # -- helpers
    def _api(self, path: str) -> str:
        return f"https://{self.host}/{self.zone.name}/{_quote(self.bucket + '/' + path)}"

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {"AccessKey": self.zone.key}
        h.update(extra or {})
        return h

    # -- writing
    def upload(self, path: str, file: Any, file_options: Any = None) -> dict:
        """file: bytes, a local file path (str), or an open file. Overwrites (Bunny has no "fail if exists")."""
        ctype = _content_type(file_options) or "application/octet-stream"
        opened = None
        if isinstance(file, (bytes, bytearray, memoryview)):
            body: Any = bytes(file)
        elif isinstance(file, (str, os.PathLike)):
            opened = open(file, "rb")
            body = opened
        else:
            body = file
        try:
            r = self.http.put(self._api(path), data=body, headers=self._headers({"Content-Type": ctype}),
                              timeout=(CONNECT_TIMEOUT, TRANSFER_TIMEOUT))
        finally:
            if opened:
                opened.close()
        if r.status_code not in (200, 201):
            raise FileStoreError(f"upload {self.bucket}/{path}: HTTP {r.status_code}")
        return {"path": path, "Key": f"{self.bucket}/{path}"}

    def remove(self, paths: Iterable[str]) -> list[dict]:
        removed = []
        for p in paths or []:
            r = self.http.delete(self._api(p), headers=self._headers(), timeout=(CONNECT_TIMEOUT, 60))
            if r.status_code in (200, 201, 204):
                removed.append({"name": p})
            elif r.status_code != 404:
                raise FileStoreError(f"remove {self.bucket}/{p}: HTTP {r.status_code}")
        return removed

    # -- reading
    def download(self, path: str, *args, **kwargs) -> bytes:
        r = self.http.get(self._api(path), headers=self._headers(), timeout=(CONNECT_TIMEOUT, TRANSFER_TIMEOUT))
        if r.status_code == 404:
            raise FileNotFound(f"{self.bucket}/{path} not found")
        if r.status_code != 200:
            raise FileStoreError(f"download {self.bucket}/{path}: HTTP {r.status_code}")
        return r.content

    def list(self, path: str = "", options: Any = None) -> list[dict]:
        """Entries directly under path. Folders have id None (as on Supabase); files have metadata.size."""
        folder = path.strip("/")
        url = f"https://{self.host}/{self.zone.name}/{_quote(self.bucket + ('/' + folder if folder else ''))}/"
        r = self.http.get(url, headers=self._headers({"Accept": "application/json"}),
                          timeout=(CONNECT_TIMEOUT, 60))
        if r.status_code == 404:
            return []
        if r.status_code != 200:
            raise FileStoreError(f"list {self.bucket}/{folder}: HTTP {r.status_code}")
        out = []
        for e in r.json() or []:
            name = e.get("ObjectName")
            if not name:
                continue
            if e.get("IsDirectory"):
                out.append({"name": name, "id": None, "metadata": None})
            else:
                out.append({"name": name, "id": e.get("Guid") or name,
                            "updated_at": e.get("LastChanged"),
                            "metadata": {"size": e.get("Length"), "mimetype": e.get("ContentType")}})
        limit = (options or {}).get("limit") if isinstance(options, dict) else None
        return out[:limit] if limit else out

    def info(self, path: str) -> dict:
        folder, _, name = path.strip("/").rpartition("/")
        for e in self.list(folder):
            if e["name"] == name and e.get("id") is not None:
                return {"name": name, "size": e["metadata"]["size"], "metadata": e["metadata"]}
        raise FileNotFound(f"{self.bucket}/{path} not found")

    # -- links
    def get_public_url(self, path: str, *args, **kwargs) -> str:
        return public_url(self.zone.cdn, self.bucket, path)

    def create_signed_url(self, path: str, expires_in: int, options: Any = None) -> dict:
        url = (public_url(self.zone.cdn, self.bucket, path) if self.public
               else signed_url(self.zone, self.bucket, path, expires_in))
        return {"signedURL": url, "signedUrl": url, "path": path}

    def create_signed_urls(self, paths: Iterable[str], expires_in: int, options: Any = None) -> list[dict]:
        out = []
        for p in paths or []:
            u = self.create_signed_url(p, expires_in)["signedURL"]
            out.append({"path": p, "signedURL": u, "signedUrl": u, "error": None})
        return out

    def create_signed_upload_url(self, path: str, *args, **kwargs) -> dict:
        raise FileStoreError("Direct uploads aren't available on Bunny Storage; upload through the server.")


def from_(client: Any, bucket: str):
    """The bucket to use: Bunny when switched on, else `client`'s Supabase bucket (unchanged behaviour)."""
    cfg = settings()
    if cfg is None:
        if strict():
            raise StorageNotConfigured("Files are stopped: Bunny Storage isn't fully set up (missing: "
                                       + ", ".join(missing_settings()) + "), and FILE_STORE_STRICT keeps them off "
                                       "Supabase.")
        return client.storage.from_(bucket)
    host, public, private = cfg
    return BunnyBucket(bucket, host, public, private)


# ---------------------------------------------------------------- moving old files

def list_all(store: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, Optional[int]]]:
    """Every (path, size) under prefix in a bucket, walking folders (Supabase pages 1000 at a time)."""
    out: list[tuple[str, Optional[int]]] = []
    offset = 0
    while True:
        page = store.list(prefix, {"limit": 1000, "offset": offset}) or []
        for f in page:
            name = f.get("name")
            if not name or name == ".emptyFolderPlaceholder":
                continue
            path = f"{prefix}/{name}" if prefix else name
            if f.get("id") is None:
                if depth < 8:
                    out += list_all(store, path, depth + 1)
            else:
                size = (f.get("metadata") or {}).get("size")
                out.append((path, int(size) if size is not None else None))
        if len(page) < 1000:
            return out
        offset += 1000


def copy_bucket(supabase_client: Any, bucket: str, dry_run: bool = True, delete_after: bool = False,
                progress=print) -> dict:
    """Copies every file of one Supabase bucket to Bunny at the same <bucket>/<path>, checking each copy's size.
    Files already on Bunny with the same size are skipped, so it can be run again safely. With delete_after,
    a Supabase file is removed only after its Bunny copy was checked (and only once FILE_STORE=bunny, so the
    app is already reading from Bunny)."""
    cfg = settings(require_switch=False)
    if cfg is None:
        raise FileStoreError("Bunny settings are incomplete (all BUNNY_* settings).")
    if delete_after and not using_bunny():
        raise FileStoreError("Deleting from Supabase needs FILE_STORE=bunny first.")
    host, public, private = cfg
    src = supabase_client.storage.from_(bucket)
    dst = BunnyBucket(bucket, host, public, private)
    files = list_all(src)
    report = {"bucket": bucket, "files": len(files), "bytes": sum(s or 0 for _, s in files),
              "copied": 0, "skipped": 0, "deleted": 0, "failed": []}
    if dry_run:
        return report
    existing: dict[str, dict] = {}
    for path, size in files:
        try:
            folder = path.rpartition("/")[0]
            if folder not in existing:
                existing[folder] = {e["name"]: e for e in dst.list(folder) if e.get("id") is not None}
            have = existing[folder].get(path.rpartition("/")[2])
            if have and size is not None and have["metadata"]["size"] == size:
                report["skipped"] += 1
            else:
                data = src.download(path)
                dst.upload(path, data, {"content-type": _guess_type(path)})
                check = dst.info(path)
                if int(check["size"]) != len(data):
                    raise FileStoreError(f"size check failed ({check['size']} vs {len(data)})")
                report["copied"] += 1
            if delete_after:
                src.remove([path])
                report["deleted"] += 1
        except Exception as e:
            report["failed"].append(f"{path}: {e}")
        if progress and (report["copied"] + report["skipped"]) % 100 == 0:
            progress(f"{bucket}: {report['copied']} copied, {report['skipped']} already there")
    return report


def _guess_type(path: str) -> str:
    import mimetypes
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


_URL_IN_TEXT = re.compile(r"https://[^\s\"'<>]+")


def find_files(text: Optional[str]) -> set[tuple[str, str]]:
    """Every (bucket, path) of this project's files linked anywhere in a piece of text."""
    out: set[tuple[str, str]] = set()
    for url in _URL_IN_TEXT.findall(text or ""):
        loc = parse_url(url)
        if loc:
            out.add(loc)
    return out


# ---------------------------------------------------------------- the one-time move, run by the worker

ALL_BUCKETS = ["posts-media", "avatars", "studio-images", "studio-audio", "studio-videos", "processed-videos",
               "creator-studio", "repurpose-sources", "glow-up-photos", "glow-up-previews"]


def move_all(supabase_client: Any, mode: str, progress=print) -> list[dict]:
    """mode "count", "copy" or "copy_and_delete". Returns one report per bucket."""
    reports = []
    for bucket in ALL_BUCKETS:
        try:
            r = copy_bucket(supabase_client, bucket, dry_run=(mode == "count"),
                            delete_after=(mode == "copy_and_delete"), progress=progress)
        except Exception as e:
            r = {"bucket": bucket, "error": str(e)}
        progress(f"[files] {mode} {r}" if r.get("error") else
                 f"[files] {mode} {bucket}: {r['files']} files, {r['bytes'] / 1048576:.1f} MB, copied {r['copied']}, "
                 f"already on Bunny {r['skipped']}, deleted from Supabase {r['deleted']}, failed {len(r['failed'])}"
                 + (f" (first: {r['failed'][0]})" if r["failed"] else ""))
        reports.append(r)
    return reports


def start_mover(supabase_client: Any) -> bool:
    """Runs the move once in the background when BUNNY_MOVE_FILES is "count", "copy" or "copy_and_delete".
    Progress and the result go to the worker's log. Running it again is safe (copied files are skipped)."""
    mode = _env("BUNNY_MOVE_FILES").lower()
    if mode not in ("count", "copy", "copy_and_delete") or supabase_client is None:
        return False
    import threading

    def run():
        try:
            move_all(supabase_client, mode)
            print(f"[files] {mode} finished")
        except Exception as e:
            print(f"[files] {mode} stopped: {e}")

    threading.Thread(target=run, name="bunny-file-move", daemon=True).start()
    return True
