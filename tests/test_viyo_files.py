"""Files on Bunny Storage: the bucket shim, signed links, URL parsing, the move, and app uploads.
Bunny's HTTP API is replaced by an in-memory fake; nothing here talks to Bunny or Supabase."""
import base64
import hashlib
import io
import json
import urllib.parse

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

import viyo_files as vf

SUPA = "https://proj.supabase.co"
BUNNY_ENV = {
    "FILE_STORE": "bunny", "BUNNY_STORAGE_HOST": "jh.storage.bunnycdn.com",
    "BUNNY_PUBLIC_ZONE": "viyo-public", "BUNNY_PUBLIC_KEY": "pub-key", "BUNNY_PUBLIC_CDN": "https://pub.b-cdn.net/",
    "BUNNY_PRIVATE_ZONE": "viyo-private", "BUNNY_PRIVATE_KEY": "priv-key", "BUNNY_PRIVATE_CDN": "priv.b-cdn.net",
    "BUNNY_PRIVATE_TOKEN_KEY": "token-secret", "SUPABASE_URL": SUPA,
}


class Resp:
    def __init__(self, status, content=b"", data=None):
        self.status_code, self.content, self._data = status, content, data

    def json(self):
        return self._data


class FakeBunny:
    """Bunny Storage API in memory: PUT/GET/DELETE files, GET <folder>/ lists."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple] = []
        self.fail_put = False
        self.truncate = False

    def _key(self, url, headers):
        p = urllib.parse.urlparse(url)
        assert p.netloc == "jh.storage.bunnycdn.com"
        zone = p.path.lstrip("/").split("/")[0]
        assert headers["AccessKey"] == {"viyo-public": "pub-key", "viyo-private": "priv-key"}[zone]
        return urllib.parse.unquote(p.path)

    def put(self, url, data=None, headers=None, timeout=None):
        key = self._key(url, headers)
        body = data if isinstance(data, bytes) else data.read()
        self.calls.append(("put", key, headers.get("Content-Type")))
        if self.fail_put:
            return Resp(500)
        self.files[key] = body[:-1] if self.truncate else body
        return Resp(201)

    def get(self, url, headers=None, timeout=None):
        key = self._key(url, headers)
        if key.endswith("/"):
            entries, seen = [], set()
            for k, v in self.files.items():
                if k.startswith(key):
                    rest = k[len(key):]
                    name, _, more = rest.partition("/")
                    if name in seen:
                        continue
                    seen.add(name)
                    entries.append({"ObjectName": name, "IsDirectory": bool(more), "Length": 0 if more else len(v)})
            return Resp(200, data=entries)
        if key not in self.files:
            return Resp(404)
        return Resp(200, content=self.files[key])

    def delete(self, url, headers=None, timeout=None):
        key = self._key(url, headers)
        self.calls.append(("delete", key))
        return Resp(200) if self.files.pop(key, None) is not None else Resp(404)


@pytest.fixture
def bunny(monkeypatch):
    for k, v in BUNNY_ENV.items():
        monkeypatch.setenv(k, v)
    fake = FakeBunny()
    monkeypatch.setattr(vf, "requests", fake)
    return fake


class FakeSupabaseBucket:
    def __init__(self, files):
        self.files = files       # path -> bytes
        self.removed = []

    def list(self, prefix, opts=None):
        prefix = prefix.strip("/")
        out, seen = [], set()
        for k, v in sorted(self.files.items()):
            if prefix and not k.startswith(prefix + "/"):
                continue
            rest = k[len(prefix) + 1:] if prefix else k
            name, _, more = rest.partition("/")
            if name in seen:
                continue
            seen.add(name)
            out.append({"name": name, "id": None, "metadata": None} if more
                       else {"name": name, "id": "x", "metadata": {"size": len(v)}})
        offset = (opts or {}).get("offset", 0)
        return out[offset:offset + (opts or {}).get("limit", 1000)]

    def download(self, path):
        return self.files[path]

    def remove(self, paths):
        for p in paths:
            self.removed.append(p)
            self.files.pop(p, None)
        return [{"name": p} for p in paths]


class FakeSupabase:
    def __init__(self, buckets):
        self.buckets = {b: FakeSupabaseBucket(f) for b, f in buckets.items()}

    @property
    def storage(self):
        return self

    def from_(self, bucket):
        return self.buckets.setdefault(bucket, FakeSupabaseBucket({}))


# ---------------------------------------------------------------- switching

def test_supabase_stays_until_switched(monkeypatch):
    for k in BUNNY_ENV:
        monkeypatch.delenv(k, raising=False)
    client = FakeSupabase({"posts-media": {}})
    assert vf.from_(client, "posts-media") is client.buckets["posts-media"]
    assert not vf.using_bunny()


def test_half_configured_bunny_keeps_supabase(monkeypatch):
    for k, v in BUNNY_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("BUNNY_PRIVATE_TOKEN_KEY")
    client = FakeSupabase({"posts-media": {}})
    assert vf.from_(client, "posts-media") is client.buckets["posts-media"]
    assert vf.settings(require_switch=False) is None


def test_switched_on_returns_bunny(bunny):
    b = vf.from_(object(), "posts-media")
    assert isinstance(b, vf.BunnyBucket) and b.public and b.zone.name == "viyo-public"
    p = vf.from_(object(), "creator-studio")
    assert not p.public and p.zone.name == "viyo-private"
    # An unknown bucket is private, never public by accident.
    assert not vf.from_(object(), "something-new").public


# ---------------------------------------------------------------- bucket operations

def test_upload_bytes_path_and_file(bunny, tmp_path):
    b = vf.from_(None, "posts-media")
    b.upload("u1/a.jpg", b"img", {"content-type": "image/jpeg"})
    f = tmp_path / "x.mp4"
    f.write_bytes(b"video")
    b.upload("u1/b.mp4", str(f), file_options={"content-type": "video/mp4", "upsert": "true"})
    b.upload("u1/c.png", io.BytesIO(b"png"), {"content-type": "image/png"})
    assert bunny.files == {"/viyo-public/posts-media/u1/a.jpg": b"img", "/viyo-public/posts-media/u1/b.mp4": b"video",
                           "/viyo-public/posts-media/u1/c.png": b"png"}
    assert ("put", "/viyo-public/posts-media/u1/a.jpg", "image/jpeg") in bunny.calls


def test_upload_failure_raises(bunny):
    bunny.fail_put = True
    with pytest.raises(vf.FileStoreError):
        vf.from_(None, "posts-media").upload("u1/a.jpg", b"img")


def test_download_list_info_remove(bunny):
    b = vf.from_(None, "creator-studio")
    b.upload("u1/p1/image/a.png", b"aaaa")
    b.upload("u1/p1/video/b.mp4", b"bb")
    assert b.download("u1/p1/image/a.png") == b"aaaa"
    with pytest.raises(vf.FileNotFound, match="not found"):
        b.download("u1/missing.png")
    top = b.list("u1/p1")
    assert {e["name"] for e in top} == {"image", "video"} and all(e["id"] is None for e in top)
    files = b.list("u1/p1/image")
    assert files[0]["name"] == "a.png" and files[0]["metadata"]["size"] == 4
    assert b.info("u1/p1/video/b.mp4")["size"] == 2
    assert b.list("nobody") == []
    assert b.remove(["u1/p1/image/a.png", "u1/not-there.png"]) == [{"name": "u1/p1/image/a.png"}]
    assert "/viyo-private/creator-studio/u1/p1/image/a.png" not in bunny.files


def test_signed_upload_not_offered(bunny):
    with pytest.raises(vf.FileStoreError):
        vf.from_(None, "repurpose-sources").create_signed_upload_url("u1/x.mp4")


# ---------------------------------------------------------------- links

def test_public_bucket_links_are_plain_cdn_urls(bunny):
    b = vf.from_(None, "posts-media")
    assert b.get_public_url("u1/a b.jpg") == "https://pub.b-cdn.net/posts-media/u1/a%20b.jpg"
    assert b.create_signed_url("u1/a.jpg", 60)["signedURL"] == "https://pub.b-cdn.net/posts-media/u1/a.jpg"


def test_private_links_carry_a_bunny_token(bunny):
    zone = vf.Zone("viyo-private", "k", "priv.b-cdn.net", "token-secret")
    url = vf.signed_url(zone, "creator-studio", "u1/p/a.png", 3600, now=1_000_000)
    path = "/creator-studio/u1/p/a.png"
    expected = base64.b64encode(hashlib.sha256(f"token-secret{path}1003600".encode()).digest()).decode()
    expected = expected.replace("+", "-").replace("/", "_").replace("=", "")
    assert url == f"https://priv.b-cdn.net{path}?token={expected}&expires=1003600"

    b = vf.from_(None, "creator-studio")
    many = b.create_signed_urls(["u1/a.png", "u1/b.png"], 600)
    assert [m["path"] for m in many] == ["u1/a.png", "u1/b.png"]
    assert all("token=" in m["signedURL"] and m["signedURL"] == m["signedUrl"] for m in many)
    single = b.create_signed_url("u1/a.png", 600, {"download": "film.mp4"})
    assert single["signedURL"].startswith("https://priv.b-cdn.net/creator-studio/u1/a.png?token=")


def test_parse_url_understands_both_stores(bunny):
    assert vf.parse_url(f"{SUPA}/storage/v1/object/public/posts-media/u1/a%20b.jpg") == ("posts-media", "u1/a b.jpg")
    assert vf.parse_url(f"{SUPA}/storage/v1/object/sign/creator-studio/u1/a.png?token=x") == ("creator-studio",
                                                                                            "u1/a.png")
    assert vf.parse_url("https://pub.b-cdn.net/posts-media/u1/a.jpg") == ("posts-media", "u1/a.jpg")
    assert vf.parse_url("https://priv.b-cdn.net/creator-studio/u1/a.png?token=t&expires=1") == ("creator-studio",
                                                                                                "u1/a.png")
    assert vf.parse_url("https://evil.example/storage/v1/object/public/posts-media/u1/a.jpg") is None
    assert vf.parse_url("https://other.b-cdn.net/posts-media/u1/a.jpg") is None
    assert vf.parse_url(None) is None
    text = 'cover "https://pub.b-cdn.net/studio-images/u1/c.jpg" and ' + f"{SUPA}/storage/v1/object/public/avatars/u1/x.png"
    assert vf.find_files(text) == {("studio-images", "u1/c.jpg"), ("avatars", "u1/x.png")}
    assert vf.file_hosts() == {"proj.supabase.co", "pub.b-cdn.net", "priv.b-cdn.net"}


# ---------------------------------------------------------------- moving old files

def test_count_copy_rerun_and_delete(bunny, monkeypatch):
    supa = FakeSupabase({"posts-media": {"u1/a.jpg": b"aaa", "u1/sub/b.jpg": b"bb"},
                         "creator-studio": {"u1/p/image/c.png": b"c"}})
    count = vf.copy_bucket(supa, "posts-media", dry_run=True)
    assert (count["files"], count["bytes"], count["copied"]) == (2, 5, 0) and not bunny.files

    r = vf.copy_bucket(supa, "posts-media", dry_run=False, progress=None)
    assert (r["copied"], r["skipped"], r["failed"]) == (2, 0, [])
    assert bunny.files["/viyo-public/posts-media/u1/sub/b.jpg"] == b"bb"
    again = vf.copy_bucket(supa, "posts-media", dry_run=False, progress=None)
    assert (again["copied"], again["skipped"]) == (0, 2)

    r = vf.copy_bucket(supa, "creator-studio", dry_run=False, delete_after=True, progress=None)
    assert r["copied"] == 1 and r["deleted"] == 1
    assert bunny.files["/viyo-private/creator-studio/u1/p/image/c.png"] == b"c"
    assert supa.buckets["creator-studio"].files == {}


def test_copy_works_before_the_switch_but_delete_needs_it(bunny, monkeypatch):
    monkeypatch.delenv("FILE_STORE")
    supa = FakeSupabase({"avatars": {"u1/x.png": b"x"}})
    assert vf.copy_bucket(supa, "avatars", dry_run=False, progress=None)["copied"] == 1
    with pytest.raises(vf.FileStoreError, match="FILE_STORE=bunny"):
        vf.copy_bucket(supa, "avatars", dry_run=False, delete_after=True, progress=None)
    assert supa.buckets["avatars"].files == {"u1/x.png": b"x"}


def test_a_bad_copy_is_never_deleted(bunny):
    bunny.truncate = True
    supa = FakeSupabase({"posts-media": {"u1/a.jpg": b"aaa"}})
    r = vf.copy_bucket(supa, "posts-media", dry_run=False, delete_after=True, progress=None)
    assert r["deleted"] == 0 and len(r["failed"]) == 1 and "size check" in r["failed"][0]
    assert supa.buckets["posts-media"].files == {"u1/a.jpg": b"aaa"}


def test_failed_upload_keeps_the_original(bunny):
    bunny.fail_put = True
    supa = FakeSupabase({"posts-media": {"u1/a.jpg": b"aaa"}})
    r = vf.copy_bucket(supa, "posts-media", dry_run=False, delete_after=True, progress=None)
    assert r["deleted"] == 0 and r["failed"]
    assert supa.buckets["posts-media"].removed == []


def test_mover_only_runs_when_asked(bunny, monkeypatch):
    assert vf.start_mover(FakeSupabase({})) is False
    monkeypatch.setenv("BUNNY_MOVE_FILES", "delete-everything")
    assert vf.start_mover(FakeSupabase({})) is False
    reports = vf.move_all(FakeSupabase({"posts-media": {"u/a.jpg": b"a"}}), "count", progress=lambda *_: None)
    assert [r["bucket"] for r in reports] == vf.ALL_BUCKETS
    assert reports[0]["files"] == 1 and not bunny.files


# ---------------------------------------------------------------- app uploads

@pytest.fixture
def api(bunny, monkeypatch):
    import files_api
    monkeypatch.setattr(files_api, "_client", lambda: None)
    app = FastAPI()
    app.include_router(files_api.router)

    async def fake_user(authorization: str = Header(None)):
        if not authorization:
            raise HTTPException(status_code=401)
        return authorization.removeprefix("Bearer ")

    app.dependency_overrides[files_api._user] = fake_user
    return TestClient(app)


def test_config_tells_the_app_where_to_upload(api, monkeypatch):
    assert api.get("/api/v1/files/config").json() == {"store": "bunny"}
    monkeypatch.delenv("FILE_STORE")
    assert api.get("/api/v1/files/config").json() == {"store": "supabase"}


def test_upload_goes_to_the_users_own_folder(api, bunny):
    r = api.post("/api/v1/files/upload", headers={"Authorization": "Bearer u1"}, data={"kind": "post_media"},
                 files={"file": ("x.jpg", b"jpegbytes", "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["bucket"] == "posts-media" and body["path"].startswith("u1/") and body["path"].endswith(".jpg")
    assert body["url"] == f"https://pub.b-cdn.net/posts-media/{body['path']}"
    assert bunny.files[f"/viyo-public/posts-media/{body['path']}"] == b"jpegbytes"

    av = api.post("/api/v1/files/upload", headers={"Authorization": "Bearer u1"}, data={"kind": "avatar"},
                  files={"file": ("a.png", b"png", "image/png")}).json()
    assert av["bucket"] == "avatars" and av["path"].startswith("u1/avatar_")


def test_upload_refuses_bad_files(api, monkeypatch):
    import files_api
    h = {"Authorization": "Bearer u1"}
    assert api.post("/api/v1/files/upload", headers=h, data={"kind": "post_media"},
                    files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert api.post("/api/v1/files/upload", headers=h, data={"kind": "avatar"},
                    files={"file": ("x.mp4", b"v", "video/mp4")}).status_code == 400
    assert api.post("/api/v1/files/upload", headers=h, data={"kind": "other"},
                    files={"file": ("x.jpg", b"j", "image/jpeg")}).status_code == 400
    assert api.post("/api/v1/files/upload", headers=h, data={"kind": "post_media"},
                    files={"file": ("x.jpg", b"", "image/jpeg")}).status_code == 400
    monkeypatch.setattr(files_api, "MAX_IMAGE_BYTES", 10)
    assert api.post("/api/v1/files/upload", headers=h, data={"kind": "post_media"},
                    files={"file": ("x.jpg", b"x" * 11, "image/jpeg")}).status_code == 413
    assert api.post("/api/v1/files/upload", data={"kind": "post_media"},
                    files={"file": ("x.jpg", b"j", "image/jpeg")}).status_code == 401


def test_delete_only_removes_your_own_files(api, bunny):
    bunny.files["/viyo-public/posts-media/u1/mine.jpg"] = b"1"
    bunny.files["/viyo-public/posts-media/u2/theirs.jpg"] = b"2"
    bunny.files["/viyo-private/creator-studio/u1/p.png"] = b"3"
    r = api.post("/api/v1/files/delete", headers={"Authorization": "Bearer u1"}, json={"urls": [
        "https://pub.b-cdn.net/posts-media/u1/mine.jpg",
        "https://pub.b-cdn.net/posts-media/u2/theirs.jpg",
        "https://priv.b-cdn.net/creator-studio/u1/p.png?token=t&expires=1",
        "https://pub.b-cdn.net/posts-media/u1/../u2/theirs.jpg",
        "https://evil.example/posts-media/u1/mine.jpg",
    ]}).json()
    assert r == {"removed": 1, "skipped": 4}
    assert set(bunny.files) == {"/viyo-public/posts-media/u2/theirs.jpg", "/viyo-private/creator-studio/u1/p.png"}


# ---------------------------------------------------------------- downloads from our own hosts

def test_repurpose_downloads_accept_bunny_hosts(bunny):
    from repurpose_storage import SourceMissing, stream_url_to_file
    with pytest.raises(SourceMissing, match="refusing"):
        stream_url_to_file("https://evil.example/x.mp4", "/dev/null", 10, vf.file_hosts())
    with pytest.raises(SourceMissing, match="refusing"):
        stream_url_to_file("http://pub.b-cdn.net/x.mp4", "/dev/null", 10, vf.file_hosts())


# ------------------------------------------------------------------ strict: Bunny or nowhere, never Supabase

class _NoSupabase:
    class storage:
        @staticmethod
        def from_(bucket):
            raise AssertionError("strict mode must never reach Supabase storage")


def test_strict_mode_never_falls_back_to_supabase(monkeypatch):
    for k in list(BUNNY_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FILE_STORE_STRICT", "true")
    with pytest.raises(vf.StorageNotConfigured) as e:
        vf.from_(_NoSupabase(), "studio")
    assert "FILE_STORE=bunny" in str(e.value) and "BUNNY_PUBLIC_KEY" in str(e.value)
    monkeypatch.setenv("FILE_STORE", "bunny")
    monkeypatch.setenv("BUNNY_PUBLIC_ZONE", "z")
    with pytest.raises(vf.StorageNotConfigured) as e:
        vf.from_(_NoSupabase(), "studio")
    assert "BUNNY_PRIVATE_TOKEN_KEY" in str(e.value) and "BUNNY_PUBLIC_ZONE" not in str(e.value)


def test_strict_mode_with_bunny_set_up_uses_bunny(bunny, monkeypatch):
    monkeypatch.setenv("FILE_STORE_STRICT", "true")
    b = vf.from_(_NoSupabase(), "studio")
    assert isinstance(b, vf.BunnyBucket)


def test_without_strict_the_old_fallback_is_unchanged(monkeypatch):
    for k in list(BUNNY_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("FILE_STORE_STRICT", raising=False)

    class Sb:
        class storage:
            @staticmethod
            def from_(bucket):
                return ("supabase", bucket)
    assert vf.from_(Sb(), "posts") == ("supabase", "posts")


def test_the_storage_status_names_what_is_missing_and_never_a_value(bunny, monkeypatch):
    st = vf.status()
    assert st["files"] == "bunny" and st["missing"] == []
    assert not any(v in str(st) for k, v in BUNNY_ENV.items() if k != "FILE_STORE"), "no setting values in the status"
    monkeypatch.delenv("BUNNY_PRIVATE_KEY")
    monkeypatch.setenv("FILE_STORE_STRICT", "true")
    st = vf.status()
    assert st["files"] == "blocked" and st["missing"] == ["BUNNY_PRIVATE_KEY"] and st["strict"] is True
