"""Creator downloads: owner-only links to a creator's own videos; nothing is
ever deleted, moved, edited or published by downloading."""
import time
import urllib.parse

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

import bunny_stream
import creator_downloads as cd

SUPA = "https://proj.supabase.co"
A, B = "aaaaaaaa-0000-0000-0000-000000000001", "bbbbbbbb-0000-0000-0000-000000000002"
S1 = "51515151-0000-0000-0000-000000000001"


def pid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


class Destructive(AssertionError):
    pass


class FakeQuery:
    def __init__(self, db, table):
        self.db, self.table, self.filters, self._limit, self._order = db, table, [], None, None
        self._insert = None

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self.filters.append((col, val))
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def insert(self, row):
        self._insert = row
        return self

    def update(self, *_a, **_k):
        raise Destructive(f"update on {self.table}")

    def delete(self, *_a, **_k):
        raise Destructive(f"delete on {self.table}")

    def upsert(self, *_a, **_k):
        raise Destructive(f"upsert on {self.table}")

    def execute(self):
        if self._insert is not None:
            if self.table not in self.db.tables_that_exist:
                raise RuntimeError(f'relation "{self.table}" does not exist')
            self.db.rows.setdefault(self.table, []).append(dict(self._insert))
            return type("R", (), {"data": [self._insert]})()
        rows = [r for r in self.db.rows.get(self.table, []) if all(r.get(c) == v for c, v in self.filters)]
        if self._order:
            rows.sort(key=lambda r: r.get(self._order[0]) or "", reverse=self._order[1])
        if self._limit:
            rows = rows[: self._limit]
        return type("R", (), {"data": [dict(r) for r in rows]})()


class FakeBucket:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def create_signed_url(self, path, expires_in, options=None):
        self.db.signed.append((self.name, path, expires_in, dict(options or {})))
        token = f"tok{len(self.db.signed)}"
        q = f"token={token}"
        if options and options.get("download"):
            q += "&download=" + urllib.parse.quote(options["download"])
        return {"signedURL": f"{SUPA}/storage/v1/object/sign/{self.name}/{path}?{q}"}

    def remove(self, *_a, **_k):
        raise Destructive(f"remove from {self.name}")

    def upload(self, *_a, **_k):
        raise Destructive(f"upload to {self.name}")

    def move(self, *_a, **_k):
        raise Destructive(f"move in {self.name}")


class FakeStorage:
    def __init__(self, db):
        self.db = db

    def from_(self, name):
        return FakeBucket(self.db, name)


class FakeAdmin:
    def __init__(self):
        self.rows = {"posts": [], "series": []}
        self.signed = []
        self.tables_that_exist = {"creator_download_events"}
        self.storage = FakeStorage(self)

    def table(self, name):
        return FakeQuery(self, name)


def media(path, bucket="posts-media"):
    return f"{SUPA}/storage/v1/object/public/{bucket}/{path}"


def video_post(n, owner=A, **over):
    p = {"id": pid(n), "user_id": owner, "post_type": "video", "caption": f"My film {n}\nmore",
         "media_url": media(f"{owner}/v{n}.mp4"), "thumbnail_url": media(f"{owner}/t{n}.jpg"),
         "is_private": False, "is_archived": False, "series_id": None, "episode_number": None,
         "video_provider": None, "bunny_video_id": None, "video_status": None,
         "duration_seconds": 30, "created_at": f"2026-11-0{n % 9 + 1}"}
    p.update(over)
    return p


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", SUPA)
    admin = FakeAdmin()
    monkeypatch.setattr(cd, "_admin", admin)
    monkeypatch.setattr(cd, "_link_log", cd.defaultdict(cd.deque))
    bunny = {}

    def fake_info(video_id):
        if video_id not in bunny:
            raise bunny_stream.BunnyVideoNotFound(video_id)
        v = bunny[video_id]
        if v == "down":
            raise bunny_stream.BunnyError("timeout")
        return v

    monkeypatch.setattr(cd, "bunny_info", fake_info)
    monkeypatch.setattr(bunny_stream, "BUNNY_STREAM_PULL_ZONE", "vz-test.b-cdn.net")
    monkeypatch.setattr(bunny_stream, "delete_video",
                        lambda *_a: (_ for _ in ()).throw(Destructive("bunny delete")))

    app = FastAPI()
    app.include_router(cd.router)

    async def fake_creator(authorization: str = Header(None)):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Not signed in")
        if authorization == "Bearer guest":
            raise HTTPException(status_code=403, detail="Create an account to continue.")
        return authorization.removeprefix("Bearer ")

    app.dependency_overrides[cd._creator] = fake_creator
    return TestClient(app), admin, bunny


def as_user(u):
    return {"Authorization": f"Bearer {u}"}


# 1 ------------------------------------------------------------------ own download

def test_creator_downloads_own_video_with_expiring_signed_link(env):
    client, admin, _ = env
    admin.rows["posts"].append(video_post(1))
    before = [dict(p) for p in admin.rows["posts"]]
    t0 = int(time.time())
    r = client.post(f"/api/v1/creator/downloads/{pid(1)}", headers=as_user(A))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "supabase" and body["kind"] == "video"
    assert body["file_name"] == "my-film-1-000001.mp4"
    bucket, path, seconds, opts = admin.signed[-1]
    assert (bucket, path) == ("posts-media", f"{A}/v1.mp4")
    assert seconds == cd.LINK_SECONDS and opts == {"download": "my-film-1-000001.mp4"}
    assert t0 + cd.LINK_SECONDS <= body["expires_at"] <= int(time.time()) + cd.LINK_SECONDS
    assert "/object/sign/" in body["url"] and "token=" in body["url"], "a signed link, not the permanent one"
    assert body["url"] != before[0]["media_url"]
    # 3 + 6: the stored copy and the post are untouched (no publish, no delete)
    assert admin.rows["posts"] == before
    assert admin.rows["creator_download_events"] == [{"user_id": A, "post_id": pid(1), "kind": "video"}]


# 2 ------------------------------------------------------------------ isolation

def test_creator_cannot_download_someone_elses_private_video(env):
    client, admin, _ = env
    admin.rows["posts"].append(video_post(2, owner=B, is_private=True))
    for path in (f"/api/v1/creator/downloads/{pid(2)}", f"/api/v1/creator/downloads/{pid(2)}/thumbnail"):
        r = client.post(path, headers=as_user(A))
        assert r.status_code == 404, "same answer as a missing post, so nothing leaks"
    assert admin.signed == []
    assert client.post(f"/api/v1/creator/downloads/{pid(2)}", headers=as_user(B)).status_code == 200


def test_signed_out_guest_and_bad_ids_are_refused(env):
    client, admin, _ = env
    admin.rows["posts"].append(video_post(3))
    assert client.post(f"/api/v1/creator/downloads/{pid(3)}").status_code == 401
    assert client.post(f"/api/v1/creator/downloads/{pid(3)}", headers=as_user("guest")).status_code == 403
    assert client.post("/api/v1/creator/downloads/not-a-uuid", headers=as_user(A)).status_code == 404
    assert client.post(f"/api/v1/creator/downloads/{pid(99)}", headers=as_user(A)).status_code == 404


def test_listing_only_shows_the_callers_own_productions(env):
    client, admin, _ = env
    admin.rows["series"] += [{"id": S1, "user_id": A, "title": "Night Shift", "content_type": "series",
                              "orientation": "vertical", "status": "ongoing", "cover_image_url": "c.jpg",
                              "created_at": "2026-11-01"},
                             {"id": "52525252-0000-0000-0000-000000000002", "user_id": B, "title": "Not mine",
                              "created_at": "2026-11-02"}]
    admin.rows["posts"] += [
        video_post(10, series_id=S1, episode_number=2, caption=""),
        video_post(11, series_id=S1, episode_number=1, is_private=True),
        video_post(12),
        video_post(13, owner=B),
        video_post(14, video_provider="bunny", bunny_video_id="vid-fail", video_status="failed", media_url=None),
    ]
    r = client.get("/api/v1/creator/productions", headers=as_user(A))
    assert r.status_code == 200
    prods = r.json()["productions"]
    assert [p["title"] for p in prods if p["kind"] == "series"] == ["Night Shift"]
    series = prods[0]
    assert [e["episode_number"] for e in series["episodes"]] == [1, 2]
    assert series["episodes"][0]["watch_status"] == "private"
    assert series["episodes"][1]["title"] == "Episode 2"
    assert series["episode_count"] == 2 and series["public_episode_count"] == 1
    singles = {p["id"]: p for p in prods if p["kind"] == "video"}
    assert set(singles) == {pid(12), pid(14)}, "other creators' videos never listed"
    assert singles[pid(14)]["episodes"][0]["downloadable"] is False
    assert singles[pid(14)]["episodes"][0]["not_downloadable_reason"] == "Processing failed"


# 4 + 5 ------------------------------------------------------------ retry / expiry

def test_each_request_gets_a_fresh_link_so_a_failed_or_expired_download_can_retry(env):
    client, admin, _ = env
    admin.rows["posts"].append(video_post(4))
    u1 = client.post(f"/api/v1/creator/downloads/{pid(4)}", headers=as_user(A)).json()["url"]
    u2 = client.post(f"/api/v1/creator/downloads/{pid(4)}", headers=as_user(A)).json()["url"]
    assert u1 != u2, "a retry signs a new link instead of reusing an old one"
    assert all(s[2] == cd.LINK_SECONDS for s in admin.signed), "every link carries an expiry"
    assert cd.LINK_SECONDS <= 3600


def test_storage_failure_is_a_clean_retryable_error(env, monkeypatch):
    client, admin, _ = env
    admin.rows["posts"].append(video_post(5))
    monkeypatch.setattr(cd, "sign_storage", lambda *a: (_ for _ in ()).throw(RuntimeError("503 from storage")))
    r = client.post(f"/api/v1/creator/downloads/{pid(5)}", headers=as_user(A))
    assert r.status_code == 502 and "Try again" in r.json()["detail"]
    assert "creator_download_events" not in admin.rows, "nothing logged for a link that wasn't made"


def test_rate_limit_stops_bulk_pulling(env, monkeypatch):
    client, admin, _ = env
    admin.rows["posts"].append(video_post(6))
    monkeypatch.setattr(cd, "LINKS_PER_WINDOW", 3)
    codes = [client.post(f"/api/v1/creator/downloads/{pid(6)}", headers=as_user(A)).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert client.post(f"/api/v1/creator/downloads/{pid(6)}", headers=as_user(B)).status_code == 404, \
        "the limit is per creator"


# 8 ------------------------------------------------------------------ episodes

def test_downloading_one_episode_touches_only_that_episode(env, monkeypatch):
    client, admin, _ = env
    admin.rows["series"].append({"id": S1, "user_id": A, "title": "Night Shift!", "created_at": "x"})
    admin.rows["posts"] += [video_post(20, series_id=S1, episode_number=1),
                            video_post(21, series_id=S1, episode_number=2)]
    before = [dict(p) for p in admin.rows["posts"]]
    body = client.post(f"/api/v1/creator/downloads/{pid(21)}", headers=as_user(A)).json()
    assert body["file_name"] == "night-shift-ep02-000021.mp4"
    assert [s[1] for s in admin.signed] == [f"{A}/v21.mp4"]
    assert admin.rows["posts"] == before


# Bunny ---------------------------------------------------------------

def test_bunny_video_gives_best_mp4_and_checks_processing_live(env):
    client, admin, bunny = env
    admin.rows["posts"].append(video_post(30, video_provider="bunny", bunny_video_id="vid30",
                                          video_status="processing", media_url=None))
    bunny["vid30"] = {"status": 3, "availableResolutions": "360p,720p"}
    r = client.post(f"/api/v1/creator/downloads/{pid(30)}", headers=as_user(A))
    assert r.status_code == 409 and "still processing" in r.json()["detail"]
    bunny["vid30"] = {"status": 4, "availableResolutions": "240p,360p,720p,1080p", "width": 1920, "height": 1080}
    body = client.post(f"/api/v1/creator/downloads/{pid(30)}", headers=as_user(A)).json()
    assert body["url"] == "https://vz-test.b-cdn.net/vid30/play_1080p.mp4"
    assert body["quality"] == "1080p" and body["orientation"] == "16:9 landscape" and body["source"] == "bunny"
    bunny["vid30"] = {"status": 4, "availableResolutions": "360p,480p", "width": 480, "height": 854}
    body = client.post(f"/api/v1/creator/downloads/{pid(30)}", headers=as_user(A)).json()
    assert body["quality"] == "480p" and body["orientation"] == "9:16 vertical", "never promises a size it doesn't have"


def test_bunny_failures_are_explained(env):
    client, admin, bunny = env
    admin.rows["posts"] += [video_post(31, video_provider="bunny", bunny_video_id="gone", media_url=None),
                            video_post(32, video_provider="bunny", bunny_video_id="vid32", media_url=None),
                            video_post(33, video_provider="bunny", bunny_video_id="vid33", media_url=None)]
    bunny["vid32"] = "down"
    bunny["vid33"] = {"status": 5}
    assert client.post(f"/api/v1/creator/downloads/{pid(31)}", headers=as_user(A)).status_code == 410
    assert client.post(f"/api/v1/creator/downloads/{pid(32)}", headers=as_user(A)).status_code == 502
    assert client.post(f"/api/v1/creator/downloads/{pid(33)}", headers=as_user(A)).status_code == 409


def test_posts_without_a_stored_file_are_not_offered(env):
    client, admin, _ = env
    admin.rows["posts"] += [video_post(40, media_url="https://elsewhere.example/v.mp4"),
                            video_post(41, media_url=None)]
    for n in (40, 41):
        assert client.post(f"/api/v1/creator/downloads/{pid(n)}", headers=as_user(A)).status_code == 409
    assert admin.signed == []


def test_thumbnail_download(env):
    client, admin, _ = env
    admin.rows["posts"].append(video_post(50))
    body = client.post(f"/api/v1/creator/downloads/{pid(50)}/thumbnail", headers=as_user(A)).json()
    assert body["kind"] == "thumbnail" and body["file_name"] == "my-film-50-000050-thumbnail.jpg"
    assert admin.signed[-1][1] == f"{A}/t50.jpg"


def test_download_still_works_when_the_log_table_is_missing(env):
    client, admin, _ = env
    admin.tables_that_exist = set()
    admin.rows["posts"].append(video_post(60))
    assert client.post(f"/api/v1/creator/downloads/{pid(60)}", headers=as_user(A)).status_code == 200


def test_storage_location_only_accepts_this_projects_storage(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", SUPA)
    assert cd.storage_location(media("u/a%20b.mp4")) == ("posts-media", "u/a b.mp4")
    assert cd.storage_location("https://evil.example/storage/v1/object/public/posts-media/u/a.mp4") is None
    assert cd.storage_location("https://vz.b-cdn.net/x/play_720p.mp4") is None
    assert cd.storage_location(None) is None


def test_download_module_has_no_destructive_calls():
    """Belt and braces for 'downloading never deletes the master'."""
    import inspect
    src = inspect.getsource(cd)
    for bad in (".delete(", ".remove(", ".update(", ".upsert(", ".move(", "delete_video(", ".upload("):
        assert bad not in src, bad
