"""Shared post pages and the "Made with Viyo" clip tag."""
import subprocess

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import clip_brand
import repurpose as r
import share_pages
from conftest import connect, make_user, pg_rpc
from repurpose_jobs import JobConfig, JobStore
from repurpose_storage import CLIPS_BUCKET, LocalStorage

PID = "11111111-2222-3333-4444-555555555555"


def _post(**over):
    post = {"id": PID, "post_type": "video", "caption": "How I price my work <fast>\nSecond line",
            "media_url": "https://cdn.example/secret-episode.mp4", "thumbnail_url": "https://cdn.example/t.jpg",
            "is_private": False, "is_archived": False, "episode_number": None,
            "profiles": {"username": "ada", "display_name": "Ada"}, "series": None}
    post.update(over)
    return post


@pytest.fixture
def share_client(monkeypatch):
    app = FastAPI()
    app.include_router(share_pages.router)
    posts = {}
    monkeypatch.setattr(share_pages, "load_public_post",
                        lambda pid: (lambda p: None if not p or p["is_private"] or p["is_archived"] else p)(posts.get(pid)))
    return TestClient(app), posts


def test_shared_post_page_has_preview_tags_and_opens_the_app(share_client):
    client, posts = share_client
    posts[PID] = _post()
    resp = client.get(f"/share/p/{PID}")
    assert resp.status_code == 200
    page = resp.text
    assert f'<link rel="canonical" href="https://www.viyo.site/p/{PID}">' in page
    assert 'property="og:image" content="https://cdn.example/t.jpg"' in page
    assert "How I price my work &lt;fast&gt;" in page and "<fast>" not in page, "caption is escaped"
    assert f"https://www.viyo.site/app/?post={PID}" in page
    assert "secret-episode.mp4" not in page, "the video file is never exposed"


def test_private_archived_missing_and_bad_ids_are_not_shown(share_client):
    client, posts = share_client
    posts[PID] = _post(is_private=True)
    for url in [f"/share/p/{PID}", "/share/p/not-a-uuid", "/share/p/99999999-2222-3333-4444-555555555555"]:
        resp = client.get(url)
        assert resp.status_code == 404 and 'name="robots" content="noindex"' in resp.text
    posts[PID] = _post(is_archived=True)
    assert client.get(f"/share/p/{PID}").status_code == 404


def test_drama_episode_page_names_the_series():
    page = share_pages.render_post_page(_post(series={"title": "Night Shift"}, episode_number=4))
    assert "Night Shift, episode 4" in page


def _video(path, seconds=2):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=1080x1920:rate=30:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=f=440:d={seconds}", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", path], check=True)


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return out


def test_brand_clip_keeps_size_and_sound(tmp_path):
    src, out = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4")
    _video(src)
    clip_brand.brand_clip(src, out)
    info = _probe(out)
    assert "video,1080,1920" in info and "audio" in info
    with pytest.raises(clip_brand.BrandError):
        (tmp_path / "bad.mp4").write_bytes(b"nope")
        clip_brand.brand_clip(str(tmp_path / "bad.mp4"), str(tmp_path / "o2.mp4"))


@pytest.fixture
def api(db, tmp_path, monkeypatch):
    store = JobStore(pg_rpc(connect()), JobConfig(cost=40, daily_limit=5, free_trial_daily_cap=0))
    storage = LocalStorage(str(tmp_path / "storage"))
    monkeypatch.setattr(r, "job_store", store)
    monkeypatch.setattr(r, "storage", storage)
    app = FastAPI()
    app.include_router(r.router)
    state = {"user": None}
    app.dependency_overrides[r._get_current_user_id] = lambda: state["user"]
    return {"db": db, "store": store, "storage": storage, "client": TestClient(app), "state": state, "tmp": tmp_path}


def test_branded_copy_is_made_once_for_the_owner_only(api):
    db = api["db"]
    owner, other = make_user(db, 100), make_user(db, 100)
    job = api["store"].create_job(str(owner), source_kind="legacy_public_url", source_bucket=None, source_path=None,
                                  source_url="https://x/a.mp4", creative_prompt=None, client_request_id="brand-1")
    jid = str(job["job_id"])
    src = str(api["tmp"] / "clip.mp4")
    _video(src)
    api["storage"].upload(CLIPS_BUCKET, f"{owner}/{jid}/clip_1.mp4", src, "video/mp4")
    db.execute("insert into repurpose_clips (job_id, user_id, rank, status, start_time, end_time, video_url) "
               "values (%s, %s, 1, 'rendered', 0, 2, 'file://x')", (jid, owner))
    url = f"/api/v1/repurpose/{jid}/clips/1/branded"
    api["state"]["user"] = str(other)
    assert api["client"].post(url).status_code == 403
    api["state"]["user"] = str(owner)
    first = api["client"].post(url)
    assert first.status_code == 200 and first.json()["url"].endswith(f"{jid}/clip_1_viyo.mp4")
    assert "video,1080,1920" in _probe(first.json()["url"].replace("file://", ""))
    assert api["client"].post(url).json() == first.json(), "reused, not re-rendered"
    assert api["client"].post(f"/api/v1/repurpose/{jid}/clips/2/branded").status_code == 404
