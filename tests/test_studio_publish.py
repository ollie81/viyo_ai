"""Publishing a Studio production to Viyo: an episode post in the creator's
own drama, live at once, priced by the creator, with the main cover as the
poster. Supabase, Bunny and push are replaced by a recording fake."""
import os
import uuid

import pytest

import creator_studio as cs
from conftest import make_user
from test_character_identity import setup
from test_creator_studio import H, env, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import TALK

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_publish.sql")


class FakePublisher:
    def __init__(self):
        self.series_rows, self.posts, self.images, self.after, self.uploads = {}, [], [], [], []

    def public_image(self, data, mime, owner_id):
        self.images.append((owner_id, len(data)))
        return f"https://cdn.example/cover-{len(self.images)}.png"

    def series(self, sid):
        return self.series_rows.get(sid)

    def create_series(self, row):
        row = dict(row, id=str(uuid.uuid4()))
        self.series_rows[row["id"]] = row
        return row

    def update_series(self, sid, fields):
        self.series_rows[sid].update(fields)

    def last_episode(self, sid):
        return max([p["episode_number"] for p in self.posts if p["series_id"] == sid] or [0])

    def insert_post(self, row):
        row = dict(row, id=str(uuid.uuid4()))
        self.posts.append(row)
        return row

    def after_post(self, post, owner, sid, ep):
        self.after.append((post["id"], owner, sid, ep))

    def bunny_playback(self, vid):
        return f"https://bunny.example/{vid}/play_720p.mp4"

    def bunny_upload(self, path, title):
        self.uploads.append(title)
        return "vid-" + str(len(self.uploads))


@pytest.fixture
def pub(monkeypatch):
    f = FakePublisher()
    monkeypatch.setattr(cs, "_publisher", f)
    return f


def render(db, files, u, pid, version=1):
    path = f"{u}/{pid}/render/{uuid.uuid4().hex}.mp4"
    files.put(path, b"\x00" * 64, "video/mp4")
    return db.execute("insert into public.studio_renders (project_id, owner_id, version, path, aspect_ratio, duration_seconds) "
                      "values (%s, %s, %s, %s, '9:16', 61) returning id", (pid, u, version, path)).fetchone()[0]


def publish(client, u, rid, **body):
    return client.post(f"/api/v1/studio/renders/{rid}/publish", headers=H(u), json={"confirm": True, **body})


def test_publishing_makes_a_live_episode_with_the_cover_as_poster(env, pub):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    _, _, store, files, *_ = env
    start(client, u, pid, "cover")
    run(gen)
    rid = render(db, files, u, pid)
    assert client.post(f"/api/v1/studio/renders/{rid}/publish", headers=H(u), json={}).status_code == 400, "must confirm"
    r = publish(client, u, rid, coin_price=0, caption="Episode one!")
    assert r.status_code == 200, r.text
    out = r.json()
    s = pub.series_rows[out["series_id"]]
    assert s["user_id"] == str(u) and s["title"] == "Rooftop" and s["cover_image_url"] == out["cover_url"]
    assert s["content_type"] == "short_drama" and s["orientation"] == "vertical"
    post = pub.posts[0]
    assert post["user_id"] == str(u) and post["is_private"] is False and post["series_id"] == out["series_id"]
    assert post["episode_number"] == 1 and post["is_free"] is True and post["thumbnail_url"] == out["cover_url"]
    assert post["video_provider"] == "bunny" and post["media_url"].startswith("https://bunny.example/")
    assert post["caption"] == "Episode one!" and pub.after == [(post["id"], str(u), out["series_id"], 1)]
    assert publish(client, u, rid).status_code == 409, "a version is published once"
    detail = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()
    assert detail["renders"][0]["post_id"] == post["id"]


def test_the_next_episode_joins_the_same_drama_and_a_new_main_cover_updates_the_poster(env, pub):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    _, _, store, files, *_ = env
    start(client, u, pid, "cover")
    run(gen)
    first = publish(client, u, render(db, files, u, pid), coin_price=40).json()
    assert pub.series_rows[first["series_id"]]["coin_price_per_episode"] == 40 and "is_free" not in pub.posts[0]
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u)).json()["id"]
    start(client, u, ep2, "cover")
    run(gen)
    second = publish(client, u, render(db, files, u, ep2), coin_price=40).json()
    assert second["series_id"] == first["series_id"] and second["episode_number"] == 2
    assert pub.series_rows[first["series_id"]]["cover_image_url"] == second["cover_url"]
    assert len(pub.series_rows) == 1


def test_only_the_owner_can_publish(env, pub):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    _, _, store, files, *_ = env
    rid = render(db, files, u, pid)
    other = make_user(db)
    assert publish(client, other, rid).status_code == 404
    assert pub.posts == []


def test_a_film_is_a_single_title_and_free_means_free(env, pub):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    _, _, store, files, *_ = env
    db.execute("update public.studio_projects set kind = 'film' where id = %s", (pid,))
    out = publish(client, u, render(db, files, u, pid), coin_price=0).json()
    assert pub.series_rows[out["series_id"]]["content_type"] == "movie" and pub.posts[0]["is_free"] is True
    assert pub.uploads == ["Rooftop"], "a render stored only in Studio is uploaded to the video host"
