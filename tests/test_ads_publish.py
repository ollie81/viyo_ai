"""Publish to VIYO: a finished ad becomes one ordinary post (post_kind "ad") playing the same Bunny video;
only its owner can publish, change or take it down; drafts and failed ads can't be published; publishing
again updates the same post; taking it down archives (never deletes); private pictures become public only
when picked as the thumbnail."""
import uuid

import pytest

import ads_publish
import ads_studio as core
from test_ads_maker import ALICE, BOB, env, new_campaign, token  # noqa: F401  (fixture reused)

BASE = "/api/v1/ads-maker/campaigns"


@pytest.fixture
def ready(env, monkeypatch):
    uploads = []
    monkeypatch.setattr(core, "_upload_image", lambda data, mime, path: uploads.append((path, data)) or f"https://public.test/{path}")
    monkeypatch.setattr(core, "_bunny_thumbnail_url", lambda vid: f"https://bunny.test/{vid}/thumbnail.jpg")
    store = env["db"].storage

    env["db"].storage.from_ = (lambda orig: (lambda bucket: _with_download(orig(bucket), store, bucket)))(store.from_)
    c = new_campaign(env, ALICE, promote_target="physical_product", product_subtype="drink", target_name="Kivu Sunrise",
                     target_description="Sparkling hibiscus drink", destination_link="kivusunrise.rw",
                     facts={"prices": ["RWF 800"], "features": ["330 ml can"]})
    row = env["db"].db["ad_campaigns"][0]
    row.update({"status": "ready", "bunny_video_id": "vid123", "video_url": "https://play.test/vid123",
                "thumbnail_url": "private:creator-studio/ads/a/thumb.jpg", "duration_actual_seconds": 15})
    store.files["creator-studio/ads/a/thumb.jpg"] = b"jpegbytes"
    env["db"].db.setdefault("posts", [])
    return {**env, "cid": c["id"], "row": row, "uploads": uploads}


def _with_download(b, store, bucket):
    b.download = lambda path: store.files[f"{bucket}/{path}"]
    return b


def pub(ready, uid=ALICE, **body):
    return ready["c"].post(f"{BASE}/{ready['cid']}/publish", json={"caption": "Taste the sunrise", "category": "food_drink",
                                                                    "link": "kivusunrise.rw", **body}, headers=token(uid))


def test_options_suggest_from_the_campaign_and_its_facts(ready):
    o = ready["c"].get(f"{BASE}/{ready['cid']}/publish", headers=token(ALICE)).json()
    assert o["can_publish"] and o["published"] is None
    assert o["suggested"]["category"] == "food_drink" and o["suggested"]["link"] == "kivusunrise.rw"
    assert o["suggested"]["product_info"] == "RWF 800 · 330 ml can"
    keys = [t["key"] for t in o["thumbnails"]]
    assert keys[:2] == ["video", "cover"]
    assert o["thumbnails"][1]["url"].startswith("https://signed.test/"), "private cover shown as a signed link only"


def test_publish_makes_one_ad_post_with_the_same_bunny_video(ready):
    r = pub(ready, product_info="RWF 800")
    assert r.status_code == 200, r.text
    posts = ready["db"].db["posts"]
    assert len(posts) == 1
    p = posts[0]
    assert p["user_id"] == ALICE and p["post_kind"] == "ad" and p["post_type"] == "video"
    assert p["bunny_video_id"] == "vid123" and p["media_url"] == "https://play.test/vid123" and p["video_provider"] == "bunny"
    assert p["is_private"] is False and p["ad_link"] == "https://kivusunrise.rw" and p["ad_info"] == "RWF 800"
    assert p["ad_campaign_id"] == ready["cid"] and p["width"] == 720 and p["height"] == 1280
    assert p["thumbnail_url"] == "https://bunny.test/vid123/thumbnail.jpg"
    assert ready["row"]["published_post_id"] == p["id"]
    assert not ready["uploads"], "no file copied for the video thumbnail"


def test_publishing_again_updates_the_same_post(ready):
    pub(ready)
    r = pub(ready, caption="New caption", public=False)
    assert r.status_code == 200 and r.json()["public"] is False
    posts = ready["db"].db["posts"]
    assert len(posts) == 1 and posts[0]["caption"] == "New caption" and posts[0]["is_private"] is True


def test_private_post_is_kept_private(ready):
    pub(ready, public=False)
    assert ready["db"].db["posts"][0]["is_private"] is True


def test_a_private_cover_is_copied_public_only_when_chosen(ready):
    r = pub(ready, thumbnail="cover")
    assert r.status_code == 200
    assert ready["uploads"] and ready["uploads"][0][1] == b"jpegbytes"
    assert ready["db"].db["posts"][0]["thumbnail_url"].startswith("https://public.test/ad-posts/")


def test_another_campaigns_asset_cannot_be_the_thumbnail(ready):
    ready["db"].db["ad_assets"].append({"id": "other", "campaign_id": str(uuid.uuid4()), "url": "https://x.test/a.png",
                                        "asset_type": "product_photo"})
    assert pub(ready, thumbnail="asset:other").status_code == 400
    assert not ready["db"].db["posts"]


@pytest.mark.parametrize("status", ["draft", "failed", "needs_correction", "generating"])
def test_only_a_finished_ad_can_be_published(ready, status):
    ready["row"]["status"] = status
    assert pub(ready).status_code == 409
    assert not ready["db"].db["posts"]


def test_bad_link_and_category_are_refused(ready):
    assert pub(ready, link="not a link").status_code == 400
    assert pub(ready, category="weapons").status_code == 400
    assert not ready["db"].db["posts"]


def test_contact_links_are_allowed(ready):
    assert pub(ready, link="tel:+250788123456").status_code == 200
    assert ready["db"].db["posts"][0]["ad_link"] == "tel:+250788123456"


@pytest.mark.parametrize("method,path", [("get", "/publish"), ("post", "/publish"), ("post", "/unpublish")])
def test_another_creator_cannot_publish_or_take_down(ready, method, path):
    if path == "/unpublish":
        pub(ready)
    kw = {"json": {"category": "food_drink"}} if path == "/publish" and method == "post" else {}
    r = getattr(ready["c"], method)(f"{BASE}/{ready['cid']}{path}", headers=token(BOB), **kw)
    assert r.status_code == 404
    assert all(p["user_id"] == ALICE for p in ready["db"].db["posts"])
    assert not any(p.get("is_archived") for p in ready["db"].db["posts"])


def test_take_down_archives_and_never_deletes(ready):
    pub(ready)
    r = ready["c"].post(f"{BASE}/{ready['cid']}/unpublish", headers=token(ALICE))
    assert r.status_code == 200 and r.json()["archived"] is True and r.json()["public"] is False
    posts = ready["db"].db["posts"]
    assert len(posts) == 1 and posts[0]["is_archived"] is True
    assert ready["row"]["bunny_video_id"] == "vid123" and ready["row"]["status"] == "ready", "the ad itself is untouched"
    # and it can be put back
    assert pub(ready).json()["public"] is True and posts[0]["is_archived"] is False


def test_unpublish_before_publishing_is_not_found(ready):
    assert ready["c"].post(f"{BASE}/{ready['cid']}/unpublish", headers=token(ALICE)).status_code == 404


def test_a_post_that_changed_owner_is_never_updated(ready):
    pub(ready)
    ready["db"].db["posts"][0]["user_id"] = BOB      # e.g. a stale/forged link to someone else's post
    pub(ready, caption="mine")
    assert len(ready["db"].db["posts"]) == 2 and ready["db"].db["posts"][0]["caption"] == "Taste the sunrise"


def test_admin_publish_needs_the_admin_key(ready, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import studio
    monkeypatch.setattr(studio, "ADMIN_API_KEY", "k")
    app = FastAPI()
    app.include_router(ads_publish.admin_router)
    c = TestClient(app)
    path = f"/api/v1/admin/ads-studio/campaign/{ready['cid']}/publish"
    assert c.post(path, json={"category": "food_drink"}, headers=token(ALICE)).status_code in (401, 403)
    assert c.post(path, json={"category": "food_drink"}, headers={"X-Admin-Key": "k"}).status_code == 200
