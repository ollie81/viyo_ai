"""My Brands: a brand belongs to one creator; its pictures are private; a new ad made from it is filled in
from the brand, gets its own copy of the pictures and reuses the brand's character (no new portrait);
saving an ad as a brand works; archiving keeps everything; character portraits survive deleting their ad."""
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ads_brands
import ads_maker
import ads_studio as core
from test_ads_maker import ALICE, BOB, FakeDB, new_campaign, png, token  # noqa: F401

B = "/api/v1/ads-maker/brands"
CH1, CHB = str(uuid.uuid4()), str(uuid.uuid4())


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    for t in ("ad_brands", "ad_brand_assets", "ad_characters"):
        db.db[t] = []
    store = db.storage
    orig = store.from_

    def from_(bucket):
        b = orig(bucket)
        b.download = lambda path: store.files[f"{bucket}/{path}"]
        return b
    store.from_ = from_
    monkeypatch.setattr(core, "supabase_admin", db)
    monkeypatch.setattr(core, "_require_configured", lambda: None)
    monkeypatch.setattr(core, "_load_campaign_assets",
                        lambda cid: [a for a in db.db["ad_assets"] if a["campaign_id"] == cid])
    monkeypatch.setenv("STUDIO_CREATORS_OPEN", "true")
    app = FastAPI()
    app.include_router(ads_brands.router)
    app.include_router(ads_maker.router)
    return {"db": db, "c": TestClient(app)}


KIVU = {"name": "Kivu Sunrise", "description": "Sparkling hibiscus drink", "promote_target": "physical_product",
        "product_subtype": "drink", "website": "kivusunrise.rw", "cta_text": "Find it in shops",
        "facts": {"prices": ["RWF 800"], "features": ["330 ml can"]}, "brand_colors": ["#e8452c"],
        "social_links": ["instagram.com/kivusunrise"], "contact": {"phone": "+250 788 123 456"}}


def brand(env, uid=ALICE, **over):
    r = env["c"].post(B, json={**KIVU, **over}, headers=token(uid))
    assert r.status_code == 200, r.text
    return r.json()


def upload(env, bid, kind="product_photo", uid=ALICE):
    return env["c"].post(f"{B}/{bid}/assets?asset_type={kind}", files={"file": ("p.png", png(), "image/png")},
                         headers=token(uid))


def test_create_brand_with_verified_details(env):
    b = brand(env)
    assert b["name"] == "Kivu Sunrise" and b["brand_colors"] == ["#E8452C"] and b["facts"]["prices"] == ["RWF 800"]
    assert env["db"].db["ad_brands"][0]["owner_id"] == ALICE


@pytest.mark.parametrize("over", [{"name": " "}, {"website": "not a site"}, {"brand_colors": ["blue"]},
                                  {"promote_target": "viyo"}, {"promote_target": "weapons"}])
def test_bad_brand_details_are_refused(env, over):
    assert env["c"].post(B, json={**KIVU, **over}, headers=token(ALICE)).status_code == 400


def test_brands_are_private_to_their_owner(env):
    b = brand(env)
    assert [x["name"] for x in env["c"].get(B, headers=token(ALICE)).json()] == ["Kivu Sunrise"]
    assert env["c"].get(B, headers=token(BOB)).json() == []
    for method, path in [("get", ""), ("put", ""), ("post", "/campaigns"), ("delete", "/character")]:
        kw = {"json": {}} if method in ("put", "post") else {}
        assert getattr(env["c"], method)(f"{B}/{b['id']}{path}", headers=token(BOB), **kw).status_code == 404
    assert upload(env, b["id"], uid=BOB).status_code == 404


def test_brand_pictures_are_private_and_the_first_logo_is_the_logo(env):
    b = brand(env)
    assert upload(env, b["id"], "logo").status_code == 200
    a = upload(env, b["id"]).json()
    got = env["c"].get(f"{B}/{b['id']}", headers=token(ALICE)).json()
    assert got["logo_url"].startswith("https://signed.test/creator-studio/ads/")
    assert len(got["assets"]) == 2 and a["url"].startswith("https://signed.test/")
    assert all(k.startswith(f"creator-studio/ads/{ALICE}/brands/") for k in env["db"].storage.files)


def test_new_ad_from_brand_is_filled_in_and_gets_its_own_pictures(env):
    b = brand(env)
    upload(env, b["id"], "logo")
    upload(env, b["id"])
    r = env["c"].post(f"{B}/{b['id']}/campaigns", json={"ad_style": "lifestyle", "aspect_ratio": "1:1"},
                      headers=token(ALICE))
    assert r.status_code == 200, r.text
    c = r.json()
    row = next(x for x in env["db"].db["ad_campaigns"] if x["id"] == c["id"])
    assert row["user_id"] == ALICE and row["brand_id"] == b["id"] and row["target_name"] == "Kivu Sunrise"
    assert row["destination_link"] == "kivusunrise.rw" and row["facts"]["prices"] == ["RWF 800"]
    assert row["ad_style"] == "lifestyle" and row["aspect_ratio"] == "1:1" and row.get("character_mode", "auto") == "auto"
    assets = [a for a in env["db"].db["ad_assets"] if a["campaign_id"] == c["id"]]
    assert sorted(a["asset_type"] for a in assets) == ["logo", "product_photo"]
    assert all(f"/ads/{ALICE}/{c['id']}/" in a["url"] for a in assets), "the ad's own copy, in its own folder"
    # deleting the ad removes its copies, never the brand's pictures
    env["c"].delete(f"/api/v1/ads-maker/campaigns/{c['id']}", headers=token(ALICE))
    assert len([k for k in env["db"].storage.files if "/brands/" in k]) == 2


def test_the_brand_character_is_reused_without_a_new_portrait(env):
    b = brand(env)
    env["db"].db["ad_characters"].append({"id": CH1, "owner_id": ALICE, "name": "Aline",
                                          "spec": {"appearance": "curly hair"}, "portrait": "private:x/p.png",
                                          "voice_name": "Kore"})
    assert env["c"].put(f"{B}/{b['id']}/character/{CH1}", headers=token(ALICE)).status_code == 200
    c = env["c"].post(f"{B}/{b['id']}/campaigns", json={}, headers=token(ALICE)).json()
    row = next(x for x in env["db"].db["ad_campaigns"] if x["id"] == c["id"])
    assert row["character_mode"] == "reuse" and row["character"]["saved_id"] == CH1
    assert row["character"]["portrait"] == "private:x/p.png" and row["character"]["voice_name"] == "Kore"
    assert ads_maker.estimate(row, assets=[])["video"] == ads_maker.estimate({**row, "character_mode": "none"},
                                                                                assets=[])["video"], "no portrait cost"
    # product-only on request
    c2 = env["c"].post(f"{B}/{b['id']}/campaigns", json={"use_character": False}, headers=token(ALICE)).json()
    assert next(x for x in env["db"].db["ad_campaigns"] if x["id"] == c2["id"])["character_mode"] == "none"


def test_someone_elses_character_cannot_be_the_brand_character(env):
    b = brand(env)
    env["db"].db["ad_characters"].append({"id": CHB, "owner_id": BOB, "name": "Bob's"})
    assert env["c"].put(f"{B}/{b['id']}/character/{CHB}", headers=token(ALICE)).status_code == 404


def test_save_an_ad_as_a_brand(env):
    c = new_campaign(env, ALICE, promote_target="physical_product", target_name="Kivu Sunrise",
                     destination_link="kivusunrise.rw", facts={"prices": ["RWF 800"]}, brand_colors=["#E8452C"])
    env["db"].storage.files[f"creator-studio/ads/{ALICE}/{c['id']}/logo.png"] = b"logo"
    env["db"].db["ad_assets"].append({"id": "a1", "campaign_id": c["id"], "asset_type": "logo",
                                      "url": f"private:creator-studio/ads/{ALICE}/{c['id']}/logo.png"})
    r = env["c"].post(f"{B}/from-campaign/{c['id']}", headers=token(ALICE))
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["name"] == "Kivu Sunrise" and got["website"] == "kivusunrise.rw" and got["facts"]["prices"] == ["RWF 800"]
    assert got["logo_url"] and len(got["assets"]) == 1 and [x["id"] for x in got["campaigns"]] == [c["id"]]
    # saving again returns the same brand
    assert env["c"].post(f"{B}/from-campaign/{c['id']}", headers=token(ALICE)).json()["id"] == got["id"]
    assert len(env["db"].db["ad_brands"]) == 1
    # someone else can't
    assert env["c"].post(f"{B}/from-campaign/{c['id']}", headers=token(BOB)).status_code == 404


def test_archive_hides_but_keeps_everything(env):
    b = brand(env)
    env["c"].post(f"{B}/{b['id']}/campaigns", json={}, headers=token(ALICE))
    assert env["c"].put(f"{B}/{b['id']}", json={"archived": True}, headers=token(ALICE)).status_code == 200
    assert env["c"].get(B, headers=token(ALICE)).json() == []
    assert len(env["c"].get(f"{B}?include_archived=true", headers=token(ALICE)).json()) == 1
    assert len(env["db"].db["ad_brands"]) == 1 and len(env["db"].db["ad_campaigns"]) == 1
    assert env["c"].post(f"{B}/{b['id']}/campaigns", json={}, headers=token(ALICE)).status_code == 400


def test_removing_a_brand_picture_keeps_ads_already_made(env):
    b = brand(env)
    a = upload(env, b["id"]).json()
    c = env["c"].post(f"{B}/{b['id']}/campaigns", json={}, headers=token(ALICE)).json()
    assert env["c"].delete(f"{B}/{b['id']}/assets/{a['id']}", headers=token(ALICE)).status_code == 200
    copies = [x for x in env["db"].db["ad_assets"] if x["campaign_id"] == c["id"]]
    ref = copies[0]["url"][len("private:"):]
    assert ref in env["db"].storage.files


def test_character_portrait_is_stored_outside_the_campaign_folder(env):
    ref = core._store_character_portrait({"id": "c1", "user_id": ALICE, "created_via": "creator"}, b"png")
    assert f"/ads/{ALICE}/characters/" in ref and "/c1/" not in ref
