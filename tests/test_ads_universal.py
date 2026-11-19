"""Universal Ads Maker: any product/service/business, verified facts only, optional consistent characters,
real product photos in product scenes, brand colours, and the extra final checks. End-to-end tests run the
real pipeline (real FFmpeg, layout, OCR); only paid AI calls, storage and Bunny are local stand-ins."""
import os
import subprocess
import uuid

import pytest
from PIL import Image, ImageDraw, ImageFont

import ad_claims
import ad_compose
import ad_qc
import ads_studio
from test_ads_quality import pipeline, _img, _tone  # noqa: F401  (fixture reused)

FONT = ad_compose.FONT_BOLD
DRINK = {
    "promote_target": "physical_product", "product_subtype": "drink", "target_name": "Kivu Sunrise",
    "brand_name": "Kivu Sunrise", "target_description": "A sparkling hibiscus and ginger drink from Rwanda.",
    "destination_link": "", "cta_text": "Find it in shops near you",
    "facts": {"features": ["Made with hibiscus and ginger", "330 ml can"], "prices": ["RWF 800"],
              "selling_points": ["No added sugar"]},
    "brand_colors": ["#E8452C", "#7A1E5B"], "social_links": ["https://instagram.com/kivusunrise"],
    "contact": {"phone": "+250 788 123 456"}, "character_mode": "auto",
}


# ---------------------------------------------------------------- facts and claims

def sheet(**over):
    return ad_claims.FactSheet.from_campaign({**DRINK, **over})


@pytest.mark.parametrize("line", ["Made with hibiscus and ginger.", "No added sugar.", "Just RWF 800 a can.",
                                  "330 ml of sunshine.", "Follow @kivusunrise.", "Call +250 788 123 456.",
                                  "Ice cold. Bright. Yours."])
def test_supplied_facts_are_kept(line):
    assert ad_claims.enforce(line, sheet()) == (line, [])


@pytest.mark.parametrize("line,why", [
    ("Only $1 a can.", "price"), ("Now 50% off!", "50%"), ("Packed with vitamin C.", "ingredient"),
    ("100% organic.", "organic"), ("Guaranteed to wake you up.", "guaranteed"), ("Clinically proven energy.", "clinically"),
    ("The best drink in Kigali.", "best"), ("Lasts 24 hours.", "24 hours"), ("Email hello@kivu.com.", "email"),
    ("Call 0788 999 000.", "phone"), ("Follow @kivu_deals.", "@kivu_deals"), ("Cures headaches.", "cures")])
def test_invented_claims_are_removed(line, why):
    kept, removed = ad_claims.enforce(line, sheet())
    assert kept == "" and removed and why.lower() in removed[0].lower()


def test_prices_compare_as_numbers():
    s = ad_claims.FactSheet.from_campaign({"facts": {"prices": ["1,500 Frw"]}})
    assert ad_claims.enforce("Only RWF 1500.", s)[1] == []
    assert ad_claims.enforce("Only RWF 1200.", s)[1]


def test_no_facts_means_no_specific_claims():
    s = ad_claims.FactSheet.from_campaign({"target_name": "X"})
    assert "none supplied" in s.prompt_block()
    assert ad_claims.enforce("Only $5!", s)[0] == ""


def test_only_unsupported_sentences_go():
    kept, removed = ad_claims.enforce("Made with hibiscus and ginger. Clinically proven.", sheet())
    assert kept == "Made with hibiscus and ginger." and len(removed) == 1


def test_script_enforcement_reports_each_removal():
    scenes = [{"caption_text": "100% natural", "dialogue_or_vo": "Made with hibiscus and ginger. Best in Africa!"},
              {"caption_text": "keep", "dialogue_or_vo": "Kivu Sunrise."}]
    out, report = ad_claims.enforce_script(scenes, sheet())
    assert out[0]["caption_text"] == "" and out[0]["dialogue_or_vo"] == "Made with hibiscus and ginger."
    assert len(report) == 2


# ---------------------------------------------------------------- profile validation

def test_profile_is_validated_and_kept_exactly():
    p = ads_studio._clean_profile({"facts": {"prices": [" RWF 800 "]}, "brand_colors": ["e8452c", "#7A1E5B"],
                                   "contact": {"phone": "+250 788 123 456", "email": "hi@kivu.rw"},
                                   "character_mode": "none", "ad_style": "product_cinematic"})
    assert p["facts"]["prices"] == ["RWF 800"] and p["brand_colors"] == ["#E8452C", "#7A1E5B"]
    assert p["character_mode"] == "none" and p["ad_style"] == "product_cinematic"


@pytest.mark.parametrize("bad", [{"brand_colors": ["red"]}, {"contact": {"email": "nope"}}, {"contact": {"phone": "12"}},
                                 {"character_mode": "famous_person"}, {"ad_style": "deepfake"},
                                 {"facts": {"features": ["x"] * 13}}])
def test_bad_profile_values_are_refused(bad):
    with pytest.raises(ads_studio.HTTPException):
        ads_studio._clean_profile(bad)


@pytest.mark.parametrize("kind", ["physical_product", "service", "digital_product", "event", "music", "personal_brand",
                                  "business", "film", "website", "other_app"])
def test_every_advertiser_type_has_a_prompt_description(kind):
    assert kind in ads_studio.PRODUCT_KINDS
    assert ads_studio.CreateCampaignRequest(user_id="u", promote_target=kind)


# ---------------------------------------------------------------- story and characters

def test_story_check():
    good = [{"purpose": "hook", "transition": "a"}, {"purpose": "product_intro", "transition": "b"},
            {"purpose": "selling_point", "transition": "c"}, {"purpose": "cta"}]
    assert ads_studio.story_notes(good) == []
    bad = [{"purpose": "story", "transition": ""}, {"purpose": "story", "transition": "x"}, {"purpose": "story"}]
    notes = " ".join(ads_studio.story_notes(bad))
    assert "hook" in notes and "never introduced" in notes and "call to action" in notes and "no lead" in notes


class _Parsed:
    def __init__(self, uses, ch=None):
        self.uses_character = uses
        self.character = ads_studio.AdCharacterSchema(**(ch or {}))


def test_character_is_only_used_when_chosen():
    auto = {"character_mode": "auto"}
    assert ads_studio._character_from_script(auto, _Parsed(False)) == {"character": {}}
    got = ads_studio._character_from_script(auto, _Parsed(True, {"name": "Aline", "appearance": "short curly hair"}))
    assert got["character"]["name"] == "Aline"
    assert ads_studio._character_from_script({"character_mode": "custom"}, _Parsed(True, {"name": "X"})) == {}
    assert ads_studio._character_from_script({"character_mode": "none"}, _Parsed(True, {"name": "X"})) == {}
    assert "none" in ads_studio._character_block({"character_mode": "none"}).lower()
    assert "Don't force a character" in ads_studio._character_block({"character_mode": "auto"})


def test_product_photos_go_to_product_scenes_only_for_physical_products():
    assets = [{"id": "p1", "asset_type": "product_photo"}]
    scenes = [{"purpose": "hook"}, {"purpose": "product_intro"}, {"features_product": True}, {"purpose": "cta"}]
    out = ads_studio._assign_product_photos([dict(s) for s in scenes], assets, {"p1": "x"}, "physical_product")
    assert [s.get("product_asset_id") for s in out] == [None, "p1", "p1", None]
    out = ads_studio._assign_product_photos([dict(s) for s in scenes], assets, {"p1": "x"}, "website")
    assert not any(s.get("product_asset_id") for s in out)


# ---------------------------------------------------------------- rendering and checks

def _can(path, w=500, h=900):
    """A drawn drink can standing in for the advertiser's product photo."""
    im = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([90, 60, w - 90, h - 60], 60, fill=(232, 69, 44))
    d.rectangle([90, 330, w - 90, 560], fill=(122, 30, 91))
    d.text((140, 400), "KIVU", fill="white", font=ImageFont.truetype(FONT, 90))
    im.save(path)
    return path


@pytest.mark.parametrize("w,h", [(720, 1280), (1280, 720), (720, 720)])
def test_product_hero_shot_shows_the_real_photo_and_is_checked(tmp_path, w, h):
    work = str(tmp_path)
    can, other = _can(os.path.join(work, "can.png")), _img(os.path.join(work, "other.png"), 500, 900)
    bg = _img(os.path.join(work, "bg.png"), 1024, 1024)
    ad_compose.silence(2, os.path.join(work, "s.wav"))
    tp = ad_compose.time_pages(ad_compose.layout_caption("No added sugar", w, h), 0, 2)
    drawn = ad_compose.render_hero_scene(bg, can, w, h, 2, os.path.join(work, "s.wav"), tp,
                                         os.path.join(work, "hero.mp4"), work, "h")
    frame = ad_qc.frame(os.path.join(work, "hero.mp4"), 1.0, os.path.join(work, "f.png"))
    assert ad_qc.logo_visible(frame, drawn["product"], can, max_diff=42)
    assert not ad_qc.logo_visible(frame, drawn["product"], other, max_diff=42), "a different product doesn't pass"
    lay = ad_compose.frame_layout(w, h)
    assert ad_compose.Box(**drawn["product"]).inside(lay["safe"])
    assert not any(ad_compose.Box(**c).intersects(ad_compose.Box(**drawn["product"])) for c in drawn["captions"])


def test_brand_colours_pick_readable_text():
    color, rgb = ad_compose.brand_background(["#FFE600"], 100, 100, "/tmp/_bg_test.png")
    assert color != "white", "dark text on yellow"
    color, rgb = ad_compose.brand_background(["#1A1035"], 100, 100, "/tmp/_bg_test.png")
    assert color == "white"


def test_sound_and_picture_out_of_sync_is_detected(tmp_path):
    p = os.path.join(str(tmp_path), "drift.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=720x1280:d=4", "-f", "lavfi",
                    "-i", "sine=frequency=300:duration=2.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", p],
                   check=True)
    rep = ad_qc.inspect(p, {"scenes": [], "end_card": None}, ads_studio.ad_brand.BrandFacts(), 720, 1280, str(tmp_path))
    assert next(i for i in rep["items"] if i["key"] == "av_sync")["status"] == "failed"


# ---------------------------------------------------------------- end to end: a drink brand

DRINK_SCRIPT = [
    {"order": 1, "seconds": 2.5, "purpose": "hook", "transition": "the sun hits her face", "features_character": True,
     "features_product": False, "visual_description": "A student squints into the hot afternoon sun at a Kigali bus stop",
     "camera_shot": "close-up", "dialogue_or_vo": "Kigali afternoons hit different.", "caption_text": "Kigali afternoons hit different",
     "is_cta": False, "featured_asset_id": "", "fact_ids": []},
    {"order": 2, "seconds": 2.5, "purpose": "product_intro", "transition": "the can's colour matches the sunset",
     "features_character": False, "features_product": True, "visual_description": "A cold can on a wooden market table",
     "camera_shot": "close-up", "dialogue_or_vo": "Meet Kivu Sunrise. Made with hibiscus and ginger.",
     "caption_text": "Made with hibiscus and ginger", "is_cta": False, "featured_asset_id": "", "fact_ids": ["F1"]},
    {"order": 3, "seconds": 2.5, "purpose": "selling_point", "transition": "she raises the can", "features_character": True,
     "features_product": True, "visual_description": "She sips, smiling, at golden hour",
     "camera_shot": "medium", "dialogue_or_vo": "No added sugar. 100% organic and clinically proven to energise you.",
     "caption_text": "No added sugar, only $1!", "is_cta": False, "featured_asset_id": "", "fact_ids": ["F4"]},
    {"order": 4, "seconds": 2.5, "purpose": "cta", "transition": "", "features_character": False, "features_product": False,
     "visual_description": "End card", "camera_shot": "medium", "dialogue_or_vo": "Kivu Sunrise. Follow @kivu_official.",
     "caption_text": "Buy now", "is_cta": True, "featured_asset_id": "", "fact_ids": []},
]


def _drink_campaign(pipeline, aspect="9:16", **over):
    cid = pipeline["make"](aspect, script=[dict(s) for s in DRINK_SCRIPT], assets=("logo",))
    db = ads_studio.supabase_admin
    c = db.db["ad_campaigns"][0]
    c.update({**DRINK, "user_id": str(uuid.uuid4()),
              "character": {"name": "Aline", "appearance": "a young woman with short curly hair and round glasses",
                            "clothing": "a yellow hoodie", "gender": "female", "age": "adult"}, **over})
    can = _can(os.path.join(pipeline["work"], "can.png"))
    db.db["ad_assets"].append({"id": str(uuid.uuid4()), "campaign_id": cid, "url": can, "asset_type": "product_photo",
                               "label": "can"})
    db.db.setdefault("ad_characters", [])
    return cid, can


@pytest.mark.parametrize("aspect", ["9:16", "16:9", "1:1"])
def test_drink_ad_end_to_end(pipeline, aspect):
    cid, can = _drink_campaign(pipeline, aspect)
    c, job, video = pipeline["run"](cid)
    rep = c["quality_report"]
    failed = [i for i in rep["items"] if i["status"] == "failed"]
    assert c["status"] == "ready", failed
    keys = {i["key"]: i for i in rep["items"]}
    assert keys["product"]["status"] == "passed", "the real can is visible in the product scenes"
    assert keys["character"]["status"] == "passed" and "one reference portrait" in keys["character"]["note"]
    assert keys["final_scene"]["status"] == "passed"
    # invented claims never made it to the ad
    removed = " ".join(rep["claim_corrections"])
    for claim in ("organic", "clinically", "$1", "@kivu_official"):
        assert claim in removed
    # one portrait, saved for reuse, then 2 generated settings (hook + hero background for scene 2 & 3... )
    db = ads_studio.supabase_admin
    assert len(db.db["ad_characters"]) == 1 and db.db["ad_characters"][0]["name"] == "Aline"
    assert c["character"]["portrait"] and c["character"]["voice_name"]
    if ad_qc.ocr_available():
        info = ad_qc.probe(video)
        end = ad_qc.frame(video, info["duration"] - 1.0, os.path.join(pipeline["work"], "end.png"))
        text = (ad_qc.ocr(end) or "").lower().replace(" ", "")
        assert "kivusunrise" in text and "@kivusunrise" in text


def test_product_only_cinematic_ad_has_no_character(pipeline):
    cid, can = _drink_campaign(pipeline, "9:16", character_mode="none", character={})
    c, job, video = pipeline["run"](cid)
    assert c["status"] == "ready", [i for i in c["quality_report"]["items"] if i["status"] == "failed"]
    assert not any(i["key"] == "character" for i in c["quality_report"]["items"])
    assert not ads_studio.supabase_admin.db.get("ad_characters")


def test_reused_character_makes_no_new_portrait(pipeline):
    portrait = _img(os.path.join(pipeline["work"], "aline.png"), 512, 512)
    cid, can = _drink_campaign(pipeline, "9:16", character_mode="reuse",
                               character={"name": "Aline", "appearance": "short curly hair", "portrait": portrait,
                                          "voice_name": "Kore", "saved_id": "saved-1"})
    before = pipeline["paid"]["images"]
    c, job, video = pipeline["run"](cid)
    assert c["status"] == "ready"
    assert pipeline["paid"]["images"] - before == 3, "one picture per story scene, no new portrait"
    assert not ads_studio.supabase_admin.db.get("ad_characters"), "already saved"


def test_text_reader_timeout_is_unchecked_not_a_crash(tmp_path, monkeypatch):
    img = _img(os.path.join(str(tmp_path), "f.png"), 200, 200)
    monkeypatch.setattr(ad_qc, "ocr_available", lambda: True)

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 60)
    monkeypatch.setattr(ad_qc.subprocess, "run", slow)
    assert ad_qc.ocr(img) is None
