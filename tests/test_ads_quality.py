"""Ads Maker: exact brand facts, layout in every aspect ratio, and the post-render check of the finished
file. The end-to-end tests run the real pipeline (_run_ad_generation) with real FFmpeg, real layout and
real OCR; only the paid AI calls, storage and Bunny are replaced by local stand-ins (no money spent)."""
import os
import shutil
import subprocess
import uuid

import pytest
from PIL import Image, ImageDraw, ImageFont

import ad_brand
import ad_compose
import ad_qc
import ads_studio

OLLIE = {"target_name": "Ollie", "destination_link": "https://ourollie.space", "cta_text": "Try Ollie free",
         "target_description": "An AI companion app that helps families stay organised.",
         "target_features": ["Shared family reminders", "Smart daily plans"], "promote_target": "ollie_ai"}
SIZES = [(720, 1280), (1080, 1920), (1280, 720), (1920, 1080), (720, 720), (1080, 1080)]
FONT = ad_compose.FONT_BOLD


# ---------------------------------------------------------------- 1. exact advertiser facts

def facts(**over):
    return ad_brand.BrandFacts.from_campaign({**OLLIE, **over})


def test_exact_url_is_preserved():
    f = facts()
    assert (f.url, f.host, f.display_url) == ("https://ourollie.space", "ourollie.space", "ourollie.space")
    assert ad_brand.sanitize("Get started at ourollie.space", f) == ("Get started at ourollie.space", [])
    assert ad_brand.sanitize("https://ourollie.space", f)[0] == "https://ourollie.space"


@pytest.mark.parametrize("bad", ["olliecare.com", "www.olliecare.com", "https://ollie.ai/signup", "ollie-app.io",
                                 "ourollie.com", "ourollie.space/pricing"])
def test_invented_or_substituted_website_is_replaced_with_the_real_one(bad):
    out, changes = ad_brand.sanitize(f"Visit {bad} today", facts())
    assert out == "Visit ourollie.space today" and changes


def test_spoken_invented_website_is_corrected_in_voiceover():
    out, _ = ad_brand.sanitize("Just go to ollie care dot com now", facts(), spoken=True)
    assert out == "Just go to ourollie dot space now"


def test_no_website_in_campaign_means_none_is_shown():
    f = facts(destination_link="")
    out, changes = ad_brand.sanitize("Order at kigalibakes.rw!", f)
    assert "kigalibakes" not in out and out == "Order!" and changes
    assert "NONE" in ad_brand.prompt_facts(f)


def test_invented_brand_name_variant_is_replaced():
    assert ad_brand.sanitize("OllieCare keeps you on track", facts())[0] == "Ollie keeps you on track"
    # the advertiser's own words aren't touched
    assert ad_brand.sanitize("Ollie's plans", facts())[0] == "Ollie's plans"


def test_invalid_website_is_reported_not_guessed():
    f = facts(destination_link="ourollie")
    assert f.url == "" and any("valid web address" in i for i in f.issues)


def test_script_is_sanitized_everywhere_and_cta_is_the_advertisers():
    scenes = [{"caption_text": "olliecare.com", "dialogue_or_vo": "visit olliecare.com", "visual_description": "a sign reading olliecare.com"},
              {"caption_text": "Download now!", "dialogue_or_vo": "Ollie. ourollie.space", "visual_description": "logo"}]
    out, report = ad_brand.sanitize_script(scenes, facts())
    assert out[0]["caption_text"] == "ourollie.space"
    assert out[0]["dialogue_or_vo"] == "visit ourollie dot space"
    assert "olliecare" not in out[0]["visual_description"] and "ourollie" not in out[0]["visual_description"]
    assert out[-1]["caption_text"] == "Try Ollie free" and out[-1]["is_cta"]
    assert len(report) >= 3


def test_nothing_is_hardcoded_to_one_advertiser():
    f = ad_brand.BrandFacts.from_campaign({"target_name": "Kigali Bakes", "destination_link": "kigalibakes.rw",
                                           "cta_text": "Order today"})
    assert f.display_url == "kigalibakes.rw"
    assert ad_brand.sanitize("Find us at ourollie.space", f)[0] == "Find us at kigalibakes.rw"


# ---------------------------------------------------------------- 2. layout

LONG = ("Ollie remembers every appointment, every school trip and every birthday so your whole family stays "
        "organised without anyone keeping it all in their head")


@pytest.mark.parametrize("w,h", SIZES)
def test_long_captions_wrap_inside_the_safe_band_in_every_aspect_ratio(w, h):
    lay = ad_compose.frame_layout(w, h)
    pages = ad_compose.layout_caption(LONG, w, h)
    big, small = ad_compose.font_limits(w, h)
    assert pages and " ".join(p.text for p in pages) == " ".join(LONG.split()), "nothing truncated"
    for p in pages:
        assert small <= p.font_size <= big
        assert len(p.lines) <= lay["max_lines"]
        for line, b in p.lines:
            assert b.inside(lay["caption"]) and b.inside(lay["safe"])
            assert ad_compose.text_width(line, p.font_size) <= lay["caption"].w


@pytest.mark.parametrize("w,h", SIZES)
def test_unbreakable_words_like_urls_still_fit(w, h):
    pages = ad_compose.layout_caption("https://averyveryverylongadvertiserdomainname.example/campaign/summer", w, h)
    lay = ad_compose.frame_layout(w, h)
    assert all(b.inside(lay["caption"]) for p in pages for _, b in p.lines)


@pytest.mark.parametrize("w,h", SIZES)
def test_captions_never_collide_with_logo_or_product(w, h):
    lay = ad_compose.frame_layout(w, h)
    for p in ad_compose.layout_caption(LONG, w, h):
        assert not p.box.intersects(lay["brand_mark"])
        assert not p.box.intersects(lay["content"])
        assert not p.box.intersects(lay["content_below_mark"])
    assert not lay["brand_mark"].intersects(lay["content_below_mark"])


def test_short_caption_uses_the_largest_size():
    w, h = 720, 1280
    assert ad_compose.layout_caption("Never miss a thing", w, h)[0].font_size == ad_compose.font_limits(w, h)[0]


@pytest.mark.parametrize("w,h", SIZES)
def test_end_card_shows_logo_name_website_and_cta_in_their_own_areas(w, h):
    lay = ad_compose.end_card_layout("Ollie", "ourollie.space", "Try Ollie free", w, h, has_logo=True)
    kinds = [it["kind"] for it in lay["items"]]
    assert kinds == ["logo", "name", "url", "cta"] and lay["fits"]
    boxes = [it["box"] for it in lay["items"]]
    assert all(b.inside(lay["safe"]) for b in boxes)
    assert not any(a.intersects(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])


def test_end_card_leaves_out_what_the_advertiser_didnt_give():
    lay = ad_compose.end_card_layout("Ollie", "", "", 720, 1280, has_logo=False)
    assert [it["kind"] for it in lay["items"]] == ["name"]


def test_very_long_website_wraps_on_the_end_card():
    url = "shop.averyveryverylongadvertiserdomainname.example/summer-campaign"
    lay = ad_compose.end_card_layout("X", url, "Buy", 720, 1280, has_logo=False)
    urls = [it for it in lay["items"] if it["kind"] == "url"]
    assert "".join(it["text"] for it in urls).replace(" ", "") == url
    assert all(it["box"].inside(lay["safe"]) for it in urls)


# ---------------------------------------------------------------- 3. checks on rendered frames

@pytest.fixture
def work(tmp_path):
    return str(tmp_path)


def _img(path, w, h, color=(40, 110, 150), text=None, black_screen=False, size=None):
    im = Image.new("RGB", (w, h), color)
    d = ImageDraw.Draw(im)
    d.ellipse([w * 0.2, h * 0.15, w * 0.8, h * 0.6], fill=(220, 170, 120))
    if black_screen:
        d.rectangle([w * 0.25, h * 0.2, w * 0.75, h * 0.8], fill=(4, 4, 6))
    if text:
        # A sign or shop front with the address on it, the way image models sometimes add one.
        fs = size or max(28, w // 14)
        f = ImageFont.truetype(FONT, fs)
        d.rectangle([0, h * 0.4, w, h * 0.4 + fs * 2], fill=(255, 255, 255))
        tw = f.getlength(text)
        d.text(((w - tw) / 2, h * 0.4 + fs * 0.5), text, fill=(10, 10, 10), font=f)
    im.save(path)
    return path


def test_edge_check_catches_cut_off_captions(work):
    w, h = 720, 1280
    im = Image.new("RGB", (w, h), (90, 90, 90))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 48)
    d.text((-40, 1000), "This caption runs off the edge", fill="white", font=f, stroke_width=3, stroke_fill="black")
    im.save(os.path.join(work, "clip.png"))
    assert ad_qc.edge_text(os.path.join(work, "clip.png"), ad_compose.Box(0, 980, w, 100))
    ok = Image.new("RGB", (w, h), (90, 90, 90))
    ImageDraw.Draw(ok).text((120, 1000), "Fits fine", fill="white", font=f, stroke_width=3, stroke_fill="black")
    ok.save(os.path.join(work, "ok.png"))
    assert not ad_qc.edge_text(os.path.join(work, "ok.png"), ad_compose.Box(0, 980, w, 100))


def test_black_screen_detector(work):
    assert ad_qc.flat_dark_block(_img(os.path.join(work, "b.png"), 720, 1280, black_screen=True))
    assert not ad_qc.flat_dark_block(_img(os.path.join(work, "g.png"), 720, 1280))


def test_a_dark_night_scene_is_not_mistaken_for_a_blank_screen(work):
    im = Image.new("RGB", (1024, 1024))
    d = ImageDraw.Draw(im)
    for y in range(1024):                       # a dark sky fading lighter, edge to edge
        d.line([(0, y), (1024, y)], fill=(int(8 + 60 * y / 1024), int(6 + 30 * y / 1024), int(20 + 90 * y / 1024)))
    d.ellipse([330, 300, 700, 650], fill=(230, 150, 110))
    im.save(os.path.join(work, "night.png"))
    assert not ad_qc.flat_dark_block(os.path.join(work, "night.png"))


@pytest.mark.skipif(not ad_qc.ocr_available(), reason="Tesseract not installed")
def test_ocr_reads_an_invented_website_off_a_frame(work):
    p = _img(os.path.join(work, "t.png"), 1280, 720, text="olliecare.com", size=70)
    assert "olliecare.com" in ad_brand.ocr_foreign_domains(ad_qc.ocr(p) or "", facts())


# ---------------------------------------------------------------- 4. the whole pipeline, end to end

class _Res:
    def __init__(self, data):
        self.data = data


def _match(have, want):
    if isinstance(want, tuple) and len(want) == 2 and isinstance(want[0], str) and want[0].startswith("__"):
        op, v = want
        if op == "__in__":
            return have in v
        if op == "__neq__":
            return have != v
        if have is None:
            return False
        return str(have) >= str(v) if op == "__gte__" else str(have) < str(v)
    return have == want


class _Q:
    def __init__(self, db, table):
        self.db, self.table, self.filters, self.patch, self.row = db, table, [], None, None

    def select(self, *_a, **_k):
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def order(self, *_a, **_k):
        return self

    # Filters used by the intelligence modules (in-memory versions of PostgREST's).
    def in_(self, k, values):
        values = list(values)
        self.filters.append((k, ("__in__", values)))
        return self

    def gte(self, k, v):
        self.filters.append((k, ("__gte__", v)))
        return self

    def lt(self, k, v):
        self.filters.append((k, ("__lt__", v)))
        return self

    def neq(self, k, v):
        self.filters.append((k, ("__neq__", v)))
        return self

    def range(self, *_a):
        return self

    def limit(self, *_a):
        return self

    def update(self, patch):
        self.patch = patch
        return self

    def insert(self, row):
        self.row = row
        return self

    def delete(self):
        self.deleting = True
        return self

    def execute(self):
        rows = self.db.setdefault(self.table, [])
        if getattr(self, "deleting", False):
            hit = [r for r in rows if all(_match(r.get(k), v) for k, v in self.filters)]
            self.db[self.table] = [r for r in rows if r not in hit]
            return _Res(hit)
        if self.row is not None:
            r = {"id": str(uuid.uuid4()), **self.row}
            rows.append(r)
            return _Res([r])
        hit = [r for r in rows if all(_match(r.get(k), v) for k, v in self.filters)]
        if self.patch is not None:
            for r in hit:
                r.update(self.patch)
        return _Res(hit)


class FakeDB:
    def __init__(self):
        self.db = {}

    def table(self, name):
        return _Q(self.db, name)


def _tone(path, seconds):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}",
                    "-ar", "24000", "-ac", "1", path], check=True)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    work = str(tmp_path)
    db = FakeDB()
    monkeypatch.setattr(ads_studio, "supabase_admin", db)
    images = {"queue": []}
    paid = {"images": 0, "tts": 0}

    def gen_image(prompt, refs, campaign_id):
        paid["images"] += 1
        assert "olliecare" not in prompt and "ourollie" not in prompt, "no website ever goes into an image prompt"
        maker = images["queue"].pop(0) if images["queue"] else (lambda p: _img(p, 1024, 1024))
        return maker(os.path.join(work, f"gen_{uuid.uuid4().hex}.png"))

    def tts(text, voice, campaign_id):
        paid["tts"] += 1
        assert "olliecare" not in text.replace(" ", "").replace("dot", ".")
        p = os.path.join(work, f"tts_{uuid.uuid4().hex}.wav")
        _tone(p, min(2.5, 0.3 * len(text.split())))
        return p

    uploaded = {}
    monkeypatch.setattr(ads_studio, "_generate_ad_scene_image", gen_image)
    monkeypatch.setattr(ads_studio, "_generate_ad_line_audio", tts)
    monkeypatch.setattr(ads_studio, "_auto_assign_voice", lambda *a: "Kore")
    monkeypatch.setattr(ads_studio, "_upload_image", lambda *a: "https://example.test/thumb.jpg")
    previews = []
    monkeypatch.setattr(ads_studio, "_upload_video_preview", lambda data, path: previews.append(path) or f"https://store.test/{path}")

    def bunny(path, title):
        vid = uuid.uuid4().hex
        uploaded[vid] = shutil.copy(path, os.path.join(work, f"final_{vid}.mp4"))
        return vid
    monkeypatch.setattr(ads_studio, "_upload_finished_video_to_bunny", bunny)
    monkeypatch.setattr(ads_studio, "_bunny_playback_url", lambda vid: f"https://play.test/{vid}")

    def make(aspect="9:16", script=None, assets=("screenshot", "logo"), **campaign):
        shot = Image.new("RGB", (720, 1560), (246, 247, 252))
        d = ImageDraw.Draw(shot)
        d.rectangle([0, 0, 720, 170], fill=(88, 80, 236))
        for i in range(6):
            d.rounded_rectangle([40, 230 + i * 210, 680, 400 + i * 210], radius=24, fill="white", outline=(200, 200, 225))
        shot.save(os.path.join(work, "shot.png"))
        logo = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
        ImageDraw.Draw(logo).ellipse([40, 40, 560, 560], fill=(88, 80, 236, 255))
        logo.save(os.path.join(work, "logo.png"))
        cid = str(uuid.uuid4())
        db.db["ad_campaigns"] = [{"id": cid, **OLLIE, "aspect_ratio": aspect, "resolution": "720p", "use_veo": False,
                                  "script": script or BAD_OLLIE_SCRIPT, "cost_usd_cents": 0, **campaign}]
        db.db["ad_assets"] = []
        for kind in assets:
            url = os.path.join(work, "shot.png" if kind == "screenshot" else "logo.png" if kind == "logo" else "missing.png")
            db.db["ad_assets"].append({"id": str(uuid.uuid4()), "campaign_id": cid, "url": url, "asset_type": kind,
                                       "label": kind})
        db.db["ad_generation_jobs"] = [{"id": "job1", "campaign_id": cid, "status": "pending"}]
        return cid

    def run(cid):
        ads_studio._run_ad_generation(cid, "job1")
        c = db.db["ad_campaigns"][0]
        assert c.get("status") != "failed", c.get("error")      # a crash, not a quality result
        vid = c.get("bunny_video_id")
        return c, db.db["ad_generation_jobs"][0], uploaded.get(vid)

    return {"make": make, "run": run, "images": images, "paid": paid, "work": work}


# What went wrong in the real Ollie test: the script invented "olliecare.com", one generated picture had the
# address written in it, and a phone scene came out as a black screen.
BAD_OLLIE_SCRIPT = [
    {"order": 1, "seconds": 2.5, "visual_description": "A tired mum juggles a calendar, keys and a lunchbox",
     "camera_shot": "medium", "dialogue_or_vo": "Who remembered the dentist?", "caption_text": "Who remembered the dentist?",
     "is_cta": False, "featured_asset_id": ""},
    {"order": 2, "seconds": 2.5, "visual_description": "She opens the app on her phone and sees the family plan",
     "camera_shot": "close-up", "dialogue_or_vo": "Ollie did. Everything in one family plan.",
     "caption_text": "Everything in one family plan", "is_cta": False, "featured_asset_id": ""},
    {"order": 3, "seconds": 2.5, "visual_description": "The family laughs together in the kitchen",
     "camera_shot": "wide", "dialogue_or_vo": "Sign up at olliecare.com", "caption_text": "Get OllieCare at olliecare.com",
     "is_cta": False, "featured_asset_id": ""},
    {"order": 4, "seconds": 2.5, "visual_description": "Logo", "camera_shot": "medium",
     "dialogue_or_vo": "Ollie. Visit olliecare.com", "caption_text": "Download OllieCare now!", "is_cta": True,
     "featured_asset_id": ""},
]


def _texts_drawn(campaign):
    return str(campaign.get("quality_report"))


@pytest.mark.parametrize("aspect", ["9:16", "16:9", "1:1"])
def test_ollie_ad_renders_accurately_in_every_aspect_ratio(pipeline, aspect):
    cid = pipeline["make"](aspect)
    c, job, video = pipeline["run"](cid)
    rep = c["quality_report"]
    failed = [i for i in rep["items"] if i["status"] == "failed"]
    assert c["status"] == "ready" and job["status"] == "succeeded", failed
    assert c["video_url"] and video
    if os.environ.get("AD_KEEP_DIR"):          # keep the render to look at it frame by frame
        shutil.copy(video, os.path.join(os.environ["AD_KEEP_DIR"], f"ollie_{aspect.replace(':', 'x')}.mp4"))
    w, h = ads_studio._ad_target_dimensions(aspect, "720p")
    info = ad_qc.probe(video)
    assert (info["width"], info["height"]) == (w, h) and info["has_audio"]
    # the invented domain was corrected in the script before anything was rendered
    assert any("olliecare.com" in x for x in rep["brand_corrections"])
    # the phone scene used the real screenshot instead of a generated phone
    assert pipeline["paid"]["images"] == 2, "scene 2 used the real screenshot, scenes 1 and 3 were generated"
    if ad_qc.ocr_available():
        url_item = next(i for i in rep["items"] if i["key"] == "url_visible")
        assert url_item["status"] == "passed", url_item
        end = ad_qc.frame(video, info["duration"] - 1.0, os.path.join(pipeline["work"], "end.png"))
        text = ad_qc.ocr(end) or ""
        assert "ourollie.space" in text.replace(" ", "").lower()
        assert "olliecare" not in text.lower()


@pytest.mark.skipif(not ad_qc.ocr_available(), reason="Tesseract not installed")
def test_generated_picture_with_a_fake_website_is_caught_and_corrected(pipeline):
    q = pipeline["images"]["queue"]
    q.append(lambda p: _img(p, 1024, 1024, text="olliecare.com", size=64))   # scene 1: the model wrote text
    cid = pipeline["make"]("9:16")
    c, job, video = pipeline["run"](cid)
    rep = c["quality_report"]
    assert c["status"] == "ready", [i for i in rep["items"] if i["status"] == "failed"]
    assert any("Scene 1" in x for x in rep["corrections"]), rep["corrections"]


def test_black_phone_screen_is_caught_and_corrected(pipeline):
    q = pipeline["images"]["queue"]
    q.append(lambda p: _img(p, 1024, 1024, black_screen=True))          # scene 1 came out with a blank screen
    cid = pipeline["make"]("9:16")
    c, job, video = pipeline["run"](cid)
    rep = c["quality_report"]
    assert c["status"] == "ready", [i for i in rep["items"] if i["status"] == "failed"]
    assert any("Scene 1" in x for x in rep["corrections"])


def test_uncorrectable_problem_is_never_delivered_as_finished(pipeline, monkeypatch):
    q = pipeline["images"]["queue"]
    for _ in range(4):
        q.append(lambda p: _img(p, 1024, 1024, black_screen=True))      # every attempt comes out blank
    cid = pipeline["make"]("9:16", assets=("logo",))                    # no screenshot to fall back on
    c, job, video = pipeline["run"](cid)
    assert c["status"] == "needs_correction" and job["status"] == "needs_correction"
    assert c["video_url"] is None and c["preview_video_url"], "a preview to look at, never a finished ad"
    assert "blank" in c["error"].lower() or "black" in c["error"].lower()


def test_missing_asset_is_reported(pipeline):
    cid = pipeline["make"]("9:16", assets=("screenshot", "logo", "broken"))
    c, job, video = pipeline["run"](cid)
    assert c["status"] == "needs_correction"
    assert "couldn't be loaded" in c["error"]


def test_no_website_campaign_shows_none(pipeline):
    cid = pipeline["make"]("1:1", destination_link="")
    c, job, video = pipeline["run"](cid)
    assert c["status"] == "ready", [i for i in c["quality_report"]["items"] if i["status"] == "failed"]
    if ad_qc.ocr_available():
        end = ad_qc.frame(video, ad_qc.probe(video)["duration"] - 1.0, os.path.join(pipeline["work"], "end.png"))
        assert "olliecare" not in (ad_qc.ocr(end) or "").lower()


# ---------------------------------------------------------------- 5. more checks on real files

def _video(path, seconds=3, audio=True, black_from=None, w=720, h=1280):
    """A real test video: a coloured picture (optionally black for its second half), with or without sound."""
    vf = f"color=c=0x2a6f99:s={w}x{h}:d={seconds}"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", vf]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}"]
    if black_from is not None:
        cmd += ["-vf", f"drawbox=enable='gte(t,{black_from})':x=0:y=0:w=iw:h=ih:color=black:t=fill"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac", "-shortest"] if audio else []) + [path]
    subprocess.run(cmd, check=True)
    return path


def test_missing_audio_is_detected(work):
    plan = {"scenes": [{"index": 0, "start": 0, "end": 3, "kind": "generated", "has_voice": True, "drawn": {}}],
            "end_card": None}
    rep = ad_qc.inspect(_video(os.path.join(work, "mute.mp4"), audio=False), plan, facts(), 720, 1280, work)
    assert next(i for i in rep["items"] if i["key"] == "audio_track")["status"] == "failed"
    assert not rep["passed"]


def test_unexpected_silence_in_a_voiceover_scene_is_detected(work):
    p = os.path.join(work, "silent.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x2a6f99:s=720x1280:d=3", "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=stereo", "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    p], check=True)
    plan = {"scenes": [{"index": 0, "start": 0, "end": 3, "kind": "generated", "has_voice": True, "drawn": {}}],
            "end_card": None}
    rep = ad_qc.inspect(p, plan, facts(), 720, 1280, work)
    assert any(i["key"] == "voice" and i["status"] == "failed" and i["scene"] == 0 for i in rep["items"])


def test_black_frames_in_a_real_video_are_detected(work):
    p = _video(os.path.join(work, "black.mp4"), seconds=4, black_from=2)
    spans = ad_qc.black_intervals(p)
    assert spans and spans[0][0] == pytest.approx(2.0, abs=0.2)
    assert not ad_qc.black_intervals(_video(os.path.join(work, "ok.mp4"), seconds=3))


def test_wrong_size_is_detected(work):
    plan = {"scenes": [], "end_card": None}
    rep = ad_qc.inspect(_video(os.path.join(work, "landscape.mp4"), w=1280, h=720), plan, facts(), 720, 1280, work)
    assert next(i for i in rep["items"] if i["key"] == "format")["status"] == "failed"


@pytest.mark.parametrize("w,h", SIZES)
def test_logo_has_a_reserved_place(w, h):
    lay = ad_compose.frame_layout(w, h)
    mark = lay["brand_mark"]
    assert mark.inside(lay["safe"]) and not mark.intersects(lay["caption"])
    end = ad_compose.end_card_layout("Ollie", "ourollie.space", "Try Ollie free", w, h, has_logo=True)
    logo = next(it["box"] for it in end["items"] if it["kind"] == "logo")
    assert logo.inside(lay["safe"])
    for it in end["items"]:
        if it["kind"] != "logo":
            assert not it["box"].intersects(logo)


def test_captions_never_appear_on_the_end_card_with_the_cta(pipeline):
    cid = pipeline["make"]("9:16")
    c, job, video = pipeline["run"](cid)
    # What the end card drew is only the advertiser's facts; the closing scene's caption isn't drawn there.
    texts = c["quality_report"]["items"]
    assert c["status"] == "ready", [i for i in texts if i["status"] == "failed"]
    final = next(i for i in texts if i["key"] == "final_scene")
    assert final["status"] == "passed" and "url" in final["note"] and "cta" in final["note"]


def test_a_failed_ad_is_never_uploaded_to_bunny(pipeline):
    for _ in range(4):
        pipeline["images"]["queue"].append(lambda p: _img(p, 1024, 1024, black_screen=True))
    cid = pipeline["make"]("9:16", assets=("logo",))
    c, job, video = pipeline["run"](cid)
    assert c["status"] == "needs_correction" and c["bunny_video_id"] is None and video is None
    assert c["preview_video_url"], "kept privately as a preview"
