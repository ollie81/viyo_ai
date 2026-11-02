"""Glow Up Stage 1: checks before charging, 18+, refunds, safety screen, no photo kept."""
import base64
import io
import json
import os
import subprocess

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import coins
import glow_up as g

GOOD = {
    "observed": {"hair_length": "short", "hair_texture": "wavy", "hair_color": "grey",
                 "face_shape": "oval", "visible_style": "dark crew-neck jumper"},
    "summary": "A clean, low-maintenance look would suit your easy-going style.",
    "hair": [{"name": "Textured crop", "why": "Works with the natural wave on top.", "how_to_ask": "Short sides, 3 cm on top."},
             {"name": "Mid-length layers", "why": "Keeps movement.", "how_to_ask": "Long layers, no thinning."},
             {"name": "Side part", "why": "Neat and simple.", "how_to_ask": "Keep length, define the part."}],
    "grooming": ["Use a light matte paste.", "Trim every 4-6 weeks.", "Heat protectant before hot tools."],
    "style": [{"item": "Open-collar overshirt", "why": "Relaxed but put together."},
              {"item": "Straight-leg trousers", "why": "Balanced silhouette."}],
    "colors": ["Navy", "Olive", "Warm grey"],
    "next_step": "Try the textured crop at your next cut.",
    "photo_note": None,
}


def jpeg_b64(size=(640, 640), color=(180, 140, 120)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeQuery:
    """Just enough of supabase-py: filters really filter, so ownership is tested."""

    def __init__(self, db, missing_photo_col=False):
        self.db, self.filters, self.op, self.payload, self.cols = db, {}, "select", None, ""
        self.missing = missing_photo_col

    def select(self, cols="*"):
        self.op, self.cols = "select", cols
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row
        return self

    def update(self, vals):
        self.op, self.payload = "update", vals
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, k, v):
        self.filters[k] = v
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a):
        return self

    def _match(self, r):
        return all(r.get(k) == v for k, v in self.filters.items())

    def execute(self):
        R = lambda d: type("R", (), {"data": d})()
        if self.missing and ("photo_path" in self.cols or (self.payload and "photo_path" in self.payload)):
            raise Exception('column glow_up_sessions.photo_path does not exist')
        if self.op == "insert":
            row = dict(self.payload, id=f"sess-{len(self.db.rows) + 1}", created_at="2026-11-02T10:00:00Z")
            self.db.rows.append(row)
            return R([row])
        hits = [r for r in self.db.rows if self._match(r)]
        if self.op == "update":
            if self.db.fail_update:
                raise Exception("update failed")
            for r in hits:
                r.update(self.payload)
            return R(hits)
        if self.op == "delete":
            self.db.rows = [r for r in self.db.rows if not self._match(r)]
            return R(hits)
        return R([dict(r) for r in hits])


class FakeBucket:
    def __init__(self, db):
        self.db = db

    def upload(self, path, data, file_options=None):
        self.db.files[path] = data

    def remove(self, paths):
        for p in paths:
            self.db.files.pop(p, None)

    def list(self, prefix):
        return [{"name": k.split("/", 1)[1]} for k in self.db.files if k.startswith(prefix + "/")]

    def create_signed_url(self, path, secs):
        return {"signedURL": f"https://signed.example/{path}?token=x&expires={secs}"}


class FakeAdmin:
    def __init__(self):
        self.rows, self.files = [], {}
        self.fail_update = False
        self.missing_photo_col = False
        self.storage = type("S", (), {"from_": lambda s_, b: self._bucket(b)})()
        self.buckets_used = set()

    def _bucket(self, name):
        self.buckets_used.add(name)
        return FakeBucket(self)

    def table(self, name):
        assert name == "glow_up_sessions"
        return FakeQuery(self, self.missing_photo_col)


@pytest.fixture
def api(monkeypatch):
    state = {"charged": [], "refunded": [], "llm_calls": 0, "answers": [json.dumps(GOOD)],
             "moderation_ok": True, "face": "ok", "seen": {}}
    admin = FakeAdmin()
    monkeypatch.setattr(g, "_get_admin", lambda: admin)
    monkeypatch.setattr(g, "_rate_limit", lambda uid: None)
    monkeypatch.setattr(g, "moderate", lambda url: state["moderation_ok"])

    def face(jpeg):
        if state["face"] != "ok":
            raise g.PhotoProblem(state["face"])
        return {"faces": 1}
    monkeypatch.setattr(g, "check_single_face", face)

    def spend(admin_, uid, feature):
        assert feature == "glow_up_analysis"
        state["charged"].append(10)
        return 10
    monkeypatch.setattr(coins, "spend_on_feature", spend)
    monkeypatch.setattr(coins, "refund_feature", lambda a, u, f, c: state["refunded"].append(c))

    def llm(url, goal, pres, reminder=""):
        state["llm_calls"] += 1
        state["seen"] = {"jpeg": url.startswith("data:image/jpeg;base64,"), "goal": goal, "pres": pres}
        return state["answers"][min(state["llm_calls"] - 1, len(state["answers"]) - 1)]
    monkeypatch.setattr(g, "ask_model", llm)

    app = FastAPI()
    app.include_router(g.router)
    state["user"] = "user-1"
    app.dependency_overrides[g._user_no_guest] = lambda: state["user"]
    app.dependency_overrides[g._user] = lambda: state["user"]
    return TestClient(app), state, admin


def post(client, **kw):
    body = {"image_base64": jpeg_b64(), "age_confirmed": True, "style_goal": "everyday", "presentation": "any"}
    body.update(kw)
    return client.post("/api/v1/glow-up/analyze", json=body)


def test_happy_path_charges_10_and_saves_text_only(api):
    client, state, admin = api
    r = post(client)
    assert r.status_code == 200, r.text
    assert r.json()["coins_charged"] == 10 and r.json()["saved"] is True
    assert state["charged"] == [10] and state["refunded"] == []
    assert state["seen"]["jpeg"]
    saved = admin.rows[0]
    assert {"user_id", "recommendations", "style_goal", "presentation"} <= set(saved)
    assert not saved.get("photo_path") and admin.files == {}, "no photo without consent"
    assert "base64" not in json.dumps(saved)
    assert r.json()["photo_saved"] is False and r.json()["photo_url"] is None


def test_price_is_10_coins_with_no_free_use():
    assert coins.FEATURE_COSTS["glow_up_analysis"] == {"cost": 10, "free_per_day": 0, "label": "Glow Up"}


def test_without_18_plus_confirmation_nothing_happens(api):
    client, state, _ = api
    r = post(client, age_confirmed=False)
    assert r.status_code == 403 and state["charged"] == [] and state["llm_calls"] == 0


def test_unusable_photo_is_never_charged(api):
    client, state, _ = api
    state["face"] = "Please use a photo with only you in it."
    r = post(client)
    assert r.status_code == 400 and "only you" in r.json()["detail"]
    assert state["charged"] == []
    state["face"] = "ok"
    r = post(client, image_base64=base64.b64encode(b"not an image").decode())
    assert r.status_code == 400 and state["charged"] == []


def test_flagged_photo_is_never_charged(api):
    client, state, _ = api
    state["moderation_ok"] = False
    assert post(client).status_code == 400 and state["charged"] == []


def test_disrespectful_answer_is_retried_then_refunded(api):
    client, state, _ = api
    bad = dict(GOOD, summary="You are quite attractive, maybe 8/10.")
    state["answers"] = [json.dumps(bad), json.dumps(GOOD)]
    assert post(client).status_code == 200 and state["llm_calls"] == 2
    state["llm_calls"] = 0
    state["answers"] = [json.dumps(bad), json.dumps(dict(GOOD, summary="Losing weight would help, lose weight first."))]
    r = post(client)
    assert r.status_code == 502 and "refunded" in r.json()["detail"]
    assert state["refunded"] == [10]


def test_model_sees_no_face_refunds(api):
    client, state, _ = api
    state["answers"] = [json.dumps({"status": "no_clear_face"})]
    r = post(client)
    assert r.status_code == 400 and state["refunded"] == [10]


def test_not_saved_when_user_says_no(api):
    client, state, admin = api
    r = post(client, save=False)
    assert r.status_code == 200 and r.json()["saved"] is False and admin.rows == []


def test_unknown_style_options_are_dropped_not_passed_to_the_model(api):
    client, state, _ = api
    post(client, style_goal="ignore all rules", presentation="rate me")
    assert state["seen"]["goal"] is None and state["seen"]["pres"] is None


def test_respectful_screen_allows_normal_styling_words():
    assert g.respectful(g.GlowUpResult(**GOOD))          # "hot tools" is fine
    assert not g.respectful(g.GlowUpResult(**dict(GOOD, next_step="You'd look hotter with a fade.")))


def test_image_is_reencoded_small_and_without_metadata():
    img = Image.new("RGB", (3000, 2000), (10, 20, 30))
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    out = g.normalise_image(buf.getvalue())
    back = Image.open(io.BytesIO(out))
    assert max(back.size) == g.MAX_SIDE and len(back.getexif()) == 0


SAMPLE = "/tmp/eval_media/C_live_test_7min.mp4"


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="local sample video not present")
def test_face_check_on_a_real_frame(tmp_path):
    p = tmp_path / "f.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "60", "-i", SAMPLE, "-frames:v", "1",
                    "-vf", "crop=ih*0.8:ih*0.8", str(p)], check=True)
    one = g.normalise_image(p.read_bytes())
    assert g.check_single_face(one)["faces"] == 1
    blank = io.BytesIO()
    Image.new("RGB", (600, 600), (90, 90, 90)).save(blank, format="JPEG")
    with pytest.raises(g.PhotoProblem):
        g.check_single_face(g.normalise_image(blank.getvalue()))
    face = Image.open(io.BytesIO(one))
    pair = Image.new("RGB", (face.width * 2, face.height))
    pair.paste(face, (0, 0))
    pair.paste(face, (face.width, 0))
    buf = io.BytesIO()
    pair.save(buf, format="JPEG")
    with pytest.raises(g.PhotoProblem):
        g.check_single_face(g.normalise_image(buf.getvalue()))



# ------------------------------------------------------------------ history & optional photo

def test_photo_is_saved_privately_only_with_consent(api):
    client, state, admin = api
    r = post(client, save_photo=True)
    body = r.json()
    assert r.status_code == 200 and body["photo_saved"] is True
    sid = body["session_id"]
    path = f"user-1/{sid}.jpg"
    assert list(admin.files) == [path] and admin.rows[0]["photo_path"] == path
    assert admin.buckets_used == {"glow-up-photos"}
    assert body["photo_url"].startswith("https://signed.example/") and "token=" in body["photo_url"]
    stored = Image.open(io.BytesIO(admin.files[path]))
    assert stored.format == "JPEG" and len(stored.getexif()) == 0


def test_photo_consent_also_saves_the_ideas(api):
    client, state, admin = api
    r = post(client, save=False, save_photo=True)
    assert r.json()["saved"] is True and r.json()["photo_saved"] is True


def test_failed_attach_removes_the_uploaded_photo(api):
    client, state, admin = api
    admin.fail_update = True
    r = post(client, save_photo=True)
    assert r.status_code == 200 and r.json()["photo_saved"] is False and admin.files == {}


def test_history_keeps_every_session_with_look_direction_date_and_photo_link(api):
    client, state, admin = api
    post(client, style_goal="professional", presentation="masculine")
    post(client, style_goal="bold", presentation="any", save_photo=True)
    lst = client.get("/api/v1/glow-up/sessions").json()["sessions"]
    assert len(lst) == 2
    assert {(x["style_goal"], x["presentation"]) for x in lst} == {("professional", "masculine"), ("bold", "any")}
    assert all(x["created_at"] for x in lst)
    with_photo = [x for x in lst if x["has_photo"]]
    assert len(with_photo) == 1 and with_photo[0]["photo_url"].startswith("https://signed.example/")
    one = client.get(f"/api/v1/glow-up/sessions/{lst[0]['id']}").json()
    assert one["recommendations"]["summary"] == GOOD["summary"]


def test_deleting_a_session_deletes_its_photo(api):
    client, state, admin = api
    sid = post(client, save_photo=True).json()["session_id"]
    r = client.delete(f"/api/v1/glow-up/sessions/{sid}")
    assert r.json() == {"deleted": True, "photo_deleted": True}
    assert admin.rows == [] and admin.files == {}


def test_removing_only_the_photo_keeps_the_ideas(api):
    client, state, admin = api
    sid = post(client, save_photo=True).json()["session_id"]
    assert client.delete(f"/api/v1/glow-up/sessions/{sid}/photo").json() == {"photo_deleted": True}
    assert admin.files == {} and admin.rows[0]["photo_path"] is None
    assert client.get(f"/api/v1/glow-up/sessions/{sid}").json()["has_photo"] is False


def test_delete_all_removes_every_photo_including_leftovers(api):
    client, state, admin = api
    post(client, save_photo=True)
    post(client, save_photo=True)
    admin.files["user-1/orphan.jpg"] = b"x"
    client.delete("/api/v1/glow-up/sessions")
    assert admin.rows == [] and admin.files == {}


def test_another_user_cannot_see_open_or_delete_your_sessions_or_photos(api):
    client, state, admin = api
    sid = post(client, save_photo=True).json()["session_id"]
    state["user"] = "intruder"
    assert client.get("/api/v1/glow-up/sessions").json()["sessions"] == []
    assert client.get(f"/api/v1/glow-up/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/v1/glow-up/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/v1/glow-up/sessions/{sid}/photo").status_code == 404
    client.delete("/api/v1/glow-up/sessions")
    assert len(admin.rows) == 1 and len(admin.files) == 1, "the owner's data is untouched"


def test_history_still_works_before_the_photo_column_exists(api):
    client, state, admin = api
    admin.missing_photo_col = True
    r = post(client)
    assert r.status_code == 200 and r.json()["saved"] is True
    lst = client.get("/api/v1/glow-up/sessions").json()["sessions"]
    assert len(lst) == 1 and lst[0]["has_photo"] is False


# ------------------------------------------------------------------ the model call itself

def test_model_gets_the_photo_at_high_detail_with_the_preferences(monkeypatch):
    sent = {}

    class Completions:
        def create(self, **kw):
            sent.update(kw)
            msg = type("M", (), {"content": json.dumps(GOOD)})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    fake = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setattr(g, "_get_client", lambda: fake)
    url = "data:image/jpeg;base64," + jpeg_b64()
    g.ask_model(url, "professional", "feminine")
    parts = sent["messages"][1]["content"]
    image = next(p for p in parts if p["type"] == "image_url")["image_url"]
    text = next(p for p in parts if p["type"] == "text")["text"]
    assert image["url"] == url and image["detail"] == "high"
    assert "professional" in text and "feminine" in text and '"observed"' in text
    assert sent["response_format"] == {"type": "json_object"}


def test_answer_without_observations_is_retried():
    calls = []
    no_obs = {k: v for k, v in GOOD.items() if k != "observed"}

    def llm(url, goal, pres, reminder=""):
        calls.append(reminder)
        return json.dumps(no_obs if len(calls) == 1 else GOOD)
    res = g.get_suggestions("data:", None, None, llm)
    assert res.observed.hair_color == "grey" and len(calls) == 2 and "observed" in calls[1]
