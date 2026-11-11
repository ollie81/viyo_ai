"""Proof that real image bytes reach the multimodal model, with the right
labels; that nothing else's images are read; and that a 100-scene production
is audited one scene at a time with honest progress and a report covering
every scene. Fake AI and local Postgres: no real AI credits."""
import io
import json
import os
import types
import uuid

import pytest
from PIL import Image

import creator_studio as cs
import studio_ai
import studio_inspect
import studio_worker
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import project

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the visual audit SQL")

COLORS = {"red": (220, 20, 20), "blue": (20, 20, 220), "green": (20, 200, 20)}


def png(color):
    out = io.BytesIO()
    Image.new("RGB", (1200, 800), COLORS[color]).save(out, "PNG")
    return out.getvalue()


def color_of(jpeg: bytes) -> str:
    """Which test colour an image really is (decoding the bytes the model got)."""
    r, g, b = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((1, 1)).getpixel((0, 0))
    return max(COLORS, key=lambda k: -sum(abs(a - c) for a, c in zip((r, g, b), COLORS[k])))


def production(env, colors, title="Night Ward", checked=0):
    """A production whose scene pictures are real files of known colours."""
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 5000)
    pid = project(client, u, title=title, production_style="pictures")["id"]
    for i, c in enumerate(colors):
        path = f"{u}/{pid}/image/s{i}.png" if c else None
        if path:
            files.put(path, png(c), "image/png")
        ins = {"result": {"scores": {"overall": 90}, "problems": [], "observed": {}}, "image_path": path,
               "signature": "x"} if i < checked else {}
        db.execute("insert into public.studio_scenes (project_id, owner_id, position, heading, visual, image_path, "
                   "inspection) values (%s, %s, %s, %s, %s, %s, %s)",
                   (pid, u, i, f"INT. WARD {i + 1} - NIGHT", f"Maya in the ward, shot {i + 1}", path, json.dumps(ins)))
    return client, db, store, files, ai, u, pid


@pytest.fixture
def wired(env, monkeypatch):
    monkeypatch.setattr(cs, "_ai", env[4])
    monkeypatch.setattr(cs, "_files", env[3])
    return env


def ask(client, u, pid, text):
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": text, "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    return r.json()


def reply():
    return {"reply": "Compared.", "proposals": [], "memory": []}


# ------------------------------------------------------------------ the bytes

def test_the_actual_scene_pictures_reach_the_model_labelled_by_production_and_scene(wired):
    client, db, store, files, ai, u, pid = production(wired, ["red", "blue", "green"])
    ai.chat_result = reply()
    r = ask(client, u, pid, "Does Maya look the same in scene 1, scene 2 and scene 3?")
    labels, sent = ai.chat_images[-1], ai.chat_image_bytes[-1]
    assert labels == ["Night Ward · scene 1 picture (INT. WARD 1 - NIGHT)", "Night Ward · scene 2 picture (INT. WARD 2 - NIGHT)",
                      "Night Ward · scene 3 picture (INT. WARD 3 - NIGHT)"]
    assert [color_of(b) for b in sent] == ["red", "blue", "green"], "the model got the real pixels of each scene"
    assert all(Image.open(io.BytesIO(b)).format == "JPEG" and max(Image.open(io.BytesIO(b)).size) <= 768 for b in sent)
    assert r["seen"] == labels
    assert "- Image 2: Night Ward · scene 2 picture" in ai.chat_prompts[-1]


def test_the_gemini_request_itself_carries_the_image_bytes():
    sent = {}

    class Models:
        def generate_content(self, model, contents, config):
            sent["contents"] = contents
            return types.SimpleNamespace(text='{"reply": "ok", "proposals": [], "memory": []}', usage_metadata=None)

    ai = object.__new__(studio_ai.GoogleStudioAI)
    ai.client = types.SimpleNamespace(models=Models())
    red, blue = png("red"), png("blue")
    ai.chat("Compare them.", mode="story", images=[("scene 1 picture", red), ("scene 2 picture", blue)])
    parts = sent["contents"]
    assert [p.text for p in parts if getattr(p, "text", None)] == ["Image 1: scene 1 picture", "Image 2: scene 2 picture",
                                                                    "Compare them."]
    blobs = [p.inline_data for p in parts if getattr(p, "inline_data", None)]
    assert [b.data for b in blobs] == [red, blue] and blobs[0].mime_type == "image/png"


def test_two_different_scenes_are_compared_side_by_side(wired):
    client, db, store, files, ai, u, pid = production(wired, ["red", "red", "red", "blue"])
    ai.chat_result = reply()
    ask(client, u, pid, "Scene 4 looks different from scene 3, why?")
    assert [color_of(b) for b in ai.chat_image_bytes[-1]] == ["red", "blue"]
    assert ["scene 3" in ai.chat_images[-1][0], "scene 4" in ai.chat_images[-1][1]] == [True, True]


def test_missing_and_unreadable_pictures_are_reported_and_never_invented(wired, monkeypatch):
    client, db, store, files, ai, u, pid = production(wired, ["red", None, "blue"])
    real = files.get
    monkeypatch.setattr(files, "get", lambda p: b"corrupt" if p.endswith("s2.png") else real(p))
    ai.chat_result = reply()
    ask(client, u, pid, "Do scenes 2 and 3 look consistent?")
    prompt = ai.chat_prompts[-1]
    assert "Night Ward · scene 2: no picture or video made yet" in prompt
    assert "scene 3 picture (INT. WARD 3 - NIGHT): couldn't be loaded" in prompt
    assert [color_of(b) for b in ai.chat_image_bytes[-1]] == [], "nothing unreadable is sent as if it were seen"


def test_pictures_stored_under_someone_else_are_never_read(wired):
    client, db, store, files, ai, u, pid = production(wired, ["red", "blue"])
    other = make_user(db)
    files.put(f"{other}/x/image/secret.png", png("green"), "image/png")
    db.execute("update public.studio_scenes set image_path = %s where project_id = %s and position = 1",
               (f"{other}/x/image/secret.png", pid))
    ai.chat_result = reply()
    ask(client, u, pid, "Compare scene 1 and scene 2.")
    assert [color_of(b) for b in ai.chat_image_bytes[-1]] == ["red"]
    assert "scene 2 picture (INT. WARD 2 - NIGHT): not accessible" in ai.chat_prompts[-1]


# ------------------------------------------------------------------ big audits

def test_a_100_scene_production_is_audited_in_small_batches_with_honest_progress(wired):
    client, db, store, files, ai, u, pid = production(wired, ["red", "blue"] * 50, title="Long Film")
    insp = studio_worker.StudioWorker(store, files, ai, "inspect")
    gen = studio_worker.StudioWorker(store, files, ai, "generation")
    ids = [str(r[0]) for r in db.execute("select id from public.studio_scenes where project_id = %s order by position",
                                         (pid,)).fetchall()]
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "inspect", "target_ids": ids}).json()
    assert est["ready_count"] == 100 and est["total_credits"] > 0, "the full price is shown first"
    ai.inspect_result = lambda prompt, labels: {"scores": {"overall": 60 if "WARD 37 " in prompt else 95},
                                                "problems": [{"kind": "outfit", "severity": "high", "certainty": "likely",
                                                              "expected": "red", "seen": "blue", "intentional": False,
                                                              "fix": "keep red"}] if "WARD 37 " in prompt else [],
                                                "summary": "x"}
    before = balance(client, u)["available"]
    assert start(client, u, pid, "inspect", ids).status_code == 200
    assert run(gen) == 0, "audits don't block the picture queue"
    run(insp, times=40)
    mid = client.get(f"/api/v1/studio/projects/{pid}/visual-audit", headers=H(u)).json()
    assert mid["checked"] == 40 and mid["queued"] == 60 and mid["complete"] is False
    ai.chat_result = reply()
    ask(client, u, pid, "Have you checked every scene?")
    prompt = ai.chat_prompts[-1]
    assert "VISUAL AUDIT (every scene of the production): 40 of 100 scene pictures checked (NOT complete); 60 queued" in prompt
    assert "Being checked now: scenes 41-100" in prompt
    run(insp, times=100)
    done = client.get(f"/api/v1/studio/projects/{pid}/visual-audit", headers=H(u)).json()
    assert done["checked"] == 100 and done["complete"] is True and len(done["scenes"]) == 100
    assert [p["scene"] for p in done["problems"]] == [37] and done["attention"] == [37]
    assert all(len(x["labels"]) <= studio_inspect.MAX_IMAGES for x in ai.inspect_log), "each request is small"
    charged = sum(j["charged_credits"] for j in client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"])
    assert before - balance(client, u)["available"] <= est["total_credits"] and charged > 0


def test_more_than_one_batch_is_needed_for_a_very_long_film(wired):
    client, db, store, files, ai, u, pid = production(wired, ["red"] * 160, title="Epic")
    ids = [str(r[0]) for r in db.execute("select id from public.studio_scenes where project_id = %s order by position",
                                         (pid,)).fetchall()]
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "inspect", "target_ids": ids}).json()
    assert est["ready_count"] == cs.MAX_AUDIT
    assert sum(1 for i in est["items"] if (i["blocked"] or "").startswith("Next batch")) == 160 - cs.MAX_AUDIT


def test_a_failed_check_is_reported_as_not_checked_with_no_findings(wired, monkeypatch):
    client, db, store, files, ai, u, pid = production(wired, ["red", "blue"])
    insp = studio_worker.StudioWorker(store, files, ai, "inspect")
    real = files.get
    monkeypatch.setattr(files, "get", lambda p: (_ for _ in ()).throw(OSError("storage down")) if p.endswith("s1.png") else real(p))
    before = balance(client, u)["available"]
    sid = str(db.execute("select id from public.studio_scenes where project_id = %s and position = 1", (pid,)).fetchone()[0])
    start(client, u, pid, "inspect", [sid])
    run(insp)
    a = client.get(f"/api/v1/studio/projects/{pid}/visual-audit", headers=H(u)).json()
    assert a["failed"] == 1 and a["problems"] == [] and a["complete"] is False
    assert db.execute("select inspection from public.studio_scenes where id = %s", (sid,)).fetchone()[0] == {}
    assert balance(client, u)["available"] == before, "a failed check is refunded"


def test_looking_and_checking_never_change_the_production(wired):
    client, db, store, files, ai, u, pid = production(wired, ["red", "blue", "green"])
    insp = studio_worker.StudioWorker(store, files, ai, "inspect")
    snap = lambda: db.execute("select id, image_path, visual, heading, continuity, image_approved_path from "
                              "public.studio_scenes where project_id = %s order by position", (pid,)).fetchall()
    before = snap()
    ai.chat_result = {"reply": "Scene 2 looks blue.", "memory": [], "proposals": [
        {"type": "fix_scene_picture", "title": "Fix scene 2", "summary": "x", "data": {"scene": 2, "correction": "red"}}]}
    ask(client, u, pid, "Check whether my characters look consistent in every scene.")
    ids = [str(r[0]) for r in before]
    start(client, u, pid, "inspect", ids)
    run(insp)
    assert snap() == before, "pictures, text and approvals untouched"
    kinds = {r[0] for r in db.execute("select kind from public.studio_jobs where project_id = %s", (pid,)).fetchall()}
    assert kinds == {"inspect"}, "no picture or video was made: the fix is only a proposal"
