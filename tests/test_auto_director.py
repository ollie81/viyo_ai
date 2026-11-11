"""The automatic AI director: one-click production directs every scene that
isn't made yet (camera shots planned, reviewed for cinematography, re-planned
once if weak, approved by the run), then every shot's picture and clip is
made from those decisions and the film is assembled in shot order. A creator
never has to touch the shot settings. Approved or already-made scenes are
never re-directed; a scene whose directing fails is made as one clip; unused
reservations come back.

Fake AI and local Postgres, real ffmpeg render. These tests prove the
decisions reach the providers and the film; they do NOT prove the real models
draw what was asked. That needs a real-model test.
"""
import os
import uuid

import pytest

import studio_shots
from conftest import make_user
from test_creator_studio import H, balance, env, grant, run, studio_schema  # noqa: F401
from test_video_drama import DAN, MAYA, detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_shots.sql")

TALK = {"heading": "EXT. GATES - NIGHT", "visual": "Maya faces Daniel at the manor gates in the rain", "duration_seconds": 8,
        "dialogue": [{"character": "Daniel", "text": "Drop the money and walk.", "emotion": "frightened"},
                     {"character": "Maya", "text": "Not without you."},
                     {"character": "Daniel", "text": "Then we both die here.", "emotion": "desperate"}],
        "continuity": {"transition": "new_place", "characters": [
            {"name": "Maya", "in_frame": True, "position": "left"}, {"name": "Daniel", "in_frame": True, "position": "right"}]}}
QUIET = {"heading": "EXT. ROAD - NIGHT", "visual": "Maya runs down the empty road", "duration_seconds": 4, "dialogue": [],
         "continuity": {"transition": "time_jump", "characters": [{"name": "Maya", "in_frame": True}]}}


def setup(env, style="video", credits=200000):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, credits)
    pid = project(client, u, production_style=style)["id"]
    split(client, ai, gen, u, pid, [dict(TALK), dict(QUIET)], characters=(MAYA, DAN))
    return client, db, store, ai, gen, ren, u, pid


def quote(client, u, pid, **body):
    r = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json=body)
    assert r.status_code == 200, r.text
    return r.json()


def produce(client, gen, ren, u, pid, rounds=30, **body):
    q = quote(client, u, pid, **body)
    r = client.post(f"/api/v1/studio/projects/{pid}/production", headers=H(u),
                    json={"request_id": uuid.uuid4().hex, "confirm_credits": q["total_credits"], **body})
    assert r.status_code == 200, r.text
    for _ in range(rounds):
        run(gen)
        run(ren)
    return q, client.get(f"/api/v1/studio/projects/{pid}/production", headers=H(u)).json()["production"]


# ------------------------------------------------------------------ directing is the default

def test_every_unmade_scene_is_directed_by_default_and_the_shots_are_reserved(env):
    client, db, store, ai, gen, ren, u, pid = setup(env)
    q = quote(client, u, pid)
    stages = {s["stage"]: s for s in q["stages"]}
    assert [i["label"] for i in stages["direct"]["items"]] == ["Scene 1 directing", "Scene 2 directing"]
    assert any("up to" in i["label"] and "shot pictures" in i["label"] for i in stages["images"]["items"])
    assert not any(i["label"].endswith(" picture") and "shot" not in i["label"] for i in stages["images"]["items"]), \
        "no single-picture scenes when the director takes them"
    assert q["directed_scenes"] == 2 and "reserved at most" in q["note_direct"] and q["advice"] == []
    assert studio_shots.estimate_shots(TALK) >= 3 and studio_shots.estimate_shots(QUIET) <= 2


def test_one_click_production_directs_makes_and_assembles_every_shot_without_the_creator(env):
    client, db, store, ai, gen, ren, u, pid = setup(env)
    before = balance(client, u)["available"]
    q, prod = produce(client, gen, ren, u, pid)
    assert prod["status"] == "done", prod
    sc = detail(client, u, pid)["scenes"]
    assert all(s["shot_plan_approved"] for s in sc), "the run approved the director's plans"
    talk, quiet = sc[0]["shots"], sc[1]["shots"]
    assert [x["shot_type"] for x in talk] == ["wide_two_shot", "over_the_shoulder", "reaction"]
    assert all(x["has_image"] and x["has_video"] for x in talk + quiet), "every planned shot was made"
    # The director's decisions reached the image and video models.
    pics = [x["prompt"] for x in ai.image_log if "CAMERA:" in x["prompt"]]
    assert any("CAMERA: Wide two-shot" in p for p in pics) and any("CAMERA: Over-the-shoulder" in p for p in pics)
    assert any("CAMERA: Reaction shot" in p for p in pics)
    assert "CAMERA:" in ai.last_video_prompt
    # The film follows the shot order and every shot is in it.
    r = db.execute("select duration_seconds from public.studio_renders where project_id = %s", (pid,)).fetchone()
    assert r and float(r[0]) >= sum(x["duration_seconds"] for x in talk + quiet) - 0.6
    made = db.execute("select made_with->'_direction' from public.studio_scenes where project_id = %s order by position",
                      (pid,)).fetchall()
    assert all(m[0]["auto_approved"] and m[0]["auto"] for m in made)
    spent = before - balance(client, u)["available"]
    assert 0 < spent < q["total_credits"], "the unused part of the shot reservation came back"


# ------------------------------------------------------------------ the director's own review

def test_a_weak_plan_is_rejected_by_the_review_and_re_planned(env):
    client, db, store, ai, gen, ren, u, pid = setup(env)
    calls = []

    def director(prompt, scene):
        calls.append(prompt)
        if "YOUR PLAN WAS SENT BACK" in prompt or "argue" not in (scene.get("visual") or "") and "faces" not in (scene.get("visual") or ""):
            return studio_shots.default_plan(scene, ["Maya", "Daniel"])
        return {"shots": [{"type": "medium", "description": "Both at the gate", "characters": [{"name": "Maya", "position": "left"}],
                           "line_indices": [0, 1, 2], "reason": ""}]}           # one flat shot, Daniel never seen
    ai.shots_result = director
    sid = detail(client, u, pid)["scenes"][0]["id"]
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "shotplan", "target_ids": [sid], "request_id": uuid.uuid4().hex})
    assert r.status_code == 200
    run(gen)
    sent_back = [c for c in calls if "YOUR PLAN WAS SENT BACK" in c]
    assert len(sent_back) == 1
    assert "(coverage): One shot for a conversation" in sent_back[0]
    assert "Daniel speaks but isn't in frame" in sent_back[0] and "(purpose)" in sent_back[0]
    shots = detail(client, u, pid)["scenes"][0]["shots"]
    assert len(shots) == 3, "the better plan was kept"
    d = db.execute("select made_with->'_direction' from public.studio_scenes where id = %s", (sid,)).fetchone()[0]
    assert d["replanned"] is True and d["review"] == []


@pytest.mark.parametrize("shots,kinds", [
    ([{"shot_type": "medium", "reason": "x", "characters": [{"name": "Maya"}], "dialogue": [TALK["dialogue"][0]]},
      {"shot_type": "medium", "reason": "x", "characters": [{"name": "Maya"}], "dialogue": TALK["dialogue"][1:]}],
     {"variety", "speaker", "reaction", "emotion", "geography"} - {"geography"}),
    ([{"shot_type": "wide_two_shot", "reason": "x", "characters": [{"name": "Maya", "position": "left"},
                                                                  {"name": "Daniel", "position": "right"}], "dialogue": []},
      {"shot_type": "medium", "reason": "x", "characters": [{"name": "Maya", "position": "right"}], "dialogue": []},
      {"shot_type": "close_up", "reason": "x", "characters": [{"name": "Daniel", "position": "right"}],
       "dialogue": TALK["dialogue"]}], {"screen_direction"}),
])
def test_the_cinematography_review_names_what_a_director_would_fix(shots, kinds):
    got = {p["kind"] for p in studio_shots.plan_quality(TALK, shots)}
    assert kinds <= got


def test_a_good_plan_passes_the_review():
    plan = studio_shots.clean_plan(studio_shots.default_plan(TALK, ["Maya", "Daniel"]), TALK, ["Maya", "Daniel"])
    assert studio_shots.plan_quality(TALK, plan) == []


# ------------------------------------------------------------------ what the director never overrides

def test_made_or_creator_directed_scenes_are_reused_and_failed_directing_falls_back_to_one_clip(env):
    client, db, store, ai, gen, ren, u, pid = setup(env)
    s1, s2 = (s["id"] for s in detail(client, u, pid)["scenes"])
    # Scene 2 already has its clip: reused, never directed again.
    db.execute("update public.studio_scenes set video_path = image_path, image_path = 'x.png' where id = %s", (s2,))
    db.execute("update public.studio_scenes set video_path = %s where id = %s", (f"{u}/{pid}/video/old.mp4", s2))
    q = quote(client, u, pid)
    assert [i["label"] for i in {s["stage"]: s for s in q["stages"]}["direct"]["items"]] == ["Scene 1 directing"]
    # The director fails twice on scene 1: it's made as one clip instead and the film still finishes.
    db.execute("update public.studio_scenes set video_path = null, image_path = null where id = %s", (s2,))
    ai.fail = {"plan_shots": [RuntimeError("down")] * 8}
    q, prod = produce(client, gen, ren, u, pid, rounds=40)
    assert prod["status"] == "done", prod
    scenes = detail(client, u, pid)["scenes"]
    row = db.execute("select made_with->>'_direct_failed', video_path from public.studio_scenes where id = %s", (s1,)).fetchone()
    assert row[0] == "true", "the director's note survives the picture being made"
    assert row[1], "the scene was still made, as one clip"
    assert not scenes[0]["shots"]


def test_runs_started_before_automatic_directing_keep_their_plan(env):
    import studio_production as prod
    client, db, store, ai, gen, ren, u, pid = setup(env)
    p = db.execute("select id, owner_id from public.studio_projects where id = %s", (pid,)).fetchone()
    plan = prod.plan({"id": str(p[0]), "owner_id": str(p[1])}, {"direct": False})
    assert "direct" not in [s["stage"] for s in plan["stages"]]


# ------------------------------------------------------------------ the creator can still take over

def test_the_creator_can_rearrange_shots_without_breaking_the_line_order(env):
    client, db, store, ai, gen, ren, u, pid = setup(env)
    ai.shots_result = lambda prompt, scene: {"shots": [
        {"type": "establishing_wide", "description": "The gates in the rain", "characters": [], "line_indices": [],
         "reason": "where we are"},
        {"type": "insert", "description": "The duffel bag", "characters": [], "line_indices": [], "reason": "the money"},
        {"type": "wide_two_shot", "description": "Both", "characters": [{"name": "Maya", "position": "left"},
                                                                      {"name": "Daniel", "position": "right"}],
         "line_indices": [0, 1, 2], "reason": "the argument"}]} if scene.get("dialogue") else \
        studio_shots.default_plan(scene, ["Maya"])
    sid = detail(client, u, pid)["scenes"][0]["id"]
    client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                json={"kind": "shotplan", "target_ids": [sid], "request_id": uuid.uuid4().hex})
    run(gen)
    a, b, c = (x["id"] for x in detail(client, u, pid)["scenes"][0]["shots"])
    r = client.post(f"/api/v1/studio/scenes/{sid}/shots/order", headers=H(u), json={"shot_ids": [b, a, c]})
    assert r.status_code == 200, r.text
    assert [x["id"] for x in detail(client, u, pid)["scenes"][0]["shots"]] == [b, a, c]
    ai.shots_result = None
