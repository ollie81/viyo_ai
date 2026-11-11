"""The multi-shot director: scenes planned as ordered camera shots chosen from
the story, continuity carried from shot to shot, each shot made on its own,
approved plans and shots that are never overwritten, honest prices and
refunds, ordered rendering, and old single-clip scenes untouched.
Fake AI and local Postgres: no real AI or video credits."""
import os
import uuid

import pytest

import creator_studio as cs
import studio_ai
import studio_shots
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import DAN, MAYA, detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_shots.sql")

TALK = {"heading": "INT. KITCHEN - NIGHT", "visual": "Maya and Daniel argue across the kitchen table", "duration_seconds": 10,
        "dialogue": [{"character": "Maya", "text": "You lied to me about the letter."},
                     {"character": "Daniel", "text": "I was protecting you."},
                     {"character": "Maya", "text": "From what?"},
                     {"character": "Daniel", "text": "From our father.", "emotion": "ashamed"}],
        "continuity": {"transition": "new_place", "time_of_day": "night", "lighting": "one warm lamp",
                       "start_state": "Maya standing by the table, Daniel by the sink",
                       "characters": [{"name": "Maya", "in_frame": True, "position": "left, by the table"},
                                      {"name": "Daniel", "in_frame": True, "position": "right, by the sink"}]}}
QUIET = {"heading": "EXT. ROOF - DAWN", "visual": "Maya alone on the roof at dawn", "duration_seconds": 4, "dialogue": [],
         "continuity": {"transition": "time_jump", "characters": [{"name": "Maya", "in_frame": True}]}}
LETTER = {"heading": "EXT. ROOF - DAWN", "visual": "Maya unfolds the letter", "duration_seconds": 5, "dialogue": [],
          "continuity": {"transition": "continuous", "characters": [{"name": "Maya", "in_frame": True}]}}


@pytest.fixture
def prod(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 50000)
    pid = project(client, u, title="The Letter")["id"]          # talking-video production
    split(client, ai, gen, u, pid, [dict(TALK), dict(QUIET), dict(LETTER)], characters=(MAYA, DAN))
    return client, db, store, ai, gen, ren, u, pid


def scenes(client, u, pid):
    return detail(client, u, pid)["scenes"]


def plan(client, gen, u, pid, ids=None):
    ids = ids or [s["id"] for s in scenes(client, u, pid)]
    assert start(client, u, pid, "shotplan", ids).status_code == 200
    run(gen)
    return scenes(client, u, pid)


def approve(client, u, sid, ok=True):
    return client.post(f"/api/v1/studio/scenes/{sid}/shots/approve", headers=H(u), json={"approved": ok})


def make(client, gen, u, pid, kind, shot_ids, rid=None):
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": kind, "target_ids": shot_ids, "request_id": rid or uuid.uuid4().hex})
    run(gen)
    return r


# ------------------------------------------------------------------ planning

def test_each_scene_gets_ordered_shots_chosen_from_the_story(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "shotplan", "target_ids": [s["id"] for s in scenes(client, u, pid)]}).json()
    assert est["ready_count"] == 3 and est["total_credits"] > 0, "the price is shown first"
    sc = plan(client, gen, u, pid)
    talk, quiet, letter = (s["shots"] for s in sc)
    assert [s["shot_type"] for s in talk] == ["wide_two_shot", "over_the_shoulder", "reaction"]
    assert [s["position"] for s in talk] == [0, 1, 2]
    assert [s["shot_type"] for s in quiet] == ["establishing_wide"], "a quiet moment: one held shot"
    assert [s["shot_type"] for s in letter] == ["medium", "insert"], "the letter matters: an insert shot"
    assert all(s["reason"] for s in talk + quiet + letter)
    assert sc[0]["shot_plan_approved"] is False


def test_shot_lengths_add_up_to_the_scene_and_every_line_is_heard_once(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    for s in plan(client, gen, u, pid):
        assert abs(sum(x["duration_seconds"] for x in s["shots"]) - s["duration_seconds"]) < 0.05
        assert all(x["duration_seconds"] >= studio_shots.MIN_SHOT for x in s["shots"])
    talk = scenes(client, u, pid)[0]["shots"]
    heard = [d["text"] for x in talk for d in x["dialogue"]]
    assert heard == [d["text"] for d in TALK["dialogue"]]


def test_the_director_is_told_how_to_direct_and_what_came_before(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    plan(client, gen, u, pid)
    first, second, third = ai.shot_prompts[-3:]
    assert "Plan the camera shots for scene 1" in first and "Daniel: \"From our father.\" (ashamed)" in first
    assert "180-degree rule" in first and "No two shots in a row with the same type" in first
    assert "Its last shot: establishing_wide" in third, "the director knows the previous scene's last shot"


def test_a_messy_plan_from_the_model_is_cleaned():
    scene = {"duration_seconds": 8, "dialogue": [{"character": "Maya", "text": "One two three four five"},
                                                 {"character": "Daniel", "text": "Six seven"}]}
    raw = {"shots": [{"type": "drone", "description": "x"},
                     {"type": "medium", "duration_seconds": 30, "description": "Maya talks", "line_indices": [0, 0],
                      "characters": [{"name": "maya"}, {"name": "Stranger"}], "frame_prompt": "", "video_prompt": "v",
                      "reason": "r"},
                     {"type": "close_up", "duration_seconds": 1, "description": "Daniel", "line_indices": [],
                      "characters": [{"name": "Daniel"}], "frame_prompt": "f", "video_prompt": "v", "reason": "r"}]}
    shots = studio_shots.clean_plan(raw, scene, ["Maya", "Daniel"])
    assert [s["shot_type"] for s in shots] == ["medium", "close_up"]
    assert [c["name"] for c in shots[0]["characters"]] == ["Maya"]
    assert shots[0]["line_indices"] == [0] and shots[1]["line_indices"] == [1], "every line once, in order"
    assert abs(sum(s["duration_seconds"] for s in shots) - 8) < 0.05 and min(s["duration_seconds"] for s in shots) >= 1.5
    with pytest.raises(ValueError):
        studio_shots.clean_plan({"shots": [{"type": "nonsense"}]}, scene, ["Maya"])


def test_a_bad_plan_fails_and_is_refunded(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    before = balance(client, u)["available"]
    ai.shots_result = {"shots": []}
    start(client, u, pid, "shotplan", [scenes(client, u, pid)[0]["id"]])
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed" and scenes(client, u, pid)[0]["shots"] == []
    assert balance(client, u)["available"] == before


# ------------------------------------------------------------------ making shots

def test_shots_are_only_made_after_the_plan_is_approved(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    talk = plan(client, gen, u, pid)[0]
    r = make(client, gen, u, pid, "shot_image", [talk["shots"][0]["id"]])
    assert r.status_code == 400 and "Approve the scene's shot plan first" in r.text
    assert approve(client, u, talk["id"]).status_code == 200
    assert make(client, gen, u, pid, "shot_image", [talk["shots"][0]["id"]]).status_code == 200


def test_every_shot_keeps_continuity_with_the_previous_one(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    start(client, u, pid, "portrait")
    run(gen)
    talk = plan(client, gen, u, pid)[0]
    approve(client, u, talk["id"])
    ai.image_log = []
    make(client, gen, u, pid, "shot_image", [s["id"] for s in talk["shots"]])
    wide, ots, reaction = (x["prompt"] for x in ai.image_log)
    assert "First shot of the scene: it starts with Maya standing by the table, Daniel by the sink" in wide
    for p in (wide, ots, reaction):
        assert "Screen direction (keep it): Maya on the left; Daniel on the right" in p
        assert "Same light and time in every shot: night, one warm lamp" in p
    assert "CAMERA: Over-the-shoulder shot" in ots and "The previous shot (wide two shot) showed: Maya and Daniel" in ots
    assert "Reference image" in ots and "the previous shot of this scene (same place, light and clothes" in ots
    cast_part = reaction.split("SHOT CONTINUITY")[0]
    assert "- Daniel" in cast_part and "- Maya" not in cast_part, "only who is in the shot"


def test_each_shot_gets_its_own_clip_with_only_its_lines(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    talk = plan(client, gen, u, pid)[0]
    approve(client, u, talk["id"])
    shots = talk["shots"]
    make(client, gen, u, pid, "shot_image", [s["id"] for s in shots])
    r = make(client, gen, u, pid, "shot_video", [shots[0]["id"]])
    assert r.status_code == 200
    v = ai.last_video_prompt
    assert "You lied to me about the letter." in v and "I was protecting you." not in v
    assert "CAMERA: Wide two-shot" in v and "BEGINNING: Maya and Daniel face each other" in v
    assert scenes(client, u, pid)[0]["shots"][0]["has_video"] is True


def test_a_video_needs_its_shot_picture_and_the_price_covers_every_shot(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    talk = plan(client, gen, u, pid)[0]
    approve(client, u, talk["id"])
    ids = [s["id"] for s in talk["shots"]]
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "shot_video", "target_ids": ids}).json()
    assert est["ready_count"] == 0 and all("picture first" in i["blocked"] for i in est["items"])
    imgs = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "shot_image", "target_ids": ids}).json()
    assert imgs["ready_count"] == 3 and imgs["total_credits"] == sum(i["credits"] for i in imgs["items"])


def test_a_failed_shot_is_refunded_and_remade_alone(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    talk = plan(client, gen, u, pid)[0]
    approve(client, u, talk["id"])
    ids = [s["id"] for s in talk["shots"]]
    make(client, gen, u, pid, "shot_image", ids)
    make(client, gen, u, pid, "shot_video", ids)
    before_paths = {s["id"]: s["video_url"] for s in scenes(client, u, pid)[0]["shots"]}
    credits = balance(client, u)["available"]
    ai.fail["video"] = studio_ai.ProviderError("down")
    make(client, gen, u, pid, "shot_video", [ids[1]])
    assert balance(client, u)["available"] == credits, "a failed shot is free"
    ai.fail.pop("video")
    make(client, gen, u, pid, "shot_video", [ids[1]])
    after = scenes(client, u, pid)[0]["shots"]
    assert after[1]["has_video"] and after[0]["video_url"] == before_paths[ids[0]] and after[2]["video_url"] == before_paths[ids[2]]


def test_the_same_request_never_makes_or_charges_twice(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    talk = plan(client, gen, u, pid)[0]
    approve(client, u, talk["id"])
    rid = uuid.uuid4().hex
    client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u), json={"kind": "shot_image", "target_ids": [talk["shots"][0]["id"]], "request_id": rid})
    client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u), json={"kind": "shot_image", "target_ids": [talk["shots"][0]["id"]], "request_id": rid})
    n = db.execute("select count(*) from public.studio_jobs where kind = 'shot_image'").fetchone()[0]
    assert n == 1


def test_approved_plans_and_shots_are_never_overwritten(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    talk = plan(client, gen, u, pid)[0]
    approve(client, u, talk["id"])
    r = start(client, u, pid, "shotplan", [talk["id"]])
    assert r.status_code == 400 and "approved" in r.text
    shot = talk["shots"][0]
    make(client, gen, u, pid, "shot_image", [shot["id"]])
    assert client.post(f"/api/v1/studio/shots/{shot['id']}/approve", headers=H(u), json={"approved": True}).status_code == 200
    assert make(client, gen, u, pid, "shot_image", [shot["id"]]).status_code == 400
    assert client.patch(f"/api/v1/studio/shots/{shot['id']}", headers=H(u), json={"shot_type": "close_up"}).status_code == 409
    client.post(f"/api/v1/studio/shots/{shot['id']}/approve", headers=H(u), json={"approved": False})
    edited = client.patch(f"/api/v1/studio/shots/{shot['id']}", headers=H(u), json={"shot_type": "close_up"}).json()
    assert edited["shot_type"] == "close_up" and edited["status"] == "planned", "the old picture no longer matches"
    other = make_user(db)
    assert client.patch(f"/api/v1/studio/shots/{shot['id']}", headers=H(other), json={"duration_seconds": 3}).status_code == 404


# ------------------------------------------------------------------ the film

def test_the_film_joins_the_shots_in_order_and_old_scenes_still_work(prod):
    client, db, store, ai, gen, ren, u, pid = prod
    sc = plan(client, gen, u, pid, [scenes(client, u, pid)[0]["id"]])       # only scene 1 uses shots
    talk = sc[0]
    approve(client, u, talk["id"])
    ids = [s["id"] for s in talk["shots"]]
    r = start(client, u, pid, "render")
    assert r.status_code == 400 and "Scene 1, shot 1 has no picture yet" in r.text
    make(client, gen, u, pid, "shot_image", ids)
    assert "talking video isn't made yet" in start(client, u, pid, "render").text
    make(client, gen, u, pid, "shot_video", ids)
    for s in sc[1:]:                     # scenes 2 and 3: the old way, one picture and one clip each
        start(client, u, pid, "image", [s["id"]])
        run(gen)
        start(client, u, pid, "video", [s["id"]])
        run(gen)
    assert start(client, u, pid, "render").status_code == 200
    run(ren)
    job = next(j for j in client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"] if j["kind"] == "render")
    assert job["status"] == "succeeded", job
    dur = float(db.execute("select duration_seconds from public.studio_renders where project_id = %s", (pid,)).fetchone()[0])
    assert dur >= 10 + 4 + 5 - 0.5, "every shot and every scene is in the film"


def test_breakdown_aims_for_about_8_scenes_a_minute():
    p = studio_ai.breakdown_prompt({"kind": "episode", "aspect_ratio": "9:16"}, "INT. ROOM")
    assert "a 60-second drama is about 8 scenes" in p


# ------------------------------------------------------------------ assistant

def test_the_assistant_explains_and_proposes_shot_changes(prod, monkeypatch):
    client, db, store, ai, gen, ren, u, pid = prod
    monkeypatch.setattr(cs, "_ai", ai)
    talk = plan(client, gen, u, pid)[0]
    ai.chat_result = {"reply": "I chose a reaction shot so we see Daniel's shame.", "memory": [], "proposals": [
        {"type": "edit_shot", "title": "Closer on Daniel", "summary": "x",
         "data": {"scene": 1, "shot": 3, "shot_type": "extreme_close_up"}},
        {"type": "remake_shot_video", "title": "Remake shot 2", "summary": "x", "data": {"scene": 1, "shot": 2}},
        {"type": "remake_shot_picture", "title": "Nope", "summary": "x", "data": {"scene": 1, "shot": 9}}]}
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": "Why did you choose these shots for scene 1?", "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    prompt = ai.chat_prompts[-1]
    assert "SHOTS (3, plan not approved): 1. wide two shot" in prompt and "We need to see the line land on Daniel" in prompt
    edit, remake, bad = r.json()["reply"]["proposals"]
    assert remake["job"] == "shot_video" and remake["targets"] == [talk["shots"][1]["id"]]
    assert bad["status"] == "failed"
    mid = r.json()["reply"]["id"]
    assert client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/0/apply", headers=H(u)).status_code == 200
    assert scenes(client, u, pid)[0]["shots"][2]["shot_type"] == "extreme_close_up"
