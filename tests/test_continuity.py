"""Scene continuity: the breakdown's continuity records, the handoff from one
scene to the next in the picture and video prompts, the free scene-by-scene
review, and the assistant as continuity supervisor. Fake AI, local Postgres:
no real AI credits are spent."""
import os
import uuid

import pytest

import creator_studio as cs
import studio_ai
import studio_chat
import studio_continuity as cont
from conftest import make_user
from test_creator_studio import H, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import DAN, MAYA, detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_continuity.sql")


def maya(**kw):
    return {"name": "Maya", "in_frame": True, "outfit": "red coat", **kw}


# A woman arrives with a suitcase, finds a photograph, confronts Daniel; the
# next morning she is at a hospital in another outfit; then she phones him.
STORY = [
    {"heading": "EXT. FAMILY HOUSE - NIGHT", "visual": "Maya walks up the path to the front door carrying a red suitcase",
     "duration_seconds": 4, "dialogue": [],
     "continuity": {"purpose": "Maya comes home after years away", "transition": "new_place", "time_of_day": "night",
                    "location": "family house", "objects": ["red suitcase"],
                    "characters": [maya(position="on the path, walking to the door", holding=["red suitcase (right hand)"],
                                        holding_at_end=["red suitcase (right hand)"], emotion_start="nervous",
                                        emotion_end="nervous")],
                    "start_state": "Maya on the garden path, suitcase in her right hand",
                    "action": "She walks to the door and opens it", "end_state": "Maya steps inside, suitcase in her right hand",
                    "camera": "wide shot from the gate", "camera_movement": "static", "lighting": "porch light, rain"}},
    {"heading": "INT. HALLWAY - NIGHT", "visual": "Maya in the dark hallway, suitcase in hand",
     "duration_seconds": 4, "dialogue": [],
     "continuity": {"purpose": "She discovers the photograph", "transition": "continuous", "time_of_day": "night",
                    "location": "hallway", "objects": ["red suitcase", "photograph"],
                    "characters": [maya(position="just inside the door", holding=["red suitcase (right hand)"],
                                        holding_at_end=[], emotion_start="nervous", emotion_end="shocked")],
                    "start_state": "Maya just inside the front door, suitcase in her right hand, rain on her coat",
                    "action": "She puts the suitcase down and sees a photograph on the side table",
                    "end_state": "Maya staring at the photograph on the table, the suitcase on the floor beside her",
                    "camera": "medium shot", "camera_movement": "slow push in", "lighting": "one lamp"}},
    {"heading": "INT. HALLWAY - NIGHT", "visual": "Maya picks up the photograph as Daniel comes down the stairs",
     "duration_seconds": 6, "dialogue": [{"character": "Maya", "text": "Who is she?"},
                                         {"character": "Daniel", "text": "You weren't supposed to see that."}],
     "continuity": {"purpose": "The confrontation", "transition": "continuous", "time_of_day": "night",
                    "location": "hallway", "objects": ["photograph", "red suitcase"],
                    "characters": [maya(position="by the side table, facing the stairs", holding=[],
                                        holding_at_end=["photograph"], emotion_start="shocked", emotion_end="angry"),
                                   {"name": "Daniel", "in_frame": True, "position": "on the stairs", "enters": True,
                                    "emotion_start": "guilty", "emotion_end": "guilty", "outfit": "grey coat"}],
                    "start_state": "Maya by the table, the photograph in front of her, suitcase on the floor",
                    "action": "Maya picks up the photograph and turns to Daniel, who stops on the stairs",
                    "end_state": "Maya holding the photograph up at Daniel, Daniel frozen on the stairs",
                    "camera": "two-shot", "lighting": "one lamp"}},
    {"heading": "INT. HOSPITAL - DAY", "visual": "Maya alone in a hospital corridor in a black dress",
     "duration_seconds": 4, "dialogue": [],
     "continuity": {"purpose": "The morning after", "transition": "time_jump", "transition_note": "the next morning",
                    "time_of_day": "morning", "location": "hospital",
                    "characters": [maya(outfit="black dress", position="on a bench", emotion_start="numb")],
                    "start_state": "Maya sitting on a bench", "action": "She waits", "end_state": "She stands up",
                    "camera": "wide", "intentional_changes": ["Maya now wears the black dress", "night becomes morning"]}},
    {"heading": "INT. HOSPITAL - DAY", "visual": "Maya on the phone in the corridor",
     "duration_seconds": 4, "dialogue": [{"character": "Daniel", "text": "Come home."}],
     "continuity": {"purpose": "Daniel calls", "transition": "continuous", "time_of_day": "morning",
                    "characters": [maya(outfit="black dress", position="standing by the window"),
                                   {"name": "Daniel", "in_frame": False}],
                    "start_state": "Maya standing, phone to her ear", "action": "She listens",
                    "end_state": "She lowers the phone", "camera": "close-up"}},
]


def build(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 20000)
    p = project(client, u, production_style="pictures")
    split(client, ai, gen, u, p["id"], [dict(s) for s in STORY], characters=(MAYA, DAN))
    return client, db, ai, gen, u, p["id"]


def make_pictures(client, ai, gen, u, pid, scenes):
    prompts = {}
    for s in scenes:
        ai.image_log = []
        assert start(client, u, pid, "image", [s["id"]]).status_code == 200
        run(gen)
        prompts[s["position"]] = ai.image_log[-1]["prompt"]
    return prompts


def test_the_breakdown_is_asked_to_direct_one_continuous_film():
    p = studio_ai.breakdown_prompt({"kind": "episode", "aspect_ratio": "9:16"}, "INT. ROOM")
    assert "ONE continuous film" in p and "starting exactly where the scene before it ended" in p
    assert "Objects only appear, move or disappear" in p and "intentional_changes" in p
    scene = studio_ai.BREAKDOWN_SCHEMA["properties"]["scenes"]["items"]
    assert "continuity" in scene["required"]
    rec = scene["properties"]["continuity"]
    for k in ("start_state", "end_state", "transition", "characters", "camera", "intentional_changes", "setup_next"):
        assert k in rec["properties"]


def test_every_scene_keeps_its_continuity_record(env):
    client, db, ai, gen, u, pid = build(env)
    scenes = detail(client, u, pid)["scenes"]
    assert [s["continuity"]["transition"] for s in scenes] == ["new_place", "continuous", "continuous", "time_jump",
                                                               "continuous"]
    assert scenes[1]["continuity"]["end_state"].startswith("Maya staring at the photograph")
    assert scenes[2]["continuity"]["characters"][1]["enters"] is True


def test_each_picture_starts_where_the_scene_before_ended(env):
    client, db, ai, gen, u, pid = build(env)
    scenes = detail(client, u, pid)["scenes"]
    p = make_pictures(client, ai, gen, u, pid, scenes)
    assert "opening shot of the film" in p[0]
    # Scene 2 continues scene 1: its end state, and the previous picture as a reference.
    assert "The previous scene ended with: Maya steps inside, suitcase in her right hand" in p[1]
    assert "This scene starts with: Maya just inside the front door, suitcase in her right hand" in p[1]
    assert "holding red suitcase (right hand)" in p[1] and "the previous shot, same place and moment" in p[1]
    assert "on the path" not in p[1], "the previous scene's END, not where it started"
    assert "CAMERA: medium shot." in p[1], "the director's framing, not a generic one"
    # Scene 3: the suitcase was put down, the photograph is picked up, Daniel enters.
    assert "the suitcase on the floor beside her" in p[2]
    assert "- Maya" in p[2] and "- Daniel" in p[2], "both characters' identities are sent"
    assert "Daniel: on the stairs" in p[2]


def test_a_time_jump_and_an_outfit_change_are_new_shots_not_copies(env):
    client, db, ai, gen, u, pid = build(env)
    scenes = detail(client, u, pid)["scenes"]
    p = make_pictures(client, ai, gen, u, pid, scenes)
    assert "New sequence (time jump: the next morning)" in p[3]
    assert "the previous shot" not in p[3], "a time jump doesn't copy last night's picture"
    assert "Intended changes from the previous scene: Maya now wears the black dress" in p[3]


def test_a_voice_on_the_phone_is_not_drawn(env):
    client, db, ai, gen, u, pid = build(env)
    scenes = detail(client, u, pid)["scenes"]
    p = make_pictures(client, ai, gen, u, pid, scenes)
    assert "Only heard, NOT in the picture: Daniel" in p[4]
    assert "- Daniel" not in p[4].split("SCENE:")[0], "Daniel's identity isn't sent for a phone call"


def test_the_video_knows_the_beginning_middle_end_and_what_must_not_change(env):
    client, db, ai, gen, u, pid = build(env)
    s = detail(client, u, pid)["scenes"][1]
    start(client, u, pid, "image", [s["id"]])
    run(gen)
    start(client, u, pid, "video", [s["id"]])
    run(gen)
    v = ai.last_video_prompt
    assert "The first frame is the scene picture: keep its composition" in v
    assert "BEGINNING: Maya just inside the front door" in v
    assert "DURING: She puts the suitcase down and sees a photograph" in v
    assert "END: Maya staring at the photograph" in v
    assert "CAMERA: medium shot, slow push in" in v
    assert "MUST NOT CHANGE: Maya wears red coat" in v and "nobody else appears" in v


def test_a_clean_story_has_no_continuity_problems(env):
    client, db, ai, gen, u, pid = build(env)
    issues = detail(client, u, pid)["continuity_issues"]
    assert [i for i in issues if i["severity"] == "fix"] == []


def _scene(n, **c):
    return {"id": f"s{n}", "position": n - 1, "heading": "INT. ROOM", "visual": "", "continuity": c}


def test_review_flags_accidental_changes_but_not_intended_ones():
    a = _scene(1, transition="new_place", time_of_day="night",
               characters=[{"name": "Maya", "in_frame": True, "holding_at_end": ["letter"], "leaves": False}])
    b = _scene(2, transition="continuous", time_of_day="day",
               characters=[{"name": "Maya", "in_frame": True, "holding": []}], start_state="Maya stands still")
    outfits = {"s1": {"maya": "red"}, "s2": {"maya": "black"}}
    names = {"red": "Red dress", "black": "Black dress"}
    issues = cont.review([a, b], outfits, names, {"maya": "Maya"}, {"s1": "L1", "s2": "L2"})
    kinds = {i["kind"] for i in issues}
    assert {"outfit", "location", "time", "object"} <= kinds
    outfit = next(i for i in issues if i["kind"] == "outfit")
    assert "Red dress" in outfit["message"] and "scene 2" in outfit["message"] and outfit["scene_id"] == "s2"
    # The same changes, intended by the script and marked as a time jump to a new place: no problems.
    b2 = _scene(2, transition="time_jump", time_of_day="day", characters=[{"name": "Maya", "in_frame": True}],
                intentional_changes=["Maya changes into the black dress"])
    assert cont.review([a, b2], outfits, names, {"maya": "Maya"}, {"s1": "L1", "s2": "L2"}) == []
    # Putting the letter down in the scene is fine.
    b3 = _scene(2, transition="continuous", time_of_day="night", characters=[{"name": "Maya", "in_frame": True}],
                action="She puts the letter on the table")
    assert [i for i in cont.review([a, b3], {}, {}, {}, {}) if i["kind"] == "object"] == []


def test_review_flags_people_who_left_coming_back_without_entering():
    a = _scene(1, transition="new_place", characters=[{"name": "Daniel", "in_frame": True, "leaves": True}])
    b = _scene(2, transition="continuous", characters=[{"name": "Daniel", "in_frame": True}])
    assert [i["kind"] for i in cont.review([a, b], {}, {}, {}, {})] == ["presence"]
    b["continuity"]["characters"][0]["enters"] = True
    assert cont.review([a, b], {}, {}, {}, {}) == []


def test_remaking_one_scene_leaves_the_rest_and_flags_the_next_one(env):
    client, db, ai, gen, u, pid = build(env)
    scenes = detail(client, u, pid)["scenes"]
    make_pictures(client, ai, gen, u, pid, scenes[:3])
    before = {s["id"]: s["image_url"] for s in detail(client, u, pid)["scenes"]}
    assert not [i for i in detail(client, u, pid)["continuity_issues"] if i["kind"] == "stale"]
    make_pictures(client, ai, gen, u, pid, [scenes[1]])          # only scene 2 again
    after = detail(client, u, pid)
    paths = db.execute("select position, image_path from public.studio_scenes order by position").fetchall()
    assert paths[1][1] and before[scenes[0]["id"]] and before[scenes[2]["id"]]
    stale = [i for i in after["continuity_issues"] if i["kind"] == "stale"]
    assert [i["scene"] for i in stale] == [3], "scene 3 was made from the old scene 2"
    assert len({row[1] for row in paths[:3]}) == 3


def test_creators_can_fix_a_scenes_start_and_end(env):
    client, db, ai, gen, u, pid = build(env)
    s = detail(client, u, pid)["scenes"][1]
    r = client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u),
                     json={"continuity": {"end_state": "Maya sits on the suitcase", "transition": "continuous"}})
    assert r.status_code == 200
    c = r.json()["continuity"]
    assert c["end_state"] == "Maya sits on the suitcase" and c["start_state"].startswith("Maya just inside")


def test_the_assistant_sees_the_scenes_in_full_and_the_problems(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    _c, _d, _a, _g, u, pid = build(env)
    s2 = detail(client, u, pid)["scenes"][1]
    client.patch(f"/api/v1/studio/scenes/{s2['id']}", headers=H(u),
                 json={"continuity": {"time_of_day": "afternoon"}})
    ai.chat_result = {"reply": "Scene 2 is in the afternoon, but scene 1 ends at night.",
                      "proposals": [{"type": "rewrite_scene", "title": "Fix scene 2", "summary": "Suitcase in hand",
                                     "data": {"scene": 2, "start_state": "Maya inside the door, suitcase in her right hand",
                                              "transition": "continuous"}},
                                    {"type": "remake_picture", "title": "New picture for scene 2", "summary": "Remake it",
                                     "data": {"scene": 2}}],
                      "decisions": []}
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": "Why does scene two not continue from scene 1?", "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    prompt = ai.chat_prompts[-1]
    assert "Starts: Maya just inside the front door" in prompt and "Ends: Maya steps inside" in prompt
    assert "CONTINUITY CHECK" in prompt and "Scene 1→2 [time]" in prompt
    assert "Never say a visual problem is fixed" in prompt
    props = r.json()["reply"]["proposals"]
    assert props[1]["paid"] and props[1]["job"] == "image" and props[1]["targets"] == [s2["id"]]
    mid = r.json()["reply"]["id"]
    ok = client.post(f"/api/v1/studio/assistant/messages/{mid}/proposals/0/apply", headers=H(u))
    assert ok.status_code == 200, ok.text
    assert detail(client, u, pid)["scenes"][1]["continuity"]["start_state"] == "Maya inside the door, suitcase in her right hand"


def test_scenes_without_a_record_still_work():
    prev = {"heading": "INT. ROOM", "continuity": {}}
    cur = {"heading": "INT. ROOM", "continuity": {}}
    assert cont.transition(prev, cur) == "same_place_cut" and cont.carries_look(prev, cur)
    assert cont.video_block(prev, cur) == ""
    assert "follows straight on" in cont.image_block(prev, cur)


def test_the_chat_finds_scene_numbers_in_words():
    assert studio_chat.scenes_named("make scene two continue from scene 1") == {1, 2}
