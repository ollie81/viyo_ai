"""Places and objects that stay the same in every scene. Fake AI, local Postgres."""
import os
import uuid

import pytest

import studio_elements as world
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import TALK, detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_elements.sql")

ROOF = dict(TALK, heading="EXT. ROOFTOP - NIGHT", visual="Maya clutches her black leather bag at the railing")
FLAT = {"heading": "INT. FLAT - DAY", "visual": "Maya drops her bag on the sofa", "duration_seconds": 4, "dialogue": []}
ROOF2 = {"heading": "EXT. ROOFTOP - LATER", "visual": "Daniel alone at the railing", "duration_seconds": 4, "dialogue": []}


def setup(env, scenes=(ROOF, FLAT, ROOF2), style="pictures"):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 5000)
    p = project(client, u, production_style=style)
    split(client, ai, gen, u, p["id"], list(scenes))
    client.patch(f"/api/v1/studio/projects/{p['id']}", headers=H(u), json={"script": "EXT. ROOFTOP - NIGHT\n\nMaya..."})
    return client, db, ai, gen, ren, u, p["id"]


def find(client, ai, gen, u, pid):
    assert start(client, u, pid, "elements").status_code == 200
    run(gen)
    return detail(client, u, pid)


def test_the_ai_finds_places_and_objects_and_assigns_scenes(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "elements"}).json()
    assert est["ready_count"] == 1 and est["total_credits"] <= 10
    d = find(client, ai, gen, u, pid)
    assert "production designer" in ai.last_elements_prompt and "EXT. ROOFTOP - NIGHT" in ai.last_elements_prompt
    els = {e["name"]: e for e in d["elements"]}
    assert els["Rooftop"]["kind"] == "location" and els["Maya's black leather bag"]["kind"] == "prop"
    assert els["Rooftop"]["details"] == {"time_of_day": "night", "mood": "tense"}
    s = d["scenes"]
    assert s[0]["location_id"] == els["Rooftop"]["id"] and s[0]["props"] == [els["Maya's black leather bag"]["id"]]
    # Scene 3 wasn't assigned but its heading names the rooftop; scene 2 mentions the bag.
    assert s[2]["location_used"]["name"] == "Rooftop"
    assert [x["name"] for x in s[1]["props_used"]] == ["Maya's black leather bag"]
    # Running it again adds nothing twice.
    d = find(client, ai, gen, u, pid)
    assert len(d["elements"]) == 2


def test_each_place_and_object_gets_a_picture_used_in_every_scene(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    d = find(client, ai, gen, u, pid)
    ids = [e["id"] for e in d["elements"]]
    ai.image_log = []
    assert start(client, u, pid, "element", ids).status_code == 200
    run(gen)
    prompts = [x["prompt"] for x in ai.image_log]
    assert any("Establishing shot of the empty set" in x for x in prompts) and any("Prop reference photo" in x for x in prompts)
    assert all(e["has_image"] for e in detail(client, u, pid)["elements"])
    assert detail(client, u, pid)["project"]["covers"] == [], "place and object pictures are not covers"
    ai.image_log = []
    start(client, u, pid, "image", [s["id"] for s in d["scenes"]])
    run(gen)
    roof, flat, roof2 = (x["prompt"] for x in ai.image_log)
    for prompt in (roof, roof2):
        assert "LOCATION (the same place every time" in prompt and "rusty railing" in prompt
        assert "the location, Rooftop (how the place looks" in prompt
    assert "OBJECTS (exactly the same in every scene" in roof and "gold clasp" in roof
    assert "the object Maya's black leather bag" in flat and "LOCATION" not in flat
    assert [x["refs"] for x in ai.image_log] == [2, 1, 1], "place + object; object; place"


def test_the_creator_controls_a_scenes_place_and_objects(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    d = find(client, ai, gen, u, pid)
    els = {e["name"]: e for e in d["elements"]}
    r = client.post(f"/api/v1/studio/projects/{pid}/elements", headers=H(u),
                    json={"kind": "location", "name": "Maya's flat", "description": "small bright flat, green sofa",
                          "details": {"time_of_day": "day", "junk": "x"}})
    flat = r.json()
    assert r.status_code == 200 and flat["details"] == {"time_of_day": "day"}
    s = d["scenes"]
    assert client.patch(f"/api/v1/studio/scenes/{s[1]['id']}", headers=H(u),
                        json={"location_id": flat["id"], "props": []}).status_code == 200
    s2 = detail(client, u, pid)["scenes"][1]
    assert s2["location_used"]["name"] == "Maya's flat" and s2["props_used"] == []
    assert client.patch(f"/api/v1/studio/scenes/{s[1]['id']}", headers=H(u), json={"props_auto": True}).status_code == 200
    assert [x["name"] for x in detail(client, u, pid)["scenes"][1]["props_used"]] == ["Maya's black leather bag"]
    # A prop can't be a scene's place; deleting an element frees the scenes.
    assert client.patch(f"/api/v1/studio/scenes/{s[1]['id']}", headers=H(u),
                        json={"location_id": els["Maya's black leather bag"]["id"]}).status_code == 400
    assert client.delete(f"/api/v1/studio/elements/{flat['id']}", headers=H(u)).status_code == 200
    assert detail(client, u, pid)["scenes"][1]["location_id"] is None
    # Approve needs a picture; edits work.
    bag = els["Maya's black leather bag"]["id"]
    assert client.patch(f"/api/v1/studio/elements/{bag}", headers=H(u), json={"status": "approved"}).status_code == 409
    r = client.patch(f"/api/v1/studio/elements/{bag}", headers=H(u), json={"description": "small RED leather bag"})
    assert r.json()["description"] == "small RED leather bag"


def test_only_the_owner_sees_or_changes_places_and_objects(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    d = find(client, ai, gen, u, pid)
    other = make_user(db)
    eid = d["elements"][0]["id"]
    assert client.patch(f"/api/v1/studio/elements/{eid}", headers=H(other), json={"name": "x"}).status_code == 404
    assert client.delete(f"/api/v1/studio/elements/{eid}", headers=H(other)).status_code == 404
    assert client.post(f"/api/v1/studio/projects/{pid}/elements", headers=H(other),
                       json={"kind": "prop", "name": "x"}).status_code == 404


def test_next_episode_and_one_click_production_use_them(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    find(client, ai, gen, u, pid)
    q = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json={"sheets": False}).json()
    assert q["stages"][0]["stage"] == "places" and len(q["stages"][0]["items"]) == 2
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u)).json()["id"]
    assert {e["name"] for e in detail(client, u, ep2)["elements"]} == {"Rooftop", "Maya's black leather bag"}


def test_matching_rules():
    roof = {"id": "r", "kind": "location", "name": "Vance Mansion - Living Room"}
    bag = {"id": "b", "kind": "prop", "name": "Maya's black leather bag"}
    s = {"heading": "INT. VANCE MANSION - NIGHT", "visual": "She grabs the bag", "dialogue": []}
    assert world.scene_location(s, [roof, bag]) is roof and world.scene_props(s, [roof, bag]) == [bag]
    assert world.scene_props(dict(s, props=[]), [roof, bag]) == []
    assert world.scene_location({"heading": "EXT. PARK", "visual": ""}, [roof]) is None


def test_framing_comes_from_the_scene_not_the_place_picture():
    import studio_identity as ident
    loc = {"name": "Vance mansion", "description": "Gothic gates", "reference_path": "x.png", "details": {}}
    assert "only the part of it this shot would see" in world.world_block(loc, [])
    assert "not like this picture" in world.reference_labels(loc, [], 4)[0][0]
    empty = ident.scene_prompt("Extreme close-up of a phone in a puddle", [], "front", "", "9:16")
    assert "characters" not in empty.split("SCENE:")[1].split("STYLE")[0].replace("people the SCENE", "")
    assert "Frame the shot exactly as the SCENE says" in empty


def test_object_pictures_and_descriptions_leave_out_the_scene():
    phone = {"kind": "prop", "name": "Maya's smartphone", "details": {},
             "description": "Cracked black phone, underwater in a mud puddle, showing INCOMING CALL: JULIAN"}
    p = world.picture_prompt(phone, "", "9:16")
    assert "plain neutral background" in p and "ignore any situation" in p and "switched off" in p
    assert "Never what happens to it in a scene" in world.extract_prompt("x", [], [], "")


def test_the_next_shot_in_the_same_place_carries_the_look_over(env):
    client, db, ai, gen, ren, u, pid = setup(env, scenes=(ROOF, ROOF2, FLAT))
    d = find(client, ai, gen, u, pid)
    start(client, u, pid, "element", [e["id"] for e in d["elements"]])
    run(gen)
    roof, roof2, flat = d["scenes"]
    for s in (roof, roof2, flat):
        ai.image_log = []
        start(client, u, pid, "image", [s["id"]])
        run(gen)
        if s is roof:
            first = ai.image_log[-1]["prompt"]
        elif s is roof2:
            second = ai.image_log[-1]["prompt"]
        else:
            third = ai.image_log[-1]["prompt"]
    assert "the previous shot" not in first, "nothing comes before the first scene"
    assert "the previous shot, same place and moment" in second
    assert "the location, Rooftop" not in second, "the previous shot already shows the place"
    assert "the previous shot" not in third, "a different place doesn't copy the rooftop"
