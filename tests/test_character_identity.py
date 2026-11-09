"""The Character Consistency Engine: a permanent identity, a reference sheet
made from one anchor view, approval and locking, outfits, and identity in
every scene picture and video. Fake AI and local Postgres only."""
import os
import uuid

import pytest

import studio_ai
import studio_identity as ident
import studio_pricing as pricing
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import DAN, MAYA, TALK, detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")),
                                reason="needs creator_studio_identity.sql")

ALL = list(ident.VIEWS)
BEHIND = {"heading": "EXT. STREET - NIGHT", "visual": "Maya walks away from the camera, seen from behind, into the rain",
          "duration_seconds": 4, "dialogue": []}
FACE = {"heading": "INT. FLAT - NIGHT", "visual": "Close-up of Maya's face as she reads the letter", "duration_seconds": 4,
        "dialogue": []}


def jobs(client, u, pid, kind, cid, views=None, outfit_id=None):
    return client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                       json={"kind": kind, "target_ids": [cid], "request_id": uuid.uuid4().hex,
                             "views": views or [], "outfit_id": outfit_id})


def char(client, u, pid, name="Maya"):
    return next(c for c in detail(client, u, pid)["characters"] if c["name"] == name)


def views_of(c):
    return {v["view"]: v for v in c["reference_views"]}


def paths(db, cid):
    return db.execute("select reference_views from public.studio_characters where id = %s", (cid,)).fetchone()[0]


def setup(env, scenes=(TALK,), credits=5000, style="video"):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, credits)
    p = project(client, u, production_style=style)
    split(client, ai, gen, u, p["id"], list(scenes))
    return client, db, ai, gen, u, p["id"]


def make_sheet(client, ai, gen, u, pid, name="Maya", views=ALL):
    c = char(client, u, pid, name)
    r = jobs(client, u, pid, "portrait", c["id"], views)
    assert r.status_code == 200, r.text
    run(gen)
    return char(client, u, pid, name)


# ------------------------------------------------------------- identity

def test_the_sheet_creates_a_structured_identity_and_keeps_it(env):
    client, db, ai, gen, u, pid = setup(env)
    c = make_sheet(client, ai, gen, u, pid)
    assert c["identity"]["basic"]["height_cm"] == 168 and c["identity"]["face"]["marks"].startswith("small scar")
    assert ("identity", "Maya") in ai.calls
    # The creator edits one field; the rest stays.
    r = client.patch(f"/api/v1/studio/characters/{c['id']}/identity", headers=H(u),
                     json={"identity": {"basic": {"height_cm": 171}, "face": {"marks": ""}}})
    assert r.status_code == 200, r.text
    saved = r.json()["identity"]
    assert saved["basic"]["height_cm"] == 171 and saved["face"]["eyes"].startswith("almond") and "marks" not in saved["face"]
    assert char(client, u, pid)["identity"] == saved
    # A second sheet never asks the AI for a new identity.
    ai.calls.clear()
    jobs(client, u, pid, "portrait", c["id"], ["left_profile"])
    run(gen)
    assert ("identity", "Maya") not in ai.calls


def test_identity_is_cleaned_and_rendered_as_one_permanent_block():
    raw = {"basic": {"height_cm": "999", "build": "  broad\n shoulders "}, "face": {"hair": "x" * 500, "junk": 1}, "x": 2}
    clean = ident.clean_identity(raw)
    assert "height_cm" not in clean["basic"] and clean["basic"]["build"] == "broad shoulders"
    assert len(clean["face"]["hair"]) == ident.TEXT_MAX and "junk" not in clean["face"] and "x" not in clean
    text = ident.identity_text("Ama", {"basic": {"height_cm": 160, "skin_tone": "deep brown"},
                                       "face": {"facial_hair": "none", "eyes": "round, hazel"}})
    assert "height 160 cm" in text and "skin tone deep brown" in text and "facial hair" not in text


# ---------------------------------------------------------- privacy

def test_another_creator_cannot_see_or_change_a_character(env):
    client, db, ai, gen, u, pid = setup(env)
    c = make_sheet(client, ai, gen, u, pid)
    other = make_user(db)
    grant(client, other, 500)
    cid = c["id"]
    assert client.patch(f"/api/v1/studio/characters/{cid}/identity", headers=H(other),
                        json={"identity": {"basic": {"age": "60"}}}).status_code == 404
    assert client.post(f"/api/v1/studio/characters/{cid}/identity/lock", headers=H(other)).status_code == 404
    assert client.post(f"/api/v1/studio/characters/{cid}/views/front_full/approve", headers=H(other)).status_code == 404
    assert client.post(f"/api/v1/studio/characters/{cid}/outfits", headers=H(other), json={"name": "x"}).status_code == 404
    assert client.get(f"/api/v1/studio/projects/{pid}", headers=H(other)).status_code == 404
    assert jobs(client, other, pid, "portrait", cid, ["back_full"]).status_code == 404
    # Someone else's outfit can't be used in my job either.
    mine = client.post(f"/api/v1/studio/characters/{cid}/outfits", headers=H(u), json={"name": "Gala"}).json()
    oid = next(o["id"] for o in mine["outfits"] if o["name"] == "Gala")
    assert client.patch(f"/api/v1/studio/outfits/{oid}", headers=H(other), json={"name": "x"}).status_code == 404
    assert client.delete(f"/api/v1/studio/outfits/{oid}", headers=H(other)).status_code == 404
    other_pid = project(client, other)["id"]
    assert client.post(f"/api/v1/studio/projects/{other_pid}/estimate", headers=H(other),
                       json={"kind": "portrait", "target_ids": [], "outfit_id": oid}).status_code == 404
    assert client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["characters"][0]["identity"]["basic"]["age"] \
        == "late 20s"


# ------------------------------------------------------ reference sheet

def test_every_view_is_made_from_the_front_view(env):
    client, db, ai, gen, u, pid = setup(env)
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "portrait", "target_ids": [char(client, u, pid)["id"]], "views": ALL}).json()
    assert est["total_credits"] == pricing.hold_credits("portrait", pricing.estimate_sheet(7, "1K", True))
    assert 50 <= est["total_credits"] <= 80
    ai.image_log = []
    c = make_sheet(client, ai, gen, u, pid)
    prompts = [x["prompt"] for x in ai.image_log]
    assert [x["refs"] for x in ai.image_log] == [0, 1, 1, 1, 1, 2, 1], "front first; the rest from it"
    assert "first, defining picture" in prompts[0]
    assert all("Draw THE SAME PERSON" in x for x in prompts[1:])
    assert all("small scar above the left eyebrow" in x for x in prompts[:5])
    assert "left side profile" in prompts[2] and "directly behind" in prompts[4] and "2x2 grid" in prompts[5]
    raw = paths(db, c["id"])
    anchor = raw["front_full"]["path"]
    assert all(raw[v]["anchor"] == anchor for v in ALL) and all(raw[v]["status"] == "draft" for v in ALL)
    assert all(v["status"] == "draft" and v["url"] for v in c["reference_views"])
    face = db.execute("select reference_path from public.studio_characters where id = %s", (c["id"],)).fetchone()[0]
    assert face == raw["face_closeup"]["path"], "the face view is the portrait every older step uses"
    assert [o["name"] for o in c["outfits"]] == ["Main outfit"] and c["outfits"][0]["has_image"]
    assert c["outfits"][0]["description"] == "red jacket, black jeans, white sneakers"


def test_a_view_needs_the_front_view_first(env):
    client, db, ai, gen, u, pid = setup(env)
    c = char(client, u, pid)
    r = jobs(client, u, pid, "portrait", c["id"], ["back_full"])
    assert r.status_code == 400 and "front view first" in r.json()["detail"]["message"]
    assert jobs(client, u, pid, "portrait", c["id"], ["nope"]).status_code == 400


def test_an_existing_portrait_seeds_the_front_view(env):
    client, db, ai, gen, u, pid = setup(env)
    start(client, u, pid, "portrait", [char(client, u, pid)["id"]])
    run(gen)
    ai.image_log = []
    make_sheet(client, ai, gen, u, pid, views=["front_full"])
    assert ai.image_log[0]["refs"] == 1 and "existing portrait" in ai.image_log[0]["prompt"]


def test_redoing_one_view_never_changes_the_others(env):
    client, db, ai, gen, u, pid = setup(env)
    c = make_sheet(client, ai, gen, u, pid)
    cid = c["id"]
    assert client.post(f"/api/v1/studio/characters/{cid}/views/back_full/approve", headers=H(u)).status_code == 200
    before = paths(db, cid)
    make_sheet(client, ai, gen, u, pid, views=["left_profile"])
    after = paths(db, cid)
    assert after["left_profile"]["path"] != before["left_profile"]["path"]
    assert after["left_profile"]["anchor"] == before["front_full"]["path"]
    assert {v: after[v] for v in ALL if v != "left_profile"} == {v: before[v] for v in ALL if v != "left_profile"}
    assert after["back_full"]["status"] == "approved"


def test_a_new_front_view_marks_older_views_outdated_without_changing_them(env):
    client, db, ai, gen, u, pid = setup(env)
    c = make_sheet(client, ai, gen, u, pid)
    before = paths(db, c["id"])
    c = make_sheet(client, ai, gen, u, pid, views=["front_full"])
    v = views_of(c)
    assert v["front_full"]["status"] == "draft" and v["left_profile"]["status"] == "outdated"
    assert paths(db, c["id"])["left_profile"] == before["left_profile"]
    r = client.post(f"/api/v1/studio/characters/{c['id']}/identity/lock", headers=H(u))
    assert r.status_code == 409 and "older front view" in r.json()["detail"]["message"]


# ----------------------------------------------------------- locking

def test_approve_and_lock_then_unlock(env):
    client, db, ai, gen, u, pid = setup(env)
    c = char(client, u, pid)
    r = client.post(f"/api/v1/studio/characters/{c['id']}/identity/lock", headers=H(u))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "not_ready"
    c = make_sheet(client, ai, gen, u, pid)
    assert c["lock_problem"] is None and not c["identity_locked"]
    r = client.post(f"/api/v1/studio/characters/{c['id']}/identity/lock", headers=H(u))
    assert r.status_code == 200 and r.json()["identity_locked"]
    assert all(v["status"] == "approved" for v in r.json()["reference_views"])
    # Locked: no edits, no redos, no charge.
    held = balance(client, u)
    assert client.patch(f"/api/v1/studio/characters/{c['id']}/identity", headers=H(u),
                        json={"identity": {"basic": {"age": "50"}}}).json()["detail"]["code"] == "identity_locked"
    r = jobs(client, u, pid, "portrait", c["id"], ["back_full"])
    assert r.status_code == 400 and "locked" in r.json()["detail"]["message"]
    assert balance(client, u) == held
    assert client.post(f"/api/v1/studio/characters/{c['id']}/views/back_full/approve", headers=H(u)).status_code == 409
    r = client.post(f"/api/v1/studio/characters/{c['id']}/identity/unlock", headers=H(u))
    assert r.status_code == 200 and not r.json()["identity_locked"]
    assert jobs(client, u, pid, "portrait", c["id"], ["back_full"]).status_code == 200


def test_the_worker_refuses_a_locked_character_and_refunds(env):
    client, db, ai, gen, u, pid = setup(env)
    c = make_sheet(client, ai, gen, u, pid)
    r = jobs(client, u, pid, "portrait", c["id"], ["back_full"])
    assert r.status_code == 200
    client.post(f"/api/v1/studio/characters/{c['id']}/identity/lock", headers=H(u))
    before = balance(client, u)
    run(gen)
    after = balance(client, u)
    assert after["held"] == 0 and after["available"] == before["available"] + before["held"]


# ------------------------------------------------------ scene pictures

def test_scene_pictures_get_the_identity_and_the_view_for_the_camera_angle(env):
    client, db, ai, gen, u, pid = setup(env, scenes=(BEHIND, FACE), style="pictures")
    make_sheet(client, ai, gen, u, pid)
    scenes = detail(client, u, pid)["scenes"]
    assert [s["camera_used"] for s in scenes] == ["back", "close_up"]
    ai.image_log = []
    start(client, u, pid, "image", [s["id"] for s in scenes])
    run(gen)
    back, close = (x["prompt"] for x in ai.image_log)
    for prompt in (back, close):
        assert "permanent identity: never change" in prompt and "small scar above the left eyebrow" in prompt
        assert "height 168 cm" in prompt and "Wearing in this scene: red jacket, black jeans" in prompt
        assert prompt.index("CHARACTERS") < prompt.index("SCENE:"), "identity before the scene text"
    assert "Reference image 1: Maya: back, full body" in back and "Shot from behind" in back
    assert "Reference image 1: Maya: face close-up" in close and "Close-up shot." in close
    assert all(x["refs"] >= 2 for x in ai.image_log), "view + outfit"


def test_a_character_without_a_sheet_still_uses_their_portrait(env):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    start(client, u, pid, "portrait")
    run(gen)
    ai.image_log = []
    start(client, u, pid, "image")
    run(gen)
    assert ai.image_log[0]["refs"] == 2 and "Maya (keep this exact face, hair and clothes)" in ai.image_log[0]["prompt"]


# ----------------------------------------------------------------- video

def test_talking_video_gets_identity_views_and_outfits(env):
    client, db, ai, gen, u, pid = setup(env)
    make_sheet(client, ai, gen, u, pid, "Maya")
    make_sheet(client, ai, gen, u, pid, "Daniel")
    s = detail(client, u, pid)["scenes"][0]
    start(client, u, pid, "image", [s["id"]])
    run(gen)
    assert start(client, u, pid, "video", [s["id"]]).status_code == 200
    run(gen)
    refs = ai.last_video_refs
    assert refs[0].startswith("the scene") and len(refs) <= 6
    assert any(r.startswith("Maya: face close-up") for r in refs) and any(r.startswith("Daniel: face close-up") for r in refs)
    assert any("Maya wearing the outfit for this scene" in r for r in refs)
    assert "permanent identity: never change" in ai.last_video_prompt and "height 168 cm" in ai.last_video_prompt


def test_a_silent_video_without_a_picture_gets_the_reference_views(env):
    client, db, ai, gen, u, pid = setup(env, scenes=(BEHIND,))
    make_sheet(client, ai, gen, u, pid)
    s = detail(client, u, pid)["scenes"][0]
    assert start(client, u, pid, "video", [s["id"]]).status_code == 200
    run(gen)
    assert ai.last_video_refs and ai.last_video_refs[0].startswith("Maya: back, full body")
    assert "small scar above the left eyebrow" in ai.last_video_prompt


# --------------------------------------------------------------- outfits

def test_outfits_carry_across_connected_scenes(env):
    scenes = [dict(FACE, heading=f"SCENE {i}") for i in range(4)]
    client, db, ai, gen, u, pid = setup(env, scenes=scenes, style="pictures")
    c = make_sheet(client, ai, gen, u, pid)
    r = client.post(f"/api/v1/studio/characters/{c['id']}/outfits", headers=H(u),
                    json={"name": "Wedding dress", "description": "long ivory wedding dress, pearl earrings"})
    gala = next(o for o in r.json()["outfits"] if o["name"] == "Wedding dress")
    assert not gala["is_default"] and not gala["has_image"]
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "portrait", "target_ids": [c["id"]], "outfit_id": gala["id"]}).json()
    assert est["ready_count"] == 1 and 5 <= est["total_credits"] <= 15
    ai.image_log = []
    assert jobs(client, u, pid, "portrait", c["id"], outfit_id=gala["id"]).status_code == 200
    run(gen)
    assert ai.image_log[0]["refs"] == 1 and "long ivory wedding dress" in ai.image_log[0]["prompt"]
    main = next(o for o in char(client, u, pid)["outfits"] if o["is_default"])
    s = detail(client, u, pid)["scenes"]
    # Scene 2 changes into the dress; scene 3 says nothing (keeps it); scene 4 changes back.
    assert client.patch(f"/api/v1/studio/scenes/{s[1]['id']}", headers=H(u),
                        json={"outfits": {c["id"]: gala["id"]}}).status_code == 200
    client.patch(f"/api/v1/studio/scenes/{s[3]['id']}", headers=H(u), json={"outfits": {c["id"]: main["id"]}})
    ai.image_log = []
    start(client, u, pid, "image", [x["id"] for x in s])
    run(gen)
    wears = ["wedding dress" in x["prompt"] for x in ai.image_log]
    assert wears == [False, True, True, False]
    # The permanent identity never changed with the clothes.
    assert char(client, u, pid)["identity"]["face"]["marks"].startswith("small scar")
    # An outfit of another character is refused for this one.
    dan = char(client, u, pid, "Daniel")
    assert client.patch(f"/api/v1/studio/scenes/{s[0]['id']}", headers=H(u),
                        json={"outfits": {dan["id"]: gala["id"]}}).status_code == 400
    # The main outfit can't be removed; another can, and scenes fall back.
    assert client.delete(f"/api/v1/studio/outfits/{main['id']}", headers=H(u)).status_code == 409
    assert client.delete(f"/api/v1/studio/outfits/{gala['id']}", headers=H(u)).status_code == 200
    assert detail(client, u, pid)["scenes"][1]["outfits"] == {}


def test_resolve_outfits_rules():
    cast = [{"id": "m", "name": "Maya"}]
    outfits = [{"id": "main", "character_id": "m", "is_default": True}, {"id": "red", "character_id": "m"},
               {"id": "dan", "character_id": "d"}]
    scenes = [{"id": f"s{i}", "position": i, "outfits": o} for i, o in
              enumerate([{}, {"m": "red"}, {}, {"m": "dan"}, {"m": "main"}])]
    got = ident.resolve_outfits(scenes, outfits, cast, lambda s, c: c)
    assert [got[f"s{i}"]["m"]["id"] for i in range(5)] == ["main", "red", "red", "red", "main"]


def test_reference_choice_follows_the_camera():
    views = {v: {"path": f"{v}.png", "status": "approved", "anchor": "front_full.png"} for v in ident.VIEWS}
    c = {"name": "Ama", "reference_views": views}
    assert [p for _, p in ident.references_for(c, "side", None, 3)] == ["face_closeup.png", "left_profile.png",
                                                                         "right_profile.png"]
    assert [p for _, p in ident.references_for(c, "back", {"reference_path": "o.png"}, 2)] == ["back_full.png", "o.png"]
    assert ident.split_limit(2, 5) == [3, 2] and ident.split_limit(3, 4) == [2, 1, 1]
    assert ident.infer_camera("Ama walks away, seen from behind") == "back"
    assert ident.infer_camera("Close-up on his hands") == "close_up"


# ------------------------------------------------- failures and refunds

def test_a_failed_first_view_refunds_everything(env):
    client, db, ai, gen, u, pid = setup(env)
    before = balance(client, u)
    ai.fail["image"] = studio_ai.ProviderError("boom")
    jobs(client, u, pid, "portrait", char(client, u, pid)["id"], ALL)
    run(gen)
    assert balance(client, u) == before
    assert all(v["status"] == "missing" for v in char(client, u, pid)["reference_views"])


def test_a_later_failure_keeps_the_finished_views_and_charges_only_them(env):
    client, db, ai, gen, u, pid = setup(env)
    before = balance(client, u)["available"]
    calls = {"n": 0}
    real = ai.image

    def flaky(prompt, references, aspect_ratio, size="1K"):
        calls["n"] += 1
        if calls["n"] == 3:
            raise studio_ai.ProviderError("boom")
        return real(prompt, references, aspect_ratio, size)

    ai.image = flaky
    c = char(client, u, pid)
    jobs(client, u, pid, "portrait", c["id"], ALL)
    run(gen)
    st = {v["view"]: v["status"] for v in char(client, u, pid)["reference_views"]}
    assert st["left_profile"] == "missing" and st["front_full"] == st["face_closeup"] == st["back_full"] == "draft"
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded" and job["charged_credits"] < job["estimated_credits"]
    after = balance(client, u)
    assert after["held"] == 0 and before - after["available"] == job["charged_credits"]


def test_the_next_episode_keeps_the_locked_identity_sheet_and_outfits(env):
    client, db, ai, gen, u, pid = setup(env)
    c = make_sheet(client, ai, gen, u, pid)
    client.post(f"/api/v1/studio/characters/{c['id']}/identity/lock", headers=H(u))
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u)).json()["id"]
    split(client, ai, gen, u, ep2, [TALK])
    c2 = char(client, u, ep2)
    assert c2["id"] != c["id"] and c2["identity_locked"] and c2["identity"] == char(client, u, pid)["identity"]
    assert paths(db, c2["id"]) == paths(db, c["id"])
    assert [o["name"] for o in c2["outfits"]] == ["Main outfit"] and c2["outfits"][0]["has_image"]


def test_an_outfit_needs_a_real_name(env):
    client, db, ai, gen, u, pid = setup(env)
    c = char(client, u, pid)
    assert client.post(f"/api/v1/studio/characters/{c['id']}/outfits", headers=H(u), json={"name": "   "}).status_code == 400
    o = client.post(f"/api/v1/studio/characters/{c['id']}/outfits", headers=H(u), json={"name": "Coat"}).json()["outfits"][0]
    assert client.patch(f"/api/v1/studio/outfits/{o['id']}", headers=H(u), json={"name": "  "}).status_code == 400


# ------------------------------------------------ flexible outfit system

def add_outfit(client, u, cid, name, **kw):
    r = client.post(f"/api/v1/studio/characters/{cid}/outfits", headers=H(u), json={"name": name, **kw})
    assert r.status_code == 200, r.text
    return next(o for o in r.json()["outfits"] if o["name"] == name)


def test_unlimited_outfits_with_independent_parts(env):
    client, db, ai, gen, u, pid = setup(env)
    c = char(client, u, pid)
    for i in range(15):
        add_outfit(client, u, c["id"], f"Look {i}")
    work = add_outfit(client, u, c["id"], "Work", parts={"clothing": "blue hospital scrubs", "shoes": "white clogs",
                                                          "accessories": "ID badge", "makeup": "none, tired eyes"})
    assert len(char(client, u, pid)["outfits"]) == 16
    assert work["parts"] == {"clothing": "blue hospital scrubs", "shoes": "white clogs", "accessories": "ID badge",
                             "makeup": "none, tired eyes"}
    # Change only the shoes: the other parts stay.
    r = client.patch(f"/api/v1/studio/outfits/{work['id']}", headers=H(u),
                     json={"parts": {**work["parts"], "shoes": "black trainers"}})
    got = next(o for o in r.json()["outfits"] if o["id"] == work["id"])
    assert got["parts"]["shoes"] == "black trainers" and got["parts"]["clothing"] == "blue hospital scrubs"
    assert client.post(f"/api/v1/studio/characters/{c['id']}/outfits", headers=H(u),
                       json={"name": "x", "parts": {"face": "new"}}).json()["outfits"][-1]["parts"] == {}


def test_an_outfit_picture_never_remakes_the_face_or_body_and_goes_out_of_date_when_a_part_changes(env):
    client, db, ai, gen, u, pid = setup(env)
    c = make_sheet(client, ai, gen, u, pid)
    sheet_before, ident_before = paths(db, c["id"]), char(client, u, pid)["identity"]
    work = add_outfit(client, u, c["id"], "Work", parts={"clothing": "blue scrubs", "makeup": "red lipstick"})
    ai.image_log, ai.calls = [], []
    jobs(client, u, pid, "portrait", c["id"], outfit_id=work["id"])
    run(gen)
    assert len(ai.image_log) == 1 and ai.image_log[0]["refs"] == 1 and ("identity", "Maya") not in ai.calls
    prompt = ai.image_log[0]["prompt"]
    assert "clothing: blue scrubs" in prompt and "makeup: red lipstick" in prompt and "only the clothes" in prompt
    assert paths(db, c["id"]) == sheet_before and char(client, u, pid)["identity"] == ident_before
    o = next(x for x in char(client, u, pid)["outfits"] if x["id"] == work["id"])
    assert o["has_image"] and not o["image_outdated"]
    client.patch(f"/api/v1/studio/outfits/{work['id']}", headers=H(u), json={"parts": {"clothing": "blue scrubs",
                                                                                        "makeup": "none"}})
    o = next(x for x in char(client, u, pid)["outfits"] if x["id"] == work["id"])
    assert o["image_outdated"]


def test_a_scene_only_change_does_not_touch_the_outfit_or_the_next_scene(env):
    scenes = [dict(FACE, heading=f"SCENE {i}") for i in range(3)]
    client, db, ai, gen, u, pid = setup(env, scenes=scenes, style="pictures")
    c = make_sheet(client, ai, gen, u, pid)
    s = detail(client, u, pid)["scenes"]
    r = client.patch(f"/api/v1/studio/scenes/{s[1]['id']}", headers=H(u),
                     json={"outfit_overrides": {c["id"]: {"makeup": "smudged mascara", "clothing": ""}}})
    assert r.status_code == 200 and r.json()["outfit_overrides"] == {c["id"]: {"makeup": "smudged mascara"}}
    assert client.patch(f"/api/v1/studio/scenes/{s[1]['id']}", headers=H(u),
                        json={"outfit_overrides": {c["id"]: {"hair": "blonde"}}}).status_code == 400
    ai.image_log = []
    start(client, u, pid, "image", [x["id"] for x in s])
    run(gen)
    smudged = ["smudged mascara" in x["prompt"] for x in ai.image_log]
    assert smudged == [False, True, False]
    assert "for this scene only" in ai.image_log[1]["prompt"] and "red jacket" in ai.image_log[1]["prompt"]
    main = next(o for o in char(client, u, pid)["outfits"] if o["is_default"])
    assert "makeup" not in main["parts"]


def test_ai_suggests_outfits_and_the_creator_decides(env):
    scenes = [dict(FACE, heading=f"SCENE {i}") for i in range(3)]
    client, db, ai, gen, u, pid = setup(env, scenes=scenes, style="pictures")
    c = make_sheet(client, ai, gen, u, pid)
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": "INT. CHURCH - DAY\n\nMaya marries.\n"})
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "outfits"}).json()
    assert est["ready_count"] == 1 and 1 <= est["total_credits"] <= 10
    assert start(client, u, pid, "outfits").status_code == 200
    run(gen)
    assert "costume designer" in ai.last_outfit_prompt and "Maya marries." in ai.last_outfit_prompt
    me = char(client, u, pid)
    sug = me["outfit_suggestions"]
    assert [o["name"] for o in sug] == ["Wedding"] and sug[0]["parts"]["shoes"] == "satin heels"
    assert sug[0]["reason"] == "Her wedding day." and [o["name"] for o in me["outfits"]] == ["Main outfit"]
    s = detail(client, u, pid)["scenes"]
    assert sug[0]["suggested_scenes"] == [s[1]["id"]] and s[1]["outfits"] == {}, "nothing applied yet"
    # A suggestion can't be used or pictured until accepted.
    assert client.patch(f"/api/v1/studio/scenes/{s[0]['id']}", headers=H(u),
                        json={"outfits": {c["id"]: sug[0]["id"]}}).status_code == 400
    r = client.post(f"/api/v1/studio/outfits/{sug[0]['id']}/accept", headers=H(u), json={"use_in_suggested_scenes": True})
    assert r.status_code == 200 and [o["name"] for o in r.json()["outfits"]] == ["Main outfit", "Wedding"]
    assert detail(client, u, pid)["scenes"][1]["outfits"] == {c["id"]: sug[0]["id"]}
    # New suggestions replace old ones, never the accepted outfit; dismissing deletes a suggestion.
    start(client, u, pid, "outfits")
    run(gen)
    me = char(client, u, pid)
    assert me["outfit_suggestions"] == [] and [o["name"] for o in me["outfits"]] == ["Main outfit", "Wedding"]
    ai.outfit_result = [{"character": "Maya", "name": "Nightwear", "clothing": "grey pyjamas", "scenes": [3]}]
    start(client, u, pid, "outfits")
    run(gen)
    night = char(client, u, pid)["outfit_suggestions"][0]
    assert client.delete(f"/api/v1/studio/outfits/{night['id']}", headers=H(u)).status_code == 200
    assert char(client, u, pid)["outfit_suggestions"] == []


def test_outfits_are_reused_in_later_episodes_but_temporary_ones_are_not(env):
    client, db, ai, gen, u, pid = setup(env)
    c = make_sheet(client, ai, gen, u, pid)
    add_outfit(client, u, c["id"], "Hospital", parts={"clothing": "hospital gown"})
    add_outfit(client, u, c["id"], "Soaked", parts={"clothing": "rain-soaked jacket"}, is_temporary=True)
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u)).json()["id"]
    split(client, ai, gen, u, ep2, [TALK])
    c2 = char(client, u, ep2)
    assert [o["name"] for o in c2["outfits"]] == ["Main outfit", "Hospital"]
    # An outfit added to episode 1 later shows in episode 2's library and is reused without a new picture.
    gala = add_outfit(client, u, c["id"], "Gala", parts={"clothing": "black gown"})
    lib = char(client, u, ep2)["outfit_library"]
    assert [o["name"] for o in lib] == ["Gala"]
    ai.calls = []
    r = client.post(f"/api/v1/studio/characters/{c2['id']}/outfits/import", headers=H(u), json={"outfit_id": gala["id"]})
    assert r.status_code == 200 and "Gala" in [o["name"] for o in r.json()["outfits"]] and r.json()["outfit_library"] == []
    assert not [x for x in ai.calls if x[0] == "image"]
    other = make_user(db)
    assert client.post(f"/api/v1/studio/characters/{c2['id']}/outfits/import", headers=H(other),
                       json={"outfit_id": gala["id"]}).status_code == 404
