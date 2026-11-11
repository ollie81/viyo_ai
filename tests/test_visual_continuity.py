"""The visual continuity engine: a vision check after every scene picture (or on
request), structured results, the observed state handed to the next scene,
corrections that remake one scene only, approved pictures that are never
replaced, restoring versions, credit protection, and the assistant reading the
checks. Fake AI and local Postgres: no real AI credits."""
import os
import uuid

import pytest

import creator_studio as cs
import studio_ai
import studio_inspect
from conftest import make_user
from test_continuity import STORY, build
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import detail

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_inspection.sql")


def problem(kind, severity="high", certainty="likely", intentional=False, **kw):
    return {"kind": kind, "severity": severity, "certainty": certainty, "intentional": intentional,
            "expected": kw.get("expected", "as before"), "seen": kw.get("seen", "different"),
            "fix": kw.get("fix", f"fix the {kind}"), "characters": kw.get("characters", ["Maya"])}


def result(overall=90, problems=(), observed=None, **scores):
    return {"scores": {"overall": overall, **scores}, "problems": list(problems), "observed": observed or {},
            "summary": "checked"}


def job_of(client, u, pid, kind):
    return next(j for j in client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"]
                if j["kind"] == kind)


def picture(client, gen, u, pid, sid, **extra):
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "image", "target_ids": [sid], "request_id": uuid.uuid4().hex, **extra})
    run(gen)
    return r


def scenes(client, u, pid):
    return detail(client, u, pid)["scenes"]


# ------------------------------------------------------------------ the check

def test_every_new_picture_is_checked_and_the_price_includes_it(env):
    client, db, ai, gen, u, pid = build(env)
    s = scenes(client, u, pid)[0]
    est_on = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                         json={"kind": "image", "target_ids": [s["id"]]}).json()["total_credits"]
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"auto_inspect": False})
    est_off = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                          json={"kind": "image", "target_ids": [s["id"]]}).json()["total_credits"]
    assert est_on > est_off
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"auto_inspect": True})
    start(client, u, pid, "portrait")       # character reference pictures to compare against
    run(gen)
    picture(client, gen, u, pid, s["id"])
    job = job_of(client, u, pid, "image")
    assert job["status"] == "succeeded" and job["charged_credits"] <= est_on
    ins = scenes(client, u, pid)[0]["inspection"]
    assert ins["scores"]["overall"] == 92 and ins["outdated"] is False and ins["needs_attention"] is False
    assert ai.inspect_log[-1]["labels"][0] == "this scene's picture (check this one)"
    assert any("reference:" in label and "Maya" in label for label in ai.inspect_log[-1]["labels"])


def test_turning_the_check_off_makes_no_vision_call(env):
    client, db, ai, gen, u, pid = build(env)
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"auto_inspect": False})
    picture(client, gen, u, pid, scenes(client, u, pid)[0]["id"])
    assert ("inspect",) not in ai.calls and scenes(client, u, pid)[0]["inspection"] is None


def test_a_face_outfit_prop_and_place_mismatch_are_reported_with_fixes(env):
    client, db, ai, gen, u, pid = build(env)
    ai.inspect_result = result(48, [
        problem("character", expected="short black hair", seen="long blonde hair"),
        problem("outfit", "medium", expected="red coat", seen="blue dress", fix="Maya must wear the red coat"),
        problem("prop", expected="red suitcase in her right hand", seen="no suitcase", fix="Put the red suitcase in her right hand"),
        problem("location", "medium", "certain", expected="family house path", seen="city street")],
        character=40, outfit=50, location=45, props=30)
    picture(client, gen, u, pid, scenes(client, u, pid)[0]["id"])
    ins = scenes(client, u, pid)[0]["inspection"]
    assert {p["kind"] for p in ins["problems"]} == {"character", "outfit", "prop", "location"}
    assert ins["needs_attention"] and ins["scores"]["props"] == 30
    assert "Maya must wear the red coat" in ins["correction"] and "Put the red suitcase" in ins["correction"]


def test_an_intended_change_is_not_a_mistake(env):
    client, db, ai, gen, u, pid = build(env)
    ai.inspect_result = result(85, [problem("outfit", intentional=True, expected="red coat", seen="black dress")])
    picture(client, gen, u, pid, scenes(client, u, pid)[3]["id"])
    ins = scenes(client, u, pid)[3]["inspection"]
    assert ins["needs_attention"] is False and ins["correction"] == ""


def test_uncertain_findings_stay_uncertain(env):
    client, db, ai, gen, u, pid = build(env)
    ai.inspect_result = result(80, [problem("prop", certainty="uncertain", fix="Show the suitcase clearly")])
    picture(client, gen, u, pid, scenes(client, u, pid)[0]["id"])
    ins = scenes(client, u, pid)[0]["inspection"]
    assert ins["problems"][0]["certainty"] == "uncertain"
    assert ins["needs_attention"] is False, "one uncertain finding is not a confirmed mistake"
    assert ins["correction"].startswith("Check: ")


def test_the_check_compares_with_the_previous_scene_and_hands_on_what_it_saw(env):
    client, db, ai, gen, u, pid = build(env)
    s1, s2 = scenes(client, u, pid)[:2]
    ai.inspect_result = result(90, observed={"characters": [{"name": "Maya", "outfit": "red coat, soaked",
                                                             "holding": ["red suitcase"]}], "weather": "heavy rain"})
    picture(client, gen, u, pid, s1["id"])
    ai.image_log, ai.inspect_result = [], None
    picture(client, gen, u, pid, s2["id"])
    assert "The previous picture actually shows (keep it unless listed as a change): Maya: red coat, soaked, holding red suitcase" \
        in ai.image_log[-1]["prompt"]
    check = ai.inspect_log[-1]
    assert "the previous scene's picture (this scene follows it)" in check["labels"]
    assert "It ended with: Maya steps inside, suitcase in her right hand" in check["prompt"]
    assert "Its picture was seen to show" in check["prompt"] and "heavy rain" in check["prompt"]


def test_a_time_jump_is_not_compared_with_the_last_picture(env):
    client, db, ai, gen, u, pid = build(env)
    sc = scenes(client, u, pid)
    picture(client, gen, u, pid, sc[2]["id"])
    picture(client, gen, u, pid, sc[3]["id"])
    assert "the previous scene's picture (this scene follows it)" not in ai.inspect_log[-1]["labels"]


# ------------------------------------------------------------------ money

def test_a_failed_check_never_fails_the_picture_or_costs_anything(env):
    client, db, ai, gen, u, pid = build(env)
    s = scenes(client, u, pid)[0]
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"auto_inspect": False})
    picture(client, gen, u, pid, s["id"])
    plain = job_of(client, u, pid, "image")["charged_credits"]
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"auto_inspect": True})
    ai.fail["inspect"] = studio_ai.ProviderError("vision down")
    picture(client, gen, u, pid, s["id"])
    job = job_of(client, u, pid, "image")
    assert job["status"] == "succeeded" and job["charged_credits"] == plain, "only the picture is charged"
    assert scenes(client, u, pid)[0]["inspection"] is None


def test_checking_an_existing_picture_is_refunded_if_it_fails_and_reused_if_nothing_changed(env):
    client, db, ai, gen, u, pid = build(env)
    s = scenes(client, u, pid)[0]
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"auto_inspect": False})
    picture(client, gen, u, pid, s["id"])
    before = balance(client, u)["available"]
    ai.fail["inspect"] = studio_ai.ProviderError("vision down")
    assert start(client, u, pid, "inspect", [s["id"]]).status_code == 200
    run(gen)
    assert job_of(client, u, pid, "inspect")["status"] == "failed" and balance(client, u)["available"] == before
    ai.fail.pop("inspect")
    start(client, u, pid, "inspect", [s["id"]])
    run(gen)
    assert job_of(client, u, pid, "inspect")["status"] == "succeeded"
    calls = ai.calls.count(("inspect",))
    again = start(client, u, pid, "inspect", [s["id"]])
    assert again.status_code == 400 and "Already checked" in again.text and ai.calls.count(("inspect",)) == calls
    client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u), json={"visual": "Maya at the gate, looking back"})
    assert start(client, u, pid, "inspect", [s["id"]]).status_code == 200, "the scene changed, so it's checked again"


def test_a_failed_new_picture_keeps_the_old_one_and_is_refunded(env):
    client, db, ai, gen, u, pid = build(env)
    s = scenes(client, u, pid)[0]
    picture(client, gen, u, pid, s["id"])
    old = db.execute("select image_path from public.studio_scenes where id = %s", (s["id"],)).fetchone()[0]
    before = balance(client, u)["available"]
    ai.fail["image"] = studio_ai.ProviderError("down")
    picture(client, gen, u, pid, s["id"], correction="Maya must wear the red coat")
    assert db.execute("select image_path from public.studio_scenes where id = %s", (s["id"],)).fetchone()[0] == old
    assert balance(client, u)["available"] == before


# ------------------------------------------------------------------ correcting

def test_a_correction_remakes_only_that_scene_and_is_checked_again(env):
    client, db, ai, gen, u, pid = build(env)
    sc = scenes(client, u, pid)
    for s in sc[:3]:
        picture(client, gen, u, pid, s["id"])
    paths = {str(k): v for k, v in db.execute("select id, image_path from public.studio_scenes").fetchall()}
    ai.image_log = []
    checks = ai.calls.count(("inspect",))
    picture(client, gen, u, pid, sc[1]["id"], correction="Maya must keep the red suitcase in her right hand")
    assert len(ai.image_log) == 1, "one picture, not the whole episode"
    assert "CORRECTION (the last picture of this scene got this wrong" in ai.image_log[0]["prompt"]
    assert "red suitcase in her right hand" in ai.image_log[0]["prompt"]
    assert ai.calls.count(("inspect",)) == checks + 1, "the corrected picture is checked again"
    now = {str(k): v for k, v in db.execute("select id, image_path from public.studio_scenes").fetchall()}
    assert now[sc[0]["id"]] == paths[sc[0]["id"]] and now[sc[2]["id"]] == paths[sc[2]["id"]]
    assert now[sc[1]["id"]] != paths[sc[1]["id"]]


def test_an_approved_picture_is_never_replaced(env):
    client, db, ai, gen, u, pid = build(env)
    s = scenes(client, u, pid)[0]
    picture(client, gen, u, pid, s["id"])
    picture(client, gen, u, pid, s["id"])
    assert client.post(f"/api/v1/studio/scenes/{s['id']}/approve-image", headers=H(u), json={"approved": True}).status_code == 200
    assert scenes(client, u, pid)[0]["image_approved"] is True
    r = picture(client, gen, u, pid, s["id"], correction="anything")
    assert r.status_code == 400 and "approved" in r.text.lower()
    versions = client.get(f"/api/v1/studio/scenes/{s['id']}/versions", headers=H(u)).json()["versions"]
    older = next(v for v in versions if not v["current"])
    assert client.post(f"/api/v1/studio/scenes/{s['id']}/restore", headers=H(u), json={"asset_id": older["id"]}).status_code == 409
    client.post(f"/api/v1/studio/scenes/{s['id']}/approve-image", headers=H(u), json={"approved": False})
    assert picture(client, gen, u, pid, s["id"]).status_code == 200


def test_earlier_versions_can_be_restored_and_nothing_is_deleted(env):
    client, db, ai, gen, u, pid = build(env)
    s = scenes(client, u, pid)[0]
    picture(client, gen, u, pid, s["id"])
    first = db.execute("select image_path from public.studio_scenes where id = %s", (s["id"],)).fetchone()[0]
    picture(client, gen, u, pid, s["id"])
    versions = client.get(f"/api/v1/studio/scenes/{s['id']}/versions", headers=H(u)).json()["versions"]
    assert len([v for v in versions if v["kind"] == "image"]) == 2 and sum(v["current"] for v in versions) == 1
    older = next(v for v in versions if not v["current"])
    other = make_user(db)
    assert client.post(f"/api/v1/studio/scenes/{s['id']}/restore", headers=H(other), json={"asset_id": older["id"]}).status_code == 404
    assert client.post(f"/api/v1/studio/scenes/{s['id']}/restore", headers=H(u), json={"asset_id": older["id"]}).status_code == 200
    assert db.execute("select image_path from public.studio_scenes where id = %s", (s["id"],)).fetchone()[0] == first
    assert scenes(client, u, pid)[0]["inspection"]["outdated"] is True, "the check was for the other picture"
    assert len(client.get(f"/api/v1/studio/scenes/{s['id']}/versions", headers=H(u)).json()["versions"]) == 2


# ------------------------------------------------------------------ assistant

def test_the_assistant_reads_the_checks_and_proposes_one_scene_fixes(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    _c, _d, _a, _g, u, pid = build(env)
    sc = scenes(client, u, pid)
    ai.inspect_result = result(55, [problem("outfit", expected="red coat", seen="blue dress",
                                            fix="Maya's coat must stay red, as in scene 1")], outfit=40)
    picture(client, gen, u, pid, sc[1]["id"])
    picture(client, gen, u, pid, sc[2]["id"])
    client.post(f"/api/v1/studio/scenes/{sc[2]['id']}/approve-image", headers=H(u), json={"approved": True})
    ai.chat_result = {"reply": "The check thinks Maya's coat turned blue in scene 2.", "memory": [], "proposals": [
        {"type": "fix_scene_picture", "title": "Fix scene 2", "summary": "Red coat",
         "data": {"scene": 2, "correction": "Maya's coat must stay red, as in scene 1"}},
        {"type": "fix_scene_picture", "title": "Fix scene 3", "summary": "x", "data": {"scene": 3, "correction": "y"}},
        {"type": "check_scenes", "title": "Check the episode", "summary": "All pictures", "data": {"scenes": []}}]}
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": "Why does Maya look different in scene two?", "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    prompt = ai.chat_prompts[-1]
    assert "PICTURE CHECK" in prompt and "[outfit, high, likely] expected red coat; seen blue dress" in prompt
    assert "[picture APPROVED by the creator]" in prompt and "An APPROVED picture is never replaced" in prompt
    fix, approved, check = r.json()["reply"]["proposals"]
    assert fix["job"] == "image" and fix["targets"] == [sc[1]["id"]]
    assert fix["extra"] == {"correction": "Maya's coat must stay red, as in scene 1"}
    assert approved["status"] == "failed" and "approved" in approved["error"]
    assert check["job"] == "inspect" and set(check["targets"]) == {sc[1]["id"], sc[2]["id"]}, "only scenes with pictures"


def test_the_result_is_cleaned_and_bad_answers_are_rejected():
    r = studio_inspect.clean({"scores": {"overall": 140, "props": -5, "x": 3}, "problems": [
        {"kind": "outfit", "severity": "weird", "certainty": "??", "expected": "a", "seen": "b", "intentional": 0, "fix": "c"},
        {"kind": "nonsense"}], "observed": {"characters": [{"name": "Maya", "holding": ["bag"]}, {"outfit": "x"}]}})
    assert r["scores"] == {"overall": 100, "props": 0}
    assert r["problems"] == [{"kind": "outfit", "severity": "medium", "certainty": "uncertain", "characters": [],
                              "expected": "a", "seen": "b", "intentional": False, "explanation": "", "fix": "c"}]
    assert r["observed"]["characters"] == [{"name": "Maya", "holding": ["bag"]}]
    with pytest.raises(ValueError):
        studio_inspect.clean({"problems": []})
