"""Shot-to-shot visual continuity: every shot picture is compared with the
picture it follows and the approved references; cuts, reverse angles, time
jumps and approved changes are not mistakes; a critical identity error fails
the shot whatever the average; one corrected picture at most (its price shown
first, unused credits returned); approved shots are never overwritten; the
creator can accept an intentional change; the assistant explains and proposes.

Fake AI and local Postgres. These tests prove the pipeline (which pictures and
references reach the vision model and the image model, what is decided and
charged), NOT that real AI pictures look alike: that needs a live test.
"""
import io
import os
import uuid

import pytest
from PIL import Image

import creator_studio as cs
import studio_shot_continuity as shotcheck
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import detail, project

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_shot_continuity.sql")

AMA = {"name": "Ama", "description": "Woman, 29, braided hair, mustard raincoat", "gender": "female"}
KOFI = {"name": "Kofi", "description": "Man, 24, cut above his left eyebrow, hospital gown", "gender": "male"}


def who(*people):
    return [{"name": n, "in_frame": True, "position": pos, "outfit": wear} for n, pos, wear in people]


ENTER = {"heading": "INT. WARD - NIGHT", "visual": "Ama enters the ward and finds Kofi in bed twelve", "duration_seconds": 10,
         "dialogue": [{"character": "Ama", "text": "Kofi?"}, {"character": "Kofi", "text": "You shouldn't be here."}],
         "continuity": {"transition": "new_place", "time_of_day": "night", "lighting": "cold fluorescent",
                        "characters": who(("Ama", "left", "mustard raincoat"), ("Kofi", "right, in bed", "hospital gown"))}}
CARD = {"heading": "INT. WARD - NIGHT", "visual": "Kofi hands Ama his real ID card", "duration_seconds": 8,
        "dialogue": [{"character": "Kofi", "text": "This is who I am."}],
        "continuity": {"transition": "continuous", "time_of_day": "night", "action": "Kofi hands Ama the ID card",
                       "characters": who(("Ama", "left", "mustard raincoat"), ("Kofi", "right, in bed", "hospital gown"))}}
LATER = {"heading": "EXT. HOSPITAL - DAWN", "visual": "Ama alone outside at dawn, in a borrowed black coat", "duration_seconds": 6,
         "dialogue": [],
         "continuity": {"transition": "time_jump", "time_of_day": "dawn",
                        "intentional_changes": ["Ama now wears a borrowed black coat"],
                        "characters": who(("Ama", "centre", "borrowed black coat"))}}


def plan_for(scene):
    """The director's shots (the same story beats as a real plan would give)."""
    v = scene.get("visual") or ""
    if "enters" in v:
        return {"shots": [
            {"type": "establishing_wide", "description": "Ama enters the ward", "characters": [{"name": "Ama", "position": "left"}],
             "line_indices": [], "reason": "Where we are."},
            {"type": "medium", "description": "Ama approaches the bed", "characters": [{"name": "Ama", "position": "left"}],
             "line_indices": [0], "reason": "Her step closer."},
            {"type": "close_up", "description": "Ama recognises her brother", "characters": [{"name": "Ama", "position": "left"}],
             "line_indices": [], "reason": "The recognition."},
            {"type": "reaction", "description": "Reverse: Kofi reacts", "characters": [{"name": "Kofi", "position": "right"}],
             "line_indices": [1], "reason": "His fear."}]}
    if "ID card" in v:
        return {"shots": [
            {"type": "wide_two_shot", "description": "Both at the bed", "characters": [{"name": "Ama", "position": "left"},
                                                                                    {"name": "Kofi", "position": "right"}],
             "line_indices": [0], "reason": "Both of them."},
            {"type": "insert", "description": "The ID card passes between their hands", "characters": [],
             "line_indices": [], "reason": "The card matters."},
            {"type": "over_the_shoulder", "description": "Over Kofi's shoulder onto Ama holding the card",
             "characters": [{"name": "Kofi", "position": "right"}, {"name": "Ama", "position": "left"}],
             "line_indices": [], "reason": "Her reaction."},
            {"type": "close_up", "description": "Kofi watches her", "characters": [{"name": "Kofi", "position": "right"}],
             "line_indices": [], "reason": "His relief."}]}
    return {"shots": [{"type": "establishing_wide", "description": "Ama outside at dawn",
                       "characters": [{"name": "Ama", "position": "centre"}], "line_indices": [], "reason": "Time has passed."}]}


def png(color=(40, 30, 90)):
    out = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def film(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 100000)
    pid = project(client, u, title="False Name", production_style="pictures")["id"]
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": "INT. WARD - NIGHT\n\nA story.\n"})
    ai.breakdown_result = {"characters": [AMA, KOFI], "scenes": [dict(ENTER), dict(CARD), dict(LATER)]}
    assert start(client, u, pid, "breakdown").status_code == 200
    run(gen)
    for name in ("Ama", "Kofi"):          # each character's approved portrait (a real file)
        path = f"{u}/{pid}/portrait/{name}.png"
        files.put(path, png(), "image/png")
        db.execute("update public.studio_characters set reference_path = %s where project_id = %s and name = %s",
                   (path, pid, name))
    ai.shots_result = lambda prompt, scene: plan_for(scene)
    ids = [s["id"] for s in detail(client, u, pid)["scenes"]]
    assert start(client, u, pid, "shotplan", ids).status_code == 200
    run(gen)
    for sid in ids:
        assert client.post(f"/api/v1/studio/scenes/{sid}/shots/approve", headers=H(u), json={"approved": True}).status_code == 200
    return client, db, store, files, ai, gen, u, pid


def shots(client, u, pid):
    return [sh for s in detail(client, u, pid)["scenes"] for sh in s["shots"]]


def make(client, gen, u, pid, ids, **extra):
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "shot_image", "target_ids": ids, "request_id": uuid.uuid4().hex, **extra})
    assert r.status_code == 200, r.text
    run(gen, times=len(ids) + 2)
    return r


def problem(kind, severity="high", certainty="certain", intentional=False, expected="", seen="", fix=""):
    return {"kind": kind, "severity": severity, "certainty": certainty, "intentional": intentional,
            "expected": expected, "seen": seen, "fix": fix, "explanation": "x"}


def clean(overall=94, problems=(), observed=None):
    return {"scores": {"overall": overall, "character": 95, "outfit": 92}, "problems": list(problems),
            "observed": observed or {"characters": [{"name": "Ama", "outfit": "mustard raincoat", "position": "left"}]},
            "summary": "ok"}


# ------------------------------------------------------------------ the chain of comparisons

def test_the_same_character_across_eight_consecutive_shots_is_compared_shot_by_shot(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    first8 = [sh["id"] for sh in shots(client, u, pid)][:8]
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "shot_image", "target_ids": first8}).json()
    assert "one corrected picture" in est["note"], "the reserve for a correction is shown, not hidden"
    before = balance(client, u)["available"]
    make(client, gen, u, pid, first8)
    log = ai.inspect_log[-8:]
    assert len(log) == 8, "every shot picture was checked"
    assert [x["labels"][0] for x in log] == ["this shot's picture (check this one)"] * 8
    assert all(not x["labels"][1].startswith("shot") for x in log[:1]), "the first shot of the film follows nothing"
    for k in range(1, 4):
        assert log[k]["labels"][1].startswith(f"shot {k} of this scene (the shot just before)")
    assert log[4]["labels"][1].startswith("the last shot of the previous scene"), "a continuous scene follows the last shot"
    assert all(any("Ama" in label for label in x["labels"]) for x in log[:3]), "Ama's approved reference is compared"
    assert all(s["continuity_check"]["status"] == "passed" for s in shots(client, u, pid)[:8])
    spent = before - balance(client, u)["available"]
    assert 0 < spent < est["total_credits"], "the unused correction reserve is returned"
    # The image model got the previous shot as a real reference and the state it was seen to show.
    second = ai.image_log[1]
    assert "Reference image" in second["prompt"] and "the shot just before" in second["prompt"]
    assert "WAS SEEN TO SHOW" in second["prompt"] and "mustard raincoat" in second["prompt"]


def test_a_time_jump_to_dawn_starts_fresh_and_does_not_compare_with_the_night(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    every = [sh["id"] for sh in shots(client, u, pid)]
    make(client, gen, u, pid, every)
    dawn = ai.inspect_log[-1]
    assert "INT. WARD" not in dawn["prompt"].split("THE PICTURE IT FOLLOWS")[0][-50:]
    assert "THE PICTURE IT FOLLOWS" not in dawn["prompt"], "day/night change after a time jump: nothing to match"
    assert not any("previous scene" in label for label in dawn["labels"])


def test_who_is_in_the_shot_decides_whose_face_is_compared(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    every = [sh["id"] for sh in shots(client, u, pid)]
    make(client, gen, u, pid, every[:5])
    ama_only, kofi_only, both = ai.inspect_log[-5], ai.inspect_log[-2], ai.inspect_log[-1]
    assert any("Ama" in x for x in ama_only["labels"]) and not any("Kofi:" in x for x in ama_only["labels"])
    assert any("Kofi" in x for x in kofi_only["labels"]) and not any(x.startswith("reference: Ama") for x in kofi_only["labels"])
    assert any("Ama" in x for x in both["labels"]) and any("Kofi" in x for x in both["labels"]), "two people: both faces"


# ------------------------------------------------------------------ what counts as a mistake

def test_an_outfit_change_fails_and_one_corrected_picture_is_made_and_kept(film):
    client, db, store, files, ai, gen, u, pid = film
    seen = {"n": 0}

    def judge(prompt, labels):
        seen["n"] += 1
        if seen["n"] == 1:
            return clean(70, [problem("outfit", expected="mustard raincoat", seen="red dress",
                                      fix="Ama's raincoat must be mustard, as in shot 1")])
        return clean()
    ai.inspect_result = judge
    target = shots(client, u, pid)[0]["id"]
    before = balance(client, u)["available"]
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "shot_image", "target_ids": [target]}).json()["total_credits"]
    make(client, gen, u, pid, [target])
    assert len(ai.image_log) == 2, "exactly one automatic correction"
    assert "CORRECTION" in ai.image_log[1]["prompt"] and "raincoat must be mustard" in ai.image_log[1]["prompt"]
    sh = shots(client, u, pid)[0]
    assert sh["continuity_check"]["status"] == "passed"
    out = db.execute("select output from public.studio_jobs where project_id = %s and kind = 'shot_image' "
                     "order by created_at desc limit 1", (pid,)).fetchone()[0]
    assert out["attempts"] == 2 and out["first_try"] == "failed"
    rejected = db.execute("select made_with from public.studio_shots where id = %s", (target,)).fetchone()[0]["_rejected"]
    assert db.execute("select count(*) from public.studio_assets where path = %s", (rejected,)).fetchone()[0] == 1, \
        "the rejected picture is kept in the history"
    assert before - balance(client, u)["available"] <= est


def test_if_the_correction_is_worse_the_first_picture_is_kept_and_retries_stop(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean(60, [problem("character", expected="Ama", seen="another woman",
                                                                  fix="Keep Ama's face")]) \
        if "check this one" in labels[0] else clean()
    target = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [target])
    assert len(ai.image_log) == 2, "never more than one corrected picture"
    sh = shots(client, u, pid)[0]
    assert sh["continuity_check"]["status"] == "failed" and sh["continuity_check"]["critical"] == ["identity"]
    kept_over, path = db.execute("select made_with->>'_kept_over', image_path from public.studio_shots where id = %s",
                                 (target,)).fetchone()
    assert kept_over and kept_over != path, "the first picture is back; the correction stays in the history"
    assert db.execute("select count(*) from public.studio_assets where path in (%s, %s)", (kept_over, path)).fetchone()[0] == 2


def test_auto_fix_can_be_switched_off_and_then_nothing_is_reserved(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean(50, [problem("outfit")])
    target = shots(client, u, pid)[0]["id"]
    on = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                     json={"kind": "shot_image", "target_ids": [target]}).json()["total_credits"]
    off = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "shot_image", "target_ids": [target], "auto_fix": False}).json()["total_credits"]
    assert off < on
    make(client, gen, u, pid, [target], auto_fix=False)
    assert len(ai.image_log) == 1 and shots(client, u, pid)[0]["continuity_check"]["status"] == "failed"


def test_intentional_changes_cuts_and_uncertain_details_are_not_mistakes():
    ins = {"result": {"scores": {"overall": 90}, "summary": "", "observed": {}, "problems": [
        problem("outfit", intentional=True, expected="mustard raincoat", seen="black coat"),      # approved wardrobe change
        problem("location", severity="medium", certainty="uncertain", expected="ward", seen="background hidden"),
        problem("prop", severity="medium", intentional=True, expected="card in Kofi's hand", seen="card in Ama's hand")]},
        "image_path": "p.png"}
    rep = shotcheck.report(ins, "p.png")
    assert rep["status"] == "passed"
    assert all(c["passed"] for c in rep["checks"])
    assert any(c["problems"] for c in rep["checks"] if c["check"] == "clothing"), "still shown, marked intentional"


def test_a_critical_identity_error_fails_even_with_a_high_average():
    ins = {"result": {"scores": {"overall": 93, "character": 40}, "summary": "", "observed": {},
                      "problems": [problem("character", expected="Ama", seen="a different woman")]}, "image_path": "p.png"}
    rep = shotcheck.report(ins, "p.png")
    assert rep["status"] == "failed" and rep["critical"] == ["identity"]
    assert [c["check"] for c in rep["checks"] if not c["passed"]] == ["identity"]


def test_the_vision_check_is_told_how_shots_change(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    every = [sh["id"] for sh in shots(client, u, pid)]
    make(client, gen, u, pid, every[:6])
    reverse, insert = ai.inspect_log[3]["prompt"], ai.inspect_log[5]["prompt"]
    assert "THIS SHOT: reaction" in reverse and "A reverse angle shows the other side of the room" in reverse
    assert "a CUT to a new framing" in reverse and "Framing: close up" in reverse
    assert "THIS SHOT: insert" in insert and "Objects may move when the action moves them" in insert
    assert "Kofi hands Ama the ID card" in insert, "the action that moves the prop is given"
    assert "People enter and leave when the scene says so" in insert


# ------------------------------------------------------------------ failures

def test_an_unreadable_shot_picture_is_not_checked_and_the_check_is_refunded(film, monkeypatch):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    target = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [target], auto_fix=False)
    db.execute("update public.studio_shots set inspection = '{}'::jsonb where id = %s", (target,))
    path = shots(client, u, pid)[0]["id"] and db.execute("select image_path from public.studio_shots where id = %s",
                                                          (target,)).fetchone()[0]
    real = files.get
    monkeypatch.setattr(files, "get", lambda p: (_ for _ in ()).throw(OSError("gone")) if p == path else real(p))
    insp = __import__("studio_worker").StudioWorker(store, files, ai, "inspect")
    before = balance(client, u)["available"]
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "shot_inspect", "target_ids": [target], "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    run(insp)
    assert balance(client, u)["available"] == before
    assert shots(client, u, pid)[0]["continuity_check"]["status"] == "unchecked"
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed" and "couldn't be loaded" in job["error"]


def test_a_failed_comparison_keeps_the_picture_and_charges_only_the_picture(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.fail = {"inspect": RuntimeError("vision model down")}
    target = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [target])
    sh = shots(client, u, pid)[0]
    assert sh["image_url"] and sh["continuity_check"]["status"] == "unchecked"
    status, out = db.execute("select status, output from public.studio_jobs where project_id = %s and kind = 'shot_image' "
                             "order by created_at desc limit 1", (pid,)).fetchone()
    assert status == "succeeded" and out["check"] == "skipped"


# ------------------------------------------------------------------ the creator decides

def test_the_creator_can_accept_an_intentional_change_for_that_picture_only(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean(65, [problem("outfit", expected="mustard raincoat", seen="black coat")])
    target = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [target], auto_fix=False)
    r = client.post(f"/api/v1/studio/shots/{target}/continuity/accept", headers=H(u),
                    json={"accepted": True, "note": "She borrowed a coat"})
    assert r.status_code == 200 and r.json()["continuity_check"]["status"] == "accepted"
    assert shots(client, u, pid)[0]["continuity_check"]["accepted"]["note"] == "She borrowed a coat"
    make(client, gen, u, pid, [target], auto_fix=False)             # a new picture: the acceptance doesn't carry over
    assert shots(client, u, pid)[0]["continuity_check"]["status"] == "failed"


def test_an_approved_shot_is_never_overwritten_by_a_fix(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    target = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [target])
    path = db.execute("select image_path from public.studio_shots where id = %s", (target,)).fetchone()[0]
    client.post(f"/api/v1/studio/shots/{target}/approve", headers=H(u), json={"approved": True})
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "shot_image", "target_ids": [target], "request_id": uuid.uuid4().hex,
                          "correction": "different coat"})
    assert r.status_code == 400 and "approved" in r.text
    assert db.execute("select image_path from public.studio_shots where id = %s", (target,)).fetchone()[0] == path


def test_an_unchanged_shot_is_not_checked_or_charged_twice(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    target = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [target])
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "shot_inspect", "target_ids": [target]}).json()
    assert est["items"][0]["blocked"] == "Already checked, and nothing has changed since."


# ------------------------------------------------------------------ the assistant

def test_the_assistant_explains_a_failed_shot_and_proposes_a_fix_for_that_shot(film, monkeypatch):
    client, db, store, files, ai, gen, u, pid = film
    monkeypatch.setattr(cs, "_ai", ai)
    ai.inspect_result = lambda prompt, labels: clean(60, [problem("outfit", expected="mustard raincoat", seen="red dress",
                                                                  fix="Ama's raincoat must be mustard")]) \
        if "Ama recognises" in prompt else clean()
    ids = [sh["id"] for sh in shots(client, u, pid)][:4]
    make(client, gen, u, pid, ids, auto_fix=False)
    ai.chat_result = {"reply": "Ama's raincoat changed in shot 3. Should I fix it?", "memory": [], "proposals": [
        {"type": "fix_shot_picture", "title": "Fix shot 3", "summary": "x",
         "data": {"scene": 1, "shot": 3, "correction": "Ama's raincoat must be mustard"}},
        {"type": "check_shots", "title": "Check scene 1", "summary": "x", "data": {"scenes": [1]}}]}
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": "The previous image doesn't look like Ama.", "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    prompt = ai.chat_prompts[-1]
    assert "[continuity FAILED" in prompt and "clothing (high): expected mustard raincoat, seen red dress" in prompt
    assert "[continuity passed" in prompt
    assert "Should I fix shot 4?" in prompt, "the assistant is shown how to explain it"
    fix, check = r.json()["reply"]["proposals"]
    assert fix["job"] == "shot_image" and fix["targets"] == [ids[2]] and fix["extra"]["correction"]
    assert check["job"] == "shot_inspect" and check["targets"] == ids
    assert fix["status"] == "proposed", "nothing is made until the creator confirms the price"


def test_the_continuity_memory_is_structured_for_every_scene_and_shot(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    make(client, gen, u, pid, [shots(client, u, pid)[0]["id"]])
    m = client.get(f"/api/v1/studio/projects/{pid}/continuity-memory", headers=H(u)).json()
    assert [c["name"] for c in m["characters"]] == ["Ama", "Kofi"]
    assert [s["scene"] for s in m["scenes"]] == [1, 2, 3]
    s1 = m["scenes"][0]
    assert s1["time_of_day"] == "night" and s1["characters"][0]["position"] == "left"
    assert [x["type"] for x in s1["shots"]] == ["establishing_wide", "medium", "close_up", "reaction"]
    assert s1["shots"][0]["continuity"] == "passed" and s1["shots"][0]["seen"]["characters"][0]["outfit"] == "mustard raincoat"
    assert s1["shots"][1]["previous"] == 1 and s1["shots"][3]["next"] is None
    assert m["scenes"][2]["intentional_changes"] == ["Ama now wears a borrowed black coat"]
