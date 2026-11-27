"""Free defect fixes, paid remakes and the full production estimate.

* A free fix happens only when VIYO's own check confirms a defect (a mismatch with the approved references or
  storyboard, or a broken clip); otherwise nothing is remade or charged. One successful free fix per shot and output,
  at most two checks, never two at once: it can't be turned into unlimited free generation.
* A remake (replacing what exists) needs the creator to confirm its exact price.
* The production quote shows every cost category, the maximum, its value per credit and the rules.
Fake AI and local Postgres."""
import os
import uuid

import pytest

import studio_worker as sw
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run
from test_shot_continuity import clean, film, make, problem, shots  # noqa: F401
from test_studio_production import quote, setup

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_budget.sql")


def wrong_face():
    return clean(overall=40, problems=[problem("character", expected="Ama's face from her reference",
                                               seen="a different woman", fix="Use Ama's face from her reference")])


def fix(client, u, shot_id, target="picture"):
    return client.post(f"/api/v1/studio/shots/{shot_id}/fix", headers=H(u),
                       json={"target": target, "request_id": uuid.uuid4().hex})


def jobs(db, kind):
    return db.execute("select status, charged_credits, estimated_credits, error_code from public.studio_jobs "
                      "where kind = %s order by created_at", (kind,)).fetchall()


def test_a_confirmed_defect_is_fixed_for_free_once(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    first = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [first])
    before = balance(client, u)["available"]
    ai.inspect_result = lambda prompt, labels: wrong_face()      # VIYO's check sees the wrong face
    r = fix(client, u, first)
    assert r.status_code == 200 and r.json()["free"] is True
    old = shots(client, u, pid)[0]
    run(gen)
    assert jobs(db, "shot_fix")[-1][:2] == ("succeeded", 0), "remade, and charged nothing"
    assert balance(client, u)["available"] == before
    assert db.execute("select count(*) from public.studio_usage u join public.studio_jobs j on j.id = u.job_id "
                      "where j.kind = 'shot_fix' and u.credits_charged = 0").fetchone()[0] >= 1, "VIYO's cost is recorded"
    st = client.get(f"/api/v1/studio/shots/{first}/fix", headers=H(u)).json()
    assert st["picture"]["used"] is True and st["picture"]["available"] is False
    again = fix(client, u, first)
    assert again.status_code == 409 and again.json()["detail"]["code"] == "free_fix_used", "only one free fix per shot"
    assert old is not None


def test_no_defect_found_means_nothing_remade_and_nothing_charged(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    first = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [first])
    path = db.execute("select image_path from public.studio_shots where id = %s", (first,)).fetchone()[0]
    before = balance(client, u)["available"]
    assert fix(client, u, first).status_code == 200
    run(gen)
    status, charged, _held, code = jobs(db, "shot_fix")[-1]
    assert (status, code) == ("failed", "no_defect_found") and not charged
    assert db.execute("select image_path from public.studio_shots where id = %s", (first,)).fetchone()[0] == path
    assert balance(client, u)["available"] == before
    assert db.execute("select count(*) from public.studio_usage where status = 'absorbed'").fetchone()[0] >= 1


def test_free_checks_are_limited_and_never_run_twice_at_once(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    first = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [first])
    assert fix(client, u, first).status_code == 200
    busy = fix(client, u, first)
    assert busy.status_code == 409 and "already being checked" in busy.text, "no second check while one runs"
    run(gen)                                              # check 1: no defect
    assert fix(client, u, first).status_code == 200
    run(gen)                                              # check 2: no defect
    third = fix(client, u, first)
    assert third.status_code == 409 and third.json()["detail"]["code"] == "free_fix_checks_used"
    assert len(jobs(db, "shot_fix")) == 2


def test_a_broken_clip_is_detected_without_ai_and_fixed_for_free(film, monkeypatch):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    first = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [first])
    db.execute("update public.studio_shots set video_path = image_path where id = %s", (first,))   # a "clip"
    monkeypatch.setattr(sw.StudioWorker, "clip_defects", lambda self, data, secs: ["The picture is frozen for 4.0s of 5.0s."])
    calls = []
    monkeypatch.setattr(sw.StudioWorker, "do_shot_video",
                        lambda self, job, project: (calls.append(job["input"]) or {"path": "x"}, sw.pricing.video_cost(5)))
    assert fix(client, u, first, "picture").status_code == 409, "a shot with a clip is fixed as a clip"
    before = balance(client, u)["available"]
    assert fix(client, u, first, "clip").status_code == 200
    run(gen)
    assert jobs(db, "shot_fix")[-1][:2] == ("succeeded", 0) and calls == [{"shot_id": first}]
    assert balance(client, u)["available"] == before


def test_approved_shots_need_unapproving_before_a_fix(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    first = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [first])
    client.post(f"/api/v1/studio/shots/{first}/approve", headers=H(u), json={"approved": True})
    assert fix(client, u, first).status_code == 409


def test_a_paid_remake_needs_the_exact_price_confirmed(film):
    client, db, store, files, ai, gen, u, pid = film
    ai.inspect_result = lambda prompt, labels: clean()
    first = shots(client, u, pid)[0]["id"]
    make(client, gen, u, pid, [first])
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u),
                      json={"kind": "shot_image", "target_ids": [first]}).json()
    assert est["remakes"] == 1 and est["items"][0]["remake"] is True and "free" in est["remake_note"]
    body = {"kind": "shot_image", "target_ids": [first], "request_id": uuid.uuid4().hex}
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u), json=body)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "confirm_remake"
    assert r.json()["detail"]["total_credits"] == est["total_credits"]
    wrong = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                        json={**body, "confirm_credits": est["total_credits"] - 1})
    assert wrong.status_code == 409
    ok = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                     json={**body, "confirm_credits": est["total_credits"]})
    assert ok.status_code == 200, ok.text


def test_making_something_for_the_first_time_needs_no_extra_confirmation(film):
    client, db, store, files, ai, gen, u, pid = film
    first = shots(client, u, pid)[0]["id"]
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "shot_image", "target_ids": [first], "request_id": uuid.uuid4().hex})
    assert r.status_code == 200


def test_the_quote_answers_what_a_creator_needs_before_paying(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    q = quote(client, u, pid)
    e = q["estimate"]
    keys = [c["key"] for c in e["categories"]]
    assert keys == ["story", "characters", "storyboard", "video", "voices", "music", "render"]
    assert e["max_credits"] == q["total_credits"] == e["steps_max"] + e["safety_credits"]
    assert e["max_value_usd"] == round(q["total_credits"] * 0.01, 2)
    assert sum(c["remaining_max"] for c in e["categories"]) == e["steps_max"]
    assert e["episodes"]["in_this_production"] == 1 and "separate productions" in e["episodes"]["note"]
    t = q["terms"]
    assert "not cash value" in t["credit_value_label"]
    assert any("Premium" in x and "doesn't include Studio" in x for x in t["currencies"])
    assert any("never spend more" in x for x in t["budget"]) and any("automatically" in x for x in t["budget"])
    assert any("never charged" in x for x in t["failures"])
    assert any("one free fix" in x for x in t["revisions"]) and any("confirm" in x for x in t["revisions"])
    assert any("no minimum" in x for x in t["publishing"]) and any("non-exclusive" in x for x in t["publishing"])
    assert any("storyboard" in x for x in t["storyboard"])
    assert client.get("/api/v1/studio/terms", headers=H(u)).json()["credit_value_usd"] == 0.01
