"""One-click production: one confirmed price reserved up front, every step
made in order in the background, failed steps refunded and retried, the
film assembled only when every scene is ready. Fake AI and local Postgres."""
import os
import uuid

import pytest

import studio_ai
import studio_production as prod
from conftest import make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run
from test_video_drama import DAN, MAYA, QUIET, TALK, detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_production.sql")

SCENES = [dict(TALK, heading="EXT. ROOFTOP - NIGHT"), dict(QUIET),
          dict(TALK, heading="INT. FLAT - DAY", visual="Maya alone in her flat, Daniel at the door")]


def setup(env, style="pictures", scenes=SCENES, credits=8000):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, credits)
    p = project(client, u, production_style=style)
    split(client, ai, gen, u, p["id"], list(scenes))
    return client, db, ai, gen, ren, u, p["id"]


def quote(client, u, pid, **body):
    r = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json=body)
    assert r.status_code == 200, r.text
    return r.json()


def start(client, u, pid, total, **body):
    return client.post(f"/api/v1/studio/projects/{pid}/production", headers=H(u),
                       json={"request_id": uuid.uuid4().hex, "confirm_credits": total, **body})


def drive(gen, ren, rounds=12):
    for _ in range(rounds):
        run(gen)
        run(ren)


def status(client, u, pid):
    return client.get(f"/api/v1/studio/projects/{pid}/production", headers=H(u)).json()["production"]


def test_one_price_reserved_then_everything_made_in_order(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    q = quote(client, u, pid)
    assert [s["stage"] for s in q["stages"]] == ["sheets", "images", "voices", "render"]
    assert [len(s["items"]) for s in q["stages"]] == [2, 3, 2, 1] and not q["blocked"]
    total = q["total_credits"]
    assert total == sum(s["credits"] for s in q["stages"]) and "reserved" in q["note"]
    before = balance(client, u)
    r = start(client, u, pid, total)
    assert r.status_code == 200, r.text
    after = balance(client, u)
    assert after["available"] == before["available"] - total and after["held"] == total, "whole price reserved at once"
    drive(gen, ren)
    s = status(client, u, pid)
    assert s["status"] == "done" and s["percent"] == 100 and s["render_id"], s
    assert all(x["done"] == x["total"] for x in s["stages"])
    d = detail(client, u, pid)
    assert len(d["renders"]) == 1 and all(x["has_image"] for x in d["scenes"])
    assert all(x["status"] == "approved" or x["status"] == "draft" for c in d["characters"] for x in c["reference_views"])
    end = balance(client, u)
    assert end["held"] == 0 and end["available"] == before["available"] - s["spent_credits"]
    assert 0 < s["spent_credits"] <= total, "only the real cost is kept; the rest came back"
    # Sheets were made before scene pictures, so the pictures used them.
    kinds = [c[0] for c in ai.calls if c[0] in ("identity", "image")]
    assert kinds.index("identity") < len(kinds) - 3


def test_talking_video_productions_make_clips_not_separate_voices(env):
    client, db, ai, gen, ren, u, pid = setup(env, style="video")
    q = quote(client, u, pid)
    assert [s["stage"] for s in q["stages"]] == ["sheets", "images", "videos", "render"]
    assert start(client, u, pid, q["total_credits"]).status_code == 200
    drive(gen, ren)
    assert status(client, u, pid)["status"] == "done"
    assert all(x["has_video"] for x in detail(client, u, pid)["scenes"])


def test_what_is_already_made_is_not_made_or_paid_again(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    first = quote(client, u, pid, sheets=False)
    s1 = detail(client, u, pid)["scenes"][0]["id"]
    client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                json={"kind": "image", "target_ids": [s1], "request_id": uuid.uuid4().hex})
    run(gen)
    again = quote(client, u, pid, sheets=False)
    imgs = lambda q: next(s for s in q["stages"] if s["stage"] == "images")["items"]
    assert len(imgs(again)) == len(imgs(first)) - 1 and again["total_credits"] < first["total_credits"]


def test_a_failed_step_is_refunded_and_retried_once_automatically(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    q = quote(client, u, pid, sheets=False)
    ai.fail["image"] = [studio_ai.ProviderError("boom")]
    assert start(client, u, pid, q["total_credits"], sheets=False).status_code == 200
    drive(gen, ren)
    s = status(client, u, pid)
    assert s["status"] == "done"
    assert sum(x["failed"] for x in s["stages"]) == 1
    assert balance(client, u)["held"] == 0


def test_a_step_that_fails_twice_pauses_the_run_and_resume_finishes_it(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    q = quote(client, u, pid, sheets=False)
    start_balance = balance(client, u)["available"]
    boom = studio_ai.ProviderError("boom")
    real = ai.image

    def flaky(prompt, refs, ar, size="1K"):
        if "SCENE:" in prompt and "face each other in the rain" in prompt:
            raise boom                                   # scene 1's picture fails every time
        return real(prompt, refs, ar, size)

    ai.image = flaky
    assert start(client, u, pid, q["total_credits"], sheets=False).status_code == 200
    drive(gen, ren)
    s = status(client, u, pid)
    assert s["status"] == "needs_attention" and "Scene 1 picture" in s["problem"], s
    assert not detail(client, u, pid)["renders"], "no film until every scene is ready"
    held_while_paused = balance(client, u)["held"]
    assert held_while_paused > 0, "the rest stays reserved while it waits"
    ai.image = real
    r = client.post(f"/api/v1/studio/production/{s['id']}/resume", headers=H(u))
    assert r.status_code == 200 and r.json()["production"]["status"] == "running"
    drive(gen, ren)
    s = status(client, u, pid)
    assert s["status"] == "done" and balance(client, u)["held"] == 0
    assert s["spent_credits"] <= s["quoted_credits"]
    # Every credit is accounted for: the starting grant minus what was kept.
    assert balance(client, u)["available"] == start_balance - s["spent_credits"]


def test_cancel_returns_the_reservation_and_keeps_what_was_made(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    before = balance(client, u)
    q = quote(client, u, pid)
    rid = start(client, u, pid, q["total_credits"]).json()["production"]["id"]
    run(gen)        # makes the first stage's jobs and works through them
    r = client.post(f"/api/v1/studio/production/{rid}/cancel", headers=H(u))
    assert r.status_code == 200 and r.json()["production"]["status"] == "canceled"
    end = balance(client, u)
    spent = r.json()["production"]["spent_credits"]
    assert end["held"] == 0 and end["available"] == before["available"] - spent
    drive(gen, ren)
    assert status(client, u, pid)["status"] == "canceled" and not detail(client, u, pid)["renders"]


def test_price_check_blocks_balance_and_ownership(env):
    client, db, ai, gen, ren, u, pid = setup(env, credits=50)
    q = quote(client, u, pid)
    r = start(client, u, pid, q["total_credits"] + 1)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "price_changed"
    r = start(client, u, pid, q["total_credits"])
    assert r.status_code == 402 and balance(client, u)["held"] == 0
    grant(client, u, 9000)
    assert start(client, u, pid, q["total_credits"]).status_code == 200
    r = start(client, u, pid, q["total_credits"])
    assert r.status_code == 409 and r.json()["detail"]["code"] in ("already_running", "price_changed")
    other = make_user(db)
    assert client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(other), json={}).status_code == 404
    rid = status(client, u, pid)["id"]
    assert client.post(f"/api/v1/studio/production/{rid}/cancel", headers=H(other)).status_code == 404
    empty = project(client, u)["id"]
    r = start(client, u, empty, 0)
    assert r.status_code == 409 and "scenes first" in r.json()["detail"]["message"]
