"""Studio budget protection (creator_studio_budget.sql + studio_production): a production never spends more than the
creator approved, concurrent workers can't get past the cap, retries stay inside it, more credits need the creator's
explicit approval (reserved first), video waits for the storyboard review, and every credit is in the history.
Fake AI and a real local Postgres."""
import os
import threading
import uuid

import pytest

import studio_ai
import studio_production as prod
from conftest import connect, make_user
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run
from test_studio_production import SCENES, drive, quote, setup, status

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_budget.sql")


def start(client, u, pid, total, **body):
    return client.post(f"/api/v1/studio/projects/{pid}/production", headers=H(u),
                       json={"request_id": uuid.uuid4().hex, "confirm_credits": total, "direct": False, **body})


def run_row(db, rid):
    cur = db.execute("select * from public.studio_production_runs where id = %s", (rid,))
    return dict(zip([c.name for c in cur.description], cur.fetchone()))


def create_job(db, rid, credits, key=None, target=None):
    return db.execute("select public.studio_run_create_job(%s, 'image', 'generation', %s, %s, '{}'::jsonb, %s, null, true)",
                      (rid, key or f"run:{rid}:images:{uuid.uuid4()}:0:0", credits, target)).fetchone()[0]


def charged(db, rid):
    return db.execute("select public.studio_run_committed(%s)", (rid,)).fetchone()[0]


def test_the_quote_is_the_maximum_and_a_step_can_never_go_past_the_approved_budget(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    q = quote(client, u, pid)
    assert q["total_credits"] == q["steps_credits"] + q["safety_credits"]
    r = start(client, u, pid, q["total_credits"], storyboard_review=False)
    rid = r.json()["production"]["id"]
    approved = run_row(db, rid)["approved_credits"]
    assert approved == q["total_credits"]
    held_before = balance(client, u)
    out = create_job(db, rid, approved + 1)
    assert out["ok"] is False and out["error"] == "budget_exceeded" and out["needed"] == 1
    row = run_row(db, rid)
    assert row["status"] == "needs_approval" and row["needs_credits"] == 1
    assert balance(client, u) == held_before, "nothing was taken from the wallet"
    assert db.execute("select count(*) from public.studio_jobs where idempotency_key like %s",
                      (f"run:{rid}:%",)).fetchone()[0] == 0


def test_concurrent_workers_cannot_get_past_the_cap_together(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    q = quote(client, u, pid)
    rid = start(client, u, pid, q["total_credits"], storyboard_review=False).json()["production"]["id"]
    approved = run_row(db, rid)["approved_credits"]
    each = approved // 2 + 1                        # one fits, two together don't
    results, barrier = [], threading.Barrier(2)

    def worker():
        c = connect()
        try:
            barrier.wait()
            r = c.execute("select public.studio_run_create_job(%s, 'image', 'generation', %s, %s, '{}'::jsonb, null, "
                          "null, true)", (rid, f"run:{rid}:images:{uuid.uuid4()}:0:0", each)).fetchone()[0]
            results.append(r)
        finally:
            c.close()
    threads = [threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(bool(r.get("ok")) for r in results) == [False, True]
    assert next(r for r in results if not r.get("ok"))["error"] == "budget_exceeded"
    assert charged(db, rid) <= approved


def test_a_retry_reuses_what_its_failed_try_returned_and_stays_inside_the_budget(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    q = quote(client, u, pid)
    rid = start(client, u, pid, q["total_credits"], storyboard_review=False).json()["production"]["id"]
    approved = run_row(db, rid)["approved_credits"]
    first = create_job(db, rid, approved)
    assert first["ok"]
    job = first["job_id"]
    db.execute("update public.studio_jobs set status = 'running', lease = gen_random_uuid() where id = %s", (job,))
    lease = db.execute("select lease from public.studio_jobs where id = %s", (job,)).fetchone()[0]
    assert db.execute("select public.studio_fail_job(%s, %s, 'boom', 'boom', false)", (job, lease)).fetchone()[0] == "failed"
    assert charged(db, rid) == 0, "the failed try was refunded"
    retry = create_job(db, rid, approved)
    assert retry["ok"], "the retry fits: it reuses the refund"
    again = create_job(db, rid, 1)
    assert again["ok"] is False and again["error"] == "budget_exceeded", "but nothing beyond the budget"
    assert charged(db, rid) == approved


def test_more_credits_need_the_creators_exact_approval_and_are_reserved_first(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    q = quote(client, u, pid, sheets=False)
    rid = start(client, u, pid, q["total_credits"], sheets=False, storyboard_review=False).json()["production"]["id"]
    # The plan grew after approval (e.g. the director planned more shots): pretend the approved budget is short.
    db.execute("update public.studio_production_runs set approved_credits = 3 where id = %s", (rid,))
    drive(gen, ren, rounds=3)
    s = status(client, u, pid)
    assert s["status"] == "needs_approval" and s["needs_credits"] > 0, s
    assert "more credits than you approved" in s["problem"] and "new total" in s["problem"]
    jobs_before = db.execute("select count(*) from public.studio_jobs where idempotency_key like %s",
                             (f"run:{rid}:%",)).fetchone()[0]
    drive(gen, ren, rounds=3)
    assert db.execute("select count(*) from public.studio_jobs where idempotency_key like %s",
                      (f"run:{rid}:%",)).fetchone()[0] == jobs_before, "nothing starts while it waits"
    other = make_user(db)
    assert client.post(f"/api/v1/studio/production/{rid}/budget", headers=H(other),
                       json={"credits": s["needs_credits"]}).status_code == 404
    wrong = client.post(f"/api/v1/studio/production/{rid}/budget", headers=H(u), json={"credits": s["needs_credits"] + 1})
    assert wrong.status_code == 409 and wrong.json()["detail"]["code"] == "amount_changed"
    before = balance(client, u)
    ok = client.post(f"/api/v1/studio/production/{rid}/budget", headers=H(u), json={"credits": s["needs_credits"]})
    assert ok.status_code == 200, ok.text
    after = balance(client, u)
    assert after["available"] == before["available"] - s["needs_credits"], "the increase was reserved"
    assert ok.json()["production"]["approved_credits"] == 3 + s["needs_credits"]
    hist = client.get(f"/api/v1/studio/production/{rid}/history", headers=H(u)).json()
    assert [e["kind"] for e in hist["events"]][:3] == ["reserved", "increase_requested", "increase_approved"]


def test_an_increase_without_enough_credits_is_refused_and_nothing_is_bought(env):
    client, db, ai, gen, ren, u, pid = setup(env, credits=8000)
    q = quote(client, u, pid, sheets=False)
    rid = start(client, u, pid, q["total_credits"], sheets=False, storyboard_review=False).json()["production"]["id"]
    db.execute("update public.studio_production_runs set status = 'needs_approval', needs_credits = 999999 where id = %s",
               (rid,))
    r = client.post(f"/api/v1/studio/production/{rid}/budget", headers=H(u), json={"credits": 999999})
    assert r.status_code == 402
    assert run_row(db, rid)["status"] == "needs_approval"


def test_a_whole_run_with_failures_never_spends_more_than_approved_and_returns_the_rest(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    start_bal = balance(client, u)["available"]
    q = quote(client, u, pid, sheets=False)
    ai.fail["image"] = [studio_ai.ProviderError("boom"), studio_ai.ProviderError("boom")]
    rid = start(client, u, pid, q["total_credits"], sheets=False, storyboard_review=False).json()["production"]["id"]
    drive(gen, ren, rounds=16)
    s = status(client, u, pid)
    assert s["status"] == "done", s
    assert s["spent_credits"] <= s["approved_credits"] == q["total_credits"]
    end = balance(client, u)
    assert end["held"] == 0 and end["available"] == start_bal - s["spent_credits"], "every unused credit came back"
    hist = client.get(f"/api/v1/studio/production/{rid}/history", headers=H(u)).json()
    assert hist["totals"]["charged"] == s["spent_credits"]
    assert any(st["status"] == "failed" and st["charged"] == 0 and st["refunded"] == st["held"] for st in hist["steps"])
    # Unused credits come back either step by step (refunds) or at the end (what's left of the reservation):
    # together with what was kept they add up to the approved budget minus what never left it.
    assert hist["totals"]["charged"] + hist["totals"]["refunded"] == sum(x["held"] for x in hist["steps"])
    assert all(e["kind"] != "returned" or e["credits"] > 0 for e in hist["events"])


def test_storyboard_review_is_on_by_default_and_gates_video(env):
    client, db, ai, gen, ren, u, pid = setup(env, style="video")
    q = quote(client, u, pid)
    r = start(client, u, pid, q["total_credits"])                  # no opt-out: review is on
    rid = r.json()["production"]["id"]
    drive(gen, ren)
    s = status(client, u, pid)
    assert s["status"] == "storyboard_review" and "approve it to start the video" in s["problem"], s
    assert s["storyboard_review"] is True
    videos = lambda: db.execute("select count(*) from public.studio_jobs where idempotency_key like %s and kind in "
                                "('video', 'shot_video')", (f"run:{rid}:%",)).fetchone()[0]
    assert videos() == 0, "no paid video before approval"
    other = make_user(db)
    assert client.post(f"/api/v1/studio/production/{rid}/storyboard/approve", headers=H(other)).status_code == 404
    ok = client.post(f"/api/v1/studio/production/{rid}/storyboard/approve", headers=H(u))
    assert ok.status_code == 200 and ok.json()["production"]["status"] == "running"
    drive(gen, ren)
    assert status(client, u, pid)["status"] == "done" and videos() > 0
    assert client.post(f"/api/v1/studio/production/{rid}/storyboard/approve", headers=H(u)).status_code == 409


def test_turning_storyboard_review_off_is_a_deliberate_choice(env):
    client, db, ai, gen, ren, u, pid = setup(env, style="video")
    q = quote(client, u, pid)
    start(client, u, pid, q["total_credits"], storyboard_review=False)
    drive(gen, ren)
    assert status(client, u, pid)["status"] == "done"


def test_a_paused_run_can_be_canceled_and_everything_unused_comes_back(env):
    client, db, ai, gen, ren, u, pid = setup(env, style="video")
    before = balance(client, u)
    q = quote(client, u, pid)
    rid = start(client, u, pid, q["total_credits"]).json()["production"]["id"]
    drive(gen, ren)
    assert status(client, u, pid)["status"] == "storyboard_review"
    r = client.post(f"/api/v1/studio/production/{rid}/cancel", headers=H(u))
    assert r.status_code == 200 and r.json()["production"]["status"] == "canceled"
    end = balance(client, u)
    assert end["held"] == 0 and end["available"] == before["available"] - r.json()["production"]["spent_credits"]


def test_only_the_server_can_call_the_budget_functions(env):
    client, db, ai, gen, ren, u, pid = setup(env)
    db.execute("begin")
    try:
        db.execute("set local role authenticated")
        with pytest.raises(Exception):
            db.execute("select public.studio_run_approve_budget(gen_random_uuid(), gen_random_uuid(), 1)")
    finally:
        db.execute("rollback")
