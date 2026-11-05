"""Studio credits and the Studio job queue, against the real SQL in
creator_studio_schema.sql on a local Postgres."""
from decimal import Decimal
import threading
import uuid

import pytest

from conftest import ROOT, connect, make_user, pg_rpc


@pytest.fixture(scope="module")
def studio_schema(migrated):
    conn = connect()
    conn.execute((ROOT / "creator_studio_schema.sql").read_text())
    conn.close()
    return migrated


@pytest.fixture
def sdb(studio_schema, db):
    db.execute("truncate public.studio_usage, public.studio_jobs, public.studio_credit_ledger, "
               "public.studio_credit_hold_allocations, public.studio_credit_holds, public.studio_credit_lots, "
               "public.studio_credit_purchases, public.studio_renders, public.studio_assets, public.studio_scenes, "
               "public.studio_characters, public.studio_projects cascade")
    return db


def project(db, owner):
    pid = uuid.uuid4()
    db.execute("insert into public.studio_projects (id, owner_id, title) values (%s, %s, 'Test')", (pid, owner))
    return pid


def bal(rpc, user):
    return rpc("studio_credit_balance", {"p_user_id": user})


def grant(rpc, user, credits, key=None, source="grant", **kw):
    return rpc("studio_add_credits", {"p_user_id": user, "p_credits": credits, "p_source": source,
                                      "p_key": key or uuid.uuid4().hex, **kw})


def job(rpc, owner, proj, credits, key=None, kind="image", lane="generation"):
    return rpc("studio_create_job", {"p_owner": owner, "p_project": proj, "p_kind": kind, "p_lane": lane,
                                     "p_key": key or uuid.uuid4().hex, "p_credits": credits, "p_input": {}})


def test_credits_are_added_once_per_key_and_per_order(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    assert grant(rpc, u, 100, key="welcome")["ok"]
    assert grant(rpc, u, 100, key="welcome")["duplicate"]
    r1 = grant(rpc, u, 500, key="ls-1", source="purchase", p_provider="lemonsqueezy", p_order_id="o1", p_gross_usd=Decimal("5.5"), p_fee_usd=Decimal("0.77"))
    r2 = grant(rpc, u, 500, key="ls-1-retry", source="purchase", p_provider="lemonsqueezy", p_order_id="o1")
    assert r1["ok"] and r2["duplicate"], "the same order is never credited twice"
    b = bal(rpc, u)
    assert (b["available"], b["purchased"], b["promotional"]) == (600, 500, 100)


def test_studio_credits_never_touch_watch_coins(sdb):
    rpc = pg_rpc(sdb)
    u = make_user(sdb, balance=77)
    p = project(sdb, u)
    grant(rpc, u, 300, source="purchase", p_provider="x", p_order_id="o")
    jid = job(rpc, u, p, 120)["job_id"]
    claimed = rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180})
    rpc("studio_complete_job", {"p_job_id": jid, "p_lease": claimed["lease"], "p_output": {}, "p_credits": 100, "p_usage": None})
    assert sdb.execute("select points_balance from public.profiles where id=%s", (u,)).fetchone()[0] == 77
    assert sdb.execute("select count(*) from public.transactions").fetchone()[0] == 0
    assert sdb.execute("select count(*) from public.creator_earnings").fetchone()[0] == 0


def test_job_holds_the_estimate_and_a_repeat_tap_charges_nothing_more(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 100)
    a = job(rpc, u, p, 40, key="tap-1")
    b = job(rpc, u, p, 40, key="tap-1")
    assert a["ok"] and b["duplicate"] and a["job_id"] == b["job_id"]
    assert bal(rpc, u)["available"] == 60 and bal(rpc, u)["held"] == 40


def test_not_enough_credits_changes_nothing(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 30)
    r = job(rpc, u, p, 31)
    assert r == {"ok": False, "error": "insufficient_credits", "balance": 30, "needed": 31}
    assert bal(rpc, u)["available"] == 30
    assert sdb.execute("select count(*) from public.studio_jobs").fetchone()[0] == 0
    assert sdb.execute("select count(*) from public.studio_credit_holds where credits_held > 0").fetchone()[0] == 0


def test_someone_elses_project_is_refused(sdb):
    rpc, a, b = pg_rpc(sdb), make_user(sdb), make_user(sdb)
    grant(rpc, b, 100)
    assert job(rpc, b, project(sdb, a), 10) == {"ok": False, "error": "project_not_found"}


def test_promotional_credits_are_spent_first_and_stay_spent_after_a_partial_refund(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 50)  # promotional
    grant(rpc, u, 100, source="purchase", p_provider="x", p_order_id="o")
    jid = job(rpc, u, p, 70)["job_id"]
    b = bal(rpc, u)
    assert (b["promotional"], b["purchased"]) == (0, 80)
    c = rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180})
    assert rpc("studio_complete_job", {"p_job_id": jid, "p_lease": c["lease"], "p_output": {"ok": 1},
                                       "p_credits": 55, "p_usage": {"provider": "google", "model": "m",
                                                                    "provider_cost_usd": 0.41}})
    b = bal(rpc, u)
    assert b["available"] == 95 and b["held"] == 0
    assert (b["promotional"], b["purchased"]) == (0, 95), "the unused part goes back to the bought credits"
    row = sdb.execute("select charged_credits, status from public.studio_jobs where id=%s", (jid,)).fetchone()
    assert row == (55, "succeeded")
    usage = sdb.execute("select provider, credits_charged, status, provider_cost_usd from public.studio_usage").fetchone()
    assert usage[:3] == ("google", 55, "charged") and float(usage[3]) == 0.41


def test_never_charges_more_than_the_approved_hold(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 100)
    jid = job(rpc, u, p, 20)["job_id"]
    c = rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180})
    rpc("studio_complete_job", {"p_job_id": jid, "p_lease": c["lease"], "p_output": {}, "p_credits": 999, "p_usage": None})
    assert bal(rpc, u)["available"] == 80


def test_failure_returns_every_credit_and_retry_requeues(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 100)
    jid = job(rpc, u, p, 40)["job_id"]
    c = rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180})
    assert rpc("studio_fail_job", {"p_job_id": jid, "p_lease": c["lease"], "p_code": "provider_busy",
                                   "p_message": "busy", "p_retry": True, "p_absorbed_usage": None}) == "retrying"
    assert bal(rpc, u)["held"] == 40, "credits stay held while it retries"
    c2 = rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180})
    assert c2["id"] == jid and c2["attempts"] == 2
    assert rpc("studio_fail_job", {"p_job_id": jid, "p_lease": c2["lease"], "p_code": "provider_error",
                                   "p_message": "We couldn't make this.", "p_retry": True,
                                   "p_absorbed_usage": {"provider": "google", "model": "m", "provider_cost_usd": 0.03}}) == "failed"
    b = bal(rpc, u)
    assert b["available"] == 100 and b["held"] == 0
    assert sdb.execute("select status from public.studio_usage").fetchone()[0] == "absorbed"
    assert rpc("studio_complete_job", {"p_job_id": jid, "p_lease": c2["lease"], "p_output": {}, "p_credits": 5,
                                       "p_usage": None}) is False, "a finished job can't be charged later"


def test_cancel_before_start_returns_credits_but_a_started_job_cannot_be_cancelled(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 100)
    j1 = job(rpc, u, p, 30)["job_id"]
    assert rpc("studio_cancel_job", {"p_owner": u, "p_job_id": j1}) == {"ok": True}
    assert bal(rpc, u)["available"] == 100
    j2 = job(rpc, u, p, 30)["job_id"]
    rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180})
    assert rpc("studio_cancel_job", {"p_owner": u, "p_job_id": j2})["error"] == "already_started"
    other = make_user(sdb)
    assert rpc("studio_cancel_job", {"p_owner": other, "p_job_id": j2})["error"] == "not_found"


def test_a_dead_worker_job_is_taken_over_then_failed_and_refunded(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 100)
    jid = job(rpc, u, p, 50)["job_id"]
    c1 = rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180})
    rpc("studio_heartbeat", {"p_job_id": jid, "p_lease": c1["lease"], "p_provider_job_id": "op-123"})
    assert rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180}) is None, "a live job isn't stolen"
    sdb.execute("update public.studio_jobs set heartbeat_at = now() - interval '10 minutes' where id=%s", (jid,))
    c2 = rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180})
    assert c2["id"] == jid and c2["provider_job_id"] == "op-123", "the new worker can resume the provider job"
    assert rpc("studio_heartbeat", {"p_job_id": jid, "p_lease": c1["lease"], "p_provider_job_id": None}) is False
    sdb.execute("update public.studio_jobs set heartbeat_at = now() - interval '10 minutes' where id=%s", (jid,))
    assert rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180}) is None
    row = sdb.execute("select status, error_code from public.studio_jobs where id=%s", (jid,)).fetchone()
    assert row == ("failed", "worker_lost")
    assert bal(rpc, u)["available"] == 100


def test_lanes_are_separate(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 100)
    job(rpc, u, p, 10, kind="render", lane="render")
    assert rpc("studio_claim_job", {"p_lane": "generation", "p_stale_seconds": 180}) is None
    assert rpc("studio_claim_job", {"p_lane": "render", "p_stale_seconds": 180})["kind"] == "render"


def test_open_job_cap_stops_runaway_loops(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 1000)
    for _ in range(3):
        assert rpc("studio_create_job", {"p_owner": u, "p_project": p, "p_kind": "image", "p_lane": "generation",
                                         "p_key": uuid.uuid4().hex, "p_credits": 1, "p_input": {}, "p_max_open": 3})["ok"]
    r = rpc("studio_create_job", {"p_owner": u, "p_project": p, "p_kind": "image", "p_lane": "generation",
                                  "p_key": uuid.uuid4().hex, "p_credits": 1, "p_input": {}, "p_max_open": 3})
    assert r["error"] == "too_many_jobs"


def test_parallel_requests_can_never_overspend(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 100)
    results = []

    def worker():
        conn = connect()
        try:
            results.append(job(pg_rpc(conn), u, p, 30))
        finally:
            conn.close()

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for r in results if r["ok"]) == 3
    assert sum(1 for r in results if r.get("error") == "insufficient_credits") == 9
    b = bal(rpc, u)
    assert b["available"] == 10 and b["held"] == 90
    assert sdb.execute("select min(credits_remaining) from public.studio_credit_lots").fetchone()[0] >= 0


def test_expired_promotional_credits_cannot_be_spent(sdb):
    rpc, u = pg_rpc(sdb), make_user(sdb)
    p = project(sdb, u)
    grant(rpc, u, 100, p_expires_at="2020-01-01T00:00:00Z")
    assert bal(rpc, u)["available"] == 0
    assert job(rpc, u, p, 10)["error"] == "insufficient_credits"
