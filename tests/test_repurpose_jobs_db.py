"""
Behaviour tests for repurpose_jobs_schema.sql, run against a real
Postgres (not mocks): the guarantees that matter here — one charge per
job, one refund per failure, one owner per job — are properties of the
SQL functions and their locking, which only a real database exercises.

Needs TEST_DATABASE_URL (default: local postgres/postgres@viyo_test).
The test database is reset from tests/sql/supabase_stub.sql plus the
real migration file at the start of the session.
"""
import os
import threading
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.types.json import Jsonb  # noqa: E402

from conftest import connect as _connect, make_user  # noqa: E402

COST = 40
DAILY_LIMIT = 5


def balance(db, uid):
    return db.execute("select points_balance from profiles where id=%s", (uid,)).fetchone()[0]


def ledger(db, uid):
    return db.execute(
        "select amount, type from transactions where user_id=%s order by id", (uid,)
    ).fetchall()


def create_job(db, uid, crid=None, kind="legacy_public_url", path=None, bucket=None, cost=COST,
               daily_limit=DAILY_LIMIT, max_attempts=3):
    return db.execute(
        "select repurpose_create_job(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (uid, cost, daily_limit, max_attempts, kind, bucket, path,
         "https://x.supabase.co/v.mp4" if kind == "legacy_public_url" else None, None, crid),
    ).fetchone()[0]


def claim(db, worker="w1", stale=120):
    return db.execute("select repurpose_claim_job(%s,%s)", (worker, stale)).fetchone()[0]


def job(db, job_id):
    return db.execute(
        "select status, attempts, coins_charged, refunded_at, refund_amount, lease_token, "
        "next_attempt_at > now() as delayed, error_code from repurpose_jobs where id=%s",
        (job_id,),
    ).fetchone()


# ---------------------------------------------------------------- charging

def test_create_charges_once_and_logs_ledger(db):
    uid = make_user(db, 100)
    r = create_job(db, uid, crid="tap-1")
    assert r["ok"] and r["created"] and r["coins_charged"] == COST
    assert balance(db, uid) == 60
    assert ledger(db, uid) == [(-COST, "repurpose")]


def test_retried_request_with_same_client_id_is_not_charged_again(db):
    uid = make_user(db, 100)
    first = create_job(db, uid, crid="tap-1")
    second = create_job(db, uid, crid="tap-1")
    assert second["ok"] and not second["created"]
    assert second["job_id"] == first["job_id"]
    assert balance(db, uid) == 60
    assert len(ledger(db, uid)) == 1


def test_concurrent_duplicate_requests_charge_once(db):
    uid = make_user(db, 1000)
    results = []

    def worker():
        with _connect() as conn:
            results.append(create_job(conn, uid, crid="double-tap"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r["ok"] for r in results)
    assert len({r["job_id"] for r in results}) == 1
    assert sum(1 for r in results if r["created"]) == 1
    assert balance(db, uid) == 1000 - COST
    assert len(ledger(db, uid)) == 1


def test_insufficient_coins_creates_nothing(db):
    uid = make_user(db, 39)
    r = create_job(db, uid, crid="x")
    assert r == {"ok": False, "error": "insufficient_coins", "balance": 39, "needed": COST}
    assert balance(db, uid) == 39
    assert db.execute("select count(*) from repurpose_jobs").fetchone()[0] == 0


def test_daily_limit_is_enforced_in_the_database(db):
    uid = make_user(db, 10_000)
    for i in range(DAILY_LIMIT):
        assert create_job(db, uid, crid=f"t{i}")["ok"]
    r = create_job(db, uid, crid="one-too-many")
    assert r["ok"] is False and r["error"] == "daily_limit"
    assert balance(db, uid) == 10_000 - DAILY_LIMIT * COST


def test_private_upload_must_belong_to_the_caller_and_be_unused(db):
    owner, other = make_user(db, 500), make_user(db, 500)
    db.execute(
        "insert into repurpose_uploads (source_path, bucket, user_id) values (%s,%s,%s)",
        (f"{owner}/a.mp4", "repurpose-sources", owner),
    )
    stolen = create_job(db, other, crid="s", kind="private_upload",
                        path=f"{owner}/a.mp4", bucket="repurpose-sources")
    assert stolen == {"ok": False, "error": "unknown_upload"}
    assert balance(db, other) == 500

    ok = create_job(db, owner, crid="o1", kind="private_upload",
                    path=f"{owner}/a.mp4", bucket="repurpose-sources")
    assert ok["ok"]
    linked = db.execute("select job_id from repurpose_uploads where source_path=%s",
                        (f"{owner}/a.mp4",)).fetchone()[0]
    assert str(linked) == ok["job_id"]

    reuse = create_job(db, owner, crid="o2", kind="private_upload",
                       path=f"{owner}/a.mp4", bucket="repurpose-sources")
    assert reuse == {"ok": False, "error": "upload_already_used"}
    assert balance(db, owner) == 500 - COST


def test_missing_profile_is_reported_not_crashed(db):
    r = create_job(db, uuid.uuid4(), crid="x")
    assert r == {"ok": False, "error": "profile_not_found"}


# ---------------------------------------------------------------- claiming

def test_concurrent_workers_never_claim_the_same_job(db):
    job_ids = set()
    for _ in range(20):
        job_ids.add(create_job(db, make_user(db))["job_id"])

    claimed = []
    lock = threading.Lock()

    def worker(n):
        with _connect() as conn:
            while True:
                j = claim(conn, worker=f"w{n}")
                if j is None:
                    return
                with lock:
                    claimed.append(j["id"])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(claimed) == sorted(job_ids)
    assert len(claimed) == len(set(claimed))


def test_one_running_job_per_creator_at_a_time(db):
    busy, waiting = make_user(db, 500), make_user(db, 500)
    a = create_job(db, busy, crid="a")["job_id"]
    b = create_job(db, busy, crid="b")["job_id"]
    c = create_job(db, waiting, crid="c")["job_id"]

    first = claim(db)
    assert first["id"] == a
    second = claim(db)
    assert second["id"] == c, "the other creator's job must not wait behind busy's queue"
    assert claim(db) is None, "busy's second job waits while the first is running"

    db.execute("select repurpose_complete_job(%s,%s,%s,30)", (a, first["lease_token"], Jsonb({})))
    assert claim(db)["id"] == b


def test_heartbeat_and_completion_require_the_current_lease(db):
    uid = make_user(db)
    jid = create_job(db, uid)["job_id"]
    j = claim(db)
    wrong = str(uuid.uuid4())
    assert db.execute("select repurpose_heartbeat(%s,%s,'x',0.5,null,null)", (jid, wrong)).fetchone()[0] is False
    assert db.execute(
        "select repurpose_heartbeat(%s,%s,'transcribing',0.3,'Transcribing',%s)",
        (jid, j["lease_token"], Jsonb({"transcript": {"sha256": "abc"}})),
    ).fetchone()[0] is True
    row = db.execute("select stage, progress, checkpoint from repurpose_jobs where id=%s", (jid,)).fetchone()
    assert row[0] == "transcribing" and abs(row[1] - 0.3) < 1e-6
    assert row[2] == {"transcript": {"sha256": "abc"}}
    assert db.execute("select repurpose_complete_job(%s,%s,%s,30)", (jid, wrong, Jsonb({}))).fetchone()[0] is False


def test_stalled_worker_loses_its_lease_to_a_new_worker(db):
    uid = make_user(db)
    jid = create_job(db, uid)["job_id"]
    old = claim(db, worker="old")
    db.execute("update repurpose_jobs set heartbeat_at = now() - interval '10 minutes' where id=%s", (jid,))

    new = claim(db, worker="new")
    assert new["id"] == jid and new["attempts"] == 2
    assert new["lease_token"] != old["lease_token"]
    # The stalled worker wakes up and tries to finish: refused.
    assert db.execute("select repurpose_heartbeat(%s,%s)", (jid, old["lease_token"])).fetchone()[0] is False
    assert db.execute("select repurpose_complete_job(%s,%s,%s,30)", (jid, old["lease_token"], Jsonb({}))).fetchone()[0] is False
    assert db.execute("select repurpose_complete_job(%s,%s,%s,30)", (jid, new["lease_token"], Jsonb({"ok": 1}))).fetchone()[0] is True


# ---------------------------------------------------------------- failure, retry, refund

def fail(db, jid, lease, retryable, code="x"):
    return db.execute(
        "select repurpose_fail_job(%s,%s,%s,'user msg','internal',%s,60,30)",
        (jid, lease, code, retryable),
    ).fetchone()[0]


def test_retryable_failure_requeues_with_delay_and_no_refund(db):
    uid = make_user(db, 100)
    jid = create_job(db, uid)["job_id"]
    j = claim(db)
    r = fail(db, jid, j["lease_token"], retryable=True)
    assert r == {"owned": True, "final": False, "refunded": 0}
    status, attempts, *_rest = job(db, jid)
    assert status == "queued" and attempts == 1
    assert job(db, jid)[6] is True, "next attempt is delayed"
    assert claim(db) is None, "not runnable until the delay passes"
    assert balance(db, uid) == 60


def test_permanent_failure_refunds_exactly_once(db):
    uid = make_user(db, 100)
    jid = create_job(db, uid)["job_id"]
    j = claim(db)
    r = fail(db, jid, j["lease_token"], retryable=False, code="no_audio")
    assert r == {"owned": True, "final": True, "refunded": COST}
    assert balance(db, uid) == 100
    # Any later refund attempt (sweep, manual, duplicate call) is a no-op.
    assert db.execute("select repurpose_refund_job(%s)", (jid,)).fetchone()[0] == 0
    assert db.execute("select repurpose_sweep(120,30)").fetchone()[0]["refunded"] == 0
    assert balance(db, uid) == 100
    assert ledger(db, uid) == [(-COST, "repurpose"), (COST, "refund_repurpose")]


def test_retryable_failure_on_last_attempt_is_final_and_refunded(db):
    uid = make_user(db, 100)
    jid = create_job(db, uid, max_attempts=2)["job_id"]
    j = claim(db)
    assert fail(db, jid, j["lease_token"], retryable=True)["final"] is False
    db.execute("update repurpose_jobs set next_attempt_at = now() where id=%s", (jid,))
    j = claim(db)
    assert j["attempts"] == 2
    r = fail(db, jid, j["lease_token"], retryable=True)
    assert r["final"] is True and r["refunded"] == COST
    assert balance(db, uid) == 100


def test_jobs_that_keep_crashing_are_failed_and_refunded_by_the_sweep(db):
    uid = make_user(db, 100)
    jid = create_job(db, uid, max_attempts=2)["job_id"]
    for _ in range(2):
        assert claim(db)["id"] == jid
        db.execute("update repurpose_jobs set heartbeat_at = now() - interval '1 hour' where id=%s", (jid,))
    assert claim(db) is None, "exhausted jobs are never handed out again"
    swept = db.execute("select repurpose_sweep(120,30)").fetchone()[0]
    assert swept == {"failed": 1, "refunded": 1}
    status, _a, _c, refunded_at, amount, *_ = job(db, jid)
    assert status == "failed" and refunded_at is not None and amount == COST
    assert balance(db, uid) == 100


def test_free_job_failure_moves_no_coins(db):
    uid = make_user(db, 0)
    jid = create_job(db, uid, cost=0)["job_id"]
    j = claim(db)
    assert fail(db, jid, j["lease_token"], retryable=False)["refunded"] == 0
    assert balance(db, uid) == 0 and ledger(db, uid) == []


def test_refund_that_cannot_land_is_retried_later_not_lost(db):
    uid = make_user(db, 100)
    jid = create_job(db, uid)["job_id"]
    j = claim(db)
    # Simulate the profile row being unavailable at refund time.
    db.execute("alter table profiles rename to profiles_away")
    db.execute("create table profiles (id uuid primary key, points_balance int)")
    try:
        with pytest.raises(psycopg.errors.RaiseException):
            fail(db, jid, j["lease_token"], retryable=False)
    finally:
        db.execute("drop table profiles")
        db.execute("alter table profiles_away rename to profiles")
    # The whole failure transaction rolled back; the job is still owned.
    assert job(db, jid)[0] == "processing"
    assert fail(db, jid, j["lease_token"], retryable=False)["refunded"] == COST
    assert balance(db, uid) == 100


def test_sweep_refunds_a_failed_job_whose_refund_never_ran(db):
    uid = make_user(db, 100)
    jid = create_job(db, uid)["job_id"]
    db.execute("update repurpose_jobs set status='failed' where id=%s", (jid,))
    assert db.execute("select repurpose_sweep(120,30)").fetchone()[0]["refunded"] == 1
    assert balance(db, uid) == 100


def test_graceful_release_does_not_count_the_attempt(db):
    uid = make_user(db)
    jid = create_job(db, uid)["job_id"]
    j = claim(db)
    assert db.execute("select repurpose_release_job(%s,%s)", (jid, j["lease_token"])).fetchone()[0]
    status, attempts, *_ = job(db, jid)
    assert status == "queued" and attempts == 0
    assert claim(db)["attempts"] == 1


# ---------------------------------------------------------------- security

@pytest.mark.parametrize("role", ["anon", "authenticated"])
def test_clients_cannot_call_job_functions(db, role):
    uid = make_user(db, 100)
    db.execute(f"set role {role}")
    try:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            db.execute(
                "select repurpose_create_job(%s,0,5,3,'legacy_public_url',null,null,'u',null,'c')",
                (uid,),
            )
    finally:
        db.execute("reset role")


def test_service_role_can_call_job_functions(db):
    uid = make_user(db, 100)
    db.execute("set role service_role")
    try:
        r = db.execute(
            "select repurpose_create_job(%s,40,5,3,'legacy_public_url',null,null,'u',null,'c')",
            (uid,),
        ).fetchone()[0]
    finally:
        db.execute("reset role")
    assert r["ok"]
