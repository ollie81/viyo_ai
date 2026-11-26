"""nardopay_orders.sql on a real Postgres: runs twice, permissions (RLS), ownership, unique transactions, valid
statuses, and the rule that an order can't be crediting or credited without a verified payment. No NardoPay call,
no money."""
import hashlib
import uuid

import psycopg
import pytest

from conftest import ROOT, connect


@pytest.fixture(scope="module")
def schema(migrated):
    conn = connect()
    sql = (ROOT / "nardopay_orders.sql").read_text()
    conn.execute(sql)
    conn.execute(sql)                       # safe to run a second time
    conn.close()


@pytest.fixture
def db(schema):
    conn = connect()
    conn.execute("truncate public.nardopay_orders, public.nardopay_webhook_events, public.nardopay_rates cascade")
    yield conn
    conn.close()


def user(db):
    uid = uuid.uuid4()
    db.execute("insert into auth.users (id) values (%s)", (uid,))
    return uid


def rate(db, value=1410):
    return db.execute("insert into public.nardopay_rates (rate, set_by) values (%s, 'owner') returning id",
                      (value,)).fetchone()[0]


def order(db, uid, **over):
    row = {"user_id": uid, "package_id": "starter", "coins": 100, "usd_cents": 99, "rate_id": rate(db),
           "usd_rwf_rate": 1410, "rwf_amount": 1396, **over}
    cols = ", ".join(row)
    return db.execute(f"insert into public.nardopay_orders ({cols}) values ({', '.join(['%s'] * len(row))}) returning id",
                      list(row.values())).fetchone()[0]


def event(db, body=b'{"event":"payment.completed"}', **over):
    row = {"body_sha256": hashlib.sha256(body).hexdigest(), "signature_ok": True, "raw_body": body.decode(), **over}
    cols = ", ".join(row)
    return db.execute(f"insert into public.nardopay_webhook_events ({cols}) values ({', '.join(['%s'] * len(row))}) "
                      "returning id", list(row.values())).fetchone()[0]


def as_role(db, role, uid, sql, params=()):
    """Runs one statement as a client would through Supabase (anon or authenticated, with its user id)."""
    db.execute("begin")
    try:
        db.execute(f"set local role {role}")
        if uid:
            db.execute("select set_config('request.jwt.claim.sub', %s, true)", (str(uid),))
        rows = db.execute(sql, params).fetchall() if sql.lstrip().lower().startswith("select") else db.execute(sql, params)
        db.execute("commit")
        return rows
    except Exception:
        db.execute("rollback")
        raise


# ------------------------------------------------------------------------------------------- the migration

def test_the_migration_creates_three_tables_with_row_level_security(db):
    tables = {r[0]: r[1] for r in db.execute(
        "select relname, relrowsecurity from pg_class where relkind = 'r' and relname like 'nardopay_%'").fetchall()}
    assert {"nardopay_rates", "nardopay_webhook_events", "nardopay_orders"} <= set(tables)
    # Later migrations (e.g. nardopay_rates_auto.sql's nardopay_rate_runs) may add tables: every one keeps RLS on.
    assert set(tables) <= {"nardopay_rates", "nardopay_webhook_events", "nardopay_orders", "nardopay_rate_runs",
                           "nardopay_subscriptions", "nardopay_subscription_payments"}
    assert all(tables.values())


def test_running_it_again_keeps_existing_data(schema, db):
    uid = user(db)
    oid = order(db, uid)
    db.execute((ROOT / "nardopay_orders.sql").read_text())
    assert db.execute("select count(*) from public.nardopay_orders where id = %s", (oid,)).fetchone()[0] == 1


# ------------------------------------------------------------------------------------------- permissions and ownership

def test_a_user_reads_only_their_own_orders(db):
    alice, bob = user(db), user(db)
    a = order(db, alice)
    order(db, bob)
    rows = as_role(db, "authenticated", alice, "select id from public.nardopay_orders")
    assert [str(r[0]) for r in rows] == [str(a)]


def test_anonymous_visitors_read_nothing(db):
    order(db, user(db))
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        as_role(db, "anon", None, "select id from public.nardopay_orders")


@pytest.mark.parametrize("sql", [
    "insert into public.nardopay_orders (user_id, package_id, coins, usd_cents, usd_rwf_rate, rwf_amount) "
    "values ('{uid}', 'creator', 999999, 1, 1, 1)",
    "update public.nardopay_orders set coins = 999999 where user_id = '{uid}'",
    "update public.nardopay_orders set status = 'credited' where user_id = '{uid}'",
    "delete from public.nardopay_orders where user_id = '{uid}'",
])
def test_signed_in_users_cannot_create_change_or_delete_orders(db, sql):
    uid = user(db)
    oid = order(db, uid)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        as_role(db, "authenticated", uid, sql.format(uid=uid))
    coins, status = db.execute("select coins, status from public.nardopay_orders where id = %s", (oid,)).fetchone()
    assert (coins, status) == (100, "pending")


@pytest.mark.parametrize("table", ["nardopay_rates", "nardopay_webhook_events"])
def test_rates_and_webhook_events_are_server_only(db, table):
    uid = user(db)
    rate(db)
    event(db)
    for role in ("authenticated", "anon"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            as_role(db, role, uid, f"select * from public.{table}")


def test_the_service_role_can_write_everything(db):
    uid = user(db)
    db.execute("begin")
    db.execute("set local role service_role")
    rid = db.execute("insert into public.nardopay_rates (rate, set_by) values (1400, 'owner') returning id").fetchone()[0]
    db.execute("insert into public.nardopay_orders (user_id, package_id, coins, usd_cents, rate_id, usd_rwf_rate, "
               "rwf_amount) values (%s, 'starter', 100, 99, %s, 1400, 1386)", (uid, rid))
    db.execute("commit")
    assert db.execute("select count(*) from public.nardopay_orders").fetchone()[0] == 1


# ------------------------------------------------------------------------------------------- integrity

def test_a_transaction_can_pay_only_one_order(db):
    uid = user(db)
    ev = event(db)
    order(db, uid, status="credited", transaction_id="tx_1", verified_event_id=ev,
          payment_verified_at="2026-11-24T10:00:00Z", credited_at="2026-11-24T10:00:01Z")
    with pytest.raises(psycopg.errors.UniqueViolation):
        order(db, uid, status="crediting", transaction_id="tx_1", verified_event_id=ev,
              payment_verified_at="2026-11-24T10:00:00Z")


def test_only_valid_statuses_and_rwf_are_accepted(db):
    uid = user(db)
    with pytest.raises(psycopg.errors.CheckViolation):
        order(db, uid, status="paid")
    with pytest.raises(psycopg.errors.CheckViolation):
        order(db, uid, currency="USD")
    with pytest.raises(psycopg.errors.CheckViolation):
        order(db, uid, coins=0)
    with pytest.raises(psycopg.errors.CheckViolation):
        order(db, uid, rwf_amount=-5)
    for s in ("pending", "failed", "mismatch", "expired"):
        order(db, uid, status=s)


def test_an_order_cannot_be_crediting_or_credited_without_a_verified_payment(db):
    uid = user(db)
    with pytest.raises(psycopg.errors.CheckViolation):
        order(db, uid, status="crediting")                                   # no transaction, no verified event
    with pytest.raises(psycopg.errors.CheckViolation):
        order(db, uid, status="crediting", transaction_id="tx_2")            # no verified event
    oid = order(db, uid)
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("update public.nardopay_orders set status = 'credited' where id = %s", (oid,))
    ev = event(db)
    with pytest.raises(psycopg.errors.CheckViolation):                       # credited also needs credited_at
        db.execute("update public.nardopay_orders set status = 'credited', transaction_id = 'tx_3', "
                   "verified_event_id = %s, payment_verified_at = now() where id = %s", (ev, oid))
    db.execute("update public.nardopay_orders set status = 'crediting', transaction_id = 'tx_3', verified_event_id = %s, "
               "payment_verified_at = now() where id = %s", (ev, oid))
    db.execute("update public.nardopay_orders set status = 'credited', credited_at = now() where id = %s", (oid,))


def test_a_repeated_webhook_delivery_finds_the_existing_event(db):
    body = b'{"event":"payment.completed","n":1}'
    first = event(db, body)
    again = db.execute("insert into public.nardopay_webhook_events (body_sha256, signature_ok, raw_body) "
                       "values (%s, true, %s) on conflict (body_sha256) do nothing returning id",
                       (hashlib.sha256(body).hexdigest(), body.decode())).fetchall()
    assert again == []
    found = db.execute("select id from public.nardopay_webhook_events where body_sha256 = %s",
                       (hashlib.sha256(body).hexdigest(),)).fetchone()[0]
    assert found == first


def test_only_one_of_two_simultaneous_deliveries_claims_the_event(db):
    """The claim is a conditional update: the second one sees nothing to claim."""
    ev = event(db)
    claim = ("update public.nardopay_webhook_events set status = 'processing', locked_at = now(), "
             "attempts = attempts + 1 where id = %s and (status in ('received', 'error') or "
             "(status = 'processing' and locked_at < now() - interval '5 minutes')) returning id")
    other = connect()
    db.execute("begin")
    first = db.execute(claim, (ev,)).fetchall()
    other.execute("set lock_timeout = '200ms'")
    with pytest.raises(psycopg.errors.LockNotAvailable):          # the second waits on the first's row lock
        other.execute(claim, (ev,))
    db.execute("commit")
    second = other.execute(claim, (ev,)).fetchall()                # after the first commits: nothing left to claim
    other.close()
    assert len(first) == 1 and second == []


def test_provider_event_ids_are_unique_when_present_and_hashes_must_be_sha256(db):
    event(db, b"a", provider_event_id="evt_1")
    event(db, b"b")                                                # no provider id: allowed
    event(db, b"c")
    with pytest.raises(psycopg.errors.UniqueViolation):
        event(db, b"d", provider_event_id="evt_1")
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("insert into public.nardopay_webhook_events (body_sha256, signature_ok) values ('nothex', true)")


def test_a_purged_raw_body_stays_purged(db):
    ev = event(db)
    db.execute("update public.nardopay_webhook_events set raw_body = null, raw_purged_at = now() where id = %s", (ev,))
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("update public.nardopay_webhook_events set raw_body = 'x' where id = %s", (ev,))


def test_rates_must_be_positive_and_name_who_set_them(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("insert into public.nardopay_rates (rate, set_by) values (0, 'owner')")
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("insert into public.nardopay_rates (rate, set_by) values (1400, '')")
