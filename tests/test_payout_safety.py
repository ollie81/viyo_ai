"""Creator withdrawals made safe for real money (creator_payouts_safety.sql + wallet.py): atomic requests, no double
reservation under concurrency, atomic and idempotent admin actions, rejected earnings returned exactly once, no double
payment. Real Postgres; no payout provider, no money."""
import sys
import threading
import uuid

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import wallet
from conftest import ROOT, connect, make_user

MIN = wallet.MIN_WITHDRAWAL_COINS          # 1,667 coins = $10
RATE = wallet.COIN_USD


@pytest.fixture(scope="module")
def schema(migrated):
    conn = connect()
    conn.execute((ROOT / "creator_payouts_schema.sql").read_text())
    conn.execute((ROOT / "creator_payouts_safety.sql").read_text())
    conn.execute((ROOT / "creator_payouts_safety.sql").read_text())       # safe to run twice
    conn.close()


@pytest.fixture
def pg(schema, db):
    clean = "truncate public.creator_earnings, public.withdrawal_requests, public.creator_payout_settings cascade"
    db.execute(clean)
    yield db
    db.execute(clean)        # leave nothing behind for other test files that expect these tables empty


def earn(db, creator, coins, status="available"):
    return db.execute("insert into public.creator_earnings (creator_id, coins, status) values (%s, %s, %s) returning id",
                      (creator, coins, status)).fetchone()[0]


def request(db, creator, method="paypal", dest="PayPal a@b.co"):
    return db.execute("select public.viyo_request_withdrawal(%s, %s, %s, %s, %s)",
                      (creator, MIN, RATE, method, dest)).fetchone()[0]


def settle(db, req_id, action, note=None):
    return db.execute("select public.viyo_settle_withdrawal(%s, %s, %s)", (req_id, action, note)).fetchone()[0]


def statuses(db, creator):
    return dict(db.execute("select status, coalesce(sum(coins), 0) from public.creator_earnings where creator_id = %s "
                           "group by status", (creator,)).fetchall())


# ------------------------------------------------------------------------------------------- requesting

def test_a_request_reserves_all_available_earnings_atomically(pg):
    c = make_user(pg)
    earn(pg, c, 1000)
    earn(pg, c, 800)
    r = request(pg, c)
    assert r["ok"] and r["coins"] == 1800 and float(r["usd_amount"]) == round(1800 * RATE, 2)
    assert statuses(pg, c) == {"pending_withdrawal": 1800}
    linked = pg.execute("select count(*) from public.creator_earnings where withdrawal_request_id = %s",
                        (r["request_id"],)).fetchone()[0]
    assert linked == 2
    row = pg.execute("select status, coins, payout_method, payout_destination from public.withdrawal_requests "
                     "where id = %s", (r["request_id"],)).fetchone()
    assert row == ("pending", 1800, "paypal", "PayPal a@b.co")


def test_below_the_minimum_nothing_is_reserved(pg):
    c = make_user(pg)
    earn(pg, c, MIN - 1)
    r = request(pg, c)
    assert r == {"ok": False, "error": "below_minimum", "coins": MIN - 1}
    assert statuses(pg, c) == {"available": MIN - 1}
    assert pg.execute("select count(*) from public.withdrawal_requests").fetchone()[0] == 0


def test_only_one_pending_request_per_creator(pg):
    c = make_user(pg)
    earn(pg, c, 2000)
    assert request(pg, c)["ok"]
    earn(pg, c, 2000)                                  # new earnings arrive while the first is pending
    assert request(pg, c) == {"ok": False, "error": "pending_exists"}
    assert statuses(pg, c) == {"pending_withdrawal": 2000, "available": 2000}
    # The database itself refuses a second pending row, even bypassing the function.
    with pytest.raises(psycopg.errors.UniqueViolation):
        pg.execute("insert into public.withdrawal_requests (creator_id, coins, status) values (%s, 5, 'pending')", (c,))


def test_simultaneous_requests_never_reserve_the_same_earnings(pg):
    c = make_user(pg)
    for _ in range(5):
        earn(pg, c, 500)
    results, barrier = [], threading.Barrier(6)

    def go():
        conn = connect()
        barrier.wait()
        try:
            results.append(conn.execute("select public.viyo_request_withdrawal(%s, %s, %s, 'paypal', 'x')",
                                        (c, MIN, RATE)).fetchone()[0])
        finally:
            conn.close()
    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = [r for r in results if r["ok"]]
    assert len(ok) == 1 and ok[0]["coins"] == 2500
    assert sorted(r.get("error") for r in results if not r["ok"]) == ["pending_exists"] * 5
    assert pg.execute("select count(*) from public.withdrawal_requests where creator_id = %s", (c,)).fetchone()[0] == 1
    assert statuses(pg, c) == {"pending_withdrawal": 2500}


def test_a_failure_inside_the_request_leaves_nothing_reserved_and_no_request(pg):
    c = make_user(pg)
    earn(pg, c, 2000)
    pg.execute("""create or replace function public.test_fail_reserve() returns trigger language plpgsql as $$
                  begin raise exception 'disk full'; end $$""")
    pg.execute("create trigger test_fail_reserve before update on public.creator_earnings "
               "for each row execute function public.test_fail_reserve()")
    try:
        with pytest.raises(psycopg.errors.RaiseException):
            request(pg, c)
    finally:
        pg.execute("drop trigger test_fail_reserve on public.creator_earnings")
    assert pg.execute("select count(*) from public.withdrawal_requests").fetchone()[0] == 0
    assert statuses(pg, c) == {"available": 2000}
    assert request(pg, c)["ok"]                       # and it works once the failure is gone


# ------------------------------------------------------------------------------------------- admin actions

def test_paying_marks_request_and_earnings_paid_and_can_never_pay_twice(pg):
    c = make_user(pg)
    earn(pg, c, 2000)
    rid = request(pg, c)["request_id"]
    first = settle(pg, rid, "paid", "sent via PayPal")
    assert first["ok"] and first["changed"] and first["status"] == "paid"
    again = settle(pg, rid, "paid")
    assert again["ok"] and again["changed"] is False           # a double tap changes nothing
    assert settle(pg, rid, "rejected") == {"ok": False, "error": "already_paid"}
    assert statuses(pg, c) == {"paid": 2000}
    paid_at, note = pg.execute("select paid_at, admin_note from public.withdrawal_requests where id = %s", (rid,)).fetchone()
    assert paid_at is not None and note == "sent via PayPal"   # the first action's note is kept


def test_rejecting_returns_earnings_exactly_once(pg):
    c = make_user(pg)
    earn(pg, c, 2000)
    rid = request(pg, c)["request_id"]
    assert settle(pg, rid, "rejected", "wrong number")["changed"] is True
    assert statuses(pg, c) == {"available": 2000}
    assert settle(pg, rid, "rejected")["changed"] is False     # repeated: nothing returned a second time
    assert settle(pg, rid, "paid") == {"ok": False, "error": "already_rejected"}
    assert statuses(pg, c) == {"available": 2000}
    assert pg.execute("select count(*) from public.creator_earnings where withdrawal_request_id is not null").fetchone()[0] == 0
    # The returned earnings can be requested again, once.
    again = request(pg, c)
    assert again["ok"] and again["coins"] == 2000


def test_simultaneous_paid_and_reject_leave_one_consistent_outcome(pg):
    c = make_user(pg)
    earn(pg, c, 2000)
    rid = request(pg, c)["request_id"]
    out, barrier = {}, threading.Barrier(2)

    def go(action):
        conn = connect()
        barrier.wait()
        try:
            out[action] = conn.execute("select public.viyo_settle_withdrawal(%s, %s, null)", (rid, action)).fetchone()[0]
        finally:
            conn.close()
    ts = [threading.Thread(target=go, args=(a,)) for a in ("paid", "rejected")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    winners = [a for a, r in out.items() if r["ok"]]
    assert len(winners) == 1
    final = pg.execute("select status from public.withdrawal_requests where id = %s", (rid,)).fetchone()[0]
    assert final == winners[0]
    assert statuses(pg, c) == ({"paid": 2000} if final == "paid" else {"available": 2000})


def test_a_request_whose_earnings_dont_add_up_is_not_settled(pg):
    """An old request from before this fix, whose earnings were never reserved: paying it could pay twice."""
    c = make_user(pg)
    earn(pg, c, 2000)                                     # still available, not linked
    rid = pg.execute("insert into public.withdrawal_requests (creator_id, coins, status) values (%s, 2000, 'pending') "
                     "returning id", (c,)).fetchone()[0]
    r = settle(pg, rid, "paid")
    assert r["ok"] is False and r["error"] == "earnings_mismatch" and r["reserved_coins"] == 0
    assert pg.execute("select status from public.withdrawal_requests where id = %s", (rid,)).fetchone()[0] == "pending"
    assert statuses(pg, c) == {"available": 2000}


def test_unknown_requests_and_bad_states_are_refused(pg):
    assert settle(pg, str(uuid.uuid4()), "paid") == {"ok": False, "error": "not_found"}
    with pytest.raises(psycopg.errors.RaiseException):
        settle(pg, str(uuid.uuid4()), "refund")
    c = make_user(pg)
    with pytest.raises(psycopg.errors.CheckViolation):
        earn(pg, c, 10, status="spent")
    with pytest.raises(psycopg.errors.CheckViolation):    # reserved earnings must belong to a withdrawal
        earn(pg, c, 10, status="pending_withdrawal")


def test_only_the_server_can_call_the_payout_functions(pg):
    c = make_user(pg)
    for role in ("anon", "authenticated"):
        pg.execute("begin")
        try:
            pg.execute(f"set local role {role}")
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                pg.execute("select public.viyo_request_withdrawal(%s, 1, 0.006, 'paypal', 'x')", (c,))
        finally:
            pg.execute("rollback")


# ------------------------------------------------------------------------------------------- the API on top

class _Res:
    def __init__(self, data):
        self.data = data


class _Q:
    """Just enough of the Supabase query builder for wallet._load_settings / _pending_request."""
    def __init__(self, conn, table):
        self.conn, self.table, self.filters, self.order_by, self.n = conn, table, [], None, None

    def select(self, *_a):
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def order(self, col, desc=False):
        self.order_by = f"{col} {'desc' if desc else 'asc'}"
        return self

    def limit(self, n):
        self.n = n
        return self

    def execute(self):
        where = " and ".join(f"{k} = %s" for k, _ in self.filters) or "true"
        sql = f"select * from public.{self.table} where {where}"
        if self.order_by:
            sql += f" order by {self.order_by}"
        if self.n:
            sql += f" limit {int(self.n)}"
        cur = self.conn.execute(sql, [v for _, v in self.filters])
        cols = [c.name for c in cur.description]
        return _Res([{c: (str(v) if c.endswith("id") and v is not None else v) for c, v in zip(cols, row)}
                     for row in cur.fetchall()])


class PgAdmin:
    def __init__(self, conn, fail=None):
        self.conn, self.fail = conn, fail or {}

    def table(self, name):
        return _Q(self.conn, name)

    def rpc(self, name, params):
        conn, fail = self.conn, self.fail

        class Call:
            def execute(self_inner):
                if name in fail:
                    raise fail[name]
                args = ", ".join(f"{k} => %({k})s" for k in params)
                return _Res(conn.execute(f"select public.{name}({args})", params).fetchone()[0])
        return Call()


@pytest.fixture
def api(pg, monkeypatch):
    admin = PgAdmin(pg)
    monkeypatch.setattr(wallet, "supabase_admin", admin)
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    monkeypatch.setattr(wallet, "ADMIN_API_KEY", "admin-secret", raising=False)
    sent = []
    import notify as notify_mod
    monkeypatch.setattr(notify_mod, "notify", lambda *a, **k: sent.append((a, k)))
    app = FastAPI()
    app.include_router(wallet.router)

    async def creator_user(authorization: str = wallet.Header(None)):
        return authorization.removeprefix("Bearer ")
    app.dependency_overrides[wallet._get_current_user_id_no_guest] = creator_user
    return TestClient(app), pg, admin, sent


def H(u):
    return {"Authorization": f"Bearer {u}"}


A = {"X-Admin-Key": "admin-secret"}


def with_settings(db, u):
    db.execute("insert into public.creator_payout_settings (user_id, method, momo_phone, momo_network, account_name) "
               "values (%s, 'mobile_money', '+250788000000', 'MTN', 'Amani')", (u,))


def test_api_withdraw_then_pay_notifies_once(api):
    client, db, admin, sent = api
    u = make_user(db)
    with_settings(db, u)
    earn(db, u, 2000)
    r = client.post("/api/v1/wallet/withdraw", headers=H(u))
    assert r.status_code == 200 and r.json()["coins"] == 2000 and r.json()["usd_amount"] == 12.0
    assert client.post("/api/v1/wallet/withdraw", headers=H(u)).status_code == 409
    rid = db.execute("select id from public.withdrawal_requests").fetchone()[0]
    first = client.post(f"/api/v1/admin/withdrawals/{rid}/paid", headers=A, json={"note": "sent"})
    again = client.post(f"/api/v1/admin/withdrawals/{rid}/paid", headers=A, json={})
    assert first.json()["changed"] is True and again.status_code == 200 and again.json()["changed"] is False
    assert len(sent) == 1 and sent[0][0][1] == "payout_paid"
    assert client.post(f"/api/v1/admin/withdrawals/{rid}/reject", headers=A, json={}).status_code == 409


def test_api_reject_returns_earnings_and_allows_a_new_request(api):
    client, db, admin, sent = api
    u = make_user(db)
    with_settings(db, u)
    earn(db, u, 2000)
    client.post("/api/v1/wallet/withdraw", headers=H(u))
    rid = db.execute("select id from public.withdrawal_requests").fetchone()[0]
    assert client.post(f"/api/v1/admin/withdrawals/{rid}/reject", headers=A, json={"note": "bad number"}).status_code == 200
    assert client.post(f"/api/v1/admin/withdrawals/{rid}/reject", headers=A, json={}).json()["changed"] is False
    assert [s[0][1] for s in sent] == ["payout_rejected"]
    assert client.post("/api/v1/wallet/withdraw", headers=H(u)).status_code == 200


def test_api_messages_for_minimum_missing_details_and_unknown_requests(api):
    client, db, admin, sent = api
    u = make_user(db)
    assert "payout details" in client.post("/api/v1/wallet/withdraw", headers=H(u)).json()["detail"]
    with_settings(db, u)
    earn(db, u, 100)
    r = client.post("/api/v1/wallet/withdraw", headers=H(u))
    assert r.status_code == 400 and "$10" in r.json()["detail"] and "100 coins" in r.json()["detail"]
    assert client.post(f"/api/v1/admin/withdrawals/{uuid.uuid4()}/paid", headers=A, json={}).status_code == 404
    assert client.post("/api/v1/admin/withdrawals/not-a-uuid/paid", headers=A, json={}).status_code == 404
    assert client.post(f"/api/v1/admin/withdrawals/{uuid.uuid4()}/paid", json={}).status_code == 403


def test_api_a_failed_database_call_changes_nothing_and_a_missing_function_pauses_withdrawals(api, monkeypatch):
    client, db, admin, sent = api
    u = make_user(db)
    with_settings(db, u)
    earn(db, u, 2000)
    admin.fail = {"viyo_request_withdrawal": Exception("connection reset")}
    r = client.post("/api/v1/wallet/withdraw", headers=H(u))
    assert r.status_code == 500 and "Nothing was changed" in r.json()["detail"]
    admin.fail = {"viyo_request_withdrawal": Exception("PGRST202 Could not find the function viyo_request_withdrawal")}
    r = client.post("/api/v1/wallet/withdraw", headers=H(u))
    assert r.status_code == 503                         # no fallback to the old non-atomic steps
    assert statuses(db, u) == {"available": 2000}
    assert db.execute("select count(*) from public.withdrawal_requests").fetchone()[0] == 0


def test_api_rules_are_unchanged():
    assert wallet.MIN_WITHDRAWAL_USD == 10.0 and wallet.MIN_WITHDRAWAL_COINS == 1667 and wallet.COIN_USD == 0.006
    import episodes
    assert episodes.CREATOR_SHARE == 0.65
    assert wallet.PayoutSettings.model_fields["method"].annotation.__args__ == ("paypal", "mobile_money")


# ------------------------------------------------------------------------------------------- earnings: bought coins only

class _EarnQ:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.row = db, table, "select", None

    def select(self, *_a):
        return self

    def insert(self, row):
        self.op, self.row = "insert", row
        return self

    def eq(self, *_a):
        return self

    def single(self):
        return self

    def execute(self):
        if self.table == "profiles":
            return _Res(self.db.profile)
        if self.op == "insert":
            self.db.rows.setdefault(self.table, []).append(self.row)
        return _Res([])


class EarnDB:
    """The viewer's balance as the unlock code reads it just before the debit."""
    def __init__(self, balance, paid):
        self.profile = {"points_balance": balance, "paid_coins": paid}
        self.rows = {}

    def table(self, name):
        return _EarnQ(self, name)


@pytest.mark.parametrize("balance,paid,price,expected", [
    (100, 0, 20, None),        # only free coins (check-ins, missions, ads, gifts): the creator earns nothing
    (100, 10, 20, None),       # 90 free coins are spent first: none of this spend was bought
    (100, 100, 20, 13),        # all bought: 65% of 20 = 13 coins
    (30, 25, 20, 10),          # 5 free first, then 15 bought: 65% of 15 = 9.75 -> 10
])
def test_creators_earn_only_from_bought_coins(monkeypatch, balance, paid, price, expected):
    import asyncio

    import episodes
    db = EarnDB(balance, paid)
    creator, viewer, post = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    monkeypatch.setattr(episodes, "supabase_admin", db)
    monkeypatch.setattr(episodes, "is_active_subscriber", lambda *_: False)
    monkeypatch.setattr(episodes, "_get_episode", lambda pid: {"id": pid, "user_id": creator, "series_id": "s",
                                                              "episode_number": 9, "is_free": None})
    monkeypatch.setattr(episodes, "_get_series", lambda sid: {"id": sid, "content_type": "short_drama", "title": "T",
                                                            "coin_price_per_episode": price})
    monkeypatch.setattr(episodes, "debit_coins", lambda *a, **k: None)
    monkeypatch.setattr(episodes, "_notify_unlock", lambda *a, **k: None)
    asyncio.run(episodes.unlock_episode(post, user_id=viewer))
    earned = db.rows.get("creator_earnings") or []
    assert [e["coins"] for e in earned] == ([] if expected is None else [expected])
