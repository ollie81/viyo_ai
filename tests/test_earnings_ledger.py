"""Creator earnings ledger (creator_earnings_ledger.sql + wallet.py): each unlock credited once, held 14 days, then
withdrawable; reversals for refunds (cancel if unpaid, deduct if paid), once; failed credits kept and retried without
doubling. Real Postgres; no payout provider, no money."""
import datetime

import pytest

import wallet
from conftest import ROOT, connect, make_user
from test_payout_safety import A, H, MIN, PgAdmin, api, pg, request, schema, settle, with_settings  # noqa: F401


@pytest.fixture(scope="module")
def ledger(schema):
    conn = connect()
    for _ in range(2):                                               # safe to run again
        conn.execute((ROOT / "creator_earnings_ledger.sql").read_text())
    conn.close()


@pytest.fixture
def lg(ledger, pg):
    pg.execute("truncate public.creator_earning_failures")
    return pg


class _Insert:
    def __init__(self, conn, table):
        self.conn, self.table = conn, table

    def insert(self, row):
        cols = list(row)
        self.sql = (f"insert into public.{self.table} ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))})",
                    [row[c] for c in cols])
        return self

    def execute(self):
        self.conn.execute(*self.sql)


class LedgerAdmin(PgAdmin):
    """PgAdmin that can also insert rows (the failure log)."""
    def table(self, name):
        return _Insert(self.conn, name)


def rows(db, creator):
    cur = db.execute("select status, coins, kind, earning_key, available_at, reversal_of from public.creator_earnings "
                     "where creator_id = %s order by created_at", (creator,))
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def credit(db, creator, post, viewer, coins=13):
    return wallet.credit_earning(LedgerAdmin(db), creator, post, viewer, coins)


def age(db, creator, days=15):
    db.execute("update public.creator_earnings set available_at = now() - make_interval(days => %s) "
               "where creator_id = %s and status = 'pending'", (days - 14, creator))


def test_an_unlock_is_credited_once_and_held_for_14_days(lg):
    c, v, post = make_user(lg), make_user(lg), make_user(lg)       # any uuids
    assert credit(lg, c, post, v) is True
    assert credit(lg, c, post, v) is True                          # a retry: recorded, not doubled
    r = rows(lg, c)
    assert len(r) == 1 and r[0]["status"] == "pending" and r[0]["coins"] == 13 and r[0]["kind"] == "unlock"
    assert r[0]["earning_key"] == f"unlock:{v}:{post}"
    held = r[0]["available_at"] - datetime.datetime.now(datetime.timezone.utc)
    assert datetime.timedelta(days=13, hours=23) < held <= datetime.timedelta(days=14)


def test_held_earnings_cant_be_withdrawn_until_their_14_days_are_up(lg):
    c = make_user(lg)
    for i in range(MIN // 13 + 1):
        credit(lg, c, make_user(lg), make_user(lg))
    assert request(lg, c)["error"] == "below_minimum"               # all still pending
    age(lg, c, days=15)
    assert lg.execute("select public.viyo_release_earnings(%s)", (c,)).fetchone()[0] == MIN // 13 + 1
    assert request(lg, c)["ok"] is True


def test_reversing_an_unpaid_earning_cancels_it_once(lg):
    c, v, post = make_user(lg), make_user(lg), make_user(lg)
    credit(lg, c, post, v)
    eid = lg.execute("select id from public.creator_earnings where creator_id = %s", (c,)).fetchone()[0]
    r = lg.execute("select public.viyo_reverse_earning(%s, 'refunded')", (eid,)).fetchone()[0]
    assert r["ok"] and r["changed"] and r["status"] == "reversed"
    again = lg.execute("select public.viyo_reverse_earning(%s, 'refunded')", (eid,)).fetchone()[0]
    assert again["ok"] and again["changed"] is False
    assert [x["status"] for x in rows(lg, c)] == ["reversed"]


def test_reversing_a_paid_earning_deducts_it_from_the_next_withdrawal_once(lg):
    c = make_user(lg)
    for _ in range(MIN // 13 + 1):
        credit(lg, c, make_user(lg), make_user(lg))
    age(lg, c)
    lg.execute("select public.viyo_release_earnings(%s)", (c,))
    req = request(lg, c)
    settle(lg, req["request_id"], "paid")
    eid = lg.execute("select id from public.creator_earnings where creator_id = %s and status = 'paid' limit 1",
                     (c,)).fetchone()[0]
    r = lg.execute("select public.viyo_reverse_earning(%s, 'chargeback')", (eid,)).fetchone()[0]
    assert r["ok"] and r["status"] == "deducted"
    assert lg.execute("select public.viyo_reverse_earning(%s, 'chargeback')", (eid,)).fetchone()[0]["changed"] is False
    neg = [x for x in rows(lg, c) if x["kind"] == "reversal"]
    assert len(neg) == 1 and neg[0]["coins"] == -13 and neg[0]["status"] == "available" and neg[0]["reversal_of"] == eid
    # The next withdrawal nets it: 13 coins less.
    for _ in range(MIN // 13 + 2):
        credit(lg, c, make_user(lg), make_user(lg))
    age(lg, c)
    lg.execute("select public.viyo_release_earnings(%s)", (c,))
    req2 = request(lg, c)
    assert req2["ok"] and req2["coins"] == (MIN // 13 + 2) * 13 - 13


def test_an_earning_inside_a_pending_withdrawal_needs_the_withdrawal_rejected_first(lg):
    c = make_user(lg)
    for _ in range(MIN // 13 + 1):
        credit(lg, c, make_user(lg), make_user(lg))
    age(lg, c)
    lg.execute("select public.viyo_release_earnings(%s)", (c,))
    req = request(lg, c)
    eid = lg.execute("select id from public.creator_earnings where creator_id = %s limit 1", (c,)).fetchone()[0]
    assert lg.execute("select public.viyo_reverse_earning(%s, 'x')", (eid,)).fetchone()[0]["error"] == "in_withdrawal"
    settle(lg, req["request_id"], "rejected")
    assert lg.execute("select public.viyo_reverse_earning(%s, 'x')", (eid,)).fetchone()[0]["status"] == "reversed"


def test_a_failed_credit_is_kept_and_a_retry_never_doubles(lg):
    c, v, post = make_user(lg), make_user(lg), make_user(lg)
    broken = LedgerAdmin(lg, fail={"viyo_credit_earning": RuntimeError("database unavailable")})
    assert wallet.credit_earning(broken, c, post, v, 13) is False
    fails = lg.execute("select earning_key, resolved_at from public.creator_earning_failures").fetchall()
    assert fails == [(f"unlock:{v}:{post}", None)]
    assert rows(lg, c) == []
    assert credit(lg, c, post, v) is True                           # the retry with the same key
    assert credit(lg, c, post, v) is True                           # and again: still once
    assert len(rows(lg, c)) == 1
    assert lg.execute("select resolved_at is not null from public.creator_earning_failures").fetchone()[0] is True


def test_only_the_server_can_call_the_ledger_functions(lg):
    lg.execute("begin")
    try:
        lg.execute("set local role authenticated")
        for sql in ("select public.viyo_credit_earning(gen_random_uuid(), null, null, 1000, 'k', 0)",
                    "select public.viyo_release_earnings(null)",
                    "select public.viyo_reverse_earning(gen_random_uuid(), 'x')"):
            lg.execute("savepoint s")
            with pytest.raises(Exception):
                lg.execute(sql)
            lg.execute("rollback to savepoint s")
    finally:
        lg.execute("rollback")


# ------------------------------------------------------------------------------------------- the API

def test_the_wallet_shows_held_and_available_separately(api, ledger):
    client, db, admin, sent = api
    c = make_user(db)
    wallet.credit_earning(admin, c, make_user(db), make_user(db), 100)
    wallet.credit_earning(admin, c, make_user(db), make_user(db), 50)
    db.execute("update public.creator_earnings set available_at = now() - interval '1 minute' "
               "where creator_id = %s and coins = 50", (c,))
    e = client.get("/api/v1/wallet/earnings", headers=H(c)).json()
    assert e["held_coins"] == 100 and e["available_coins"] == 50 and e["hold_days"] == 14
    assert e["next_available_at"] and e["lifetime_coins"] == 150


def test_admin_reversal_endpoint_is_admin_only_and_explains_refusals(api, ledger):
    client, db, admin, sent = api
    c = make_user(db)
    wallet.credit_earning(admin, c, make_user(db), make_user(db), 13)
    eid = str(db.execute("select id from public.creator_earnings where creator_id = %s", (c,)).fetchone()[0])
    url = f"/api/v1/admin/creator-earnings/{eid}/reverse"
    assert client.post(url, json={"note": "refund"}).status_code == 403
    assert client.post(url, headers={"X-Admin-Key": "wrong"}, json={"note": "refund"}).status_code == 403
    r = client.post(url, headers=A, json={"note": "refunded purchase"})
    assert r.status_code == 200 and r.json() == {"id": eid, "changed": True, "status": "reversed"}
    assert client.get("/api/v1/wallet/earnings", headers=H(c)).json()["lifetime_coins"] == 0
    assert client.post("/api/v1/admin/creator-earnings/not-a-uuid/reverse", headers=A,
                       json={"note": "x y z"}).status_code == 404
