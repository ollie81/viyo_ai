"""Watch coins: atomic balance changes, one credit per payment, and the
paid-coin rules creator earnings depend on stay the same."""
import threading
import uuid

import pytest
from fastapi import HTTPException

import coins
from conftest import ROOT, connect, make_user


class _Res:
    def __init__(self, data):
        self.data = data


class PgAdmin:
    """Just enough of the Supabase client for coins.py's rpc path."""

    def __init__(self, conn):
        self.conn = conn

    def rpc(self, name, params):
        conn = self.conn

        class Call:
            def execute(self_inner):
                args = ", ".join(f"{k} => %({k})s" for k in params)
                row = conn.execute(f"select public.{name}({args})", params).fetchone()
                return _Res(row[0])
        return Call()


@pytest.fixture(scope="module")
def ledger_schema(migrated):
    conn = connect()
    conn.execute((ROOT / "creator_payouts_schema.sql").read_text())
    conn.execute((ROOT / "coin_ledger_fix.sql").read_text())
    conn.close()
    return migrated


@pytest.fixture
def ldb(ledger_schema, db):
    return db


def state(db, u):
    bal, paid = db.execute("select points_balance, paid_coins from public.profiles where id=%s", (u,)).fetchone()
    return int(bal), float(paid)


def test_a_payment_is_credited_once_even_if_the_webhook_repeats(ldb):
    admin, u = PgAdmin(ldb), make_user(ldb, balance=0)
    assert coins.credit_coins(admin, str(u), 550, "coin_purchase", "Purchased 550 Coins — Stripe pi_1", external_ref="stripe:pi_1")
    assert coins.credit_coins(admin, str(u), 550, "coin_purchase", "Purchased 550 Coins — Stripe pi_1", external_ref="stripe:pi_1") is False
    assert state(ldb, u) == (550, 550.0), "balance and bought-coin count move once"
    assert ldb.execute("select count(*) from public.transactions where external_ref='stripe:pi_1'").fetchone()[0] == 1


def test_parallel_webhook_deliveries_credit_once(ldb):
    u = make_user(ldb, balance=0)
    results = []

    def deliver():
        conn = connect()
        try:
            results.append(coins.credit_coins(PgAdmin(conn), str(u), 100, "coin_purchase", "Purchased — Paystack R9",
                                              external_ref="paystack:R9"))
        finally:
            conn.close()

    threads = [threading.Thread(target=deliver) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1 and results.count(False) == 7
    assert state(ldb, u) == (100, 100.0)


def test_different_payments_both_credit(ldb):
    admin, u = PgAdmin(ldb), make_user(ldb, balance=0)
    coins.credit_coins(admin, str(u), 100, "coin_purchase", "a", external_ref="google_play:tok-a")
    coins.credit_coins(admin, str(u), 100, "coin_purchase", "b", external_ref="google_play:tok-b")
    assert state(ldb, u)[0] == 200


def test_spending_is_atomic_and_never_goes_below_zero(ldb):
    admin, u = PgAdmin(ldb), make_user(ldb, balance=8)
    coins.FEATURE_COSTS.setdefault("hook_check", {"cost": 5, "free_per_day": 1, "label": "Hook Check"})
    ldb.execute("insert into public.transactions (user_id, amount, type, description, created_at) "
                "values (%s, 0, 'hook_check', 'Free daily use', now())", (u,))  # free taste already used today
    admin.table = lambda name: _FreeTasteUsed()
    assert coins.spend_on_feature(admin, str(u), "hook_check") == 5
    with pytest.raises(HTTPException) as e:
        coins.spend_on_feature(admin, str(u), "hook_check")
    assert e.value.status_code == 402 and e.value.detail["balance"] == 3
    assert state(ldb, u)[0] == 3
    rows = ldb.execute("select amount, type from public.transactions where user_id=%s and amount <> 0", (u,)).fetchall()
    assert rows == [(-5, "hook_check")], "one history row per charge, written with the balance change"


class _FreeTasteUsed:
    """coins._has_free_taste reads transactions through the table API;
    answer 'already used today' so the paid path runs."""

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def gte(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return _Res([{"id": 1}])


def test_free_coins_are_still_spent_first_so_creator_earnings_rules_hold(ldb):
    admin, u = PgAdmin(ldb), make_user(ldb, balance=0)
    coins.credit_coins(admin, str(u), 100, "coin_purchase", "bought", external_ref="stripe:pi_free_first")
    coins.credit_coins(admin, str(u), 50, "daily_checkin", "free")
    assert state(ldb, u) == (150, 100.0)
    coins.debit_coins(admin, str(u), 60, "episode_unlock", "Unlocked ep")
    # 50 free coins used first, then 10 bought: 90 bought coins left.
    assert state(ldb, u) == (90, 90.0)
    with pytest.raises(HTTPException) as e:
        coins.debit_coins(admin, str(u), 1000, "gift", "too much")
    assert e.value.status_code == 402 and state(ldb, u) == (90, 90.0)


def test_old_purchases_are_backfilled_so_they_cannot_be_credited_again(ldb):
    u = make_user(ldb, balance=0)
    for desc in ["Purchased 550 Coins — Stripe pi_old", "Purchased 550 Coins — Stripe pi_old",   # already doubled once
                 "Purchased 100 Coins — Lemon Squeezy order 777", "Purchased 100 Coins — Google Play tok_old"]:
        ldb.execute("insert into public.transactions (user_id, amount, type, description) values (%s, 1, 'coin_purchase', %s)",
                    (u, desc))
    ldb.execute((ROOT / "coin_ledger_fix.sql").read_text())  # safe to run again
    refs = sorted(r[0] for r in ldb.execute(
        "select external_ref from public.transactions where user_id=%s and external_ref is not null", (u,)).fetchall())
    assert refs == ["google_play:tok_old", "lemonsqueezy:777", "stripe:pi_old"], "the double row keeps no ref"
    assert coins.credit_coins(PgAdmin(ldb), str(u), 550, "coin_purchase", "retry", external_ref="stripe:pi_old") is False


def test_falls_back_to_the_old_path_before_the_sql_is_run():
    class NoFunction:
        def rpc(self, *a):
            raise Exception("{'code': 'PGRST202', 'message': 'Could not find the function public.viyo_coin_apply'}")

    with pytest.raises(coins._NotMigrated):
        coins._apply(NoFunction(), str(uuid.uuid4()), 1, "x", "y")
