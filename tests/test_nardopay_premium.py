"""VIYO Premium as a prepaid NardoPay pass: real Postgres, NardoPay HTTP mocked, webhooks signed with a test secret.
Premium is real money: these tests also prove a pass never touches coins."""
import datetime

import pytest

import checkout_options as co
import nardopay_payments as np_
import subscriptions as sub
from conftest import ROOT, connect, make_user
from test_nardopay_payments import (ADMIN, FakeResponse, H, balance, credits, deliver, env, payload,  # noqa: F401
                                    schema)


@pytest.fixture(scope="module")
def premium_schema(schema):
    conn = connect()
    conn.execute((ROOT / "subscriptions_schema.sql").read_text())
    for _ in range(2):                                               # safe to run again
        conn.execute((ROOT / "nardopay_premium.sql").read_text())
    conn.close()


@pytest.fixture
def pv(premium_schema, env):
    return env


def buy_pass(client, u, plan="monthly"):
    r = client.post("/api/v1/subscription/nardopay/create", headers=H(u), json={"plan": plan})
    assert r.status_code == 200, r.text
    return r.json()


def profile(db, u):
    cols = ("is_subscribed", "subscription_provider", "subscription_plan", "subscription_status", "subscription_ends_at")
    return dict(zip(cols, db.execute(f"select {', '.join(cols)} from public.profiles where id = %s", (u,)).fetchone()))


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def test_a_pass_is_priced_on_the_server_and_sold_as_premium_not_coins(pv):
    client, db, calls, st = pv
    u = make_user(db)
    assert client.post("/api/v1/subscription/nardopay/create", headers=H(u),
                       json={"plan": "monthly", "usd_cents": 1}).status_code == 422       # no client price
    assert client.post("/api/v1/subscription/nardopay/create", headers=H(u),
                       json={"plan": "daily"}).status_code == 422
    o = buy_pass(client, u)
    assert o["kind"] == "premium" and o["plan"] == "monthly" and o["pass_days"] == 30 and o["coins"] is None
    assert o["usd_cents"] == 999 and o["rwf_amount"] == 14086                       # 9.99 x 1410 = 14085.9 -> 14086
    body = calls[-1]["body"]
    assert body["product_name"] == "VIYO Premium" and body["amount"] == 14086
    assert "doesn't renew" in body["description"] and body["webhook_url"]


def test_a_verified_payment_grants_30_days_once_and_never_coins(pv):
    client, db, calls, st = pv
    u = make_user(db)
    coins_before = balance(db, u)
    o = buy_pass(client, u)
    r = deliver(client, payload(o["order_id"], tx="tx_p1", amount=14086))
    assert r.status_code == 200 and r.json()["result"] == "processed", r.text
    p = profile(db, u)
    assert p["is_subscribed"] is True and p["subscription_provider"] == "nardopay" and p["subscription_plan"] == "monthly"
    assert abs((p["subscription_ends_at"] - now()) - datetime.timedelta(days=30)) < datetime.timedelta(minutes=2)
    assert balance(db, u) == coins_before and credits(db, "tx_p1") == 0                 # no coins, no coin ledger
    ends = p["subscription_ends_at"]
    # The same payment delivered again adds nothing.
    assert deliver(client, payload(o["order_id"], tx="tx_p1", amount=14086, retry=1)).status_code == 200
    assert profile(db, u)["subscription_ends_at"] == ends
    status = client.get(f"/api/v1/coins/purchase/nardopay/orders/{o['order_id']}", headers=H(u)).json()
    assert status["status"] == "credited" and status["kind"] == "premium"


def test_buying_again_while_active_adds_the_time_after_the_current_end(pv):
    client, db, calls, st = pv
    u = make_user(db)
    a = buy_pass(client, u, "weekly")
    deliver(client, payload(a["order_id"], tx="tx_w", amount=a["rwf_amount"]))
    first_end = profile(db, u)["subscription_ends_at"]
    st["response"] = lambda body: FakeResponse(200, {"x_link": "https://pay.example/checkout/second"})
    b = buy_pass(client, u, "monthly")
    deliver(client, payload(b["order_id"], tx="tx_m", amount=b["rwf_amount"]))
    assert profile(db, u)["subscription_ends_at"] == first_end + datetime.timedelta(days=30)


def test_wrong_amount_or_a_test_event_grants_nothing(pv):
    client, db, calls, st = pv
    u = make_user(db)
    o = buy_pass(client, u)
    assert deliver(client, payload(o["order_id"], tx="tx_bad", amount=100)).json()["result"] == "mismatch"
    assert deliver(client, payload(o["order_id"], tx="tx_t", amount=14086, test=True)).json()["result"] == "ignored"
    assert profile(db, u)["is_subscribed"] is False


def test_someone_with_another_active_subscription_is_not_charged_twice(pv):
    client, db, calls, st = pv
    u = make_user(db)
    db.execute("update public.profiles set is_subscribed = true, subscription_provider = 'lemonsqueezy', "
               "subscription_status = 'active' where id = %s", (u,))
    r = client.post("/api/v1/subscription/nardopay/create", headers=H(u), json={"plan": "monthly"})
    assert r.status_code == 409 and calls == []


def test_an_ended_pass_gives_nothing_and_is_switched_off(pv):
    client, db, calls, st = pv
    u = make_user(db)
    o = buy_pass(client, u)
    deliver(client, payload(o["order_id"], tx="tx_e", amount=14086))
    db.execute("update public.profiles set subscription_ends_at = now() - interval '1 minute' where id = %s", (u,))
    row = {k: (v.isoformat() if isinstance(v, datetime.datetime) else v) for k, v in profile(db, u).items()}
    assert sub._pass_ended(row) is True                       # the paywall stops honouring it at once
    assert sub._pass_ended({**row, "subscription_provider": "lemonsqueezy"}) is False
    out = np_.reconcile()
    assert out["passes_ended"] == 1
    p = profile(db, u)
    assert p["is_subscribed"] is False and p["subscription_status"] == "expired"


def test_a_crash_after_the_payment_is_verified_is_finished_by_reconciliation(pv, monkeypatch):
    client, db, calls, st = pv
    u = make_user(db)
    o = buy_pass(client, u)
    real = np_._rpc

    def flaky(name, params):
        if name == "nardopay_grant_premium":
            raise RuntimeError("crash")
        return real(name, params)
    monkeypatch.setattr(np_, "_rpc", flaky)
    assert deliver(client, payload(o["order_id"], tx="tx_c", amount=14086)).status_code == 500
    assert profile(db, u)["is_subscribed"] is False
    monkeypatch.setattr(np_, "_rpc", real)
    out = np_.reconcile(min_age_seconds=0)
    assert out["finished"] == 1
    assert profile(db, u)["is_subscribed"] is True
    assert np_.reconcile(min_age_seconds=0)["finished"] == 0                      # once


def test_coin_orders_still_work_next_to_premium(pv):
    client, db, calls, st = pv
    u = make_user(db)
    before = balance(db, u)
    r = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u), json={"package_id": "starter"})
    assert r.status_code == 200 and r.json()["kind"] == "coins"
    assert deliver(client, payload(r.json()["order_id"], tx="tx_coin", amount=1396)).json()["result"] == "processed"
    assert balance(db, u) == before + 100 and profile(db, u)["is_subscribed"] is False


def test_checkout_options_offer_the_pass_only_when_nardopay_is_available(pv, monkeypatch):
    plans = co.nardopay_premium_plans()
    assert [(p["plan"], p["price_cents"], p["days"]) for p in plans] == [
        ("weekly", 399, 7), ("monthly", 999, 30), ("yearly", 5999, 365)]
    monkeypatch.setenv("NARDOPAY_ENABLED", "false")
    assert co.nardopay_premium_plans() == []
