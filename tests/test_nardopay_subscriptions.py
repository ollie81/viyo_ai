"""VIYO Premium as a recurring NardoPay subscription: real Postgres, NardoPay HTTP mocked, webhooks signed with a test
secret and shaped like NardoPay's real payment webhook (matched by link_code). Premium is real money: never coins."""
import datetime

import pytest

import nardopay_payments as np_
import subscriptions as sub
from conftest import ROOT, connect, make_user
from test_nardopay_payments import (FakeResponse, H, balance, buy, deliver, env, real, real_payment,  # noqa: F401
                                    schema)


@pytest.fixture(scope="module")
def subs_schema(schema):
    conn = connect()
    conn.execute((ROOT / "subscriptions_schema.sql").read_text())
    for _ in range(2):                                               # safe to run again
        conn.execute((ROOT / "nardopay_subscriptions.sql").read_text())
    conn.close()


@pytest.fixture
def sv(subs_schema, real, db):
    client, db, calls, st = real
    db.execute("truncate public.nardopay_subscriptions cascade")
    codes = iter([f"vsub{i:012d}" for i in range(1, 50)])
    st["response"] = lambda body: FakeResponse(200, {"url": f"https://nardopay.com/s/{next(codes)}?pay=1"})
    return client, db, calls, st


def subscribe(client, u, plan="monthly"):
    r = client.post("/api/v1/subscription/nardopay/create", headers=H(u), json={"plan": plan})
    assert r.status_code == 200, r.text
    return r.json()


def code_of(db, sid):
    return db.execute("select link_code from public.nardopay_subscriptions where id = %s", (sid,)).fetchone()[0]


def profile(db, u):
    cols = ("is_subscribed", "subscription_provider", "subscription_plan", "subscription_status",
            "subscription_renews_at", "subscription_ends_at", "subscription_id")
    return dict(zip(cols, db.execute(f"select {', '.join(cols)} from public.profiles where id = %s", (u,)).fetchone()))


def paid(code, amount, tx, event="payment.completed", **extra):
    return real_payment(code, amount=amount, tx=tx, event=event, link_type="subscription", **extra)


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def payments(db, sid):
    return db.execute("select count(*) from public.nardopay_subscription_payments where subscription_id = %s",
                      (sid,)).fetchone()[0]


def test_checkout_creates_this_users_own_subscription_link_with_the_owners_plan(sv):
    client, db, calls, st = sv
    u = make_user(db)
    assert client.post("/api/v1/subscription/nardopay/create", headers=H(u),
                       json={"plan": "monthly", "amount": 1}).status_code == 422          # no client price
    s = subscribe(client, u, "monthly")
    assert s["status"] == "pending" and s["plan"] == "monthly" and s["usd_cents"] == 999
    assert s["payment_link"].startswith("https://nardopay.com/s/")
    body = calls[-1]["body"]
    assert body["link_type"] == "subscription" and body["amount"] == 14800 and body["billing_cycle"] == "monthly"
    assert body["trial_days"] == 0 and body["redirect_url"] == "https://viyo.site/app" and body["webhook_url"]
    assert body["metadata"] == {"subscription_id": s["subscription_id"]}
    for plan, rwf, cycle in (("weekly", 5900, "weekly"), ("yearly", 88600, "yearly")):
        subscribe(client, make_user(db), plan)
        assert (calls[-1]["body"]["amount"], calls[-1]["body"]["billing_cycle"]) == (rwf, cycle)


def test_the_redirect_alone_grants_nothing(sv):
    client, db, calls, st = sv
    u = make_user(db)
    s = subscribe(client, u)
    status = client.get(f"/api/v1/subscription/nardopay/{s['subscription_id']}", headers=H(u)).json()
    assert status["status"] == "pending" and profile(db, u)["is_subscribed"] is False


def test_a_verified_payment_starts_premium_for_one_month_plus_grace_and_never_coins(sv):
    client, db, calls, st = sv
    u = make_user(db)
    coins = balance(db, u)
    s = subscribe(client, u)
    r = deliver(client, paid(code_of(db, s["subscription_id"]), 14800, "tx_first"))
    assert r.status_code == 200 and r.json()["result"] == "processed", r.text
    p = profile(db, u)
    assert p["is_subscribed"] is True and p["subscription_provider"] == "nardopay" and p["subscription_plan"] == "monthly"
    assert p["subscription_id"] == s["subscription_id"]
    assert 27 <= (p["subscription_renews_at"] - now()).days <= 31                  # one calendar month
    assert p["subscription_ends_at"] - p["subscription_renews_at"] == datetime.timedelta(hours=48)
    assert balance(db, u) == coins                                                  # no coins, ever
    assert client.get(f"/api/v1/subscription/nardopay/{s['subscription_id']}", headers=H(u)).json()["status"] == "active"


def test_the_same_payment_twice_or_as_two_events_counts_once(sv):
    client, db, calls, st = sv
    u = make_user(db)
    s = subscribe(client, u, "weekly")
    code = code_of(db, s["subscription_id"])
    deliver(client, paid(code, 5900, "tx_w1"))
    ends = profile(db, u)["subscription_ends_at"]
    assert deliver(client, paid(code, 5900, "tx_w1")).json()["duplicate"] is True           # NardoPay retry
    assert deliver(client, paid(code, 5900, "tx_w1", event="subscription.renewed")).json()["result"] == "processed"
    assert profile(db, u)["subscription_ends_at"] == ends and payments(db, s["subscription_id"]) == 1


def test_a_renewal_extends_from_the_current_period_end(sv):
    client, db, calls, st = sv
    u = make_user(db)
    s = subscribe(client, u, "weekly")
    code = code_of(db, s["subscription_id"])
    deliver(client, paid(code, 5900, "tx_a"))
    first = profile(db, u)["subscription_renews_at"]
    deliver(client, paid(code, 5900, "tx_b", event="subscription.renewed"))
    p = profile(db, u)
    assert p["subscription_renews_at"] == first + datetime.timedelta(days=7)
    assert payments(db, s["subscription_id"]) == 2


@pytest.mark.parametrize("change", [
    {"amount": 5000}, {"currency": "USD"}, {"test": True}, {"event": "payment.failed"}, {"event": "refund.created"}])
def test_wrong_amount_currency_test_or_other_events_grant_nothing(sv, change):
    client, db, calls, st = sv
    u = make_user(db)
    s = subscribe(client, u)
    body = paid(code_of(db, s["subscription_id"]), 14800, "tx_x")
    body.update(change)
    assert deliver(client, body).status_code == 200
    assert profile(db, u)["is_subscribed"] is False and payments(db, s["subscription_id"]) == 0


def test_an_unknown_subscription_link_grants_nothing(sv):
    client, db, calls, st = sv
    u = make_user(db)
    subscribe(client, u)
    assert deliver(client, paid("viyopremiummonthly", 14800, "tx_dash")).json()["result"] == "ignored"
    assert profile(db, u)["is_subscribed"] is False


def test_no_renewal_means_premium_ends_after_the_grace_period(sv):
    client, db, calls, st = sv
    u = make_user(db)
    s = subscribe(client, u)
    deliver(client, paid(code_of(db, s["subscription_id"]), 14800, "tx_e"))
    db.execute("update public.profiles set subscription_ends_at = now() - interval '1 minute' where id = %s", (u,))
    row = {k: (v.isoformat() if isinstance(v, datetime.datetime) else v) for k, v in profile(db, u).items()}
    assert sub.nardopay_premium_ended(row) is True                                  # the paywall stops at once
    assert np_.reconcile()["premium_ended"] == 1
    p = profile(db, u)
    assert p["is_subscribed"] is False and p["subscription_status"] == "expired"


def test_cant_subscribe_twice_while_premium_is_active(sv):
    client, db, calls, st = sv
    u = make_user(db)
    s = subscribe(client, u)
    deliver(client, paid(code_of(db, s["subscription_id"]), 14800, "tx_one"))
    n = len(calls)
    assert client.post("/api/v1/subscription/nardopay/create", headers=H(u), json={"plan": "yearly"}).status_code == 409
    v = make_user(db)
    db.execute("update public.profiles set is_subscribed = true, subscription_provider = 'lemonsqueezy', "
               "subscription_status = 'active' where id = %s", (v,))
    assert client.post("/api/v1/subscription/nardopay/create", headers=H(v), json={"plan": "monthly"}).status_code == 409
    assert len(calls) == n                                                          # no link created


def test_only_the_owner_can_see_a_subscription(sv):
    client, db, calls, st = sv
    u, other = make_user(db), make_user(db)
    s = subscribe(client, u)
    assert client.get(f"/api/v1/subscription/nardopay/{s['subscription_id']}", headers=H(other)).status_code == 404


def test_a_link_without_a_readable_code_is_never_sold(sv):
    client, db, calls, st = sv
    st["response"] = lambda body: FakeResponse(200, {"url": "https://nardopay.com/checkout?x=1"})
    u = make_user(db)
    r = client.post("/api/v1/subscription/nardopay/create", headers=H(u), json={"plan": "monthly"})
    assert r.status_code == 502
    assert db.execute("select status from public.nardopay_subscriptions where user_id = %s", (u,)).fetchone()[0] == "failed"


def test_switched_off_means_no_premium_checkout(sv, monkeypatch):
    client, db, calls, st = sv
    monkeypatch.setenv("NARDOPAY_ENABLED", "false")
    r = client.post("/api/v1/subscription/nardopay/create", headers=H(make_user(db)), json={"plan": "monthly"})
    assert r.status_code == 503 and calls == []


def test_coin_purchases_are_unchanged_next_to_subscriptions(sv):
    client, db, calls, st = sv
    u = make_user(db)
    st["response"] = lambda body: FakeResponse(200, {"url": "https://nardopay.com/p/coinlink0001?pay=1"})
    buy(client, u)
    before = balance(db, u)
    assert deliver(client, real_payment("coinlink0001", amount=1396, tx="tx_coin")).json()["result"] == "processed"
    assert balance(db, u) == before + 100 and profile(db, u)["is_subscribed"] is False
