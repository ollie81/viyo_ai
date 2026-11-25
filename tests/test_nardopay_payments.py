"""NardoPay web coin checkout (nardopay_payments.py): real Postgres, the real coin ledger (viyo_coin_apply), NardoPay
HTTP mocked, webhooks signed with a test secret. No real NardoPay call and no money.

The NardoPay field NAMES used here (x_link, x_event, ...) are TEST PLACEHOLDERS for the settings an operator fills in
from NardoPay's documentation; they are not NardoPay's real names."""
import datetime
import hashlib
import json
import uuid

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

import nardopay_payments as np_
from conftest import ROOT, connect, make_user
from studio_store import PgStore

SECRET = "test-webhook-secret"
HOOK = "https://api.viyo.test/api/v1/webhooks/nardopay"
FIELDS = {"NARDOPAY_LINK_URL_FIELD": "x_link", "NARDOPAY_FIELD_EVENT_TYPE": "x_event",
          "NARDOPAY_FIELD_ORDER_ID": "x_meta.order", "NARDOPAY_FIELD_TRANSACTION_ID": "x_tx",
          "NARDOPAY_FIELD_AMOUNT": "x_amount", "NARDOPAY_FIELD_CURRENCY": "x_currency"}


class _Res:
    def __init__(self, data):
        self.data = data


class PgAdmin:
    """What coins.credit_coins needs from the Supabase client: rpc(...).execute()."""
    def __init__(self, conn):
        self.conn = conn

    def rpc(self, name, params):
        conn = self.conn

        class Call:
            def execute(self_inner):
                args = ", ".join(f"{k} => %({k})s" for k in params)
                return _Res(conn.execute(f"select public.{name}({args})", params).fetchone()[0])
        return Call()


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.fixture(scope="module")
def schema(migrated):
    conn = connect()
    conn.execute((ROOT / "creator_payouts_schema.sql").read_text())
    conn.execute((ROOT / "coin_ledger_fix.sql").read_text())
    conn.execute((ROOT / "nardopay_orders.sql").read_text())
    conn.close()


@pytest.fixture
def env(schema, db, monkeypatch):
    db.execute("truncate public.nardopay_orders, public.nardopay_webhook_events, public.nardopay_rates cascade")
    monkeypatch.setattr(np_, "_store", PgStore(db))
    monkeypatch.setattr(np_, "_admin", PgAdmin(db))
    for k, v in {**FIELDS, "NARDOPAY_ENABLED": "true", "NARDOPAY_LEDGER_VERIFIED": "true", "NARDOPAY_API_KEY": "test-key",
                 "NARDOPAY_AMOUNT_UNIT": "rwf", "NARDOPAY_WEBHOOK_SECRET": SECRET, "ADMIN_API_KEY": "admin-secret",
                 "NARDOPAY_WEBHOOK_URL": HOOK}.items():
        monkeypatch.setenv(k, v)
    calls = []

    def post(url, headers, body):
        calls.append({"url": url, "headers": headers, "body": body})
        return env_state["response"](body)
    env_state = {"response": lambda body: FakeResponse(200, {"x_link": "https://pay.example/checkout/abc"})}
    monkeypatch.setattr(np_, "_http_post", post)
    app = FastAPI()
    app.include_router(np_.router)

    async def buyer(authorization: str = Header(None)):
        if not authorization:
            raise HTTPException(status_code=401, detail="Not signed in")
        return authorization.removeprefix("Bearer ")
    app.dependency_overrides[np_._buyer] = buyer
    client = TestClient(app)
    db.execute("insert into public.nardopay_rates (rate, set_by) values (1410, 'owner')")
    return client, db, calls, env_state


def H(u):
    return {"Authorization": f"Bearer {u}"}


ADMIN = {"X-Admin-Key": "admin-secret"}


def balance(db, u):
    return db.execute("select points_balance from public.profiles where id = %s", (u,)).fetchone()[0]


def buy(client, u, package="starter"):
    r = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u), json={"package_id": package})
    assert r.status_code == 200, r.text
    return r.json()


def payload(order_id, tx="tx_1", amount=1396, currency="RWF", event="payment.completed", **extra):
    return {"x_event": event, "x_meta": {"order": order_id}, "x_tx": tx, "x_amount": amount, "x_currency": currency,
            **extra}


def deliver(client, body, secret=SECRET, signature=None):
    raw = json.dumps(body, separators=(",", ":")).encode() if not isinstance(body, bytes) else body
    sig = signature if signature is not None else np_.sign(raw, secret)
    return client.post("/api/v1/webhooks/nardopay", content=raw,
                       headers={"X-NardoPay-Signature": sig, "Content-Type": "application/json"})


def order_row(db, oid):
    cols = ("status", "transaction_id", "credited_at", "verified_event_id")
    return dict(zip(cols, db.execute(f"select {', '.join(cols)} from public.nardopay_orders where id = %s", (oid,)).fetchone()))


def credits(db, tx):
    return db.execute("select count(*) from public.transactions where external_ref = %s", (f"nardopay:{tx}",)).fetchone()[0]


# ------------------------------------------------------------------------------------------- rates

def test_conversion_rounds_up_to_whole_francs():
    assert np_.rwf_for(99, 1410) == 1396           # 1395.9 -> 1396
    assert np_.rwf_for(499, 1410) == 7036          # 7035.9 -> 7036
    assert np_.rwf_for(100, 1400) == 1400


def test_only_the_admin_sets_rates_and_mistakes_are_refused(env):
    client, db, calls, st = env
    body = {"rate": 1420, "set_by": "owner"}
    assert client.post("/api/v1/admin/nardopay/rate", json=body).status_code == 403
    assert client.post("/api/v1/admin/nardopay/rate", headers={"X-Admin-Key": "wrong"}, json=body).status_code == 403
    assert client.post("/api/v1/admin/nardopay/rate", headers=ADMIN, json={**body, "rate": 14100}).status_code == 422
    big = client.post("/api/v1/admin/nardopay/rate", headers=ADMIN, json={**body, "rate": 1700})
    assert big.status_code == 409 and "confirm_large_change" in big.json()["detail"]
    assert client.post("/api/v1/admin/nardopay/rate", headers=ADMIN,
                       json={**body, "rate": 1700, "confirm_large_change": True}).status_code == 200
    assert client.post("/api/v1/admin/nardopay/rate", headers=ADMIN, json=body).status_code == 409   # back down >10%
    assert client.get("/api/v1/admin/nardopay/rate", headers=ADMIN).json()["rate"]["rate"] in (1700, "1700.0000", 1700.0)
    assert db.execute("select count(*) from public.nardopay_rates").fetchone()[0] == 2               # history kept


def test_a_stale_or_missing_rate_pauses_checkout(env):
    client, db, calls, st = env
    u = make_user(db)
    db.execute("update public.nardopay_rates set set_at = now() - interval '8 days'")
    r = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u), json={"package_id": "starter"})
    assert r.status_code == 503 and "out of date" in r.json()["detail"]
    db.execute("delete from public.nardopay_rates")
    r = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u), json={"package_id": "starter"})
    assert r.status_code == 503 and "No exchange rate" in r.json()["detail"]
    assert calls == [] and db.execute("select count(*) from public.nardopay_orders").fetchone()[0] == 0


# ------------------------------------------------------------------------------------------- orders

def test_an_order_is_created_server_side_with_the_documented_request(env):
    client, db, calls, st = env
    u = make_user(db)
    out = buy(client, u)
    assert out["coins"] == 100 and out["usd_cents"] == 99 and out["rwf_amount"] == 1396
    assert out["payment_link"] == "https://pay.example/checkout/abc" and out["status"] == "pending"
    sent = calls[0]
    assert sent["url"] == np_.CREATE_LINK_URL
    assert sent["headers"] == {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
    assert sent["body"] == {"link_type": "payment", "product_name": "VIYO Coins", "amount_mode": "fixed",
                            "amount": 1396, "description": "VIYO Coins purchase: 100 Coins",
                            "metadata": {"order_id": out["order_id"]}, "webhook_url": HOOK}
    row = db.execute("select user_id, usd_rwf_rate, rate_id is not null from public.nardopay_orders where id = %s",
                     (out["order_id"],)).fetchone()
    assert str(row[0]) == str(u) and float(row[1]) == 1410 and row[2] is True


def test_minor_units_multiply_the_amount_when_configured(env, monkeypatch):
    client, db, calls, st = env
    monkeypatch.setenv("NARDOPAY_AMOUNT_UNIT", "rwf_minor")
    buy(client, make_user(db))
    assert calls[0]["body"]["amount"] == 139600


def test_the_client_cannot_set_price_or_coins(env):
    client, db, calls, st = env
    u = make_user(db)
    r = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u),
                    json={"package_id": "starter", "coins": 999999, "amount": 1})
    assert r.status_code == 422 and calls == []
    assert client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u),
                       json={"package_id": "free_stuff"}).status_code == 400


@pytest.mark.parametrize("unset", ["NARDOPAY_ENABLED", "NARDOPAY_LEDGER_VERIFIED", "NARDOPAY_API_KEY",
                                   "NARDOPAY_AMOUNT_UNIT", "NARDOPAY_LINK_URL_FIELD", "NARDOPAY_FIELD_TRANSACTION_ID"])
def test_checkout_stays_off_until_everything_is_set(env, monkeypatch, unset):
    client, db, calls, st = env
    monkeypatch.delenv(unset)
    r = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(make_user(db)), json={"package_id": "starter"})
    assert r.status_code == 503 and calls == []


@pytest.mark.parametrize("response", [
    FakeResponse(500, {"error": "boom"}), FakeResponse(200, ValueError("not json")), FakeResponse(200, {}),
    FakeResponse(200, {"x_link": "http://insecure.example"}), FakeResponse(200, {"x_link": 42})])
def test_a_bad_nardopay_response_fails_the_order_without_guessing(env, response):
    client, db, calls, st = env
    st["response"] = lambda body: response
    r = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(make_user(db)), json={"package_id": "starter"})
    assert r.status_code == 502
    assert db.execute("select status, payment_link from public.nardopay_orders").fetchone() == ("failed", None)


def test_only_the_buyer_sees_their_order(env):
    client, db, calls, st = env
    u, other = make_user(db), make_user(db)
    oid = buy(client, u)["order_id"]
    assert client.get(f"/api/v1/coins/purchase/nardopay/orders/{oid}", headers=H(u)).json()["status"] == "pending"
    assert client.get(f"/api/v1/coins/purchase/nardopay/orders/{oid}", headers=H(other)).status_code == 404
    assert client.get("/api/v1/coins/purchase/nardopay/orders/not-a-uuid", headers=H(u)).status_code == 404


def test_no_redirect_or_get_route_can_credit_coins():
    routes = {(r.path, m) for r in np_.router.routes for m in r.methods}
    gets = sorted(p for p, m in routes if m == "GET")
    assert gets == ["/api/v1/admin/nardopay/duplicate-payments", "/api/v1/admin/nardopay/rate",
                    "/api/v1/coins/purchase/nardopay/availability",
                    "/api/v1/coins/purchase/nardopay/orders/{order_id}"]


# ------------------------------------------------------------------------------------------- webhook: signature

def test_invalid_or_missing_signatures_are_rejected_and_nothing_is_stored(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    body = payload(oid)
    assert deliver(client, body, secret="wrong-secret").status_code == 400
    raw = json.dumps(body).encode()
    assert client.post("/api/v1/webhooks/nardopay", content=raw).status_code == 400            # no header
    tampered = json.dumps({**body, "x_amount": 1}).encode()
    assert deliver(client, tampered, signature=np_.sign(raw, SECRET)).status_code == 400       # body changed
    assert db.execute("select count(*) from public.nardopay_webhook_events").fetchone()[0] == 0
    assert balance(db, u) == 100


def test_the_signature_is_hex_hmac_sha256_of_the_raw_body():
    raw = b'{"a":1}'
    assert np_.sign(raw, "k") == __import__("hmac").new(b"k", raw, hashlib.sha256).hexdigest()
    assert np_.signature_ok(raw, np_.sign(raw, "k").upper(), "k")          # hex, any case
    assert not np_.signature_ok(raw, "sha256=" + np_.sign(raw, "k"), "k")   # no undocumented prefix accepted
    assert not np_.signature_ok(raw, np_.sign(raw, "k"), "")


# ------------------------------------------------------------------------------------------- webhook: crediting

def test_a_verified_payment_credits_the_package_once(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    r = deliver(client, payload(oid))
    assert r.status_code == 200 and r.json()["result"] == "processed"
    assert balance(db, u) == 200 and credits(db, "tx_1") == 1
    o = order_row(db, oid)
    assert o["status"] == "credited" and o["transaction_id"] == "tx_1" and o["credited_at"] and o["verified_event_id"]
    assert db.execute("select status from public.nardopay_webhook_events").fetchone()[0] == "processed"
    st_ = client.get(f"/api/v1/coins/purchase/nardopay/orders/{oid}", headers=H(u)).json()
    assert st_["status"] == "credited" and st_["payment_link"] is None


def test_the_same_delivery_twice_credits_once(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    body = payload(oid)
    assert deliver(client, body).status_code == 200
    again = deliver(client, body)
    assert again.status_code == 200 and again.json()["duplicate"] is True
    assert balance(db, u) == 200 and db.execute("select count(*) from public.nardopay_webhook_events").fetchone()[0] == 1


def test_a_retry_of_the_same_transaction_with_a_different_body_credits_once(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    deliver(client, payload(oid, sent_at="10:00"))
    r = deliver(client, payload(oid, sent_at="10:05"))                      # same transaction, new body
    assert r.status_code == 200
    assert balance(db, u) == 200 and credits(db, "tx_1") == 1


def test_a_second_transaction_for_a_paid_order_is_never_credited(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    deliver(client, payload(oid, tx="tx_1"))
    r = deliver(client, payload(oid, tx="tx_2"))
    assert r.status_code == 200 and r.json()["result"] == "duplicate_payment"
    assert balance(db, u) == 200 and credits(db, "tx_2") == 0
    dup = client.get("/api/v1/admin/nardopay/duplicate-payments", headers=ADMIN).json()["payments"]
    assert [d["transaction_id"] for d in dup] == ["tx_2"]


def test_a_transaction_already_used_for_another_order_is_not_credited_again(env):
    client, db, calls, st = env
    u = make_user(db)
    a, b = buy(client, u)["order_id"], buy(client, u)["order_id"]
    deliver(client, payload(a, tx="tx_same"))
    r = deliver(client, payload(b, tx="tx_same"))
    assert r.json()["result"] == "duplicate_payment" and balance(db, u) == 200
    assert order_row(db, b)["status"] == "pending"


@pytest.mark.parametrize("change", [{"amount": 1395}, {"amount": 13960}, {"currency": "USD"}, {"currency": None},
                                    {"tx": None}, {"amount": "abc"}])
def test_a_payment_that_doesnt_match_its_order_credits_nothing(env, change):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    r = deliver(client, payload(oid, **change))
    assert r.status_code == 200 and r.json()["result"] == "mismatch"
    assert balance(db, u) == 100 and order_row(db, oid)["status"] == "mismatch"


def test_unknown_orders_and_other_events_are_ignored(env):
    client, db, calls, st = env
    u = make_user(db)
    assert deliver(client, payload(str(uuid.uuid4()))).json()["result"] == "ignored"
    assert deliver(client, payload("not-a-uuid", tx="t2")).json()["result"] == "ignored"
    oid = buy(client, u)["order_id"]
    assert deliver(client, payload(oid, event="payment.refunded", tx="t3")).json()["result"] == "ignored"
    assert deliver(client, b"not json").json()["result"] == "ignored"
    assert balance(db, u) == 100 and order_row(db, oid)["status"] == "pending"


def test_a_failed_payment_marks_the_order_and_a_later_success_still_credits(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    deliver(client, payload(oid, event="payment.failed", tx="tx_f"))
    assert order_row(db, oid)["status"] == "failed" and balance(db, u) == 100
    deliver(client, payload(oid, tx="tx_ok"))
    assert order_row(db, oid)["status"] == "credited" and balance(db, u) == 200


def test_a_late_payment_for_an_expired_order_is_honoured(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    db.execute("update public.nardopay_orders set expires_at = now() - interval '1 hour' where id = %s", (oid,))
    assert np_.reconcile()["expired"] == 1 and order_row(db, oid)["status"] == "expired"
    deliver(client, payload(oid))
    assert order_row(db, oid)["status"] == "credited" and balance(db, u) == 200


# ------------------------------------------------------------------------------------------- recovery

def test_a_crash_after_crediting_is_finished_by_the_retry_without_double_credit(env, monkeypatch):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    real = np_._rpc

    def crash_on_finish(name, params):
        if name == "nardopay_finish_order":
            raise RuntimeError("server died")
        return real(name, params)
    monkeypatch.setattr(np_, "_rpc", crash_on_finish)
    r = deliver(client, payload(oid))
    assert r.status_code == 500
    assert balance(db, u) == 200 and order_row(db, oid)["status"] == "crediting"        # coins in, order not finished
    monkeypatch.setattr(np_, "_rpc", real)
    assert deliver(client, payload(oid)).status_code == 200                            # NardoPay retries
    assert balance(db, u) == 200 and credits(db, "tx_1") == 1 and order_row(db, oid)["status"] == "credited"


def test_reconciliation_finishes_a_verified_payment_when_no_retry_comes(env, monkeypatch):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    real = np_._rpc
    monkeypatch.setattr(np_, "_rpc", lambda n, p: (_ for _ in ()).throw(RuntimeError("died"))
                        if n == "nardopay_finish_order" else real(n, p))
    deliver(client, payload(oid))
    monkeypatch.setattr(np_, "_rpc", real)
    assert np_.reconcile(min_age_seconds=3600)["finished"] == 0                        # too recent: left alone
    out = np_.reconcile(min_age_seconds=0)
    assert out["finished"] == 1 and order_row(db, oid)["status"] == "credited"
    assert balance(db, u) == 200 and credits(db, "tx_1") == 1


def test_reconciliation_never_credits_an_unverified_order(env):
    client, db, calls, st = env
    u = make_user(db)
    pending = buy(client, u)["order_id"]
    # A claimed order whose recorded event doesn't match it (as if tampered with): must not be credited.
    bad = buy(client, u)["order_id"]
    ev = db.execute("insert into public.nardopay_webhook_events (body_sha256, signature_ok, transaction_id, amount, "
                    "currency, status) values (%s, true, 'tx_bad', 1, 'RWF', 'processing') returning id",
                    (hashlib.sha256(b"x").hexdigest(),)).fetchone()[0]
    db.execute("update public.nardopay_orders set status = 'crediting', transaction_id = 'tx_bad', verified_event_id = %s, "
               "payment_verified_at = now(), updated_at = now() - interval '1 hour' where id = %s", (ev, bad))
    out = np_.reconcile(min_age_seconds=0)
    assert out["finished"] == 0 and "doesn't match" in out["skipped"][0]["reason"]
    assert balance(db, u) == 100 and order_row(db, pending)["status"] == "pending"
    assert order_row(db, bad)["status"] == "crediting"


def test_without_the_ledger_check_nothing_is_credited_and_it_recovers_after(env, monkeypatch):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    monkeypatch.setenv("NARDOPAY_LEDGER_VERIFIED", "false")
    r = deliver(client, payload(oid))
    assert r.status_code == 503 and balance(db, u) == 100 and order_row(db, oid)["status"] == "pending"
    assert np_.reconcile(min_age_seconds=0)["skipped"] == [{"reason": "ledger not verified"}]
    monkeypatch.setenv("NARDOPAY_LEDGER_VERIFIED", "true")
    assert deliver(client, payload(oid)).status_code == 200 and balance(db, u) == 200


def test_events_received_before_the_fields_were_set_are_stored_then_reprocessed(env, monkeypatch):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    monkeypatch.delenv("NARDOPAY_FIELD_AMOUNT")
    r = deliver(client, payload(oid))
    assert r.status_code == 202 and balance(db, u) == 100
    assert db.execute("select status, amount from public.nardopay_webhook_events").fetchone() == ("received", None)
    monkeypatch.setenv("NARDOPAY_FIELD_AMOUNT", "x_amount")
    out = client.post("/api/v1/admin/nardopay/reprocess", headers=ADMIN).json()
    assert out["processed"] == 1 and balance(db, u) == 200
    # The stored event now carries the real amount, so reconciliation could re-check it.
    assert float(db.execute("select amount from public.nardopay_webhook_events").fetchone()[0]) == 1396


def test_a_delivery_already_being_processed_is_told_to_retry(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    body = payload(oid)
    raw = json.dumps(body, separators=(",", ":")).encode()
    db.execute("insert into public.nardopay_webhook_events (body_sha256, signature_ok, status, locked_at) "
               "values (%s, true, 'processing', now())", (hashlib.sha256(raw).hexdigest(),))
    r = deliver(client, body)
    assert r.status_code == 409 and balance(db, u) == 100


def test_raw_bodies_are_purged_after_the_retention_period(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    deliver(client, payload(oid))
    db.execute("insert into public.nardopay_webhook_events (body_sha256, signature_ok, status, raw_body, received_at) "
               "values (%s, true, 'mismatch', 'customer details', now() - interval '40 days')",
               (hashlib.sha256(b"old-mismatch").hexdigest(),))
    db.execute("update public.nardopay_webhook_events set received_at = now() - interval '31 days' where status = 'processed'")
    assert np_.reconcile()["purged"] == 1                   # the processed one (31 days); the mismatch is kept to 90
    rows = dict(db.execute("select status, raw_body is null from public.nardopay_webhook_events").fetchall())
    assert rows == {"processed": True, "mismatch": False}
    assert db.execute("select count(*) from public.nardopay_webhook_events where status = 'processed' and "
                      "body_sha256 is not null and transaction_id = 'tx_1'").fetchone()[0] == 1


# ------------------------------------------------------------------------------------------- availability and schedule

def test_availability_is_a_plain_yes_or_no(env, monkeypatch):
    client, db, calls, st = env
    assert client.get("/api/v1/coins/purchase/nardopay/availability").json() == {"available": True}
    monkeypatch.setenv("NARDOPAY_ENABLED", "false")
    assert client.get("/api/v1/coins/purchase/nardopay/availability").json() == {"available": False}
    monkeypatch.setenv("NARDOPAY_ENABLED", "true")
    db.execute("update public.nardopay_rates set set_at = now() - interval '30 days'")
    assert client.get("/api/v1/coins/purchase/nardopay/availability").json() == {"available": False}


def test_the_scheduled_loop_reconciles_and_survives_errors(env, monkeypatch):
    import threading
    client, db, calls, st = env
    seen, stop = [], threading.Event()

    def fake_reconcile():
        seen.append(1)
        if len(seen) == 1:
            raise RuntimeError("database blip")           # an error must not stop the loop
        stop.set()
        return {"finished": 0, "skipped": [], "expired": 0, "purged": 0}
    monkeypatch.setattr(np_, "reconcile", fake_reconcile)
    np_.reconcile_forever(interval_seconds=0, stop=stop)
    assert len(seen) == 2


def test_the_schedule_does_nothing_until_nardopay_is_set_up(env, monkeypatch):
    import threading
    client, db, calls, st = env
    for k in ("NARDOPAY_ENABLED", "NARDOPAY_WEBHOOK_SECRET"):
        monkeypatch.delenv(k)
    assert np_.reconciler_should_run() is False
    seen, stop = [], threading.Event()
    monkeypatch.setattr(np_, "reconcile", lambda: seen.append(1))
    threading.Timer(0.05, stop.set).start()
    np_.reconcile_forever(interval_seconds=0.01, stop=stop)
    assert seen == []
    monkeypatch.setenv("NARDOPAY_RECONCILE", "off")
    assert np_.start_reconciler() is False


# ====================================================================== test deliveries are never credited

def event_status(db):
    return db.execute("select status, detail from public.nardopay_webhook_events order by received_at desc limit 1").fetchone()


@pytest.mark.parametrize("flag", [True, "true", "TRUE", 1])
def test_a_test_payment_completed_with_a_real_order_amount_and_currency_credits_nothing(env, flag):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    body = payload(oid, tx="tx_test", test=flag)          # valid order, exact amount, RWF, valid signature
    r = deliver(client, body)
    assert r.status_code == 200 and r.json()["result"] == "ignored" and r.json()["test"] is True
    assert balance(db, u) == 100 and credits(db, "tx_test") == 0
    assert order_row(db, oid)["status"] == "pending" and order_row(db, oid)["transaction_id"] is None
    assert event_status(db) == ("ignored", "test event: never credited")
    # Sent again: still ignored, still one row, still nothing credited.
    again = deliver(client, body)
    assert again.status_code == 200 and again.json()["result"] == "ignored"
    assert db.execute("select count(*) from public.nardopay_webhook_events").fetchone()[0] == 1
    assert balance(db, u) == 100
    # Reconciliation never finishes it either.
    client.post("/api/v1/admin/nardopay/reconcile", headers=ADMIN)
    assert balance(db, u) == 100 and credits(db, "tx_test") == 0


def test_webhook_test_events_are_ignored_at_once(env):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    r = deliver(client, payload(oid, tx="tx_wt", event="webhook.test"))       # no test flag, event type alone
    assert r.status_code == 200 and r.json()["result"] == "ignored"
    assert balance(db, u) == 100 and event_status(db)[0] == "ignored"


def test_nardopays_own_test_webhook_is_ignored_even_before_the_fields_are_set(env, monkeypatch):
    """The exact shape NardoPay sent on 25 Nov 2026, with the field settings still empty (as in production now)."""
    client, db, calls, st = env
    for k in FIELDS:
        monkeypatch.delenv(k)
    body = {"test": True, "event": "webhook.test", "amount": 10, "status": "completed", "currency": "USD",
            "metadata": {}, "reference": "TEST-WEBHOOK", "payer_name": "Test Customer",
            "payer_email": "customer@example.com", "transaction_id": "00000000-0000-0000-0000-000000000000"}
    r = deliver(client, body)
    assert r.status_code == 200 and r.json()["result"] == "ignored"             # not left at "received"
    assert event_status(db) == ("ignored", "test event: never credited")
    assert db.execute("select count(*) from public.transactions where external_ref like 'nardopay:%'").fetchone()[0] == 0


def test_a_test_event_stored_earlier_as_received_is_ignored_when_reprocessed(env, monkeypatch):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    body = payload(oid, tx="tx_old", test=True)
    raw = json.dumps(body, separators=(",", ":"))
    # As the production row looks now: stored before this fix, status received.
    db.execute("insert into public.nardopay_webhook_events (body_sha256, signature_ok, raw_body) values (%s, true, %s)",
               (hashlib.sha256(raw.encode()).hexdigest(), raw))
    out = client.post("/api/v1/admin/nardopay/reprocess", headers=ADMIN).json()
    assert out["results"][0]["result"] == "ignored"
    assert balance(db, u) == 100 and credits(db, "tx_old") == 0 and order_row(db, oid)["status"] == "pending"
    assert event_status(db)[0] == "ignored"


@pytest.mark.parametrize("flag", [False, "false", 0, None])
def test_a_real_payment_marked_test_false_is_still_credited_once(env, flag):
    client, db, calls, st = env
    u = make_user(db)
    oid = buy(client, u)["order_id"]
    extra = {} if flag is None else {"test": flag}
    r = deliver(client, payload(oid, tx="tx_real", **extra))
    assert r.status_code == 200 and r.json()["result"] == "processed"
    assert balance(db, u) == 200 and credits(db, "tx_real") == 1
    assert order_row(db, oid)["status"] == "credited"


def test_is_test_event():
    assert np_.is_test_event({"test": True}) and np_.is_test_event({"event": "webhook.test"})
    assert np_.is_test_event({"test": "yes"}) and np_.is_test_event({"test": 1})
    assert not np_.is_test_event({"test": False}) and not np_.is_test_event({"test": 0})
    assert not np_.is_test_event({"event": "payment.completed"}) and not np_.is_test_event(None)
    assert not np_.is_test_event({"test": "no"}) and not np_.is_test_event([])


# ====================================================================== webhook_url and the admin test link

def test_every_link_tells_nardopay_where_to_report_the_payment(env):
    client, db, calls, st = env
    buy(client, make_user(db))
    assert calls[-1]["body"]["webhook_url"] == HOOK
    assert calls[-1]["body"]["metadata"]["order_id"]


@pytest.mark.parametrize("value", [None, "http://insecure.example/hook", "not a url"])
def test_without_a_valid_webhook_url_no_link_is_created(env, monkeypatch, value):
    client, db, calls, st = env
    if value is None:
        monkeypatch.delenv("NARDOPAY_WEBHOOK_URL")
    else:
        monkeypatch.setenv("NARDOPAY_WEBHOOK_URL", value)
    r = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(make_user(db)), json={"package_id": "starter"})
    assert r.status_code == 503 and calls == []
    assert db.execute("select count(*) from public.nardopay_orders").fetchone()[0] == 0
    assert client.get("/api/v1/coins/purchase/nardopay/availability").json() == {"available": False}


def test_the_admin_test_link_works_while_checkout_is_off_and_belongs_to_no_order(env, monkeypatch):
    client, db, calls, st = env
    for k in ("NARDOPAY_ENABLED", "NARDOPAY_LEDGER_VERIFIED", "NARDOPAY_LINK_URL_FIELD"):
        monkeypatch.delenv(k)
    st["response"] = lambda body: FakeResponse(200, {"ok": True, "data": {"payment_url": "https://pay.example/l/xyz",
                                                                          "id": "lnk_1"}})
    assert client.post("/api/v1/admin/nardopay/test-link", json={}).status_code == 403          # admin only
    r = client.post("/api/v1/admin/nardopay/test-link", headers=ADMIN, json={"amount_rwf": 100})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["link"] == "https://pay.example/l/xyz" and out["link_field_setting"] is None
    assert out["links_found"] == [{"field": "data.payment_url", "url": "https://pay.example/l/xyz"}]
    sent = calls[-1]["body"]
    assert sent["webhook_url"] == HOOK and "order_id" not in sent["metadata"] and sent["amount"] == 100
    assert db.execute("select count(*) from public.nardopay_orders").fetchone()[0] == 0           # no VIYO order


def test_paying_the_test_link_never_credits_coins(env):
    client, db, calls, st = env
    u = make_user(db)
    buy(client, u)                                       # a real pending order exists for the same amount
    body = {"x_event": "payment.completed", "x_meta": {"viyo_test": "admin"}, "x_tx": "tx_link", "x_amount": 1396,
            "x_currency": "RWF"}
    r = deliver(client, body)
    assert r.status_code == 200 and r.json()["result"] == "ignored"
    assert balance(db, u) == 100 and credits(db, "tx_link") == 0
    assert db.execute("select detail from public.nardopay_webhook_events").fetchone()[0] == "no order for this payment's order id"


def test_the_test_link_needs_its_settings_and_reports_a_bad_answer(env, monkeypatch):
    client, db, calls, st = env
    monkeypatch.delenv("NARDOPAY_WEBHOOK_URL")
    r = client.post("/api/v1/admin/nardopay/test-link", headers=ADMIN, json={})
    assert r.status_code == 503 and "NARDOPAY_WEBHOOK_URL" in r.json()["detail"]
    monkeypatch.setenv("NARDOPAY_WEBHOOK_URL", HOOK)
    st["response"] = lambda body: FakeResponse(400, {"error": "bad amount"})
    r = client.post("/api/v1/admin/nardopay/test-link", headers=ADMIN, json={})
    assert r.status_code == 502 and r.json()["detail"]["response_fields"] == ["error"]
    assert client.post("/api/v1/admin/nardopay/test-link", headers=ADMIN, json={"amount_rwf": 50}).status_code == 422
    assert client.post("/api/v1/admin/nardopay/test-link", headers=ADMIN,
                       json={"amount_rwf": 100, "order_id": "x"}).status_code == 422


def test_https_fields():
    data = {"a": "https://x.example/1", "b": {"c": ["http://no", "https://x.example/2"]}, "d": 5}
    assert np_.https_fields(data) == [{"field": "a", "url": "https://x.example/1"},
                                      {"field": "b.c.1", "url": "https://x.example/2"}]
