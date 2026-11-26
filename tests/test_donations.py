"""Fan donations through NardoPay (donations.py + creator_donations.sql): real Postgres, NardoPay HTTP mocked, webhooks
signed with a test secret and shaped like NardoPay's real payment webhook. Real money, never coins."""
from decimal import Decimal

import pytest
from fastapi import FastAPI

import donations
import nardopay_payments as np_
import wallet
from conftest import ROOT, connect, make_user
from test_nardopay_payments import FakeResponse, H, balance, deliver, env, real, real_payment, schema  # noqa: F401


@pytest.fixture(scope="module")
def don_schema(schema):
    conn = connect()
    conn.execute((ROOT / "creator_payouts_safety.sql").read_text())
    conn.execute((ROOT / "creator_earnings_ledger.sql").read_text())
    for _ in range(2):                                               # safe to run again
        conn.execute((ROOT / "creator_donations.sql").read_text())
    conn.close()


@pytest.fixture
def dv(don_schema, real, monkeypatch):
    client, db, calls, st = real
    db.execute("truncate public.creator_donations, public.creator_earnings cascade")
    codes = iter([f"vdon{i:012d}" for i in range(1, 50)])
    st["response"] = lambda body: FakeResponse(200, {"url": f"https://nardopay.com/d/{next(codes)}?pay=1"})
    monkeypatch.setattr(np_, "LINK_CODE_IN_URL", __import__("re").compile(r"/[psd]/([A-Za-z0-9_-]{6,64})(?:[/?#]|$)"))
    sent = []
    import notify as notify_mod
    monkeypatch.setattr(notify_mod, "notify", lambda *a, **k: sent.append((a, k)))
    client.app.include_router(donations.router)
    return client, db, calls, st, sent


def donate(client, donor, creator, message="Love your drama!"):
    r = client.post(f"/api/v1/creators/{creator}/donate", headers=H(donor), json={"message": message})
    assert r.status_code == 200, r.text
    return r.json()


def code_of(db, did):
    return db.execute("select link_code from public.creator_donations where id = %s", (did,)).fetchone()[0]


def donation(db, did):
    cur = db.execute("select * from public.creator_donations where id = %s", (did,))
    return dict(zip([c.name for c in cur.description], cur.fetchone()))


def earnings(db, creator):
    return db.execute("select kind, coins, status, earning_key from public.creator_earnings where creator_id = %s",
                      (creator,)).fetchall()


def paid(code, amount, tx, **extra):
    return real_payment(code, amount=amount, tx=tx, link_type="donation", **extra)


def test_a_donation_gets_its_own_nardopay_donation_link_and_no_client_amount(dv):
    client, db, calls, st, sent = dv
    donor, creator = make_user(db), make_user(db)
    assert client.post(f"/api/v1/creators/{creator}/donate", headers=H(donor),
                       json={"amount": 100000}).status_code == 422
    d = donate(client, donor, creator)
    assert d["status"] == "pending" and d["payment_link"].startswith("https://nardopay.com/d/")
    body = calls[-1]["body"]
    assert body["link_type"] == "donation" and body["title"].startswith("Support ")
    assert body["metadata"] == {"donation_id": d["donation_id"]} and body["webhook_url"]
    assert body["redirect_url"] == "https://viyo.site/app" and "amount" not in body
    row = donation(db, d["donation_id"])
    assert float(row["usd_rwf_rate"]) == 1410 and float(row["creator_share_rate"]) == 0.8


def test_cant_donate_to_yourself_or_to_nobody(dv):
    client, db, calls, st, sent = dv
    u = make_user(db)
    assert client.post(f"/api/v1/creators/{u}/donate", headers=H(u), json={}).status_code == 400
    assert client.post("/api/v1/creators/00000000-0000-0000-0000-000000000000/donate", headers=H(u),
                       json={}).status_code == 404
    assert calls == []


def test_a_verified_donation_gives_the_creator_80_percent_held_14_days_once(dv):
    client, db, calls, st, sent = dv
    donor, creator = make_user(db), make_user(db)
    coins_before = balance(db, donor)
    d = donate(client, donor, creator)
    r = deliver(client, paid(code_of(db, d["donation_id"]), 14100, "tx_don1"))
    assert r.status_code == 200 and r.json()["result"] == "processed", r.text
    row = donation(db, d["donation_id"])
    # 14,100 RWF -> 80% = 11,280 RWF -> / 1410 = $8.00 -> / $0.006 = 1,333 coins (rounded down)
    assert row["status"] == "paid" and row["amount_rwf"] == Decimal("14100.00")
    assert row["share_rwf"] == Decimal("11280.00") and row["share_usd"] == Decimal("8.0000") and row["share_coins"] == 1333
    assert earnings(db, creator) == [("donation", 1333, "pending", f"donation:{d['donation_id']}")]
    assert balance(db, donor) == coins_before                       # never coins
    assert sent and sent[0][0][1] == "donation" and "$8.00" in sent[0][0][2]
    # NardoPay retries: nothing more.
    assert deliver(client, paid(code_of(db, d["donation_id"]), 14100, "tx_don1")).json()["duplicate"] is True
    assert deliver(client, paid(code_of(db, d["donation_id"]), 14100, "tx_don1", retry=1)).status_code == 200
    assert len(earnings(db, creator)) == 1 and len(sent) == 1


def test_a_second_payment_on_a_paid_donation_link_is_not_counted_again(dv):
    client, db, calls, st, sent = dv
    donor, creator = make_user(db), make_user(db)
    d = donate(client, donor, creator)
    code = code_of(db, d["donation_id"])
    deliver(client, paid(code, 5000, "tx_a"))
    assert deliver(client, paid(code, 5000, "tx_b")).json()["result"] == "duplicate_payment"
    assert len(earnings(db, creator)) == 1


@pytest.mark.parametrize("change", [{"currency": "USD"}, {"amount": 0}, {"test": True}, {"event": "payment.failed"}])
def test_wrong_currency_no_amount_test_or_failed_records_nothing(dv, change):
    client, db, calls, st, sent = dv
    donor, creator = make_user(db), make_user(db)
    d = donate(client, donor, creator)
    body = paid(code_of(db, d["donation_id"]), 5000, "tx_x")
    body.update(change)
    assert deliver(client, body).status_code == 200
    assert earnings(db, creator) == [] and donation(db, d["donation_id"])["status"] in ("pending", "failed")


def test_the_redirect_back_records_nothing_and_only_the_donor_sees_the_status(dv):
    client, db, calls, st, sent = dv
    donor, creator, other = make_user(db), make_user(db), make_user(db)
    d = donate(client, donor, creator)
    assert client.get(f"/api/v1/donations/{d['donation_id']}", headers=H(donor)).json()["status"] == "pending"
    assert client.get(f"/api/v1/donations/{d['donation_id']}", headers=H(other)).status_code == 404
    assert earnings(db, creator) == []


def test_a_refunded_donation_is_reversed_once(dv):
    client, db, calls, st, sent = dv
    donor, creator = make_user(db), make_user(db)
    d = donate(client, donor, creator)
    deliver(client, paid(code_of(db, d["donation_id"]), 14100, "tx_r"))
    eid = db.execute("select id from public.creator_earnings where creator_id = %s", (creator,)).fetchone()[0]
    r = db.execute("select public.viyo_reverse_earning(%s, 'refunded donation')", (eid,)).fetchone()[0]
    assert r["ok"] and r["status"] == "reversed"
    assert db.execute("select public.viyo_reverse_earning(%s, 'again')", (eid,)).fetchone()[0]["changed"] is False


def test_the_creator_sees_paid_donations_with_messages(dv):
    client, db, calls, st, sent = dv
    donor, creator = make_user(db), make_user(db)
    d = donate(client, donor, creator, "Keep going!")
    donate(client, donor, creator, "never paid")
    deliver(client, paid(code_of(db, d["donation_id"]), 14100, "tx_list"))
    out = client.get("/api/v1/creator/donations", headers=H(creator)).json()
    assert out["creator_share"] == 0.8
    assert [(x["message"], x["share_usd"]) for x in out["donations"]] == [("Keep going!", 8.0)]


def test_switched_off_means_no_donations(dv, monkeypatch):
    client, db, calls, st, sent = dv
    monkeypatch.setenv("NARDOPAY_ENABLED", "false")
    assert client.get("/api/v1/donations/availability").json()["available"] is False
    r = client.post(f"/api/v1/creators/{make_user(db)}/donate", headers=H(make_user(db)), json={})
    assert r.status_code == 503 and calls == []


def test_coin_purchases_and_withdrawals_are_unchanged(dv):
    client, db, calls, st, sent = dv
    assert wallet.MIN_WITHDRAWAL_USD == 10.0 and wallet.COIN_USD == 0.006
