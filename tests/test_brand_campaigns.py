"""Brand campaigns (brand_campaigns.py + brand_campaigns.sql) on a real Postgres, NardoPay HTTP mocked and webhooks
signed like NardoPay's real ones. The business pays upfront; creators at Rising+ apply; the creator gets 80% of the
fee, held, once; unused money comes back as campaign credit, once."""
import pytest

import brand_campaigns as bc
import wallet
from conftest import ROOT, connect, make_user
from test_nardopay_payments import FakeResponse, H, deliver, env, real, real_payment, schema  # noqa: F401


@pytest.fixture(scope="module")
def bc_schema(schema):
    conn = connect()
    conn.execute("alter table public.posts add column if not exists caption text; "
                 "alter table public.posts add column if not exists created_at timestamptz default now(); "
                 "alter table public.posts add column if not exists is_private boolean default false; "
                 "alter table public.posts add column if not exists is_archived boolean default false; "
                 "alter table public.posts add column if not exists post_type text")
    for f in ("creator_payouts_safety.sql", "creator_earnings_ledger.sql", "creator_donations.sql",
              "creator_donations_usd.sql", "brand_campaigns.sql", "brand_campaigns.sql"):     # twice: safe to rerun
        conn.execute((ROOT / f).read_text())
    conn.close()


@pytest.fixture
def b(bc_schema, real, monkeypatch):
    client, db, calls, st = real
    db.execute("truncate public.brand_credit_ledger, public.brand_campaign_applications, public.brand_campaigns, "
               "public.creator_earnings cascade")
    codes = iter([f"vbrand{i:010d}" for i in range(1, 50)])
    st["response"] = lambda body: FakeResponse(200, {"url": f"https://nardopay.com/p/{next(codes)}?pay=1"})
    import notify as notify_mod
    sent = []
    monkeypatch.setattr(notify_mod, "notify", lambda *a, **k: sent.append((a, k)))
    monkeypatch.setattr(notify_mod, "display_name", lambda uid: "Amani")
    levels = {}
    monkeypatch.setattr(bc, "_level", lambda uid: {"rank": levels.get(str(uid), 2),
                                                   "name": "Rising" if levels.get(str(uid), 2) >= 2 else "New"})
    import creator_levels
    monkeypatch.setattr(creator_levels, "hold_days_for", lambda db_, c: 14)
    client.app.include_router(bc.router)
    return client, db, calls, sent, levels


BRIEF = {"brand_name": "Inyange", "title": "Summer juice", "brief": "Show our new mango juice in a fun short video.",
         "deliverable": "One 30-60s video", "fee_usd_cents": 5000, "slots": 2, "days": 14}


def create(client, owner, **over):
    r = client.post("/api/v1/brand-campaigns", headers=H(owner), json={**BRIEF, **over})
    assert r.status_code == 200, r.text
    return r.json()


def code_of(db, cid):
    return db.execute("select link_code from public.brand_campaigns where id = %s", (cid,)).fetchone()[0]


def pay(client, db, cid, amount, tx="tx_brand_1"):
    return deliver(client, real_payment(code_of(db, cid), amount=amount, tx=tx)).json()["result"]


def post(db, user):
    return db.execute("insert into public.posts (user_id, caption, post_type) values (%s, 'my video', 'video') "
                      "returning id", (user,)).fetchone()[0]


def credit(client, owner):
    return client.get("/api/v1/brand-campaigns/credit", headers=H(owner)).json()["credit_usd"]


def open_paid(client, db, owner):
    c = create(client, owner)
    assert pay(client, db, c["id"], 141000) == "processed"           # $100 at 1,410 RWF/USD
    return c


def test_the_business_pays_the_whole_budget_in_dollars_converted_by_viyo(b):
    client, db, calls, sent, levels = b
    owner = make_user(db)
    for bad in ({**BRIEF, "fee_usd_cents": 499}, {**BRIEF, "slots": 51}, {**BRIEF, "price_rwf": 1}, {**BRIEF, "days": 1}):
        assert client.post("/api/v1/brand-campaigns", headers=H(owner), json=bad).status_code == 422
    c = create(client, owner)
    assert c["status"] == "awaiting_payment" and c["budget_usd"] == 100 and c["pay_usd"] == 100
    assert c["creator_gets_usd"] == 40 and c["payment_link"].startswith("https://nardopay.com/p/")
    body = calls[-1]["body"]
    assert body["amount"] == 141000 and body["amount_mode"] == "fixed" and body["redirect_url"] == "https://viyo.site/app"
    assert client.get(f"/api/v1/brand-campaigns/{c['id']}", headers=H(owner)).json()["status"] == "awaiting_payment"
    # Only the signed webhook opens it (never a redirect); the fee on top is allowed, less is not.
    assert pay(client, db, c["id"], 141000 - 1, "tx_short") == "mismatch"
    assert pay(client, db, c["id"], 164970) == "processed"                    # + 17% fee
    assert client.get(f"/api/v1/brand-campaigns/{c['id']}", headers=H(owner)).json()["status"] == "open"
    assert pay(client, db, c["id"], 164970) == "processed"                    # the same webhook again: nothing new
    assert pay(client, db, c["id"], 164970, "tx_other") == "duplicate_payment"


def test_the_whole_deal_creator_gets_80_percent_held_once(b):
    client, db, calls, sent, levels = b
    owner, creator = make_user(db), make_user(db)
    c = open_paid(client, db, owner)
    listed = client.get("/api/v1/brand-campaigns/creator/open", headers=H(creator)).json()
    assert listed["eligible"] is True and listed["campaigns"][0]["creator_gets_usd"] == 40
    a = client.post(f"/api/v1/brand-campaigns/{c['id']}/apply", headers=H(creator), json={"pitch": "I love mango"}).json()
    assert client.post(f"/api/v1/brand-campaigns/{c['id']}/apply", headers=H(creator), json={}).status_code == 409
    # Only the business decides; only the creator submits.
    assert client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/accept", headers=H(creator)).status_code == 403
    assert client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/accept", headers=H(owner)).json()["status"] == "accepted"
    other = make_user(db)
    assert client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/submit", headers=H(creator),
                       json={"post_id": str(post(db, other))}).status_code == 400          # someone else's post
    pid = post(db, creator)
    s = client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/submit", headers=H(creator),
                    json={"post_id": str(pid)}).json()
    assert s["status"] == "submitted" and "auto_approves_at" in s
    assert db.execute("select sponsored_by from public.posts where id = %s", (pid,)).fetchone()[0] == "Inyange"
    ok = client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/approve", headers=H(owner)).json()
    assert ok["status"] == "approved"
    rows = db.execute("select kind, coins, status, earning_key from public.creator_earnings where creator_id = %s",
                      (creator,)).fetchall()
    assert rows == [("brand", int(40 / wallet.COIN_USD), "pending", f"brand:{a['id']}")]     # $40 = 80% of $50
    assert client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/approve", headers=H(owner)).status_code == 409
    assert any("You earned $40.00" in x[0][2] for x in sent)


def test_only_rising_and_above_can_apply(b):
    client, db, calls, sent, levels = b
    owner, newbie = make_user(db), make_user(db)
    c = open_paid(client, db, owner)
    levels[str(newbie)] = 1
    assert client.get("/api/v1/brand-campaigns/creator/open", headers=H(newbie)).json()["eligible"] is False
    r = client.post(f"/api/v1/brand-campaigns/{c['id']}/apply", headers=H(newbie), json={})
    assert r.status_code == 403 and "Rising" in r.json()["detail"]
    assert client.post(f"/api/v1/brand-campaigns/{c['id']}/apply", headers=H(owner), json={}).status_code == 400


def test_slots_are_never_exceeded(b):
    client, db, calls, sent, levels = b
    owner = make_user(db)
    c = open_paid(client, db, owner)
    apps = [client.post(f"/api/v1/brand-campaigns/{c['id']}/apply", headers=H(make_user(db)), json={}).json()
            for _ in range(3)]
    codes = [client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/accept", headers=H(owner)).status_code
             for a in apps]
    assert codes == [200, 200, 409]


def test_a_rejection_needs_a_reason_and_removes_the_sponsored_label(b):
    client, db, calls, sent, levels = b
    owner, creator = make_user(db), make_user(db)
    c = open_paid(client, db, owner)
    a = client.post(f"/api/v1/brand-campaigns/{c['id']}/apply", headers=H(creator), json={}).json()
    client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/accept", headers=H(owner))
    pid = post(db, creator)
    client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/submit", headers=H(creator), json={"post_id": str(pid)})
    assert client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/reject", headers=H(owner),
                       json={"note": ""}).status_code == 400
    r = client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/reject", headers=H(owner),
                    json={"note": "The juice isn't shown"})
    assert r.json()["status"] == "rejected" and r.json()["decision_note"] == "The juice isn't shown"
    assert db.execute("select sponsored_by from public.posts where id = %s", (pid,)).fetchone()[0] is None
    assert db.execute("select count(*) from public.creator_earnings").fetchone()[0] == 0


def test_unused_money_comes_back_as_credit_once_and_pays_the_next_campaign(b):
    client, db, calls, sent, levels = b
    owner, creator = make_user(db), make_user(db)
    c = open_paid(client, db, owner)                                       # $100: 2 x $50
    a = client.post(f"/api/v1/brand-campaigns/{c['id']}/apply", headers=H(creator), json={}).json()
    client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/accept", headers=H(owner))
    out = client.post(f"/api/v1/brand-campaigns/{c['id']}/close", headers=H(owner)).json()
    assert out["credited_usd"] == 50 and credit(client, owner) == 50      # the accepted creator's $50 stays reserved
    client.post(f"/api/v1/brand-campaigns/{c['id']}/close", headers=H(owner))
    assert credit(client, owner) == 50                                     # once
    # The creator can still deliver until the deadline; the business rejects -> that $50 comes back too.
    pid = post(db, creator)
    client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/submit", headers=H(creator), json={"post_id": str(pid)})
    client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/reject", headers=H(owner), json={"note": "Off brief"})
    assert credit(client, owner) == 100
    n = len(calls)
    c2 = create(client, owner, fee_usd_cents=2500, slots=2)                # $50: paid fully from credit
    assert c2["status"] == "open" and c2["credit_used_usd"] == 50 and c2["pay_usd"] == 0 and len(calls) == n
    assert credit(client, owner) == 50
    c3 = create(client, owner, fee_usd_cents=5000, slots=2)                # $100: $50 credit + $50 by card
    assert c3["status"] == "awaiting_payment" and c3["credit_used_usd"] == 50 and c3["pay_usd"] == 50
    assert calls[-1]["body"]["amount"] == 70500 and credit(client, owner) == 0
    client.post(f"/api/v1/brand-campaigns/{c3['id']}/close", headers=H(owner))          # cancel unpaid
    assert credit(client, owner) == 50


def test_maintenance_auto_approves_after_7_days_and_expires_late_creators(b):
    client, db, calls, sent, levels = b
    owner, on_time, late = make_user(db), make_user(db), make_user(db)
    c = open_paid(client, db, owner)
    a1 = client.post(f"/api/v1/brand-campaigns/{c['id']}/apply", headers=H(on_time), json={}).json()
    a2 = client.post(f"/api/v1/brand-campaigns/{c['id']}/apply", headers=H(late), json={}).json()
    for a in (a1, a2):
        client.post(f"/api/v1/brand-campaigns/applications/{a['id']}/accept", headers=H(owner))
    client.post(f"/api/v1/brand-campaigns/applications/{a1['id']}/submit", headers=H(on_time),
                json={"post_id": str(post(db, on_time))})
    assert bc.maintain()["auto_approved"] == 0                             # not 7 days yet
    db.execute("update public.brand_campaign_applications set submitted_at = now() - interval '8 days' where id = %s",
               (a1["id"],))
    db.execute("update public.brand_campaigns set deadline = now() - interval '1 minute' where id = %s", (c["id"],))
    out = bc.maintain()
    assert out["auto_approved"] == 1 and out["expired"] == 1
    st = dict(db.execute("select id::text, status from public.brand_campaign_applications").fetchall())
    assert st[a1["id"]] == "approved" and st[a2["id"]] == "expired"
    assert credit(client, owner) == 50                                     # the late creator's slot
    assert bc.maintain() == {"auto_approved": 0, "expired": 0, "abandoned": 0}


def test_unpaid_campaigns_are_cancelled_after_a_day_and_a_late_payment_becomes_credit(b):
    client, db, calls, sent, levels = b
    owner = make_user(db)
    c = create(client, owner)
    db.execute("update public.brand_campaigns set created_at = now() - interval '25 hours' where id = %s", (c["id"],))
    assert bc.maintain()["abandoned"] == 1
    assert credit(client, owner) == 0
    assert pay(client, db, c["id"], 141000, "tx_late") == "processed"
    assert credit(client, owner) == 100
    assert client.get(f"/api/v1/brand-campaigns/{c['id']}", headers=H(owner)).json()["status"] == "cancelled"


def test_nobody_else_sees_or_changes_a_campaign(b):
    client, db, calls, sent, levels = b
    owner, stranger = make_user(db), make_user(db)
    c = open_paid(client, db, owner)
    assert client.get(f"/api/v1/brand-campaigns/{c['id']}", headers=H(stranger)).status_code == 404
    assert client.post(f"/api/v1/brand-campaigns/{c['id']}/close", headers=H(stranger)).status_code == 404


def test_only_the_server_can_call_the_functions(b):
    client, db, calls, sent, levels = b
    db.execute("begin")
    try:
        db.execute("set local role authenticated")
        with pytest.raises(Exception):
            db.execute("select public.brand_credit_balance(gen_random_uuid())")
    finally:
        db.execute("rollback")
