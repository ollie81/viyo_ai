"""Ad revenue sharing (ad_revenue.py + ad_revenue.sql) on a real Postgres: half of the real monthly AdMob totals to
creators, shared by the ads on their content (verified unlock ads + in-feed ads next to any post), once per month."""
import datetime as dt
import uuid

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

import ad_revenue as ar
import wallet
from conftest import ROOT, connect, make_user
from studio_store import PgStore

ADMIN = {"X-Admin-Key": "admin-secret"}


@pytest.fixture(scope="module")
def ar_schema(migrated):
    conn = connect()
    conn.execute("alter table public.posts add column if not exists caption text")
    for f in ("creator_payouts_schema.sql", "creator_payouts_safety.sql", "nardopay_orders.sql",
              "creator_earnings_ledger.sql", "creator_donations.sql", "brand_campaigns.sql",
              "ad_revenue.sql", "ad_revenue.sql"):                                     # twice: safe to rerun
        conn.execute((ROOT / f).read_text())
    conn.execute("create table if not exists public.ad_episode_tickets (id bigserial primary key, user_id uuid, "
                 "delta int not null, reason text not null, external_ref text unique, "
                 "created_at timestamptz not null default now())")
    conn.close()


@pytest.fixture
def a(ar_schema, db, monkeypatch):
    db.execute("truncate public.ad_impressions, public.ad_revenue_months, public.ad_revenue_payouts, "
               "public.ad_episode_tickets, public.creator_earnings cascade")
    import nardopay_payments as np_
    monkeypatch.setattr(np_, "_store", PgStore(db))
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    import creator_levels
    monkeypatch.setattr(creator_levels, "hold_days_for", lambda s, c: 7 if str(c) == pro[0] else 14)
    pro = [""]
    import notify as notify_mod
    sent = []
    monkeypatch.setattr(notify_mod, "notify", lambda *x, **k: sent.append(x))
    app = FastAPI()
    app.include_router(ar.router)

    async def user(authorization: str = Header(None)):
        if not authorization:
            raise HTTPException(status_code=401)
        return authorization.removeprefix("Bearer ")
    app.dependency_overrides[ar._user] = user
    return TestClient(app), db, sent, pro


def H(u):
    return {"Authorization": f"Bearer {u}"}


def post(db, creator):
    return db.execute("insert into public.posts (user_id) values (%s) returning id", (creator,)).fetchone()[0]


def last_month():
    t = dt.datetime.now(dt.timezone.utc).date().replace(day=1)
    return (t - dt.timedelta(days=1)).replace(day=1)


def unlock(db, viewer, post_id, when):
    db.execute("insert into public.ad_episode_tickets (user_id, delta, reason, external_ref, created_at) "
               "values (%s, -2, 'unlock', %s, %s)", (viewer, f"unlock:{viewer}:{post_id}", when))


def shown(db, viewer, creator, post_id, day):
    db.execute("insert into public.ad_impressions (viewer_id, creator_id, post_id, day) values (%s, %s, %s, %s)",
               (viewer, creator, post_id, day))


def test_in_feed_ads_count_once_per_viewer_per_post_per_day_never_on_your_own_posts(a):
    client, db, sent, pro = a
    creator, viewer = make_user(db), make_user(db)
    p = post(db, creator)
    assert client.post("/api/v1/ads/impression", headers=H(viewer), json={"post_id": str(p)}).json() == {"counted": True}
    assert client.post("/api/v1/ads/impression", headers=H(viewer), json={"post_id": str(p)}).json() == {"counted": False}
    assert client.post("/api/v1/ads/impression", headers=H(creator), json={"post_id": str(p)}).json() == {"counted": False}
    assert client.post("/api/v1/ads/impression", headers=H(viewer), json={"post_id": "nope"}).json() == {"counted": False}
    assert client.post("/api/v1/ads/impression", json={"post_id": str(p)}).status_code == 401
    assert db.execute("select count(*) from public.ad_impressions").fetchone()[0] == 1


def test_a_viewer_counts_at_most_40_in_feed_ads_a_day(a):
    client, db, sent, pro = a
    creator, viewer = make_user(db), make_user(db)
    counted = [client.post("/api/v1/ads/impression", headers=H(viewer), json={"post_id": str(post(db, creator))})
               .json()["counted"] for _ in range(42)]
    assert counted.count(True) == 40


def test_closing_a_month_shares_half_of_each_real_total_by_ads_on_each_creators_content(a):
    client, db, sent, pro = a
    c1, c2, v = make_user(db), make_user(db), make_user(db)
    pro[0] = str(c2)
    m = last_month()
    p1, p2 = post(db, c1), post(db, c2)
    for i in range(3):                                               # in-feed: c1 3 ads, c2 1 ad
        shown(db, make_user(db), c1, p1, m + dt.timedelta(days=i))
    shown(db, v, c2, p2, m)
    unlock(db, v, p2, m + dt.timedelta(days=2))                       # unlock ads: c2 2 verified ads (1 episode)
    unlock(db, c1, p1, m)                                            # a creator unlocking their own: left out
    shown(db, v, c1, p1, m - dt.timedelta(days=3))                    # the month before: left out
    view = client.get(f"/api/v1/admin/ad-revenue/{m:%Y-%m}", headers=ADMIN).json()
    assert view["feed_ads"] == 4 and view["unlock_ads"] == 2 and view["can_close"] is True
    out = client.post(f"/api/v1/admin/ad-revenue/{m:%Y-%m}/close", headers=ADMIN,
                      json={"feed_usd": 40, "unlock_usd": 10}).json()
    assert out["ok"] is True and out["creators"] == 2
    # feed pool $20 (half of $40): c1 3/4 = $15, c2 1/4 = $5; unlock pool $5: all c2.
    rows = dict(db.execute("select creator_id::text, usd from public.ad_revenue_payouts").fetchall())
    assert float(rows[str(c1)]) == 15 and float(rows[str(c2)]) == 10
    earn = dict(db.execute("select creator_id::text, coins from public.creator_earnings where kind = 'ads'").fetchall())
    assert earn[str(c1)] == int(15 / wallet.COIN_USD) and earn[str(c2)] == int(10 / wallet.COIN_USD)
    holds = dict(db.execute("select creator_id::text, (available_at - now()) > interval '10 days' "
                            "from public.creator_earnings where kind = 'ads'").fetchall())
    assert holds[str(c1)] is True and holds[str(c2)] is False             # c2 is Pro: 7-day hold
    assert len(sent) == 2
    again = client.post(f"/api/v1/admin/ad-revenue/{m:%Y-%m}/close", headers=ADMIN, json={"feed_usd": 99, "unlock_usd": 0})
    assert again.status_code == 409
    assert db.execute("select count(*) from public.creator_earnings where kind = 'ads'").fetchone()[0] == 2


def test_the_current_month_cant_be_closed_and_creators_see_counts_but_no_dollars_yet(a):
    client, db, sent, pro = a
    c, v = make_user(db), make_user(db)
    p = post(db, c)
    client.post("/api/v1/ads/impression", headers=H(v), json={"post_id": str(p)})
    this = dt.datetime.now(dt.timezone.utc).date()
    r = client.post(f"/api/v1/admin/ad-revenue/{this:%Y-%m}/close", headers=ADMIN, json={"feed_usd": 1, "unlock_usd": 1})
    assert r.status_code == 409
    me = client.get("/api/v1/creator/ad-revenue", headers=H(c)).json()
    assert me["available"] is True and me["this_month"]["feed_ads"] == 1 and me["months"] == []
    assert me["creator_share"] == 0.5


def test_admin_only(a):
    client, db, sent, pro = a
    m = last_month()
    assert client.get(f"/api/v1/admin/ad-revenue/{m:%Y-%m}").status_code == 403
    assert client.post(f"/api/v1/admin/ad-revenue/{m:%Y-%m}/close", headers={"X-Admin-Key": "wrong"},
                       json={"feed_usd": 1, "unlock_usd": 1}).status_code == 403
    assert client.post(f"/api/v1/admin/ad-revenue/{m:%Y-%m}/close", headers=ADMIN,
                       json={"feed_usd": -1, "unlock_usd": 1}).status_code == 422


def test_a_month_with_no_ads_closes_with_nothing_paid(a):
    client, db, sent, pro = a
    m = last_month()
    out = client.post(f"/api/v1/admin/ad-revenue/{m:%Y-%m}/close", headers=ADMIN, json={"feed_usd": 5, "unlock_usd": 5}).json()
    assert out["creators"] == 0 and float(out["paid_usd"]) == 0


def test_only_the_server_can_call_the_functions(a):
    client, db, sent, pro = a
    db.execute("begin")
    try:
        db.execute("set local role authenticated")
        with pytest.raises(Exception):
            db.execute("select public.ad_revenue_counts(current_date, null)")
    finally:
        db.execute("rollback")
