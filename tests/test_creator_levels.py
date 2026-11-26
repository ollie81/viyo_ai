"""Creator levels (creator_levels.sql + creator_levels.py): New, Rising, Pro, Star from real records. Everyone earns
the same share; the only perk is a shorter hold for Pro and Star, never given on unreadable numbers or after a
refund in the last 90 days."""
import uuid

import pytest

import creator_levels as cl
import wallet
from conftest import ROOT, connect, make_user


def m(followers=0, viewers=0, days=0, usd=0.0, reversals=0):
    return {"followers": followers, "viewers_30d": viewers, "account_days": days,
            "earned_coins": round(usd / wallet.COIN_USD) + 1 if usd else 0, "reversals_90d": reversals}


def test_levels_are_climbed_with_every_requirement_met():
    assert cl.evaluate(m())["key"] == "new"
    assert cl.evaluate(m(followers=400, viewers=4999))["key"] == "new"
    assert cl.evaluate(m(followers=399, viewers=5000))["key"] == "new"
    assert cl.evaluate(m(followers=400, viewers=5000))["key"] == "rising"
    assert cl.evaluate(m(followers=1000, viewers=20000, days=60, usd=50))["key"] == "pro"
    assert cl.evaluate(m(followers=1000, viewers=20000, days=59, usd=50))["key"] == "rising"     # too new
    assert cl.evaluate(m(followers=1000, viewers=20000, days=60, usd=49))["key"] == "rising"     # not enough earned
    assert cl.evaluate(m(followers=10000, viewers=50000, days=90, usd=500))["key"] == "star"


def test_a_recent_refund_keeps_a_creator_below_pro():
    out = cl.evaluate(m(followers=20000, viewers=90000, days=400, usd=900, reversals=1))
    assert out["key"] == "rising" and out["hold_days"] == wallet.EARNINGS_HOLD_DAYS
    clean = [r for r in out["next"]["requirements"] if r["key"] == "clean"][0]
    assert clean["met"] is False


def test_only_the_hold_changes_and_everyone_earns_the_same_share():
    new, pro = cl.evaluate(m()), cl.evaluate(m(followers=1000, viewers=20000, days=60, usd=50))
    assert new["hold_days"] == wallet.EARNINGS_HOLD_DAYS and new["perks"] == []
    assert pro["hold_days"] == cl.SHORT_HOLD_DAYS == 7 and "7 days" in pro["perks"][0]
    assert new["next"]["key"] == "rising" and new["next"]["perk"] is None
    assert cl.evaluate(m(followers=400, viewers=5000))["next"]["perk"] is not None             # Pro brings the perk
    assert "same share" in pro["share_note"]
    assert cl.evaluate(m(followers=10000, viewers=50000, days=90, usd=500))["next"] is None


def test_unreadable_numbers_mean_the_normal_hold():
    class Broken:
        def rpc(self, *a):
            raise Exception("PGRST202 function not found")
    cl._cache.clear()
    assert cl.hold_days_for(Broken(), str(uuid.uuid4())) == wallet.EARNINGS_HOLD_DAYS
    assert cl.hold_days_for(None, str(uuid.uuid4())) == wallet.EARNINGS_HOLD_DAYS
    lvl = cl.level_for(Broken(), str(uuid.uuid4()))
    assert lvl["ready"] is False and lvl["key"] == "new"


def test_hold_is_cached_and_forgotten_after_a_refund():
    calls = []

    class Pro:
        def rpc(self, name, params):
            calls.append(name)
            return {"followers": 1000, "viewers_30d": 20000, "account_days": 60,
                    "earned_coins": 10000, "reversals_90d": 0}
    cl._cache.clear()
    c = str(uuid.uuid4())
    assert cl.hold_days_for(Pro(), c) == 7 and cl.hold_days_for(Pro(), c) == 7
    assert len(calls) == 1
    cl.forget(c)
    cl.hold_days_for(Pro(), c)
    assert len(calls) == 2


def test_an_unlock_is_credited_with_the_creators_hold(monkeypatch):
    sent = []

    class Admin:
        def rpc(self, name, params):
            sent.append((name, params))

            class R:
                data = {"ok": True}

                def execute(self_inner):
                    return self_inner
            return R()
    monkeypatch.setattr(cl, "hold_days_for", lambda db, c: 7)
    assert wallet.credit_earning(Admin(), "c", "p", "v", 30) is True
    assert sent[-1][1]["p_hold_days"] == 7
    monkeypatch.setattr(cl, "hold_days_for", lambda db, c: 14)
    wallet.credit_earning(Admin(), "c", "p2", "v", 30)
    assert sent[-1][1]["p_hold_days"] == 14


# ---------------------------------------------------------------- the SQL on a real Postgres

@pytest.fixture(scope="module")
def lv_schema(migrated):
    conn = connect()
    conn.execute("alter table auth.users add column if not exists created_at timestamptz default now(); "
                 "create table if not exists public.follows (follower_id uuid, following_id uuid)")
    conn.execute("create table if not exists public.series (id uuid primary key default gen_random_uuid(), "
                 "user_id uuid references auth.users(id) on delete cascade, title text, cover_url text); "
                 "alter table public.profiles add column if not exists username text; "
                 "alter table public.profiles add column if not exists display_name text")
    for f in ("intelligence.sql", "creator_payouts_schema.sql", "creator_payouts_safety.sql",
              "creator_earnings_ledger.sql", "creator_levels.sql", "creator_levels.sql"):   # twice: safe to rerun
        conn.execute((ROOT / f).read_text())
    conn.close()


@pytest.fixture
def d(lv_schema, db):
    db.execute("truncate public.viewer_events, public.creator_earnings, public.follows cascade")
    return db


class Rpc:
    def __init__(self, conn):
        self.conn = conn

    def rpc(self, name, params):
        args = ", ".join(f"{k} => %({k})s" for k in params)
        return self.conn.execute(f"select public.{name}({args})", params).fetchone()[0]


def test_metrics_come_from_real_records(d):
    c = make_user(d)
    d.execute("update auth.users set created_at = now() - interval '75 days' where id = %s", (c,))
    for _ in range(3):
        d.execute("insert into public.follows (follower_id, following_id) values (%s, %s)", (make_user(d), c))
    v1, v2 = make_user(d), make_user(d)
    for v, days in ((v1, 1), (v1, 2), (v2, 3), (c, 1), (make_user(d), 40)):      # own play and old play left out
        d.execute("insert into public.viewer_events (user_id, creator_id, kind, post_id, created_at) "
                  "values (%s, %s, 'view', %s, now() - make_interval(days => %s))", (v, c, uuid.uuid4(), days))
    d.execute("insert into public.creator_earnings (creator_id, coins, kind, status) values "
              "(%s, 500, 'unlock', 'available'), (%s, 200, 'unlock', 'reversed'), (%s, -50, 'reversal', 'available')",
              (c, c, c))
    d.execute("update public.creator_earnings set reversed_at = now() where status = 'reversed'")
    out = Rpc(d).rpc("creator_level_metrics", {"p_creator": c})
    assert out["followers"] == 3 and out["viewers_30d"] == 2 and out["account_days"] == 75
    assert out["earned_coins"] == 450 and out["reversals_90d"] == 2
    assert cl.level_for(Rpc(d), str(c))["ready"] is True


def test_only_the_server_can_read_metrics(d):
    d.execute("begin")
    try:
        d.execute("set local role authenticated")
        with pytest.raises(Exception):
            d.execute("select public.creator_level_metrics(gen_random_uuid())")
    finally:
        d.execute("rollback")
