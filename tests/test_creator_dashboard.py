"""Unified creator dashboard (creator_dashboard.sql + creator_hub.dashboard): exact counts from real records on a real
Postgres, the creator's own plays left out, earnings split by type from the ledger with reversed rows left out, Studio
credits actually charged, and advertising / brand campaigns reported as not available (nothing estimated)."""
import uuid

import pytest

import creator_hub
import wallet
from conftest import ROOT, connect, make_user


@pytest.fixture(scope="module")
def dash_schema(migrated):
    conn = connect()
    # intelligence.sql indexes the series table's title (search); the minimal table other tests use.
    conn.execute("create table if not exists public.series (id uuid primary key default gen_random_uuid(), "
                 "user_id uuid references auth.users(id) on delete cascade, title text, cover_url text); "
                 "alter table public.profiles add column if not exists username text; "
                 "alter table public.profiles add column if not exists display_name text")
    for f in ("intelligence.sql", "creator_payouts_schema.sql", "creator_payouts_safety.sql", "nardopay_orders.sql",
              "creator_earnings_ledger.sql", "creator_donations.sql", "creator_studio_schema.sql",
              "creator_dashboard.sql", "creator_dashboard.sql"):           # the last one twice: safe to run again
        conn.execute((ROOT / f).read_text())
    conn.close()


class _Res:
    def __init__(self, data):
        self.data = data


class Rpc:
    """What creator_hub.dashboard needs from the Supabase client: rpc(...).execute()."""
    def __init__(self, conn):
        self.conn = conn

    def rpc(self, name, params):
        conn = self.conn

        class Call:
            def execute(self_inner):
                args = ", ".join(f"{k} => %({k})s" for k in params)
                return _Res(conn.execute(f"select public.{name}({args})", params).fetchone()[0])
        return Call()


@pytest.fixture
def d(dash_schema, db):
    db.execute("truncate public.viewer_events, public.creator_earnings, public.withdrawal_requests, "
               "public.studio_jobs, public.studio_projects cascade")
    return db


def event(db, user, creator, kind, post, value=None, days_ago=0):
    db.execute("insert into public.viewer_events (user_id, creator_id, kind, post_id, value, created_at) "
               "values (%s, %s, %s, %s, %s, now() - make_interval(days => %s))", (user, creator, kind, post, value, days_ago))


def earning(db, creator, coins, kind="unlock", status="available"):
    req = None
    if status in ("pending_withdrawal", "paid"):
        req = db.execute("insert into public.withdrawal_requests (creator_id, coins, status, usd_amount) "
                         "values (%s, %s, %s, %s) returning id",
                         (creator, coins, "pending" if status == "pending_withdrawal" else "paid", coins * 0.006)).fetchone()[0]
    db.execute("insert into public.creator_earnings (creator_id, coins, kind, status, withdrawal_request_id) "
               "values (%s, %s, %s, %s, %s)", (creator, coins, kind, status, req))


def test_audience_counts_other_peoples_plays_in_the_last_30_days(d):
    c, v1, v2 = make_user(d), make_user(d), make_user(d)
    p1, p2 = uuid.uuid4(), uuid.uuid4()
    event(d, v1, c, "view", p1, 120)
    event(d, v1, c, "complete", p1)
    event(d, v1, c, "view", p2, 60)
    event(d, v2, c, "view", p1, 30)
    event(d, c, c, "view", p1, 999)                     # the creator's own play: left out
    event(d, v2, c, "view", p2, 600, days_ago=40)       # older than 30 days: left out
    out = creator_hub.dashboard(Rpc(d), str(c))
    a = out["audience"]
    assert out["ready"] is True and out["days"] == 30
    assert a["plays"] == 3 and a["unique_viewers"] == 2 and a["watch_minutes"] == 4      # (120+60+30)/60 = 3.5 -> 4
    assert a["completion_rate"] == round(1 / 3, 3)                                       # 1 of 3 viewer-episode starts


def test_earnings_are_split_by_type_from_the_ledger_and_reversed_ones_dont_count(d):
    c = make_user(d)
    earning(d, c, 100, "unlock", "pending")
    earning(d, c, 200, "unlock", "available")
    earning(d, c, 300, "unlock", "paid")
    earning(d, c, 50, "unlock", "reversed")             # refunded: never counts
    earning(d, c, 1333, "donation", "pending")
    earning(d, c, -13, "reversal", "available")        # a paid earning reversed later: deducted
    e = creator_hub.dashboard(Rpc(d), str(c))["earnings"]
    content, donations, adj = e["by_type"]["content"], e["by_type"]["donations"], e["by_type"]["adjustments"]
    assert content["coins"] == 600 and content["held_coins"] == 100 and content["paid_coins"] == 300
    assert donations["coins"] == 1333 and donations["held_coins"] == 1333 and donations["usd"] == wallet.coins_to_usd(1333)
    assert adj["coins"] == -13
    assert e["held_coins"] == 1433 and e["available_coins"] == 200 - 13 and e["hold_days"] == 14
    assert e["advertising"] == {"available": False, "note": "Not available on VIYO yet."}
    assert e["brand_campaigns"]["available"] is False


def test_withdrawals_and_studio_activity(d):
    c = make_user(d)
    earning(d, c, 2000, "unlock", "paid")
    proj = d.execute("insert into public.studio_projects (owner_id, title) values (%s, 'Pilot') returning id",
                     (c,)).fetchone()[0]
    d.execute("insert into public.studio_projects (owner_id, title, status) values (%s, 'Old', 'archived')", (c,))
    for status, credits in (("succeeded", 40), ("succeeded", 60), ("running", 0), ("failed", 0)):
        d.execute("insert into public.studio_jobs (owner_id, project_id, kind, lane, status, charged_credits, "
                  "idempotency_key) values (%s, %s, 'video', 'generation', %s, %s, %s)",
                  (c, proj, status, credits, str(uuid.uuid4())))
    out = creator_hub.dashboard(Rpc(d), str(c))
    assert out["withdrawals"][0]["status"] == "paid" and out["withdrawals"][0]["usd"] == 12.0
    assert out["studio"] == {"projects": 1, "active_jobs": 1, "completed_jobs": 2, "failed_jobs": 1,
                             "credits_used": 100, "credits_used_period": 100}


def test_another_creators_records_never_show(d):
    c, other = make_user(d), make_user(d)
    earning(d, other, 999, "unlock", "available")
    event(d, make_user(d), other, "view", uuid.uuid4(), 60)
    out = creator_hub.dashboard(Rpc(d), str(c))
    assert out["audience"]["plays"] == 0 and out["earnings"]["available_coins"] == 0


def test_only_the_server_can_call_it(d):
    d.execute("begin")
    try:
        d.execute("set local role authenticated")
        with pytest.raises(Exception):
            d.execute("select public.creator_dashboard_stats(gen_random_uuid(), 30)")
    finally:
        d.execute("rollback")


def test_without_the_function_the_dashboard_says_not_ready_and_estimates_nothing():
    class NoRpc:
        def rpc(self, *a):
            raise Exception("PGRST202 function not found")
    out = creator_hub.dashboard(NoRpc(), str(uuid.uuid4()))
    assert out["ready"] is False and "audience" not in out and out["advertising"]["available"] is False
