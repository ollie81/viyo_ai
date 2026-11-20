"""Coin unlocks keep working when the premium-subscription columns are missing from the database (live error
42703), and subscriptions_schema.sql adds them without letting a user mark themselves subscribed."""
import asyncio
import uuid

import pytest

import episodes
import subscriptions
from conftest import ROOT, connect, make_user


class _Res:
    def __init__(self, data):
        self.data = data


class _Missing(Exception):
    pass


class _Q:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.row = db, table, "select", None

    def select(self, *_a, **_k):
        return self

    def insert(self, row):
        self.op, self.row = "insert", row
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, *_a):
        return self

    def limit(self, *_a):
        return self

    def execute(self):
        if self.table == "profiles":
            raise _Missing("{'code': '42703', 'message': 'column profiles.is_subscribed does not exist'}")
        if self.op == "insert":
            self.db.rows.setdefault(self.table, []).append(self.row)
            return _Res([self.row])
        if self.op == "delete":
            self.db.rows[self.table] = []
        return _Res([])


class FakeDB:
    """The live database as the audit found it: profiles has no is_subscribed column."""
    def __init__(self):
        self.rows = {}

    def table(self, name):
        return _Q(self, name)


def test_missing_subscription_column_means_not_subscribed():
    assert subscriptions.is_active_subscriber(FakeDB(), "u1") is False


@pytest.fixture
def unlock_env(monkeypatch):
    db = FakeDB()
    creator, viewer, post = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    monkeypatch.setattr(episodes, "supabase_admin", db)
    monkeypatch.setattr(episodes, "is_active_subscriber", subscriptions.is_active_subscriber)
    monkeypatch.setattr(episodes, "_get_episode", lambda pid: {"id": pid, "user_id": creator, "series_id": "s1",
                                                              "episode_number": 5, "is_free": None})
    monkeypatch.setattr(episodes, "_get_series", lambda sid: {"id": sid, "content_type": "short_drama",
                                                            "title": "T", "coin_price_per_episode": 20})
    debits = []
    monkeypatch.setattr(episodes, "debit_coins", lambda _a, uid, n, kind, note: debits.append((uid, n, kind)))
    monkeypatch.setattr(episodes, "_bought_coins_spent", lambda uid, price: price)
    monkeypatch.setattr(episodes, "_notify_unlock", lambda *a, **k: None)
    return db, creator, viewer, post, debits


def test_coin_unlock_works_without_subscription_columns(unlock_env):
    db, creator, viewer, post, debits = unlock_env
    res = asyncio.run(episodes.unlock_episode(post, user_id=viewer))
    assert res.unlocked and res.coins_spent == 20
    assert debits == [(viewer, 20, "episode_unlock")]
    assert db.rows["episode_unlocks"] == [{"user_id": viewer, "post_id": post, "coins_spent": 20}]
    assert db.rows["creator_earnings"] == [{"creator_id": creator, "source_post_id": post,
                                            "coins": round(20 * episodes.CREATOR_SHARE)}]


def test_subscription_status_without_columns(monkeypatch):
    monkeypatch.setattr(subscriptions, "supabase_admin", FakeDB())
    res = asyncio.run(subscriptions.get_subscription_status(user=("u1", "a@b.c")))
    assert res.is_subscribed is False


# ---- the migration, on a real Postgres ----

@pytest.fixture(scope="module")
def schema(migrated):
    conn = connect()
    conn.execute((ROOT / "subscriptions_schema.sql").read_text())
    conn.execute((ROOT / "subscriptions_schema.sql").read_text())        # safe to run twice
    conn.execute("grant select, insert, update on public.profiles to authenticated")
    conn.close()


def _as_user(db, uid, sql, params=()):
    db.execute("begin")
    try:
        db.execute("set local role authenticated")
        db.execute("select set_config('request.jwt.claim.sub', %s, true)", (str(uid),))
        db.execute(sql, params)
        db.execute("commit")
    except Exception:
        db.execute("rollback")
        raise


def test_new_accounts_are_not_subscribed(schema, db):
    uid = make_user(db)
    assert db.execute("select is_subscribed from public.profiles where id = %s", (uid,)).fetchone()[0] is False


def test_user_cannot_make_themselves_subscribed(schema, db):
    uid = make_user(db)
    with pytest.raises(Exception, match="can only be changed by VIYO"):
        _as_user(db, uid, "update public.profiles set is_subscribed = true where id = %s", (uid,))
    with pytest.raises(Exception, match="can only be changed by VIYO"):
        _as_user(db, uid, "update public.profiles set subscription_ends_at = now() where id = %s", (uid,))
    assert db.execute("select is_subscribed from public.profiles where id = %s", (uid,)).fetchone()[0] is False


def test_user_can_still_edit_other_profile_fields(schema, db):
    uid = make_user(db, balance=5)
    _as_user(db, uid, "update public.profiles set points_balance = points_balance where id = %s", (uid,))


def test_user_cannot_sign_up_subscribed(schema, db):
    uid = uuid.uuid4()
    db.execute("insert into auth.users (id) values (%s)", (uid,))
    _as_user(db, uid, "insert into public.profiles (id, is_subscribed) values (%s, true)", (uid,))
    assert db.execute("select is_subscribed from public.profiles where id = %s", (uid,)).fetchone()[0] is False


def test_server_can_set_subscription(schema, db):
    uid = make_user(db)
    db.execute("update public.profiles set is_subscribed = true, subscription_plan = 'monthly' where id = %s", (uid,))
    assert db.execute("select is_subscribed from public.profiles where id = %s", (uid,)).fetchone()[0] is True
