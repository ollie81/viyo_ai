"""Deleting an account: dramas stay on Viyo only when the creator chooses it, under 'Former creator'."""
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import coach
from conftest import ROOT, connect, make_user


class _Res:
    def __init__(self, data):
        self.data = data


class _Query:
    """Just enough of the Supabase query builder for account deletion, run against Postgres."""

    def __init__(self, conn, table):
        self.conn, self.table, self.where, self.args, self.op, self.payload = conn, table, [], [], "select", None

    def select(self, _cols="*"):
        self.op = "select"
        return self

    def eq(self, col, val):
        self.where.append(f"{col} = %s")
        self.args.append(val)
        return self

    def in_(self, col, vals):
        self.where.append(f"{col}::text = any(%s)")
        self.args.append([str(v) for v in vals])
        return self

    def limit(self, _n):
        return self

    def delete(self):
        self.op = "delete"
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row
        return self

    def upsert(self, row):
        self.op, self.payload = "upsert", row
        return self

    def execute(self):
        w = (" where " + " and ".join(self.where)) if self.where else ""
        t = f"public.{self.table}"
        if self.op == "select":
            cur = self.conn.execute(f"select * from {t}{w}", self.args)
            cols = [d.name for d in cur.description]
            return _Res([{c: (str(v) if isinstance(v, uuid.UUID) else v) for c, v in zip(cols, r)} for r in cur.fetchall()])
        if self.op == "delete":
            self.conn.execute(f"delete from {t}{w}", self.args)
            return _Res([])
        cols = list(self.payload)
        sql = f"insert into {t} ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))})"
        if self.op == "upsert":
            sql += " on conflict (id) do update set " + ", ".join(f"{c} = excluded.{c}" for c in cols if c != "id")
        self.conn.execute(sql, [self.payload[c] for c in cols])
        return _Res([self.payload])


class _Store:
    def __init__(self, files, bucket):
        self.files, self.bucket = files, bucket

    def list(self, prefix, _opts=None):
        names = {p[len(prefix) + 1:].split("/")[0] for b, p in self.files if b == self.bucket and p.startswith(prefix + "/")}
        return [{"name": n, "id": None if any(p.startswith(f"{prefix}/{n}/") for b, p in self.files if b == self.bucket)
                 else "f"} for n in names]

    def remove(self, paths):
        for p in paths:
            self.files.discard((self.bucket, p))


class FakeAdmin:
    def __init__(self, conn):
        self.conn, self.files, self.deleted_users, self.created = conn, set(), [], []
        admin = self

        class _AuthAdmin:
            def delete_user(self, uid):
                admin.deleted_users.append(uid)
                conn.execute("delete from auth.users where id = %s", (uid,))

            def create_user(self, attrs):
                uid = uuid.uuid4()
                conn.execute("insert into auth.users (id) values (%s)", (uid,))
                admin.created.append(attrs["email"])
                return {"id": str(uid)}

        class _Auth:
            pass

        self.auth = _Auth()
        self.auth.admin = _AuthAdmin()

        class _Storage:
            def from_(self, bucket):
                return _Store(admin.files, bucket)

        self.storage = _Storage()

    # Tables other test modules create for real (Studio, Glow Up, Coach) get test-only stand-ins here.
    STAND_INS = {"studio_renders", "video_coach_messages", "glow_up_sessions", "glow_up_previews"}

    def table(self, name):
        return _Query(self.conn, f"kd_{name}" if name in self.STAND_INS else name)

    def rpc(self, name, params):
        conn = self.conn

        class Call:
            def execute(self_inner):
                args = ", ".join(f"{k} => %({k})s" for k in params)
                return _Res(conn.execute(f"select public.{name}({args})", params).fetchone()[0])
        return Call()


@pytest.fixture(scope="module")
def schema(migrated):
    conn = connect()
    conn.execute("""
      alter table public.profiles add column if not exists username text;
      alter table public.profiles add column if not exists display_name text;
      create table if not exists public.series (id uuid primary key default gen_random_uuid(),
        user_id uuid references auth.users(id) on delete cascade, title text, cover_url text);
      alter table public.posts add column if not exists series_id uuid;
      alter table public.posts add column if not exists bunny_video_id text;
      create table if not exists public.kd_video_coach_messages (user_id uuid);
      create table if not exists public.kd_glow_up_sessions (user_id uuid);
      create table if not exists public.kd_glow_up_previews (user_id uuid);
      create table if not exists public.kd_studio_renders (owner_id uuid, bunny_video_id text);
    """)
    conn.execute((ROOT / "account_keep_dramas.sql").read_text())
    conn.execute((ROOT / "account_keep_dramas.sql").read_text())          # safe to run twice
    conn.close()


STORE = "https://x.supabase.co/storage/v1/object/public/posts-media"


@pytest.fixture
def env(schema, db, monkeypatch):
    db.execute("truncate public.series, public.viyo_system_accounts, public.kd_studio_renders cascade")
    admin = FakeAdmin(db)
    monkeypatch.setattr(coach, "supabase_admin", admin)
    deleted_videos = []
    import bunny_stream
    monkeypatch.setattr(bunny_stream, "delete_video", lambda vid: deleted_videos.append(vid) or True)
    creator = make_user(db)
    other = make_user(db)
    series = db.execute("insert into public.series (user_id, title, cover_url) values (%s, 'Love in Kigali', %s) returning id",
                        (creator, f"{STORE}/{creator}/cover.jpg")).fetchone()[0]
    db.execute("insert into public.posts (user_id, series_id, bunny_video_id, thumbnail_url) values (%s, %s, 'ep1', %s)",
               (creator, series, f"{STORE}/{creator}/ep1.jpg"))
    db.execute("insert into public.posts (user_id, series_id, bunny_video_id) values (%s, %s, 'ep2')", (creator, series))
    db.execute("insert into public.posts (user_id, bunny_video_id) values (%s, 'selfie-vlog')", (creator,))
    db.execute("insert into public.posts (user_id, bunny_video_id) values (%s, 'others')", (other,))
    db.execute("insert into public.kd_studio_renders (owner_id, bunny_video_id) values (%s, 'ep1'), (%s, 'draft')",
               (creator, creator))
    admin.files |= {("posts-media", f"{creator}/cover.jpg"), ("posts-media", f"{creator}/ep1.jpg"),
                    ("posts-media", f"{creator}/vlog.jpg"), ("avatars", f"{creator}/me.jpg")}
    app = FastAPI()
    app.include_router(coach.router)
    current = {"id": str(creator)}
    app.dependency_overrides[coach._get_current_user_id] = lambda: current["id"]
    yield {"db": db, "admin": admin, "client": TestClient(app), "creator": creator, "other": other,
           "series": series, "videos": deleted_videos}
    # The Former creator account is protected from deletion (no cascade): clear it for the next test.
    db.execute("truncate public.viyo_system_accounts, public.series cascade")


def owners(db):
    return {r[0]: (str(r[1]) if r[1] else None) for r in db.execute("select bunny_video_id, user_id from public.posts")}


def test_summary_counts_the_creators_dramas(env):
    assert env["client"].get("/api/v1/account/deletion-summary").json() == {"dramas": 1, "episodes": 2}


def test_default_deletes_everything_as_before(env):
    r = env["client"].delete("/api/v1/account")
    assert r.status_code == 200 and r.json()["kept_dramas"] == 0
    db = env["db"]
    assert set(owners(db)) == {"others"}, "the creator's episodes and posts are gone; other users' stay"
    assert sorted(env["videos"]) == ["draft", "ep1", "ep2", "selfie-vlog"]
    assert env["admin"].created == [], "no Former creator account when nothing is kept"


def test_keeping_dramas_moves_them_to_former_creator(env):
    db, admin = env["db"], env["admin"]
    r = env["client"].delete("/api/v1/account?keep_dramas=true")
    assert r.status_code == 200, r.text
    assert (r.json()["kept_dramas"], r.json()["kept_episodes"]) == (1, 2)
    former = db.execute("select user_id from public.viyo_system_accounts where role='former_creator'").fetchone()[0]
    assert db.execute("select display_name from public.profiles where id=%s", (former,)).fetchone()[0] == "Former creator"
    o = owners(db)
    assert o["ep1"] == o["ep2"] == str(former), "the episodes now belong to Former creator"
    assert "selfie-vlog" not in o, "posts that aren't dramas are still deleted"
    assert db.execute("select user_id from public.series where id=%s", (env["series"],)).fetchone()[0] == former
    assert sorted(env["videos"]) == ["draft", "selfie-vlog"], "kept episodes' videos survive, even if Studio made them"
    assert admin.files == {("posts-media", f"{env['creator']}/cover.jpg"), ("posts-media", f"{env['creator']}/ep1.jpg")}, \
        "a kept drama's cover and thumbnail stay; everything else of theirs is removed"
    assert admin.deleted_users == [str(env["creator"])], "the account itself is still deleted"


def test_second_creator_reuses_the_same_former_creator_account(env):
    db, client = env["db"], env["client"]
    client.delete("/api/v1/account?keep_dramas=true")
    second = make_user(db)
    db.execute("insert into public.series (user_id, title) values (%s, 'Second')", (second,))
    app = client.app
    app.dependency_overrides[coach._get_current_user_id] = lambda: str(second)
    assert client.delete("/api/v1/account?keep_dramas=true").status_code == 200
    assert env["admin"].created == [coach.FORMER_CREATOR_EMAIL]
    assert db.execute("select count(distinct user_id) from public.series").fetchone()[0] == 1


def test_if_keeping_fails_nothing_is_deleted(env, monkeypatch):
    def broken():
        raise RuntimeError("auth down")
    monkeypatch.setattr(coach, "_former_creator_id", broken)
    r = env["client"].delete("/api/v1/account?keep_dramas=true")
    assert r.status_code == 503 and "nothing was deleted" in r.json()["detail"]
    assert len(owners(env["db"])) == 4 and env["videos"] == [] and env["admin"].deleted_users == []


def test_keep_function_never_moves_other_users_content(env):
    db = env["db"]
    former = make_user(db)
    r = db.execute("select public.viyo_keep_dramas(%s, %s)", (env["other"], former)).fetchone()[0]
    assert r["ok"] and r["dramas"] == 0 and r["episodes"] == 0
    assert owners(db)["others"] == str(env["other"]) and owners(db)["ep1"] == str(env["creator"])
    assert db.execute("select public.viyo_keep_dramas(%s, %s)", (env["creator"], env["creator"])).fetchone()[0]["ok"] is False
