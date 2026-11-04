"""Notifications: the helper, the worker's job alerts, and the SQL fix."""
import pathlib

import notify
from test_repurpose_worker import env, source_video  # noqa: F401  (pytest fixtures)

ROOT = pathlib.Path(__file__).resolve().parent.parent


class FakeTable:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def insert(self, row):
        self.db.setdefault(self.name, []).append(row)
        return self

    def select(self, *a):
        return self

    def eq(self, *a):
        return self

    def limit(self, *a):
        return self

    def execute(self):
        class R:
            data = [{"display_name": "Ada", "username": "ada"}]
        return R()


class FakeAdmin:
    def __init__(self):
        self.rows = {}

    def table(self, name):
        return FakeTable(self.rows, name)


def test_notify_saves_row_with_post_and_never_notifies_yourself(monkeypatch):
    admin = FakeAdmin()
    pushes = []
    import push
    monkeypatch.setattr(push, "send_push_to_user", lambda a, uid, t, b, d: pushes.append((uid, t, d)))
    assert notify.notify("u1", "like", "Ada liked your post", actor_id="u2", post_id="p1",
                         push_title="New like", admin=admin)
    assert admin.rows["notifications"] == [{"user_id": "u1", "type": "like", "message": "Ada liked your post",
                                            "actor_id": "u2", "post_id": "p1"}]
    assert pushes == [("u1", "New like", {"type": "like", "post_id": "p1", "actor_id": "u2"})]
    assert not notify.notify("u1", "like", "x", actor_id="u1", admin=admin), "no alert for your own action"
    assert notify.display_name("u2", admin) == "Ada"


def test_sql_fix_drops_only_duplicate_triggers_and_old_type_check(db):
    db.execute("""
      create table if not exists public.notifications (
        id uuid primary key default gen_random_uuid(), user_id uuid, actor_id uuid, type text
          check (type in ('like','comment','follow')), entity_id uuid, post_id uuid, message text,
        is_read boolean default false, created_at timestamptz default now());
      create table if not exists public.likes (user_id uuid, post_id uuid);
      create table if not exists public.follows (follower_id uuid, following_id uuid);
      create or replace function public.old_like_notify() returns trigger language plpgsql as $f$
        begin insert into public.notifications (user_id, type, message) values (new.user_id, 'like', 'New activity');
        return new; end $f$;
      create or replace function public.count_likes() returns trigger language plpgsql as $f$
        begin return new; end $f$;
      create or replace function public.follow_notify() returns trigger language plpgsql as $f$
        begin insert into public.notifications (user_id, type, message) values (new.following_id, 'follow', 'x');
        return new; end $f$;
      create trigger old_like_notify after insert on public.likes for each row execute function public.old_like_notify();
      create trigger count_likes after insert on public.likes for each row execute function public.count_likes();
      create trigger follow_notify after insert on public.follows for each row execute function public.follow_notify();
    """)
    sql = (ROOT / "notifications_fix.sql").read_text()
    db.execute(sql)
    db.execute(sql)   # safe to re-run
    triggers = {r[0] for r in db.execute(
        "select tgname from pg_trigger where not tgisinternal and tgrelid in "
        "('public.likes'::regclass, 'public.follows'::regclass)").fetchall()}
    assert triggers == {"count_likes", "follow_notify"}, "only the duplicate like notifier is removed"
    db.execute("insert into public.notifications (type, message) values ('unlock', 'ok')")


def test_worker_tells_the_creator_when_a_job_fails_for_good(env):
    sent = []
    env["worker"].notifier = lambda uid, t, msg, job_id: sent.append((uid, t, msg, job_id))
    env["scripted"].weak = True
    from test_repurpose_worker import submit
    uid, job_id = submit(env)
    env["worker"].run_once()
    assert len(sent) == 1
    who, kind, message, jid = sent[0]
    assert (who, kind, jid) == (str(uid), "clips_failed", str(job_id))
    assert "couldn't make clips" in message and "40 coins were refunded" in message
