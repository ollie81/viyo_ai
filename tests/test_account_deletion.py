"""Deleting an account removes files in sub-folders (Viyo Studio) and the
user's Bunny videos, not just top-level files and database rows."""
import types

from fastapi.testclient import TestClient

import bunny_stream
import coach
import main

U = "22222222-aaaa-bbbb-cccc-000000000002"


class FakeBucket:
    def __init__(self, files, removed):
        self.files, self.removed = files, removed

    def list(self, prefix, opts=None):
        seen, out = set(), []
        for f in self.files:
            if not f.startswith(prefix + "/"):
                continue
            rest = f[len(prefix) + 1:]
            head = rest.split("/")[0]
            if head in seen:
                continue
            seen.add(head)
            out.append({"name": head, "id": None if "/" in rest else "file-id"})
        return out

    def remove(self, paths):
        self.removed.extend(paths)


class FakeQuery:
    def __init__(self, admin, table):
        self.admin, self.table = admin, table

    def select(self, *a):
        return self

    def delete(self):
        self.admin.deleted_tables.append(self.table)
        return self

    def eq(self, *a):
        return self

    def execute(self):
        return types.SimpleNamespace(data=self.admin.rows.get(self.table, []))


class FakeAdmin:
    def __init__(self):
        self.removed, self.deleted_tables, self.deleted_users = [], [], []
        self.buckets = {
            "creator-studio": [f"{U}/proj1/image/a.png", f"{U}/proj1/audio/b.m4a", f"{U}/proj2/portrait/c.png"],
            "posts-media": [f"{U}/v1.mp4", "someone-else/x.mp4"],
        }
        self.rows = {"posts": [{"bunny_video_id": "bv-post"}, {"bunny_video_id": None}],
                     "studio_renders": [{"bunny_video_id": "bv-film"}]}
        self.storage = types.SimpleNamespace(from_=lambda b: FakeBucket(self.buckets.get(b, []), self.removed))
        self.auth = types.SimpleNamespace(admin=types.SimpleNamespace(delete_user=self.deleted_users.append))

    def table(self, name):
        return FakeQuery(self, name)


def test_account_deletion_removes_studio_files_and_bunny_videos(monkeypatch):
    admin = FakeAdmin()
    bunny_deleted = []
    monkeypatch.setattr(coach, "supabase_admin", admin)
    monkeypatch.setattr(bunny_stream, "delete_video", lambda vid: bunny_deleted.append(vid) or True)
    main.app.dependency_overrides[coach._get_current_user_id] = lambda: U
    try:
        r = TestClient(main.app).delete("/api/v1/account")
    finally:
        main.app.dependency_overrides.clear()
    assert r.status_code == 200 and r.json()["deleted"] is True, r.text
    assert sorted(admin.removed) == sorted(admin.buckets["creator-studio"] + [f"{U}/v1.mp4"])
    assert "someone-else/x.mp4" not in admin.removed
    assert sorted(bunny_deleted) == ["bv-film", "bv-post"]
    assert admin.deleted_users == [U]
