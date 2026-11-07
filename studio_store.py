"""
Creator Studio data and file access.

Two small interfaces, so the API and worker don't care where data lives:
  Store - table rows + the SQL functions in creator_studio_schema.sql
  Files - the private "creator-studio" bucket (signed links only)

SupabaseStore/SupabaseFiles are production. PgStore/LocalFiles run the
same code against a local Postgres and a temp folder in tests.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Optional

BUCKET = "creator-studio"


class Store:
    def select(self, table: str, filters: dict, order: Optional[str] = None, desc: bool = False,
               limit: Optional[int] = None) -> list[dict]: ...
    def insert(self, table: str, rows: list[dict]) -> list[dict]: ...
    def update(self, table: str, filters: dict, fields: dict) -> list[dict]: ...
    def delete(self, table: str, filters: dict) -> None: ...
    def rpc(self, name: str, params: dict) -> Any: ...

    def one(self, table: str, filters: dict) -> Optional[dict]:
        rows = self.select(table, filters, limit=1)
        return rows[0] if rows else None


class SupabaseStore(Store):
    def __init__(self, admin):
        self.admin = admin

    def _q(self, table, filters):
        q = self.admin.table(table)
        return q, filters

    def select(self, table, filters, order=None, desc=False, limit=None):
        q = self.admin.table(table).select("*")
        for k, v in filters.items():
            q = q.in_(k, list(v)) if isinstance(v, (list, tuple, set)) else q.eq(k, v)
        if order:
            q = q.order(order, desc=desc)
        if limit:
            q = q.limit(limit)
        return q.execute().data or []

    def insert(self, table, rows):
        if not rows:
            return []
        return self.admin.table(table).insert(rows).execute().data or []

    def update(self, table, filters, fields):
        q = self.admin.table(table).update(fields)
        for k, v in filters.items():
            q = q.eq(k, v)
        return q.execute().data or []

    def delete(self, table, filters):
        q = self.admin.table(table).delete()
        for k, v in filters.items():
            q = q.in_(k, list(v)) if isinstance(v, (list, tuple, set)) else q.eq(k, v)
        q.execute()

    def rpc(self, name, params):
        return self.admin.rpc(name, params).execute().data


class PgStore(Store):
    """Same behaviour as PostgREST for the calls the Studio makes."""

    def __init__(self, conn):
        self.conn = conn

    @staticmethod
    def _adapt(v):
        from psycopg.types.json import Jsonb
        return Jsonb(v) if isinstance(v, (dict, list)) else v

    def _where(self, filters):
        parts, args = [], []
        for k, v in filters.items():
            if isinstance(v, (list, tuple, set)):
                parts.append(f"{k}::text = any(%s)")
                args.append([str(x) for x in v])
            else:
                parts.append(f"{k} = %s")
                args.append(v)
        return (" where " + " and ".join(parts)) if parts else "", args

    def _rows(self, cur):
        cols = [c.name for c in cur.description]
        return [{c: _jsonable(v) for c, v in zip(cols, r)} for r in cur.fetchall()]

    def select(self, table, filters, order=None, desc=False, limit=None):
        w, args = self._where(filters)
        sql = f"select * from public.{table}{w}"
        if order:
            sql += f" order by {order} {'desc' if desc else 'asc'}"
        if limit:
            sql += f" limit {int(limit)}"
        return self._rows(self.conn.execute(sql, args))

    def insert(self, table, rows):
        out = []
        for row in rows:
            cols = list(row)
            sql = (f"insert into public.{table} ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))}) returning *")
            out += self._rows(self.conn.execute(sql, [self._adapt(row[c]) for c in cols]))
        return out

    def update(self, table, filters, fields):
        w, args = self._where(filters)
        sets = ", ".join(f"{k} = %s" for k in fields)
        return self._rows(self.conn.execute(f"update public.{table} set {sets}{w} returning *",
                                            [self._adapt(v) for v in fields.values()] + args))

    def delete(self, table, filters):
        w, args = self._where(filters)
        self.conn.execute(f"delete from public.{table}{w}", args)

    def rpc(self, name, params):
        args = ", ".join(f"{k} => %({k})s" for k in params)
        row = self.conn.execute(f"select public.{name}({args})",
                                {k: self._adapt(v) for k, v in params.items()}).fetchone()
        return _jsonable(row[0])


def _jsonable(v):
    import datetime
    import decimal
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return v


# ---------------------------------------------------------------- files

class Files:
    def put(self, path: str, data: bytes, content_type: str) -> None: ...
    def put_file(self, path: str, local_path: str, content_type: str) -> None: ...
    def get(self, path: str) -> bytes: ...
    def get_to(self, path: str, local_path: str) -> None:
        with open(local_path, "wb") as f:
            f.write(self.get(path))
    def sign(self, paths: list[str], seconds: int, download_name: Optional[str] = None) -> dict[str, str]: ...
    def list(self, prefix: str) -> list[str]: ...   # file names directly under prefix


class SupabaseFiles(Files):
    def __init__(self, admin, bucket: str = BUCKET):
        self.admin, self.bucket = admin, bucket

    def _b(self):
        return self.admin.storage.from_(self.bucket)

    def put(self, path, data, content_type):
        self._b().upload(path, data, file_options={"content-type": content_type, "cache-control": "3600", "upsert": "false"})

    def put_file(self, path, local_path, content_type):
        # The storage client takes a path or bytes; a path is streamed from disk.
        self._b().upload(path, local_path, file_options={"content-type": content_type, "cache-control": "3600", "upsert": "false"})

    def get(self, path):
        return self._b().download(path)

    def list(self, prefix):
        return [r["name"] for r in (self._b().list(prefix.rstrip("/"), {"limit": 1000}) or []) if r.get("name")]

    def sign(self, paths, seconds, download_name=None):
        out = {}
        if download_name and len(paths) == 1:
            r = self._b().create_signed_url(paths[0], seconds, {"download": download_name})
            out[paths[0]] = r.get("signedURL") or r.get("signedUrl")
            return out
        if not paths:
            return out
        for r in self._b().create_signed_urls(paths, seconds) or []:
            url = r.get("signedURL") or r.get("signedUrl")
            if r.get("path") and url:
                out[r["path"]] = url
        return out


class LocalFiles(Files):
    def __init__(self, root: str):
        self.root = root

    def _p(self, path):
        full = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        return full

    def put(self, path, data, content_type):
        full = self._p(path)
        if os.path.exists(full):
            raise FileExistsError(path)
        with open(full, "wb") as f:
            f.write(data)

    def put_file(self, path, local_path, content_type):
        with open(local_path, "rb") as f:
            self.put(path, f.read(), content_type)

    def get(self, path):
        with open(os.path.join(self.root, path), "rb") as f:
            return f.read()

    def sign(self, paths, seconds, download_name=None):
        exp = int(time.time()) + seconds
        return {p: f"https://files.test/{p}?exp={exp}" + (f"&download={download_name}" if download_name else "")
                for p in paths}

    def exists(self, path):
        return os.path.exists(os.path.join(self.root, path))

    def list(self, prefix):
        d = os.path.join(self.root, prefix)
        return sorted(os.listdir(d)) if os.path.isdir(d) else []


def new_path(owner_id: str, project_id: str, kind: str, ext: str) -> str:
    return f"{owner_id}/{project_id}/{kind}/{uuid.uuid4().hex}.{ext}"
