import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# repurpose.py imports main.py's auth chain lazily; local runs have no
# Supabase project to fetch signing keys from.
os.environ.setdefault("ALLOW_INSECURE_AUTH", "true")
# repurpose.py builds its OpenAI client at import; no test calls it.
os.environ.setdefault("OPENAI_API_KEY", "test-key-never-used")

DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/viyo_test"
)


def connect():
    import psycopg
    return psycopg.connect(DB_URL, autocommit=True)


@pytest.fixture(scope="session")
def migrated():
    """Fresh schema: Supabase stand-ins + the real migration file."""
    psycopg = pytest.importorskip("psycopg")
    try:
        conn = connect()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"No test database: {e}")
    with conn:
        conn.execute("drop schema if exists public cascade; create schema public;")
        conn.execute("drop schema if exists auth cascade; drop schema if exists storage cascade;")
        conn.execute((ROOT / "tests/sql/supabase_stub.sql").read_text())
        conn.execute((ROOT / "repurpose_jobs_schema.sql").read_text())
        conn.execute((ROOT / "repurpose_clip_count.sql").read_text())
    return DB_URL


@pytest.fixture
def db(migrated):
    conn = connect()
    conn.execute(
        "truncate public.repurpose_clips, public.repurpose_uploads, public.repurpose_jobs, "
        "public.repurpose_transcripts, public.transactions, public.profiles, public.posts cascade; "
        "delete from auth.users;"
    )
    yield conn
    conn.close()


def make_user(db, balance=100):
    uid = uuid.uuid4()
    db.execute("insert into auth.users (id) values (%s)", (uid,))
    db.execute("insert into public.profiles (id, points_balance) values (%s, %s)", (uid, balance))
    return uid


def pg_rpc(conn):
    """Calls the SQL functions the way PostgREST does (named args), so
    JobStore runs unmodified against a local Postgres."""
    from psycopg.types.json import Jsonb

    def adapt(name, value):
        if name == "p_job_ids":
            return [uuid.UUID(str(v)) for v in value]
        if name == "p_paths":
            return list(value)
        if isinstance(value, (dict, list)):
            return Jsonb(value)
        return value

    def call(name, params):
        args = ", ".join(f"{k} => %({k})s" for k in params)
        row = conn.execute(f"select public.{name}({args})",
                           {k: adapt(k, v) for k, v in params.items()}).fetchone()
        return row[0]

    return call
