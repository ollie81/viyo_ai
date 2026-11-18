"""Unlocking episodes with ads: 2 Google-verified ads = 1 episode, leftovers kept, never charged twice."""
import base64
import urllib.parse
import uuid

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from fastapi.testclient import TestClient

import episodes
import rewarded_ads as ra
from conftest import ROOT, connect, make_user


class _Res:
    def __init__(self, data):
        self.data = data


class PgAdmin:
    def __init__(self, conn):
        self.conn = conn

    def rpc(self, name, params):
        conn = self.conn

        class Call:
            def execute(self_inner):
                args = ", ".join(f"{k} => %({k})s" for k in params)
                return _Res(conn.execute(f"select public.{name}({args})", params).fetchone()[0])
        return Call()


KEY = ec.generate_private_key(ec.SECP256R1())
KEY_ID = "3335741209"


@pytest.fixture(scope="module")
def schema(migrated):
    conn = connect()
    conn.execute((ROOT / "creator_payouts_schema.sql").read_text())
    conn.execute((ROOT / "coin_ledger_fix.sql").read_text())
    conn.execute((ROOT / "rewarded_ads_ssv.sql").read_text())
    conn.execute("""create table if not exists public.episode_unlocks (
        user_id uuid not null, post_id uuid not null, coins_spent int not null default 0,
        created_at timestamptz not null default now(), unique (user_id, post_id))""")
    conn.execute((ROOT / "ad_episode_unlocks.sql").read_text())
    conn.execute((ROOT / "ad_episode_unlocks.sql").read_text())          # safe to run twice
    conn.close()


@pytest.fixture
def env(schema, db, monkeypatch):
    db.execute("truncate public.rewarded_ad_sessions, public.ad_episode_tickets, public.episode_unlocks")
    admin = PgAdmin(db)
    monkeypatch.setattr(ra, "supabase_admin", admin)
    monkeypatch.setattr(episodes, "supabase_admin", admin)
    monkeypatch.setenv("REWARDED_AD_SSV", "on")
    pem = KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    monkeypatch.setattr(ra, "_fetch_keys", lambda: {KEY_ID: pem})
    monkeypatch.setattr(ra, "_keys", {})
    creator, viewer = make_user(db, balance=0), make_user(db, balance=0)
    eps = {n: str(uuid.uuid4()) for n in range(1, 9)}
    by_id = {pid: {"id": pid, "user_id": str(creator), "series_id": "s1", "episode_number": n, "is_free": None}
             for n, pid in eps.items()}

    def get_episode(post_id):
        if post_id not in by_id:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Episode not found.")
        return by_id[post_id]
    monkeypatch.setattr(episodes, "_get_episode", get_episode)
    monkeypatch.setattr(episodes, "_get_series", lambda sid: {"id": sid, "content_type": "short_drama", "title": "T"})
    subscribers = set()
    monkeypatch.setattr(episodes, "is_active_subscriber", lambda _a, uid: uid in subscribers)
    app = FastAPI()
    app.include_router(ra.router)
    app.include_router(episodes.router)
    current = {"id": str(viewer)}
    app.dependency_overrides[ra._get_current_user_id_no_guest] = lambda: current["id"]
    app.dependency_overrides[episodes._get_current_user_id_no_guest] = lambda: current["id"]
    return {"db": db, "c": TestClient(app), "viewer": viewer, "creator": creator, "eps": eps,
            "current": current, "subscribers": subscribers}


def watch(env, txn, purpose="episode"):
    """Start a session, then Google's signed confirmation arrives."""
    r = env["c"].post(f"/api/v1/coins/rewarded-ad/start?purpose={purpose}")
    assert r.status_code == 200, r.text
    s = r.json()
    params = [("ad_network", "1"), ("ad_unit", "4402469307"), ("custom_data", s["session"]),
              ("reward_amount", "1"), ("reward_item", "Reward"), ("timestamp", "1"),
              ("transaction_id", txn), ("user_id", s["user_id"])]
    msg = urllib.parse.urlencode(params)
    sig = base64.urlsafe_b64encode(KEY.sign(msg.encode(), ec.ECDSA(hashes.SHA256()))).decode().rstrip("=")
    assert env["c"].get(f"/api/v1/admob/ssv?{msg}&signature={sig}&key_id={KEY_ID}").status_code == 200
    return s


def tickets(env):
    return env["c"].get("/api/v1/episodes/ad-tickets").json()["tickets"]


def coins(env):
    return int(env["db"].execute("select points_balance from public.profiles where id=%s", (env["viewer"],)).fetchone()[0])


def unlocks(env):
    return {str(r[0]) for r in env["db"].execute("select post_id from public.episode_unlocks where user_id=%s",
                                                  (env["viewer"],))}


def test_two_ads_unlock_one_episode(env):
    ep4 = env["eps"][4]
    watch(env, "t1")
    r = env["c"].post(f"/api/v1/episodes/{ep4}/unlock-with-ads")
    assert r.status_code == 402 and r.json()["detail"] == {"error": "not_enough_tickets", "tickets": 1, "needed": 2}
    watch(env, "t2")
    r = env["c"].post(f"/api/v1/episodes/{ep4}/unlock-with-ads")
    assert r.status_code == 200 and r.json() == {"unlocked": True, "already_unlocked": False, "tickets": 0}
    assert unlocks(env) == {ep4}
    assert coins(env) == 0, "episode ads give tickets, not coins"


def test_several_episodes_at_once_and_leftovers_are_kept(env):
    for k in range(7):                       # 3 episodes need 6 ads; one extra stays for later
        watch(env, f"t{k}")
    for n in (4, 5, 6):
        assert env["c"].post(f"/api/v1/episodes/{env['eps'][n]}/unlock-with-ads").status_code == 200
    assert unlocks(env) == {env["eps"][4], env["eps"][5], env["eps"][6]}
    assert tickets(env) == 1


def test_never_charged_twice(env):
    for k in range(4):
        watch(env, f"t{k}")
    ep = env["eps"][4]
    env["c"].post(f"/api/v1/episodes/{ep}/unlock-with-ads")
    r = env["c"].post(f"/api/v1/episodes/{ep}/unlock-with-ads")
    assert r.json()["already_unlocked"] is True and tickets(env) == 2


def test_free_own_and_premium_episodes_cost_nothing(env):
    watch(env, "t1")
    watch(env, "t2")
    assert env["c"].post(f"/api/v1/episodes/{env['eps'][2]}/unlock-with-ads").json()["unlocked"] is True   # free ep 2
    env["subscribers"].add(str(env["viewer"]))
    assert env["c"].post(f"/api/v1/episodes/{env['eps'][5]}/unlock-with-ads").json()["unlocked"] is True
    env["subscribers"].clear()
    env["current"]["id"] = str(env["creator"])
    assert env["c"].post(f"/api/v1/episodes/{env['eps'][6]}/unlock-with-ads").json()["unlocked"] is True
    env["current"]["id"] = str(env["viewer"])
    assert tickets(env) == 2 and unlocks(env) == set()


def test_coin_ads_still_give_coins(env):
    watch(env, "c1", purpose="coins")
    assert coins(env) == ra.REWARDED_AD_COINS and tickets(env) == 0


def test_google_retries_give_one_ticket(env):
    s = watch(env, "t1")
    params = [("ad_network", "1"), ("ad_unit", "4402469307"), ("custom_data", s["session"]),
              ("reward_amount", "1"), ("reward_item", "Reward"), ("timestamp", "1"),
              ("transaction_id", "t1"), ("user_id", s["user_id"])]
    msg = urllib.parse.urlencode(params)
    sig = base64.urlsafe_b64encode(KEY.sign(msg.encode(), ec.ECDSA(hashes.SHA256()))).decode().rstrip("=")
    for _ in range(3):
        env["c"].get(f"/api/v1/admob/ssv?{msg}&signature={sig}&key_id={KEY_ID}")
    assert tickets(env) == 1


def test_episode_ads_have_their_own_daily_cap(env, monkeypatch):
    monkeypatch.setattr(ra, "EPISODE_AD_DAILY_CAP", 2)
    watch(env, "t1")
    watch(env, "t2")
    r = env["c"].post("/api/v1/coins/rewarded-ad/start?purpose=episode")
    assert r.status_code == 429 and r.json()["detail"]["error"] == "daily_ad_cap_reached"
    watch(env, "c1", purpose="coins")      # coin ads are counted separately
    assert coins(env) == ra.REWARDED_AD_COINS


def test_needs_google_verification(env, monkeypatch):
    monkeypatch.setenv("REWARDED_AD_SSV", "off")
    r = env["c"].post("/api/v1/coins/rewarded-ad/start?purpose=episode")
    assert r.status_code == 409


def test_parallel_unlocks_never_overspend(env):
    import threading
    for k in range(2):
        watch(env, f"t{k}")
    results = []

    def go(n):
        conn = connect()
        try:
            r = conn.execute("select public.ad_unlock_episode(%s, %s, 2)", (env["viewer"], env["eps"][n])).fetchone()[0]
            results.append(r["ok"] and not r["already_unlocked"])
        finally:
            conn.close()
    threads = [threading.Thread(target=go, args=(n,)) for n in (4, 5, 6, 7)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1 and tickets(env) == 0 and len(unlocks(env)) == 1
