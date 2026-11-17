"""Rewarded ads: coins only after a Google-signed confirmation, once per ad, inside the daily cap."""
import base64
import urllib.parse
import uuid

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from fastapi.testclient import TestClient

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


@pytest.fixture(scope="module")
def ads_schema(migrated):
    conn = connect()
    conn.execute((ROOT / "creator_payouts_schema.sql").read_text())
    conn.execute((ROOT / "coin_ledger_fix.sql").read_text())
    conn.execute((ROOT / "rewarded_ads_ssv.sql").read_text())
    conn.execute((ROOT / "rewarded_ads_ssv.sql").read_text())          # safe to run twice
    conn.close()


GOOGLE_KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())
KEY_ID = "3335741209"
UNIT = "4402469307"


def _pem(k):
    return k.public_key().public_bytes(serialization.Encoding.PEM,
                                       serialization.PublicFormat.SubjectPublicKeyInfo).decode()


@pytest.fixture
def env(ads_schema, db, monkeypatch):
    db.execute("truncate public.rewarded_ad_sessions")
    monkeypatch.setattr(ra, "supabase_admin", PgAdmin(db))
    monkeypatch.setattr(ra, "_fetch_keys", lambda: {KEY_ID: _pem(GOOGLE_KEY)})
    monkeypatch.setattr(ra, "_keys", {})
    monkeypatch.setenv("REWARDED_AD_SSV", "on")
    user = make_user(db, balance=0)
    app = FastAPI()
    app.include_router(ra.router)
    current = {"id": str(user)}
    app.dependency_overrides[ra._get_current_user_id_no_guest] = lambda: current["id"]
    return {"db": db, "user": user, "client": TestClient(app), "current": current}


def balance(db, u):
    return int(db.execute("select points_balance from public.profiles where id=%s", (u,)).fetchone()[0])


def callback(session, user, txn="txn-1", unit=UNIT, key=GOOGLE_KEY, tamper=None):
    """A reward callback as AdMob sends it: parameters in alphabetical order, then signature and key_id."""
    params = [("ad_network", "5450213213286189855"), ("ad_unit", unit), ("custom_data", str(session)),
              ("reward_amount", "1"), ("reward_item", "Reward"), ("timestamp", "1763380000000"),
              ("transaction_id", txn), ("user_id", str(user))]
    message = urllib.parse.urlencode(params)
    sig = key.sign(message.encode(), ec.ECDSA(hashes.SHA256()))
    signed = message if tamper is None else message.replace(*tamper)
    return f"/api/v1/admob/ssv?{signed}&signature={base64.urlsafe_b64encode(sig).decode().rstrip('=')}&key_id={KEY_ID}"


def start(env):
    r = env["client"].post("/api/v1/coins/rewarded-ad/start")
    assert r.status_code == 200, r.text
    return r.json()


def test_coins_arrive_only_after_googles_confirmation(env):
    c, db, u = env["client"], env["db"], env["user"]
    s = start(env)
    assert s["verification"] == "server" and s["user_id"] == str(u)
    assert c.get(f"/api/v1/coins/rewarded-ad/{s['session']}").json() == {"credited": False, "coins": 0}
    assert balance(db, u) == 0, "starting (or showing) an ad pays nothing"
    assert c.get(callback(s["session"], u)).status_code == 200
    assert c.get(f"/api/v1/coins/rewarded-ad/{s['session']}").json() == {"credited": True, "coins": ra.REWARDED_AD_COINS}
    assert balance(db, u) == ra.REWARDED_AD_COINS
    row = db.execute("select amount, type, external_ref from public.transactions where user_id=%s", (u,)).fetchone()
    assert row == (ra.REWARDED_AD_COINS, "rewarded_ad", "admob:txn-1"), "a permanent ledger row with AdMob's id"


def test_google_retrying_the_same_callback_credits_once(env):
    c, db, u = env["client"], env["db"], env["user"]
    s = start(env)
    for _ in range(3):
        assert c.get(callback(s["session"], u)).status_code == 200
    assert balance(db, u) == ra.REWARDED_AD_COINS


def test_one_ad_cannot_pay_two_sessions(env):
    c, db, u = env["client"], env["db"], env["user"]
    a, b = start(env), start(env)
    c.get(callback(a["session"], u, txn="same"))
    c.get(callback(b["session"], u, txn="same"))
    assert balance(db, u) == ra.REWARDED_AD_COINS


def test_watching_several_ads_in_a_row_pays_each(env):
    c, db, u = env["client"], env["db"], env["user"]
    for k in range(3):
        s = start(env)
        c.get(callback(s["session"], u, txn=f"t{k}"))
    assert balance(db, u) == 3 * ra.REWARDED_AD_COINS


def test_forged_or_tampered_callbacks_pay_nothing(env):
    c, db, u = env["client"], env["db"], env["user"]
    s = start(env)
    assert c.get(callback(s["session"], u, key=OTHER_KEY)).status_code == 403, "not signed by Google"
    assert c.get(callback(s["session"], u, tamper=("reward_amount=1", "reward_amount=9"))).status_code == 403
    assert c.get(f"/api/v1/admob/ssv?custom_data={s['session']}&user_id={u}&transaction_id=x").status_code == 403
    assert balance(db, u) == 0


def test_someone_elses_session_or_another_ad_unit_pays_nothing(env):
    c, db, u = env["client"], env["db"], env["user"]
    s = start(env)
    other = make_user(db, balance=0)
    assert c.get(callback(s["session"], other)).status_code == 400
    assert c.get(callback(s["session"], u, unit="9999999999")).status_code == 400
    assert c.get(callback(uuid.uuid4(), u)).status_code == 400
    assert balance(db, u) == 0 and balance(db, other) == 0


def test_daily_cap_still_applies(env, monkeypatch):
    monkeypatch.setattr(ra, "DAILY_AD_CAP", 2)
    c, db, u = env["client"], env["db"], env["user"]
    for k in range(2):
        s = start(env)
        c.get(callback(s["session"], u, txn=f"t{k}"))
    r = c.post("/api/v1/coins/rewarded-ad/start")
    assert r.status_code == 429 and r.json()["detail"]["error"] == "daily_ad_cap_reached"
    assert balance(db, u) == 2 * ra.REWARDED_AD_COINS


def test_old_session_expires(env):
    c, db, u = env["client"], env["db"], env["user"]
    s = start(env)
    db.execute("update public.rewarded_ad_sessions set created_at = now() - interval '3 hours'")
    assert c.get(callback(s["session"], u)).status_code == 200      # handled, but too late to pay
    assert balance(db, u) == 0


def test_status_is_private(env):
    c, db = env["client"], env["db"]
    s = start(env)
    env["current"]["id"] = str(make_user(db))
    assert c.get(f"/api/v1/coins/rewarded-ad/{s['session']}").status_code == 404


def test_self_claim_is_refused_once_verification_is_on(env):
    r = env["client"].post("/api/v1/coins/rewarded-ad/claim")
    assert r.status_code == 409 and r.json()["detail"]["error"] == "update_required"
    assert balance(env["db"], env["user"]) == 0


def test_switch_off_keeps_todays_behaviour(env, monkeypatch):
    monkeypatch.setenv("REWARDED_AD_SSV", "off")
    assert start(env)["verification"] == "client"


def test_console_verify_url_check_is_answered(env):
    """The AdMob console's test request is signed but carries no session or user."""
    params = "ad_network=1&ad_unit=" + UNIT + "&reward_amount=1&reward_item=Reward&timestamp=1&transaction_id=test"
    sig = GOOGLE_KEY.sign(params.encode(), ec.ECDSA(hashes.SHA256()))
    url = f"/api/v1/admob/ssv?{params}&signature={base64.urlsafe_b64encode(sig).decode().rstrip('=')}&key_id={KEY_ID}"
    assert env["client"].get(url).status_code == 200
