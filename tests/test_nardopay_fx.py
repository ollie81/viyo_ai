"""Automatic USD/RWF rates (nardopay_fx.py) for NardoPay checkout: real Postgres, BNR and AllRatesToday MOCKED (no
real HTTP call). Proves: parsing, source fallback, history kept, orders keep their rate, stale/failed fetch pauses
checkout, jumps and out-of-bounds rates rejected, manual override and hold, no duplicates."""
import datetime

import pytest

import nardopay_fx as fx
import nardopay_payments as np_
from conftest import make_user
from test_nardopay_payments import ADMIN, H, FakeResponse, env, schema  # noqa: F401

KIGALI = fx.KIGALI


def at(y, m, d, hour=11):
    return datetime.datetime(y, m, d, hour, 0, tzinfo=KIGALI)


def bnr_row(day, rate, cur="USD"):
    return {"currency_name": cur, "average_rate": rate, "buying_rate": "1", "selling_rate": "1",
            "post_date": day.strftime("%d-%b-%y")}


def art_body(rate="1477.88", day="2026-11-25", **kw):
    return {"source": "USD", "target": "RWF", "rate": rate, "rate_type": "reference", "rate_date": day,
            "derived": False, "stale": False, "age_days": 0, **kw}


class FakeGet:
    """Answers BNR and AllRatesToday requests from canned responses; records every call."""
    def __init__(self, bnr=None, art=None):
        self.bnr, self.art, self.calls = bnr, art, []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        which = self.bnr if url == fx.BNR_URL else self.art
        if isinstance(which, Exception):
            raise which
        if which is None:
            raise AssertionError(f"unexpected call to {url}")
        return which


@pytest.fixture
def fxenv(env, monkeypatch):
    client, db, calls, st = env
    monkeypatch.setenv("NARDOPAY_RATE_SOURCES", "bnr,allratestoday")
    monkeypatch.setenv("ALLRATESTODAY_API_KEY", "art_test_key")
    monkeypatch.setenv("NARDOPAY_RATE_MAX_AGE_HOURS", "96")
    return client, db, calls, st


def rows(db):
    return db.execute("select source, rate::float, published_on, status, reject_reason from public.nardopay_rates "
                      "order by set_at").fetchall()


def runs(db):
    return db.execute("select ok, source, detail from public.nardopay_rate_runs order by ran_at").fetchall()


# ------------------------------------------------------------------------------------------- parsing

def test_bnr_rows_are_parsed_newest_first_with_commas_and_unsorted_dates():
    today = datetime.date(2026, 11, 25)
    get = FakeGet(bnr=FakeResponse(200, [
        bnr_row(datetime.date(2026, 11, 21), "1,474.10"),
        bnr_row(datetime.date(2026, 11, 25), "1,477.88"),
        bnr_row(datetime.date(2026, 11, 24), "1476.50"),
        bnr_row(datetime.date(2026, 11, 25), "1600", cur="EUR"),            # another currency: ignored
        {"currency_name": "USD", "average_rate": "x", "post_date": "26-Nov-26"},  # bad rate
        {"currency_name": "USD", "average_rate": "1490", "post_date": "garbage"},
    ]))
    got = fx.fetch_bnr(get=get, today=today)
    assert got == {"source": "bnr", "rate": fx.decimal.Decimal("1477.88"), "published_on": today, "rate_type": "average"}
    assert get.calls[0]["params"] == {"currency_name": "USD", "start_date": "2026-11-18", "end_date": "2026-11-25"}
    assert get.calls[0]["headers"] is None                                 # BNR needs no key


@pytest.mark.parametrize("resp", [FakeResponse(500, []), FakeResponse(200, []), FakeResponse(200, ValueError("x")),
                                  FakeResponse(200, {"error": "x"}), ConnectionError("down")])
def test_bnr_failures_raise(resp):
    with pytest.raises(fx.RateSourceError):
        fx.fetch_bnr(get=FakeGet(bnr=resp), today=datetime.date(2026, 11, 25))


def test_allratestoday_accepts_only_bnrs_published_reference_rate(monkeypatch):
    monkeypatch.setenv("ALLRATESTODAY_API_KEY", "art_test_key")
    today = datetime.date(2026, 11, 25)
    get = FakeGet(art=FakeResponse(200, art_body()))
    got = fx.fetch_allratestoday(get=get, today=today)
    assert got["rate"] == fx.decimal.Decimal("1477.88") and got["published_on"] == today
    assert get.calls[0]["headers"] == {"Authorization": "Bearer art_test_key"}
    for bad in (art_body(derived=True), art_body(rate_type="mid"), art_body(target="KES"), art_body(rate="0"),
                art_body(day="2026-11-30"), {"derived": False}):
        with pytest.raises(fx.RateSourceError):
            fx.fetch_allratestoday(get=FakeGet(art=FakeResponse(200, bad)), today=today)
    monkeypatch.delenv("ALLRATESTODAY_API_KEY")
    with pytest.raises(fx.RateSourceError, match="isn't set"):
        fx.fetch_allratestoday(get=FakeGet(), today=today)


# ------------------------------------------------------------------------------------------- the daily update

def test_bnr_rate_is_stored_with_source_dates_and_used_by_checkout(fxenv):
    client, db, calls, st = fxenv
    now = at(2026, 11, 25)
    get = FakeGet(bnr=FakeResponse(200, [bnr_row(datetime.date(2026, 11, 25), "1,477.88")]))
    out = fx.update(np_.get_store(), get=get, at=now)
    assert out["ok"] and out["stored"] and out["source"] == "bnr"
    row = db.execute("select source, rate_type, published_on, fetched_at, status, set_by from public.nardopay_rates "
                     "where source = 'bnr'").fetchone()
    assert row[0] == "bnr" and row[1] == "average" and row[2] == datetime.date(2026, 11, 25)
    assert row[3] == now and row[4] == "active" and row[5] == "bnr-auto"
    assert runs(db)[-1][0] is True
    # Once the manual rate was set before BNR's publication day, BNR's rate is the one in use.
    db.execute("update public.nardopay_rates set set_at = '2026-11-20T08:00:00+02' where source = 'manual'")
    cur = fx.current_rate(np_.get_store().select("nardopay_rates", {}), now)
    assert cur["source"] == "bnr" and float(cur["rate"]) == 1477.88


def test_bnr_down_falls_back_to_allratestoday(fxenv):
    client, db, calls, st = fxenv
    get = FakeGet(bnr=ConnectionError("down"), art=FakeResponse(200, art_body("1477.88", "2026-11-25")))
    out = fx.update(np_.get_store(), get=get, at=at(2026, 11, 25))
    assert out["ok"] and out["source"] == "allratestoday"
    assert [r[0] for r in rows(db)] == ["manual", "allratestoday"]
    assert [c["url"] for c in get.calls] == [fx.BNR_URL, fx.ART_URL]


def test_both_sources_down_is_logged_and_never_silent(fxenv):
    client, db, calls, st = fxenv
    get = FakeGet(bnr=FakeResponse(503, []), art=FakeResponse(401, {}))
    out = fx.update(np_.get_store(), get=get, at=at(2026, 11, 25))
    assert out["ok"] is False and "bnr: BNR answered HTTP 503" in out["error"] and "HTTP 401" in out["error"]
    ok, source, detail = runs(db)[-1]
    assert ok is False and "503" in detail
    assert [r[0] for r in rows(db)] == ["manual"]                          # nothing added


def test_a_failed_fetch_with_only_an_old_rate_pauses_checkout(fxenv):
    client, db, calls, st = fxenv
    u = make_user(db)
    # Yesterday's BNR rate is good; the manual one is old.
    db.execute("update public.nardopay_rates set set_at = now() - interval '20 days'")
    today = datetime.datetime.now(KIGALI).date()
    store = np_.get_store()
    fx.update(store, get=FakeGet(bnr=FakeResponse(200, [bnr_row(today - datetime.timedelta(days=1), "1477")])))
    assert client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u),
                       json={"package_id": "starter"}).status_code == 200
    # Five days of failures later, that rate is past 96 hours: checkout pauses instead of using it.
    db.execute("update public.nardopay_rates set published_on = published_on - 5 where source = 'bnr'")
    fx.update(store, get=FakeGet(bnr=FakeResponse(500, []), art=FakeResponse(500, {})))
    r = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u), json={"package_id": "starter"})
    assert r.status_code == 503 and "out of date" in r.json()["detail"]
    assert client.get("/api/v1/coins/purchase/nardopay/availability").json() == {"available": False}
    info = client.get("/api/v1/admin/nardopay/rate", headers=ADMIN).json()
    assert info["usable"] is False and info["recent_runs"][0]["ok"] is False


def test_same_publication_is_stored_once_and_older_ones_never_replace_newer(fxenv):
    client, db, calls, st = fxenv
    store = np_.get_store()
    day = datetime.date(2026, 11, 25)
    for _ in range(3):
        fx.update(store, get=FakeGet(bnr=FakeResponse(200, [bnr_row(day, "1477.88")])), at=at(2026, 11, 25))
    assert [r[0] for r in rows(db)].count("bnr") == 1
    # The backup reporting the day before adds nothing.
    out = fx.update(store, get=FakeGet(bnr=FakeResponse(500, []), art=FakeResponse(200, art_body(day="2026-11-24"))),
                    at=at(2026, 11, 25, 15))
    assert out["stored"] is False and len(rows(db)) == 2


def test_out_of_bounds_and_big_jumps_are_kept_as_rejected_and_never_used(fxenv):
    client, db, calls, st = fxenv
    store = np_.get_store()
    db.execute("update public.nardopay_rates set set_at = '2026-11-20T08:00:00+02'")
    out = fx.update(store, get=FakeGet(bnr=FakeResponse(200, [bnr_row(datetime.date(2026, 11, 24), "147.788")])),
                    at=at(2026, 11, 24))
    assert out["ok"] is False and "outside 1000-3000" in out["rejected"]
    out = fx.update(store, get=FakeGet(bnr=FakeResponse(200, [bnr_row(datetime.date(2026, 11, 25), "1700")])),
                    at=at(2026, 11, 25))
    assert out["ok"] is False and "more than 10% away" in out["rejected"]
    assert [(r[0], r[3]) for r in rows(db)] == [("manual", "active"), ("bnr", "rejected"), ("bnr", "rejected")]
    cur = fx.current_rate(store.select("nardopay_rates", {}), at(2026, 11, 25))
    assert cur["source"] == "manual" and float(cur["rate"]) == 1410        # rejected rates never used
    assert [r[0] for r in runs(db)] == [False, False]
    info = client.get("/api/v1/admin/nardopay/rate", headers=ADMIN).json()
    assert len(info["rejected"]) == 2


def test_orders_keep_the_rate_they_were_created_with(fxenv):
    client, db, calls, st = fxenv
    u = make_user(db)
    first = client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u), json={"package_id": "starter"}).json()
    rate_before = db.execute("select rate_id, usd_rwf_rate::float, rwf_amount from public.nardopay_orders").fetchone()
    db.execute("update public.nardopay_rates set set_at = now() - interval '3 days'")
    today = datetime.datetime.now(KIGALI).date()
    fx.update(np_.get_store(), get=FakeGet(bnr=FakeResponse(200, [bnr_row(today, "1477.88")])))
    st["response"] = lambda body: FakeResponse(200, {"x_link": "https://pay.example/checkout/def"})
    client.post("/api/v1/coins/purchase/nardopay/create", headers=H(u), json={"package_id": "starter"})
    got = db.execute("select id::text, rate_id, usd_rwf_rate::float, rwf_amount from public.nardopay_orders "
                     "order by created_at").fetchall()
    assert got[0][0] == first["order_id"] and got[0][1:] == rate_before          # unchanged
    assert rate_before[1] == 1410 and rate_before[2] == 1396
    assert got[1][2] == 1477.88 and got[1][3] == 1464                            # 0.99 * 1477.88 = 1463.1 -> 1464
    assert db.execute("select count(*) from public.nardopay_rates").fetchone()[0] == 2   # history kept


# ------------------------------------------------------------------------------------------- manual override

def test_manual_rate_wins_until_bnr_publishes_again_and_a_hold_wins_regardless(fxenv):
    client, db, calls, st = fxenv
    store = np_.get_store()
    today = datetime.datetime.now(KIGALI).date()
    db.execute("delete from public.nardopay_rates")
    fx.update(store, get=FakeGet(bnr=FakeResponse(200, [bnr_row(today - datetime.timedelta(days=1), "1470")])))
    assert float(np_.latest_rate()["rate"]) == 1470
    # The admin corrects it today: the manual rate is used, even though BNR's latest is newer than nothing.
    assert client.post("/api/v1/admin/nardopay/rate", headers=ADMIN,
                       json={"rate": 1480, "set_by": "owner", "note": "check"}).status_code == 200
    assert float(np_.latest_rate()["rate"]) == 1480
    # BNR publishing tomorrow takes over again.
    tomorrow = datetime.datetime.now(KIGALI) + datetime.timedelta(days=1)
    fx.update(store, get=FakeGet(bnr=FakeResponse(200, [bnr_row(tomorrow.date(), "1478")])), at=tomorrow)
    rws = store.select("nardopay_rates", {})
    assert float(fx.current_rate(rws, tomorrow)["rate"]) == 1478
    # A held manual rate wins over BNR until the hold ends.
    r = client.post("/api/v1/admin/nardopay/rate", headers=ADMIN,
                    json={"rate": 1490, "set_by": "owner", "hold_hours": 72})
    assert r.status_code == 200 and r.json()["rate"]["hold_until"]
    rws = store.select("nardopay_rates", {})
    assert float(fx.current_rate(rws, tomorrow)["rate"]) == 1490
    assert float(fx.current_rate(rws, tomorrow + datetime.timedelta(days=4))["rate"]) == 1478
    assert client.post("/api/v1/admin/nardopay/rate", headers=ADMIN,
                       json={"rate": 1490, "set_by": "o", "hold_hours": 9999}).status_code == 422


def test_fetch_now_endpoint_is_admin_only_and_uses_the_same_checks(fxenv, monkeypatch):
    client, db, calls, st = fxenv
    assert client.post("/api/v1/admin/nardopay/rate/fetch").status_code == 403
    today = datetime.datetime.now(KIGALI).date()
    fake = FakeGet(bnr=FakeResponse(200, [bnr_row(today, "1477.88")]))
    real_bnr = fx.SOURCES["bnr"]
    monkeypatch.setitem(fx.SOURCES, "bnr", lambda get, today: real_bnr(get=fake, today=today))
    r = client.post("/api/v1/admin/nardopay/rate/fetch", headers=ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"]["ok"] and body["result"]["source"] == "bnr" and len(fake.calls) == 1
    # NardoPay activation settings are untouched by any of this.
    import os
    assert os.environ["NARDOPAY_ENABLED"] == "true" and os.environ["NARDOPAY_LEDGER_VERIFIED"] == "true"



# ------------------------------------------------------------------------------------------- schedule

def test_due_once_a_day_after_the_fetch_hour_and_retried_after_failure(fxenv, monkeypatch):
    client, db, calls, st = fxenv
    store = np_.get_store()
    monkeypatch.setenv("NARDOPAY_RATE_FETCH_HOUR", "10")
    assert fx.due(store, at(2026, 11, 25, 9)) is False                     # before 10:00 Kigali
    assert fx.due(store, at(2026, 11, 25, 10)) is True
    store.insert("nardopay_rate_runs", [{"ok": False, "source": "bnr", "detail": "down",
                                         "ran_at": at(2026, 11, 25, 10).isoformat()}])
    assert fx.due(store, at(2026, 11, 25, 11)) is True                     # failed: try again
    store.insert("nardopay_rate_runs", [{"ok": True, "source": "bnr", "detail": "ok",
                                         "ran_at": at(2026, 11, 25, 11).isoformat()}])
    assert fx.due(store, at(2026, 11, 25, 15)) is False                    # done for today
    assert fx.due(store, at(2026, 11, 26, 10)) is True


def test_updater_is_off_unless_switched_on(monkeypatch):
    started = []
    monkeypatch.setattr(fx.threading, "Thread", lambda **kw: started.append(kw) or type("T", (), {"start": lambda s: None})())
    monkeypatch.delenv("NARDOPAY_RATE_AUTO", raising=False)
    assert fx.start_updater(lambda: None) is False and started == []
    monkeypatch.setenv("NARDOPAY_RATE_AUTO", "on")
    assert fx.start_updater(lambda: None) is True and started[0]["name"] == "nardopay-rate"


def test_unknown_sources_are_ignored(monkeypatch):
    monkeypatch.setenv("NARDOPAY_RATE_SOURCES", "allratestoday, nope ,BNR")
    assert fx.sources() == ["allratestoday", "bnr"]
