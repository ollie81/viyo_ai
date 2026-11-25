"""Automatic USD/RWF rates for NardoPay checkout, from the National Bank of Rwanda's published average rate.

Sources, tried in order (NARDOPAY_RATE_SOURCES, default "bnr,allratestoday")
  bnr            BNR's own endpoint, https://fxrates.bnr.rw/currency_history/?currency_name=USD&start_date&end_date
                 (no key; BNR's average_rate and post_date). BNR doesn't document it publicly: formal access is by
                 application to BNR. Checked working on 25 Nov 2026 (average 1477.88).
  allratestoday  AllRatesToday's copy of BNR's table, https://allratestoday.com/api/v1/central-bank/bnrw/latest
                 ?source=USD&target=RWF with Authorization: Bearer ALLRATESTODAY_API_KEY (free plan). Only a
                 published (not derived) "reference" rate, which is BNR's average, is accepted.

Every day (worker, NARDOPAY_RATE_AUTO=on, after NARDOPAY_RATE_FETCH_HOUR Kigali time) the newest publication is
stored with its source, publication date and fetch time. Rows are only ever added: an order keeps the rate it was
created with. A rate outside NARDOPAY_RATE_MIN..MAX, or more than 10% away from the rate in use, is stored as
REJECTED (kept for review, never used). Every attempt is logged in nardopay_rate_runs, so a failure is visible.

Which rate checkout uses (current_rate)
  * a manual rate the admin set with a hold (hold_until in the future) wins;
  * otherwise the newer of: the latest manual rate, and the latest automatic rate published AFTER the day that manual
    rate was set (so a manual correction holds until BNR publishes again);
  * rejected rates are never used.
  Freshness: an automatic rate's age is counted from its publication date, a manual one's from when it was set;
  past NARDOPAY_RATE_MAX_AGE_HOURS checkout pauses. An old rate is never used silently when fetching fails.
"""
from __future__ import annotations

import datetime
import decimal
import os
import re
import threading
from typing import Callable, Optional

import requests

KIGALI = datetime.timezone(datetime.timedelta(hours=2))      # Rwanda has no daylight saving time
BNR_URL = "https://fxrates.bnr.rw/currency_history/"
ART_URL = "https://allratestoday.com/api/v1/central-bank/bnrw/latest"
LOOKBACK_DAYS = 7
LARGE_CHANGE = 0.10
TIMEOUT = 20


class RateSourceError(Exception):
    pass


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _dec(x) -> Optional[decimal.Decimal]:
    try:
        d = decimal.Decimal(str(x).replace(",", "").strip())
    except (decimal.InvalidOperation, AttributeError):
        return None
    return d if d.is_finite() and d > 0 else None


def _parse_time(v) -> datetime.datetime:
    t = datetime.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def _date(v) -> Optional[datetime.date]:
    if isinstance(v, datetime.date) and not isinstance(v, datetime.datetime):
        return v
    try:
        return datetime.date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


# ====================================================================== sources

def fetch_bnr(get: Callable = requests.get, today: Optional[datetime.date] = None) -> dict:
    """BNR's newest published USD average rate: {"rate", "published_on", "rate_type", "source"}."""
    today = today or now().astimezone(KIGALI).date()
    params = {"currency_name": "USD", "start_date": (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat(),
              "end_date": today.isoformat()}
    try:
        r = get(BNR_URL, params=params, timeout=TIMEOUT)
    except Exception as e:
        raise RateSourceError(f"BNR unreachable ({type(e).__name__})")
    if r.status_code != 200:
        raise RateSourceError(f"BNR answered HTTP {r.status_code}")
    try:
        rows = r.json()
    except ValueError:
        raise RateSourceError("BNR's answer isn't JSON")
    best = None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or str(row.get("currency_name") or "").upper() != "USD":
            continue
        try:
            day = datetime.datetime.strptime(str(row.get("post_date") or ""), "%d-%b-%y").date()
        except ValueError:
            continue
        rate = _dec(row.get("average_rate"))
        if rate is None or day > today:
            continue
        if best is None or day > best[0]:
            best = (day, rate)
    if best is None:
        raise RateSourceError("BNR returned no USD average rate for the last 7 days")
    return {"source": "bnr", "rate": best[1], "published_on": best[0], "rate_type": "average"}


def fetch_allratestoday(get: Callable = requests.get, today: Optional[datetime.date] = None) -> dict:
    """AllRatesToday's copy of BNR's published reference (average) USD rate."""
    key = _env("ALLRATESTODAY_API_KEY")
    if not key:
        raise RateSourceError("ALLRATESTODAY_API_KEY isn't set")
    today = today or now().astimezone(KIGALI).date()
    try:
        r = get(ART_URL, params={"source": "USD", "target": "RWF"},
                headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
    except Exception as e:
        raise RateSourceError(f"AllRatesToday unreachable ({type(e).__name__})")
    if r.status_code != 200:
        raise RateSourceError(f"AllRatesToday answered HTTP {r.status_code}")
    try:
        d = r.json()
    except ValueError:
        raise RateSourceError("AllRatesToday's answer isn't JSON")
    if not isinstance(d, dict):
        raise RateSourceError("AllRatesToday's answer has an unexpected shape")
    if str(d.get("source") or "").upper() != "USD" or str(d.get("target") or "").upper() != "RWF":
        raise RateSourceError("AllRatesToday answered for a different currency pair")
    if str(d.get("rate_type") or "").lower() != "reference" or d.get("derived") is not False:
        raise RateSourceError("AllRatesToday's rate isn't BNR's published reference rate")
    rate, day = _dec(d.get("rate")), _date(d.get("rate_date"))
    if rate is None or day is None or day > today:
        raise RateSourceError("AllRatesToday's rate or date is missing")
    return {"source": "allratestoday", "rate": rate, "published_on": day, "rate_type": "average"}


SOURCES = {"bnr": fetch_bnr, "allratestoday": fetch_allratestoday}


def sources() -> list[str]:
    names = [s.strip().lower() for s in _env("NARDOPAY_RATE_SOURCES", "bnr,allratestoday").split(",") if s.strip()]
    return [n for n in names if n in SOURCES]


# ====================================================================== which rate is in use

def bounds() -> tuple[float, float]:
    return (float(_env("NARDOPAY_RATE_MIN", "1000") or 1000), float(_env("NARDOPAY_RATE_MAX", "3000") or 3000))


def current_rate(rows: list[dict], at: Optional[datetime.datetime] = None) -> Optional[dict]:
    """The rate checkout uses now, from the rate history (see the module docstring). Rejected rates never count."""
    at = at or now()
    active = [r for r in rows if (r.get("status") or "active") == "active"]
    manual = sorted((r for r in active if (r.get("source") or "manual") == "manual"),
                    key=lambda r: _parse_time(r["set_at"]), reverse=True)
    auto = sorted((r for r in active if (r.get("source") or "manual") != "manual" and _date(r.get("published_on"))),
                  key=lambda r: (_date(r["published_on"]), _parse_time(r.get("fetched_at") or r["set_at"])), reverse=True)
    held = [r for r in manual if r.get("hold_until") and _parse_time(r["hold_until"]) > at]
    if held:
        return held[0]
    m, a = (manual[0] if manual else None), (auto[0] if auto else None)
    if m and a:
        manual_day = _parse_time(m["set_at"]).astimezone(KIGALI).date()
        return a if _date(a["published_on"]) > manual_day else m
    return m or a


def age_hours(rate: dict, at: Optional[datetime.datetime] = None) -> float:
    """How old a rate is: an automatic one from the start of its publication day (Kigali), a manual one from when
    it was set."""
    at = at or now()
    if (rate.get("source") or "manual") != "manual" and _date(rate.get("published_on")):
        start = datetime.datetime.combine(_date(rate["published_on"]), datetime.time(0, 0), KIGALI)
    else:
        start = _parse_time(rate["set_at"])
    return (at - start).total_seconds() / 3600


def problem(rate: Optional[dict], at: Optional[datetime.datetime] = None) -> Optional[str]:
    """Why the rate can't be used for a new order, or None."""
    if not rate:
        return "No exchange rate has been set."
    max_age = float(_env("NARDOPAY_RATE_MAX_AGE_HOURS", "168") or 168)
    if age_hours(rate, at) > max_age:
        return "The exchange rate is out of date."
    return None


# ====================================================================== the daily update

def _log(store, ok: bool, source: str, detail: str, rate_id: Optional[str] = None) -> None:
    store.insert("nardopay_rate_runs", [{"ok": ok, "source": source, "detail": detail[:1000], "rate_id": rate_id}])


def update(store, get: Callable = requests.get, at: Optional[datetime.datetime] = None) -> dict:
    """Fetches the newest BNR rate from the first source that answers and stores it (or records why not)."""
    at = at or now()
    today = at.astimezone(KIGALI).date()
    errors = []
    for name in sources():
        try:
            got = SOURCES[name](get=get, today=today)
        except RateSourceError as e:
            errors.append(f"{name}: {e}")
            continue
        rows = store.select("nardopay_rates", {})
        same = [r for r in rows if r.get("source") == name and _date(r.get("published_on")) == got["published_on"]]
        if same:
            _log(store, True, name, f"already have BNR's rate of {got['published_on']}", same[0]["id"])
            return {"ok": True, "stored": False, "source": name, "published_on": str(got["published_on"])}
        newest_auto = max((_date(r.get("published_on")) for r in rows
                           if (r.get("source") or "manual") != "manual" and _date(r.get("published_on"))), default=None)
        if newest_auto and got["published_on"] < newest_auto:
            _log(store, True, name, f"{got['published_on']} is older than the rate already stored ({newest_auto})")
            return {"ok": True, "stored": False, "source": name, "published_on": str(got["published_on"])}
        lo, hi = bounds()
        value = float(got["rate"])
        reason = ""
        if not lo <= value <= hi:
            reason = f"{value:g} is outside {lo:g}-{hi:g} RWF per USD"
        else:
            cur = current_rate(rows, at)
            if cur and abs(value - float(cur["rate"])) / float(cur["rate"]) > LARGE_CHANGE:
                reason = f"{value:g} is more than {int(LARGE_CHANGE * 100)}% away from the rate in use ({float(cur['rate']):g})"
        row = store.insert("nardopay_rates", [{
            "rate": str(got["rate"]), "set_by": f"{name}-auto", "source": name, "rate_type": got["rate_type"],
            "published_on": got["published_on"].isoformat(), "fetched_at": at.isoformat(),
            "status": "rejected" if reason else "active", "reject_reason": reason,
            "note": f"BNR average rate published {got['published_on']}"}])[0]
        if reason:
            print(f"[WARN] NardoPay rate from {name} rejected: {reason}")
            _log(store, False, name, f"rejected: {reason}", row["id"])
            return {"ok": False, "stored": True, "rejected": reason, "source": name}
        _log(store, True, name, f"stored {value:g} published {got['published_on']}", row["id"])
        return {"ok": True, "stored": True, "source": name, "rate": value, "published_on": str(got["published_on"])}
    detail = "; ".join(errors) or "no rate source is configured"
    print(f"[WARN] NardoPay rate update failed: {detail}")
    _log(store, False, ",".join(sources()), detail)
    return {"ok": False, "stored": False, "error": detail}


def due(store, at: Optional[datetime.datetime] = None) -> bool:
    """True once per Kigali day after the fetch hour, until a fetch that day has succeeded."""
    at = at or now()
    local = at.astimezone(KIGALI)
    if local.hour < int(_env("NARDOPAY_RATE_FETCH_HOUR", "10") or 10):
        return False
    runs = store.select("nardopay_rate_runs", {"ok": True}, order="ran_at", desc=True, limit=1)
    return not runs or _parse_time(runs[0]["ran_at"]).astimezone(KIGALI).date() < local.date()


def updater_forever(get_store: Callable, every_seconds: int = 1800, stop=None) -> None:
    """Checks every half hour; fetches when due (so a failed morning is retried later the same day)."""
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            store = get_store()
            if due(store):
                out = update(store)
                print(f"[INFO] NardoPay rate update: {out}")
        except Exception as e:
            print(f"[WARN] NardoPay rate updater error: {type(e).__name__}: {str(e)[:160]}")
        stop.wait(every_seconds)


def start_updater(get_store: Callable) -> bool:
    """Started by the worker. Off unless NARDOPAY_RATE_AUTO=on."""
    if _env("NARDOPAY_RATE_AUTO").lower() not in ("on", "true", "1", "yes"):
        return False
    threading.Thread(target=updater_forever, args=(get_store,), name="nardopay-rate", daemon=True).start()
    print(f"[INFO] NardoPay daily rate updates on (sources: {', '.join(sources()) or 'none'})")
    return True
