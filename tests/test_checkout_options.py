"""GET /api/v1/checkout/options: only genuinely configured checkouts are reported, Premium prices come only from
Lemon Squeezy (mocked here, no real call), and nothing is invented when a price can't be read."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import checkout_options as co
import flutterwave_payments as fw
import lemonsqueezy_payments as ls
import nardopay_payments as nd
import payments as p
import paystack_payments as ps
import subscriptions as sub


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


@pytest.fixture
def nothing(monkeypatch):
    """Every provider unconfigured, as in production today (NardoPay switched off)."""
    for mod, names in ((p, ["STRIPE_SECRET_KEY", "STRIPE_PUBLISHABLE_KEY", "STRIPE_WEBHOOK_SECRET"]),
                       (ps, ["PAYSTACK_SECRET_KEY"]),
                       (fw, ["FLUTTERWAVE_SECRET_KEY", "FLUTTERWAVE_WEBHOOK_SECRET_HASH"]),
                       (ls, ["LEMONSQUEEZY_API_KEY", "LEMONSQUEEZY_STORE_ID", "LEMONSQUEEZY_VARIANT_ID",
                             "LEMONSQUEEZY_WEBHOOK_SECRET"]),
                       (sub, ["LEMONSQUEEZY_API_KEY", "LEMONSQUEEZY_STORE_ID"])):
        for n in names:
            monkeypatch.setattr(mod, n, "")
    for mod in (p, ps, fw, ls, sub):
        monkeypatch.setattr(mod, "supabase_admin", object())
    monkeypatch.setattr(nd, "checkout_problem", lambda: "switched off")
    monkeypatch.setattr(co, "_cache", {"at": 0.0, "key": None, "value": None})


def client():
    app = FastAPI()
    app.include_router(co.router)
    return TestClient(app)


def test_nothing_configured_offers_nothing_and_keeps_real_package_prices(nothing):
    r = client().get("/api/v1/checkout/options")
    assert r.status_code == 200
    body = r.json()
    assert body["coins"]["providers"] == []
    assert {x["id"]: x["usd_cents"] for x in body["coins"]["packages"]} == {
        k: v["usd_cents"] for k, v in p.COIN_PACKAGES.items()}
    assert body["premium"]["provider"] is None
    assert [x["plan"] for x in body["premium"]["plans"]] == ["weekly", "monthly", "yearly"]
    assert all(x["available"] is False and x["price_cents"] is None and x["reason"] for x in body["premium"]["plans"])


def test_a_checkout_without_its_webhook_verification_is_not_offered(nothing, monkeypatch):
    monkeypatch.setattr(p, "STRIPE_SECRET_KEY", "sk")
    monkeypatch.setattr(p, "STRIPE_PUBLISHABLE_KEY", "pk")
    monkeypatch.setattr(fw, "FLUTTERWAVE_SECRET_KEY", "fw")
    for n in ("LEMONSQUEEZY_API_KEY", "LEMONSQUEEZY_STORE_ID", "LEMONSQUEEZY_VARIANT_ID"):
        monkeypatch.setattr(ls, n, "x")
    assert co.coin_providers() == []                       # no webhook secrets: coins could never be credited
    monkeypatch.setattr(p, "STRIPE_WEBHOOK_SECRET", "wh")
    monkeypatch.setattr(fw, "FLUTTERWAVE_WEBHOOK_SECRET_HASH", "h")
    monkeypatch.setattr(ls, "LEMONSQUEEZY_WEBHOOK_SECRET", "s")
    monkeypatch.setattr(ps, "PAYSTACK_SECRET_KEY", "ps")
    monkeypatch.setattr(nd, "checkout_problem", lambda: None)
    assert co.coin_providers() == ["stripe", "paystack", "flutterwave", "lemonsqueezy", "nardopay"]


def test_nardopay_only_when_its_checkout_is_genuinely_available(nothing, monkeypatch):
    for problem in ("switched off", "not configured", "The exchange rate is out of date."):
        monkeypatch.setattr(nd, "checkout_problem", lambda problem=problem: problem)
        assert "nardopay" not in co.coin_providers()
    monkeypatch.setattr(nd, "checkout_problem", lambda: None)
    assert co.coin_providers() == ["nardopay"]


@pytest.fixture
def premium_configured(nothing, monkeypatch):
    for mod in (ls, sub):
        monkeypatch.setattr(mod, "LEMONSQUEEZY_API_KEY", "ls-key")
        monkeypatch.setattr(mod, "LEMONSQUEEZY_STORE_ID", "77")
    monkeypatch.setattr(ls, "LEMONSQUEEZY_WEBHOOK_SECRET", "s")
    for plan, vid in (("weekly", "1"), ("monthly", "2"), ("yearly", "3")):
        monkeypatch.setattr(sub, f"LEMONSQUEEZY_{plan.upper()}_VARIANT_ID", vid)
    monkeypatch.setattr(sub, "_VARIANT_BY_PLAN", {"weekly": "1", "monthly": "2", "yearly": "3"})


def fake_ls(prices, currency="USD", calls=None):
    def get(url, timeout=None, headers=None):
        if calls is not None:
            calls.append(url)
        assert headers["Authorization"] == "Bearer ls-key"
        if url.endswith("/stores/77"):
            return Resp(200, {"data": {"attributes": {"currency": currency}}})
        vid = url.rsplit("/", 1)[1]
        price, interval = prices[vid]
        if price is None:
            return Resp(404, {})
        return Resp(200, {"data": {"attributes": {"price": price, "is_subscription": True, "interval": interval,
                                                  "interval_count": 1}}})
    return get


def test_premium_prices_come_from_lemon_squeezy_and_are_cached(premium_configured):
    calls = []
    get = fake_ls({"1": (199, "week"), "2": (599, "month"), "3": (4999, "year")}, calls=calls)
    out = co.premium_plans(get=get, now=1000)
    assert out["provider"] == "lemonsqueezy"
    assert [(x["plan"], x["available"], x["price_cents"], x["currency"], x["interval"]) for x in out["plans"]] == [
        ("weekly", True, 199, "USD", "week"), ("monthly", True, 599, "USD", "month"),
        ("yearly", True, 4999, "USD", "year")]
    n = len(calls)
    co.premium_plans(get=get, now=1000 + 60)
    assert len(calls) == n                                   # cached
    co.premium_plans(get=get, now=1000 + co.PRICE_TTL_SECONDS + 1)
    assert len(calls) == 2 * n                               # refreshed after an hour


def test_a_plan_whose_price_cant_be_read_is_unavailable_never_guessed(premium_configured):
    out = co.premium_plans(get=fake_ls({"1": (None, "week"), "2": (599, "month"), "3": (0, "year")}), now=1)
    by = {x["plan"]: x for x in out["plans"]}
    assert by["monthly"]["available"] is True and by["monthly"]["price_cents"] == 599
    for plan in ("weekly", "yearly"):
        assert by[plan]["available"] is False and by[plan]["price_cents"] is None and by[plan]["reason"]
    # A failed store lookup (no currency) makes every plan unavailable.
    def down(url, timeout=None, headers=None):
        raise ConnectionError("down")
    co._cache.update(at=0.0, key=None, value=None)
    assert not any(x["available"] for x in co.premium_plans(get=down, now=2)["plans"])


def test_premium_needs_the_webhook_that_activates_it(premium_configured, monkeypatch):
    monkeypatch.setattr(ls, "LEMONSQUEEZY_WEBHOOK_SECRET", "")
    out = co.premium_plans(get=fake_ls({"1": (199, "week"), "2": (599, "month"), "3": (4999, "year")}), now=1)
    assert out["provider"] is None and not any(x["available"] for x in out["plans"])
