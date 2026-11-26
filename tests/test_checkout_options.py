"""GET /api/v1/checkout/options: only genuinely configured checkouts are reported; Lemon Squeezy is never offered;
Premium plans are listed only while NardoPay subscriptions can really be started."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import checkout_options as co
import flutterwave_payments as fw
import lemonsqueezy_payments as ls
import nardopay_payments as nd
import payments as p
import paystack_payments as ps


@pytest.fixture
def nothing(monkeypatch):
    """Every provider unconfigured, NardoPay switched off."""
    for mod, names in ((p, ["STRIPE_SECRET_KEY", "STRIPE_PUBLISHABLE_KEY", "STRIPE_WEBHOOK_SECRET"]),
                       (ps, ["PAYSTACK_SECRET_KEY"]),
                       (fw, ["FLUTTERWAVE_SECRET_KEY", "FLUTTERWAVE_WEBHOOK_SECRET_HASH"])):
        for n in names:
            monkeypatch.setattr(mod, n, "")
    for mod in (p, ps, fw):
        monkeypatch.setattr(mod, "supabase_admin", object())
    monkeypatch.setattr(nd, "checkout_problem", lambda: "switched off")
    monkeypatch.setattr(nd, "_premium_problem", lambda: "switched off")


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
    assert co.coin_providers() == []                       # no webhook secrets: coins could never be credited
    monkeypatch.setattr(p, "STRIPE_WEBHOOK_SECRET", "wh")
    monkeypatch.setattr(fw, "FLUTTERWAVE_WEBHOOK_SECRET_HASH", "h")
    monkeypatch.setattr(ps, "PAYSTACK_SECRET_KEY", "ps")
    monkeypatch.setattr(nd, "checkout_problem", lambda: None)
    assert co.coin_providers() == ["stripe", "paystack", "flutterwave", "nardopay"]


def test_lemon_squeezy_is_never_offered_even_when_fully_configured(nothing, monkeypatch):
    for n in ("LEMONSQUEEZY_API_KEY", "LEMONSQUEEZY_STORE_ID", "LEMONSQUEEZY_VARIANT_ID", "LEMONSQUEEZY_WEBHOOK_SECRET"):
        monkeypatch.setattr(ls, n, "x")
    monkeypatch.setattr(ls, "supabase_admin", object())
    body = client().get("/api/v1/checkout/options").json()
    assert "lemonsqueezy" not in body["coins"]["providers"]
    assert body["premium"]["provider"] is None


def test_nardopay_coins_only_when_its_checkout_is_genuinely_available(nothing, monkeypatch):
    for problem in ("switched off", "not configured", "The exchange rate is out of date."):
        monkeypatch.setattr(nd, "checkout_problem", lambda problem=problem: problem)
        assert "nardopay" not in co.coin_providers()
    monkeypatch.setattr(nd, "checkout_problem", lambda: None)
    assert co.coin_providers() == ["nardopay"]


def test_premium_plans_are_the_server_prices_only_when_nardopay_subscriptions_can_start(nothing, monkeypatch):
    for problem in ("switched off", "not configured", "not set up"):
        monkeypatch.setattr(nd, "_premium_problem", lambda problem=problem: problem)
        assert not any(x["available"] for x in co.premium_plans()["plans"])
    monkeypatch.setattr(nd, "_premium_problem", lambda: None)
    out = client().get("/api/v1/checkout/options").json()["premium"]
    assert out["provider"] == "nardopay"
    assert [(x["plan"], x["available"], x["price_cents"], x["currency"], x["interval"]) for x in out["plans"]] == [
        ("weekly", True, 399, "USD", "week"), ("monthly", True, 999, "USD", "month"),
        ("yearly", True, 5999, "USD", "year")]
