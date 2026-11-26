"""The web ads switch: off unless WEB_ADS_ENABLED=true, and only with a well-formed publisher id and slot."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

import web_ads


def get(monkeypatch, **env):
    for k in ("WEB_ADS_ENABLED", "WEB_ADS_CLIENT", "WEB_ADS_SLOT", "WEB_ADS_INFEED_SLOT", "WEB_ADS_INFEED_LAYOUT_KEY",
              "WEB_ADS_MULTIPLEX_SLOT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app = FastAPI()
    app.include_router(web_ads.router)
    return TestClient(app).get("/api/v1/web-ads/config").json()


def test_off_by_default_with_the_real_ids(monkeypatch):
    assert get(monkeypatch) == {"enabled": False, "client": "ca-pub-4161816109605295", "slot": "6118261645",
                                "infeed_slot": "3636015645", "infeed_layout_key": "-fb+5w+4e-db+86",
                                "multiplex_slot": "2527655942"}


def test_on_only_when_switched_on(monkeypatch):
    assert get(monkeypatch, WEB_ADS_ENABLED="true")["enabled"] is True
    assert get(monkeypatch, WEB_ADS_ENABLED="yes")["enabled"] is False


def test_a_malformed_id_keeps_ads_off(monkeypatch):
    out = get(monkeypatch, WEB_ADS_ENABLED="true", WEB_ADS_CLIENT="pub-123")
    assert out["enabled"] is False and out["client"] is None and out["slot"] is None


def test_the_three_real_units_by_default_and_ads_stay_off(monkeypatch):
    out = get(monkeypatch)
    assert out["enabled"] is False
    assert (out["slot"], out["infeed_slot"], out["multiplex_slot"]) == ("6118261645", "3636015645", "2527655942")


def test_a_bad_override_turns_that_unit_off(monkeypatch):
    out = get(monkeypatch, WEB_ADS_INFEED_LAYOUT_KEY="bad key!", WEB_ADS_MULTIPLEX_SLOT="abc")
    assert out["infeed_slot"] is None and out["infeed_layout_key"] is None and out["multiplex_slot"] is None
