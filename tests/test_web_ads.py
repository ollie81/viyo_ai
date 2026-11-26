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
                                "infeed_slot": None, "infeed_layout_key": None, "multiplex_slot": None}


def test_on_only_when_switched_on(monkeypatch):
    assert get(monkeypatch, WEB_ADS_ENABLED="true")["enabled"] is True
    assert get(monkeypatch, WEB_ADS_ENABLED="yes")["enabled"] is False


def test_a_malformed_id_keeps_ads_off(monkeypatch):
    out = get(monkeypatch, WEB_ADS_ENABLED="true", WEB_ADS_CLIENT="pub-123")
    assert out["enabled"] is False and out["client"] is None and out["slot"] is None


def test_in_feed_needs_its_slot_and_layout_key_and_multiplex_its_slot(monkeypatch):
    out = get(monkeypatch, WEB_ADS_INFEED_SLOT="1234567890")
    assert out["infeed_slot"] is None                                  # no layout key: not used
    out = get(monkeypatch, WEB_ADS_INFEED_SLOT="1234567890", WEB_ADS_INFEED_LAYOUT_KEY="-fb+5w+4e-db+86",
              WEB_ADS_MULTIPLEX_SLOT="9876543210")
    assert out["infeed_slot"] == "1234567890" and out["infeed_layout_key"] == "-fb+5w+4e-db+86"
    assert out["multiplex_slot"] == "9876543210"
    assert get(monkeypatch, WEB_ADS_MULTIPLEX_SLOT="abc")["multiplex_slot"] is None
