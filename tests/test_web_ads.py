"""The web ads switch: off unless WEB_ADS_ENABLED=true, and only with a well-formed publisher id and slot."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

import web_ads


def get(monkeypatch, **env):
    for k in ("WEB_ADS_ENABLED", "WEB_ADS_CLIENT", "WEB_ADS_SLOT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app = FastAPI()
    app.include_router(web_ads.router)
    return TestClient(app).get("/api/v1/web-ads/config").json()


def test_off_by_default_with_the_real_ids(monkeypatch):
    assert get(monkeypatch) == {"enabled": False, "client": "ca-pub-4161816109605295", "slot": "6118261645"}


def test_on_only_when_switched_on(monkeypatch):
    assert get(monkeypatch, WEB_ADS_ENABLED="true")["enabled"] is True
    assert get(monkeypatch, WEB_ADS_ENABLED="yes")["enabled"] is False


def test_a_malformed_id_keeps_ads_off(monkeypatch):
    out = get(monkeypatch, WEB_ADS_ENABLED="true", WEB_ADS_CLIENT="pub-123")
    assert out == {"enabled": False, "client": None, "slot": None}
