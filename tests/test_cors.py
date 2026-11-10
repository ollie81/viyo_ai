"""The web app's browser asks permission (a CORS preflight) before every
call that isn't a plain GET. A method missing from the server's list gets a
400 and the app shows "Couldn't reach Viyo"; this keeps every method the app
uses allowed."""
import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize("path", ["/api/v1/studio/elements/abc", "/api/v1/studio/projects/abc",
                                  "/api/v1/studio/characters/abc/identity"])
def test_preflight_allows_every_method_the_app_uses(method, path):
    r = client.options(path, headers={"Origin": "https://viyo.site", "Access-Control-Request-Method": method,
                                      "Access-Control-Request-Headers": "authorization,content-type"})
    assert r.status_code == 200, r.text
    assert method in r.headers["access-control-allow-methods"]
    assert r.headers["access-control-allow-origin"] == "https://viyo.site"
