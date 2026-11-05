"""The four coin-charged AI tools give the coins back when the AI fails."""
import types

import pytest
from fastapi.testclient import TestClient

import coach
import main

U = "11111111-aaaa-bbbb-cccc-000000000001"


class Boom:
    """An OpenAI client whose every call fails."""

    def __getattr__(self, name):
        return self

    def create(self, *a, **k):
        raise RuntimeError("openai down")


class FakePosts:
    def table(self, name):
        return self

    def select(self, *a):
        return self

    def eq(self, *a):
        return self

    def neq(self, *a):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a):
        return self

    def execute(self):
        return types.SimpleNamespace(data=[{"id": "p1", "user_id": U, "caption": "Hi", "like_count": 3, "comment_count": 1}])


@pytest.fixture
def client(monkeypatch):
    ledger = []

    def spend(admin, user_id, feature):
        ledger.append(("spend", feature, 5))
        return 5

    def refund(admin, user_id, feature, charged):
        ledger.append(("refund", feature, charged))

    for mod in (main, coach):
        monkeypatch.setattr(mod, "spend_on_feature", spend)
        monkeypatch.setattr(mod, "refund_feature", refund)
    monkeypatch.setattr(main, "_call_openai_with_retry", lambda **k: (_ for _ in ()).throw(RuntimeError("openai down")))
    monkeypatch.setattr(main, "_check_rate_limit", lambda uid: None)
    monkeypatch.setattr(coach, "ai_client", Boom())
    monkeypatch.setattr(coach, "supabase_admin", FakePosts())
    if hasattr(coach, "_check_caption_variants_rate_limit"):
        monkeypatch.setattr(coach, "_check_caption_variants_rate_limit", lambda uid: None)
    main.app.dependency_overrides[main.get_current_user_id_no_guest] = lambda: U
    main.app.dependency_overrides[coach._get_current_user_id_no_guest] = lambda: U
    yield TestClient(main.app), ledger
    main.app.dependency_overrides.clear()


@pytest.mark.parametrize("path,body,feature", [
    ("/improve-caption", {"caption": "my caption"}, "improve_caption"),
    ("/analyze-hook", {"hook_text": "Wait for it"}, "hook_check"),
    ("/api/v1/caption-variants", {"draft": "my draft"}, "caption_variants"),
])
def test_failed_ai_call_refunds_the_coins(client, path, body, feature):
    c, ledger = client
    r = c.post(path, json=body)
    assert r.status_code == 502
    assert ledger == [("spend", feature, 5), ("refund", feature, 5)]


def test_post_insight_refunds_when_only_the_numbers_could_be_shown(client):
    c, ledger = client
    r = c.get("/api/v1/post-insight/p1")
    assert r.status_code == 200 and "4 likes and comments" in r.json()["explanation"]
    assert ledger == [("spend", "post_insight", 5), ("refund", "post_insight", 5)]
