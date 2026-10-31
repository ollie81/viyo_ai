"""The selection LLM call must survive a model that rejects the
configured reasoning level, and must not keep retrying it."""
import httpx
import openai
import pytest

import repurpose as r
from repurpose_jobs import TransientJobError


def bad_request(msg):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


class FakeClient:
    def __init__(self, reject_efforts):
        self.reject = set(reject_efforts)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        if kw["reasoning_effort"] in self.reject:
            raise bad_request(f"Unsupported value: reasoning_effort {kw['reasoning_effort']}")
        msg = type("M", (), {"content": '{"candidates": []}'})
        return type("R", (), {"choices": [type("C", (), {"message": msg, "finish_reason": "stop"})]})


def test_rejected_reasoning_level_falls_back_once_and_sticks(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING", "low")
    client = FakeClient(reject_efforts={"low"})
    llm = r.selection_llm(client)
    assert llm([{"role": "user", "content": "x"}]) == '{"candidates": []}'
    assert [c["reasoning_effort"] for c in client.calls] == ["low", "none"]
    assert client.calls[1]["temperature"] == 0.3
    llm([{"role": "user", "content": "y"}])
    assert [c["reasoning_effort"] for c in client.calls] == ["low", "none", "none"]


def test_rejection_of_the_proven_settings_is_reported_not_looped(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING", "low")
    client = FakeClient(reject_efforts={"low", "none"})
    with pytest.raises(TransientJobError) as e:
        r.selection_llm(client)([{"role": "user", "content": "x"}])
    assert "LLM error" in e.value.internal
    assert len(client.calls) == 2
