"""The selection LLM call must work with non-reasoning models (no
reasoning_effort sent) and survive a model that rejects a configured
reasoning level without retrying it forever."""
import httpx
import openai
import pytest

import repurpose as r
from repurpose_jobs import TransientJobError


def bad_request(msg):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


class FakeClient:
    def __init__(self, reject_reasoning=False, reject_all=False):
        self.reject_reasoning, self.reject_all = reject_reasoning, reject_all
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.reject_all or (self.reject_reasoning and "reasoning_effort" in kw):
            raise bad_request("Unsupported parameter")
        msg = type("M", (), {"content": '{"candidates": []}'})
        return type("R", (), {"choices": [type("C", (), {"message": msg, "finish_reason": "stop"})]})


def test_default_sends_temperature_and_no_reasoning(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_REASONING_EFFORT", "")
    client = FakeClient()
    assert r.selection_llm(client)([{"role": "user", "content": "x"}]) == '{"candidates": []}'
    assert "reasoning_effort" not in client.calls[0] and client.calls[0]["temperature"] == 0


def test_rejected_reasoning_falls_back_once_and_sticks(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_REASONING_EFFORT", "low")
    client = FakeClient(reject_reasoning=True)
    llm = r.selection_llm(client)
    llm([{"role": "user", "content": "x"}])
    llm([{"role": "user", "content": "y"}])
    assert ["reasoning_effort" in c for c in client.calls] == [True, False, False]


def test_rejection_without_reasoning_is_reported_not_looped(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_REASONING_EFFORT", "")
    client = FakeClient(reject_all=True)
    with pytest.raises(TransientJobError) as e:
        r.selection_llm(client)([{"role": "user", "content": "x"}])
    assert "rejected" in e.value.internal and len(client.calls) == 1


def test_other_calls_never_send_both_parameters(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_REASONING_EFFORT", "")
    assert r._llm_kwargs(0.4) == {"temperature": 0.4}
    monkeypatch.setattr(r, "REPURPOSE_REASONING_EFFORT", "low")
    assert r._llm_kwargs(0.4) == {"reasoning_effort": "low"}
