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
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING_EFFORT", "")
    client = FakeClient()
    assert r.selection_llm(client)([{"role": "user", "content": "x"}]) == '{"candidates": []}'
    assert "reasoning_effort" not in client.calls[0] and client.calls[0]["temperature"] == 0


def test_rejected_reasoning_falls_back_once_and_sticks(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING_EFFORT", "low")
    client = FakeClient(reject_reasoning=True)
    llm = r.selection_llm(client)
    llm([{"role": "user", "content": "x"}])
    llm([{"role": "user", "content": "y"}])
    assert ["reasoning_effort" in c for c in client.calls] == [True, False, False]


def test_rejection_without_reasoning_is_reported_not_looped(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING_EFFORT", "")
    client = FakeClient(reject_all=True)
    with pytest.raises(TransientJobError) as e:
        r.selection_llm(client)([{"role": "user", "content": "x"}])
    assert "rejected" in e.value.internal and len(client.calls) == 1


def test_selection_reasoning_never_leaks_into_the_other_calls(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING_EFFORT", "low")
    assert r._llm_kwargs(0.4) == {"temperature": 0.4}


def test_reasoning_model_gets_effort_and_no_temperature(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING_EFFORT", "low")
    client = FakeClient()
    r.selection_llm(client)([{"role": "user", "content": "x"}])
    assert client.calls[0]["reasoning_effort"] == "low" and "temperature" not in client.calls[0]


def _err(kind, status, msg):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return kind(msg, response=httpx.Response(status, request=req), body=None)


class UnavailableModel(FakeClient):
    """An account without access to `missing`: every call with it fails, other models work."""

    def __init__(self, missing, error):
        super().__init__()
        self.missing, self.error = missing, error

    def create(self, **kw):
        if kw["model"] == self.missing:
            self.calls.append(kw)
            raise self.error
        return super().create(**kw)


@pytest.mark.parametrize("error", [
    _err(openai.NotFoundError, 404, "The model `gpt-6-luna` does not exist or you do not have access to it."),
    _err(openai.PermissionDeniedError, 403, "You are not allowed to use this model."),
    _err(openai.BadRequestError, 400, "Invalid model: gpt-6-luna"),
])
def test_a_model_the_account_cant_use_falls_back_and_jobs_keep_working(monkeypatch, error):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_MODEL", "gpt-6-luna")
    monkeypatch.setattr(r, "REPURPOSE_FALLBACK_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING_EFFORT", "low")
    client = UnavailableModel("gpt-6-luna", error)
    llm = r.selection_llm(client)
    assert llm([{"role": "user", "content": "x"}]) == '{"candidates": []}'
    llm([{"role": "user", "content": "y"}])
    assert [c["model"] for c in client.calls] == ["gpt-6-luna", "gpt-4o-mini", "gpt-4o-mini"], "tried once, then stuck"
    assert "reasoning_effort" not in client.calls[1] and client.calls[1]["temperature"] == 0


def test_a_model_that_only_rejects_the_reasoning_setting_is_kept(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_MODEL", "gpt-6-luna")
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING_EFFORT", "high")

    class RejectsEffort(FakeClient):
        def create(self, **kw):
            if "reasoning_effort" in kw:
                self.calls.append(kw)
                raise bad_request("reasoning_effort is not supported with this model.")
            return super().create(**kw)
    client = RejectsEffort()
    r.selection_llm(client)([{"role": "user", "content": "x"}])
    assert [c["model"] for c in client.calls] == ["gpt-6-luna", "gpt-6-luna"]


def test_the_working_model_is_used_as_configured(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_MODEL", "gpt-6-luna")
    client = FakeClient()
    r.selection_llm(client)([{"role": "user", "content": "x"}])
    assert client.calls[0]["model"] == "gpt-6-luna"


def test_gpt_6_luna_is_the_default_selection_model_with_a_short_reasoning_level():
    import importlib
    import os
    saved = {k: os.environ.pop(k) for k in ("REPURPOSE_SELECTION_MODEL", "REPURPOSE_SELECTION_REASONING_EFFORT")
             if k in os.environ}
    try:
        m = importlib.reload(r)
        assert m.REPURPOSE_SELECTION_MODEL == "gpt-6-luna" and m.REPURPOSE_SELECTION_REASONING_EFFORT == "low"
        assert m.REPURPOSE_FALLBACK_MODEL == "gpt-4o-mini"
    finally:
        os.environ.update(saved)
        importlib.reload(r)


def test_a_model_that_rejects_temperature_is_retried_without_it(monkeypatch):
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_MODEL", "gpt-6-luna")
    monkeypatch.setattr(r, "REPURPOSE_SELECTION_REASONING_EFFORT", "")

    class RejectsTemperature(FakeClient):
        def create(self, **kw):
            if "temperature" in kw:
                self.calls.append(kw)
                raise bad_request("Unsupported value: 'temperature' does not support 0 with this model.")
            return super().create(**kw)
    client = RejectsTemperature()
    llm = r.selection_llm(client)
    llm([{"role": "user", "content": "x"}])
    llm([{"role": "user", "content": "y"}])
    assert [("temperature" in c, c["model"]) for c in client.calls] == [
        (True, "gpt-6-luna"), (False, "gpt-6-luna"), (False, "gpt-6-luna")]
