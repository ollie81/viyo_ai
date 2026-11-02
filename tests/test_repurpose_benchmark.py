"""The benchmark endpoint is test-only: allow-listed users, cost accounting correct."""
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

import repurpose_benchmark as b


def client_as(uid):
    app = FastAPI()
    app.include_router(b.router)
    app.dependency_overrides[b._user] = lambda: uid
    return TestClient(app)


def test_everyone_else_gets_not_found():
    r = client_as("someone-else").post("/api/v1/repurpose/benchmark",
                                       json={"sha256": "a" * 64, "model": "gpt-6-luna", "video_seconds": 60})
    assert r.status_code == 404


def test_meter_sends_reasoning_effort_to_gpt6_and_temperature_to_older_models():
    sent = []

    class Comp:
        def create(self, **kw):
            sent.append(kw)
            usage = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=500,
                                          prompt_tokens_details=types.SimpleNamespace(cached_tokens=200),
                                          completion_tokens_details=types.SimpleNamespace(reasoning_tokens=300))
            msg = types.SimpleNamespace(content="{}")
            return types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Comp()))
    m = b.Meter(client, "gpt-6-luna", "low")
    m([{"role": "user", "content": "x"}])
    assert sent[0]["reasoning_effort"] == "low" and "temperature" not in sent[0]
    t = m.totals()
    # 800 uncached * 0.10 + 200 cached * 0.01 + 500 out * 0.50, per 1M
    assert abs(t["usd"] - (800 * 0.10 + 200 * 0.01 + 500 * 0.50) / 1e6) < 1e-9 and t["reasoning_tokens"] == 300
    m2 = b.Meter(client, "gpt-4o-mini", None)
    m2([{"role": "user", "content": "x"}])
    assert sent[1]["temperature"] == 0 and "reasoning_effort" not in sent[1]
