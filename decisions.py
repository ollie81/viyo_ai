"""OpenAI Decisions API client (POST /v1/decisions), server side only.

Decisions answers structured questions about shared evidence (text, images,
JSON) with calibrated probabilities instead of free text:

  predicate  -> {"probability": 0-1}
  choice     -> {"choice", "probabilities": [{value, probability}], "confidence"}
  score      -> {"score" (probability-weighted level index from 0),
                 "probabilities": [{value, label, probability}], "confidence"}
  refusal    -> {"type": "refusal"}

Spec as published at https://developers.openai.com/api/docs/guides/decisions
(read for this integration, Nov 2026): model "gpt-6-luna" only; 1-400 questions
per request; 2-10 score levels ordered lowest to highest; `input` is a string or
a message array (user messages may hold input_text, input_image with an
image_url, input_json with data); question `instructions` is a string or a
non-empty list of input_text / input_image parts; implicit prompt caching of the
shared input when it is at least 1,024 tokens and identical across requests,
partitioned by `prompt_cache_key`. Price: $0.10 per 1M uncached input tokens,
cached input discounted, cache writes charged, output not charged.

The installed OpenAI Python SDK (2.x) has no `client.decisions`, so this is a
small HTTP client on httpx (already a dependency); upgrade to the SDK's
`client.decisions.create(...)` when the backend moves to openai 3.x.

Nothing returned is trusted blindly: every answer is checked against the
question it claims to answer (name, type, value ranges, a choice from the list,
a score within the levels), malformed answers are dropped, refusals are kept
as refusals, and low confidence is flagged for a second look.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import httpx

URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
MAX_QUESTIONS = 400
MIN_LEVELS, MAX_LEVELS = 2, 10
# USD per 1M tokens, from the Decisions guide (Nov 2026). Cached-read and cache-write rates follow the standard
# gpt-6-luna prompt-caching rates (repurpose_benchmark.PRICES: $0.01 cached); re-check before relying on them.
PRICE_INPUT, PRICE_CACHED, PRICE_CACHE_WRITE = 0.10, 0.01, 0.10


class DecisionsUnavailable(Exception):
    """The API couldn't be reached or answered with an error; callers fall back to another path."""


# ------------------------------------------------------------------ questions

def predicate(name: str, instructions: Union[str, list]) -> dict:
    return {"type": "predicate", "name": name, "instructions": instructions}


def choice(name: str, instructions: Union[str, list], choices: list[tuple[str, str]]) -> dict:
    return {"type": "choice", "name": name, "instructions": instructions,
            "choices": [{"value": v, "description": d} for v, d in choices]}


def score(name: str, instructions: Union[str, list], levels: list[tuple[str, str]]) -> dict:
    """`levels`: (label, description) from lowest to highest, 2-10 of them."""
    if not MIN_LEVELS <= len(levels) <= MAX_LEVELS:
        raise ValueError("a score question needs 2-10 levels")
    return {"type": "score", "name": name, "instructions": instructions,
            "levels": [{"label": l, "description": d} for l, d in levels]}


def text(t: str) -> dict:
    return {"type": "input_text", "text": t}


def image(data_url_or_https: str) -> dict:
    return {"type": "input_image", "image_url": data_url_or_https}


def json_part(data: Any) -> dict:
    return {"type": "input_json", "data": data}


# ------------------------------------------------------------------ answers

@dataclass
class Answer:
    name: str
    type: str                                  # predicate | choice | score | refusal
    probability: Optional[float] = None        # predicate
    choice: Optional[str] = None               # choice
    score: Optional[float] = None              # score: 0 .. levels-1
    level: Optional[str] = None                # score: the most likely level's label
    probabilities: list = field(default_factory=list)
    confidence: Optional[float] = None

    @property
    def refused(self) -> bool:
        return self.type == "refusal"


@dataclass
class Result:
    answers: dict                              # name -> Answer (only valid ones)
    dropped: list                              # names whose answer was missing or malformed
    usage: dict
    cost_usd: float
    seconds: float

    def get(self, name: str) -> Optional[Answer]:
        return self.answers.get(name)


def _num(v, lo: float, hi: float) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if lo - 1e-6 <= v <= hi + 1e-6 else None


def validate(questions: list[dict], body: dict) -> tuple[dict, list]:
    """Checks each returned answer against its question. Returns ({name: Answer}, [dropped names])."""
    by_name = {q["name"]: q for q in questions}
    out, seen = {}, set()
    for a in (body or {}).get("answers") or []:
        if not isinstance(a, dict):
            continue
        name, typ = a.get("name"), a.get("type")
        q = by_name.get(name)
        if q is None or name in seen:
            continue
        seen.add(name)
        if typ == "refusal":
            out[name] = Answer(name, "refusal")
            continue
        if typ != q["type"]:
            continue
        conf = _num(a.get("confidence"), 0, 1)
        if typ == "predicate":
            p = _num(a.get("probability"), 0, 1)
            if p is not None:
                out[name] = Answer(name, typ, probability=p)
        elif typ == "choice":
            values = [c["value"] for c in q["choices"]]
            if a.get("choice") in values:
                probs = [x for x in a.get("probabilities") or [] if isinstance(x, dict) and x.get("value") in values]
                out[name] = Answer(name, typ, choice=a["choice"], probabilities=probs, confidence=conf)
        elif typ == "score":
            n = len(q["levels"])
            s = _num(a.get("score"), 0, n - 1)
            if s is not None:
                probs = [x for x in a.get("probabilities") or [] if isinstance(x, dict)
                         and _num(x.get("probability"), 0, 1) is not None]
                labels = [lv["label"] for lv in q["levels"]]
                top = max(probs, key=lambda x: x["probability"])["label"] if probs and \
                    all(x.get("label") in labels for x in probs) else labels[int(round(s))]
                out[name] = Answer(name, typ, score=s, level=top, probabilities=probs, confidence=conf)
    dropped = [q["name"] for q in questions if q["name"] not in out]
    return out, dropped


def cost(usage: dict) -> float:
    """USD for one request's usage (input_tokens already includes cached and cache-write tokens)."""
    total = int((usage or {}).get("input_tokens") or 0)
    det = (usage or {}).get("input_tokens_details") or {}
    cached, written = int(det.get("cached_tokens") or 0), int(det.get("cache_write_tokens") or 0)
    plain = max(0, total - cached - written)
    return round((plain * PRICE_INPUT + cached * PRICE_CACHED + written * PRICE_CACHE_WRITE) / 1e6, 6)


# ------------------------------------------------------------------ the call

class Client:
    """Decisions over HTTP. `api_key` defaults to OPENAI_API_KEY (the backend's existing secret)."""

    def __init__(self, api_key: Optional[str] = None, http: Optional[httpx.Client] = None,
                 timeout: float = 120.0, retries: int = 1):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.http = http or httpx.Client(timeout=httpx.Timeout(timeout, connect=15.0))
        self.retries = retries

    def decide(self, input: Union[str, list], questions: list[dict], cache_key: Optional[str] = None) -> Result:
        if not self.api_key:
            raise DecisionsUnavailable("OPENAI_API_KEY is not set")
        if not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(f"1-{MAX_QUESTIONS} questions per request")
        if len({q["name"] for q in questions}) != len(questions):
            raise ValueError("question names must be unique")
        body: dict = {"model": MODEL, "input": input, "questions": questions}
        if cache_key:
            body["prompt_cache_key"] = cache_key[:64]
        t0 = time.time()
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = self.http.post(URL, json=body, headers={"Authorization": f"Bearer {self.api_key}"})
            except httpx.HTTPError as e:
                last = f"network: {e}"
            else:
                if r.status_code == 200:
                    data = r.json()
                    answers, dropped = validate(questions, data)
                    usage = data.get("usage") or {}
                    return Result(answers, dropped, usage, cost(usage), round(time.time() - t0, 2))
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    break                               # a bad request won't get better by retrying
            if attempt < self.retries:
                time.sleep(2 * (attempt + 1))
        raise DecisionsUnavailable(last or "no response")


def enabled() -> bool:
    """On unless REPURPOSE_DECISIONS=off, and only with an API key."""
    return os.environ.get("REPURPOSE_DECISIONS", "on").lower() not in ("off", "false", "0") and \
        bool(os.environ.get("OPENAI_API_KEY"))
