"""OpenAI Decisions API in the AI Shorts Creator: the HTTP client (exact
request, retries, refusals, validation of every answer, usage and cost), the
clip judge (questions per candidate, thresholds, explanations, second opinions
for unsure answers, re-ranking), the pipeline using it and falling back when
it is unavailable, and the creator's choice of how many shorts.

Every Decisions response here is MOCKED (httpx.MockTransport): no request
reaches OpenAI and nothing is spent. That a real gpt-6-luna request succeeds,
and how good its judgements are, is NOT shown by these tests.
"""
import json
import os

import httpx
import pytest

import clip_decisions as cd
import clip_selection as cs
import decisions as dc
from test_repurpose_worker import (env, job_row, local_path, source_video, submit)  # noqa: F401
from test_shorts_editor import LONG, sentences_from
from conftest import db  # noqa: F401


def answer_all(request_json, overrides=None):
    """A plausible Decisions response for every question in a request."""
    out = []
    for q in request_json["questions"]:
        o = (overrides or {}).get(q["name"]) or (overrides or {}).get(q["name"].split("_", 1)[-1])
        if o == "refusal":
            out.append({"type": "refusal", "name": q["name"]})
            continue
        if q["type"] == "predicate":
            # A normal clip: yes to the good things, no to "are there rendering defects?".
            default = 0.05 if q["name"].endswith("defects") else 0.9
            out.append({"type": "predicate", "name": q["name"], "probability": (o or {}).get("p", default)})
        elif q["type"] == "score":
            labels = [l["label"] for l in q["levels"]]
            s = min((o or {}).get("s", 2.4), len(labels) - 1)
            top = labels[int(round(s))]
            out.append({"type": "score", "name": q["name"], "score": s, "confidence": (o or {}).get("c", 0.8),
                        "probabilities": [{"value": k, "label": l, "probability": 0.7 if l == top else 0.1}
                                          for k, l in enumerate(labels)]})
        elif q["type"] == "choice":
            out.append({"type": "choice", "name": q["name"], "choice": q["choices"][0]["value"], "confidence": 0.9,
                        "probabilities": [{"value": q["choices"][0]["value"], "probability": 0.9}]})
    return {"answers": out, "usage": {"input_tokens": 20000,
                                      "input_tokens_details": {"cached_tokens": 12000, "cache_write_tokens": 0}}}


class Fake:
    def __init__(self, statuses=(), overrides=None, body=None):
        self.requests, self.statuses, self.overrides, self.body = [], list(statuses), overrides, body

    def __call__(self, request):
        data = json.loads(request.content)
        self.requests.append({"json": data, "auth": request.headers.get("authorization"), "url": str(request.url)})
        if self.statuses:
            code = self.statuses.pop(0)
            if code != 200:
                return httpx.Response(code, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=self.body or answer_all(data, self.overrides))

    def client(self, key="sk-test", retries=1):
        return dc.Client(api_key=key, http=httpx.Client(transport=httpx.MockTransport(self)), retries=retries)


# ------------------------------------------------------------------ the client

def test_the_request_matches_the_documented_api_and_usage_becomes_cost(monkeypatch):
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    fake = Fake()
    qs = [dc.predicate("alone", "Can it stand alone?"),
          dc.score("hook", "How strong is the hook?", [("Poor", "a"), ("Good", "b")]),
          dc.choice("kind", "What is it?", [("story", "a story"), ("advice", "advice")])]
    res = fake.client().decide("evidence", qs, cache_key="sha256-of-video")
    sent = fake.requests[0]
    assert sent["url"] == "https://api.openai.com/v1/decisions" and sent["auth"] == "Bearer sk-test"
    assert set(sent["json"]) == {"model", "input", "questions", "prompt_cache_key"}
    assert sent["json"]["model"] == "gpt-6-luna" and sent["json"]["questions"][1]["levels"][0] == {"label": "Poor",
                                                                                                    "description": "a"}
    assert res.get("alone").probability == 0.9 and res.get("kind").choice == "story" and res.get("hook").level
    # 20,000 input tokens of which 12,000 cached: 8,000 x $0.10/M + 12,000 x $0.01/M.
    assert res.cost_usd == pytest.approx(0.0008 + 0.00012)


def test_busy_answers_are_retried_once_but_a_bad_request_is_not(monkeypatch):
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    q = [dc.predicate("x", "?")]
    fake = Fake(statuses=[503, 200])
    assert fake.client().decide("e", q).get("x") and len(fake.requests) == 2
    fake = Fake(statuses=[400])
    with pytest.raises(dc.DecisionsUnavailable):
        fake.client().decide("e", q)
    assert len(fake.requests) == 1
    with pytest.raises(dc.DecisionsUnavailable):
        Fake().client(key="").decide("e", q)
    with pytest.raises(ValueError):
        dc.score("s", "?", [("only one", "")])


def test_every_answer_is_checked_against_its_question():
    qs = [dc.predicate("p", "?"), dc.score("s", "?", [("Poor", ""), ("Good", ""), ("Great", "")]),
          dc.choice("c", "?", [("a", ""), ("b", "")]), dc.predicate("r", "?"), dc.predicate("missing", "?")]
    body = {"answers": [{"type": "predicate", "name": "p", "probability": 1.7},          # out of range
                        {"type": "score", "name": "s", "score": 5.0, "probabilities": []},  # beyond the levels
                        {"type": "choice", "name": "c", "choice": "z"},                     # not a listed choice
                        {"type": "refusal", "name": "r"},
                        {"type": "predicate", "name": "stranger", "probability": 0.5}]}     # not asked
    answers, dropped = dc.validate(qs, body)
    assert list(answers) == ["r"] and answers["r"].refused
    assert dropped == ["p", "s", "c", "missing"]


# ------------------------------------------------------------------ the clip judge

def _cands(n=2, hook=None, joined=False):
    words, sents = sentences_from(LONG)
    out = []
    for k in range(n):
        raw = {"start_sentence": 6 * k, "end_sentence": 6 * k + 5, "category": "story",
               "scores": {x: 8 for x in cs.RUBRIC_KEYS}, "title": f"Clip {k}", "reason": "x", "overlay_hook": "",
               "caption": "c", "hashtags": ["rent"], "topic": "rent"}
        c, why = cs.evaluate_candidate(raw, sents, words, 400.0)
        assert why is None, why
        out.append(c)
    if hook:
        out[0].overlay_hook = hook
    if joined:
        out[0].structure = {**out[0].structure, "joined": [18, 19], "ranges": [[0, 5], [18, 19]]}
    return out, sents


def test_each_candidate_gets_the_six_questions_and_only_the_ones_that_apply():
    pool, sents = _cands(2, hook="Nobody tells you this", joined=True)
    shared, qs = cd.build_request(list(enumerate(pool)), sents, None, {"summary": "A talk", "format": "talk"},
                                  {0: ["data:image/jpeg;base64,AAAA"]})
    names = [q["name"] for q in qs]
    assert len(names) == len(set(names))
    assert {n for n in names if n.startswith("c1_")} == {"c1_hook", "c1_standalone", "c1_story", "c1_suitable", "c1_ending"}
    assert {"c0_visual", "c0_hook_true", "c0_join_true"} <= set(names), "frames, a hook and a join add their questions"
    vis = next(q for q in qs if q["name"] == "c0_visual")
    assert vis["instructions"][1] == {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"}
    assert [lv["label"] for lv in next(q for q in qs if q["name"] == "c0_hook")["levels"]] == \
        ["Poor", "Average", "Good", "Excellent"]
    # The shared evidence holds the transcript and each clip once; questions only point at a clip.
    user = shared[1]["content"]
    assert "S18 Here is how the rent story ended" in user[0]["text"]
    assert user[1]["type"] == "input_json" and [c["id"] for c in user[1]["data"]["clips"]] == [0, 1]
    assert all(pool[0].transcript not in str(q["instructions"]) for q in qs)


def test_answers_become_a_review_with_explanations_and_unsure_clips_get_a_second_look():
    pool, sents = _cands(3)
    fake = Fake(overrides={"c1_standalone": {"p": 0.2}, "c2_story": {"s": 1.2, "c": 0.3}})
    asked = []

    def second(cands):
        asked.append([c.title for c in cands])
        return {0: {"standalone": True, "opening": 7, "ending": 7, "verdict": "publish", "issues": []}}
    reviews, facts = cd.review(pool, sents, None, None, None, fake.client(), "sha", second)
    assert facts["requests"] == 1 and facts["questions"] == 15 and facts["cost_usd"] > 0
    good, alone, unsure = reviews[0], reviews[1], reviews[2]
    assert good["verdict"] == "publish" and good["standalone"] and good["source"] == "decisions"
    assert good["decisions"]["explanations"]["hook"].startswith("Good:")
    assert alone["verdict"] == "reject" and not alone["standalone"] and "Needs context" in alone["issues"][0]
    assert asked == [["Clip 2"]], "only the low-confidence clip got a second opinion"
    assert unsure["decisions"]["second_opinion"]["verdict"] == "publish"


def test_a_refused_clip_is_never_treated_as_passed():
    pool, sents = _cands(1)
    reviews, _f = cd.review(pool, sents, None, None, None, Fake(overrides={"c0_standalone": "refusal"}).client())
    assert reviews == {}, "left unreviewed, so the checklist says 'not checked', not 'passed'"


def test_decisions_reorder_the_finalists_but_only_as_part_of_the_score():
    pool, _s = _cands(2)
    pool[0].score, pool[1].score = 80, 70
    reviews = {0: {"verdict": "publish", "decisions": {"composite": 20}},
               1: {"verdict": "publish", "decisions": {"composite": 95}}}
    ordered, rv = cd.rerank(pool, reviews)
    assert [c.title for c in ordered] == ["Clip 1", "Clip 0"] and ordered[0].score == 80 and ordered[1].score == 56
    assert rv[0]["decisions"]["composite"] == 95 and ordered[1].breakdown["before_decisions"] == 80


# ------------------------------------------------------------------ in a real job

def _with_decisions(env, fake):
    env["worker"].deps.analysis = env["worker"].deps.review = env["worker"].deps.qa = True
    env["worker"].deps.decisions = fake.client()


def test_a_job_is_judged_by_decisions_and_records_its_usage_and_cost(env, monkeypatch):
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    fake = Fake()
    _with_decisions(env, fake)
    s = env["scripted"]
    seen = []
    real = s.llm

    def llm(messages):
        content = messages[-1]["content"]
        prompt = content if isinstance(content, str) else content[0]["text"]
        seen.append("review" if "strict senior editor" in prompt else "other")
        return real(messages)
    env["worker"].deps.llm = llm
    uid, job_id = submit(env)
    assert env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded", job.get("error_detail")
    assert len(fake.requests) == 1 and "review" not in seen, "Decisions replaced the free-text review call"
    rv = job["result"]["selection"]["analysis"]["review"]["decisions"]
    assert rv["requests"] == 1 and rv["cost_usd"] > 0 and rv["usage"]["input_tokens"] == 20000
    clip = job["result"]["clips"][0]
    assert clip["scores"]["source"] == "decisions" and clip["scores"]["explanations"]["story"].startswith("Good")
    q = {i["key"]: i["status"] for i in clip["quality"]["items"]}
    assert q["stands_alone"] == "passed" and q["opening"] in ("passed", "warning")


def test_when_decisions_is_down_the_editorial_review_takes_over_and_the_job_still_succeeds(env, monkeypatch):
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    fake = Fake(statuses=[500, 500])
    _with_decisions(env, fake)
    uid, job_id = submit(env)
    assert env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded", job.get("error_detail")
    review = job["result"]["selection"]["analysis"]["review"]
    assert "unavailable" in review["decisions"] and review["clips_reviewed"] >= 1
    assert job["result"]["clips"][0]["scores"]["source"] == "editorial_review"


def test_the_creator_chooses_how_many_shorts(env):
    uid, job_id = submit(env)
    assert env["store"].set_clip_count(str(job_id), "00000000-0000-0000-0000-000000000000", 1) is False, \
        "only the owner can set it"
    assert env["store"].set_clip_count(str(job_id), str(uid), 1) is True
    assert env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded" and job["result"]["selection"]["target"] == 1
    assert len(job["result"]["clips"]) == 1
    assert env["store"].set_clip_count(str(job_id), str(uid), 3) is False, "not after the job has started"
