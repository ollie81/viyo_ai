"""
Model benchmark for the AI Repurposer's selection stage — test-only.

Runs the AI part of clip selection (story map -> clip proposals -> editor
review -> context recovery -> hooks) on a transcript already cached from an
earlier repurpose job, with ONE chosen model and reasoning level, and reports
what it picked, why, how long it took, the tokens it used and the cost.

Safety:
  * Only user IDs in REPURPOSE_BENCHMARK_USERS (default: the QA test account)
    may call it. No coins are charged; it never touches jobs, clips or storage.
  * It reads only transcripts by content hash — the caller must already know
    the hash of a video that was processed before.
  * Nothing in the live pipeline reads anything this module writes.

Review here is text-only (no frames), the same for every model, so the
comparison is like-for-like.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

import clip_review
import clip_selection as cs
import hooks
import video_map

router = APIRouter(prefix="/api/v1/repurpose", tags=["repurpose_benchmark"])

ALLOWED = {u.strip() for u in os.environ.get(
    "REPURPOSE_BENCHMARK_USERS", "172fdc6b-604b-4a77-91b7-987b78de8b65").split(",") if u.strip()}

# USD per 1M tokens (input, cached input, output) — OpenAI pricing page, Nov 2026.
PRICES = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
}
REASONING_MODELS = {"gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-astra",
                    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"}


async def _user(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest
    return await get_current_user_id_no_guest(authorization)


class BenchRequest(BaseModel):
    sha256: str = Field(..., min_length=64, max_length=64)
    model: str
    reasoning_effort: Optional[str] = None     # GPT-6 models: none/low/medium/high
    video_seconds: float = Field(..., gt=0)


class Meter:
    def __init__(self, client, model: str, effort: Optional[str]):
        self.client, self.model, self.effort = client, model, effort
        self.calls, self.errors = [], []

    def __call__(self, messages: list) -> str:
        kwargs = {"model": self.model, "messages": messages, "max_completion_tokens": 16000}
        if self.model in REASONING_MODELS:
            kwargs["reasoning_effort"] = self.effort or "low"
        else:
            kwargs["temperature"] = 0
        t = time.time()
        try:
            r = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            self.errors.append(str(e)[:300])
            raise cs.SelectionUnavailable(str(e)[:200])
        u = r.usage
        det_in = getattr(u, "prompt_tokens_details", None)
        det_out = getattr(u, "completion_tokens_details", None)
        self.calls.append({
            "seconds": round(time.time() - t, 2),
            "input": u.prompt_tokens, "cached": getattr(det_in, "cached_tokens", 0) or 0,
            "output": u.completion_tokens, "reasoning": getattr(det_out, "reasoning_tokens", 0) or 0,
            "finish": r.choices[0].finish_reason,
        })
        return r.choices[0].message.content or ""

    def totals(self) -> dict:
        inp = sum(c["input"] for c in self.calls)
        cached = sum(c["cached"] for c in self.calls)
        out = sum(c["output"] for c in self.calls)
        p_in, p_cached, p_out = PRICES.get(self.model, (0, 0, 0))
        cost = ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6
        return {"calls": len(self.calls), "input_tokens": inp, "cached_tokens": cached, "output_tokens": out,
                "reasoning_tokens": sum(c["reasoning"] for c in self.calls),
                "seconds": round(sum(c["seconds"] for c in self.calls), 1), "usd": round(cost, 6)}


@router.post("/benchmark")
async def benchmark(req: BenchRequest, user_id: str = Depends(_user)):
    if user_id not in ALLOWED:
        raise HTTPException(status_code=404, detail="Not found")
    import repurpose as r
    if r.job_store is None:
        raise HTTPException(status_code=503, detail="Jobs not configured")
    transcript = r.job_store.get_transcript(req.sha256, "whisper-1")
    if not transcript:
        raise HTTPException(status_code=404, detail="No cached transcript for that video")
    words, segments = transcript.get("words") or [], transcript.get("segments") or []
    meter = Meter(r.ai_client, req.model, req.reasoning_effort)
    started = time.time()
    out: dict = {"model": req.model, "reasoning_effort": req.reasoning_effort}
    try:
        sentences = cs.build_sentences(cs.attach_punctuation(words, segments))
        vm = video_map.build_video_map(sentences, None, meter)
        ctx = cs.AnalysisContext(video_map=vm)
        result = cs.run_selection(words, segments, req.video_seconds, meter, ctx, cs.DEFAULT_MIN_SCORE)
        proposed = len(result.clips) + len(result.spares)
        pool = result.clips + result.spares
        reviews = clip_review.review_candidates(pool, sentences, None, meter, None,
                                                (vm or {}).get("summary"))
        clip_review.apply_reviews(result, reviews, pool)
        recovery = clip_review.recover_with_context(result, sentences, cs.attach_punctuation(words, segments),
                                                    req.video_seconds, ctx, None, meter, None,
                                                    (vm or {}).get("summary"))
        hook_stats = hooks.fill_missing_hooks(result.clips, meter)
        out.update({
            "video_map": {"format": (vm or {}).get("format"), "summary": (vm or {}).get("summary"),
                          "avoid": len((vm or {}).get("avoid") or [])},
            "candidates_evaluated": (result.analysis or {}).get("candidates_evaluated"),
            "passed_selection": proposed,
            "final_clips": [{
                "start": c.start, "end": c.end, "seconds": round(c.end - c.start, 1), "score": c.score,
                "category": c.category, "title": c.title, "hook": c.overlay_hook, "hook_status": c.overlay_status,
                "why": c.reason, "opens": c.transcript[:160], "ends": c.transcript[-120:],
                "review": {k: (c.review or {}).get(k) for k in ("verdict", "standalone", "opening", "ending", "issues")},
            } for c in result.clips],
            "rejected": [{k: x.get(k) for k in ("title", "score", "reason", "issues")} for x in result.rejected][:20],
            "recovery": recovery, "hooks": hook_stats, "note": result.note,
        })
    except Exception as e:
        out["error"] = str(e)[:400]
    out["usage"] = meter.totals()
    out["per_call"] = meter.calls
    out["errors"] = meter.errors
    out["wall_seconds"] = round(time.time() - started, 1)
    return out
