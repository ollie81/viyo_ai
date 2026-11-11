"""
Creator Studio pricing: what each operation really costs, and what the
creator pays for it in Studio credits.

    credits = max(MIN_CREDITS, ceil(cost_usd / (1 - target_margin) / USD_PER_CREDIT))

  * cost_usd = provider cost + attributable compute + storage for the
    included retention period. Nothing else is added: no hidden reserve.
    Failures the provider bills anyway are absorbed by Viyo and recorded.
  * Payment fees are NOT in here. They're covered when credits are bought
    (see pack_price), because a fee depends on the payment, not on what
    the credits are later spent on.
  * 1 credit = $0.01 of value Viyo actually receives.

Provider prices are the official list prices checked on PRICE_CHECKED
(Google's Gemini API pricing page). Each price row has a date range so
history is preserved; every usage row stores the exact prices used.
"""
from __future__ import annotations

import datetime
import math
import os
from dataclasses import dataclass, field
from typing import Optional

PRICE_CHECKED = "2026-11-05"
USD_PER_CREDIT = 0.01
MIN_CREDITS = 1

TEXT_MODEL = os.environ.get("STUDIO_TEXT_MODEL", "gemini-3.8-flash")
IMAGE_MODEL = os.environ.get("STUDIO_IMAGE_MODEL", "gemini-3.1-flash-image")
TTS_MODEL = os.environ.get("STUDIO_TTS_MODEL", "gemini-3.8-flash-tts")
VIDEO_MODEL = os.environ.get("STUDIO_VIDEO_MODEL", "gemini-omni-1.2-flash")

# (valid_from, prices). The last row whose date has started applies.
_PRICE_TABLE: list[tuple[str, dict]] = [
    ("2026-11-01", {
        "text_input_per_m": 0.75, "text_output_per_m": 3.75,            # gemini-3.8-flash
        "image_input_per_m": 0.50, "image_per_1k_image": 0.067,          # gemini-3.1-flash-image
        "image_per_2k_image": 0.101,
        "tts_audio_per_m": 9.00, "tts_tokens_per_second": 25,            # gemini-3.8-flash-tts
        "video_input_per_m": 1.50, "video_output_per_m": 12.00,          # gemini-omni-1.2-flash ($0.07/s at 720p)
        "video_tokens_per_second_720p": 5792,
        # Railway compute and Supabase storage are our own estimates
        # (Railway ~$20 per vCPU-month; Supabase $0.021 per GB-month).
        "cpu_per_second": 20.0 / (30 * 24 * 3600),
        "storage_per_gb_month": 0.021,
    }),
    ("2027-01-01", {
        "text_input_per_m": 1.50, "text_output_per_m": 7.50,
        "tts_audio_per_m": 18.00,
    }),
]

STORAGE_MONTHS_INCLUDED = 3


def prices(on: Optional[datetime.date] = None) -> dict:
    day = (on or datetime.datetime.now(datetime.timezone.utc).date()).isoformat()
    merged: dict = {}
    for start, row in _PRICE_TABLE:
        if start <= day:
            merged.update(row)
    return merged


def price_version(on: Optional[datetime.date] = None) -> str:
    day = (on or datetime.datetime.now(datetime.timezone.utc).date()).isoformat()
    return max(start for start, _ in _PRICE_TABLE if start <= day)


def target_margin() -> float:
    try:
        m = float(os.environ.get("STUDIO_TARGET_MARGIN", "0.25"))
    except ValueError:
        m = 0.25
    return min(max(m, 0.0), 0.8)


def credits_for(cost_usd: float, margin: Optional[float] = None) -> int:
    m = target_margin() if margin is None else margin
    if cost_usd <= 0:
        return MIN_CREDITS
    # Round to micro-cents first so float noise never adds a credit.
    raw = round(cost_usd / (1 - m) / USD_PER_CREDIT, 6)
    return max(MIN_CREDITS, math.ceil(raw))


def pack_price(credits: int, pct_fee: float, fixed_fee: float) -> float:
    """What to charge for a pack so Viyo receives credits x $0.01 after the
    payment provider's fee."""
    return round((credits * USD_PER_CREDIT + fixed_fee) / (1 - pct_fee), 2)


# ---------------------------------------------------------------- estimates

@dataclass
class Cost:
    """A priced operation. `provider_cost_usd` is the provider's bill;
    infra and storage are our own attributable costs."""
    provider: str
    model: str
    units: dict = field(default_factory=dict)
    provider_cost_usd: float = 0.0
    infra_cost_usd: float = 0.0
    storage_cost_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.provider_cost_usd + self.infra_cost_usd + self.storage_cost_usd

    @property
    def credits(self) -> int:
        return credits_for(self.total_usd)

    def usage(self) -> dict:
        p = prices()
        return {"provider": self.provider, "model": self.model, "units": self.units, "prices": p,
                "price_version": price_version(), "provider_cost_usd": round(self.provider_cost_usd, 6),
                "infra_cost_usd": round(self.infra_cost_usd, 6), "storage_cost_usd": round(self.storage_cost_usd, 6)}


def text_cost(input_tokens: int, output_tokens: int) -> Cost:
    p = prices()
    return Cost("google", TEXT_MODEL, {"input_tokens": input_tokens, "output_tokens": output_tokens},
                input_tokens * p["text_input_per_m"] / 1e6 + output_tokens * p["text_output_per_m"] / 1e6)


def image_cost(images: int = 1, input_tokens: int = 0, size: str = "1K") -> Cost:
    """size "2K" is the High quality setting (sharper on 1080p video)."""
    p = prices()
    per = p["image_per_2k_image"] if size == "2K" else p["image_per_1k_image"]
    return Cost("google", IMAGE_MODEL, {"images": images, "input_tokens": input_tokens, "size": size},
                images * per + input_tokens * p["image_input_per_m"] / 1e6,
                storage_cost_usd=storage_cost((4 if size == "2K" else 1.5) * 1024 * 1024 * images))


def voice_cost(audio_seconds: float) -> Cost:
    p = prices()
    tokens = audio_seconds * p["tts_tokens_per_second"]
    return Cost("google", TTS_MODEL, {"audio_seconds": round(audio_seconds, 2), "audio_tokens": round(tokens)},
                tokens * p["tts_audio_per_m"] / 1e6,
                storage_cost_usd=storage_cost(audio_seconds * 48_000))


def video_cost(seconds: float, output_tokens: Optional[int] = None, input_tokens: int = 0) -> Cost:
    p = prices()
    out = output_tokens if output_tokens is not None else seconds * p["video_tokens_per_second_720p"]
    return Cost("google", VIDEO_MODEL, {"video_seconds": round(seconds, 2), "output_tokens": round(out),
                                        "input_tokens": input_tokens},
                out * p["video_output_per_m"] / 1e6 + input_tokens * p["video_input_per_m"] / 1e6,
                storage_cost_usd=storage_cost(seconds * 1_000_000))


def render_cost(output_seconds: float, output_bytes: Optional[int] = None, music_generated: bool = False) -> Cost:
    """FFmpeg compute/storage plus an original Lyria score when one is needed."""
    p = prices()
    cpu_seconds = max(30.0, output_seconds * 3.0)
    size = output_bytes if output_bytes is not None else output_seconds * 1_000_000
    music_cost = 0.08 if music_generated else 0.0  # Google Lyria 3.5: $0.08 per generated song
    return Cost("viyo+google" if music_generated else "viyo", "ffmpeg+lyria-3.5" if music_generated else "ffmpeg",
                {"cpu_seconds": round(cpu_seconds), "output_seconds": round(output_seconds, 2),
                 "bytes": int(size), "generated_music": music_generated},
                music_cost, infra_cost_usd=cpu_seconds * p["cpu_per_second"], storage_cost_usd=storage_cost(size))


def storage_cost(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3) * prices()["storage_per_gb_month"] * STORAGE_MONTHS_INCLUDED


# How many tokens the text steps usually use. Thinking is capped (see
# studio_ai.THINKING_BUDGET) so these stay predictable; the job holds
# HOLD_FACTOR x the estimate and only the real usage is kept.
THINKING_BUDGET = 1024
HOLD_FACTOR = 1.6


def estimate_script(target_seconds: int, brief_chars: int) -> Cost:
    """The writers' room makes three calls: story plan, draft, script doctor
    (which reads the plan and the draft and writes the whole script again)."""
    script_out = 600 + int(target_seconds * 14)
    plan_in, plan_out = 1600 + brief_chars // 3, THINKING_BUDGET + 2200
    draft_in, draft_out = 1200 + plan_out, THINKING_BUDGET + script_out
    doctor_in, doctor_out = 1300 + plan_out + script_out, THINKING_BUDGET + script_out + 700
    # The story check of the final script, and (at most once) a targeted revision and a second check.
    eval_in, eval_out = 1500 + script_out, THINKING_BUDGET + 1200
    rev_in, rev_out = 900 + script_out, THINKING_BUDGET + script_out
    return text_cost(plan_in + draft_in + doctor_in + 2 * eval_in + rev_in,
                     plan_out + draft_out + doctor_out + 2 * eval_out + rev_out)


def estimate_breakdown(script_chars: int, target_seconds: int) -> Cost:
    scenes = max(3, target_seconds // 6)
    # ~260 tokens a scene for the shot and lines, ~450 more for its continuity record.
    return text_cost(1700 + script_chars // 3, THINKING_BUDGET + 400 + scenes * 710)


def estimate_voice(text: str) -> Cost:
    words = max(1, len(text.split()))
    return voice_cost(words / 2.4 + 0.6)


VIDEO_MIN_SECONDS, VIDEO_MAX_SECONDS = 3, 15   # Omni 1.2 accepts "3s".."15s" (default 10s)


def video_seconds_for(scene_seconds) -> int:
    """The clip length we ask Omni for: the scene's length, rounded up,
    within what the model accepts. Asked for explicitly so the estimate,
    the hold and the bill all use the same number."""
    import math
    try:
        s = math.ceil(float(scene_seconds))
    except (TypeError, ValueError):
        s = 8
    return max(VIDEO_MIN_SECONDS, min(VIDEO_MAX_SECONDS, s))


def estimate_video(seconds: float = 8.0) -> Cost:
    return video_cost(video_seconds_for(seconds), input_tokens=1800)


def combine(costs: list[Cost]) -> Cost:
    """Several provider calls in one job (e.g. a character's reference sheet)."""
    costs = [c for c in costs if c is not None]
    if not costs:
        return Cost("google", IMAGE_MODEL)
    units: dict = {"calls": len(costs)}
    for c in costs:
        for k, v in (c.units or {}).items():
            if isinstance(v, (int, float)):
                units[k] = units.get(k, 0) + v
    return Cost(costs[0].provider, "+".join(sorted({c.model for c in costs})), units,
                sum(c.provider_cost_usd for c in costs), sum(c.infra_cost_usd for c in costs),
                sum(c.storage_cost_usd for c in costs))


IDENTITY_TOKENS = (900, THINKING_BUDGET + 700)   # the identity profile: one short JSON text call
SHEET_REF_TOKENS = 1300                          # one reference picture sent with each view


def estimate_outfit_suggestions(script_chars: int, characters: int) -> Cost:
    return text_cost(1200 + min(script_chars, 12000) // 3, THINKING_BUDGET + 300 + characters * 350)


def estimate_identity() -> Cost:
    return text_cost(*IDENTITY_TOKENS)


def estimate_sheet(views: int, size: str = "1K", needs_identity: bool = False) -> Cost:
    """A character's reference views: the front view has no reference, every
    other view is made from it (one reference picture each)."""
    parts = [image_cost(1, 400 + SHEET_REF_TOKENS, size) for _ in range(views)]
    if needs_identity:
        parts.append(estimate_identity())
    return combine(parts)


def hold_credits(kind: str, cost: Cost) -> int:
    """Credits to hold for a job: text jobs vary, so a margin is held and
    the unused part returned; fixed-price items hold exactly the price."""
    if kind in ("script", "breakdown", "outfits"):
        return credits_for(cost.total_usd * HOLD_FACTOR)
    return cost.credits


def estimate_inspection(images: int = 5) -> Cost:
    """One visual continuity check: the scene picture and its references read
    by the text model (studio_inspect), with a capped thinking budget."""
    return text_cost(2500 + SHEET_REF_TOKENS * max(1, images), THINKING_BUDGET + 900)


def estimate_shotplan(scene_chars: int = 1500) -> Cost:
    """The director planning one scene's shots (one text call, capped thinking)."""
    return text_cost(2200 + scene_chars // 3, THINKING_BUDGET + 1400)
