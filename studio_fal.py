"""Backup video models through fal.ai (Kling 3, Seedance 2).

When Google refuses a scene's video, the worker can try these instead, from
the scene picture. It is switched off unless both are set in the server's
environment:
  FAL_KEY               the fal.ai key
  STUDIO_VIDEO_BACKUP   which models to try, in order, e.g. "kling,seedance"

Prices are per second of video, from fal.ai's model pages (Nov 2026). Check
them on the first live test; the job only ever charges what the creator saw.
"""
from __future__ import annotations

import base64
import os
import time
from typing import Optional

import httpx

import studio_pricing as pricing

QUEUE = "https://queue.fal.run"
MODELS = {
    # Kling's own voices speak only English and Chinese.
    "kling": {"id": "fal-ai/kling-video/v3/pro/image-to-video", "image": "start_image_url",
              "usd_audio": 0.168, "usd_silent": 0.112, "min": 3, "max": 15, "voices": {"english", "chinese"}},
    "seedance": {"id": "bytedance/seedance-2.0/image-to-video", "image": "image_url",
                 "usd_audio": 0.3024, "usd_silent": 0.3024, "min": 4, "max": 15, "voices": None},
}
POLL_SECONDS, MAX_WAIT = 5, 900


class BackupFailed(Exception):
    def __init__(self, message: str, blocked: bool = False):
        super().__init__(message)
        self.blocked = blocked


def models() -> list[str]:
    """The backup models switched on, in order (empty = off)."""
    if not os.environ.get("FAL_KEY"):
        return []
    raw = os.environ.get("STUDIO_VIDEO_BACKUP", "")
    return [m for m in (x.strip().lower() for x in raw.split(",")) if m in MODELS]


def usable(model: str, talking: bool, language: str) -> bool:
    voices = MODELS[model]["voices"]
    return not (talking and voices is not None and (language or "english").strip().lower() not in voices)


def seconds_for(model: str, seconds: float) -> int:
    m = MODELS[model]
    return max(m["min"], min(m["max"], int(round(seconds))))


def cost(model: str, seconds: float, audio: bool) -> pricing.Cost:
    m = MODELS[model]
    secs = seconds_for(model, seconds)
    usd = secs * (m["usd_audio"] if audio else m["usd_silent"])
    return pricing.Cost("fal", m["id"], {"video_seconds": secs, "audio": audio}, usd,
                        storage_cost_usd=pricing.storage_cost(secs * 1_000_000))


def estimate(seconds: float, talking: bool, language: str) -> Optional[pricing.Cost]:
    """The most a backup try could cost for this scene (None when off)."""
    costs = [cost(m, seconds, True) for m in models() if usable(m, talking, language)]
    return max(costs, key=lambda c: c.total_usd) if costs else None


def _data_uri(image: bytes) -> str:
    mime = "image/png" if image[:8] == b"\x89PNG\r\n\x1a\n" else "image/webp" if image[8:12] == b"WEBP" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(image).decode("ascii")


def _blocked(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("content", "policy", "safety", "nsfw", "moderation", "blocked", "prohibited"))


def generate(model: str, prompt: str, image: bytes, aspect_ratio: str, seconds: float, audio: bool,
             client: Optional[httpx.Client] = None) -> tuple[bytes, pricing.Cost]:
    """One clip from the scene picture. Raises BackupFailed."""
    m = MODELS[model]
    secs = seconds_for(model, seconds)
    body = {"prompt": prompt[:2500], m["image"]: _data_uri(image), "duration": str(secs), "generate_audio": audio}
    if model == "seedance":
        body.update(aspect_ratio=aspect_ratio, resolution="720p")
    headers = {"Authorization": f"Key {os.environ.get('FAL_KEY', '')}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0))
    try:
        r = http.post(f"{QUEUE}/{m['id']}", headers=headers, json=body)
        if r.status_code >= 400:
            raise BackupFailed(f"{model} refused the request ({r.status_code}).", _blocked(r.text))
        sub = r.json()
        waited = 0
        while True:
            st = http.get(sub["status_url"], headers=headers)
            status = (st.json() if st.status_code < 400 else {}).get("status")
            if status == "COMPLETED":
                break
            if st.status_code >= 400 or waited >= MAX_WAIT:
                raise BackupFailed(f"{model} didn't finish the video.")
            time.sleep(POLL_SECONDS)
            waited += POLL_SECONDS
        res = http.get(sub["response_url"], headers=headers)
        if res.status_code >= 400:
            raise BackupFailed(f"{model} couldn't make this video.", _blocked(res.text))
        url = ((res.json() or {}).get("video") or {}).get("url")
        if not url:
            raise BackupFailed(f"{model} returned no video.")
        vid = http.get(url, timeout=120.0)
        if vid.status_code >= 400 or len(vid.content) < 1024:
            raise BackupFailed(f"{model}'s video couldn't be downloaded.")
        return vid.content, cost(model, secs, audio)
    except httpx.HTTPError as e:
        raise BackupFailed(f"{model} couldn't be reached.") from e
    finally:
        if client is None:
            http.close()
