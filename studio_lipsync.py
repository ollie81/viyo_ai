"""Lip sync with sync.so: the character's permanent saved voice, and their
mouth moving to it, on a video clip (or a still portrait).

API from sync.so's OpenAPI spec (6 Nov 2026):
  POST https://api.sync.so/v2/generate   header x-api-key
       {"model": ..., "input": [{"type": "video"|"image", "url": ...}, {"type": "audio", "url": ...}]}
  GET  https://api.sync.so/v2/generate/{id}
       status PENDING | PROCESSING | COMPLETED | FAILED | REJECTED, outputUrl, outputDuration, error, errorCode

Off until switched on: STUDIO_LIPSYNC_ENABLED=true and SYNC_API_KEY set
(the key lives only in Railway's variables). Not yet verified with a live
call; tested with a fake that never spends anything.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Callable, Optional

import studio_ai as ai_mod
import studio_pricing as pricing

SYNC_BASE = "https://api.sync.so"
MODEL = os.environ.get("STUDIO_LIPSYNC_MODEL", "lipsync-2")
# sync.so bills per output frame. Hobbyist/Creator list prices, sync.so
# pricing page checked 6 Nov 2026 ($0.05 / $0.08325 / $0.13325 per second
# at 25 fps).
PRICE_PER_FRAME = {"lipsync-2": 0.002, "lipsync-2-pro": 0.00333, "sync-3": 0.00533}
HOLD_FPS = 30          # holds assume 30 fps so a 30 fps clip is still covered
MAX_WAIT_SECONDS = 15 * 60


def enabled() -> bool:
    return os.environ.get("STUDIO_LIPSYNC_ENABLED", "").lower() == "true" and bool(os.environ.get("SYNC_API_KEY"))


def lipsync_cost(seconds: float, fps: float = HOLD_FPS, model: Optional[str] = None) -> pricing.Cost:
    model = model or MODEL
    frames = max(1, round(seconds * fps))
    return pricing.Cost("sync", model, {"seconds": round(seconds, 2), "frames": frames, "fps": round(fps, 2)},
                        frames * PRICE_PER_FRAME.get(model, PRICE_PER_FRAME["sync-3"]),
                        storage_cost_usd=pricing.storage_cost(seconds * 1_000_000))


def estimate_lipsync(seconds: float) -> pricing.Cost:
    return lipsync_cost(seconds, HOLD_FPS)


def _probe(data: bytes) -> tuple[float, float]:
    """(seconds, fps) of an mp4."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(data)
        f.flush()
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=r_frame_rate:format=duration", "-of", "default=nw=1", f.name],
                             capture_output=True, text=True).stdout
    secs, fps = 0.0, float(HOLD_FPS)
    for line in out.splitlines():
        if line.startswith("duration="):
            try:
                secs = float(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("r_frame_rate="):
            num, _, den = line.split("=", 1)[1].partition("/")
            try:
                fps = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                pass
    return secs, fps


class SyncLipSync:
    def __init__(self, api_key: Optional[str] = None, http=None, poll_seconds: float = 5.0):
        import requests
        self.key = api_key or os.environ.get("SYNC_API_KEY", "")
        self.http = http or requests
        self.poll_seconds = poll_seconds

    def _headers(self):
        return {"x-api-key": self.key, "Content-Type": "application/json"}

    @staticmethod
    def _error(status: int, text: str) -> Exception:
        t = (text or "")[:300]
        if status in (429, 500, 502, 503, 504):
            return ai_mod.ProviderBusy(f"sync.so is busy ({status}).")
        if status == 402:
            return ai_mod.ProviderError("Viyo's lip-sync account is out of credit. Your credits were returned.")
        if status in (401, 403):
            return ai_mod.ProviderError("Lip sync isn't set up correctly on Viyo yet. Your credits were returned.")
        return ai_mod.ProviderError(f"Lip sync refused the request ({status}): {t}")

    def start(self, media_type: str, media_url: str, audio_url: str, model: Optional[str] = None) -> str:
        body = {"model": model or MODEL, "input": [{"type": media_type, "url": media_url}, {"type": "audio", "url": audio_url}]}
        try:
            r = self.http.post(f"{SYNC_BASE}/v2/generate", json=body, headers=self._headers(), timeout=60)
        except Exception as e:
            raise ai_mod.ProviderBusy(f"sync.so unreachable: {str(e)[:120]}")
        if r.status_code not in (200, 201):
            raise self._error(r.status_code, r.text)
        gid = (r.json() or {}).get("id")
        if not gid:
            raise ai_mod.ProviderError("sync.so didn't return a job id.")
        return gid

    def status(self, gid: str) -> dict:
        try:
            r = self.http.get(f"{SYNC_BASE}/v2/generate/{gid}", headers=self._headers(), timeout=60)
        except Exception as e:
            raise ai_mod.ProviderBusy(f"sync.so unreachable: {str(e)[:120]}")
        if r.status_code != 200:
            raise self._error(r.status_code, r.text)
        return r.json() or {}

    def wait(self, gid: str, max_seconds: float = MAX_WAIT_SECONDS, tick: Optional[Callable] = None) -> dict:
        deadline = time.monotonic() + max_seconds
        while True:
            s = self.status(gid)
            st = s.get("status")
            if st == "COMPLETED":
                if not s.get("outputUrl"):
                    raise ai_mod.ProviderError("Lip sync finished without a video.")
                return s
            if st == "REJECTED":
                raise ai_mod.PolicyBlocked(f"Lip sync refused this scene: {s.get('error') or s.get('errorCode') or 'no reason given'}")
            if st == "FAILED":
                raise ai_mod.ProviderError(f"Lip sync failed: {s.get('error') or s.get('errorCode') or 'no reason given'}")
            if time.monotonic() > deadline:
                raise ai_mod.ProviderError("Lip sync took too long. Your credits were returned.")
            if tick:
                tick()
            time.sleep(self.poll_seconds)

    def download(self, url: str) -> bytes:
        try:
            r = self.http.get(url, timeout=300)
        except Exception as e:
            raise ai_mod.ProviderBusy(f"couldn't download the lip-synced video: {str(e)[:120]}")
        if r.status_code != 200 or not r.content:
            raise ai_mod.ProviderError("Couldn't download the lip-synced video.")
        return r.content

    def run(self, media_type: str, media_url: str, audio_url: str, resume_id: Optional[str] = None,
            on_started: Optional[Callable[[str], None]] = None) -> tuple[bytes, pricing.Cost, str]:
        """(mp4 bytes, cost, sync job id). With resume_id, carries on with a
        job started before a worker restart instead of paying twice."""
        gid = resume_id or self.start(media_type, media_url, audio_url)
        if not resume_id and on_started:
            on_started(gid)
        result = self.wait(gid)
        data = self.download(result["outputUrl"])
        secs, fps = _probe(data)
        secs = secs or float(result.get("outputDuration") or 0)
        return data, lipsync_cost(secs, fps, result.get("model")), gid


class FakeLipSync:
    """For tests: puts the voice track on the video (or a still) and
    returns it, like sync.so would, without any network or cost."""

    def __init__(self, fail: Optional[Exception] = None):
        self.calls: list = []
        self.fail = fail
        self.resumed: list = []

    def run(self, media_type, media_url, audio_url, resume_id=None, on_started=None):
        self.calls.append((media_type, media_url, audio_url, resume_id))
        if resume_id:
            self.resumed.append(resume_id)
        else:
            if on_started:
                on_started(f"fake-{len(self.calls)}")
        if self.fail:
            raise self.fail
        size = "720x1280"
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "o.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration=3",
                            "-f", "lavfi", "-i", "sine=f=1000:d=3", "-c:v", "libx264", "-preset", "ultrafast",
                            "-c:a", "aac", "-shortest", out], check=True)
            data = open(out, "rb").read()
        return data, lipsync_cost(3.0, 25), resume_id or f"fake-{len(self.calls)}"
