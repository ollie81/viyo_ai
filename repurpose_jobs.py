"""
Job store and error taxonomy for the AI Repurposer.

All state lives in Postgres (repurpose_jobs_schema.sql) and is only
ever changed through its SECURITY DEFINER functions, so the guarantees
— one charge per job, one refund per failure, one owner per job — are
enforced by the database, not by whichever process happens to be
running. This module is the thin, typed layer over those functions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

RpcFn = Callable[[str, dict], Any]


def supabase_rpc(admin) -> RpcFn:
    def call(name: str, params: dict):
        return admin.rpc(name, params).execute().data
    return call


# ---------------------------------------------------------------- errors

class JobError(Exception):
    """A failure with a creator-facing message. `retryable` decides
    whether the attempt is retried (if attempts remain) or the job fails
    for good and is refunded."""

    retryable = False

    def __init__(self, code: str, user_message: str, internal: Optional[str] = None):
        super().__init__(internal or user_message)
        self.code = code
        self.user_message = user_message
        self.internal = internal or user_message


class PermanentJobError(JobError):
    """Retrying can't help: bad input, no speech, nothing worth clipping."""
    retryable = False


class TransientJobError(JobError):
    """Network, provider outage, rate limit — worth another attempt."""
    retryable = True


class LeaseLost(Exception):
    """Another worker now owns this job. Stop without writing anything."""


class ShutdownRequested(Exception):
    """The worker is being stopped; hand the job back."""


# ---------------------------------------------------------------- config

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class JobConfig:
    cost: int = 40
    daily_limit: int = 5
    max_attempts: int = 3
    # A worker heartbeats every heartbeat_seconds; a job with no
    # heartbeat for stale_seconds is presumed dead and can be reclaimed.
    heartbeat_seconds: int = 20
    stale_seconds: int = 180
    # Hard ceiling on one attempt. Past it the worker stops heartbeating,
    # so a hung attempt is reclaimed instead of holding the job forever.
    max_attempt_seconds: int = 60 * 90
    retry_base_delay_seconds: int = 60
    result_retention_days: int = 30
    source_retention_hours: int = 24
    orphan_upload_hours: int = 24
    transcript_retention_days: int = 30
    maintenance_interval_seconds: int = 300
    idle_poll_seconds: float = 3.0
    # Free first video: one coin-free run per account, at most this many
    # app-wide per day (0 turns it off), for videos up to this long.
    free_trial_daily_cap: int = 20
    free_trial_max_seconds: int = 600

    @classmethod
    def from_env(cls, cost: int) -> "JobConfig":
        return cls(
            cost=cost,
            daily_limit=_env_int("REPURPOSE_DAILY_LIMIT", cls.daily_limit),
            max_attempts=_env_int("REPURPOSE_MAX_ATTEMPTS", cls.max_attempts),
            heartbeat_seconds=_env_int("REPURPOSE_HEARTBEAT_SECONDS", cls.heartbeat_seconds),
            stale_seconds=_env_int("REPURPOSE_STALE_SECONDS", cls.stale_seconds),
            max_attempt_seconds=_env_int("REPURPOSE_MAX_ATTEMPT_SECONDS", cls.max_attempt_seconds),
            retry_base_delay_seconds=_env_int("REPURPOSE_RETRY_BASE_DELAY_SECONDS", cls.retry_base_delay_seconds),
            result_retention_days=_env_int("REPURPOSE_RESULT_RETENTION_DAYS", cls.result_retention_days),
            source_retention_hours=_env_int("REPURPOSE_SOURCE_RETENTION_HOURS", cls.source_retention_hours),
            orphan_upload_hours=_env_int("REPURPOSE_ORPHAN_UPLOAD_HOURS", cls.orphan_upload_hours),
            transcript_retention_days=_env_int("REPURPOSE_TRANSCRIPT_RETENTION_DAYS", cls.transcript_retention_days),
            maintenance_interval_seconds=_env_int("REPURPOSE_MAINTENANCE_SECONDS", cls.maintenance_interval_seconds),
            free_trial_daily_cap=_env_int("REPURPOSE_FREE_TRIAL_DAILY_CAP", cls.free_trial_daily_cap),
            free_trial_max_seconds=_env_int("REPURPOSE_FREE_TRIAL_MAX_SECONDS", cls.free_trial_max_seconds),
        )


# ---------------------------------------------------------------- store

class JobStore:
    def __init__(self, rpc: RpcFn, config: JobConfig):
        self._rpc = rpc
        self.config = config

    # -- API side

    def register_upload(self, user_id: str, bucket: str, path: str, size: int, content_type: str) -> None:
        self._rpc("repurpose_register_upload", {
            "p_user_id": user_id, "p_bucket": bucket, "p_path": path,
            "p_bytes": size, "p_content_type": content_type,
        })

    def _rpc_with_length(self, fn: str, args: dict, clip_length: Optional[str]):
        """Calls fn with p_clip_length when a length was chosen. A database
        that predates the clip-length migration rejects the parameter, so
        retry without it (the job then uses the automatic length)."""
        if not clip_length:
            return self._rpc(fn, args)
        try:
            return self._rpc(fn, {**args, "p_clip_length": clip_length})
        except Exception as e:
            print(f"[WARN] {fn} without clip length (database not migrated?): {e}")
            return self._rpc(fn, args)

    def create_job(self, user_id: str, *, source_kind: str, source_bucket: Optional[str],
                   source_path: Optional[str], source_url: Optional[str],
                   creative_prompt: Optional[str], client_request_id: Optional[str],
                   clip_length: Optional[str] = None) -> dict:
        return self._rpc_with_length("repurpose_create_job", {
            "p_user_id": user_id,
            "p_cost": self.config.cost,
            "p_daily_limit": self.config.daily_limit,
            "p_max_attempts": self.config.max_attempts,
            "p_source_kind": source_kind,
            "p_source_bucket": source_bucket,
            "p_source_path": source_path,
            "p_source_url": source_url,
            "p_creative_prompt": creative_prompt,
            "p_client_request_id": client_request_id,
        }, clip_length)

    def set_clip_count(self, job_id: str, user_id: str, count: int) -> bool:
        """How many shorts the creator wants (repurpose_clip_count.sql). False on a database without it."""
        try:
            return bool(self._rpc("repurpose_set_clip_count", {"p_job_id": job_id, "p_user_id": user_id,
                                                               "p_count": int(count)}))
        except Exception as e:
            print(f"[WARN] clip count not saved (run repurpose_clip_count.sql?): {e}")
            return False

    def free_trial_status(self, user_id: str) -> dict:
        """{'eligible', 'used', 'sold_out'}; never eligible when switched off
        or when the database doesn't have the free-trial functions yet."""
        off = {"eligible": False, "used": False, "sold_out": False}
        if self.config.free_trial_daily_cap <= 0 or self.config.cost <= 0:
            return off
        try:
            out = self._rpc("repurpose_free_trial_status", {
                "p_user_id": user_id, "p_daily_cap": self.config.free_trial_daily_cap})
        except Exception as e:
            print(f"[WARN] free trial status unavailable: {e}")
            return off
        return out or off

    def create_free_job(self, user_id: str, *, source_kind: str, source_bucket: Optional[str],
                        source_path: Optional[str], source_url: Optional[str],
                        creative_prompt: Optional[str], client_request_id: Optional[str],
                        clip_length: Optional[str] = None) -> Optional[dict]:
        """Starts the account's free first video. None when it isn't available
        (already used, sold out today, switched off, or not migrated yet), so
        the caller falls back to the normal paid path."""
        if self.config.free_trial_daily_cap <= 0 or self.config.cost <= 0:
            return None
        try:
            out = self._rpc_with_length("repurpose_create_free_job", {
                "p_user_id": user_id,
                "p_daily_cap": self.config.free_trial_daily_cap,
                "p_daily_limit": self.config.daily_limit,
                "p_max_attempts": self.config.max_attempts,
                "p_source_kind": source_kind,
                "p_source_bucket": source_bucket,
                "p_source_path": source_path,
                "p_source_url": source_url,
                "p_creative_prompt": creative_prompt,
                "p_client_request_id": client_request_id,
            }, clip_length)
        except Exception as e:
            print(f"[WARN] free trial unavailable: {e}")
            return None
        if not out or out.get("error") == "free_trial_unavailable":
            return None
        return out

    def get_job(self, job_id: str) -> Optional[dict]:
        return self._rpc("repurpose_get_job", {"p_job_id": job_id})

    # -- worker side

    def claim(self, worker_id: str) -> Optional[dict]:
        return self._rpc("repurpose_claim_job", {
            "p_worker_id": worker_id, "p_stale_seconds": self.config.stale_seconds,
        })

    def heartbeat(self, job_id: str, lease: str, stage: Optional[str] = None,
                  progress: Optional[float] = None, message: Optional[str] = None,
                  checkpoint: Optional[dict] = None) -> bool:
        return bool(self._rpc("repurpose_heartbeat", {
            "p_job_id": job_id, "p_lease": lease, "p_stage": stage,
            "p_progress": progress, "p_message": message, "p_checkpoint": checkpoint,
        }))

    def save_clip(self, job_id: str, lease: str, clip: dict) -> bool:
        return bool(self._rpc("repurpose_save_clip", {
            "p_job_id": job_id, "p_lease": lease, "p_clip": clip,
        }))

    def complete(self, job_id: str, lease: str, result: dict) -> bool:
        return bool(self._rpc("repurpose_complete_job", {
            "p_job_id": job_id, "p_lease": lease, "p_result": result,
            "p_retention_days": self.config.result_retention_days,
        }))

    def retry_delay(self, attempts: int) -> int:
        return self.config.retry_base_delay_seconds * (2 ** max(0, attempts - 1))

    def fail(self, job_id: str, lease: str, error: JobError, attempts: int) -> dict:
        return self._rpc("repurpose_fail_job", {
            "p_job_id": job_id, "p_lease": lease,
            "p_error_code": error.code,
            "p_user_error": error.user_message,
            "p_internal_error": (error.internal or "")[:4000],
            "p_retryable": error.retryable,
            "p_retry_delay_seconds": self.retry_delay(attempts),
            "p_retention_days": self.config.result_retention_days,
        }) or {}

    def release(self, job_id: str, lease: str) -> bool:
        return bool(self._rpc("repurpose_release_job", {"p_job_id": job_id, "p_lease": lease}))

    def get_transcript(self, sha256: str, model: str) -> Optional[dict]:
        return self._rpc("repurpose_get_transcript", {"p_sha256": sha256, "p_model": model})

    def put_transcript(self, sha256: str, model: str, transcript: dict) -> None:
        self._rpc("repurpose_put_transcript", {
            "p_sha256": sha256, "p_model": model,
            "p_duration": transcript.get("duration"),
            "p_language": transcript.get("language"),
            "p_text": transcript.get("text", ""),
            "p_segments": transcript.get("segments", []),
            "p_words": transcript.get("words", []),
        })

    # -- maintenance

    def sweep(self) -> dict:
        return self._rpc("repurpose_sweep", {
            "p_stale_seconds": self.config.stale_seconds,
            "p_retention_days": self.config.result_retention_days,
        }) or {}

    def sources_to_delete(self, limit: int = 200) -> list[dict]:
        return self._rpc("repurpose_sources_to_delete", {
            "p_source_hours": self.config.source_retention_hours,
            "p_orphan_hours": self.config.orphan_upload_hours,
            "p_limit": limit,
        }) or []

    def mark_sources_deleted(self, paths: list[str]) -> None:
        self._rpc("repurpose_mark_sources_deleted", {"p_paths": paths})

    def expired_jobs(self, limit: int = 50) -> list[dict]:
        return self._rpc("repurpose_expired_jobs", {"p_limit": limit}) or []

    def delete_jobs(self, job_ids: list[str]) -> int:
        return int(self._rpc("repurpose_delete_jobs", {"p_job_ids": job_ids}) or 0)

    def prune_transcripts(self) -> int:
        return int(self._rpc("repurpose_prune_transcripts", {
            "p_days": self.config.transcript_retention_days,
        }) or 0)
