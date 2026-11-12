"""
AI Repurposer worker — the second Railway service.

Runs from the same Docker image as the API with the start command
`python repurpose_worker.py`. Each process works on one job at a time
(rendering is CPU-bound); scale by adding replicas — jobs are claimed
with FOR UPDATE SKIP LOCKED, so any number of workers can share the
queue without ever picking up the same job.

Reliability model:
  * heartbeat thread: refreshes the job's lease every few seconds while
    the main thread does slow work; if a write reports the lease is
    gone (another worker reclaimed a job this one stalled on), the job
    is abandoned without writing anything else.
  * SIGTERM (redeploy/scale-down): the current job is handed straight
    back to the queue without counting the attempt, then the process
    exits. Checkpoints mean the next worker resumes, not restarts.
  * hard kill / OOM: no heartbeat -> reclaimed after stale_seconds by
    any worker; the attempt counts. When attempts run out the database
    sweep fails the job and refunds it.
"""
from __future__ import annotations

import os
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from typing import Optional

from repurpose_jobs import (JobConfig, JobError, JobStore, LeaseLost, ShutdownRequested,
                            TransientJobError, supabase_rpc)
from repurpose_pipeline import HeartbeatState, JobContext, PipelineDeps, process_job


class Heartbeat(threading.Thread):
    def __init__(self, store: JobStore, job: dict, state: HeartbeatState, config: JobConfig):
        super().__init__(daemon=True)
        self.store, self.job, self.state, self.config = store, job, state, config
        # Not _stop/_started: threading.Thread uses those names itself.
        self._halt = threading.Event()
        self._began = time.monotonic()

    def run(self) -> None:
        while not self._halt.wait(self.config.heartbeat_seconds):
            if time.monotonic() - self._began > self.config.max_attempt_seconds:
                # Stop vouching for a hung attempt; it will be reclaimed.
                print(f"[WARN] job {self.job['id']} exceeded max attempt time; abandoning")
                self.state.lease_lost = True
                return
            try:
                ok = self.store.heartbeat(self.job["id"], self.job["lease_token"],
                                          self.state.stage, self.state.progress, self.state.message)
            except Exception as e:  # transient DB/network trouble: try again next tick
                print(f"[WARN] heartbeat failed for {self.job['id']}: {e}")
                continue
            if not ok:
                self.state.lease_lost = True
                return

    def stop(self) -> None:
        self._halt.set()


class Worker:
    def __init__(self, store: JobStore, deps: PipelineDeps, config: JobConfig,
                 worker_id: Optional[str] = None, maintenance=None, notifier=None):
        self.store, self.deps, self.config = store, deps, config
        # (user_id, type, message, job_id) -> None; tells the creator when
        # clips are ready or a job failed for good. None in tests.
        self.notifier = notifier
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.maintenance = maintenance
        self._stop = threading.Event()
        self._current: Optional[tuple[dict, HeartbeatState]] = None
        self._last_maintenance = 0.0

    # -- lifecycle

    def request_shutdown(self) -> None:
        """Stops claiming new jobs and hands the current one back."""
        self._stop.set()
        current = self._current
        if current:
            job, state = current
            state.shutdown = True
            try:
                if self.store.release(job["id"], job["lease_token"]):
                    print(f"[INFO] released job {job['id']} for another worker")
            except Exception as e:
                print(f"[WARN] could not release job {job['id']}: {e} (it will be reclaimed when stale)")
            state.lease_lost = True

    def run_forever(self) -> None:
        print(f"[INFO] repurpose worker {self.worker_id} started")
        while not self._stop.is_set():
            self._maybe_maintain()
            try:
                job = self.store.claim(self.worker_id)
            except Exception as e:
                print(f"[WARN] claim failed: {e}")
                self._stop.wait(self.config.idle_poll_seconds * 3)
                continue
            if not job:
                self._stop.wait(self.config.idle_poll_seconds)
                continue
            self.run_job(job)
        print(f"[INFO] repurpose worker {self.worker_id} stopped")

    def run_once(self) -> bool:
        """Claim and process at most one job. Returns whether one ran."""
        job = self.store.claim(self.worker_id)
        if not job:
            return False
        self.run_job(job)
        return True

    # -- one job

    def run_job(self, job: dict) -> None:
        state = HeartbeatState()
        self._current = (job, state)
        heartbeat = Heartbeat(self.store, job, state, self.config)
        heartbeat.start()
        ctx = JobContext(self.store, job, state)
        started = time.monotonic()
        print(f"[INFO] job {job['id']} attempt {job['attempts']}/{job['max_attempts']} started")
        try:
            result = process_job(ctx, self.deps)
            if self.store.complete(job["id"], job["lease_token"], result):
                n = len(result.get("clips", []))
                print(f"[INFO] job {job['id']} succeeded in {time.monotonic() - started:.0f}s "
                      f"with {n} clips")
                self._notify(job, "clips_ready",
                             f"Your {n} AI clip{'s' if n != 1 else ''} {'are' if n != 1 else 'is'} ready. "
                             "Tap to watch, download or post.")
            else:
                print(f"[WARN] job {job['id']} finished but the lease was gone; result discarded")
        except LeaseLost:
            print(f"[WARN] job {job['id']}: lease lost, stopping without further writes")
        except ShutdownRequested:
            print(f"[INFO] job {job['id']}: shutdown requested")
        except JobError as e:
            self._fail(job, e)
        except Exception as e:
            traceback.print_exc()
            self._fail(job, TransientJobError(
                "internal_error", "Something went wrong while editing your clips.",
                f"{type(e).__name__}: {e}"))
        finally:
            heartbeat.stop()
            self._current = None

    def _fail(self, job: dict, error: JobError) -> None:
        try:
            outcome = self.store.fail(job["id"], job["lease_token"], error, job["attempts"])
        except Exception as e:
            # The DB call itself failed: the job stays "processing" with
            # no heartbeat, gets reclaimed when stale, and the sweep
            # refunds it if attempts run out. Nothing is lost.
            print(f"[ERROR] could not record failure of job {job['id']}: {e}")
            return
        if not outcome.get("owned"):
            print(f"[WARN] job {job['id']}: lease gone before failure could be recorded")
        elif outcome.get("final"):
            print(f"[INFO] job {job['id']} failed permanently ({error.code}); "
                  f"refunded {outcome.get('refunded', 0)} coins")
            refunded = outcome.get("refunded") or 0
            self._notify(job, "clips_failed",
                         f"We couldn't make clips from your video: {error.user_message}"
                         + (f" Your {refunded} coins were refunded." if refunded else ""))
        else:
            print(f"[INFO] job {job['id']} will retry ({error.code}): {error.internal[:300]}")

    def _notify(self, job: dict, notif_type: str, message: str) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier(str(job["user_id"]), notif_type, message, str(job["id"]))
        except Exception as e:
            print(f"[WARN] job {job['id']}: notification failed: {e}")

    def _maybe_maintain(self) -> None:
        if self.maintenance is None:
            return
        now = time.monotonic()
        if now - self._last_maintenance < self.config.maintenance_interval_seconds:
            return
        self._last_maintenance = now
        try:
            self.maintenance()
        except Exception as e:
            print(f"[WARN] maintenance failed: {e}")


def build_production_worker() -> Worker:
    missing = [k for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "OPENAI_API_KEY")
               if not os.environ.get(k)]
    if missing:
        sys.exit(f"Worker can't start: set {', '.join(missing)} in this service's Variables.")

    import repurpose as r
    from openai import OpenAI
    from coins import FEATURE_COSTS
    from repurpose_cleanup import run_maintenance
    from repurpose_storage import RoutingStorage, SupabaseStorage
    from transcription import local_chunk_transcriber, openai_chunk_transcriber, transcript_cache_key
    import clip_decisions
    import clip_selection
    import creator_insights

    if r.supabase_admin is None:
        sys.exit("Worker can't start: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are invalid.")

    config = JobConfig.from_env(cost=FEATURE_COSTS["repurpose"]["cost"])
    store = JobStore(supabase_rpc(r.supabase_admin), config)
    storage = RoutingStorage(SupabaseStorage(r.supabase_admin, r.SUPABASE_URL))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300, max_retries=2)
    use_local_whisper = os.environ.get("REPURPOSE_TRANSCRIBER", "openai").strip().lower() == "local"
    print(f"[INFO] transcription: {'local ' + transcript_cache_key(True) if use_local_whisper else 'OpenAI whisper-1'}")
    deps = PipelineDeps(
        storage=storage,
        # REPURPOSE_TRANSCRIBER=local runs speech-to-text on this worker
        # (faster-whisper); anything else keeps OpenAI's whisper-1.
        transcribe_chunk=(local_chunk_transcriber() if use_local_whisper else openai_chunk_transcriber(client)),
        transcription_model=transcript_cache_key(use_local_whisper),
        llm=r.selection_llm(client),
        max_input_bytes=r.MAX_INPUT_SIZE_BYTES,
        max_input_seconds=r.MAX_INPUT_DURATION_SECONDS,
        min_clip_score=int(os.environ.get("REPURPOSE_MIN_CLIP_SCORE", clip_selection.DEFAULT_MIN_SCORE)),
        caption_style=os.environ.get("REPURPOSE_CAPTION_STYLE", "bold"),
        # The editor's passes; each is on unless switched off (REPURPOSE_<NAME>=off).
        retention=_on("REPURPOSE_RETENTION"), cold_open=_on("REPURPOSE_COLD_OPEN"), zoom=_on("REPURPOSE_ZOOM"),
        platform_versions=_on("REPURPOSE_PLATFORM_VERSIONS"), packaging=_on("REPURPOSE_PACKAGING"),
        # The OpenAI Decisions API judges clips (REPURPOSE_DECISIONS=off falls back to the editorial review).
        decisions=clip_decisions.make_client(),
        # The creator's own history only: how their published clips did, and what they kept or skipped.
        creator_prior_for=lambda user_id: creator_insights.combine(
            creator_insights.load_prior(r.supabase_admin, user_id),
            creator_insights.build_feedback_prior(store.feedback_events(user_id))),
        creator_length_for=lambda user_id: creator_insights.preferred_length(store.feedback_events(user_id)),
    )
    def notifier(user_id: str, notif_type: str, message: str, job_id: str) -> None:
        from notify import notify
        title = "Your clips are ready 🎬" if notif_type == "clips_ready" else "Couldn't make your clips"
        notify(user_id, notif_type, message, entity_id=job_id, push_title=title, admin=r.supabase_admin)

    return Worker(store, deps, config, maintenance=lambda: run_maintenance(store, storage), notifier=notifier)


def _on(name: str) -> bool:
    return os.environ.get(name, "on").strip().lower() not in ("off", "false", "0", "no")


def main() -> None:
    worker = build_production_worker()

    def on_signal(signum, _frame):
        print(f"[INFO] signal {signum}: shutting down")
        worker.request_shutdown()
        # Unwinds whatever the main thread is blocked in (subprocess.run
        # kills its ffmpeg child on the way out) so the process exits now
        # rather than after the current render.
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        import studio_worker
        studio_worker.start_production()
    except Exception as e:  # never let the Studio stop the Repurposer
        print(f"[WARN] Creator Studio worker not started: {e}")
    worker.run_forever()


if __name__ == "__main__":
    main()
