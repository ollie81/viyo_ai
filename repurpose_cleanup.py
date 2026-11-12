"""
Periodic maintenance for the AI Repurposer, run by every worker every
few minutes (all steps are idempotent, so overlapping runs are safe):

  1. Database sweep: fail + refund jobs that ran out of attempts by
     crashing, and retry any refund that didn't go through.
  2. Source videos: delete private uploads once their job finished more
     than REPURPOSE_SOURCE_RETENTION_HOURS ago, and uploads that never
     became a job.
  3. Expired results: after REPURPOSE_RESULT_RETENTION_DAYS, delete the
     job, its clip rows and generated files — except files a Viyo post
     still uses (a posted clip is that post's media now).
  4. Transcript cache entries unused for REPURPOSE_TRANSCRIPT_RETENTION_DAYS.
  5. Job JSON files left in Storage by the pre-database version.

A storage delete that fails leaves the database untouched, so the item
is simply picked up again next run.
"""
from __future__ import annotations

from collections import defaultdict

from repurpose_jobs import JobStore
from repurpose_storage import CLIPS_BUCKET


def _extras(assets: list) -> list:
    """Files made next to each clip that the clip rows don't list: the caption-free master, the platform caption
    files and versions, and the "Made with Viyo" copies. None of them is ever a post's media."""
    import platforms
    out = []
    for a in assets:
        path = a.get("path") or ""
        if not path.endswith(".mp4") or "_" not in path.rsplit("/", 1)[-1]:
            continue
        base = path[:-4]
        out += [f"{base}_edl.json"]
        # The clip itself and every version a creator made by switching edits (5 switches: 32 combinations).
        for stem in [base] + [f"{base}_e{n:05b}" for n in range(32)]:
            out += [f"{stem}_master.mp4", f"{stem}_viyo.mp4", f"{stem}_info.json"]
            if stem != base:
                out.append(f"{stem}.mp4")
            for p in platforms.PLATFORMS:
                out += [f"{stem}_{p}.ass", f"{stem}_{p}.mp4", f"{stem}_{p}_viyo.mp4"]
    return out


def run_maintenance(store: JobStore, storage) -> dict:
    report = {"sweep": store.sweep()}

    sources = store.sources_to_delete()
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for item in sources:
        by_bucket[item["bucket"]].append(item["path"])
    removed_sources = 0
    for bucket, paths in by_bucket.items():
        try:
            storage.remove(bucket, paths)
        except Exception as e:
            print(f"[WARN] cleanup: could not remove {len(paths)} sources from {bucket}: {e}")
            continue
        store.mark_sources_deleted(paths)
        removed_sources += len(paths)
    report["sources_deleted"] = removed_sources

    deletable_jobs = []
    kept_files = 0
    for entry in store.expired_jobs():
        assets = entry.get("assets") or []
        to_remove = [a["path"] for a in assets if not a.get("referenced")] + _extras(assets)
        kept_files += sum(1 for a in assets if a.get("referenced"))
        try:
            for k in range(0, len(to_remove), 400):         # storage APIs cap how many paths one call may remove
                storage.remove(CLIPS_BUCKET, to_remove[k:k + 400])
        except Exception as e:
            print(f"[WARN] cleanup: could not remove clips of job {entry['job_id']}: {e}")
            continue
        deletable_jobs.append(entry["job_id"])
    report["jobs_deleted"] = store.delete_jobs(deletable_jobs) if deletable_jobs else 0
    report["posted_files_kept"] = kept_files

    report["transcripts_pruned"] = store.prune_transcripts()

    try:
        import repurpose
        if repurpose.supabase_admin is not None and storage.__class__.__name__ in ("SupabaseStorage", "RoutingStorage"):
            repurpose._cleanup_expired_repurpose_jobs()
    except Exception as e:
        print(f"[WARN] cleanup: legacy job sweep failed: {e}")

    if any(v for k, v in report.items() if k != "sweep") or any((report["sweep"] or {}).values()):
        print(f"[INFO] repurpose maintenance: {report}")
    return report
