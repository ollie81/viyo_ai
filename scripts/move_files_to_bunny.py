"""
Copies VIYO's files from Supabase Storage to Bunny Storage, then (only when asked) frees the Supabase space.

Run on Railway (it needs the server's settings), from the repository folder:

    python scripts/move_files_to_bunny.py            # count only: files and MB per bucket, copies nothing
    python scripts/move_files_to_bunny.py --copy     # copy everything; each copy's size is checked
    python scripts/move_files_to_bunny.py --copy --delete-from-supabase
                                                     # copy, check, then delete the Supabase original

Safe to run again: files already on Bunny with the same size are skipped. A file whose copy can't be
checked is never deleted from Supabase. Needs every BUNNY_* setting; deleting also needs FILE_STORE=bunny (see viyo_files.py).
The worker can do the same without a shell: set BUNNY_MOVE_FILES=count | copy | copy_and_delete.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viyo_files  # noqa: E402

BUCKETS = viyo_files.ALL_BUCKETS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="copy files (without it, only counts)")
    ap.add_argument("--delete-from-supabase", action="store_true", help="after a checked copy, delete the original")
    ap.add_argument("--bucket", action="append", help="only this bucket (can repeat)")
    args = ap.parse_args()
    if args.delete_from_supabase and not args.copy:
        ap.error("--delete-from-supabase needs --copy")
    if viyo_files.settings(require_switch=False) is None:
        print("Bunny settings are incomplete: set every BUNNY_* setting first (see viyo_files.py).")
        return 2
    from supabase import create_client
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    total_failed = 0
    for bucket in args.bucket or BUCKETS:
        try:
            r = viyo_files.copy_bucket(client, bucket, dry_run=not args.copy,
                                       delete_after=args.delete_from_supabase)
        except Exception as e:
            print(f"{bucket}: could not read ({e})")
            continue
        mb = r["bytes"] / 1024 / 1024
        if not args.copy:
            print(f"{bucket}: {r['files']} files, {mb:.1f} MB")
            continue
        print(f"{bucket}: {r['files']} files, {mb:.1f} MB | copied {r['copied']}, already there {r['skipped']}, "
              f"deleted from Supabase {r['deleted']}, failed {len(r['failed'])}")
        for f in r["failed"][:20]:
            print(f"   failed: {f}")
        total_failed += len(r["failed"])
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
