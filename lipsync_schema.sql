-- VIYO AI real lip sync (Sync/sync.so) — extends the EXISTING
-- `series_scenes` table (not created here; it already exists and is
-- populated by studio.py's scene-splitting endpoints). Handed over to
-- run by hand in the Supabase SQL editor — same "no migration-runner
-- access from this codebase" constraint every other migration this
-- session has had.

-- The Sync-generated, already-lip-synced talking-portrait video for
-- this scene (studio.py's generate_talking_portrait) — an alternative
-- to the existing Veo `video_url` for a scene carried by one
-- character actually talking. Nullable: a scene with no real lip sync
-- generated yet (every scene before this feature, or one that still
-- just uses Veo or the default Ken Burns pan) just has this unset.
-- No RLS changes: this table already has the same service-role-only
-- posture every other Studio table uses.
alter table series_scenes add column if not exists talking_portrait_url text;
