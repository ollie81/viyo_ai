-- VIYO AI Drama Creator — extends the EXISTING `series` and
-- `series_characters` tables (not created here; they already exist and
-- are populated by studio.py / the Flutter app today). Handed over to
-- run by hand in the Supabase SQL editor — same "no migration-runner
-- access from this codebase" constraint every other migration this
-- session has had.
--
-- Every `add column if not exists` below is safe to re-run. No RLS
-- changes: these tables already have the same service-role-only
-- posture every other Studio table uses (drama_creator.py's new
-- endpoints go through the same supabase_admin client studio.py
-- already uses for series_characters/series_scenes).

-- A brand-new drama's structured story, written by the AI Drama
-- Assistant from the user's one-line idea. Nothing about "plot"
-- existed as structured data before this — it used to live only
-- inside whatever full script got pasted into the old flow.
alter table series add column if not exists plot_json jsonb;

-- "Confirmed, no re-editing later" — once true, drama_creator.py
-- rejects further edits to this series' plot/characters (see
-- _require_setup_editable in drama_creator.py). A real UI action, not
-- an automatic lock, so an accidental tap never permanently traps a
-- bad story (the flag itself can still be hand-reset in Supabase if
-- truly needed, same escape hatch every other "locked" flag in this
-- app has).
alter table series add column if not exists setup_confirmed boolean not null default false;
alter table series add column if not exists setup_confirmed_at timestamptz;

-- Captured at idea-generation time for the new AI-driven storyboard
-- step (drama_creator.py's /storyboard endpoint) to pace scene count
-- and length against. Episode rendering itself (assemble_episode)
-- stays 9:16-only for now regardless of this value — see
-- drama_creator.py's module docstring for what's and isn't wired yet.
alter table series add column if not exists episode_duration_seconds int not null default 30;
alter table series add column if not exists drama_aspect_ratio text not null default '9:16';

-- Richer per-character fields the AI Drama Assistant now generates
-- and the four-view sheet step fills in — series_characters already
-- had name/age/gender/appearance/clothing/personality/portrait_url/
-- voice_id/costume_lock; these are additive, no existing column
-- changes, and every existing row (and the old paste-a-script flow)
-- is unaffected since all of these default to null/empty.
alter table series_characters add column if not exists skin_tone text not null default '';
alter table series_characters add column if not exists face_shape text not null default '';
alter table series_characters add column if not exists eye_color text not null default '';
alter table series_characters add column if not exists hairstyle text not null default '';
alter table series_characters add column if not exists body_proportions text not null default '';
alter table series_characters add column if not exists voice_description text not null default '';

-- The three additional reference-sheet views generated from the
-- character's existing portrait_url as an image reference (the "face"
-- view IS portrait_url — no new column needed for it), plus the single
-- composed strip image shown in the UI. All nullable: a character
-- with no four-view sheet yet (every character before this feature,
-- or one whose view generation failed) just has these unset — never a
-- fake/placeholder URL standing in for a real one.
alter table series_characters add column if not exists front_view_url text;
alter table series_characters add column if not exists side_view_url text;
alter table series_characters add column if not exists rear_view_url text;
alter table series_characters add column if not exists four_view_sheet_url text;
