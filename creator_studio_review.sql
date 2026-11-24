-- Viyo Studio: Story Intelligence stage 4, VIYO's own review of each chapter. Run after creator_studio_chapters.sql.
-- Safe to run more than once. Adds one column; nothing else changes.
--
-- review: the verified scores (each with its quote), what is weak and why, the fix, and whether a targeted
-- revision was kept (and why or why not).
alter table public.studio_story_chapters add column if not exists review jsonb not null default '{}'::jsonb;
