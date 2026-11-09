-- Viyo Studio: what each scene was made with.
-- Run after creator_studio_identity.sql. Safe to run more than once. Only adds.
--
-- made_with: for each character in the scene, the identity version and
-- outfit its picture or video was made with, and the reference pictures that
-- were sent: {"<character id>": {"identity_version": 3, "locked": true,
-- "outfit_id": "..."}, "_refs": ["Maya: face close-up ...", ...]}.
-- The Studio uses it to flag scenes made before a character's identity or
-- outfit changed, so the creator can redo just those.
alter table public.studio_scenes add column if not exists made_with jsonb not null default '{}'::jsonb;
