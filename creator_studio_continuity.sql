-- Viyo Studio: scene continuity.
-- Run after creator_studio_music.sql. Safe to run more than once. Only adds.
--
-- continuity: the continuity director's record for each scene (who is in
-- frame and where, what they hold and wear, feelings, start and end state,
-- camera, light, and how it follows the scene before). The next scene's
-- picture and video are given the previous scene's end state from it.
alter table public.studio_scenes add column if not exists continuity jsonb not null default '{}'::jsonb;
