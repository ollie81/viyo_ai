-- Viyo Studio: AI-made covers (poster art) for productions.
-- Run after creator_studio_identity.sql. Safe to run more than once. Only adds.
--
-- cover_path: the latest cover picture (3:4 poster, the title in the art and
-- the small Viyo mark in a corner). Earlier covers stay in studio_assets.
alter table public.studio_projects add column if not exists cover_path text;

-- The cover job type.
alter table public.studio_jobs drop constraint if exists studio_jobs_kind_check;
alter table public.studio_jobs add constraint studio_jobs_kind_check
  check (kind in ('script', 'breakdown', 'portrait', 'image', 'voice', 'video', 'render', 'lipsync', 'outfits', 'cover'));
