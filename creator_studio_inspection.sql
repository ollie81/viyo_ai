-- Viyo Studio: visual continuity checks, approved scene pictures, restoring versions.
-- Run after creator_studio_assistant_memory.sql. Safe to run more than once. Only adds.
--
-- studio_scenes.inspection: the latest picture check (scores, problems with
--   severity/certainty/intentional, fixes, what the picture shows), the
--   signature of what it was checked against, and its cost. A check is reused
--   while nothing it depends on has changed.
-- studio_scenes.image_approved_path: the exact picture the creator approved.
--   An approved picture is never replaced until the creator unapproves it.
-- studio_projects.auto_inspect: check every new scene picture (about 2 credits,
--   included in the picture's price). On by default; the creator can turn it off.
-- Old pictures and videos are already kept in studio_assets; restoring one
-- just points the scene back at it.
alter table public.studio_scenes add column if not exists inspection jsonb not null default '{}'::jsonb;
alter table public.studio_scenes add column if not exists image_approved_path text;
alter table public.studio_projects add column if not exists auto_inspect boolean not null default true;

-- The new "inspect" job (checking pictures that already exist). Every job type
-- is listed, so re-running this file never narrows it.
alter table public.studio_jobs drop constraint if exists studio_jobs_kind_check;
alter table public.studio_jobs add constraint studio_jobs_kind_check
  check (kind in ('script', 'breakdown', 'portrait', 'image', 'voice', 'video', 'render', 'lipsync', 'outfits', 'cover',
                  'elements', 'element', 'inspect', 'shotplan', 'shot_image', 'shot_video', 'shot_inspect'));
