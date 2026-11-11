-- Viyo Studio: shot-to-shot visual continuity checks.
-- Run after creator_studio_shots.sql. Safe to run more than once. Only adds.
--
-- studio_shots.inspection: the latest continuity check of the shot's picture
-- (scores, problems, what the picture shows, which picture it was compared
-- with, and the creator's acceptance of a flagged change for that picture).
alter table public.studio_shots add column if not exists inspection jsonb not null default '{}'::jsonb;

-- shot_inspect: one shot picture's check. Every job type is listed, so re-running
-- this file never narrows it.
alter table public.studio_jobs drop constraint if exists studio_jobs_kind_check;
alter table public.studio_jobs add constraint studio_jobs_kind_check
  check (kind in ('script', 'breakdown', 'portrait', 'image', 'voice', 'video', 'render', 'lipsync', 'outfits', 'cover',
                  'elements', 'element', 'inspect', 'shotplan', 'shot_image', 'shot_video', 'shot_inspect'));
