-- Viyo Studio: background music, sound effects and image quality.
-- Run after creator_studio_schema.sql. Safe to run more than once.

alter table public.studio_projects add column if not exists music_path text;          -- creator's own track (private bucket)
alter table public.studio_projects add column if not exists music_name text;
alter table public.studio_projects add column if not exists music_volume numeric(4,2) not null default 0.25;
alter table public.studio_projects add column if not exists image_quality text not null default 'standard';
alter table public.studio_projects drop constraint if exists studio_projects_image_quality_check;
alter table public.studio_projects add constraint studio_projects_image_quality_check
  check (image_quality in ('standard', 'high'));
alter table public.studio_projects drop constraint if exists studio_projects_music_volume_check;
alter table public.studio_projects add constraint studio_projects_music_volume_check
  check (music_volume between 0 and 1);

alter table public.studio_scenes add column if not exists sfx jsonb not null default '[]'::jsonb;   -- ["rain", "whoosh"]

alter table public.studio_assets drop constraint if exists studio_assets_kind_check;
alter table public.studio_assets add constraint studio_assets_kind_check
  check (kind in ('image', 'video', 'audio', 'portrait', 'render', 'music'));
