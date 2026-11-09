-- Viyo Studio: publishing productions to Viyo.
-- Run after creator_studio_cover.sql. Safe to run more than once. Only adds.
--
-- viyo_series_id: the creator's Viyo drama (public `series` row) this
-- production's episodes are published into. Episodes of the same Studio
-- series share it.
-- studio_renders.post_id / published_at: the Viyo post a render became.
alter table public.studio_projects add column if not exists viyo_series_id uuid;
alter table public.studio_renders add column if not exists post_id uuid;
alter table public.studio_renders add column if not exists published_at timestamptz;
