-- Viyo Studio: multi-shot scenes (the AI director's camera shots).
-- Run after creator_studio_visual_audit.sql. Safe to run more than once. Only adds.
--
-- studio_shots: a scene's camera shots, in order. Each has its type (wide
-- two-shot, over-the-shoulder, close-up, reaction, insert, ...), angle,
-- movement, duration, what we see, who is where, which lines are heard, the
-- prompts for its opening picture and its video clip, what must stay the same
-- from the previous shot, why the director chose it, and its own picture and
-- clip. approved_at locks a shot: it is never replaced until unapproved.
-- studio_scenes.shot_plan_approved: the creator approved the scene's shots;
-- only then are shots made, and the film uses them instead of the single clip.
create table if not exists public.studio_shots (
  id uuid primary key default gen_random_uuid(),
  scene_id uuid not null references public.studio_scenes(id) on delete cascade,
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  position int not null,
  shot_type text not null,
  camera_angle text not null default '',
  camera_movement text not null default 'static',
  duration_seconds numeric(5,2) not null default 3 check (duration_seconds between 1 and 20),
  description text not null default '',
  characters jsonb not null default '[]'::jsonb,
  line_indices jsonb not null default '[]'::jsonb,
  dialogue jsonb not null default '[]'::jsonb,
  audio text not null default '',
  frame_prompt text not null default '',
  video_prompt text not null default '',
  continuity text not null default '',
  reason text not null default '',
  image_path text,
  video_path text,
  status text not null default 'planned' check (status in ('planned', 'picture', 'ready', 'failed')),
  made_with jsonb not null default '{}'::jsonb,
  approved_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists studio_shots_scene_idx on public.studio_shots (scene_id, position);
alter table public.studio_shots enable row level security;
drop policy if exists "owner reads" on public.studio_shots;
create policy "owner reads" on public.studio_shots for select using (auth.uid() = owner_id);

alter table public.studio_scenes add column if not exists shot_plan_approved boolean not null default false;

-- The director (shotplan) and each shot's picture and clip. Every job type is
-- listed, so re-running this file never narrows it.
alter table public.studio_jobs drop constraint if exists studio_jobs_kind_check;
alter table public.studio_jobs add constraint studio_jobs_kind_check
  check (kind in ('script', 'breakdown', 'portrait', 'image', 'voice', 'video', 'render', 'lipsync', 'outfits', 'cover',
                  'elements', 'element', 'inspect', 'shotplan', 'shot_image', 'shot_video'));
