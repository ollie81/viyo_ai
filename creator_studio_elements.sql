-- Viyo Studio: places and objects that stay the same in every scene.
-- Run after creator_studio_production.sql. Safe to run more than once. Only adds.
--
-- studio_elements: a production's recurring locations (Vance mansion, the
-- rooftop) and objects (a black leather bag, a phone, keys, a revolver),
-- each with a description and one reference picture. Every scene picture
-- and video that uses one gets that picture and description, so it looks
-- the same every time.
create table if not exists public.studio_elements (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  kind text not null check (kind in ('location', 'prop')),
  name text not null check (char_length(name) between 1 and 80),
  description text not null default '' check (char_length(description) <= 800),
  details jsonb not null default '{}'::jsonb,     -- location: time_of_day, mood; object: owner, colour, material
  reference_path text,
  status text not null default 'draft' check (status in ('draft', 'approved')),
  position int not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists studio_elements_project_idx on public.studio_elements (project_id, kind);

alter table public.studio_elements enable row level security;
drop policy if exists "owner reads" on public.studio_elements;
create policy "owner reads" on public.studio_elements for select using (auth.uid() = owner_id);

-- Which place and objects a scene uses. location_id null = worked out from
-- the scene's heading and text; props null = worked out from the text,
-- [] = no objects, a list = exactly these.
alter table public.studio_scenes add column if not exists location_id uuid;
alter table public.studio_scenes add column if not exists props jsonb;

-- The job types: find places and objects in the script; make one's picture.
alter table public.studio_jobs drop constraint if exists studio_jobs_kind_check;
alter table public.studio_jobs add constraint studio_jobs_kind_check
  check (kind in ('script', 'breakdown', 'portrait', 'image', 'voice', 'video', 'render', 'lipsync', 'outfits', 'cover',
                  'elements', 'element', 'inspect', 'shotplan', 'shot_image', 'shot_video'));
