-- Viyo Studio: the Character Consistency Engine.
-- Run after creator_studio_story.sql. Safe to run more than once. Only adds:
-- no existing column, row or file is changed or removed.
--
-- identity: the character's permanent look as structured data
--   {"basic": {...}, "face": {...}, "body": {...}}: age, height in cm, build,
--   skin tone, face shape, eyes, nose, hair, marks, proportions and so on.
--   Scene text never changes it; only the creator does (while unlocked).
-- reference_views: the reference sheet, one entry per view
--   {"front_full": {"path": "...", "status": "draft"|"approved", "anchor": "...", "at": "..."}, ...}
--   Every view is made from the approved front view (the anchor), so they
--   show the same person. Redoing one view never changes the others.
-- identity_locked_at: set when the creator approves and locks the identity.
-- identity_version: goes up whenever the identity or a view changes.
alter table public.studio_characters add column if not exists identity jsonb not null default '{}'::jsonb;
alter table public.studio_characters add column if not exists reference_views jsonb not null default '{}'::jsonb;
alter table public.studio_characters add column if not exists identity_locked_at timestamptz;
alter table public.studio_characters add column if not exists identity_version int not null default 0;

-- Outfits: what a character wears in part of the story. Each has its own
-- reference picture made from the character's anchor view. Changing clothes
-- in the story never touches the permanent identity.
create table if not exists public.studio_outfits (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  character_id uuid not null references public.studio_characters(id) on delete cascade,
  name text not null check (char_length(name) between 1 and 60),
  description text not null default '' check (char_length(description) <= 800),
  reference_path text,
  is_default boolean not null default false,
  position int not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists studio_outfits_character_idx on public.studio_outfits (character_id);
create index if not exists studio_outfits_project_idx on public.studio_outfits (project_id);

-- Scenes: which outfit each character wears ({"<character id>": "<outfit id>"};
-- a character not listed keeps the outfit from the scene before), and the
-- camera framing, which decides which reference views are sent.
alter table public.studio_scenes add column if not exists outfits jsonb not null default '{}'::jsonb;
alter table public.studio_scenes add column if not exists camera text not null default 'auto';
alter table public.studio_scenes drop constraint if exists studio_scenes_camera_check;
alter table public.studio_scenes add constraint studio_scenes_camera_check
  check (camera in ('auto', 'close_up', 'front', 'side', 'back', 'full_body', 'wide'));

-- Owners can read their own outfits; every write goes through the backend.
alter table public.studio_outfits enable row level security;
drop policy if exists "owner reads" on public.studio_outfits;
create policy "owner reads" on public.studio_outfits for select using (auth.uid() = owner_id);
