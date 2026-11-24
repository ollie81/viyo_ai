-- Viyo Studio: Story Intelligence stage 3, writing the novel's chapters. Run after creator_studio_novel.sql.
-- Safe to run more than once. Adds one table and adds 'chapter' to the job types; nothing else changes.
--
-- studio_story_chapters: one row per chapter, the latest version of its text. A chapter is written by a
-- 'chapter' job (the same queue, holds, refunds and retries as every Studio job), then the creator approves it.
-- extraction keeps what VIYO found when it read the chapter back: the facts, events and threads it proposed for
-- the story memory (as proposals only), what it rejected for lack of evidence, and any continuity problems.
create table if not exists public.studio_story_chapters (
  id uuid primary key default gen_random_uuid(),
  story_id uuid not null references public.studio_stories(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  number int not null check (number >= 1),
  title text not null default '' check (char_length(title) <= 200),
  text text not null default '',
  summary text not null default '' check (char_length(summary) <= 2000),
  words int not null default 0,
  version int not null default 1,
  outline_version int,
  status text not null default 'draft' check (status in ('draft', 'approved')),
  approved_at timestamptz,
  extraction jsonb not null default '{}'::jsonb,
  job_id uuid,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (story_id, number)
);
create index if not exists studio_story_chapters_story_idx on public.studio_story_chapters (story_id, number);

alter table public.studio_story_chapters enable row level security;
drop policy if exists "owner reads" on public.studio_story_chapters;
create policy "owner reads" on public.studio_story_chapters for select using (auth.uid() = owner_id);

-- The job type that writes a chapter (every existing type kept, so re-running this file never narrows it).
alter table public.studio_jobs drop constraint if exists studio_jobs_kind_check;
alter table public.studio_jobs add constraint studio_jobs_kind_check
  check (kind in ('script', 'breakdown', 'portrait', 'image', 'voice', 'video', 'render', 'lipsync', 'outfits', 'cover',
                  'elements', 'element', 'inspect', 'shotplan', 'shot_image', 'shot_video', 'shot_inspect', 'chapter'));
