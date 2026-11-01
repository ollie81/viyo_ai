-- Creator performance history for the AI Repurposer (inert until rows are written).
-- Run manually in the Supabase SQL editor. Safe to re-run.

create table if not exists public.repurpose_clip_performance (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  clip_id uuid,
  post_id uuid,
  category text,
  duration double precision,
  score integer,
  views integer not null default 0,
  completion_rate double precision,      -- 0-1, share of views watched to the end
  avg_watch_seconds double precision,
  rewatch_rate double precision,         -- 0-1, views that were repeats
  shares integer not null default 0,
  saves integer not null default 0,
  comments integer not null default 0,
  updated_at timestamptz not null default now(),
  unique (post_id)
);

create index if not exists repurpose_clip_performance_user_idx
  on public.repurpose_clip_performance (user_id);

alter table public.repurpose_clip_performance enable row level security;
-- No policies: only the backend (service role) reads or writes this table.
