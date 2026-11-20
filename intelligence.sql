-- VIYO intelligence: viewer activity events (recommendations, trending, analytics), viewer preferences, and
-- fraud review flags with appeals. Additive only, safe to run more than once. Changes no existing data.
-- Every table is written and read only by the backend (RLS on, no policies), so the app can't fake events.

-- What viewers do, recorded by the backend (never directly by the app): views, watch progress, completions,
-- likes, comments, shares, saves, "not interested", searches. Kept 180 days (see the cleanup at the end).
create table if not exists public.viewer_events (
  id bigserial primary key,
  user_id uuid not null,
  kind text not null check (kind in ('impression', 'view', 'progress', 'complete', 'like', 'comment', 'share',
                                     'save', 'follow', 'not_interested', 'not_interested_creator', 'search')),
  post_id uuid,
  series_id uuid,
  creator_id uuid,
  value real,                 -- progress: watched fraction 0..1; view: seconds watched (optional)
  query text,                 -- search only (trimmed, at most 100 characters)
  created_at timestamptz not null default now()
);
create index if not exists viewer_events_user on public.viewer_events (user_id, created_at desc);
create index if not exists viewer_events_post on public.viewer_events (post_id, kind, created_at desc);
create index if not exists viewer_events_series on public.viewer_events (series_id, kind, created_at desc);
create index if not exists viewer_events_creator on public.viewer_events (creator_id, kind, created_at desc);
create index if not exists viewer_events_recent on public.viewer_events (created_at desc, kind);
alter table public.viewer_events enable row level security;

-- A viewer's choices about personalization. personalized = false: For You uses no history at all.
create table if not exists public.viewer_prefs (
  user_id uuid primary key references auth.users(id) on delete cascade,
  personalized boolean not null default true,
  updated_at timestamptz not null default now()
);
alter table public.viewer_prefs enable row level security;

-- Suspicious patterns found by the backend, for a person to review. A flag never takes coins, money or the
-- account away by itself; while it is open, that account's engagement simply doesn't count toward trending,
-- and a held payout waits for review. The flagged person can see a held decision and appeal it.
create table if not exists public.fraud_flags (
  id uuid primary key default gen_random_uuid(),
  subject_type text not null check (subject_type in ('user', 'post')),
  subject_id uuid not null,
  reason text not null,
  detail jsonb not null default '{}'::jsonb,
  score real not null default 0,
  status text not null default 'open' check (status in ('open', 'cleared', 'confirmed')),
  appeal text,
  appealed_at timestamptz,
  reviewed_note text,
  reviewed_at timestamptz,
  created_at timestamptz not null default now()
);
create index if not exists fraud_flags_subject on public.fraud_flags (subject_type, subject_id, status);
create index if not exists fraud_flags_open on public.fraud_flags (status, created_at desc);
alter table public.fraud_flags enable row level security;

-- Search: trigram indexes so partial and misspelled matches stay fast as the catalogue grows.
create extension if not exists pg_trgm;
create index if not exists series_title_trgm on public.series using gin (title gin_trgm_ops);
create index if not exists profiles_username_trgm on public.profiles using gin (username gin_trgm_ops);

-- Keep 180 days of events (run occasionally, or schedule it):
-- delete from public.viewer_events where created_at < now() - interval '180 days';

-- Check (read-only):
-- select count(*) from public.viewer_events;
