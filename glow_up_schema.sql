-- Viyo Glow Up (Stage 1): saved style suggestions.
-- Run manually in the Supabase SQL editor. Safe to re-run.
--
-- Stores TEXT ONLY. No photo is ever stored for Glow Up Stage 1.
-- Rows are written by the backend (service role) and readable/deletable
-- only by their owner. Deleting the account removes them (cascade).

create table if not exists public.glow_up_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),
  style_goal text,
  presentation text,
  recommendations jsonb not null
);

create index if not exists glow_up_sessions_user_idx
  on public.glow_up_sessions (user_id, created_at desc);

alter table public.glow_up_sessions enable row level security;

drop policy if exists "glow_up_sessions_owner_read" on public.glow_up_sessions;
create policy "glow_up_sessions_owner_read" on public.glow_up_sessions
  for select using (auth.uid() = user_id);

drop policy if exists "glow_up_sessions_owner_delete" on public.glow_up_sessions;
create policy "glow_up_sessions_owner_delete" on public.glow_up_sessions
  for delete using (auth.uid() = user_id);

-- No insert/update policies: only the backend writes these rows.

-- ---------------------------------------------------------------------
-- Stage 2: saved AI hairstyle previews (only when the user taps Save).
-- The selfie itself is still never stored.

create table if not exists public.glow_up_previews (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),
  kind text not null,
  style text not null,
  image_path text not null
);

create index if not exists glow_up_previews_user_idx
  on public.glow_up_previews (user_id, created_at desc);

alter table public.glow_up_previews enable row level security;

drop policy if exists "glow_up_previews_owner_read" on public.glow_up_previews;
create policy "glow_up_previews_owner_read" on public.glow_up_previews
  for select using (auth.uid() = user_id);

drop policy if exists "glow_up_previews_owner_delete" on public.glow_up_previews;
create policy "glow_up_previews_owner_delete" on public.glow_up_previews
  for delete using (auth.uid() = user_id);

-- Private bucket: no public links; the backend hands out 1-hour signed links.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('glow-up-previews', 'glow-up-previews', false, 5242880, array['image/jpeg'])
on conflict (id) do update
  set public = false, file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

-- ---------------------------------------------------------------------
-- Optional photo saving (only with explicit consent; off by default).

alter table public.glow_up_sessions add column if not exists photo_path text;

-- Private bucket: no public links. No storage policies are created, so
-- signed-in users cannot read or list it directly; only the backend
-- (service role) can, and it hands the owner 1-hour signed links.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('glow-up-photos', 'glow-up-photos', false, 5242880, array['image/jpeg'])
on conflict (id) do update
  set public = false, file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;
