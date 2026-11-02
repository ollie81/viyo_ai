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
