-- Creator downloads: an audit log of download links handed out.
-- Only who, which post, what kind and when. No IP address, no device.
-- Written only by the backend (service role); creators can read their own rows.
-- Safe to run more than once.

create table if not exists public.creator_download_events (
  id bigserial primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  post_id uuid references public.posts(id) on delete set null,
  kind text not null check (kind in ('video', 'thumbnail')),
  created_at timestamptz not null default now()
);

create index if not exists creator_download_events_user_idx
  on public.creator_download_events (user_id, created_at desc);

alter table public.creator_download_events enable row level security;

drop policy if exists "creators read own download events" on public.creator_download_events;
create policy "creators read own download events" on public.creator_download_events
  for select using (auth.uid() = user_id);
