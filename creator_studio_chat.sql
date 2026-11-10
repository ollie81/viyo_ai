-- Viyo Studio: the assistant chat.
-- Run after creator_studio_elements.sql. Safe to run more than once. Only adds.
--
-- studio_conversations: one creator's chat about one production. `summary`
-- is a running summary of older messages (so the AI isn't sent the whole
-- history every time); `memory` keeps what the creator decided, apart from
-- ideas only discussed.
-- studio_messages: each message. Assistant messages can carry proposals:
-- concrete changes the creator applies or dismisses ({type, title, summary,
-- data, status: proposed|applied|dismissed|failed}). Nothing in a proposal is
-- a story fact until the creator applies it.
create table if not exists public.studio_conversations (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  title text not null default '',
  summary text not null default '',
  summarized_count int not null default 0,
  memory jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists studio_conversations_project_idx on public.studio_conversations (project_id, updated_at desc);

create table if not exists public.studio_messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.studio_conversations(id) on delete cascade,
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null check (char_length(content) <= 20000),
  proposals jsonb not null default '[]'::jsonb,
  credits int not null default 0,
  client_key text unique,                -- the app's request id: a retry never sends or charges twice
  created_at timestamptz not null default clock_timestamp()   -- keeps a question before its answer
);
create index if not exists studio_messages_conversation_idx on public.studio_messages (conversation_id, created_at);

alter table public.studio_conversations enable row level security;
alter table public.studio_messages enable row level security;
drop policy if exists "owner reads" on public.studio_conversations;
create policy "owner reads" on public.studio_conversations for select using (auth.uid() = owner_id);
drop policy if exists "owner reads" on public.studio_messages;
create policy "owner reads" on public.studio_messages for select using (auth.uid() = owner_id);

-- A chat reply is a quick AI call made while the creator waits, not a queued
-- job: these hold credits for it and keep only what it really cost.
create or replace function public.studio_hold_credits(p_owner uuid, p_credits bigint)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_hold uuid;
begin
  perform pg_advisory_xact_lock(hashtext('studio_credits:' || p_owner::text));
  begin
    v_hold := public._studio_hold(p_owner, greatest(p_credits, 0), null);
  exception when sqlstate 'P0001' then
    return jsonb_build_object('ok', false, 'error', 'insufficient_credits',
                              'balance', (public.studio_credit_balance(p_owner))->'available', 'needed', p_credits);
  end;
  return jsonb_build_object('ok', true, 'hold_id', v_hold);
end;
$$;

create or replace function public.studio_settle_credits(p_hold uuid, p_used bigint)
returns bigint language plpgsql security definer set search_path = public as $$
begin
  return public._studio_settle(p_hold, p_used, null);
end;
$$;

do $$
declare f text;
begin
  foreach f in array array['studio_hold_credits(uuid, bigint)', 'studio_settle_credits(uuid, bigint)'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
