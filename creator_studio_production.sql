-- Viyo Studio: one-click production.
-- Run after creator_studio_publish.sql. Safe to run more than once. Only adds:
-- one table and four functions; nothing existing is changed.
--
-- A production run makes everything a production still needs, in order:
-- character sheets -> scene pictures -> voices -> videos -> lip sync ->
-- the final film. The creator sees the total price first; on confirmation
-- the whole amount is reserved at once (one credit hold), so it can't be
-- spent elsewhere mid-production. Each step then draws its own hold from
-- that reservation and keeps only its real cost, exactly like a job started
-- by hand; failed steps are refunded. What's left of the reservation goes
-- back when the run ends.

create table if not exists public.studio_production_runs (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  status text not null default 'running'
    check (status in ('running', 'needs_attention', 'done', 'canceled')),
  options jsonb not null default '{}'::jsonb,
  plan jsonb not null default '{}'::jsonb,        -- the quote the creator confirmed, by stage
  quoted_credits bigint not null default 0,
  hold_id uuid references public.studio_credit_holds(id),
  render_id uuid,
  problem text,                                   -- why it needs the creator, in plain words
  request_key text not null unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  finished_at timestamptz
);
create index if not exists studio_production_runs_project_idx on public.studio_production_runs (project_id, created_at desc);
create index if not exists studio_production_runs_active_idx on public.studio_production_runs (status) where status = 'running';

alter table public.studio_production_runs enable row level security;
drop policy if exists "owner reads" on public.studio_production_runs;
create policy "owner reads" on public.studio_production_runs for select using (auth.uid() = owner_id);

-- Gives back part of an open hold (to the lots it came from, last-taken
-- first) and keeps the hold open with the rest.
create or replace function public._studio_hold_release(p_hold_id uuid, p_amount bigint)
returns bigint language plpgsql security definer set search_path = public as $$
declare
  h record;
  v_back bigint;
  v_give bigint;
  a record;
begin
  select * into h from public.studio_credit_holds where id = p_hold_id for update;
  if not found or h.status <> 'open' then
    return 0;
  end if;
  v_back := least(greatest(coalesce(p_amount, 0), 0), h.credits_held);
  if v_back = 0 then
    return 0;
  end if;
  for a in select * from public.studio_credit_hold_allocations where hold_id = p_hold_id and credits > 0 order by seq desc loop
    exit when v_back <= 0;
    v_give := least(v_back, a.credits);
    update public.studio_credit_lots set credits_remaining = credits_remaining + v_give where id = a.lot_id;
    if v_give = a.credits then
      delete from public.studio_credit_hold_allocations where hold_id = a.hold_id and seq = a.seq;
    else
      update public.studio_credit_hold_allocations set credits = credits - v_give where hold_id = a.hold_id and seq = a.seq;
    end if;
    v_back := v_back - v_give;
  end loop;
  v_give := least(greatest(coalesce(p_amount, 0), 0), h.credits_held) - v_back;
  update public.studio_credit_holds set credits_held = credits_held - v_give where id = p_hold_id;
  if v_give > 0 then
    insert into public.studio_credit_ledger (user_id, kind, credits, hold_id) values (h.user_id, 'release', v_give, p_hold_id);
  end if;
  return v_give;
end;
$$;

-- Reserves the confirmed total and records the run, in one step.
-- Idempotent on p_key.
create or replace function public.studio_run_reserve(
  p_owner uuid, p_project uuid, p_credits bigint, p_key text, p_options jsonb, p_plan jsonb)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_run uuid := gen_random_uuid();
  v_hold uuid;
  v_existing uuid;
begin
  perform pg_advisory_xact_lock(hashtext('studio_credits:' || p_owner::text));
  select id into v_existing from public.studio_production_runs where request_key = p_key;
  if found then
    return jsonb_build_object('ok', true, 'run_id', v_existing, 'duplicate', true);
  end if;
  if not exists (select 1 from public.studio_projects where id = p_project and owner_id = p_owner) then
    return jsonb_build_object('ok', false, 'error', 'project_not_found');
  end if;
  if exists (select 1 from public.studio_production_runs where project_id = p_project and status in ('running', 'needs_attention')) then
    return jsonb_build_object('ok', false, 'error', 'already_running');
  end if;
  begin
    v_hold := public._studio_hold(p_owner, greatest(p_credits, 0), null);
  exception when sqlstate 'P0001' then
    return jsonb_build_object('ok', false, 'error', 'insufficient_credits',
                              'balance', (public.studio_credit_balance(p_owner))->'available', 'needed', p_credits);
  end;
  insert into public.studio_production_runs (id, project_id, owner_id, options, plan, quoted_credits, hold_id, request_key)
    values (v_run, p_project, p_owner, coalesce(p_options, '{}'::jsonb), coalesce(p_plan, '{}'::jsonb),
            greatest(p_credits, 0), v_hold, p_key);
  return jsonb_build_object('ok', true, 'run_id', v_run);
end;
$$;

-- Starts one step of a run: moves its price out of the run's reservation
-- and into the step's own hold, in one transaction, so nothing else can
-- take those credits in between. A retry (p_from_reserve = false) holds
-- from the balance instead, where the failed try's refund went, so the
-- reservation stays whole for the steps still to come.
create or replace function public.studio_run_create_job(
  p_run uuid, p_kind text, p_lane text, p_key text, p_credits bigint, p_input jsonb,
  p_scene uuid default null, p_character uuid default null, p_from_reserve boolean default true)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  r record;
  v_existing uuid;
begin
  select * into r from public.studio_production_runs where id = p_run for update;
  if not found or r.status <> 'running' then
    return jsonb_build_object('ok', false, 'error', 'run_not_running');
  end if;
  select id into v_existing from public.studio_jobs where idempotency_key = p_key;
  if found then
    return jsonb_build_object('ok', true, 'job_id', v_existing, 'duplicate', true);
  end if;
  if p_from_reserve then
    perform public._studio_hold_release(r.hold_id, p_credits);
  end if;
  return public.studio_create_job(r.owner_id, r.project_id, p_kind, p_lane, p_key, p_credits, p_input,
                                  p_scene, p_character, 1000);
end;
$$;

-- Ends a run (done or canceled): what's left of the reservation goes back.
create or replace function public.studio_run_finish(p_run uuid, p_status text, p_problem text default null)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  r record;
  v_back bigint := 0;
begin
  select * into r from public.studio_production_runs where id = p_run for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'not_found');
  end if;
  if p_status in ('done', 'canceled') then
    if r.hold_id is not null then
      select credits_held into v_back from public.studio_credit_holds where id = r.hold_id;
      perform public._studio_settle(r.hold_id, 0, null);
    end if;
    update public.studio_production_runs set status = p_status, problem = p_problem, finished_at = now(), updated_at = now()
     where id = p_run;
  else
    update public.studio_production_runs set status = p_status, problem = p_problem, updated_at = now() where id = p_run;
  end if;
  return jsonb_build_object('ok', true, 'returned', coalesce(v_back, 0));
end;
$$;

do $$
declare f text;
begin
  foreach f in array array[
    '_studio_hold_release(uuid, bigint)',
    'studio_run_reserve(uuid, uuid, bigint, text, jsonb, jsonb)',
    'studio_run_create_job(uuid, text, text, text, bigint, jsonb, uuid, uuid, boolean)',
    'studio_run_finish(uuid, text, text)'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
