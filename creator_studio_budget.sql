-- Viyo Studio: budget protection, storyboard review, production credit history and free defect fixes.
-- Run after creator_studio_production.sql, creator_studio_shots.sql and creator_studio_chapters.sql.
-- Safe to run more than once. Nothing existing is removed; existing runs keep working.
--
-- Rules (agreed with the owner)
--   * A production never spends more than the creator approved. Every step of a run is checked against the run's
--     approved budget (the confirmed quote plus any increase the creator approved) in the same transaction that
--     creates it, with the run row locked, so concurrent workers can't get past the cap.
--     committed = what the run's finished steps were charged + what its waiting/running steps hold.
--     A step starts only if committed + its price <= approved budget.
--   * When the approved budget isn't enough, the run pauses ('needs_approval') with the extra amount and why. It
--     continues only after the creator approves that exact increase, which is reserved first. Never automatic.
--   * A retry reuses the credits its failed try returned; it can't take the run past its approved budget.
--   * With storyboard review on (the default for new runs), the run stops before paid video ('storyboard_review')
--     until the creator approves the storyboard.
--   * Every reservation, increase, pause and return is written to studio_run_budget_events (the run's history).
--   * Free defect fixes (job kind 'shot_fix'): at most one successful free fix per shot and per kind of output, and
--     at most two checks per shot and output, enforced by unique keys.

-- ---------------------------------------------------------------- runs
alter table public.studio_production_runs add column if not exists approved_credits bigint;
alter table public.studio_production_runs add column if not exists needs_credits bigint;
alter table public.studio_production_runs add column if not exists storyboard_approved_at timestamptz;
alter table public.studio_production_runs drop constraint if exists studio_production_runs_status_check;
alter table public.studio_production_runs add constraint studio_production_runs_status_check
  check (status in ('running', 'needs_attention', 'needs_approval', 'storyboard_review', 'done', 'canceled'));

create table if not exists public.studio_run_budget_events (
  id bigserial primary key,
  run_id uuid not null references public.studio_production_runs(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  kind text not null check (kind in ('reserved', 'increase_requested', 'increase_approved', 'returned',
                                     'storyboard_review', 'storyboard_approved')),
  credits bigint not null default 0,
  note text not null default '' check (char_length(note) <= 500),
  created_at timestamptz not null default now()
);
create index if not exists studio_run_budget_events_run on public.studio_run_budget_events (run_id, id);
alter table public.studio_run_budget_events enable row level security;
drop policy if exists "owner reads" on public.studio_run_budget_events;
create policy "owner reads" on public.studio_run_budget_events for select using (auth.uid() = owner_id);
revoke insert, update, delete on public.studio_run_budget_events from anon, authenticated;
grant all on public.studio_run_budget_events to service_role;
grant usage, select on sequence public.studio_run_budget_events_id_seq to service_role;

-- Job kinds: every kind so far, plus the free defect fix.
alter table public.studio_jobs drop constraint if exists studio_jobs_kind_check;
alter table public.studio_jobs add constraint studio_jobs_kind_check
  check (kind in ('script', 'breakdown', 'portrait', 'image', 'voice', 'video', 'render', 'lipsync', 'outfits', 'cover',
                  'elements', 'element', 'inspect', 'shotplan', 'shot_image', 'shot_video', 'shot_inspect', 'chapter',
                  'shot_fix'));

-- ---------------------------------------------------------------- the run's numbers
create or replace function public.studio_run_approved(r public.studio_production_runs)
returns bigint language sql stable as $$ select coalesce(r.approved_credits, r.quoted_credits) $$;

-- What the run has spent or tied up: finished steps at their real charge, open steps at what they hold.
create or replace function public.studio_run_committed(p_run uuid)
returns bigint language sql stable security definer set search_path = public as $$
  select coalesce(sum(case when status = 'succeeded' then coalesce(charged_credits, 0)
                           when status in ('queued', 'running') then coalesce(estimated_credits, 0)
                           else 0 end), 0)::bigint
    from public.studio_jobs where idempotency_key like 'run:' || p_run::text || ':%' $$;

-- ---------------------------------------------------------------- reserve (records the approved budget)
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
  if exists (select 1 from public.studio_production_runs where project_id = p_project
              and status in ('running', 'needs_attention', 'needs_approval', 'storyboard_review')) then
    return jsonb_build_object('ok', false, 'error', 'already_running');
  end if;
  begin
    v_hold := public._studio_hold(p_owner, greatest(p_credits, 0), null);
  exception when sqlstate 'P0001' then
    return jsonb_build_object('ok', false, 'error', 'insufficient_credits',
                              'balance', (public.studio_credit_balance(p_owner))->'available', 'needed', p_credits);
  end;
  insert into public.studio_production_runs (id, project_id, owner_id, options, plan, quoted_credits, approved_credits,
                                              hold_id, request_key)
    values (v_run, p_project, p_owner, coalesce(p_options, '{}'::jsonb), coalesce(p_plan, '{}'::jsonb),
            greatest(p_credits, 0), greatest(p_credits, 0), v_hold, p_key);
  insert into public.studio_run_budget_events (run_id, owner_id, kind, credits, note)
    values (v_run, p_owner, 'reserved', greatest(p_credits, 0), 'Approved budget reserved when the production started.');
  return jsonb_build_object('ok', true, 'run_id', v_run);
end;
$$;

-- ---------------------------------------------------------------- one step, inside the approved budget
create or replace function public.studio_run_create_job(
  p_run uuid, p_kind text, p_lane text, p_key text, p_credits bigint, p_input jsonb,
  p_scene uuid default null, p_character uuid default null, p_from_reserve boolean default true)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  r public.studio_production_runs;
  v_existing uuid;
  v_approved bigint;
  v_committed bigint;
  v_left bigint;
begin
  select * into r from public.studio_production_runs where id = p_run for update;   -- one step at a time per run
  if not found or r.status <> 'running' then
    return jsonb_build_object('ok', false, 'error', 'run_not_running');
  end if;
  select id into v_existing from public.studio_jobs where idempotency_key = p_key;
  if found then
    return jsonb_build_object('ok', true, 'job_id', v_existing, 'duplicate', true);
  end if;
  v_approved := public.studio_run_approved(r);
  v_committed := public.studio_run_committed(p_run);
  if v_committed + greatest(p_credits, 0) > v_approved then
    -- Never past the approved budget: pause and ask (the caller adds the reason and the full amount).
    update public.studio_production_runs
       set status = 'needs_approval', needs_credits = v_committed + greatest(p_credits, 0) - v_approved,
           updated_at = now()
     where id = p_run;
    return jsonb_build_object('ok', false, 'error', 'budget_exceeded', 'approved', v_approved,
                              'committed', v_committed, 'needed', v_committed + greatest(p_credits, 0) - v_approved);
  end if;
  -- Move the step's price out of the run's reservation. Whatever the reservation no longer has came back to the
  -- balance from this run's own failed or cheaper steps (the budget check above guarantees that), so the step can
  -- take it from there without going past what the creator approved.
  select coalesce(credits_held, 0) into v_left from public.studio_credit_holds where id = r.hold_id and status = 'open';
  perform public._studio_hold_release(r.hold_id, least(greatest(p_credits, 0), coalesce(v_left, 0)));
  return public.studio_create_job(r.owner_id, r.project_id, p_kind, p_lane, p_key, p_credits, p_input,
                                  p_scene, p_character, 1000);
end;
$$;

-- The run asks the creator for more: records how much and why, and pauses.
create or replace function public.studio_run_request_budget(p_run uuid, p_extra bigint, p_reason text)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  r public.studio_production_runs;
begin
  select * into r from public.studio_production_runs where id = p_run for update;
  if not found or r.status not in ('running', 'needs_approval') then
    return jsonb_build_object('ok', false, 'error', 'run_not_running');
  end if;
  update public.studio_production_runs set status = 'needs_approval', needs_credits = greatest(p_extra, 1),
         problem = left(coalesce(p_reason, ''), 500), updated_at = now() where id = p_run;
  insert into public.studio_run_budget_events (run_id, owner_id, kind, credits, note)
    values (p_run, r.owner_id, 'increase_requested', greatest(p_extra, 1), left(coalesce(p_reason, ''), 500));
  return jsonb_build_object('ok', true, 'needed', greatest(p_extra, 1));
end;
$$;

-- The creator approves the exact increase: it is reserved (added to the run's reservation) before anything continues.
create or replace function public.studio_run_approve_budget(p_owner uuid, p_run uuid, p_credits bigint)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  r public.studio_production_runs;
  v_hold uuid;
  v_base int;
begin
  perform pg_advisory_xact_lock(hashtext('studio_credits:' || p_owner::text));
  select * into r from public.studio_production_runs where id = p_run and owner_id = p_owner for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'not_found');
  end if;
  if r.status <> 'needs_approval' then
    return jsonb_build_object('ok', false, 'error', 'not_waiting');
  end if;
  if p_credits is distinct from r.needs_credits then
    return jsonb_build_object('ok', false, 'error', 'amount_changed', 'needed', r.needs_credits);
  end if;
  begin
    v_hold := public._studio_hold(p_owner, p_credits, null);
  exception when sqlstate 'P0001' then
    return jsonb_build_object('ok', false, 'error', 'insufficient_credits',
                              'balance', (public.studio_credit_balance(p_owner))->'available', 'needed', p_credits);
  end;
  -- Fold the new credits into the run's own reservation (one hold for the whole run).
  select coalesce(max(seq), 0) into v_base from public.studio_credit_hold_allocations where hold_id = r.hold_id;
  insert into public.studio_credit_hold_allocations (hold_id, lot_id, seq, credits)
    select r.hold_id, lot_id, seq + v_base, credits from public.studio_credit_hold_allocations where hold_id = v_hold
  on conflict (hold_id, lot_id) do update set credits = studio_credit_hold_allocations.credits + excluded.credits,
                                              seq = excluded.seq;
  delete from public.studio_credit_hold_allocations where hold_id = v_hold;
  update public.studio_credit_holds set credits_held = credits_held + p_credits where id = r.hold_id;
  update public.studio_credit_holds set credits_held = 0, status = 'settled', settled_at = now() where id = v_hold;
  update public.studio_production_runs
     set approved_credits = public.studio_run_approved(r) + p_credits, needs_credits = null, problem = null,
         status = 'running', updated_at = now()
   where id = p_run;
  insert into public.studio_run_budget_events (run_id, owner_id, kind, credits, note)
    values (p_run, p_owner, 'increase_approved', p_credits, 'You approved this increase; it was reserved.');
  return jsonb_build_object('ok', true, 'approved', public.studio_run_approved(r) + p_credits);
end;
$$;

-- ---------------------------------------------------------------- end of a run: record what came back
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
      select case when status = 'open' then credits_held else 0 end into v_back
        from public.studio_credit_holds where id = r.hold_id;
      perform public._studio_settle(r.hold_id, 0, null);
    end if;
    update public.studio_production_runs set status = p_status, problem = p_problem, needs_credits = null,
           finished_at = now(), updated_at = now()
     where id = p_run;
    if coalesce(v_back, 0) > 0 then
      insert into public.studio_run_budget_events (run_id, owner_id, kind, credits, note)
        values (p_run, r.owner_id, 'returned', v_back, 'Unused reservation returned to your balance.');
    end if;
  else
    update public.studio_production_runs set status = p_status, problem = p_problem, updated_at = now() where id = p_run;
    if p_status = 'storyboard_review' then
      insert into public.studio_run_budget_events (run_id, owner_id, kind, credits, note)
        values (p_run, r.owner_id, 'storyboard_review', 0, 'Paused before video for your storyboard review.');
    end if;
  end if;
  return jsonb_build_object('ok', true, 'returned', coalesce(v_back, 0));
end;
$$;

-- The creator approves the storyboard: video may start.
create or replace function public.studio_run_approve_storyboard(p_owner uuid, p_run uuid)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  r record;
begin
  select * into r from public.studio_production_runs where id = p_run and owner_id = p_owner for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'not_found');
  end if;
  if r.status <> 'storyboard_review' then
    return jsonb_build_object('ok', false, 'error', 'not_waiting');
  end if;
  update public.studio_production_runs set storyboard_approved_at = now(), status = 'running', problem = null,
         updated_at = now() where id = p_run;
  insert into public.studio_run_budget_events (run_id, owner_id, kind, credits, note)
    values (p_run, p_owner, 'storyboard_approved', 0, 'You approved the storyboard; video can start.');
  return jsonb_build_object('ok', true);
end;
$$;

do $$
declare f text;
begin
  foreach f in array array[
    'studio_run_approved(public.studio_production_runs)',
    'studio_run_committed(uuid)',
    'studio_run_reserve(uuid, uuid, bigint, text, jsonb, jsonb)',
    'studio_run_create_job(uuid, text, text, text, bigint, jsonb, uuid, uuid, boolean)',
    'studio_run_request_budget(uuid, bigint, text)',
    'studio_run_approve_budget(uuid, uuid, bigint)',
    'studio_run_finish(uuid, text, text)',
    'studio_run_approve_storyboard(uuid, uuid)'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
