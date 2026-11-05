-- Viyo Creator Studio: productions, the job queue and Studio credits.
--
-- Studio credits are a separate currency from Watch coins. Nothing here
-- reads or writes profiles.points_balance, profiles.paid_coins,
-- transactions or creator_earnings, so spending or buying credits can
-- never trigger the 65% creator payout.
--
-- 1 credit = $0.01 of value received by Viyo. Credits live in "lots":
-- one per purchase or grant, so bought and promotional credits stay apart,
-- promotional ones can expire, and a refund can target its own purchase.
-- Promotional credits are spent first (soonest expiry first), then bought
-- credits, oldest first.
--
-- Every balance change goes through a function below that locks the
-- rows it touches and carries an idempotency key, so a repeated request
-- never charges or credits twice. All functions are callable only by the
-- backend (service role).
--
-- Safe to run more than once.

-- ================================================================ credits

create table if not exists public.studio_credit_lots (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  source text not null check (source in ('purchase', 'grant')),
  purchase_id uuid,
  credits_initial bigint not null check (credits_initial > 0),
  credits_remaining bigint not null check (credits_remaining >= 0),
  expires_at timestamptz,
  note text,
  created_at timestamptz not null default now()
);
create index if not exists studio_credit_lots_user_idx on public.studio_credit_lots (user_id) where credits_remaining > 0;

create table if not exists public.studio_credit_purchases (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  payment_provider text not null,
  provider_order_id text not null,
  gross_usd numeric(10,4) not null default 0,
  fee_usd numeric(10,4) not null default 0,
  credits bigint not null,
  status text not null default 'paid' check (status in ('paid', 'refunded', 'charged_back')),
  created_at timestamptz not null default now(),
  unique (payment_provider, provider_order_id)
);

create table if not exists public.studio_credit_holds (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  credits_held bigint not null check (credits_held >= 0),
  credits_captured bigint not null default 0,
  status text not null default 'open' check (status in ('open', 'settled')),
  created_at timestamptz not null default now(),
  settled_at timestamptz
);

create table if not exists public.studio_credit_hold_allocations (
  hold_id uuid not null references public.studio_credit_holds(id) on delete cascade,
  lot_id uuid not null references public.studio_credit_lots(id) on delete cascade,
  seq int not null,
  credits bigint not null check (credits > 0),
  primary key (hold_id, lot_id)
);

create table if not exists public.studio_credit_ledger (
  id bigserial primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  kind text not null check (kind in ('purchase', 'grant', 'hold', 'capture', 'release', 'refund', 'adjust', 'chargeback', 'expire')),
  credits bigint not null,
  lot_id uuid,
  hold_id uuid,
  job_id uuid,
  idempotency_key text unique,
  note text,
  created_at timestamptz not null default now()
);
create index if not exists studio_credit_ledger_user_idx on public.studio_credit_ledger (user_id, created_at desc);

-- Available = sum of unexpired lots. Held credits are already taken out
-- of their lots, so they are never double-counted.
create or replace function public.studio_credit_balance(p_user_id uuid)
returns jsonb language sql stable security definer set search_path = public as $$
  select jsonb_build_object(
    'available', coalesce(sum(credits_remaining) filter (where expires_at is null or expires_at > now()), 0),
    'purchased', coalesce(sum(credits_remaining) filter (where source = 'purchase'), 0),
    'promotional', coalesce(sum(credits_remaining) filter (where source = 'grant' and (expires_at is null or expires_at > now())), 0),
    'held', (select coalesce(sum(credits_held - credits_captured), 0) from public.studio_credit_holds
             where user_id = p_user_id and status = 'open'))
  from public.studio_credit_lots where user_id = p_user_id;
$$;

-- Adds a lot (a purchase or a grant). Idempotent on p_key, and a purchase
-- is also unique per (provider, order id), so a webhook delivered twice
-- credits once.
create or replace function public.studio_add_credits(
  p_user_id uuid, p_credits bigint, p_source text, p_key text,
  p_note text default null, p_expires_at timestamptz default null,
  p_provider text default null, p_order_id text default null,
  p_gross_usd numeric default 0, p_fee_usd numeric default 0)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_lot uuid;
  v_purchase uuid;
begin
  if p_credits is null or p_credits <= 0 then
    return jsonb_build_object('ok', false, 'error', 'bad_amount');
  end if;
  if p_source not in ('purchase', 'grant') then
    return jsonb_build_object('ok', false, 'error', 'bad_source');
  end if;
  perform pg_advisory_xact_lock(hashtext('studio_credits:' || p_user_id::text));
  if exists (select 1 from public.studio_credit_ledger where idempotency_key = p_key) then
    return jsonb_build_object('ok', true, 'duplicate', true);
  end if;
  if p_source = 'purchase' then
    insert into public.studio_credit_purchases (user_id, payment_provider, provider_order_id, gross_usd, fee_usd, credits)
      values (p_user_id, coalesce(p_provider, 'unknown'), coalesce(p_order_id, p_key), p_gross_usd, p_fee_usd, p_credits)
      on conflict (payment_provider, provider_order_id) do nothing
      returning id into v_purchase;
    if v_purchase is null then
      return jsonb_build_object('ok', true, 'duplicate', true);
    end if;
  end if;
  insert into public.studio_credit_lots (user_id, source, purchase_id, credits_initial, credits_remaining, expires_at, note)
    values (p_user_id, p_source, v_purchase, p_credits, p_credits, p_expires_at, p_note)
    returning id into v_lot;
  insert into public.studio_credit_ledger (user_id, kind, credits, lot_id, idempotency_key, note)
    values (p_user_id, p_source, p_credits, v_lot, p_key, p_note);
  return jsonb_build_object('ok', true, 'lot_id', v_lot);
end;
$$;

-- Takes p_credits out of the user's lots into a hold. Fails (and changes
-- nothing) if there aren't enough.
create or replace function public._studio_hold(p_user_id uuid, p_credits bigint, p_job_id uuid)
returns uuid language plpgsql security definer set search_path = public as $$
declare
  v_hold uuid;
  v_need bigint := p_credits;
  v_take bigint;
  v_seq int := 0;
  r record;
begin
  insert into public.studio_credit_holds (user_id, credits_held) values (p_user_id, p_credits) returning id into v_hold;
  if p_credits > 0 then
    for r in
      select id, credits_remaining from public.studio_credit_lots
       where user_id = p_user_id and credits_remaining > 0 and (expires_at is null or expires_at > now())
       order by (source = 'purchase'), coalesce(expires_at, 'infinity'::timestamptz), created_at
       for update
    loop
      exit when v_need <= 0;
      v_take := least(v_need, r.credits_remaining);
      update public.studio_credit_lots set credits_remaining = credits_remaining - v_take where id = r.id;
      v_seq := v_seq + 1;
      insert into public.studio_credit_hold_allocations (hold_id, lot_id, seq, credits) values (v_hold, r.id, v_seq, v_take);
      v_need := v_need - v_take;
    end loop;
    if v_need > 0 then
      raise exception 'insufficient_credits' using errcode = 'P0001';
    end if;
    insert into public.studio_credit_ledger (user_id, kind, credits, hold_id, job_id)
      values (p_user_id, 'hold', -p_credits, v_hold, p_job_id);
  end if;
  return v_hold;
end;
$$;

-- Keeps p_used of a hold and returns the rest to the lots it came from,
-- last-taken first (so promotional credits stay the ones spent).
create or replace function public._studio_settle(p_hold_id uuid, p_used bigint, p_job_id uuid)
returns bigint language plpgsql security definer set search_path = public as $$
declare
  h record;
  v_used bigint;
  v_back bigint;
  v_give bigint;
  a record;
begin
  select * into h from public.studio_credit_holds where id = p_hold_id for update;
  if not found or h.status <> 'open' then
    return 0;
  end if;
  v_used := least(greatest(coalesce(p_used, 0), 0), h.credits_held);
  v_back := h.credits_held - v_used;
  for a in select * from public.studio_credit_hold_allocations where hold_id = p_hold_id order by seq desc loop
    exit when v_back <= 0;
    v_give := least(v_back, a.credits);
    update public.studio_credit_lots set credits_remaining = credits_remaining + v_give where id = a.lot_id;
    v_back := v_back - v_give;
  end loop;
  update public.studio_credit_holds set credits_captured = v_used, status = 'settled', settled_at = now() where id = p_hold_id;
  if h.credits_held - v_used > 0 then
    insert into public.studio_credit_ledger (user_id, kind, credits, hold_id, job_id)
      values (h.user_id, 'release', h.credits_held - v_used, p_hold_id, p_job_id);
  end if;
  return v_used;
end;
$$;

-- ================================================================ productions

create table if not exists public.studio_projects (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  logline text not null default '',
  kind text not null default 'episode' check (kind in ('short', 'episode', 'series', 'film')),
  aspect_ratio text not null default '9:16' check (aspect_ratio in ('9:16', '16:9')),
  genre text not null default '',
  visual_style text not null default '',
  language text not null default 'English',
  target_seconds int not null default 90 check (target_seconds between 15 and 600),
  script text not null default '',
  captions boolean not null default true,
  status text not null default 'draft' check (status in ('draft', 'archived')),
  series_id uuid,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists studio_projects_owner_idx on public.studio_projects (owner_id, updated_at desc);

create table if not exists public.studio_characters (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  description text not null default '',
  voice text not null default '',
  reference_path text,
  rights_confirmed boolean not null default false,
  position int not null default 0,
  created_at timestamptz not null default now()
);
create index if not exists studio_characters_project_idx on public.studio_characters (project_id);

create table if not exists public.studio_scenes (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  position int not null,
  heading text not null default '',
  visual text not null default '',
  dialogue jsonb not null default '[]'::jsonb,       -- [{"character": "name", "text": "..."}]
  duration_seconds numeric(5,2) not null default 6 check (duration_seconds between 2 and 20),
  mode text not null default 'still' check (mode in ('still', 'video')),
  image_path text,
  video_path text,
  audio_path text,
  audio_seconds numeric(6,2),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists studio_scenes_project_idx on public.studio_scenes (project_id, position);

-- Every generated file, kept when a scene is regenerated (older versions
-- are never overwritten).
create table if not exists public.studio_assets (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  scene_id uuid,
  character_id uuid,
  job_id uuid,
  kind text not null check (kind in ('image', 'video', 'audio', 'portrait', 'render')),
  path text not null unique,
  bytes bigint,
  duration_seconds numeric(8,2),
  width int,
  height int,
  created_at timestamptz not null default now()
);
create index if not exists studio_assets_project_idx on public.studio_assets (project_id, created_at desc);

create table if not exists public.studio_renders (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  job_id uuid,
  version int not null,
  path text,                    -- Supabase copy (only when Bunny isn't set up)
  bunny_video_id text,          -- the master lives in Bunny Stream: no file-size limit
  aspect_ratio text not null,
  width int, height int,
  duration_seconds numeric(8,2),
  bytes bigint,
  sha256 text,
  created_at timestamptz not null default now(),
  unique (project_id, version)
);

-- Older copies of this file created path as required and had no Bunny column.
alter table public.studio_renders add column if not exists bunny_video_id text;
alter table public.studio_renders alter column path drop not null;

-- ================================================================ jobs

create table if not exists public.studio_jobs (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  scene_id uuid,
  character_id uuid,
  kind text not null check (kind in ('script', 'breakdown', 'portrait', 'image', 'voice', 'video', 'render')),
  lane text not null check (lane in ('generation', 'render')),
  status text not null default 'queued' check (status in ('queued', 'running', 'succeeded', 'failed', 'canceled')),
  input jsonb not null default '{}'::jsonb,
  output jsonb,
  estimated_credits bigint not null default 0,
  charged_credits bigint not null default 0,
  hold_id uuid references public.studio_credit_holds(id),
  provider_job_id text,
  attempts int not null default 0,
  max_attempts int not null default 2,
  lease uuid,
  heartbeat_at timestamptz,
  error_code text,
  error_message text,
  idempotency_key text not null unique,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz
);
create index if not exists studio_jobs_queue_idx on public.studio_jobs (lane, status, created_at) where status in ('queued', 'running');
create index if not exists studio_jobs_project_idx on public.studio_jobs (project_id, created_at desc);

-- Real cost of every operation, with the prices used, so history survives
-- provider price changes.
create table if not exists public.studio_usage (
  id bigserial primary key,
  job_id uuid not null references public.studio_jobs(id) on delete cascade,
  owner_id uuid not null,
  project_id uuid not null,
  provider text not null,
  model text not null,
  units jsonb not null default '{}'::jsonb,
  prices jsonb not null default '{}'::jsonb,
  price_version text not null default '',
  provider_cost_usd numeric(12,6) not null default 0,
  infra_cost_usd numeric(12,6) not null default 0,
  storage_cost_usd numeric(12,6) not null default 0,
  credits_charged bigint not null default 0,
  status text not null check (status in ('charged', 'absorbed')),
  created_at timestamptz not null default now()
);

-- Holds the estimate and queues the job in one step. Idempotent on p_key:
-- a repeated tap returns the same job and charges nothing more.
create or replace function public.studio_create_job(
  p_owner uuid, p_project uuid, p_kind text, p_lane text, p_key text,
  p_credits bigint, p_input jsonb default '{}'::jsonb,
  p_scene uuid default null, p_character uuid default null, p_max_open int default 30)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_job uuid := gen_random_uuid();
  v_hold uuid;
  v_existing record;
begin
  perform pg_advisory_xact_lock(hashtext('studio_credits:' || p_owner::text));
  select id, status into v_existing from public.studio_jobs where idempotency_key = p_key;
  if found then
    return jsonb_build_object('ok', true, 'job_id', v_existing.id, 'duplicate', true);
  end if;
  if not exists (select 1 from public.studio_projects where id = p_project and owner_id = p_owner) then
    return jsonb_build_object('ok', false, 'error', 'project_not_found');
  end if;
  if (select count(*) from public.studio_jobs where owner_id = p_owner and status in ('queued', 'running')) >= p_max_open then
    return jsonb_build_object('ok', false, 'error', 'too_many_jobs');
  end if;
  begin
    v_hold := public._studio_hold(p_owner, greatest(p_credits, 0), v_job);
  exception when sqlstate 'P0001' then
    return jsonb_build_object('ok', false, 'error', 'insufficient_credits',
                              'balance', (public.studio_credit_balance(p_owner))->'available', 'needed', p_credits);
  end;
  insert into public.studio_jobs (id, owner_id, project_id, scene_id, character_id, kind, lane, input,
                                  estimated_credits, hold_id, idempotency_key)
    values (v_job, p_owner, p_project, p_scene, p_character, p_kind, p_lane, coalesce(p_input, '{}'::jsonb),
            greatest(p_credits, 0), v_hold, p_key);
  return jsonb_build_object('ok', true, 'job_id', v_job);
end;
$$;

-- Next job in a lane. A running job whose worker stopped heartbeating
-- for p_stale_seconds is taken over (its saved provider_job_id lets the
-- new worker resume instead of paying the provider twice).
create or replace function public.studio_claim_job(p_lane text, p_stale_seconds int default 180)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  j record;
  v_lease uuid := gen_random_uuid();
begin
  select * into j from public.studio_jobs
   where lane = p_lane
     and (status = 'queued' or (status = 'running' and heartbeat_at < now() - make_interval(secs => p_stale_seconds)))
   order by created_at
   for update skip locked
   limit 1;
  if not found then
    return null;
  end if;
  if j.attempts >= j.max_attempts then
    perform public._studio_settle(j.hold_id, 0, j.id);
    update public.studio_jobs set status = 'failed', error_code = 'worker_lost',
           error_message = 'Processing stopped unexpectedly. Your credits were returned.', finished_at = now(), lease = null
     where id = j.id;
    return public.studio_claim_job(p_lane, p_stale_seconds);
  end if;
  update public.studio_jobs set status = 'running', attempts = attempts + 1, lease = v_lease,
         heartbeat_at = now(), started_at = coalesce(started_at, now())
   where id = j.id;
  select * into j from public.studio_jobs where id = j.id;
  return to_jsonb(j);
end;
$$;

create or replace function public.studio_heartbeat(p_job_id uuid, p_lease uuid, p_provider_job_id text default null)
returns boolean language plpgsql security definer set search_path = public as $$
begin
  update public.studio_jobs set heartbeat_at = now(), provider_job_id = coalesce(p_provider_job_id, provider_job_id)
   where id = p_job_id and lease = p_lease and status = 'running';
  return found;
end;
$$;

-- Success: keep what it really cost (never more than the hold), give the
-- rest back, record the usage row. Returns false if the lease was lost.
create or replace function public.studio_complete_job(
  p_job_id uuid, p_lease uuid, p_output jsonb, p_credits bigint, p_usage jsonb default null)
returns boolean language plpgsql security definer set search_path = public as $$
declare
  j record;
  v_used bigint;
begin
  select * into j from public.studio_jobs where id = p_job_id for update;
  if not found or j.lease is distinct from p_lease or j.status <> 'running' then
    return false;
  end if;
  v_used := public._studio_settle(j.hold_id, p_credits, j.id);
  update public.studio_jobs set status = 'succeeded', output = p_output, charged_credits = v_used,
         finished_at = now(), lease = null, error_code = null, error_message = null
   where id = p_job_id;
  if p_usage is not null then
    insert into public.studio_usage (job_id, owner_id, project_id, provider, model, units, prices, price_version,
                                     provider_cost_usd, infra_cost_usd, storage_cost_usd, credits_charged, status)
      values (j.id, j.owner_id, j.project_id, coalesce(p_usage->>'provider', ''), coalesce(p_usage->>'model', ''),
              coalesce(p_usage->'units', '{}'::jsonb), coalesce(p_usage->'prices', '{}'::jsonb),
              coalesce(p_usage->>'price_version', ''),
              coalesce((p_usage->>'provider_cost_usd')::numeric, 0), coalesce((p_usage->>'infra_cost_usd')::numeric, 0),
              coalesce((p_usage->>'storage_cost_usd')::numeric, 0), v_used, 'charged');
  end if;
  return true;
end;
$$;

-- Failure: retry if allowed, otherwise return every held credit. A cost
-- the provider billed anyway is recorded as absorbed by Viyo.
create or replace function public.studio_fail_job(
  p_job_id uuid, p_lease uuid, p_code text, p_message text, p_retry boolean default false,
  p_absorbed_usage jsonb default null)
returns text language plpgsql security definer set search_path = public as $$
declare
  j record;
begin
  select * into j from public.studio_jobs where id = p_job_id for update;
  if not found or j.lease is distinct from p_lease or j.status <> 'running' then
    return 'lease_lost';
  end if;
  if p_absorbed_usage is not null then
    insert into public.studio_usage (job_id, owner_id, project_id, provider, model, units, prices, price_version,
                                     provider_cost_usd, credits_charged, status)
      values (j.id, j.owner_id, j.project_id, coalesce(p_absorbed_usage->>'provider', ''),
              coalesce(p_absorbed_usage->>'model', ''), coalesce(p_absorbed_usage->'units', '{}'::jsonb),
              coalesce(p_absorbed_usage->'prices', '{}'::jsonb), coalesce(p_absorbed_usage->>'price_version', ''),
              coalesce((p_absorbed_usage->>'provider_cost_usd')::numeric, 0), 0, 'absorbed');
  end if;
  if p_retry and j.attempts < j.max_attempts then
    update public.studio_jobs set status = 'queued', lease = null, error_code = p_code, error_message = p_message
     where id = p_job_id;
    return 'retrying';
  end if;
  perform public._studio_settle(j.hold_id, 0, j.id);
  update public.studio_jobs set status = 'failed', lease = null, error_code = p_code, error_message = p_message,
         finished_at = now()
   where id = p_job_id;
  return 'failed';
end;
$$;

-- The creator cancels a job that hasn't started: everything goes back.
create or replace function public.studio_cancel_job(p_owner uuid, p_job_id uuid)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  j record;
begin
  select * into j from public.studio_jobs where id = p_job_id and owner_id = p_owner for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'not_found');
  end if;
  if j.status <> 'queued' then
    return jsonb_build_object('ok', false, 'error', 'already_started');
  end if;
  perform public._studio_settle(j.hold_id, 0, j.id);
  update public.studio_jobs set status = 'canceled', finished_at = now() where id = p_job_id;
  return jsonb_build_object('ok', true);
end;
$$;

-- ================================================================ access

-- Creators can read their own rows (the app may read them directly);
-- every write goes through the backend's service role.
do $$
declare t text;
begin
  foreach t in array array['studio_projects', 'studio_characters', 'studio_scenes', 'studio_assets',
                           'studio_renders', 'studio_jobs'] loop
    execute format('alter table public.%I enable row level security', t);
    execute format('drop policy if exists "owner reads" on public.%I', t);
    execute format('create policy "owner reads" on public.%I for select using (auth.uid() = owner_id)', t);
  end loop;
  foreach t in array array['studio_credit_lots', 'studio_credit_purchases', 'studio_credit_holds',
                           'studio_credit_hold_allocations', 'studio_credit_ledger', 'studio_usage'] loop
    execute format('alter table public.%I enable row level security', t);
  end loop;
end $$;

drop policy if exists "owner reads" on public.studio_credit_ledger;
create policy "owner reads" on public.studio_credit_ledger for select using (auth.uid() = user_id);

do $$
declare f text;
begin
  foreach f in array array[
    'studio_credit_balance(uuid)',
    'studio_add_credits(uuid, bigint, text, text, text, timestamptz, text, text, numeric, numeric)',
    '_studio_hold(uuid, bigint, uuid)', '_studio_settle(uuid, bigint, uuid)',
    'studio_create_job(uuid, uuid, text, text, text, bigint, jsonb, uuid, uuid, int)',
    'studio_claim_job(text, int)', 'studio_heartbeat(uuid, uuid, text)',
    'studio_complete_job(uuid, uuid, jsonb, bigint, jsonb)',
    'studio_fail_job(uuid, uuid, text, text, boolean, jsonb)', 'studio_cancel_job(uuid, uuid)'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;

-- Private storage for everything the Studio makes. No public links: the
-- backend hands out short-lived signed links to the owner only.
insert into storage.buckets (id, name, public, file_size_limit)
values ('creator-studio', 'creator-studio', false, 2147483648)
on conflict (id) do update set public = false;
