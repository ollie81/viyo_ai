-- Creator earnings ledger: traceable, credited once, held 14 days, reversible. Run after creator_payouts_schema.sql and
-- creator_payouts_safety.sql. Safe to run more than once. Existing earnings rows are NOT changed (they keep their
-- status; they were already available). The 65% share, bought-coins-only rule, $10 minimum and the withdrawal flow are
-- unchanged.
--
-- What this adds
--   * Each earning records what it is (kind), who unlocked (viewer_id) and a unique earning_key (one per viewer per
--     episode), so a retried credit can never pay twice.
--   * New earnings are 'pending' until available_at (the holding period, 14 days by default, set by the server) and only
--     then become 'available' for withdrawal (viyo_release_earnings). This protects against refunded or fraudulent
--     coin purchases being paid out.
--   * An admin can reverse an earning (viyo_reverse_earning): pending or available -> 'reversed' (never paid); already
--     paid -> a negative 'reversal' row that is deducted from the creator's next withdrawal. Each earning is reversed
--     once at most.
--   * A credit that fails after the viewer was charged is written to creator_earning_failures, so it can be retried.

-- ---------------------------------------------------------------- columns
alter table public.creator_earnings add column if not exists kind text not null default 'unlock';
alter table public.creator_earnings add column if not exists viewer_id uuid;
alter table public.creator_earnings add column if not exists earning_key text;
alter table public.creator_earnings add column if not exists available_at timestamptz;
alter table public.creator_earnings add column if not exists reversed_at timestamptz;
alter table public.creator_earnings add column if not exists reversal_of uuid;
alter table public.creator_earnings add column if not exists note text not null default '';

do $$ begin
  alter table public.creator_earnings add constraint creator_earnings_kind_check
    check (kind in ('unlock', 'reversal')) not valid;
exception when duplicate_object then null; end $$;

-- The status rule gains 'pending' (holding period) and 'reversed'.
alter table public.creator_earnings drop constraint if exists creator_earnings_status_check;
alter table public.creator_earnings add constraint creator_earnings_status_check
  check (status in ('pending', 'available', 'pending_withdrawal', 'paid', 'reversed')) not valid;

-- Reserved or paid earnings belong to exactly one withdrawal; every other status to none.
alter table public.creator_earnings drop constraint if exists creator_earnings_withdrawal_link_check;
alter table public.creator_earnings add constraint creator_earnings_withdrawal_link_check
  check ((status in ('pending_withdrawal', 'paid')) = (withdrawal_request_id is not null)) not valid;

create unique index if not exists creator_earnings_key on public.creator_earnings (earning_key)
  where earning_key is not null;
create unique index if not exists creator_earnings_one_reversal on public.creator_earnings (reversal_of)
  where reversal_of is not null;
create index if not exists creator_earnings_pending_idx on public.creator_earnings (available_at)
  where status = 'pending';

-- ---------------------------------------------------------------- failed credits, kept for retry
create table if not exists public.creator_earning_failures (
  id uuid primary key default gen_random_uuid(),
  creator_id uuid not null,
  source_post_id uuid,
  viewer_id uuid,
  coins int not null check (coins > 0),
  earning_key text not null unique,
  error text not null default '' check (char_length(error) <= 500),
  created_at timestamptz not null default now(),
  resolved_at timestamptz
);
alter table public.creator_earning_failures enable row level security;
revoke all on public.creator_earning_failures from anon, authenticated;
grant all on public.creator_earning_failures to service_role;

-- ---------------------------------------------------------------- credit an earning (once per key)
drop function if exists public.viyo_credit_earning(uuid, uuid, uuid, int, text, int);
create or replace function public.viyo_credit_earning(
  p_creator uuid, p_post uuid, p_viewer uuid, p_coins int, p_key text, p_hold_days int)
returns jsonb language plpgsql set search_path = public as $$
declare
  v_id uuid;
begin
  if p_coins is null or p_coins <= 0 or p_key is null or p_key = '' then
    return jsonb_build_object('ok', false, 'error', 'bad_request');
  end if;
  insert into public.creator_earnings (creator_id, source_post_id, viewer_id, coins, kind, earning_key, status,
                                       available_at)
  values (p_creator, p_post, p_viewer, p_coins, 'unlock', p_key,
          case when coalesce(p_hold_days, 0) > 0 then 'pending' else 'available' end,
          now() + make_interval(days => greatest(coalesce(p_hold_days, 0), 0)))
  on conflict (earning_key) where earning_key is not null do nothing
  returning id into v_id;
  if v_id is null then
    select id into v_id from public.creator_earnings where earning_key = p_key;
    return jsonb_build_object('ok', true, 'created', false, 'id', v_id);
  end if;
  update public.creator_earning_failures set resolved_at = now() where earning_key = p_key and resolved_at is null;
  return jsonb_build_object('ok', true, 'created', true, 'id', v_id);
end $$;

-- ---------------------------------------------------------------- end of the holding period
drop function if exists public.viyo_release_earnings(uuid);
create or replace function public.viyo_release_earnings(p_creator uuid)
returns jsonb language sql set search_path = public as $$
  with u as (
    update public.creator_earnings set status = 'available'
     where status = 'pending' and available_at <= now() and (p_creator is null or creator_id = p_creator)
    returning id)
  select to_jsonb((select count(*) from u));
$$;

-- ---------------------------------------------------------------- reverse an earning (refund / fraud), once
drop function if exists public.viyo_reverse_earning(uuid, text);
create or replace function public.viyo_reverse_earning(p_earning uuid, p_note text)
returns jsonb language plpgsql set search_path = public as $$
declare
  e public.creator_earnings;
  v_id uuid;
begin
  select * into e from public.creator_earnings where id = p_earning for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'not_found');
  end if;
  if e.kind <> 'unlock' then
    return jsonb_build_object('ok', false, 'error', 'not_an_earning');
  end if;
  if e.status = 'reversed' or e.reversed_at is not null then
    return jsonb_build_object('ok', true, 'changed', false, 'status', e.status);
  end if;
  if e.status = 'pending_withdrawal' then
    -- Reject the withdrawal first (its earnings return to available), then reverse.
    return jsonb_build_object('ok', false, 'error', 'in_withdrawal');
  end if;
  if e.status in ('pending', 'available') then
    update public.creator_earnings
       set status = 'reversed', reversed_at = now(), note = left(coalesce(p_note, ''), 400)
     where id = e.id;
    return jsonb_build_object('ok', true, 'changed', true, 'status', 'reversed', 'creator_id', e.creator_id,
                              'coins', e.coins);
  end if;
  -- Already paid out: record a deduction from the creator's next withdrawal.
  insert into public.creator_earnings (creator_id, source_post_id, viewer_id, coins, kind, earning_key, status,
                                       available_at, reversal_of, note)
  values (e.creator_id, e.source_post_id, e.viewer_id, -e.coins, 'reversal', 'reversal:' || e.id::text, 'available',
          now(), e.id, left(coalesce(p_note, ''), 400))
  returning id into v_id;
  update public.creator_earnings set reversed_at = now() where id = e.id;
  return jsonb_build_object('ok', true, 'changed', true, 'status', 'deducted', 'reversal_id', v_id,
                            'creator_id', e.creator_id, 'coins', e.coins);
end $$;

do $$
declare f text;
begin
  foreach f in array array['viyo_credit_earning(uuid, uuid, uuid, int, text, int)', 'viyo_release_earnings(uuid)',
                           'viyo_reverse_earning(uuid, text)'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
