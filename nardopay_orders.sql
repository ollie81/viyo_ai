-- NardoPay coin checkout (web only): exchange rates, server-side orders and the webhook event log.
-- Plan: VIYO_NardoPay_Integration_Plan_v3. Safe to run more than once. Only adds new tables; no existing table,
-- function or payment provider (Stripe, Paystack, Flutterwave, Lemon Squeezy, Google Play) is changed.
--
-- Who can do what:
--   the backend (service role)  reads and writes everything;
--   a signed-in user            reads ONLY their own orders, and writes nothing;
--   everyone else               nothing. Rates and webhook events (which may hold customer details) are server-only.

-- ---------------------------------------------------------------- exchange rates (RWF per USD), append-only history
create table if not exists public.nardopay_rates (
  id uuid primary key default gen_random_uuid(),
  rate numeric(12,4) not null check (rate > 0),
  set_by text not null check (char_length(set_by) between 1 and 120),
  note text not null default '' check (char_length(note) <= 400),
  set_at timestamptz not null default now()
);
create index if not exists nardopay_rates_set_at_idx on public.nardopay_rates (set_at desc);

-- ---------------------------------------------------------------- the webhook event log
-- One row per distinct signed delivery (body_sha256 is unique, so a repeated delivery finds the existing row).
-- status: received -> processing -> processed | ignored | mismatch | duplicate_payment | error.
-- raw_body is kept only for a limited time (it may contain customer details) and then cleared; the hash and the
-- few fields needed for audit stay.
create table if not exists public.nardopay_webhook_events (
  id uuid primary key default gen_random_uuid(),
  body_sha256 text not null unique check (body_sha256 ~ '^[0-9a-f]{64}$'),
  provider_event_id text,
  event_type text not null default '' check (char_length(event_type) <= 80),
  order_id uuid,
  transaction_id text,
  amount numeric(14,2),
  currency text,
  signature_ok boolean not null,
  status text not null default 'received'
    check (status in ('received', 'processing', 'processed', 'ignored', 'mismatch', 'duplicate_payment', 'error')),
  detail text not null default '' check (char_length(detail) <= 1000),
  attempts int not null default 0 check (attempts >= 0),
  locked_at timestamptz,
  raw_body text,
  raw_purged_at timestamptz,
  received_at timestamptz not null default now(),
  processed_at timestamptz,
  check (raw_purged_at is null or raw_body is null)
);
create unique index if not exists nardopay_webhook_events_provider_id_key
  on public.nardopay_webhook_events (provider_event_id) where provider_event_id is not null;
create index if not exists nardopay_webhook_events_status_idx on public.nardopay_webhook_events (status, received_at);

-- ---------------------------------------------------------------- orders, created by the server before checkout
-- The price and coins come from the server's package list; the rate used is locked on the order.
-- status: pending -> crediting -> credited, or failed / mismatch / expired.
-- An order can only be crediting or credited after a VERIFIED payment (signature checked, amount and currency
-- matched): the transaction, the verified event and the time are then required, so nothing can credit an
-- unverified order, reconciliation included.
create table if not exists public.nardopay_orders (
  id uuid primary key default gen_random_uuid(),                  -- sent to NardoPay as metadata.order_id
  user_id uuid not null references auth.users(id) on delete cascade,
  package_id text not null check (char_length(package_id) between 1 and 40),
  coins int not null check (coins > 0),
  usd_cents int not null check (usd_cents > 0),
  rate_id uuid references public.nardopay_rates(id),
  usd_rwf_rate numeric(12,4) not null check (usd_rwf_rate > 0),
  rwf_amount bigint not null check (rwf_amount > 0),
  currency text not null default 'RWF' check (currency = 'RWF'),
  status text not null default 'pending'
    check (status in ('pending', 'crediting', 'credited', 'failed', 'mismatch', 'expired')),
  payment_link text,
  transaction_id text unique,
  verified_event_id uuid references public.nardopay_webhook_events(id),
  payment_verified_at timestamptz,
  failure_reason text not null default '' check (char_length(failure_reason) <= 400),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  expires_at timestamptz not null default (now() + interval '24 hours'),
  credited_at timestamptz,
  check (status not in ('crediting', 'credited')
         or (transaction_id is not null and verified_event_id is not null and payment_verified_at is not null)),
  check (status <> 'credited' or credited_at is not null)
);
create index if not exists nardopay_orders_user_idx on public.nardopay_orders (user_id, created_at desc);
create index if not exists nardopay_orders_crediting_idx on public.nardopay_orders (status, updated_at)
  where status = 'crediting';

-- ---------------------------------------------------------------- permissions
alter table public.nardopay_rates enable row level security;
alter table public.nardopay_webhook_events enable row level security;
alter table public.nardopay_orders enable row level security;

revoke all on public.nardopay_rates from anon, authenticated;
revoke all on public.nardopay_webhook_events from anon, authenticated;
revoke all on public.nardopay_orders from anon, authenticated;

grant all on public.nardopay_rates to service_role;
grant all on public.nardopay_webhook_events to service_role;
grant all on public.nardopay_orders to service_role;

-- A signed-in user can read their own orders and nothing else. There are no insert/update/delete policies, and
-- no write privileges, so clients can't change orders at all.
grant select on public.nardopay_orders to authenticated;
drop policy if exists "owner reads own orders" on public.nardopay_orders;
create policy "owner reads own orders" on public.nardopay_orders
  for select to authenticated using (auth.uid() = user_id);

-- ---------------------------------------------------------------- atomic steps used by nardopay_payments.py
drop function if exists public.nardopay_record_event(text, boolean, text, uuid, text, numeric, text, text, text);
drop function if exists public.nardopay_claim_event(uuid);
drop function if exists public.nardopay_finish_event(uuid, text, text);
drop function if exists public.nardopay_claim_order(uuid, text, uuid);
drop function if exists public.nardopay_finish_order(uuid, text);
-- Each returns jsonb (one value), and only the backend's service role may call them.

-- Records a signed delivery, or finds the one already recorded for the same body (or the same provider event id):
-- a repeated delivery never creates a second row.
create or replace function public.nardopay_record_event(
  p_hash text, p_signature_ok boolean, p_event_type text, p_order_id text, p_transaction_id text,
  p_amount text, p_currency text, p_provider_event_id text, p_raw text)
returns jsonb language plpgsql set search_path = public as $$
declare
  r public.nardopay_webhook_events;
begin
  insert into public.nardopay_webhook_events (body_sha256, signature_ok, event_type, order_id, transaction_id, amount,
                                              currency, provider_event_id, raw_body)
  values (p_hash, p_signature_ok, coalesce(p_event_type, ''), p_order_id::uuid, p_transaction_id, p_amount::numeric,
          p_currency, p_provider_event_id, p_raw)
  on conflict do nothing
  returning * into r;
  if found then
    return jsonb_build_object('new', true, 'event', to_jsonb(r));
  end if;
  select * into r from public.nardopay_webhook_events where body_sha256 = p_hash;
  if not found and p_provider_event_id is not null then
    select * into r from public.nardopay_webhook_events where provider_event_id = p_provider_event_id;
  end if;
  -- An event stored before its fields could be read (settings not filled in yet) gets them now, so later checks
  -- (reconciliation included) see the real amount and currency. A finished event is never changed.
  if found and r.status in ('received', 'error') and p_event_type is not null then
    update public.nardopay_webhook_events
       set event_type = coalesce(nullif(event_type, ''), p_event_type),
           order_id = coalesce(order_id, p_order_id::uuid),
           transaction_id = coalesce(transaction_id, p_transaction_id),
           amount = coalesce(amount, p_amount::numeric),
           currency = coalesce(currency, p_currency)
     where id = r.id
    returning * into r;
  end if;
  return jsonb_build_object('new', false, 'event', to_jsonb(r));
end $$;

-- Claims an event for processing. Of two simultaneous deliveries only one gets true; a claim older than five
-- minutes (a crashed worker) can be taken over.
create or replace function public.nardopay_claim_event(p_event text)
returns jsonb language sql set search_path = public as $$
  with c as (
    update public.nardopay_webhook_events
       set status = 'processing', locked_at = now(), attempts = attempts + 1
     where id = p_event::uuid
       and (status in ('received', 'error') or (status = 'processing' and locked_at < now() - interval '5 minutes'))
    returning id)
  select to_jsonb(exists (select 1 from c));
$$;

create or replace function public.nardopay_finish_event(p_event text, p_status text, p_detail text)
returns jsonb language sql set search_path = public as $$
  update public.nardopay_webhook_events
     set status = p_status, detail = left(coalesce(p_detail, ''), 1000), locked_at = null,
         processed_at = case when p_status in ('processed', 'ignored', 'mismatch', 'duplicate_payment') then now()
                             else processed_at end
   where id = p_event::uuid;
  select 'true'::jsonb;
$$;

-- Claims an order for a VERIFIED payment (the caller checked signature, amount and currency first). An order is
-- claimed once: a retry of the same transaction is told so; another transaction never gets it.
create or replace function public.nardopay_claim_order(p_order text, p_transaction text, p_event text)
returns jsonb language plpgsql set search_path = public as $$
declare
  o public.nardopay_orders;
begin
  begin
    update public.nardopay_orders
       set status = 'crediting', transaction_id = p_transaction, verified_event_id = p_event::uuid,
           payment_verified_at = now(), updated_at = now()
     where id = p_order::uuid and status in ('pending', 'expired', 'failed') and transaction_id is null
    returning * into o;
  exception when unique_violation then
    return jsonb_build_object('result', 'transaction_used_elsewhere');
  end;
  if found then
    return jsonb_build_object('result', 'claimed', 'order', to_jsonb(o));
  end if;
  select * into o from public.nardopay_orders where id = p_order::uuid;
  if not found then
    return jsonb_build_object('result', 'not_found');
  elsif o.transaction_id = p_transaction then
    return jsonb_build_object('result', 'retry', 'order', to_jsonb(o));
  elsif o.transaction_id is not null then
    return jsonb_build_object('result', 'other_transaction', 'order', to_jsonb(o));
  end if;
  return jsonb_build_object('result', 'not_claimable', 'order', to_jsonb(o));
end $$;

create or replace function public.nardopay_finish_order(p_order text, p_transaction text)
returns jsonb language sql set search_path = public as $$
  with u as (
    update public.nardopay_orders
       set status = 'credited', credited_at = coalesce(credited_at, now()), updated_at = now()
     where id = p_order::uuid and transaction_id = p_transaction and status in ('crediting', 'credited')
    returning id)
  select to_jsonb(exists (select 1 from u));
$$;

-- Payments that were verified and claimed but not finished (a crash between crediting and the status update), older
-- than p_min_age_seconds. Only orders in 'crediting' qualify, and the database guarantees those have a verified
-- event; the event's own signature result, amount and currency are returned so they can be checked again.
create or replace function public.nardopay_reconcile_candidates(p_min_age_seconds int)
returns jsonb language sql set search_path = public as $$
  select coalesce(jsonb_agg(jsonb_build_object(
           'order_id', o.id, 'user_id', o.user_id, 'coins', o.coins, 'package_id', o.package_id,
           'rwf_amount', o.rwf_amount, 'transaction_id', o.transaction_id, 'event_id', e.id,
           'signature_ok', e.signature_ok, 'event_amount', e.amount, 'event_currency', e.currency,
           'event_transaction_id', e.transaction_id)), '[]'::jsonb)
    from public.nardopay_orders o
    join public.nardopay_webhook_events e on e.id = o.verified_event_id
   where o.status = 'crediting' and o.updated_at < now() - make_interval(secs => p_min_age_seconds);
$$;

-- Clears raw webhook bodies after the retention period (they may contain customer details).
create or replace function public.nardopay_purge_raw(p_days int, p_problem_days int)
returns jsonb language sql set search_path = public as $$
  with u as (
    update public.nardopay_webhook_events
       set raw_body = null, raw_purged_at = now()
     where raw_body is not null
       and ((status in ('processed', 'ignored') and received_at < now() - make_interval(days => p_days))
         or (status not in ('processed', 'ignored') and received_at < now() - make_interval(days => p_problem_days)))
    returning id)
  select to_jsonb((select count(*) from u));
$$;

-- Unpaid orders past their time show as expired (a correct payment that arrives later is still honoured).
create or replace function public.nardopay_expire_orders()
returns jsonb language sql set search_path = public as $$
  with u as (
    update public.nardopay_orders set status = 'expired', updated_at = now()
     where status = 'pending' and expires_at < now()
    returning id)
  select to_jsonb((select count(*) from u));
$$;

do $$
declare f text;
begin
  foreach f in array array[
    'nardopay_record_event(text, boolean, text, text, text, text, text, text, text)',
    'nardopay_claim_event(text)', 'nardopay_finish_event(text, text, text)',
    'nardopay_claim_order(text, text, text)', 'nardopay_finish_order(text, text)',
    'nardopay_reconcile_candidates(int)', 'nardopay_purge_raw(int, int)', 'nardopay_expire_orders()'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
