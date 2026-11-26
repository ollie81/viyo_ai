-- VIYO Premium as a recurring NardoPay subscription. Run after nardopay_orders.sql, nardopay_link_code.sql and
-- subscriptions_schema.sql. Safe to run more than once. Only adds tables and functions: coin orders, the coin ledger
-- and Studio credits are untouched. Premium is real money and never touches coins.
--
-- How it works
--   * For each checkout the server creates a NardoPay subscription link (link_type "subscription") for this user and
--     saves it here with its link code. NardoPay's signed webhooks carry that link code, so a payment is matched to
--     exactly one VIYO account (never by email, never from the redirect back to VIYO).
--   * Every verified payment (signature, amount, currency) is recorded ONCE in nardopay_subscription_payments (keyed
--     by NardoPay's transaction id) and extends Premium by one billing period. Premium always has an end date: the
--     period end plus a short grace period. NardoPay documents no cancellation event, so when payments stop, Premium
--     simply ends at that date.

-- ---------------------------------------------------------------- one row per subscription checkout
create table if not exists public.nardopay_subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  plan text not null check (plan in ('weekly', 'monthly', 'yearly')),
  rwf_amount bigint not null check (rwf_amount > 0),             -- charged by NardoPay each billing cycle
  usd_cents int not null check (usd_cents > 0),                  -- the price VIYO displays
  status text not null default 'pending' check (status in ('pending', 'active', 'lapsed', 'failed')),
  link_code text,
  payment_link text,
  current_period_end timestamptz,
  last_payment_at timestamptz,
  failure_reason text not null default '' check (char_length(failure_reason) <= 400),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create unique index if not exists nardopay_subscriptions_link_code_key
  on public.nardopay_subscriptions (link_code) where link_code is not null;
create index if not exists nardopay_subscriptions_user_idx on public.nardopay_subscriptions (user_id, created_at desc);

-- ---------------------------------------------------------------- every verified payment, once (the audit trail)
create table if not exists public.nardopay_subscription_payments (
  id uuid primary key default gen_random_uuid(),
  subscription_id uuid not null references public.nardopay_subscriptions(id) on delete cascade,
  transaction_id text not null unique,
  event_id uuid references public.nardopay_webhook_events(id),
  event_type text not null default '' check (char_length(event_type) <= 80),
  amount numeric(14,2) not null,
  currency text not null,
  period_start timestamptz not null,
  period_end timestamptz not null,
  created_at timestamptz not null default now()
);
create index if not exists nardopay_subscription_payments_sub_idx
  on public.nardopay_subscription_payments (subscription_id, created_at desc);

-- ---------------------------------------------------------------- permissions
alter table public.nardopay_subscriptions enable row level security;
alter table public.nardopay_subscription_payments enable row level security;
revoke all on public.nardopay_subscriptions from anon, authenticated;
revoke all on public.nardopay_subscription_payments from anon, authenticated;
grant all on public.nardopay_subscriptions to service_role;
grant all on public.nardopay_subscription_payments to service_role;
-- A signed-in user can read their own subscriptions (never write them); payments stay server-only.
grant select on public.nardopay_subscriptions to authenticated;
drop policy if exists "owner reads own subscriptions" on public.nardopay_subscriptions;
create policy "owner reads own subscriptions" on public.nardopay_subscriptions
  for select to authenticated using (auth.uid() = user_id);

-- ---------------------------------------------------------------- recording a verified payment (atomic, once)
-- The caller has checked the signature, amount and currency. A transaction already recorded returns 'duplicate' and
-- changes nothing (NardoPay may send payment.completed and subscription.renewed for the same transaction).
create or replace function public.nardopay_record_subscription_payment(
  p_subscription text, p_transaction text, p_event text, p_event_type text, p_amount text, p_currency text,
  p_grace_hours int)
returns jsonb language plpgsql set search_path = public as $$
declare
  s public.nardopay_subscriptions;
  p public.profiles;
  starts timestamptz;
  ends timestamptz;
begin
  select * into s from public.nardopay_subscriptions where id = p_subscription::uuid for update;
  if not found then
    return jsonb_build_object('result', 'not_found');
  end if;
  if exists (select 1 from public.nardopay_subscription_payments where transaction_id = p_transaction) then
    return jsonb_build_object('result', 'duplicate');
  end if;
  select * into p from public.profiles where id = s.user_id for update;
  if not found then
    return jsonb_build_object('result', 'no_profile');
  end if;
  -- A renewal continues from the current period end; a first payment (or one after a lapse) starts now.
  starts := greatest(now(), coalesce(s.current_period_end, now()));
  ends := starts + case s.plan when 'weekly' then interval '7 days'
                               when 'monthly' then interval '1 month'
                               else interval '1 year' end;
  insert into public.nardopay_subscription_payments
    (subscription_id, transaction_id, event_id, event_type, amount, currency, period_start, period_end)
  values (s.id, p_transaction, p_event::uuid, coalesce(p_event_type, ''), p_amount::numeric, p_currency, starts, ends);
  update public.nardopay_subscriptions
     set status = 'active', current_period_end = ends, last_payment_at = now(), updated_at = now()
   where id = s.id;
  update public.profiles
     set is_subscribed = true, subscription_plan = s.plan, subscription_provider = 'nardopay',
         subscription_status = 'active', subscription_id = s.id::text, subscription_renews_at = ends,
         subscription_ends_at = ends + make_interval(hours => greatest(p_grace_hours, 0))
   where id = s.user_id;
  return jsonb_build_object('result', 'recorded', 'period_end', ends);
end $$;

-- Premium past its end date (period end + grace, i.e. no payment arrived) is switched off; the subscription shows as
-- lapsed. The paywall also checks the date itself, so access stops on time even before this runs.
create or replace function public.nardopay_end_lapsed_subscriptions()
returns jsonb language sql set search_path = public as $$
  with u as (
    update public.profiles set is_subscribed = false, subscription_status = 'expired'
     where subscription_provider = 'nardopay' and is_subscribed and subscription_ends_at < now()
    returning id),
  s as (
    update public.nardopay_subscriptions set status = 'lapsed', updated_at = now()
     where status = 'active' and current_period_end < now() - interval '7 days'
    returning id)
  select to_jsonb((select count(*) from u));
$$;

do $$
declare f text;
begin
  foreach f in array array['nardopay_record_subscription_payment(text, text, text, text, text, text, int)',
                           'nardopay_end_lapsed_subscriptions()'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
