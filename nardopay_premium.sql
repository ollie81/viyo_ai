-- VIYO Premium paid through NardoPay: a prepaid pass (7, 30 or 365 days) that does NOT renew automatically.
-- Run after nardopay_orders.sql, nardopay_link_code.sql and subscriptions_schema.sql. Safe to run more than once.
-- Reuses the NardoPay order table and its verified-payment steps (signature, amount, currency, one transaction per
-- order); coin orders are unchanged. Premium is real money: a premium order never touches coins or the coin ledger.

-- ---------------------------------------------------------------- order kind
alter table public.nardopay_orders add column if not exists kind text not null default 'coins';
alter table public.nardopay_orders add column if not exists plan text;
alter table public.nardopay_orders add column if not exists pass_days int;
alter table public.nardopay_orders alter column coins drop not null;
alter table public.nardopay_orders drop constraint if exists nardopay_orders_coins_check;
do $$ begin
  alter table public.nardopay_orders add constraint nardopay_orders_kind_check check (
    (kind = 'coins' and coins is not null and coins > 0 and plan is null and pass_days is null)
    or (kind = 'premium' and coins is null and plan in ('weekly', 'monthly', 'yearly')
        and pass_days between 1 and 366));
exception when duplicate_object then null; end $$;

-- ---------------------------------------------------------------- granting the pass (once per verified payment)
-- Called after nardopay_claim_order put the order in 'crediting' for a VERIFIED payment. In one transaction it adds
-- the pass to the profile and marks the order credited, so a retry (or reconciliation) can never add it twice.
-- An active NardoPay pass is extended from its end date; otherwise the pass starts now.
create or replace function public.nardopay_grant_premium(p_order text, p_transaction text)
returns jsonb language plpgsql set search_path = public as $$
declare
  o public.nardopay_orders;
  p public.profiles;
  starts timestamptz;
  ends timestamptz;
begin
  select * into o from public.nardopay_orders
   where id = p_order::uuid and transaction_id = p_transaction for update;
  if not found or o.kind <> 'premium' then
    return jsonb_build_object('result', 'not_found');
  elsif o.status = 'credited' then
    return jsonb_build_object('result', 'already');
  elsif o.status <> 'crediting' then
    return jsonb_build_object('result', 'not_claimable', 'status', o.status);
  end if;
  select * into p from public.profiles where id = o.user_id for update;
  if not found then
    return jsonb_build_object('result', 'no_profile');
  end if;
  starts := case when p.is_subscribed and p.subscription_provider = 'nardopay' and p.subscription_ends_at > now()
                 then p.subscription_ends_at else now() end;
  ends := starts + make_interval(days => o.pass_days);
  update public.profiles
     set is_subscribed = true, subscription_plan = o.plan, subscription_provider = 'nardopay',
         subscription_status = 'active', subscription_id = o.id::text, subscription_renews_at = null,
         subscription_ends_at = ends
   where id = o.user_id;
  update public.nardopay_orders
     set status = 'credited', credited_at = coalesce(credited_at, now()), updated_at = now()
   where id = o.id;
  return jsonb_build_object('result', 'granted', 'ends_at', ends);
end $$;

-- Passes past their end date are switched off (the app and the episode paywall also check the date themselves).
create or replace function public.nardopay_end_expired_passes()
returns jsonb language sql set search_path = public as $$
  with u as (
    update public.profiles set is_subscribed = false, subscription_status = 'expired'
     where subscription_provider = 'nardopay' and is_subscribed and subscription_ends_at < now()
    returning id)
  select to_jsonb((select count(*) from u));
$$;

-- Reconciliation now also reports the order kind, so a premium order is finished by granting the pass.
create or replace function public.nardopay_reconcile_candidates(p_min_age_seconds int)
returns jsonb language sql set search_path = public as $$
  select coalesce(jsonb_agg(jsonb_build_object(
           'order_id', o.id, 'user_id', o.user_id, 'coins', o.coins, 'package_id', o.package_id, 'kind', o.kind,
           'rwf_amount', o.rwf_amount, 'transaction_id', o.transaction_id, 'event_id', e.id,
           'signature_ok', e.signature_ok, 'event_amount', e.amount, 'event_currency', e.currency,
           'event_transaction_id', e.transaction_id)), '[]'::jsonb)
    from public.nardopay_orders o
    join public.nardopay_webhook_events e on e.id = o.verified_event_id
   where o.status = 'crediting' and o.updated_at < now() - make_interval(secs => p_min_age_seconds);
$$;

do $$
declare f text;
begin
  foreach f in array array['nardopay_grant_premium(text, text)', 'nardopay_end_expired_passes()',
                           'nardopay_reconcile_candidates(int)'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
