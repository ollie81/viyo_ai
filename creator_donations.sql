-- Fan donations: real money from a viewer to a creator through NardoPay (not coin gifting). Run after
-- creator_earnings_ledger.sql and nardopay_orders.sql. Safe to run more than once. Coins, Premium and unlock earnings
-- are unchanged.
--
-- How it works
--   * A signed-in viewer starts a donation to a creator; the server creates a NardoPay donation link for THIS donation
--     (the donor chooses the amount on NardoPay's page) and locks the exchange rate in use on the row.
--   * A donation counts only when NardoPay's signed payment webhook for that link arrives (matched by link code, RWF,
--     a positive amount). nardopay_record_donation then, in ONE transaction and once per donation and per transaction:
--     marks the donation paid, records the amount, the rate, the creator's 80% share (in RWF, USD and earnings coins)
--     and adds a creator_earnings row of kind 'donation' (earning_key 'donation:<id>'), held for the holding period
--     like every new earning. It is then withdrawn through the normal withdrawal flow.
--   * A refunded donation is reversed with the existing admin reversal (viyo_reverse_earning), once.

-- ---------------------------------------------------------------- one row per donation
create table if not exists public.creator_donations (
  id uuid primary key default gen_random_uuid(),
  creator_id uuid not null references auth.users(id) on delete cascade,
  donor_id uuid not null references auth.users(id) on delete cascade,
  message text not null default '' check (char_length(message) <= 200),
  status text not null default 'pending' check (status in ('pending', 'paid', 'failed', 'expired')),
  link_code text,
  payment_link text,
  rate_id uuid,
  usd_rwf_rate numeric(12,4) not null check (usd_rwf_rate > 0),     -- locked when the donation started
  creator_share_rate numeric(5,4) not null check (creator_share_rate > 0 and creator_share_rate <= 1),
  amount_rwf numeric(14,2),                                         -- what NardoPay reported as paid
  currency text,
  share_rwf numeric(14,2),
  share_usd numeric(12,4),
  share_coins int,
  transaction_id text unique,
  event_id uuid references public.nardopay_webhook_events(id),
  earning_id uuid,
  failure_reason text not null default '' check (char_length(failure_reason) <= 400),
  created_at timestamptz not null default now(),
  paid_at timestamptz,
  check (creator_id <> donor_id),
  check (status <> 'paid' or (transaction_id is not null and amount_rwf is not null and paid_at is not null))
);
create unique index if not exists creator_donations_link_code_key
  on public.creator_donations (link_code) where link_code is not null;
create index if not exists creator_donations_creator_idx on public.creator_donations (creator_id, created_at desc);
create index if not exists creator_donations_donor_idx on public.creator_donations (donor_id, created_at desc);

alter table public.creator_donations enable row level security;
revoke all on public.creator_donations from anon, authenticated;
grant all on public.creator_donations to service_role;

-- ---------------------------------------------------------------- earnings of kind 'donation'
alter table public.creator_earnings add column if not exists donation_id uuid;
alter table public.creator_earnings drop constraint if exists creator_earnings_kind_check;
alter table public.creator_earnings add constraint creator_earnings_kind_check
  check (kind in ('unlock', 'donation', 'reversal')) not valid;

-- ---------------------------------------------------------------- record a verified donation payment (atomic, once)
-- The caller checked the signature, link code, currency and that the amount is positive.
-- Text parameters (converted inside), like the other nardopay_* functions, so callers needn't match exact types.
drop function if exists public.nardopay_record_donation(uuid, text, uuid, numeric, text, int, numeric);
drop function if exists public.nardopay_record_donation(text, text, text, text, text, int, text);
create or replace function public.nardopay_record_donation(
  p_donation text, p_transaction text, p_event text, p_amount text, p_currency text, p_hold_days int,
  p_coin_usd text)
returns jsonb language plpgsql set search_path = public as $$
declare
  d public.creator_donations;
  v_amount numeric := p_amount::numeric;
  v_coin_usd numeric := p_coin_usd::numeric;
  v_share_rwf numeric;
  v_share_usd numeric;
  v_coins int;
  v_earning uuid;
begin
  select * into d from public.creator_donations where id = p_donation::uuid for update;
  if not found then
    return jsonb_build_object('result', 'not_found');
  end if;
  if d.status = 'paid' then
    return jsonb_build_object('result', case when d.transaction_id = p_transaction then 'duplicate'
                                             else 'other_transaction' end);
  end if;
  if exists (select 1 from public.creator_donations where transaction_id = p_transaction) then
    return jsonb_build_object('result', 'transaction_used_elsewhere');
  end if;
  v_share_rwf := round(v_amount * d.creator_share_rate, 2);
  v_share_usd := round(v_share_rwf / d.usd_rwf_rate, 4);
  v_coins := floor(v_share_usd / v_coin_usd);
  if v_coins > 0 then
    insert into public.creator_earnings (creator_id, viewer_id, coins, kind, earning_key, status, available_at,
                                         donation_id, note)
    values (d.creator_id, d.donor_id, v_coins, 'donation', 'donation:' || d.id::text,
            case when coalesce(p_hold_days, 0) > 0 then 'pending' else 'available' end,
            now() + make_interval(days => greatest(coalesce(p_hold_days, 0), 0)), d.id,
            'Donation ' || v_amount::text || ' ' || p_currency)
    on conflict (earning_key) where earning_key is not null do nothing
    returning id into v_earning;
  end if;
  update public.creator_donations
     set status = 'paid', transaction_id = p_transaction, event_id = p_event::uuid, amount_rwf = v_amount,
         currency = p_currency, share_rwf = v_share_rwf, share_usd = v_share_usd, share_coins = v_coins,
         earning_id = v_earning, paid_at = now()
   where id = d.id;
  return jsonb_build_object('result', 'recorded', 'share_coins', v_coins, 'share_usd', v_share_usd,
                            'creator_id', d.creator_id, 'donor_id', d.donor_id);
end $$;

-- Reversals (refunds) now also cover donation earnings.
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
  if e.kind not in ('unlock', 'donation') then
    return jsonb_build_object('ok', false, 'error', 'not_an_earning');
  end if;
  if e.status = 'reversed' or e.reversed_at is not null then
    return jsonb_build_object('ok', true, 'changed', false, 'status', e.status);
  end if;
  if e.status = 'pending_withdrawal' then
    return jsonb_build_object('ok', false, 'error', 'in_withdrawal');
  end if;
  if e.status in ('pending', 'available') then
    update public.creator_earnings
       set status = 'reversed', reversed_at = now(), note = left(coalesce(p_note, ''), 400)
     where id = e.id;
    return jsonb_build_object('ok', true, 'changed', true, 'status', 'reversed', 'creator_id', e.creator_id,
                              'coins', e.coins);
  end if;
  insert into public.creator_earnings (creator_id, source_post_id, viewer_id, coins, kind, earning_key, status,
                                       available_at, reversal_of, note, donation_id)
  values (e.creator_id, e.source_post_id, e.viewer_id, -e.coins, 'reversal', 'reversal:' || e.id::text, 'available',
          now(), e.id, left(coalesce(p_note, ''), 400), e.donation_id)
  returning id into v_id;
  update public.creator_earnings set reversed_at = now() where id = e.id;
  return jsonb_build_object('ok', true, 'changed', true, 'status', 'deducted', 'reversal_id', v_id,
                            'creator_id', e.creator_id, 'coins', e.coins);
end $$;

do $$
declare f text;
begin
  foreach f in array array['nardopay_record_donation(text, text, text, text, text, int, text)',
                           'viyo_reverse_earning(uuid, text)'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
