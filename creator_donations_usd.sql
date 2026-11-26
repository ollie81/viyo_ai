-- Donations in US dollars: the donor picks a USD amount in VIYO; VIYO converts it at the rate in use and asks
-- NardoPay for exactly that many francs. Run after creator_donations.sql. Safe to run more than once; changes no
-- existing donation.
--   usd_cents  the amount the donor chose in VIYO
--   price_rwf  what VIYO asked NardoPay for (usd_cents at the locked rate, rounded up to whole francs)
-- The creator's 80% is computed on price_rwf (the donation itself), never on a fee NardoPay may add on top.

alter table public.creator_donations add column if not exists usd_cents int;
alter table public.creator_donations add column if not exists price_rwf bigint;
do $$ begin
  alter table public.creator_donations add constraint creator_donations_price_check
    check ((usd_cents is null and price_rwf is null) or (usd_cents > 0 and price_rwf > 0)) not valid;
exception when duplicate_object then null; end $$;

create or replace function public.nardopay_record_donation(
  p_donation text, p_transaction text, p_event text, p_amount text, p_currency text, p_hold_days int,
  p_coin_usd text)
returns jsonb language plpgsql set search_path = public as $$
declare
  d public.creator_donations;
  v_amount numeric := p_amount::numeric;
  v_coin_usd numeric := p_coin_usd::numeric;
  v_base numeric;
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
  -- The donation itself (what VIYO asked for), not any fee on top; older donations without a price use the amount.
  v_base := coalesce(d.price_rwf::numeric, v_amount);
  v_share_rwf := round(v_base * d.creator_share_rate, 2);
  v_share_usd := round(v_share_rwf / d.usd_rwf_rate, 4);
  v_coins := floor(v_share_usd / v_coin_usd);
  if v_coins > 0 then
    insert into public.creator_earnings (creator_id, viewer_id, coins, kind, earning_key, status, available_at,
                                         donation_id, note)
    values (d.creator_id, d.donor_id, v_coins, 'donation', 'donation:' || d.id::text,
            case when coalesce(p_hold_days, 0) > 0 then 'pending' else 'available' end,
            now() + make_interval(days => greatest(coalesce(p_hold_days, 0), 0)), d.id,
            'Donation ' || v_base::text || ' ' || p_currency)
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

revoke all on function public.nardopay_record_donation(text, text, text, text, text, int, text)
  from public, anon, authenticated;
grant execute on function public.nardopay_record_donation(text, text, text, text, text, int, text) to service_role;
