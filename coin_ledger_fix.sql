-- Watch coins: one locked step per balance change, and each payment can
-- only ever be credited once.
--
-- Before: the backend read the balance, wrote the new balance, then wrote
-- the history row separately, and checked "already credited?" by searching
-- descriptions. Two webhook deliveries at once, or a failed history write,
-- could credit the same payment twice (and skip the paid-coin count that
-- creator earnings rely on).
--
-- After: viyo_coin_apply() locks the user's row, checks, changes the
-- balance and writes the history row in one transaction; a unique
-- external_ref (e.g. "stripe:pi_123") makes a second credit for the same
-- payment impossible.
--
-- The existing paid_coins triggers keep working unchanged (they react to
-- the same balance update and 'coin_purchase' insert), so the 65% creator
-- earnings rules are untouched. Safe to run more than once.

alter table public.transactions add column if not exists external_ref text;

-- Backfill references for purchases already made, so old payments can't
-- be credited again either. If a payment was already credited twice in
-- the past, only its first row gets the reference (the index stays valid).
with refs as (
  select id,
         case
           when description ~ '— Stripe (\S+)' then 'stripe:' || substring(description from '— Stripe (\S+)')
           when description ~ '— Paystack (\S+)' then 'paystack:' || substring(description from '— Paystack (\S+)')
           when description ~ '— Flutterwave (\S+)' then 'flutterwave:' || substring(description from '— Flutterwave (\S+)')
           when description ~ '— Lemon Squeezy order (\S+)' then 'lemonsqueezy:' || substring(description from '— Lemon Squeezy order (\S+)')
           when description ~ '— Google Play (\S+)' then 'google_play:' || substring(description from '— Google Play (\S+)')
         end as ref
    from public.transactions
   where type = 'coin_purchase' and external_ref is null
), ranked as (
  select id, ref, row_number() over (partition by ref order by id) as n
    from refs
   where ref is not null
     and not exists (select 1 from public.transactions t where t.external_ref = refs.ref)
)
update public.transactions t set external_ref = r.ref
  from ranked r
 where t.id = r.id and r.n = 1;

create unique index if not exists transactions_external_ref_key
  on public.transactions (external_ref) where external_ref is not null;

-- Changes a balance by p_amount (positive adds, negative takes) and records
-- it, atomically. Returns {ok, balance} or {ok:false, error}.
--   duplicate          - p_external_ref was already recorded: nothing changed
--   insufficient_coins - not enough coins (unless p_allow_negative)
--   profile_not_found
create or replace function public.viyo_coin_apply(
  p_user_id uuid, p_amount bigint, p_type text, p_description text,
  p_external_ref text default null, p_allow_negative boolean default false)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_balance bigint;
begin
  if p_external_ref is not null and exists (select 1 from public.transactions where external_ref = p_external_ref) then
    return jsonb_build_object('ok', false, 'error', 'duplicate');
  end if;
  select points_balance into v_balance from public.profiles where id = p_user_id for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'profile_not_found');
  end if;
  v_balance := coalesce(v_balance, 0);
  if p_amount < 0 and v_balance + p_amount < 0 and not p_allow_negative then
    return jsonb_build_object('ok', false, 'error', 'insufficient_coins', 'balance', v_balance, 'needed', -p_amount);
  end if;
  -- Balance first, then the history row: the paid_coins triggers expect
  -- that order. Both are undone together if the reference is a duplicate.
  begin
    update public.profiles set points_balance = v_balance + p_amount where id = p_user_id;
    insert into public.transactions (user_id, amount, type, description, external_ref)
      values (p_user_id, p_amount, p_type, p_description, p_external_ref);
  exception when unique_violation then
    return jsonb_build_object('ok', false, 'error', 'duplicate');
  end;
  return jsonb_build_object('ok', true, 'balance', v_balance + p_amount);
end;
$$;

revoke all on function public.viyo_coin_apply(uuid, bigint, text, text, text, boolean) from public, anon, authenticated;
grant execute on function public.viyo_coin_apply(uuid, bigint, text, text, text, boolean) to service_role;
