-- Creator payouts: safe for real money. Run after creator_payouts_schema.sql (it needs the columns that file adds).
-- Safe to run more than once. Changes nothing about how earnings are calculated (65% of bought coins), the $10
-- minimum, or PayPal / mobile money: those stay in wallet.py and episodes.py.
--
-- What this fixes:
--   * requesting a withdrawal is ONE locked transaction (viyo_request_withdrawal): only one request per creator at a
--     time, the earnings it reserves are locked, and if anything fails nothing is reserved and no request exists;
--   * paying or rejecting is ONE locked transaction (viyo_settle_withdrawal): a request changes status once; repeating
--     the same action changes nothing; paying after a rejection (or the reverse) is refused; rejected earnings go back
--     to available exactly once; a request whose reserved earnings don't add up is refused, not paid.

-- ---------------------------------------------------------------- 0) stop if existing data would break the rules
do $$
begin
  if exists (select 1 from public.withdrawal_requests where status = 'pending'
             group by creator_id having count(*) > 1) then
    raise exception 'Some creators have more than one pending withdrawal. Resolve them (mark paid or rejected) '
                    'before running this file; see the check query in the PR description.';
  end if;
end $$;

-- ---------------------------------------------------------------- 1) rules the database enforces from now on
-- NOT VALID: checked for every new or changed row, without failing on old rows that may predate these rules.
do $$ begin
  alter table public.withdrawal_requests add constraint withdrawal_requests_status_check
    check (status in ('pending', 'paid', 'rejected')) not valid;
exception when duplicate_object then null; end $$;

do $$ begin
  alter table public.withdrawal_requests add constraint withdrawal_requests_coins_check
    check (coins > 0) not valid;
exception when duplicate_object then null; end $$;

do $$ begin
  alter table public.creator_earnings add constraint creator_earnings_status_check
    check (status in ('available', 'pending_withdrawal', 'paid')) not valid;
exception when duplicate_object then null; end $$;

-- Available earnings belong to no withdrawal; reserved or paid earnings belong to exactly one.
do $$ begin
  alter table public.creator_earnings add constraint creator_earnings_withdrawal_link_check
    check ((status = 'available') = (withdrawal_request_id is null)) not valid;
exception when duplicate_object then null; end $$;

do $$ begin
  alter table public.creator_earnings add constraint creator_earnings_withdrawal_fk
    foreign key (withdrawal_request_id) references public.withdrawal_requests(id) not valid;
exception when duplicate_object then null; end $$;

-- One pending withdrawal per creator, guaranteed by the database (not only by the code).
create unique index if not exists withdrawal_requests_one_pending_key
  on public.withdrawal_requests (creator_id) where status = 'pending';

alter table public.withdrawal_requests add column if not exists rejected_at timestamptz;
create index if not exists creator_earnings_creator_status_idx on public.creator_earnings (creator_id, status);

-- ---------------------------------------------------------------- 2) request a withdrawal, atomically
drop function if exists public.viyo_request_withdrawal(uuid, int, numeric, text, text);
-- The minimum and the coin value come from wallet.py (the single source of those rules). Returns
-- {ok, request_id, coins, usd_amount} or {ok:false, error: pending_exists | below_minimum, coins}.
create or replace function public.viyo_request_withdrawal(
  p_creator uuid, p_min_coins int, p_coin_usd double precision, p_method text, p_destination text)
returns jsonb language plpgsql set search_path = public as $$
declare
  v_ids uuid[];
  v_total bigint;
  v_request uuid;
  v_usd numeric;
  v_marked int;
begin
  -- One request at a time per creator: a second simultaneous request waits here, then sees the first one.
  perform pg_advisory_xact_lock(hashtextextended('viyo_withdrawal:' || p_creator::text, 0));
  if exists (select 1 from public.withdrawal_requests where creator_id = p_creator and status = 'pending') then
    return jsonb_build_object('ok', false, 'error', 'pending_exists');
  end if;
  -- Lock exactly the earnings being reserved. Earnings credited after this point stay available for next time.
  select coalesce(array_agg(id), '{}'), coalesce(sum(coins), 0) into v_ids, v_total
    from (select id, coins from public.creator_earnings
           where creator_id = p_creator and status = 'available' and withdrawal_request_id is null
           for update) e;
  if v_total < p_min_coins then
    return jsonb_build_object('ok', false, 'error', 'below_minimum', 'coins', v_total);
  end if;
  v_usd := round(v_total * p_coin_usd::numeric, 2);
  insert into public.withdrawal_requests (creator_id, coins, status, usd_amount, payout_method, payout_destination)
  values (p_creator, v_total, 'pending', v_usd, p_method, p_destination)
  returning id into v_request;
  update public.creator_earnings
     set status = 'pending_withdrawal', withdrawal_request_id = v_request
   where id = any(v_ids) and status = 'available';
  get diagnostics v_marked = row_count;
  if v_marked <> coalesce(array_length(v_ids, 1), 0) then
    -- Can't happen while the rows are locked; if it ever does, undo everything rather than create a request that
    -- isn't fully backed.
    raise exception 'withdrawal % reserved % of % earnings rows', v_request, v_marked, array_length(v_ids, 1);
  end if;
  return jsonb_build_object('ok', true, 'request_id', v_request, 'coins', v_total, 'usd_amount', v_usd);
end $$;

-- ---------------------------------------------------------------- 3) mark paid / reject, atomically and idempotently
-- p_action: 'paid' or 'rejected'. Returns {ok, changed, status, creator_id, coins, usd_amount, payout_destination}
-- (changed=false when the same action was already done: nothing is touched), or {ok:false, error: not_found |
-- already_paid | already_rejected | earnings_mismatch}.
create or replace function public.viyo_settle_withdrawal(p_request uuid, p_action text, p_note text)
returns jsonb language plpgsql set search_path = public as $$
declare
  r public.withdrawal_requests;
  v_rows int;
  v_coins bigint;
  v_moved int;
begin
  if p_action not in ('paid', 'rejected') then
    raise exception 'unknown action %', p_action;
  end if;
  select * into r from public.withdrawal_requests where id = p_request for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'not_found');
  end if;
  if r.status = p_action then          -- the same action again (a double tap, a retry): nothing changes
    return jsonb_build_object('ok', true, 'changed', false, 'status', r.status, 'creator_id', r.creator_id,
                              'coins', r.coins, 'usd_amount', r.usd_amount, 'payout_destination', r.payout_destination);
  end if;
  if r.status <> 'pending' then        -- paying a rejected request, or rejecting a paid one: refused
    return jsonb_build_object('ok', false, 'error', 'already_' || r.status);
  end if;
  -- The earnings this request reserved must add up to what it asks for; otherwise a person has to look.
  select count(*), coalesce(sum(coins), 0) into v_rows, v_coins
    from (select id, coins from public.creator_earnings
           where withdrawal_request_id = p_request and status = 'pending_withdrawal'
           for update) e;
  if v_coins <> r.coins then
    return jsonb_build_object('ok', false, 'error', 'earnings_mismatch', 'request_coins', r.coins,
                              'reserved_coins', v_coins);
  end if;
  if p_action = 'paid' then
    update public.withdrawal_requests
       set status = 'paid', paid_at = now(), admin_note = p_note
     where id = p_request and status = 'pending';
    update public.creator_earnings set status = 'paid'
     where withdrawal_request_id = p_request and status = 'pending_withdrawal';
  else
    update public.withdrawal_requests
       set status = 'rejected', rejected_at = now(), admin_note = p_note
     where id = p_request and status = 'pending';
    update public.creator_earnings set status = 'available', withdrawal_request_id = null
     where withdrawal_request_id = p_request and status = 'pending_withdrawal';
  end if;
  get diagnostics v_moved = row_count;
  if v_moved <> v_rows then
    raise exception 'withdrawal % moved % of % earnings rows', p_request, v_moved, v_rows;
  end if;
  return jsonb_build_object('ok', true, 'changed', true, 'status', p_action, 'creator_id', r.creator_id,
                            'coins', r.coins, 'usd_amount', r.usd_amount, 'payout_destination', r.payout_destination);
end $$;

revoke all on function public.viyo_request_withdrawal(uuid, int, double precision, text, text) from public, anon, authenticated;
revoke all on function public.viyo_settle_withdrawal(uuid, text, text) from public, anon, authenticated;
grant execute on function public.viyo_request_withdrawal(uuid, int, double precision, text, text) to service_role;
grant execute on function public.viyo_settle_withdrawal(uuid, text, text) to service_role;
