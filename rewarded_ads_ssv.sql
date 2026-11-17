-- Rewarded ads: coins only after Google confirms the ad was watched.
--
-- Before: the app told the backend "I watched an ad" and got coins. Anyone
-- could call that endpoint without watching anything (only the daily cap
-- limited it).
--
-- After: the app asks for a one-time ad session, shows the ad with that
-- session attached, and AdMob itself calls the backend (server-side
-- verification, signed by Google) when the reward is earned. The backend
-- checks Google's signature, then this file's rewarded_ad_credit() credits
-- the coins in one locked step:
--   - one credit per session, and one per AdMob transaction id (no double claims)
--   - the session must belong to that user and be recent
--   - the daily cap still applies
--   - coins go through viyo_coin_apply (coin_ledger_fix.sql), so the balance
--     and the permanent history row in `transactions` change together
--
-- Needs coin_ledger_fix.sql (already run). Additive only, safe to run more
-- than once. Changes no existing data.

create table if not exists public.rewarded_ad_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),
  transaction_id text unique,          -- AdMob's id for the watched ad, once credited
  coins int,
  credited_at timestamptz
);
create index if not exists rewarded_ad_sessions_user on public.rewarded_ad_sessions (user_id, created_at desc);
alter table public.rewarded_ad_sessions enable row level security;   -- no policies: only the backend reads/writes

-- Ad rewards credited to this user since midnight UTC (both the old and the verified path write type 'rewarded_ad').
create or replace function public.rewarded_ad_claims_today(p_user_id uuid)
returns int language sql stable security definer set search_path = public as $$
  select count(*)::int from public.transactions
   where user_id = p_user_id and type = 'rewarded_ad'
     and created_at >= date_trunc('day', now() at time zone 'utc') at time zone 'utc';
$$;

-- Opens a session before an ad is shown. Refuses at the daily cap, and when
-- a user opens too many sessions without finishing any (a misbehaving client).
create or replace function public.rewarded_ad_start(p_user_id uuid, p_daily_cap int)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_today int := public.rewarded_ad_claims_today(p_user_id);
  v_open int;
  v_id uuid;
begin
  if v_today >= p_daily_cap then
    return jsonb_build_object('ok', false, 'error', 'daily_cap', 'claims_today', v_today);
  end if;
  select count(*) into v_open from public.rewarded_ad_sessions
   where user_id = p_user_id and credited_at is null and created_at > now() - interval '1 hour';
  if v_open >= 40 then
    return jsonb_build_object('ok', false, 'error', 'too_many_sessions', 'claims_today', v_today);
  end if;
  insert into public.rewarded_ad_sessions (user_id) values (p_user_id) returning id into v_id;
  return jsonb_build_object('ok', true, 'session', v_id, 'claims_today', v_today);
end;
$$;

-- Credits a Google-verified reward. Only the backend calls this, and only
-- after checking Google's signature. Returns {ok, balance} or {ok:false, error}:
--   already_credited     - this session was credited before (Google retried): nothing changes
--   duplicate            - this AdMob transaction was already credited
--   unknown_session / wrong_user / expired / daily_cap
create or replace function public.rewarded_ad_credit(p_session uuid, p_user_id uuid, p_transaction_id text,
                                                     p_coins int, p_daily_cap int)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  s public.rewarded_ad_sessions;
  r jsonb;
begin
  if p_coins is null or p_coins <= 0 or coalesce(p_transaction_id, '') = '' then
    return jsonb_build_object('ok', false, 'error', 'bad_request');
  end if;
  select * into s from public.rewarded_ad_sessions where id = p_session for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'unknown_session');
  end if;
  if s.user_id <> p_user_id then
    return jsonb_build_object('ok', false, 'error', 'wrong_user');
  end if;
  if s.credited_at is not null then
    return jsonb_build_object('ok', false, 'error', 'already_credited', 'coins', s.coins);
  end if;
  if exists (select 1 from public.rewarded_ad_sessions where transaction_id = p_transaction_id) then
    return jsonb_build_object('ok', false, 'error', 'duplicate');
  end if;
  if s.created_at < now() - interval '2 hours' then
    return jsonb_build_object('ok', false, 'error', 'expired');
  end if;
  if public.rewarded_ad_claims_today(p_user_id) >= p_daily_cap then
    return jsonb_build_object('ok', false, 'error', 'daily_cap');
  end if;
  r := public.viyo_coin_apply(p_user_id, p_coins, 'rewarded_ad', 'Watched a rewarded ad (verified by AdMob)',
                              'admob:' || p_transaction_id);
  if not coalesce((r ->> 'ok')::boolean, false) then
    return r;
  end if;
  update public.rewarded_ad_sessions set transaction_id = p_transaction_id, coins = p_coins, credited_at = now()
   where id = p_session;
  return r;
end;
$$;

-- What the app polls after the ad closes: has Google's confirmation arrived?
create or replace function public.rewarded_ad_status(p_session uuid, p_user_id uuid)
returns jsonb language sql stable security definer set search_path = public as $$
  select coalesce((select jsonb_build_object('found', true, 'credited', credited_at is not null, 'coins', coins)
                     from public.rewarded_ad_sessions where id = p_session and user_id = p_user_id),
                  jsonb_build_object('found', false, 'credited', false));
$$;

revoke all on function public.rewarded_ad_claims_today(uuid) from public, anon, authenticated;
revoke all on function public.rewarded_ad_start(uuid, int) from public, anon, authenticated;
revoke all on function public.rewarded_ad_credit(uuid, uuid, text, int, int) from public, anon, authenticated;
revoke all on function public.rewarded_ad_status(uuid, uuid) from public, anon, authenticated;
grant execute on function public.rewarded_ad_claims_today(uuid) to service_role;
grant execute on function public.rewarded_ad_start(uuid, int) to service_role;
grant execute on function public.rewarded_ad_credit(uuid, uuid, text, int, int) to service_role;
grant execute on function public.rewarded_ad_status(uuid, uuid) to service_role;

-- Check (read-only):
-- select count(*) from public.rewarded_ad_sessions;
