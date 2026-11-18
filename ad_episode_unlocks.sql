-- Unlock episodes by watching ads: 2 ads (verified by Google) = 1 episode.
--
-- A viewer chooses how many episodes to unlock, watches the ads back to back,
-- and each ad Google confirms adds one "ad ticket". Two tickets unlock one
-- episode. Tickets they don't use stay for next time. Up to 15 episodes (30 ads)
-- every 5 hours: after a batch, they come back a few hours later for more. Separate from coins: the
-- coin price of episodes and creator earnings from bought coins don't change.
--
-- Needs rewarded_ads_ssv.sql (already run). Additive only, safe to run more than
-- once. Changes no existing data.

-- What an ad session is for: coins (as before) or episode tickets.
alter table public.rewarded_ad_sessions add column if not exists purpose text not null default 'coins';

-- The ticket ledger: +1 per verified ad, -2 per episode unlocked. Permanent history.
create table if not exists public.ad_episode_tickets (
  id bigserial primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  delta int not null,
  reason text not null,                 -- 'ad' or 'unlock'
  external_ref text unique,             -- admob:<transaction> / unlock:<user>:<post>: never twice
  created_at timestamptz not null default now()
);
create index if not exists ad_episode_tickets_user on public.ad_episode_tickets (user_id, created_at desc);
alter table public.ad_episode_tickets enable row level security;   -- no policies: backend only

create or replace function public.ad_episode_tickets_balance(p_user_id uuid)
returns int language sql stable security definer set search_path = public as $$
  select coalesce(sum(delta), 0)::int from public.ad_episode_tickets where user_id = p_user_id;
$$;

-- Episode ads in the last p_hours (a rolling limit, separate from coin ads): a viewer unlocks a batch,
-- and a few hours later can unlock more.
create or replace function public.ad_episode_ads_recent(p_user_id uuid, p_hours int)
returns int language sql stable security definer set search_path = public as $$
  select count(*)::int from public.ad_episode_tickets
   where user_id = p_user_id and reason = 'ad' and created_at > now() - make_interval(hours => p_hours);
$$;

-- When the next episode ad frees up (the oldest ad in the window drops out), or null if not at the limit.
create or replace function public.ad_episode_next_free_at(p_user_id uuid, p_cap int, p_hours int)
returns timestamptz language sql stable security definer set search_path = public as $$
  select case when count(*) >= p_cap then min(created_at) + make_interval(hours => p_hours) end
    from (select created_at from public.ad_episode_tickets
           where user_id = p_user_id and reason = 'ad' and created_at > now() - make_interval(hours => p_hours)
           order by created_at desc limit p_cap) w;
$$;

-- Opens an ad session for an episode ticket.
create or replace function public.rewarded_ad_start_episode(p_user_id uuid, p_cap int, p_hours int)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_today int := public.ad_episode_ads_recent(p_user_id, p_hours);
  v_open int;
  v_id uuid;
begin
  if v_today >= p_cap then
    return jsonb_build_object('ok', false, 'error', 'daily_cap', 'ads_today', v_today,
                              'next_free_at', public.ad_episode_next_free_at(p_user_id, p_cap, p_hours));
  end if;
  select count(*) into v_open from public.rewarded_ad_sessions
   where user_id = p_user_id and credited_at is null and created_at > now() - interval '1 hour';
  if v_open >= 40 then
    return jsonb_build_object('ok', false, 'error', 'too_many_sessions', 'ads_today', v_today);
  end if;
  insert into public.rewarded_ad_sessions (user_id, purpose) values (p_user_id, 'episode') returning id into v_id;
  return jsonb_build_object('ok', true, 'session', v_id, 'ads_today', v_today,
                            'tickets', public.ad_episode_tickets_balance(p_user_id));
end;
$$;

-- Google confirmed an ad: coins for a coin session (rewarded_ad_credit, unchanged), one ticket for an
-- episode session. Same checks either way: the session's own user, once per session and per AdMob
-- transaction, recent, inside that kind's limit (coins: p_daily_cap a day; episodes: p_episode_cap
-- per p_episode_hours).
create or replace function public.rewarded_ad_redeem(p_session uuid, p_user_id uuid, p_transaction_id text,
                                                     p_coins int, p_daily_cap int,
                                                     p_episode_cap int, p_episode_hours int)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  s public.rewarded_ad_sessions;
begin
  select * into s from public.rewarded_ad_sessions where id = p_session for update;
  if not found or s.purpose <> 'episode' then
    return public.rewarded_ad_credit(p_session, p_user_id, p_transaction_id, p_coins, p_daily_cap);
  end if;
  if coalesce(p_transaction_id, '') = '' then
    return jsonb_build_object('ok', false, 'error', 'bad_request');
  end if;
  if s.user_id <> p_user_id then
    return jsonb_build_object('ok', false, 'error', 'wrong_user');
  end if;
  if s.credited_at is not null then
    return jsonb_build_object('ok', false, 'error', 'already_credited');
  end if;
  if exists (select 1 from public.rewarded_ad_sessions where transaction_id = p_transaction_id) then
    return jsonb_build_object('ok', false, 'error', 'duplicate');
  end if;
  if s.created_at < now() - interval '2 hours' then
    return jsonb_build_object('ok', false, 'error', 'expired');
  end if;
  if public.ad_episode_ads_recent(p_user_id, p_episode_hours) >= p_episode_cap then
    return jsonb_build_object('ok', false, 'error', 'daily_cap');
  end if;
  begin
    insert into public.ad_episode_tickets (user_id, delta, reason, external_ref)
      values (p_user_id, 1, 'ad', 'admob:' || p_transaction_id);
  exception when unique_violation then
    return jsonb_build_object('ok', false, 'error', 'duplicate');
  end;
  update public.rewarded_ad_sessions set transaction_id = p_transaction_id, coins = 0, credited_at = now()
   where id = p_session;
  return jsonb_build_object('ok', true, 'tickets', public.ad_episode_tickets_balance(p_user_id));
end;
$$;

-- Spends p_cost tickets to unlock one episode, in one locked step. Never charges twice for an
-- episode already unlocked. The backend checks first that the episode is really a locked one.
create or replace function public.ad_unlock_episode(p_user_id uuid, p_post_id uuid, p_cost int)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_balance int;
  v_inserted int;
begin
  perform pg_advisory_xact_lock(hashtext('ad_tickets:' || p_user_id::text));
  if exists (select 1 from public.episode_unlocks where user_id = p_user_id and post_id = p_post_id) then
    return jsonb_build_object('ok', true, 'already_unlocked', true,
                              'tickets', public.ad_episode_tickets_balance(p_user_id));
  end if;
  v_balance := public.ad_episode_tickets_balance(p_user_id);
  if v_balance < p_cost then
    return jsonb_build_object('ok', false, 'error', 'not_enough_tickets', 'tickets', v_balance, 'needed', p_cost);
  end if;
  insert into public.episode_unlocks (user_id, post_id, coins_spent) values (p_user_id, p_post_id, 0)
    on conflict do nothing;
  get diagnostics v_inserted = row_count;
  if v_inserted = 0 then
    return jsonb_build_object('ok', true, 'already_unlocked', true, 'tickets', v_balance);
  end if;
  insert into public.ad_episode_tickets (user_id, delta, reason, external_ref)
    values (p_user_id, -p_cost, 'unlock', 'unlock:' || p_user_id::text || ':' || p_post_id::text);
  return jsonb_build_object('ok', true, 'already_unlocked', false, 'tickets', v_balance - p_cost);
end;
$$;

revoke all on function public.ad_episode_tickets_balance(uuid) from public, anon, authenticated;
revoke all on function public.ad_episode_ads_recent(uuid, int) from public, anon, authenticated;
revoke all on function public.ad_episode_next_free_at(uuid, int, int) from public, anon, authenticated;
revoke all on function public.rewarded_ad_start_episode(uuid, int, int) from public, anon, authenticated;
revoke all on function public.rewarded_ad_redeem(uuid, uuid, text, int, int, int, int) from public, anon, authenticated;
revoke all on function public.ad_unlock_episode(uuid, uuid, int) from public, anon, authenticated;
grant execute on function public.ad_episode_tickets_balance(uuid) to service_role;
grant execute on function public.ad_episode_ads_recent(uuid, int) to service_role;
grant execute on function public.ad_episode_next_free_at(uuid, int, int) to service_role;
grant execute on function public.rewarded_ad_start_episode(uuid, int, int) to service_role;
grant execute on function public.rewarded_ad_redeem(uuid, uuid, text, int, int, int, int) to service_role;
grant execute on function public.ad_unlock_episode(uuid, uuid, int) to service_role;

-- Check (read-only):
-- select count(*) from public.ad_episode_tickets;
