-- Ad revenue sharing: creators get half of what VIYO's ads earn on their content. Run after
-- creator_earnings_ledger.sql, creator_donations.sql, brand_campaigns.sql and ad_episode_unlocks.sql.
-- Safe to run more than once. Changes no existing data.
--
-- Rules (agreed with the owner)
--   * Each month an admin enters the real totals from the AdMob report: what the in-feed ads (interstitials) earned
--     and what the episode-unlock ads (rewarded) earned. Nothing is estimated.
--   * 50% of each total goes to creators, split by how many of those ads ran on each creator's content that month:
--       - unlock ads: 2 Google-verified ads per episode unlocked with ads (ad_episode_tickets, reason 'unlock');
--       - in-feed ads: shown next to a post (any content: dramas, videos, long videos...), reported by the app,
--         counted once per viewer per post per day, never on the creator's own posts, at most 40 per viewer a day.
--     Because the money is a share of the real total, extra counts can only move money between creators; VIYO never
--     pays out more than half of what the ads actually earned.
--   * Ads watched for coins aren't tied to any content and aren't shared.
--   * Closing a month credits each creator once (creator_earnings kind 'ads', key 'ads:<month>:<creator>'), held like
--     every new earning; a month is closed once.

-- ---------------------------------------------------------------- in-feed ad impressions next to a post
create table if not exists public.ad_impressions (
  id bigserial primary key,
  viewer_id uuid not null references auth.users(id) on delete cascade,
  creator_id uuid not null references auth.users(id) on delete cascade,
  post_id uuid not null,
  unit text not null default 'interstitial' check (unit in ('interstitial')),
  day date not null default (now() at time zone 'utc')::date,
  created_at timestamptz not null default now(),
  unique (viewer_id, post_id, unit, day),
  check (viewer_id <> creator_id)
);
create index if not exists ad_impressions_creator_day on public.ad_impressions (creator_id, day);
create index if not exists ad_impressions_day on public.ad_impressions (day);
create index if not exists ad_impressions_viewer_day on public.ad_impressions (viewer_id, day);

-- ---------------------------------------------------------------- months and what each creator got
create table if not exists public.ad_revenue_months (
  month date primary key check (extract(day from month) = 1),
  interstitial_usd numeric(12,2) not null check (interstitial_usd >= 0),
  unlock_usd numeric(12,2) not null check (unlock_usd >= 0),
  creator_share numeric(5,4) not null check (creator_share > 0 and creator_share <= 1),
  interstitial_count bigint not null default 0,
  unlock_count bigint not null default 0,
  paid_usd numeric(12,4) not null default 0,
  creators int not null default 0,
  note text not null default '' check (char_length(note) <= 300),
  closed_at timestamptz not null default now()
);

create table if not exists public.ad_revenue_payouts (
  month date not null references public.ad_revenue_months(month) on delete cascade,
  creator_id uuid not null references auth.users(id) on delete cascade,
  interstitial_count bigint not null default 0,
  unlock_count bigint not null default 0,
  usd numeric(12,4) not null default 0,
  coins int not null default 0,
  earning_id uuid,
  primary key (month, creator_id)
);
create index if not exists ad_revenue_payouts_creator on public.ad_revenue_payouts (creator_id, month desc);

do $$ declare t text;
begin
  foreach t in array array['ad_impressions', 'ad_revenue_months', 'ad_revenue_payouts'] loop
    execute format('alter table public.%I enable row level security', t);
    execute format('revoke all on public.%I from anon, authenticated', t);
    execute format('grant all on public.%I to service_role', t);
  end loop;
end $$;

-- ---------------------------------------------------------------- earnings of kind 'ads'
alter table public.creator_earnings drop constraint if exists creator_earnings_kind_check;
alter table public.creator_earnings add constraint creator_earnings_kind_check
  check (kind in ('unlock', 'donation', 'brand', 'ads', 'reversal')) not valid;

-- ---------------------------------------------------------------- record one in-feed ad (from the app)
drop function if exists public.ad_impression_record(uuid, uuid, int);
create or replace function public.ad_impression_record(p_viewer uuid, p_post uuid, p_daily_cap int)
returns jsonb language plpgsql set search_path = public as $$
declare
  v_creator uuid;
  v_today date := (now() at time zone 'utc')::date;
  v_n int;
begin
  select user_id into v_creator from public.posts where id = p_post;
  if v_creator is null then return jsonb_build_object('counted', false, 'reason', 'no_post'); end if;
  if v_creator = p_viewer then return jsonb_build_object('counted', false, 'reason', 'own_post'); end if;
  perform pg_advisory_xact_lock(hashtext('ad_impr:' || p_viewer::text));
  select count(*) into v_n from public.ad_impressions where viewer_id = p_viewer and day = v_today;
  if v_n >= coalesce(p_daily_cap, 40) then return jsonb_build_object('counted', false, 'reason', 'daily_cap'); end if;
  insert into public.ad_impressions (viewer_id, creator_id, post_id, unit, day)
  values (p_viewer, v_creator, p_post, 'interstitial', v_today)
  on conflict (viewer_id, post_id, unit, day) do nothing;
  return jsonb_build_object('counted', found);
end $$;

-- ---------------------------------------------------------------- each creator's ads in a month (exact counts)
drop function if exists public.ad_revenue_counts(date, uuid);
create or replace function public.ad_revenue_counts(p_month date, p_creator uuid default null)
returns jsonb language plpgsql stable set search_path = public as $$
declare
  v_from date := date_trunc('month', p_month)::date;
  v_to date := (date_trunc('month', p_month) + interval '1 month')::date;
  out jsonb;
begin
  with feed as (
    select creator_id, count(*) as n from public.ad_impressions
     where day >= v_from and day < v_to and (p_creator is null or creator_id = p_creator)
     group by creator_id),
  unl as (
    select p.user_id as creator_id, sum(-t.delta) as n      -- 2 verified ads per episode
      from public.ad_episode_tickets t
      join public.posts p on p.id = nullif(split_part(t.external_ref, ':', 3), '')::uuid
     where t.reason = 'unlock' and t.external_ref like 'unlock:%'
       and t.created_at >= v_from and t.created_at < v_to
       and p.user_id <> nullif(split_part(t.external_ref, ':', 2), '')::uuid
       and (p_creator is null or p.user_id = p_creator)
     group by p.user_id),
  allc as (
    select coalesce(f.creator_id, u.creator_id) as creator_id, coalesce(f.n, 0) as feed, coalesce(u.n, 0) as unlock
      from feed f full outer join unl u on u.creator_id = f.creator_id)
  select jsonb_build_object(
           'month', v_from,
           'interstitial_total', coalesce(sum(feed), 0),
           'unlock_total', coalesce(sum(unlock), 0),
           'creators', coalesce(jsonb_agg(jsonb_build_object('creator_id', creator_id, 'interstitial', feed,
                                                             'unlock', unlock)) filter (where creator_id is not null),
                                '[]'::jsonb))
    into out from allc;
  return out;
end $$;

-- ---------------------------------------------------------------- close a month: credit each creator once
-- p_holds: {"<creator id>": hold days} (each creator's level); a creator not listed gets p_default_hold.
drop function if exists public.ad_revenue_close(date, text, text, text, text, int, jsonb, text);
create or replace function public.ad_revenue_close(
  p_month date, p_interstitial_usd text, p_unlock_usd text, p_share text, p_coin_usd text, p_default_hold int,
  p_holds jsonb, p_note text)
returns jsonb language plpgsql set search_path = public as $$
declare
  v_month date := date_trunc('month', p_month)::date;
  v_feed_usd numeric := p_interstitial_usd::numeric;
  v_unl_usd numeric := p_unlock_usd::numeric;
  v_share numeric := p_share::numeric;
  v_coin numeric := p_coin_usd::numeric;
  v_counts jsonb;
  v_feed_total numeric;
  v_unl_total numeric;
  r jsonb;
  v_usd numeric;
  v_coins int;
  v_hold int;
  v_earning uuid;
  v_paid numeric := 0;
  v_n int := 0;
begin
  if v_month >= date_trunc('month', now() at time zone 'utc')::date then
    return jsonb_build_object('ok', false, 'error', 'month_not_over');
  end if;
  if v_feed_usd < 0 or v_unl_usd < 0 then return jsonb_build_object('ok', false, 'error', 'bad_amount'); end if;
  perform pg_advisory_xact_lock(hashtext('ad_revenue_close'));
  if exists (select 1 from public.ad_revenue_months where month = v_month) then
    return jsonb_build_object('ok', false, 'error', 'already_closed');
  end if;
  v_counts := public.ad_revenue_counts(v_month, null);
  v_feed_total := (v_counts->>'interstitial_total')::numeric;
  v_unl_total := (v_counts->>'unlock_total')::numeric;
  insert into public.ad_revenue_months (month, interstitial_usd, unlock_usd, creator_share, interstitial_count,
                                        unlock_count, note)
  values (v_month, v_feed_usd, v_unl_usd, v_share, v_feed_total, v_unl_total, left(coalesce(p_note, ''), 300));
  for r in select * from jsonb_array_elements(v_counts->'creators') loop
    v_usd := 0;
    if v_feed_total > 0 then v_usd := v_usd + v_feed_usd * v_share * (r->>'interstitial')::numeric / v_feed_total; end if;
    if v_unl_total > 0 then v_usd := v_usd + v_unl_usd * v_share * (r->>'unlock')::numeric / v_unl_total; end if;
    v_usd := round(v_usd, 4);
    v_coins := floor(v_usd / v_coin);
    v_earning := null;
    if v_coins > 0 then
      v_hold := coalesce((p_holds->>(r->>'creator_id'))::int, p_default_hold, 14);
      insert into public.creator_earnings (creator_id, coins, kind, earning_key, status, available_at, note)
      values ((r->>'creator_id')::uuid, v_coins, 'ads', 'ads:' || to_char(v_month, 'YYYY-MM') || ':' || (r->>'creator_id'),
              case when v_hold > 0 then 'pending' else 'available' end,
              now() + make_interval(days => greatest(v_hold, 0)),
              'Ad revenue ' || to_char(v_month, 'YYYY-MM'))
      on conflict (earning_key) where earning_key is not null do nothing
      returning id into v_earning;
    end if;
    insert into public.ad_revenue_payouts (month, creator_id, interstitial_count, unlock_count, usd, coins, earning_id)
    values (v_month, (r->>'creator_id')::uuid, (r->>'interstitial')::bigint, (r->>'unlock')::bigint, v_usd, v_coins,
            v_earning);
    v_paid := v_paid + v_usd;
    v_n := v_n + 1;
  end loop;
  update public.ad_revenue_months set paid_usd = v_paid, creators = v_n where month = v_month;
  return jsonb_build_object('ok', true, 'month', v_month, 'creators', v_n, 'paid_usd', v_paid,
                            'interstitial_count', v_feed_total, 'unlock_count', v_unl_total);
end $$;

-- Reversals (refunds / fraud) also cover ad earnings.
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
  if e.kind not in ('unlock', 'donation', 'brand', 'ads') then
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
                                       available_at, reversal_of, note, donation_id, brand_application_id)
  values (e.creator_id, e.source_post_id, e.viewer_id, -e.coins, 'reversal', 'reversal:' || e.id::text, 'available',
          now(), e.id, left(coalesce(p_note, ''), 400), e.donation_id, e.brand_application_id)
  returning id into v_id;
  update public.creator_earnings set reversed_at = now() where id = e.id;
  return jsonb_build_object('ok', true, 'changed', true, 'status', 'deducted', 'reversal_id', v_id,
                            'creator_id', e.creator_id, 'coins', e.coins);
end $$;

do $$
declare f text;
begin
  foreach f in array array[
    'ad_impression_record(uuid, uuid, int)',
    'ad_revenue_counts(date, uuid)',
    'ad_revenue_close(date, text, text, text, text, int, jsonb, text)',
    'viyo_reverse_earning(uuid, text)'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
