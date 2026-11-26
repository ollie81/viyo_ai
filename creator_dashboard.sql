-- Creator dashboard: one read-only function that adds up a creator's audience, earnings and Studio numbers in the
-- database (exact counts, not capped by the API's row limit). Run after intelligence.sql, creator_earnings_ledger.sql,
-- creator_donations.sql and creator_studio_schema.sql. Safe to run more than once. Changes no data.
--
-- Everything is a real record: viewer_events (plays, completions; the creator's own plays left out), creator_earnings
-- (by kind and status, reversed rows left out), withdrawal_requests, studio_projects and studio_jobs (credits actually
-- charged). Missing tables simply give zeros, so the dashboard works before every feature is set up.

drop function if exists public.creator_dashboard_stats(uuid, int);
create or replace function public.creator_dashboard_stats(p_creator uuid, p_days int)
returns jsonb language plpgsql stable set search_path = public as $$
declare
  since timestamptz := now() - make_interval(days => greatest(coalesce(p_days, 30), 1));
  audience jsonb := '{}'::jsonb;
  earnings jsonb := '{}'::jsonb;
  withdrawals jsonb := '[]'::jsonb;
  studio jsonb := '{}'::jsonb;
begin
  -- Audience in the period (other people's plays of this creator's work).
  if to_regclass('public.viewer_events') is not null then
    select jsonb_build_object(
             'plays', count(*) filter (where kind = 'view'),
             'unique_viewers', count(distinct user_id) filter (where kind = 'view'),
             'watch_seconds', coalesce(sum(value) filter (where kind = 'view' and value > 0), 0),
             'started', count(distinct (user_id, post_id)) filter (where kind = 'view'),
             'completed', count(distinct (user_id, post_id)) filter (where kind = 'complete'))
      into audience
      from public.viewer_events
     where creator_id = p_creator and created_at >= since and user_id <> p_creator;
  end if;

  -- Earnings by kind and status (coins), reversed ones left out (a paid one is offset by its negative reversal row).
  if to_regclass('public.creator_earnings') is not null then
    select coalesce(jsonb_object_agg(k, v), '{}'::jsonb) into earnings from (
      select coalesce(kind, 'unlock') || ':' || coalesce(status, 'available') as k, sum(coins) as v
        from public.creator_earnings
       where creator_id = p_creator and coalesce(status, 'available') <> 'reversed'
       group by 1) t;
  end if;

  if to_regclass('public.withdrawal_requests') is not null then
    select coalesce(jsonb_agg(w order by w->>'created_at' desc), '[]'::jsonb) into withdrawals from (
      select jsonb_build_object('status', status, 'coins', coins, 'usd_amount', usd_amount,
                                'created_at', created_at, 'paid_at', paid_at) as w
        from public.withdrawal_requests where creator_id = p_creator
       order by created_at desc limit 10) t;
  end if;

  if to_regclass('public.studio_jobs') is not null then
    select jsonb_build_object(
             'projects', (select count(*) from public.studio_projects where owner_id = p_creator and status <> 'archived'),
             'active_jobs', count(*) filter (where status in ('queued', 'running')),
             'completed_jobs', count(*) filter (where status = 'succeeded'),
             'failed_jobs', count(*) filter (where status = 'failed'),
             'credits_used', coalesce(sum(charged_credits), 0),
             'credits_used_period', coalesce(sum(charged_credits) filter (where created_at >= since), 0))
      into studio
      from public.studio_jobs where owner_id = p_creator;
  end if;

  return jsonb_build_object('days', greatest(coalesce(p_days, 30), 1), 'audience', audience, 'earnings', earnings,
                            'withdrawals', withdrawals, 'studio', studio, 'computed_at', now());
end $$;

revoke all on function public.creator_dashboard_stats(uuid, int) from public, anon, authenticated;
grant execute on function public.creator_dashboard_stats(uuid, int) to service_role;
