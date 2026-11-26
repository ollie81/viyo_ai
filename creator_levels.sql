-- Creator levels: one read-only function that returns the real numbers a creator's level is based on. The levels
-- themselves (New, Rising, Pro, Star) and their requirements are in creator_levels.py. Safe to run more than once.
-- Changes no data. Every creator earns the same 65% at every level; the only perk is a shorter earnings hold for
-- Pro and Star, and a refund or reversal in the last 90 days drops a creator below Pro.
--   followers      follows rows pointing at the creator
--   viewers_30d    different people (not the creator) who played their work in the last 30 days
--   account_days   days since the account was created
--   earned_coins   everything earned, net of refunds (reversed rows left out, reversal rows deducted)
--   reversals_90d  earnings reversed (refund / fraud) in the last 90 days

drop function if exists public.creator_level_metrics(uuid);
create or replace function public.creator_level_metrics(p_creator uuid)
returns jsonb language plpgsql stable set search_path = public as $$
declare
  v_followers bigint := 0;
  v_viewers bigint := 0;
  v_created timestamptz;
  v_earned bigint := 0;
  v_reversals bigint := 0;
begin
  if to_regclass('public.follows') is not null then
    select count(*) into v_followers from public.follows where following_id = p_creator;
  end if;
  if to_regclass('public.viewer_events') is not null then
    select count(distinct user_id) into v_viewers from public.viewer_events
     where creator_id = p_creator and kind = 'view' and user_id <> p_creator
       and created_at >= now() - interval '30 days';
  end if;
  if exists (select 1 from information_schema.columns
              where table_schema = 'auth' and table_name = 'users' and column_name = 'created_at') then
    execute 'select created_at from auth.users where id = $1' into v_created using p_creator;
  end if;
  if to_regclass('public.creator_earnings') is not null then
    select coalesce(sum(coins) filter (where coalesce(status, 'available') <> 'reversed'), 0),
           count(*) filter (where (status = 'reversed' and reversed_at >= now() - interval '90 days')
                               or (kind = 'reversal' and created_at >= now() - interval '90 days'))
      into v_earned, v_reversals
      from public.creator_earnings where creator_id = p_creator;
  end if;
  return jsonb_build_object(
    'followers', v_followers, 'viewers_30d', v_viewers,
    'account_days', case when v_created is null then null else floor(extract(epoch from now() - v_created) / 86400) end,
    'earned_coins', greatest(v_earned, 0), 'reversals_90d', v_reversals);
end $$;

revoke all on function public.creator_level_metrics(uuid) from public, anon, authenticated;
grant execute on function public.creator_level_metrics(uuid) to service_role;
