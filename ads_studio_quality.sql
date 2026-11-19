-- Ads Maker: post-render quality check results, preview for ads that need correction, more product types.
-- Additive only, safe to run more than once. Changes no existing data.

-- The quality check of the finished video (what passed, what failed and why, what was corrected).
alter table public.ad_campaigns add column if not exists quality_report jsonb;
-- When the check fails: the video, to look at, never offered as a finished ad (video_url stays empty).
alter table public.ad_campaigns add column if not exists preview_video_url text;

-- What is advertised: apps and websites as before, plus physical products, businesses/services,
-- films/shows and creator projects.
alter table public.ad_campaigns drop constraint if exists ad_campaigns_promote_target_check;
alter table public.ad_campaigns add constraint ad_campaigns_promote_target_check
  check (promote_target in ('viyo', 'ollie_ai', 'other_app', 'website', 'physical_product', 'business', 'film',
                            'creator_project'));

-- Check (read-only):
-- select column_name from information_schema.columns where table_name = 'ad_campaigns'
--   and column_name in ('quality_report', 'preview_video_url');

-- ---------------------------------------------------------------- creators (Ads Maker)

-- The brand's own name, when it differs from the product's (target_name). Shown exactly as written.
alter table public.ad_campaigns add column if not exists brand_name text not null default '';
-- Who made the campaign: 'admin' (Ads Studio) or 'creator' (Ads Maker). Creators' files are private.
alter table public.ad_campaigns add column if not exists created_via text not null default 'admin';
create index if not exists ad_campaigns_user_created on public.ad_campaigns (user_id, created_at desc);

-- Holds Studio credits for one paid Ads Maker step (same lots and ledger as Creator Studio). Returns
-- {ok, hold_id} or {ok:false, error:'insufficient_credits', balance}; nothing is taken when it fails.
create or replace function public.ads_credit_hold(p_user_id uuid, p_credits bigint)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_hold uuid;
begin
  if p_credits is null or p_credits <= 0 then
    return jsonb_build_object('ok', false, 'error', 'bad_amount');
  end if;
  perform pg_advisory_xact_lock(hashtext('studio_credits:' || p_user_id::text));
  begin
    v_hold := public._studio_hold(p_user_id, p_credits, null);
  exception when sqlstate 'P0001' then
    return jsonb_build_object('ok', false, 'error', 'insufficient_credits',
                              'balance', public.studio_credit_balance(p_user_id));
  end;
  return jsonb_build_object('ok', true, 'hold_id', v_hold);
end;
$$;

-- Keeps p_used of a hold (the real cost) and returns the rest. Settling twice does nothing.
create or replace function public.ads_credit_settle(p_hold_id uuid, p_used bigint)
returns bigint language sql security definer set search_path = public as $$
  select public._studio_settle(p_hold_id, p_used, null);
$$;

revoke all on function public.ads_credit_hold(uuid, bigint) from public, anon, authenticated;
revoke all on function public.ads_credit_settle(uuid, bigint) from public, anon, authenticated;
grant execute on function public.ads_credit_hold(uuid, bigint) to service_role;
grant execute on function public.ads_credit_settle(uuid, bigint) to service_role;
