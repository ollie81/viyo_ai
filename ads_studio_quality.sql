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
