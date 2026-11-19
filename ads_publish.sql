-- Publish to VIYO: a finished Ads Maker ad becomes an ordinary post (same posts table, same Bunny video, no
-- copy), marked as an ad. Additive only, safe to run more than once. Changes no existing post.

-- 'post' for every existing and normal post; 'ad' for an ad published from Ads Maker.
alter table public.posts add column if not exists post_kind text not null default 'post';
alter table public.posts drop constraint if exists posts_post_kind_check;
alter table public.posts add constraint posts_post_kind_check check (post_kind in ('post', 'ad'));
alter table public.posts add column if not exists ad_campaign_id uuid references public.ad_campaigns(id) on delete set null;
alter table public.posts add column if not exists ad_category text;
alter table public.posts add column if not exists ad_link text;      -- the advertiser's website/contact ("Visit")
alter table public.posts add column if not exists ad_info text;      -- product information, the advertiser's own facts
create index if not exists posts_kind_created on public.posts (post_kind, created_at desc);

-- Which post a campaign was published as (publishing again updates that post, never a duplicate).
alter table public.ad_campaigns add column if not exists published_post_id uuid references public.posts(id) on delete set null;

-- Check (read-only):
-- select column_name from information_schema.columns where table_name = 'posts' and column_name = 'post_kind';
