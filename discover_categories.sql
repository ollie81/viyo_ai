-- Discover categories (For You, Shorts, Movies, Dramas, Ads, Photos) and share counts. Additive only, safe to
-- run more than once. Needs ads_publish.sql (posts.post_kind). Changes no existing data.

-- How many people shared a post (counted once per person per post every 6 hours by the backend).
alter table public.posts add column if not exists share_count integer not null default 0;

-- The category grids read public posts newest first, by type; these keep those pages fast.
create index if not exists posts_discover_type on public.posts (post_type, created_at desc)
  where is_private = false and is_archived = false;
create index if not exists posts_discover_kind on public.posts (post_kind, created_at desc)
  where is_private = false and is_archived = false;

-- Check (read-only):
-- select column_name from information_schema.columns where table_name = 'posts' and column_name = 'share_count';
