-- Advertising insights (daily snapshots of each ad post's counts) and the experiment lab (versions of an ad
-- in different tones). Additive only, safe to run more than once. Needs ads_publish.sql, ads_brands.sql and
-- discover_categories.sql (posts.share_count). Changes no existing data.

-- One row per ad post per day, saved when its creator opens their insights: views and shares per day are
-- worked out only between two real snapshots.
create table if not exists public.ad_post_stats_daily (
  post_id uuid not null references public.posts(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  day date not null,
  views integer not null default 0,
  likes integer not null default 0,
  comments integer not null default 0,
  shares integer not null default 0,
  boosted boolean not null default false,
  primary key (post_id, day)
);
create index if not exists ad_post_stats_owner on public.ad_post_stats_daily (owner_id, day);
alter table public.ad_post_stats_daily enable row level security;   -- no policies: backend only

-- An experiment: several versions of an ad for one brand, each in a different tone.
create table if not exists public.ad_experiments (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  brand_id uuid references public.ad_brands(id) on delete set null,
  name text not null default '',
  created_at timestamptz not null default now()
);
create index if not exists ad_experiments_owner on public.ad_experiments (owner_id, created_at desc);
alter table public.ad_experiments enable row level security;        -- no policies: backend only

alter table public.ad_campaigns add column if not exists experiment_id uuid references public.ad_experiments(id) on delete set null;
alter table public.ad_campaigns add column if not exists variant_tone text not null default '';

-- Follower growth reads follows.created_at. Check it exists (read-only):
-- select column_name from information_schema.columns where table_name = 'follows' and column_name = 'created_at';
