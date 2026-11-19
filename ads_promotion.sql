-- Paid promotion of published ads (VIYO Coins). Additive only, safe to run more than once. Needs
-- ads_publish.sql. Changes no existing data. Stays unused until PROMOTIONS_ENABLED=true is set on the backend.

create table if not exists public.ad_promotions (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  post_id uuid not null references public.posts(id) on delete cascade,
  campaign_id uuid references public.ad_campaigns(id) on delete set null,
  budget_coins integer not null check (budget_coins > 0),
  price_per_1000 integer not null check (price_per_1000 > 0),     -- coins per 1,000 impressions, fixed at start
  days integer not null check (days between 1 and 14),
  placements jsonb not null default '["for_you", "ads"]'::jsonb,
  status text not null default 'active' check (status in ('active', 'ended')),
  ended_reason text,                       -- finished | budget_used | stopped | ad_not_public
  start_at timestamptz not null default now(),
  end_at timestamptz not null,
  ended_at timestamptz,
  impressions integer not null default 0,
  clicks integer not null default 0,
  spent_millicoins bigint not null default 0 check (spent_millicoins >= 0),   -- 1/1000 coin, never above budget
  refunded_coins integer,
  created_at timestamptz not null default now()
);
create index if not exists ad_promotions_owner on public.ad_promotions (owner_id, created_at desc);
create index if not exists ad_promotions_active on public.ad_promotions (status) where status = 'active';
alter table public.ad_promotions enable row level security;          -- no policies: backend only

-- One impression / click per viewer per promotion per day (the key makes double counting impossible).
create table if not exists public.ad_promotion_events (
  promotion_id uuid not null references public.ad_promotions(id) on delete cascade,
  viewer_id uuid not null,
  day date not null,
  kind text not null check (kind in ('impression', 'click')),
  created_at timestamptz not null default now(),
  primary key (promotion_id, viewer_id, day, kind)
);
alter table public.ad_promotion_events enable row level security;    -- no policies: backend only

-- Check (read-only):
-- select count(*) from public.ad_promotions;
