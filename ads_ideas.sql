-- Campaign Universe and daily ad ideas. Additive only, safe to run more than once. Needs ads_brands.sql.
-- Changes no existing data.

-- Which campaign track an ad belongs to (product_intro, lifestyle, character_story, demo, seasonal,
-- new_concept), and the creative concept it was made from (a chosen daily idea).
alter table public.ad_campaigns add column if not exists track text not null default '';
alter table public.ad_campaigns add column if not exists creative_brief text not null default '';

-- Ad ideas for a brand: text only (title, hook, concept, story beats), free, made when the creator asks.
create table if not exists public.ad_ideas (
  id uuid primary key default gen_random_uuid(),
  brand_id uuid not null references public.ad_brands(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  day date not null default (now() at time zone 'utc')::date,
  batch integer not null default 1,
  title text not null default '',
  track text not null default 'new_concept',
  hook text not null default '',
  concept text not null default '',
  beats jsonb not null default '[]'::jsonb,
  status text not null default 'new' check (status in ('new', 'saved', 'dismissed', 'used')),
  campaign_id uuid references public.ad_campaigns(id) on delete set null,
  created_at timestamptz not null default now()
);
create index if not exists ad_ideas_brand on public.ad_ideas (brand_id, created_at desc);
create index if not exists ad_ideas_owner_day on public.ad_ideas (owner_id, day);
alter table public.ad_ideas enable row level security;    -- no policies: backend only

-- Check (read-only):
-- select count(*) from public.ad_ideas;
