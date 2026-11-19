-- My Brands: a creator saves a brand once (details, logo, colours, product photos, verified facts, links,
-- its advertising character) and new ads start from it. Additive only, safe to run more than once.
-- Needs ads_universal.sql (ad_characters). Changes no existing data.

create table if not exists public.ad_brands (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  name text not null default '',
  description text not null default '',
  promote_target text not null default 'physical_product',
  product_subtype text not null default '',
  website text not null default '',
  cta_text text not null default '',
  target_audience text not null default '',
  facts jsonb not null default '{}'::jsonb,          -- the advertiser's verified facts, same shape as a campaign
  brand_colors jsonb not null default '[]'::jsonb,
  social_links jsonb not null default '[]'::jsonb,
  contact jsonb not null default '{}'::jsonb,
  logo text,                                          -- private file reference
  character_id uuid references public.ad_characters(id) on delete set null,
  archived boolean not null default false,
  created_at timestamptz not null default now()
);
create index if not exists ad_brands_owner on public.ad_brands (owner_id, created_at desc);
alter table public.ad_brands enable row level security;          -- no policies: backend only

-- The brand's pictures (logo, product photos): private files in the owner's folder.
create table if not exists public.ad_brand_assets (
  id uuid primary key default gen_random_uuid(),
  brand_id uuid not null references public.ad_brands(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  url text not null,
  asset_type text not null default 'product_photo',
  label text not null default '',
  created_at timestamptz not null default now()
);
create index if not exists ad_brand_assets_brand on public.ad_brand_assets (brand_id, created_at);
alter table public.ad_brand_assets enable row level security;    -- no policies: backend only

-- Which brand an ad and a character belong to.
alter table public.ad_campaigns add column if not exists brand_id uuid references public.ad_brands(id) on delete set null;
alter table public.ad_characters add column if not exists brand_id uuid references public.ad_brands(id) on delete set null;
create index if not exists ad_campaigns_brand on public.ad_campaigns (brand_id, created_at desc);

-- Check (read-only):
-- select count(*) from public.ad_brands;
