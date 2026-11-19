-- Ads Maker for any product, service, business or idea: a product profile with verified facts, brand
-- colours, social accounts and contact details; a production style; and a fictional ad character that
-- stays the same across scenes and can be reused. Additive only, safe to run more than once. Changes no
-- existing data. Needs ads_studio_quality.sql (already run).

alter table public.ad_campaigns add column if not exists product_subtype text not null default '';
-- The advertiser's verified facts: {features, benefits, prices, selling_points, specs}: lists of text.
-- The only claims an ad may make.
alter table public.ad_campaigns add column if not exists facts jsonb not null default '{}'::jsonb;
alter table public.ad_campaigns add column if not exists brand_colors jsonb not null default '[]'::jsonb;
alter table public.ad_campaigns add column if not exists social_links jsonb not null default '[]'::jsonb;
alter table public.ad_campaigns add column if not exists contact jsonb not null default '{}'::jsonb;
alter table public.ad_campaigns add column if not exists ad_style text not null default 'auto';
-- auto: the AI decides whether a character helps; custom: the creator described one; reuse: a saved one;
-- none: product-led, no character.
alter table public.ad_campaigns add column if not exists character_mode text not null default 'auto';
alter table public.ad_campaigns add column if not exists character jsonb not null default '{}'::jsonb;

alter table public.ad_campaigns drop constraint if exists ad_campaigns_promote_target_check;
alter table public.ad_campaigns add constraint ad_campaigns_promote_target_check
  check (promote_target in ('viyo', 'ollie_ai', 'other_app', 'website', 'digital_product', 'physical_product',
                            'service', 'business', 'personal_brand', 'film', 'music', 'event', 'creator_project'));

-- Fictional ad characters a creator can reuse in later ads: their description, reference portrait (private
-- file reference) and voice. Only the backend reads/writes; every read is filtered by owner.
create table if not exists public.ad_characters (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  name text not null default '',
  spec jsonb not null default '{}'::jsonb,
  portrait text,
  voice_name text,
  created_at timestamptz not null default now()
);
create index if not exists ad_characters_owner on public.ad_characters (owner_id, created_at desc);
alter table public.ad_characters enable row level security;   -- no policies: backend only

-- Check (read-only):
-- select column_name from information_schema.columns where table_name = 'ad_campaigns' and column_name = 'facts';
