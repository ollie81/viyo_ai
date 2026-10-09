-- VIYO AI Ads Studio — new tables, handed over to run by hand in the
-- Supabase SQL editor (same "no migration-runner access from this
-- codebase" constraint every other migration this session has had).
--
-- Every table is RLS-enabled with zero public policies: all reads/writes
-- go exclusively through ads_studio.py's own service-role client, never
-- the Flutter app's anon-key client directly — same posture as every
-- Viyo Studio table (series_characters, series_scenes, studio_api_costs,
-- etc).

create extension if not exists pgcrypto;

create table if not exists ad_campaigns (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id),
  promote_target text not null check (promote_target in ('viyo', 'ollie_ai', 'other_app', 'website')),
  target_name text not null default '',
  target_description text not null default '',
  target_features jsonb not null default '[]'::jsonb,
  target_audience text not null default '',
  objective text not null default '',
  destination_link text not null default '',
  cta_text text not null default '',
  format text,
  recommended_format text,
  duration_seconds int not null default 15,
  aspect_ratio text not null default '9:16',
  resolution text not null default '720p',
  use_veo boolean not null default false,
  voice_gender_preference text,
  voice_name text,
  selected_hook_id uuid,
  script jsonb,
  music_url text,
  status text not null default 'draft',
  video_url text,
  bunny_video_id text,
  thumbnail_url text,
  duration_actual_seconds int,
  cost_usd_cents int not null default 0,
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists ad_assets (
  id uuid primary key default gen_random_uuid(),
  campaign_id uuid not null references ad_campaigns(id) on delete cascade,
  url text not null,
  asset_type text not null default 'reference',
  source text not null default 'uploaded',
  label text not null default '',
  created_at timestamptz not null default now()
);

create table if not exists ad_hook_candidates (
  id uuid primary key default gen_random_uuid(),
  campaign_id uuid not null references ad_campaigns(id) on delete cascade,
  hook_text text not null,
  angle text not null default '',
  rationale text not null default '',
  score_attention int not null,
  score_curiosity int not null,
  score_emotional int not null,
  score_relevance int not null,
  score_transition int not null,
  score_total numeric not null,
  rank int not null,
  selected boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists ad_generation_jobs (
  id uuid primary key default gen_random_uuid(),
  campaign_id uuid not null references ad_campaigns(id) on delete cascade,
  stage text not null default 'video',
  status text not null default 'pending',
  progress_pct int not null default 0,
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists ad_performance_metrics (
  id uuid primary key default gen_random_uuid(),
  campaign_id uuid not null references ad_campaigns(id) on delete cascade,
  hook_id uuid references ad_hook_candidates(id),
  retention_1s numeric,
  retention_3s numeric,
  retention_5s numeric,
  avg_watch_seconds numeric,
  completion_rate numeric,
  ctr numeric,
  installs int,
  source text not null default 'manual',
  notes text not null default '',
  recorded_at timestamptz not null default now()
);

-- Reusable library of real, admin-uploaded VIYO screenshots that any
-- future "Promote VIYO" campaign can attach from without re-uploading —
-- not tied to any one campaign (campaign_id lives on ad_assets, not here).
create table if not exists ad_viyo_asset_library (
  id uuid primary key default gen_random_uuid(),
  url text not null,
  label text not null default '',
  created_at timestamptz not null default now()
);

alter table ad_campaigns enable row level security;
alter table ad_assets enable row level security;
alter table ad_hook_candidates enable row level security;
alter table ad_generation_jobs enable row level security;
alter table ad_performance_metrics enable row level security;
alter table ad_viyo_asset_library enable row level security;

create index if not exists idx_ad_assets_campaign on ad_assets(campaign_id);
create index if not exists idx_ad_hook_candidates_campaign on ad_hook_candidates(campaign_id);
create index if not exists idx_ad_generation_jobs_campaign on ad_generation_jobs(campaign_id, created_at desc);
create index if not exists idx_ad_performance_metrics_campaign on ad_performance_metrics(campaign_id);
