-- NardoPay: automatic daily USD/RWF rates from the National Bank of Rwanda. Run after nardopay_orders.sql.
-- Safe to run more than once. Adds columns (existing rows become manual, active rates), one guard index and one
-- small log table; no existing rate is changed or removed. Orders keep the rate they were created with.
--
-- nardopay_rates
--   source        manual (an admin) | bnr (BNR's own endpoint) | allratestoday (third-party copy of BNR's table)
--   rate_type     average (BNR's published average) | manual
--   published_on  the BNR publication date of an automatic rate
--   fetched_at    when VIYO fetched it
--   status        active | rejected (out of bounds or a jump too large: kept for review, never used)
--   reject_reason why it was rejected
--   hold_until    a manual rate wins over automatic ones until this time (empty = until BNR's next publication)
alter table public.nardopay_rates add column if not exists source text not null default 'manual';
alter table public.nardopay_rates add column if not exists rate_type text not null default 'manual';
alter table public.nardopay_rates add column if not exists published_on date;
alter table public.nardopay_rates add column if not exists fetched_at timestamptz;
alter table public.nardopay_rates add column if not exists status text not null default 'active';
alter table public.nardopay_rates add column if not exists reject_reason text not null default '';
alter table public.nardopay_rates add column if not exists hold_until timestamptz;

do $$ begin
  alter table public.nardopay_rates add constraint nardopay_rates_source_check
    check (source in ('manual', 'bnr', 'allratestoday'));
exception when duplicate_object then null; end $$;
do $$ begin
  alter table public.nardopay_rates add constraint nardopay_rates_status_check check (status in ('active', 'rejected'));
exception when duplicate_object then null; end $$;
do $$ begin
  alter table public.nardopay_rates add constraint nardopay_rates_auto_check
    check (source = 'manual' or (published_on is not null and fetched_at is not null));
exception when duplicate_object then null; end $$;

-- One automatic rate per source and publication date (a re-run of the daily fetch adds nothing).
create unique index if not exists nardopay_rates_source_day_key
  on public.nardopay_rates (source, published_on) where source <> 'manual';

-- Every automatic fetch attempt, so a failure is visible (never silent).
create table if not exists public.nardopay_rate_runs (
  id uuid primary key default gen_random_uuid(),
  ran_at timestamptz not null default now(),
  ok boolean not null,
  source text not null default '',
  detail text not null default '' check (char_length(detail) <= 1000),
  rate_id uuid references public.nardopay_rates(id) on delete set null
);
create index if not exists nardopay_rate_runs_ran_idx on public.nardopay_rate_runs (ran_at desc);
alter table public.nardopay_rate_runs enable row level security;
revoke all on public.nardopay_rate_runs from anon, authenticated;
grant all on public.nardopay_rate_runs to service_role;
