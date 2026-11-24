-- NardoPay coin checkout (web only): exchange rates, server-side orders and the webhook event log.
-- Plan: VIYO_NardoPay_Integration_Plan_v3. Safe to run more than once. Only adds new tables; no existing table,
-- function or payment provider (Stripe, Paystack, Flutterwave, Lemon Squeezy, Google Play) is changed.
--
-- Who can do what:
--   the backend (service role)  reads and writes everything;
--   a signed-in user            reads ONLY their own orders, and writes nothing;
--   everyone else               nothing. Rates and webhook events (which may hold customer details) are server-only.

-- ---------------------------------------------------------------- exchange rates (RWF per USD), append-only history
create table if not exists public.nardopay_rates (
  id uuid primary key default gen_random_uuid(),
  rate numeric(12,4) not null check (rate > 0),
  set_by text not null check (char_length(set_by) between 1 and 120),
  note text not null default '' check (char_length(note) <= 400),
  set_at timestamptz not null default now()
);
create index if not exists nardopay_rates_set_at_idx on public.nardopay_rates (set_at desc);

-- ---------------------------------------------------------------- the webhook event log
-- One row per distinct signed delivery (body_sha256 is unique, so a repeated delivery finds the existing row).
-- status: received -> processing -> processed | ignored | mismatch | duplicate_payment | error.
-- raw_body is kept only for a limited time (it may contain customer details) and then cleared; the hash and the
-- few fields needed for audit stay.
create table if not exists public.nardopay_webhook_events (
  id uuid primary key default gen_random_uuid(),
  body_sha256 text not null unique check (body_sha256 ~ '^[0-9a-f]{64}$'),
  provider_event_id text,
  event_type text not null default '' check (char_length(event_type) <= 80),
  order_id uuid,
  transaction_id text,
  amount numeric(14,2),
  currency text,
  signature_ok boolean not null,
  status text not null default 'received'
    check (status in ('received', 'processing', 'processed', 'ignored', 'mismatch', 'duplicate_payment', 'error')),
  detail text not null default '' check (char_length(detail) <= 1000),
  attempts int not null default 0 check (attempts >= 0),
  locked_at timestamptz,
  raw_body text,
  raw_purged_at timestamptz,
  received_at timestamptz not null default now(),
  processed_at timestamptz,
  check (raw_purged_at is null or raw_body is null)
);
create unique index if not exists nardopay_webhook_events_provider_id_key
  on public.nardopay_webhook_events (provider_event_id) where provider_event_id is not null;
create index if not exists nardopay_webhook_events_status_idx on public.nardopay_webhook_events (status, received_at);

-- ---------------------------------------------------------------- orders, created by the server before checkout
-- The price and coins come from the server's package list; the rate used is locked on the order.
-- status: pending -> crediting -> credited, or failed / mismatch / expired.
-- An order can only be crediting or credited after a VERIFIED payment (signature checked, amount and currency
-- matched): the transaction, the verified event and the time are then required, so nothing can credit an
-- unverified order, reconciliation included.
create table if not exists public.nardopay_orders (
  id uuid primary key default gen_random_uuid(),                  -- sent to NardoPay as metadata.order_id
  user_id uuid not null references auth.users(id) on delete cascade,
  package_id text not null check (char_length(package_id) between 1 and 40),
  coins int not null check (coins > 0),
  usd_cents int not null check (usd_cents > 0),
  rate_id uuid references public.nardopay_rates(id),
  usd_rwf_rate numeric(12,4) not null check (usd_rwf_rate > 0),
  rwf_amount bigint not null check (rwf_amount > 0),
  currency text not null default 'RWF' check (currency = 'RWF'),
  status text not null default 'pending'
    check (status in ('pending', 'crediting', 'credited', 'failed', 'mismatch', 'expired')),
  payment_link text,
  transaction_id text unique,
  verified_event_id uuid references public.nardopay_webhook_events(id),
  payment_verified_at timestamptz,
  failure_reason text not null default '' check (char_length(failure_reason) <= 400),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  expires_at timestamptz not null default (now() + interval '24 hours'),
  credited_at timestamptz,
  check (status not in ('crediting', 'credited')
         or (transaction_id is not null and verified_event_id is not null and payment_verified_at is not null)),
  check (status <> 'credited' or credited_at is not null)
);
create index if not exists nardopay_orders_user_idx on public.nardopay_orders (user_id, created_at desc);
create index if not exists nardopay_orders_crediting_idx on public.nardopay_orders (status, updated_at)
  where status = 'crediting';

-- ---------------------------------------------------------------- permissions
alter table public.nardopay_rates enable row level security;
alter table public.nardopay_webhook_events enable row level security;
alter table public.nardopay_orders enable row level security;

revoke all on public.nardopay_rates from anon, authenticated;
revoke all on public.nardopay_webhook_events from anon, authenticated;
revoke all on public.nardopay_orders from anon, authenticated;

grant all on public.nardopay_rates to service_role;
grant all on public.nardopay_webhook_events to service_role;
grant all on public.nardopay_orders to service_role;

-- A signed-in user can read their own orders and nothing else. There are no insert/update/delete policies, and
-- no write privileges, so clients can't change orders at all.
grant select on public.nardopay_orders to authenticated;
drop policy if exists "owner reads own orders" on public.nardopay_orders;
create policy "owner reads own orders" on public.nardopay_orders
  for select to authenticated using (auth.uid() = user_id);
