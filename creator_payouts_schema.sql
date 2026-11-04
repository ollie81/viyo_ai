-- Creator payouts: bought vs free coins, payout details, withdrawal records.
-- Run manually in the Supabase SQL editor. Safe to re-run.

-- 1) Bought coins. profiles.paid_coins is the part of points_balance that
--    was bought with real money. Spending uses free coins first, so it is
--    simply capped at the balance after every change. Creators earn cash
--    only from the bought part of what a viewer spends on their work.
alter table public.profiles add column if not exists paid_coins numeric not null default 0;

create or replace function public.viyo_cap_paid_coins()
returns trigger
language plpgsql as $$
begin
  new.paid_coins := least(greatest(coalesce(new.paid_coins, 0), 0),
                          greatest(coalesce(new.points_balance, 0), 0));
  return new;
end;
$$;

drop trigger if exists viyo_cap_paid_coins on public.profiles;
create trigger viyo_cap_paid_coins
  before insert or update of points_balance, paid_coins on public.profiles
  for each row execute function public.viyo_cap_paid_coins();

create or replace function public.viyo_count_paid_coins()
returns trigger
language plpgsql security definer set search_path = public as $$
begin
  update public.profiles set paid_coins = paid_coins + new.amount where id = new.user_id;
  return new;
end;
$$;

drop trigger if exists viyo_count_paid_coins on public.transactions;
create trigger viyo_count_paid_coins
  after insert on public.transactions
  for each row when (new.type = 'coin_purchase' and new.amount > 0)
  execute function public.viyo_count_paid_coins();

-- 2) Where each creator wants to be paid. Written only by the backend.
create table if not exists public.creator_payout_settings (
  user_id uuid primary key references auth.users(id) on delete cascade,
  method text not null check (method in ('paypal', 'mobile_money')),
  paypal_email text,
  momo_phone text,
  momo_network text,
  account_name text,
  country text,
  updated_at timestamptz not null default now()
);
alter table public.creator_payout_settings enable row level security;
-- No policies: only the backend (service role) reads or writes these.

-- 3) Withdrawal records: amount in dollars, where to send it, and status.
alter table public.withdrawal_requests add column if not exists status text not null default 'pending';
alter table public.withdrawal_requests add column if not exists usd_amount numeric;
alter table public.withdrawal_requests add column if not exists payout_method text;
alter table public.withdrawal_requests add column if not exists payout_destination text;
alter table public.withdrawal_requests add column if not exists paid_at timestamptz;
alter table public.withdrawal_requests add column if not exists admin_note text;

alter table public.creator_earnings add column if not exists withdrawal_request_id uuid;
create index if not exists creator_earnings_withdrawal_idx on public.creator_earnings (withdrawal_request_id);
