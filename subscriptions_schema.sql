-- Premium subscription columns on profiles (read by subscriptions.py, episodes.py and the app).
-- They were never added to the live database, which made every coin unlock fail. Safe to run more than once;
-- it changes no existing data (every account starts as not subscribed).
alter table public.profiles add column if not exists is_subscribed boolean not null default false;
alter table public.profiles add column if not exists subscription_plan text;
alter table public.profiles add column if not exists subscription_provider text;
alter table public.profiles add column if not exists subscription_status text;
alter table public.profiles add column if not exists subscription_id text;
alter table public.profiles add column if not exists subscription_renews_at timestamptz;
alter table public.profiles add column if not exists subscription_ends_at timestamptz;

-- Only the server (the payment webhook, using the service role) may change these. A signed-in user editing their
-- own profile from the app could otherwise switch is_subscribed on and watch every paid episode for free.
create or replace function public.protect_subscription_columns() returns trigger
language plpgsql as $$
begin
  if current_user in ('anon', 'authenticated') and (
       new.is_subscribed is distinct from old.is_subscribed
    or new.subscription_plan is distinct from old.subscription_plan
    or new.subscription_provider is distinct from old.subscription_provider
    or new.subscription_status is distinct from old.subscription_status
    or new.subscription_id is distinct from old.subscription_id
    or new.subscription_renews_at is distinct from old.subscription_renews_at
    or new.subscription_ends_at is distinct from old.subscription_ends_at) then
    raise exception 'Subscription details can only be changed by VIYO.' using errcode = '42501';
  end if;
  return new;
end $$;

drop trigger if exists protect_subscription_columns on public.profiles;
create trigger protect_subscription_columns before update on public.profiles
  for each row execute function public.protect_subscription_columns();

-- A new profile can't be created already subscribed either.
create or replace function public.protect_subscription_columns_insert() returns trigger
language plpgsql as $$
begin
  if current_user in ('anon', 'authenticated') and coalesce(new.is_subscribed, false) then
    new.is_subscribed := false;
  end if;
  return new;
end $$;

drop trigger if exists protect_subscription_columns_insert on public.profiles;
create trigger protect_subscription_columns_insert before insert on public.profiles
  for each row execute function public.protect_subscription_columns_insert();
