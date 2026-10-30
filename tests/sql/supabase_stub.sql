-- Minimal stand-ins for the Supabase objects repurpose_jobs_schema.sql
-- depends on, so the real migration can be applied to a plain local
-- Postgres in tests. Only the columns the backend actually uses.
do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin bypassrls; end if;
end $$;
create schema if not exists auth;
create table if not exists auth.users (id uuid primary key);
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  points_balance int not null default 0
);
create table if not exists public.transactions (
  id bigserial primary key,
  user_id uuid not null,
  amount int not null,
  type text not null,
  description text,
  created_at timestamptz not null default now()
);
create table if not exists public.posts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid,
  media_url text,
  thumbnail_url text
);
create schema if not exists storage;
create table if not exists storage.buckets (
  id text primary key,
  name text not null,
  public boolean default false,
  file_size_limit bigint,
  allowed_mime_types text[]
);
grant usage on schema public, auth, storage to service_role;
-- Supabase gives client roles schema access; table/function privileges
-- are what actually gate them.
grant usage on schema public to anon, authenticated;
grant all on all tables in schema public to service_role;
