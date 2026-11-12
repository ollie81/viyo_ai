-- AI Shorts Creator: let the creator choose how many shorts one video becomes.
-- Safe to run more than once. Without it, jobs keep choosing the number automatically.

alter table public.repurpose_jobs add column if not exists clip_count int
  check (clip_count is null or clip_count between 1 and 10);

-- Sets the number on the creator's own job, only while it is still waiting to start.
create or replace function public.repurpose_set_clip_count(p_job_id uuid, p_user_id uuid, p_count int)
returns boolean
language sql security definer set search_path = public as $$
  with u as (
    update public.repurpose_jobs set clip_count = p_count
     where id = p_job_id and user_id = p_user_id and status = 'queued' and p_count between 1 and 10
    returning 1)
  select exists (select 1 from u);
$$;

revoke all on function public.repurpose_set_clip_count(uuid, uuid, int) from public, anon, authenticated;
grant execute on function public.repurpose_set_clip_count(uuid, uuid, int) to service_role;

-- Check (read-only):
-- select column_name from information_schema.columns where table_name = 'repurpose_jobs' and column_name = 'clip_count';
