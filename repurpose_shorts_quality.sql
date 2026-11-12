-- AI Shorts quality upgrade: what kind of moments the creator wants, and what they did with the clips.
-- Additive only, safe to run more than once. Without it, jobs pick "best moments overall" and nothing is learned.

-- 1. The kind of moments the creator asked for (the free-text description stays in creative_prompt).
alter table public.repurpose_jobs add column if not exists moment_focus text
  check (moment_focus is null or moment_focus in ('best', 'funny', 'emotional', 'educational', 'conversation',
                                                  'story', 'product', 'highlights', 'visual'));

create or replace function public.repurpose_set_moment_focus(p_job_id uuid, p_user_id uuid, p_focus text)
returns boolean
language sql security definer set search_path = public as $$
  with u as (
    update public.repurpose_jobs set moment_focus = p_focus
     where id = p_job_id and user_id = p_user_id and status = 'queued'
       and p_focus in ('best', 'funny', 'emotional', 'educational', 'conversation', 'story', 'product',
                       'highlights', 'visual')
    returning 1)
  select exists (select 1 from u);
$$;

-- 2. What creators do with their clips. Used only to tune THAT creator's own future selections
--    (never shared across creators, never used to train a model). A download is not a publish.
create table if not exists public.repurpose_feedback (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  job_id uuid not null references public.repurpose_jobs(id) on delete cascade,
  rank int not null,
  event text not null check (event in ('downloaded', 'shared', 'exported', 'not_useful', 'regenerated', 'kept')),
  category text,
  duration_seconds real,
  platform text,
  created_at timestamptz not null default now()
);
create index if not exists repurpose_feedback_user on public.repurpose_feedback (user_id, created_at desc);
alter table public.repurpose_feedback enable row level security;   -- no policies: only the backend reads/writes

-- Records one event on the creator's own clip (the job must be theirs). One of each event per clip per day.
create or replace function public.repurpose_record_feedback(p_user_id uuid, p_job_id uuid, p_rank int, p_event text,
                                                            p_category text, p_duration double precision, p_platform text)
returns boolean
language plpgsql security definer set search_path = public as $$
begin
  if not exists (select 1 from public.repurpose_jobs where id = p_job_id and user_id = p_user_id) then
    return false;
  end if;
  if exists (select 1 from public.repurpose_feedback
              where job_id = p_job_id and rank = p_rank and event = p_event
                and coalesce(platform, '') = coalesce(p_platform, '')
                and created_at > now() - interval '1 day') then
    return true;
  end if;
  insert into public.repurpose_feedback (user_id, job_id, rank, event, category, duration_seconds, platform)
  values (p_user_id, p_job_id, p_rank, p_event, left(p_category, 30), p_duration, left(p_platform, 20));
  return true;
end;
$$;

-- The creator's own recent events (for their selection prior).
create or replace function public.repurpose_feedback_events(p_user_id uuid, p_limit int)
returns jsonb
language sql stable security definer set search_path = public as $$
  select coalesce(jsonb_agg(jsonb_build_object('event', event, 'category', category,
                                               'duration', duration_seconds, 'platform', platform)), '[]'::jsonb)
    from (select * from public.repurpose_feedback where user_id = p_user_id
           order by created_at desc limit least(greatest(p_limit, 1), 1000)) f;
$$;

revoke all on function public.repurpose_set_moment_focus(uuid, uuid, text) from public, anon, authenticated;
revoke all on function public.repurpose_record_feedback(uuid, uuid, int, text, text, double precision, text) from public, anon, authenticated;
revoke all on function public.repurpose_feedback_events(uuid, int) from public, anon, authenticated;
grant execute on function public.repurpose_set_moment_focus(uuid, uuid, text) to service_role;
grant execute on function public.repurpose_record_feedback(uuid, uuid, int, text, text, double precision, text) to service_role;
grant execute on function public.repurpose_feedback_events(uuid, int) to service_role;

-- Check (read-only):
-- select column_name from information_schema.columns where table_name = 'repurpose_jobs' and column_name = 'moment_focus';
-- select count(*) from public.repurpose_feedback;
