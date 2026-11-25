-- Viyo Studio: Story Intelligence stage 7, the episode planner. Run after creator_studio_story_bible.sql.
-- Safe to run more than once. Only adds columns; no data changes.
--
-- studio_stories
--   episode_plan            the planned episodes: chapters covered, title, summary, beats, hook, turn, cliffhanger
--   episode_plan_problems   what the plan check still finds wrong (shown to the creator)
--   episode_plan_version    goes up on every new or edited plan; approval is for one version
--   episode_seconds         the planned length of each episode
--   episodes_created_at     when the approved plan became episode productions (once)
--   episode_series_id       the series those episode productions belong to
-- studio_projects
--   story_id, episode_number  an episode production adapted from a novel, and which planned episode it is
alter table public.studio_stories add column if not exists episode_plan jsonb not null default '[]'::jsonb;
alter table public.studio_stories add column if not exists episode_plan_problems jsonb not null default '[]'::jsonb;
alter table public.studio_stories add column if not exists episode_plan_version int not null default 0;
alter table public.studio_stories add column if not exists episode_plan_approved_version int;
alter table public.studio_stories add column if not exists episode_seconds int not null default 90;
alter table public.studio_stories add column if not exists episodes_created_at timestamptz;
alter table public.studio_stories add column if not exists episode_series_id uuid;

alter table public.studio_projects add column if not exists story_id uuid references public.studio_stories(id) on delete set null;
alter table public.studio_projects add column if not exists episode_number int;
create index if not exists studio_projects_story_idx on public.studio_projects (story_id, episode_number)
  where story_id is not null;

do $$ begin
  alter table public.studio_stories add constraint studio_stories_episode_seconds_check check (episode_seconds between 30 and 600);
exception when duplicate_object then null; end $$;
