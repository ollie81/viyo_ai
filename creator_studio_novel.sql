-- Viyo Studio: Story Intelligence stage 2, the interview and the novel outline. Run after
-- creator_studio_canon.sql. Safe to run more than once. Only adds columns to studio_stories (a table added by
-- creator_studio_canon.sql); no other table changes.
--
-- idea            the creator's idea, in their own words
-- length          the novel's size: 'short' (12 chapters), 'standard' (24) or 'long' (36)
-- episodes        how many drama episodes it should become later (0 = not decided)
-- stage           interview -> outline -> outline_approved
-- interview       {"turns": [{question, answer, skipped}], "known": {topic: {value, creator_words}}, "done"}
-- outline         the novel plan (premise, characters, secrets, chapters, threads, ending, assumptions)
-- outline_problems what the plan check still finds wrong (shown to the creator)
-- outline_version  goes up on every new or edited outline; approval is for one version
alter table public.studio_stories add column if not exists idea text not null default '';
alter table public.studio_stories add column if not exists length text not null default 'standard';
alter table public.studio_stories add column if not exists episodes int not null default 0;
alter table public.studio_stories add column if not exists stage text not null default 'interview';
alter table public.studio_stories add column if not exists interview jsonb not null default '{}'::jsonb;
alter table public.studio_stories add column if not exists outline jsonb not null default '{}'::jsonb;
alter table public.studio_stories add column if not exists outline_problems jsonb not null default '[]'::jsonb;
alter table public.studio_stories add column if not exists outline_version int not null default 0;
alter table public.studio_stories add column if not exists outline_approved_version int;
alter table public.studio_stories add column if not exists outline_approved_at timestamptz;

do $$ begin
  alter table public.studio_stories add constraint studio_stories_length_check
    check (length in ('short', 'standard', 'long'));
exception when duplicate_object then null; end $$;
do $$ begin
  alter table public.studio_stories add constraint studio_stories_stage_check
    check (stage in ('interview', 'outline', 'outline_approved'));
exception when duplicate_object then null; end $$;
do $$ begin
  alter table public.studio_stories add constraint studio_stories_episodes_check check (episodes between 0 and 200);
exception when duplicate_object then null; end $$;
do $$ begin
  alter table public.studio_stories add constraint studio_stories_idea_check check (char_length(idea) <= 4000);
exception when duplicate_object then null; end $$;
create unique index if not exists studio_stories_project_idx on public.studio_stories (project_id)
  where project_id is not null and status = 'draft';
