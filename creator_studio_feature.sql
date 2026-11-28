-- VIYO Studio feature films (studio_feature): a novel can be planned as ONE continuous movie made of parts,
-- instead of a series of episodes. Additive and safe to run more than once; no existing data changes.
--   plan_format       'episodes' (as before) or 'feature'
--   feature_minutes   the movie's planned length (15-90), for a feature plan
alter table public.studio_stories add column if not exists plan_format text not null default 'episodes';
alter table public.studio_stories add column if not exists feature_minutes int;
do $$ begin
  alter table public.studio_stories add constraint studio_stories_plan_format_check
    check (plan_format in ('episodes', 'feature'));
exception when duplicate_object then null; end $$;
do $$ begin
  alter table public.studio_stories add constraint studio_stories_feature_minutes_check
    check (feature_minutes is null or feature_minutes between 15 and 90);
exception when duplicate_object then null; end $$;
