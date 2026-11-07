-- Viyo Studio: the story engine's plan for each production.
-- Safe to run more than once. Only adds.
--
-- story_bible: what the writers' room decided before writing (premise,
-- characters with their want, flaw, secret and way of speaking, the beats,
-- the ending), the script doctor's scores and fixes, and the episode
-- summary and open storylines the next episode in a series builds on.
alter table public.studio_projects add column if not exists story_bible jsonb not null default '{}'::jsonb;
