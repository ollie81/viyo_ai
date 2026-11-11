-- Viyo Studio: the assistant's long-term memory of each production.
-- Run after creator_studio_continuity.sql. Safe to run more than once. Only adds.
--
-- assistant_memory: facts the creator stated or approved (story, characters and
-- relationships, preferences, decisions, production decisions), open questions,
-- and ideas not approved yet, kept apart. Shared by every conversation about
-- the production; the creator can see, correct and delete each item.
alter table public.studio_projects add column if not exists assistant_memory jsonb not null default '{}'::jsonb;
