-- Viyo Studio: Story Intelligence stage 6, approving the whole novel and compiling it into the production's Story
-- Bible. Run after creator_studio_review.sql. Safe to run more than once. Only widens one check and adds one column
-- on studio_stories; no data changes.
--
-- stage 'novel_approved'  every chapter is approved and the creator approved the novel as a whole: its memory has
--                         been compiled into the production's story_bible (saga, knowledge, novel) and its
--                         characters, places and objects linked to the production's cast and elements.
alter table public.studio_stories add column if not exists novel_approved_at timestamptz;

alter table public.studio_stories drop constraint if exists studio_stories_stage_check;
alter table public.studio_stories add constraint studio_stories_stage_check
  check (stage in ('interview', 'outline', 'outline_approved', 'novel_approved'));
