-- Viyo Studio: whole-production picture audits on their own queue.
-- Run after creator_studio_assistant_safety.sql. Safe to run more than once. Only adds.
--
-- Picture checks ("inspect" jobs) get their own lane, so checking a 100-scene
-- film one scene at a time never holds up making pictures, videos or voices.
-- Every lane is listed, so re-running this file never narrows it.
alter table public.studio_jobs drop constraint if exists studio_jobs_lane_check;
alter table public.studio_jobs add constraint studio_jobs_lane_check
  check (lane in ('generation', 'render', 'inspect'));
