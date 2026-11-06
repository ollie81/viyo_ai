-- Viyo Studio: talking-video dramas.
-- Run after creator_studio_casting.sql. Safe to run more than once. Only adds.
--
-- production_style: 'video' = every scene is a moving video clip in which the
-- characters speak their lines (Gemini Omni); 'pictures' = still images
-- with camera movement and the characters' saved voices (the cheaper way,
-- and how existing productions were made).
alter table public.studio_projects add column if not exists production_style text not null default 'pictures';
alter table public.studio_projects drop constraint if exists studio_projects_production_style_check;
alter table public.studio_projects add constraint studio_projects_production_style_check
  check (production_style in ('pictures', 'video'));

-- Who speaks a scene's lines. Empty = follow the production style.
-- 'video_model' = the characters talk inside the video clip (lips move);
-- 'saved_voice' = the character's permanent saved voice is laid over the
-- scene, and the video's own sound is not used under it;
-- 'lipsync'     = the saved voice, with the mouth made to match it (sync.so).
alter table public.studio_scenes add column if not exists speech text;
alter table public.studio_scenes drop constraint if exists studio_scenes_speech_check;
alter table public.studio_scenes add constraint studio_scenes_speech_check
  check (speech is null or speech in ('video_model', 'saved_voice', 'lipsync'));

-- The lip-synced version of a scene's clip (the original clip is kept).
alter table public.studio_scenes add column if not exists lipsync_path text;

-- The lip-sync job type.
alter table public.studio_jobs drop constraint if exists studio_jobs_kind_check;
alter table public.studio_jobs add constraint studio_jobs_kind_check
  check (kind in ('script', 'breakdown', 'portrait', 'image', 'voice', 'video', 'render', 'lipsync'));
