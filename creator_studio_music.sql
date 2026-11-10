-- Viyo Studio: automatic background music.
-- Run after creator_studio_chat.sql. Safe to run more than once. Only adds.
--
-- auto_music: when the creator hasn't uploaded their own track, the film
-- render makes an original instrumental score (Google Lyria, about $0.08 a
-- song, shown in the price). Creators can turn it off per project.
alter table public.studio_projects add column if not exists auto_music boolean not null default true;
