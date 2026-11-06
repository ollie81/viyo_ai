-- Viyo Studio: automatic voice casting.
-- Run after creator_studio_voices.sql. Safe to run more than once. Only adds.
--
-- 1. A catalog of the voices Viyo may cast, with what has actually been
--    verified about each one. Casting only picks voices whose age group is
--    listed here, so a child is never given an adult voice by accident.
-- 2. The character profile the casting engine works from (age, gender
--    presentation, personality) and what it decided (voice category,
--    a permanent voice description, a status), kept on the character.

create table if not exists public.studio_voice_catalog (
  provider text not null,
  voice_id text not null,
  gender text not null default 'unspecified' check (gender in ('female', 'male', 'unspecified')),
  descriptor text not null default '',                       -- the provider's own one-word character
  -- Age groups this voice has been checked to suit. Only these are cast.
  age_categories jsonb not null default '[]'::jsonb,
  -- 'all' = every language the provider's model speaks; otherwise a list.
  languages jsonb not null default '["all"]'::jsonb,
  status text not null default 'available' check (status in ('available', 'unavailable')),
  verified_by text not null default '',                       -- 'provider_docs' or the admin who listened
  verified_at timestamptz,
  notes text not null default '',
  updated_at timestamptz not null default now(),
  primary key (provider, voice_id)
);
alter table public.studio_voice_catalog enable row level security;

-- Google's 30 prebuilt Gemini voices. Google documents them as adult
-- speaking voices; nothing in the docs says any of them suits a child,
-- a teenager or an elderly person, so only the adult groups are recorded.
-- An admin can add more groups after listening (internal voice gallery).
insert into public.studio_voice_catalog (provider, voice_id, gender, descriptor, age_categories, verified_by, verified_at, notes)
select v.provider, v.voice_id, v.gender, v.descriptor, '["young_adult", "middle_aged"]'::jsonb, 'provider_docs', now(),
       'Gemini prebuilt voice. Adult groups from Google''s documentation; other ages not evaluated.'
  from (values
  ('google', 'Zephyr', 'female', 'Bright'),
  ('google', 'Puck', 'male', 'Upbeat'),
  ('google', 'Charon', 'male', 'Informative'),
  ('google', 'Kore', 'female', 'Firm'),
  ('google', 'Fenrir', 'male', 'Excitable'),
  ('google', 'Leda', 'female', 'Youthful'),
  ('google', 'Orus', 'male', 'Firm'),
  ('google', 'Aoede', 'female', 'Breezy'),
  ('google', 'Callirrhoe', 'female', 'Easy-going'),
  ('google', 'Autonoe', 'female', 'Bright'),
  ('google', 'Enceladus', 'male', 'Breathy'),
  ('google', 'Iapetus', 'male', 'Clear'),
  ('google', 'Umbriel', 'male', 'Easy-going'),
  ('google', 'Algieba', 'male', 'Smooth'),
  ('google', 'Despina', 'female', 'Smooth'),
  ('google', 'Erinome', 'female', 'Clear'),
  ('google', 'Algenib', 'male', 'Gravelly'),
  ('google', 'Rasalgethi', 'male', 'Informative'),
  ('google', 'Laomedeia', 'female', 'Upbeat'),
  ('google', 'Achernar', 'female', 'Soft'),
  ('google', 'Alnilam', 'male', 'Firm'),
  ('google', 'Schedar', 'male', 'Even'),
  ('google', 'Gacrux', 'female', 'Mature'),
  ('google', 'Pulcherrima', 'female', 'Forward'),
  ('google', 'Achird', 'male', 'Friendly'),
  ('google', 'Zubenelgenubi', 'male', 'Casual'),
  ('google', 'Vindemiatrix', 'female', 'Gentle'),
  ('google', 'Sadachbia', 'male', 'Lively'),
  ('google', 'Sadaltager', 'male', 'Knowledgeable'),
  ('google', 'Sulafat', 'female', 'Warm')
  ) as v(provider, voice_id, gender, descriptor)
on conflict (provider, voice_id) do nothing;

alter table public.studio_characters add column if not exists age_category text;          -- infant .. elderly, or 'unknown'
alter table public.studio_characters add column if not exists age_years int;               -- only when the story states it
alter table public.studio_characters add column if not exists age_confidence text;         -- high / medium / low
alter table public.studio_characters add column if not exists gender_presentation text;    -- female / male / unspecified
alter table public.studio_characters add column if not exists personality text not null default '';
alter table public.studio_characters add column if not exists voice_category text;         -- e.g. adult_female, child_male
alter table public.studio_characters add column if not exists voice_description text not null default '';  -- permanent, reused
-- cast       = a suitable verified voice was chosen
-- approximate = the closest available voice (e.g. an adult voice for an elderly character); shown to the creator
-- needs_voice = no suitable voice exists yet (e.g. a child): lines aren't voiced until one is added
-- unsupported_language = the voice model doesn't speak the project's language
-- manual     = chosen by the creator
alter table public.studio_characters add column if not exists voice_status text;
alter table public.studio_characters add column if not exists voice_assignment_version int not null default 0;

alter table public.studio_voice_changes add column if not exists reason text not null default 'creator';  -- auto_cast / creator / admin / series
alter table public.studio_voice_changes alter column new_voice drop not null;

-- The scene-list function again (from creator_studio_voices.sql), now also
-- saving each new character's age profile and casting decision in the same
-- single step.
create or replace function public.studio_apply_breakdown(
  p_project uuid, p_owner uuid, p_job uuid,
  p_characters jsonb, p_describe jsonb, p_keep jsonb, p_insert jsonb, p_archive jsonb)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_archived int := 0;
  v_kept int := 0;
  v_added int := 0;
  v_chars int := 0;
begin
  perform 1 from public.studio_projects where id = p_project and owner_id = p_owner for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'project_not_found');
  end if;

  insert into public.studio_characters (project_id, owner_id, name, description, voice, position,
                                        voice_style, voice_language, identity_id, reference_path,
                                        voice_provider, age_category, age_years, age_confidence, gender_presentation,
                                        personality, voice_category, voice_description, voice_status,
                                        voice_assignment_version)
  select p_project, p_owner, x.name, coalesce(x.description, ''), coalesce(x.voice, ''), coalesce(x.position, 0),
         coalesce(x.voice_style, ''), x.voice_language, x.identity_id, x.reference_path,
         coalesce(x.voice_provider, 'google'), x.age_category, x.age_years, x.age_confidence, x.gender_presentation,
         coalesce(x.personality, ''), x.voice_category, coalesce(x.voice_description, ''), x.voice_status,
         coalesce(x.voice_assignment_version, 0)
    from jsonb_to_recordset(coalesce(p_characters, '[]'::jsonb))
         as x(name text, description text, voice text, position int,
              voice_style text, voice_language text, identity_id uuid, reference_path text,
              voice_provider text, age_category text, age_years int, age_confidence text, gender_presentation text,
              personality text, voice_category text, voice_description text, voice_status text,
              voice_assignment_version int)
   where coalesce(trim(x.name), '') <> ''
     and not exists (select 1 from public.studio_characters c
                      where c.project_id = p_project and lower(trim(c.name)) = lower(trim(x.name)));
  get diagnostics v_chars = row_count;

  update public.studio_characters c set description = x.description
    from jsonb_to_recordset(coalesce(p_describe, '[]'::jsonb)) as x(id uuid, description text)
   where c.id = x.id and c.project_id = p_project and coalesce(c.description, '') = '';

  insert into public.studio_scene_archive (scene_id, project_id, owner_id, job_id, reason,
                                           image_path, video_path, audio_path, scene)
  select s.id, s.project_id, s.owner_id, p_job, 'breakdown', s.image_path, s.video_path, s.audio_path, to_jsonb(s)
    from public.studio_scenes s
   where s.project_id = p_project
     and s.id::text in (select jsonb_array_elements_text(coalesce(p_archive, '[]'::jsonb)));
  get diagnostics v_archived = row_count;

  delete from public.studio_scenes s
   where s.project_id = p_project
     and s.id::text in (select jsonb_array_elements_text(coalesce(p_archive, '[]'::jsonb)));

  update public.studio_scenes s
     set position = x.position,
         visual = coalesce(x.visual, s.visual),
         duration_seconds = coalesce(x.duration_seconds, s.duration_seconds),
         updated_at = now()
    from jsonb_to_recordset(coalesce(p_keep, '[]'::jsonb)) as x(id uuid, position int, visual text, duration_seconds numeric)
   where s.id = x.id and s.project_id = p_project;
  get diagnostics v_kept = row_count;

  insert into public.studio_scenes (project_id, owner_id, position, heading, visual, dialogue, duration_seconds,
                                    mode, image_path, video_path)
  select p_project, p_owner, x.position, coalesce(x.heading, ''), coalesce(x.visual, ''), coalesce(x.dialogue, '[]'::jsonb),
         coalesce(x.duration_seconds, 5), coalesce(x.mode, 'still'), x.image_path, x.video_path
    from jsonb_to_recordset(coalesce(p_insert, '[]'::jsonb))
         as x(position int, heading text, visual text, dialogue jsonb, duration_seconds numeric,
              mode text, image_path text, video_path text);
  get diagnostics v_added = row_count;

  update public.studio_projects set updated_at = now() where id = p_project;
  return jsonb_build_object('ok', true, 'kept', v_kept, 'added', v_added, 'archived', v_archived,
                            'characters_added', v_chars);
end;
$$;

revoke all on function public.studio_apply_breakdown(uuid, uuid, uuid, jsonb, jsonb, jsonb, jsonb, jsonb)
  from public, anon, authenticated;
grant execute on function public.studio_apply_breakdown(uuid, uuid, uuid, jsonb, jsonb, jsonb, jsonb, jsonb)
  to service_role;
