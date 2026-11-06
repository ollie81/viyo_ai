-- Viyo Studio, stage 1: permanent character voices.
-- Run after creator_studio_schema.sql (and creator_studio_breakdown.sql).
-- Safe to run more than once. Only adds columns and tables.

-- The voice a character speaks with is part of the character record.
-- `voice` (already there) holds the provider's voice id, e.g. "Kore".
alter table public.studio_characters add column if not exists voice_provider text not null default 'google';
alter table public.studio_characters add column if not exists voice_style text not null default '';  -- one short delivery note, used for every line
alter table public.studio_characters add column if not exists voice_language text;                   -- the language it was set up for
alter table public.studio_characters add column if not exists voice_locked_at timestamptz;          -- first time a line was recorded with it
alter table public.studio_characters add column if not exists voice_changed_at timestamptz;
-- The same person across episodes: copies of a character in later episodes
-- point at the first one.
alter table public.studio_characters add column if not exists identity_id uuid;
alter table public.studio_characters drop constraint if exists studio_characters_voice_style_len;
alter table public.studio_characters add constraint studio_characters_voice_style_len check (char_length(voice_style) <= 80);

-- Which voice each character used in a scene's recording:
-- {"<character id>": "Kore", "narrator": "Charon"}
alter table public.studio_scenes add column if not exists audio_voices jsonb not null default '{}'::jsonb;

-- Every deliberate voice change, kept for support and for the creator's history.
create table if not exists public.studio_voice_changes (
  id uuid primary key default gen_random_uuid(),
  character_id uuid not null references public.studio_characters(id) on delete cascade,
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  old_voice text not null default '',
  new_voice text not null,
  old_style text not null default '',
  new_style text not null default '',
  scenes_affected int not null default 0,
  changed_at timestamptz not null default now()
);
create index if not exists studio_voice_changes_character_idx on public.studio_voice_changes (character_id, changed_at desc);

alter table public.studio_voice_changes enable row level security;
drop policy if exists "owner reads" on public.studio_voice_changes;
create policy "owner reads" on public.studio_voice_changes for select using (auth.uid() = owner_id);

-- A short sample of each voice, made once by an admin, so creators can
-- listen before choosing. Files live in the private creator-studio bucket.
create table if not exists public.studio_voice_samples (
  voice text primary key,
  path text not null,
  seconds numeric(6,2),
  created_at timestamptz not null default now()
);
alter table public.studio_voice_samples enable row level security;

create index if not exists studio_projects_series_idx on public.studio_projects (owner_id, series_id);

-- The scene-list function from creator_studio_breakdown.sql, now also saving
-- the voice details and the cross-episode identity of new characters.
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
                                        voice_style, voice_language, identity_id, reference_path)
  select p_project, p_owner, x.name, coalesce(x.description, ''), coalesce(x.voice, ''), coalesce(x.position, 0),
         coalesce(x.voice_style, ''), x.voice_language, x.identity_id, x.reference_path
    from jsonb_to_recordset(coalesce(p_characters, '[]'::jsonb))
         as x(name text, description text, voice text, position int,
              voice_style text, voice_language text, identity_id uuid, reference_path text)
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
