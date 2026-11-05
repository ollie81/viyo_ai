-- Viyo Studio, stage 0: a new scene list never deletes paid work.
-- Run after creator_studio_schema.sql. Safe to run more than once.
--
-- Before this, making a new scene list deleted every scene, along with the
-- links to their pictures and voice recordings. Now the worker works out a
-- plan (scenes to keep, scenes to add, scenes to set aside) and this
-- function applies the whole plan in one transaction: it either all
-- happens or none of it does. Scenes that are set aside are copied here
-- in full first (pictures, voice and everything else) and can be restored.

create table if not exists public.studio_scene_archive (
  id uuid primary key default gen_random_uuid(),
  scene_id uuid not null,                        -- the scene's old id
  project_id uuid not null references public.studio_projects(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  job_id uuid,                                   -- the breakdown job that replaced it
  reason text not null default 'breakdown',
  image_path text,
  video_path text,
  audio_path text,
  scene jsonb not null,                          -- the full row as it was
  archived_at timestamptz not null default now()
);
create index if not exists studio_scene_archive_project_idx on public.studio_scene_archive (project_id, archived_at desc);

alter table public.studio_scene_archive enable row level security;
drop policy if exists "owner reads" on public.studio_scene_archive;
create policy "owner reads" on public.studio_scene_archive for select using (auth.uid() = owner_id);

-- p_characters: [{name, description, voice, position}]  added if the name is new
-- p_describe:   [{id, description}]                    fills empty descriptions
-- p_keep:       [{id, position, visual?, duration_seconds?}]
-- p_insert:     [{position, heading, visual, dialogue, duration_seconds, mode, image_path, video_path}]
-- p_archive:    ["scene id", ...]                       copied to the archive, then removed
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

  insert into public.studio_characters (project_id, owner_id, name, description, voice, position)
  select p_project, p_owner, x.name, coalesce(x.description, ''), coalesce(x.voice, ''), coalesce(x.position, 0)
    from jsonb_to_recordset(coalesce(p_characters, '[]'::jsonb)) as x(name text, description text, voice text, position int)
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
