-- Viyo Studio: applying an assistant suggestion exactly once.
-- Run after creator_studio_inspection.sql. Safe to run more than once. Only adds.
--
-- studio_claim_proposal: marks one suggestion of one message as 'applying' only
-- if it is still 'proposed' and belongs to the owner, in one statement. Two taps
-- (or two devices) at the same moment: only one wins, the other gets "already
-- used", so a free change is never applied twice.
create or replace function public.studio_claim_proposal(p_message uuid, p_owner uuid, p_index int)
returns boolean language plpgsql security definer set search_path = public as $$
declare
  v_rows int;
begin
  update public.studio_messages
     set proposals = jsonb_set(proposals, array[p_index::text, 'status'], '"applying"'::jsonb)
   where id = p_message and owner_id = p_owner
     and jsonb_typeof(proposals) = 'array'
     and proposals -> p_index ->> 'status' = 'proposed';
  get diagnostics v_rows = row_count;
  return v_rows = 1;
end;
$$;

revoke all on function public.studio_claim_proposal(uuid, uuid, int) from public, anon, authenticated;
grant execute on function public.studio_claim_proposal(uuid, uuid, int) to service_role;
