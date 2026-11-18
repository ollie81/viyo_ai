-- Deleting an account: the creator may choose to keep their dramas on Viyo.
--
-- When a creator deletes their account they choose between
--   "Delete everything" (the default, as before), and
--   "Keep my dramas on Viyo": their series and the episodes in them move to one
--   shared "Former creator" account. Their name and profile are gone; viewers keep
--   the episodes they unlocked. Everything else is deleted as before.
--
-- Only dramas the creator owns move, and only when they choose it. Additive only,
-- safe to run more than once. Changes no existing data.

-- Viyo's own system accounts (one row: the "Former creator" account, created by the backend on first use).
create table if not exists public.viyo_system_accounts (
  role text primary key,
  user_id uuid not null references auth.users(id),
  created_at timestamptz not null default now()
);
alter table public.viyo_system_accounts enable row level security;   -- no policies: backend only

-- How many dramas (series) and episodes this user owns: decides whether the app offers the choice.
create or replace function public.viyo_account_dramas(p_user_id uuid)
returns jsonb language sql stable security definer set search_path = public as $$
  select jsonb_build_object(
    'dramas', (select count(*) from public.series where user_id = p_user_id),
    'episodes', (select count(*) from public.posts p
                  where p.user_id = p_user_id
                    and p.series_id in (select id from public.series where user_id = p_user_id)));
$$;

-- Moves the user's series, and the episodes in them, to the Former creator account, in one step.
-- Returns {ok, series_ids, dramas, episodes}. Posts that aren't episodes of the user's own series stay
-- with the user and are deleted with the account.
create or replace function public.viyo_keep_dramas(p_user_id uuid, p_former_id uuid)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  v_series uuid[];
  v_episodes int;
begin
  if p_former_id is null or p_former_id = p_user_id then
    return jsonb_build_object('ok', false, 'error', 'bad_former_account');
  end if;
  select coalesce(array_agg(id), '{}') into v_series from public.series where user_id = p_user_id;
  update public.posts set user_id = p_former_id
   where user_id = p_user_id and series_id = any(v_series);
  get diagnostics v_episodes = row_count;
  update public.series set user_id = p_former_id where id = any(v_series);
  return jsonb_build_object('ok', true, 'series_ids', to_jsonb(v_series),
                            'dramas', coalesce(array_length(v_series, 1), 0), 'episodes', v_episodes);
end;
$$;

revoke all on function public.viyo_account_dramas(uuid) from public, anon, authenticated;
revoke all on function public.viyo_keep_dramas(uuid, uuid) from public, anon, authenticated;
grant execute on function public.viyo_account_dramas(uuid) to service_role;
grant execute on function public.viyo_keep_dramas(uuid, uuid) to service_role;

-- Check (read-only):
-- select * from public.viyo_system_accounts;
