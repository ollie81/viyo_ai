-- Notifications: stop duplicate like/comment alerts, allow the new types.
-- Run manually in the Supabase SQL editor. Safe to re-run.

-- 1) Every like (and comment) created two notifications: one from the
--    backend ("Ada liked your post") and a vaguer one ("New activity")
--    from an old database trigger. The backend one carries the post and
--    a push, so remove only triggers on likes/comments that write to
--    notifications. Nothing else is touched.
do $$
declare
  r record;
begin
  for r in
    select t.tgname, c.relname
      from pg_trigger t
      join pg_class c on c.oid = t.tgrelid
      join pg_namespace n on n.oid = c.relnamespace
      join pg_proc p on p.oid = t.tgfoid
     where n.nspname = 'public'
       and c.relname in ('likes', 'comments')
       and not t.tgisinternal
       and pg_get_functiondef(p.oid) ilike '%notifications%'
  loop
    execute format('drop trigger if exists %I on public.%I', r.tgname, r.relname);
    raise notice 'dropped trigger % on %', r.tgname, r.relname;
  end loop;
end $$;

-- 2) New notification types (unlock, payout_paid, payout_rejected,
--    clips_ready, clips_failed): drop any old check that only allowed the
--    original few types.
do $$
declare
  r record;
begin
  for r in
    select con.conname
      from pg_constraint con
      join pg_class c on c.oid = con.conrelid
      join pg_namespace n on n.oid = c.relnamespace
     where n.nspname = 'public' and c.relname = 'notifications' and con.contype = 'c'
       and pg_get_constraintdef(con.oid) ilike '%type%'
  loop
    execute format('alter table public.notifications drop constraint %I', r.conname);
    raise notice 'dropped check %', r.conname;
  end loop;
end $$;

-- 3) Fast unread counts for the bell badge.
create index if not exists notifications_user_unread_idx
  on public.notifications (user_id) where is_read = false;
