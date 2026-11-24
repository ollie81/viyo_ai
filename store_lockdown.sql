-- Coin Store lockdown. The app no longer opens the Store (it sold a "verified" check and achievement badges for
-- coins). Its two purchase functions live only in the database and take the user id, and for Premium the PRICE,
-- from the app, so a caller could pass someone else's id or a price of 0 unless the functions check. This removes
-- the ability of app users and the public key to call them at all. Nothing is deleted: owned badges, is_premium
-- and the functions themselves stay; the backend (service role) can still call them. Safe to run more than once.
do $$
declare f record;
begin
  for f in
    select p.oid::regprocedure as sig
      from pg_proc p join pg_namespace n on n.oid = p.pronamespace
     where n.nspname = 'public' and p.proname in ('purchase_premium', 'purchase_badge')
  loop
    execute format('revoke execute on function %s from public, anon, authenticated', f.sig);
    raise notice 'locked %', f.sig;
  end loop;
end $$;
