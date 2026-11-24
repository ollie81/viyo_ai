-- Points saved file links at Bunny instead of Supabase Storage. Run AFTER the worker log says the copy finished
-- and FILE_STORE=bunny is set, and BEFORE deleting anything from Supabase.
--
-- Replace the two values at the top of each block:
--   supabase_base : your project URL + /storage/v1/object/public/   e.g. https://abcd.supabase.co/storage/v1/object/public/
--   bunny_base    : your PUBLIC pull zone URL + /                    e.g. https://viyo-public.b-cdn.net/
-- Only the public buckets are changed (the same file path exists on Bunny). Private files are stored as paths,
-- not links, so they need nothing. Safe to run more than once.

-- 1) Count first (changes nothing): which columns hold Supabase file links, and how many rows.
do $$
declare
  supabase_base text := 'https://YOUR-PROJECT.supabase.co/storage/v1/object/public/';
  c record; n bigint;
begin
  for c in
    select table_schema, table_name, column_name, data_type
      from information_schema.columns
     where table_schema = 'public'
       and data_type in ('text', 'character varying', 'jsonb', 'json', 'ARRAY')
       and table_name in (select table_name from information_schema.tables
                           where table_schema = 'public' and table_type = 'BASE TABLE')
  loop
    execute format('select count(*) from %I.%I where %I::text like %L',
                   c.table_schema, c.table_name, c.column_name, '%' || supabase_base || '%') into n;
    if n > 0 then
      raise notice '%.%: % rows', c.table_name, c.column_name, n;
    end if;
  end loop;
end $$;

-- 2) Rewrite.
do $$
declare
  supabase_base text := 'https://YOUR-PROJECT.supabase.co/storage/v1/object/public/';
  bunny_base    text := 'https://YOUR-PUBLIC-ZONE.b-cdn.net/';
  buckets text[] := array['posts-media', 'avatars', 'studio-images', 'studio-audio', 'studio-videos',
                          'processed-videos'];
  c record; b text; n bigint; expr text;
begin
  if supabase_base like '%YOUR-PROJECT%' or bunny_base like '%YOUR-PUBLIC-ZONE%' then
    raise exception 'Fill in supabase_base and bunny_base first.';
  end if;
  for c in
    select table_schema, table_name, column_name, data_type, udt_name
      from information_schema.columns
     where table_schema = 'public'
       and data_type in ('text', 'character varying', 'jsonb', 'json', 'ARRAY')
       and table_name in (select table_name from information_schema.tables
                           where table_schema = 'public' and table_type = 'BASE TABLE')
  loop
    foreach b in array buckets loop
      expr := format('replace(%I::text, %L, %L)', c.column_name, supabase_base || b || '/', bunny_base || b || '/');
      if c.data_type = 'ARRAY' then
        expr := format('(%s)::%s', expr, c.udt_name::regtype);
      elsif c.data_type in ('jsonb', 'json') then
        expr := format('(%s)::%s', expr, c.data_type);
      end if;
      execute format('update %I.%I set %I = %s where %I::text like %L',
                     c.table_schema, c.table_name, c.column_name, expr, c.column_name,
                     '%' || supabase_base || b || '/%');
      get diagnostics n = row_count;
      if n > 0 then
        raise notice '%.% (%): % rows', c.table_name, c.column_name, b, n;
      end if;
    end loop;
  end loop;
end $$;
