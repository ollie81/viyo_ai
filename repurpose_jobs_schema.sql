-- AI Repurposer job infrastructure (Phase 0).
--
-- Run by hand in the Supabase SQL editor, top to bottom, once. Safe to
-- re-run: every statement is idempotent (if not exists / or replace /
-- on conflict).
--
-- What this replaces: repurpose jobs used to live as JSON files in the
-- processed-videos bucket and run as FastAPI BackgroundTasks inside the
-- API process, so a redeploy silently killed in-flight jobs (coins never
-- refunded) and results vanished after an hour. Now:
--   * jobs and clips are rows here, processed by a separate worker
--     service that claims them with FOR UPDATE SKIP LOCKED (so any
--     number of worker replicas can run side by side);
--   * charging, refunding and every job state change happen inside one
--     database function each, so a job can never be charged twice,
--     refunded twice, or charged without existing;
--   * source uploads go to a PRIVATE bucket through server-issued signed
--     upload URLs, instead of the public posts-media bucket.
--
-- Every table has RLS enabled with no policies: only the backend's
-- service role (which bypasses RLS) can read or write them. Every
-- function is SECURITY DEFINER and executable by service_role only —
-- p_cost is trusted input, so these must never be callable by a client.
--
-- Assumes the existing tables the backend already writes to:
--   profiles(id uuid, points_balance int)
--   transactions(user_id, amount, type, description) — other columns
--   must have defaults, same as for coins.py's inserts today.

-- ---------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------

create table if not exists public.repurpose_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  -- Client-generated per "Create clips" tap. A retried POST with the
  -- same value returns the existing job instead of charging again.
  client_request_id text,

  status text not null default 'queued'
    check (status in ('queued', 'processing', 'succeeded', 'failed')),
  stage text not null default 'queued',
  progress real not null default 0 check (progress >= 0 and progress <= 1),
  status_message text,

  source_kind text not null
    check (source_kind in ('private_upload', 'legacy_public_url')),
  source_bucket text,
  source_path text,
  source_url text,
  source_sha256 text,
  source_bytes bigint,
  source_duration_seconds real,
  creative_prompt text,

  coins_charged int not null default 0 check (coins_charged >= 0),
  refunded_at timestamptz,
  refund_amount int,

  attempts int not null default 0,
  max_attempts int not null default 3 check (max_attempts >= 1),
  -- Fencing token: a new value on every claim. Every write a worker
  -- makes is conditioned on it, so a worker that lost its lease (stalled,
  -- then reclaimed by another) can't overwrite the new owner's work.
  lease_token uuid,
  locked_by text,
  heartbeat_at timestamptz,
  next_attempt_at timestamptz not null default now(),

  error_code text,
  error text,          -- shown to the creator
  last_error text,     -- internal detail of the latest failed attempt

  -- Per-stage outputs (transcript reference, clip selection) so a retry
  -- resumes from the last finished stage instead of redoing it.
  checkpoint jsonb not null default '{}'::jsonb,
  -- Final payload in the original RepurposeResponse shape, so app
  -- versions that predate this table keep working unchanged.
  result jsonb,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz,
  expires_at timestamptz,
  source_deleted_at timestamptz,

  constraint repurpose_jobs_client_request_unique unique (user_id, client_request_id)
);

create index if not exists repurpose_jobs_queue_idx
  on public.repurpose_jobs (next_attempt_at, created_at) where status = 'queued';
create index if not exists repurpose_jobs_processing_idx
  on public.repurpose_jobs (heartbeat_at) where status = 'processing';
create index if not exists repurpose_jobs_user_created_idx
  on public.repurpose_jobs (user_id, created_at desc);
create index if not exists repurpose_jobs_expires_idx
  on public.repurpose_jobs (expires_at) where expires_at is not null;
create index if not exists repurpose_jobs_source_cleanup_idx
  on public.repurpose_jobs (finished_at)
  where source_deleted_at is null and source_kind = 'private_upload';
create index if not exists repurpose_jobs_unrefunded_idx
  on public.repurpose_jobs (finished_at)
  where status = 'failed' and refunded_at is null and coins_charged > 0;

create table if not exists public.repurpose_clips (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null references public.repurpose_jobs(id) on delete cascade,
  user_id uuid not null,
  rank int not null check (rank >= 1),
  status text not null default 'selected'
    check (status in ('selected', 'rendered', 'failed')),
  start_time real not null,
  end_time real not null,
  score int check (score between 0 and 100),
  score_breakdown jsonb,
  category text,
  title text,
  reason text,
  hook_line text,
  caption text,
  hashtags text[] not null default '{}',
  transcript text,
  video_path text,
  video_url text,
  thumbnail_path text,
  thumbnail_url text,
  quote_card_path text,
  quote_card_url text,
  render_meta jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint repurpose_clips_time_order check (end_time > start_time),
  constraint repurpose_clips_job_rank_unique unique (job_id, rank)
);

create index if not exists repurpose_clips_job_idx on public.repurpose_clips (job_id);

-- Transcript cache keyed by the source file's SHA-256, so a retried or
-- re-submitted video is never transcribed (and billed) twice.
create table if not exists public.repurpose_transcripts (
  source_sha256 text not null,
  model text not null,
  duration_seconds real,
  language text,
  text text not null default '',
  segments jsonb not null,
  words jsonb not null,
  created_at timestamptz not null default now(),
  last_used_at timestamptz not null default now(),
  primary key (source_sha256, model)
);

create index if not exists repurpose_transcripts_last_used_idx
  on public.repurpose_transcripts (last_used_at);

-- Every signed upload URL the backend issues, so an upload that never
-- became a job can still be found and deleted.
create table if not exists public.repurpose_uploads (
  source_path text primary key,
  bucket text not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  declared_bytes bigint,
  content_type text,
  job_id uuid references public.repurpose_jobs(id) on delete set null,
  created_at timestamptz not null default now(),
  deleted_at timestamptz
);

create index if not exists repurpose_uploads_orphan_idx
  on public.repurpose_uploads (created_at) where job_id is null and deleted_at is null;

alter table public.repurpose_jobs enable row level security;
alter table public.repurpose_clips enable row level security;
alter table public.repurpose_transcripts enable row level security;
alter table public.repurpose_uploads enable row level security;

create or replace function public.repurpose_touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists repurpose_jobs_touch on public.repurpose_jobs;
create trigger repurpose_jobs_touch before update on public.repurpose_jobs
  for each row execute function public.repurpose_touch_updated_at();

drop trigger if exists repurpose_clips_touch on public.repurpose_clips;
create trigger repurpose_clips_touch before update on public.repurpose_clips
  for each row execute function public.repurpose_touch_updated_at();

-- ---------------------------------------------------------------
-- Functions
-- ---------------------------------------------------------------

-- Charges the creator and creates the job in one transaction.
-- Returns {"ok": true, "job_id", "created", "coins_charged"} or
-- {"ok": false, "error": "insufficient_coins" | "daily_limit" |
--  "profile_not_found" | "unknown_upload" | "upload_already_used", ...}.
create or replace function public.repurpose_create_job(
  p_user_id uuid,
  p_cost int,
  p_daily_limit int,
  p_max_attempts int,
  p_source_kind text,
  p_source_bucket text,
  p_source_path text,
  p_source_url text,
  p_creative_prompt text,
  p_client_request_id text
) returns jsonb
language plpgsql security definer set search_path = public as $$
declare
  v_balance int;
  v_existing public.repurpose_jobs%rowtype;
  v_upload public.repurpose_uploads%rowtype;
  v_recent int;
  v_job_id uuid;
begin
  if p_cost < 0 then
    raise exception 'p_cost must be >= 0';
  end if;

  -- Serializes every create for this user, which is what makes the
  -- idempotency check and the daily limit below race-free.
  select points_balance into v_balance
    from public.profiles where id = p_user_id for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'profile_not_found');
  end if;

  if p_client_request_id is not null then
    select * into v_existing from public.repurpose_jobs
      where user_id = p_user_id and client_request_id = p_client_request_id;
    if found then
      return jsonb_build_object(
        'ok', true, 'job_id', v_existing.id, 'created', false,
        'coins_charged', v_existing.coins_charged);
    end if;
  end if;

  select count(*) into v_recent from public.repurpose_jobs
    where user_id = p_user_id and created_at > now() - interval '24 hours';
  if v_recent >= p_daily_limit then
    return jsonb_build_object('ok', false, 'error', 'daily_limit', 'limit', p_daily_limit);
  end if;

  if p_source_kind = 'private_upload' then
    select * into v_upload from public.repurpose_uploads
      where source_path = p_source_path for update;
    if not found or v_upload.user_id <> p_user_id or v_upload.bucket <> p_source_bucket
       or v_upload.deleted_at is not null then
      return jsonb_build_object('ok', false, 'error', 'unknown_upload');
    end if;
    if v_upload.job_id is not null then
      return jsonb_build_object('ok', false, 'error', 'upload_already_used');
    end if;
  elsif p_source_kind <> 'legacy_public_url' then
    raise exception 'unknown source kind %', p_source_kind;
  end if;

  if coalesce(v_balance, 0) < p_cost then
    return jsonb_build_object(
      'ok', false, 'error', 'insufficient_coins',
      'balance', coalesce(v_balance, 0), 'needed', p_cost);
  end if;

  insert into public.repurpose_jobs (
    user_id, client_request_id, source_kind, source_bucket, source_path,
    source_url, creative_prompt, coins_charged, max_attempts
  ) values (
    p_user_id, p_client_request_id, p_source_kind, p_source_bucket, p_source_path,
    p_source_url, p_creative_prompt, p_cost, p_max_attempts
  ) returning id into v_job_id;

  if p_cost > 0 then
    update public.profiles set points_balance = points_balance - p_cost where id = p_user_id;
    insert into public.transactions (user_id, amount, type, description)
      values (p_user_id, -p_cost, 'repurpose', 'Used AI Repurposer');
  end if;

  if p_source_kind = 'private_upload' then
    update public.repurpose_uploads set job_id = v_job_id where source_path = p_source_path;
  end if;

  return jsonb_build_object('ok', true, 'job_id', v_job_id, 'created', true, 'coins_charged', p_cost);
end;
$$;

-- Hands the oldest runnable job to a worker. Runnable = queued and due,
-- or "processing" whose worker stopped heartbeating (crashed, OOM,
-- killed mid-redeploy). A creator with a job already actively running
-- waits, so one person queuing many videos can't starve everyone else.
create or replace function public.repurpose_claim_job(
  p_worker_id text,
  p_stale_seconds int
) returns jsonb
language plpgsql security definer set search_path = public as $$
declare
  v_job public.repurpose_jobs%rowtype;
begin
  select j.* into v_job
    from public.repurpose_jobs j
   where j.attempts < j.max_attempts
     and (
       (j.status = 'queued' and j.next_attempt_at <= now())
       or (j.status = 'processing'
           and j.heartbeat_at < now() - make_interval(secs => p_stale_seconds))
     )
     and not exists (
       select 1 from public.repurpose_jobs other
        where other.user_id = j.user_id
          and other.id <> j.id
          and other.status = 'processing'
          and other.heartbeat_at >= now() - make_interval(secs => p_stale_seconds)
     )
   order by j.next_attempt_at, j.created_at
   for update of j skip locked
   limit 1;

  if not found then
    return null;
  end if;

  update public.repurpose_jobs
     set status = 'processing',
         attempts = attempts + 1,
         lease_token = gen_random_uuid(),
         locked_by = p_worker_id,
         heartbeat_at = now(),
         started_at = coalesce(started_at, now())
   where id = v_job.id
   returning * into v_job;

  return to_jsonb(v_job);
end;
$$;

-- Liveness + progress, and optionally a checkpoint merge. Returns false
-- when the caller no longer holds the lease — it must stop working.
create or replace function public.repurpose_heartbeat(
  p_job_id uuid,
  p_lease uuid,
  p_stage text default null,
  p_progress double precision default null,
  p_message text default null,
  p_checkpoint jsonb default null
) returns boolean
language plpgsql security definer set search_path = public as $$
begin
  update public.repurpose_jobs
     set heartbeat_at = now(),
         stage = coalesce(p_stage, stage),
         progress = coalesce(greatest(0, least(1, p_progress)), progress),
         status_message = coalesce(p_message, status_message),
         checkpoint = checkpoint || coalesce(p_checkpoint, '{}'::jsonb)
   where id = p_job_id and lease_token = p_lease and status = 'processing';
  return found;
end;
$$;

-- Exactly-once refund of a failed job's charge. Safe to call any number
-- of times: only the first call for a failed, unrefunded, charged job
-- moves coins.
create or replace function public.repurpose_refund_job(p_job_id uuid)
returns int
language plpgsql security definer set search_path = public as $$
declare
  v_user uuid;
  v_amount int;
begin
  update public.repurpose_jobs
     set refunded_at = now(), refund_amount = coins_charged
   where id = p_job_id and status = 'failed'
     and refunded_at is null and coins_charged > 0
   returning user_id, coins_charged into v_user, v_amount;
  if not found then
    return 0;
  end if;

  update public.profiles set points_balance = points_balance + v_amount where id = v_user;
  if not found then
    -- Rolls back the refunded_at stamp too, so the sweep retries later
    -- instead of recording a refund that never happened.
    raise exception 'refund failed: profile % not found', v_user;
  end if;
  insert into public.transactions (user_id, amount, type, description)
    values (v_user, v_amount, 'refund_repurpose', 'Refund — AI Repurposer failed');
  return v_amount;
end;
$$;

create or replace function public.repurpose_complete_job(
  p_job_id uuid,
  p_lease uuid,
  p_result jsonb,
  p_retention_days int
) returns boolean
language plpgsql security definer set search_path = public as $$
begin
  update public.repurpose_jobs
     set status = 'succeeded', stage = 'done', progress = 1,
         status_message = null, result = p_result,
         lease_token = null, locked_by = null,
         finished_at = now(),
         expires_at = now() + make_interval(days => p_retention_days)
   where id = p_job_id and lease_token = p_lease and status = 'processing';
  return found;
end;
$$;

-- Records a failed attempt. Retryable failures with attempts left go
-- back to the queue after p_retry_delay_seconds; anything else fails the
-- job for good and refunds it in the same transaction.
-- Returns {"owned": bool, "final": bool, "refunded": int}.
create or replace function public.repurpose_fail_job(
  p_job_id uuid,
  p_lease uuid,
  p_error_code text,
  p_user_error text,
  p_internal_error text,
  p_retryable boolean,
  p_retry_delay_seconds int,
  p_retention_days int
) returns jsonb
language plpgsql security definer set search_path = public as $$
declare
  v_job public.repurpose_jobs%rowtype;
  v_refunded int := 0;
begin
  select * into v_job from public.repurpose_jobs
   where id = p_job_id and lease_token = p_lease and status = 'processing'
   for update;
  if not found then
    return jsonb_build_object('owned', false, 'final', false, 'refunded', 0);
  end if;

  if p_retryable and v_job.attempts < v_job.max_attempts then
    update public.repurpose_jobs
       set status = 'queued', lease_token = null, locked_by = null,
           next_attempt_at = now() + make_interval(secs => p_retry_delay_seconds),
           last_error = p_internal_error, error_code = p_error_code,
           status_message = 'Retrying after a temporary problem'
     where id = p_job_id;
    return jsonb_build_object('owned', true, 'final', false, 'refunded', 0);
  end if;

  update public.repurpose_jobs
     set status = 'failed', stage = 'failed', lease_token = null, locked_by = null,
         error_code = p_error_code, error = p_user_error, last_error = p_internal_error,
         status_message = null, finished_at = now(),
         expires_at = now() + make_interval(days => p_retention_days)
   where id = p_job_id;
  v_refunded := public.repurpose_refund_job(p_job_id);
  return jsonb_build_object('owned', true, 'final', true, 'refunded', v_refunded);
end;
$$;

-- Graceful shutdown (SIGTERM during a redeploy): hand the job straight
-- back without counting the interrupted attempt against it.
create or replace function public.repurpose_release_job(p_job_id uuid, p_lease uuid)
returns boolean
language plpgsql security definer set search_path = public as $$
begin
  update public.repurpose_jobs
     set status = 'queued', lease_token = null, locked_by = null,
         attempts = greatest(attempts - 1, 0), next_attempt_at = now(),
         status_message = 'Waiting for a processing slot'
   where id = p_job_id and lease_token = p_lease and status = 'processing';
  return found;
end;
$$;

-- Periodic safety net, run by every worker:
--   * jobs that exhausted their attempts without a graceful failure
--     (every attempt crashed or was killed) are failed and refunded;
--   * failed jobs whose refund didn't go through are refunded again.
create or replace function public.repurpose_sweep(p_stale_seconds int, p_retention_days int)
returns jsonb
language plpgsql security definer set search_path = public as $$
declare
  v_id uuid;
  v_failed int := 0;
  v_refunded int := 0;
begin
  for v_id in
    select id from public.repurpose_jobs
     where attempts >= max_attempts
       and (status = 'queued'
            or (status = 'processing'
                and heartbeat_at < now() - make_interval(secs => p_stale_seconds)))
     for update skip locked
  loop
    update public.repurpose_jobs
       set status = 'failed', stage = 'failed', lease_token = null, locked_by = null,
           error_code = coalesce(error_code, 'processing_interrupted'),
           error = 'Processing kept getting interrupted, so we stopped and refunded your coins. Please try again.',
           finished_at = now(),
           expires_at = now() + make_interval(days => p_retention_days)
     where id = v_id;
    v_failed := v_failed + 1;
  end loop;

  for v_id in
    select id from public.repurpose_jobs
     where status = 'failed' and refunded_at is null and coins_charged > 0
     for update skip locked
  loop
    begin
      if public.repurpose_refund_job(v_id) > 0 then
        v_refunded := v_refunded + 1;
      end if;
    exception when others then
      raise warning 'repurpose_sweep: refund for % failed: %', v_id, sqlerrm;
    end;
  end loop;

  return jsonb_build_object('failed', v_failed, 'refunded', v_refunded);
end;
$$;

create or replace function public.repurpose_register_upload(
  p_user_id uuid, p_bucket text, p_path text, p_bytes bigint, p_content_type text
) returns void
language sql security definer set search_path = public as $$
  insert into public.repurpose_uploads (source_path, bucket, user_id, declared_bytes, content_type)
  values (p_path, p_bucket, p_user_id, p_bytes, p_content_type);
$$;

-- The job (minus its lease token) with its clips, best rank first.
create or replace function public.repurpose_get_job(p_job_id uuid)
returns jsonb
language sql security definer set search_path = public as $$
  select (to_jsonb(j) - 'lease_token') || jsonb_build_object(
           'clips', coalesce((select jsonb_agg(to_jsonb(c) order by c.rank)
                                from public.repurpose_clips c where c.job_id = j.id), '[]'::jsonb))
    from public.repurpose_jobs j
   where j.id = p_job_id;
$$;

-- Insert-or-update one clip, only while the caller holds the job's lease.
create or replace function public.repurpose_save_clip(p_job_id uuid, p_lease uuid, p_clip jsonb)
returns boolean
language plpgsql security definer set search_path = public as $$
declare
  v_user uuid;
begin
  select user_id into v_user from public.repurpose_jobs
   where id = p_job_id and lease_token = p_lease and status = 'processing'
   for update;
  if not found then
    return false;
  end if;

  insert into public.repurpose_clips (
    job_id, user_id, rank, status, start_time, end_time, score, score_breakdown, category,
    title, reason, hook_line, caption, hashtags, transcript, video_path, video_url,
    thumbnail_path, thumbnail_url, quote_card_path, quote_card_url, render_meta
  ) values (
    p_job_id, v_user, (p_clip->>'rank')::int, coalesce(p_clip->>'status', 'selected'),
    (p_clip->>'start_time')::real, (p_clip->>'end_time')::real, (p_clip->>'score')::int,
    p_clip->'score_breakdown', p_clip->>'category', p_clip->>'title', p_clip->>'reason',
    p_clip->>'hook_line', p_clip->>'caption',
    coalesce(array(select jsonb_array_elements_text(coalesce(p_clip->'hashtags', '[]'::jsonb))), '{}'),
    p_clip->>'transcript', p_clip->>'video_path', p_clip->>'video_url',
    p_clip->>'thumbnail_path', p_clip->>'thumbnail_url',
    p_clip->>'quote_card_path', p_clip->>'quote_card_url',
    coalesce(p_clip->'render_meta', '{}'::jsonb)
  )
  on conflict (job_id, rank) do update set
    status = excluded.status, start_time = excluded.start_time, end_time = excluded.end_time,
    score = excluded.score, score_breakdown = excluded.score_breakdown,
    category = excluded.category, title = excluded.title, reason = excluded.reason,
    hook_line = excluded.hook_line, caption = excluded.caption, hashtags = excluded.hashtags,
    transcript = excluded.transcript, video_path = excluded.video_path,
    video_url = excluded.video_url, thumbnail_path = excluded.thumbnail_path,
    thumbnail_url = excluded.thumbnail_url, quote_card_path = excluded.quote_card_path,
    quote_card_url = excluded.quote_card_url, render_meta = excluded.render_meta;
  return true;
end;
$$;

create or replace function public.repurpose_get_transcript(p_sha256 text, p_model text)
returns jsonb
language sql security definer set search_path = public as $$
  update public.repurpose_transcripts set last_used_at = now()
   where source_sha256 = p_sha256 and model = p_model
  returning to_jsonb(repurpose_transcripts);
$$;

create or replace function public.repurpose_put_transcript(
  p_sha256 text, p_model text, p_duration double precision, p_language text, p_text text,
  p_segments jsonb, p_words jsonb
) returns void
language sql security definer set search_path = public as $$
  insert into public.repurpose_transcripts
    (source_sha256, model, duration_seconds, language, text, segments, words)
  values (p_sha256, p_model, p_duration, p_language, coalesce(p_text, ''), p_segments, p_words)
  on conflict (source_sha256, model) do update set last_used_at = now();
$$;

-- Cleanup, step 1: private source files that are no longer needed —
-- sources of jobs finished more than p_source_hours ago, and uploads
-- that never became a job within p_orphan_hours.
create or replace function public.repurpose_sources_to_delete(
  p_source_hours int, p_orphan_hours int, p_limit int
) returns jsonb
language sql security definer set search_path = public as $$
  select coalesce(jsonb_agg(x), '[]'::jsonb) from (
    (select j.source_bucket as bucket, j.source_path as path
       from public.repurpose_jobs j
      where j.source_kind = 'private_upload' and j.source_deleted_at is null
        and j.status in ('succeeded', 'failed')
        and j.finished_at < now() - make_interval(hours => p_source_hours)
      limit p_limit)
    union
    (select u.bucket, u.source_path
       from public.repurpose_uploads u
      where u.job_id is null and u.deleted_at is null
        and u.created_at < now() - make_interval(hours => p_orphan_hours)
      limit p_limit)
  ) x;
$$;

create or replace function public.repurpose_mark_sources_deleted(p_paths text[])
returns void
language sql security definer set search_path = public as $$
  update public.repurpose_uploads set deleted_at = now()
   where source_path = any(p_paths) and deleted_at is null;
  update public.repurpose_jobs set source_deleted_at = now()
   where source_path = any(p_paths) and source_kind = 'private_upload' and source_deleted_at is null;
$$;

-- Cleanup, step 2: jobs past their retention, with every generated file
-- and whether a post still uses it. A clip the creator posted to Viyo
-- is the post's media now: it is reported as referenced and must not
-- be deleted.
create or replace function public.repurpose_expired_jobs(p_limit int)
returns jsonb
language sql security definer set search_path = public as $$
  select coalesce(jsonb_agg(jsonb_build_object('job_id', j.id, 'assets', coalesce((
           select jsonb_agg(jsonb_build_object(
                    'path', a.path, 'url', a.url,
                    'referenced', exists (select 1 from public.posts p
                                           where p.media_url = a.url or p.thumbnail_url = a.url)))
             from public.repurpose_clips c
             cross join lateral (values (c.video_path, c.video_url),
                                        (c.thumbnail_path, c.thumbnail_url),
                                        (c.quote_card_path, c.quote_card_url)) a(path, url)
            where c.job_id = j.id and a.path is not null), '[]'::jsonb))), '[]'::jsonb)
    from (select id from public.repurpose_jobs
           where expires_at < now() and status in ('succeeded', 'failed')
           order by expires_at limit p_limit) j;
$$;

create or replace function public.repurpose_delete_jobs(p_job_ids uuid[])
returns int
language plpgsql security definer set search_path = public as $$
declare
  v_count int;
begin
  delete from public.repurpose_jobs where id = any(p_job_ids) and status in ('succeeded', 'failed');
  get diagnostics v_count = row_count;
  return v_count;
end;
$$;

create or replace function public.repurpose_prune_transcripts(p_days int)
returns int
language plpgsql security definer set search_path = public as $$
declare
  v_count int;
begin
  delete from public.repurpose_transcripts
   where last_used_at < now() - make_interval(days => p_days);
  get diagnostics v_count = row_count;
  return v_count;
end;
$$;

revoke all on function public.repurpose_register_upload(uuid, text, text, bigint, text) from public, anon, authenticated;
revoke all on function public.repurpose_get_job(uuid) from public, anon, authenticated;
revoke all on function public.repurpose_save_clip(uuid, uuid, jsonb) from public, anon, authenticated;
revoke all on function public.repurpose_get_transcript(text, text) from public, anon, authenticated;
revoke all on function public.repurpose_put_transcript(text, text, double precision, text, text, jsonb, jsonb) from public, anon, authenticated;
revoke all on function public.repurpose_sources_to_delete(int, int, int) from public, anon, authenticated;
revoke all on function public.repurpose_mark_sources_deleted(text[]) from public, anon, authenticated;
revoke all on function public.repurpose_expired_jobs(int) from public, anon, authenticated;
revoke all on function public.repurpose_delete_jobs(uuid[]) from public, anon, authenticated;
revoke all on function public.repurpose_prune_transcripts(int) from public, anon, authenticated;

grant execute on function public.repurpose_register_upload(uuid, text, text, bigint, text) to service_role;
grant execute on function public.repurpose_get_job(uuid) to service_role;
grant execute on function public.repurpose_save_clip(uuid, uuid, jsonb) to service_role;
grant execute on function public.repurpose_get_transcript(text, text) to service_role;
grant execute on function public.repurpose_put_transcript(text, text, double precision, text, text, jsonb, jsonb) to service_role;
grant execute on function public.repurpose_sources_to_delete(int, int, int) to service_role;
grant execute on function public.repurpose_mark_sources_deleted(text[]) to service_role;
grant execute on function public.repurpose_expired_jobs(int) to service_role;
grant execute on function public.repurpose_delete_jobs(uuid[]) to service_role;
grant execute on function public.repurpose_prune_transcripts(int) to service_role;

revoke all on function public.repurpose_create_job(uuid, int, int, int, text, text, text, text, text, text) from public, anon, authenticated;
revoke all on function public.repurpose_claim_job(text, int) from public, anon, authenticated;
revoke all on function public.repurpose_heartbeat(uuid, uuid, text, double precision, text, jsonb) from public, anon, authenticated;
revoke all on function public.repurpose_refund_job(uuid) from public, anon, authenticated;
revoke all on function public.repurpose_complete_job(uuid, uuid, jsonb, int) from public, anon, authenticated;
revoke all on function public.repurpose_fail_job(uuid, uuid, text, text, text, boolean, int, int) from public, anon, authenticated;
revoke all on function public.repurpose_release_job(uuid, uuid) from public, anon, authenticated;
revoke all on function public.repurpose_sweep(int, int) from public, anon, authenticated;

grant execute on function public.repurpose_create_job(uuid, int, int, int, text, text, text, text, text, text) to service_role;
grant execute on function public.repurpose_claim_job(text, int) to service_role;
grant execute on function public.repurpose_heartbeat(uuid, uuid, text, double precision, text, jsonb) to service_role;
grant execute on function public.repurpose_refund_job(uuid) to service_role;
grant execute on function public.repurpose_complete_job(uuid, uuid, jsonb, int) to service_role;
grant execute on function public.repurpose_fail_job(uuid, uuid, text, text, text, boolean, int, int) to service_role;
grant execute on function public.repurpose_release_job(uuid, uuid) to service_role;
grant execute on function public.repurpose_sweep(int, int) to service_role;

-- ---------------------------------------------------------------
-- Private bucket for source videos
-- ---------------------------------------------------------------
-- Uploads only happen through signed upload URLs the backend issues
-- (POST /api/v1/repurpose/uploads), and only the backend reads them, so
-- no storage.objects policies are needed — and none are added, so
-- nobody can list or read another creator's source video.
--
-- 2 GB is this bucket's own cap. The project-wide limit in Dashboard ->
-- Storage -> Settings ("Upload file size limit") still applies on top
-- and wins if lower; the backend reads this bucket's value at runtime
-- and its own MAX_INPUT_SIZE_MB, and enforces the smaller.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'repurpose-sources', 'repurpose-sources', false, 2147483648,
  array['video/mp4', 'video/quicktime', 'video/webm', 'video/x-matroska',
        'video/3gpp', 'video/x-m4v', 'application/octet-stream']
)
on conflict (id) do update
  set public = false,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

-- ---------------------------------------------------------------
-- Check after running (read-only): bucket settings as they really are.
-- ---------------------------------------------------------------
select id, public, file_size_limit, allowed_mime_types
  from storage.buckets
 where id in ('repurpose-sources', 'processed-videos', 'posts-media');

-- ---------------------------------------------------------------------
-- Free first video: every account gets one AI Repurposer run without
-- coins, capped app-wide per day. A free run that fails doesn't use up
-- the free try. Safe to re-run.

alter table public.repurpose_jobs add column if not exists free_trial boolean not null default false;

create index if not exists repurpose_jobs_free_trial_idx
  on public.repurpose_jobs (created_at) where free_trial;

create or replace function public.repurpose_free_trial_status(p_user_id uuid, p_daily_cap int)
returns jsonb
language plpgsql stable security definer set search_path = public as $$
declare
  v_used boolean;
  v_today int;
begin
  select exists (select 1 from public.repurpose_jobs
                  where user_id = p_user_id and free_trial and status <> 'failed') into v_used;
  select count(*) into v_today from public.repurpose_jobs
   where free_trial and created_at > now() - interval '24 hours';
  return jsonb_build_object('eligible', not v_used and v_today < p_daily_cap,
                            'used', v_used, 'sold_out', v_today >= p_daily_cap);
end;
$$;

create or replace function public.repurpose_create_free_job(
  p_user_id uuid,
  p_daily_cap int,
  p_daily_limit int,
  p_max_attempts int,
  p_source_kind text,
  p_source_bucket text,
  p_source_path text,
  p_source_url text,
  p_creative_prompt text,
  p_client_request_id text
) returns jsonb
language plpgsql security definer set search_path = public as $$
declare
  v_status jsonb;
  v_out jsonb;
begin
  -- One free-job creation at a time app-wide, so the daily cap holds;
  -- the profile row lock in repurpose_create_job serializes the user.
  perform pg_advisory_xact_lock(hashtext('repurpose_free_trial'));
  perform 1 from public.profiles where id = p_user_id for update;
  if p_client_request_id is not null and exists (
       select 1 from public.repurpose_jobs
        where user_id = p_user_id and client_request_id = p_client_request_id) then
    return public.repurpose_create_job(p_user_id, 0, p_daily_limit, p_max_attempts, p_source_kind,
                                       p_source_bucket, p_source_path, p_source_url,
                                       p_creative_prompt, p_client_request_id);
  end if;
  v_status := public.repurpose_free_trial_status(p_user_id, p_daily_cap);
  if not (v_status->>'eligible')::boolean then
    return jsonb_build_object('ok', false, 'error', 'free_trial_unavailable');
  end if;
  v_out := public.repurpose_create_job(p_user_id, 0, p_daily_limit, p_max_attempts, p_source_kind,
                                       p_source_bucket, p_source_path, p_source_url,
                                       p_creative_prompt, p_client_request_id);
  if (v_out->>'ok')::boolean and (v_out->>'created')::boolean then
    update public.repurpose_jobs set free_trial = true where id = (v_out->>'job_id')::uuid;
    v_out := v_out || jsonb_build_object('free_trial', true);
  end if;
  return v_out;
end;
$$;

revoke all on function public.repurpose_free_trial_status(uuid, int) from public, anon, authenticated;
revoke all on function public.repurpose_create_free_job(uuid, int, int, int, text, text, text, text, text, text)
  from public, anon, authenticated;
