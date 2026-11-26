-- Brand campaigns: a business pays creators to make sponsored posts, through VIYO. Run after
-- creator_earnings_ledger.sql, creator_donations.sql and nardopay_orders.sql. Safe to run more than once.
-- Changes no existing data.
--
-- Rules (agreed with the owner)
--   * A business posts a brief with a fixed fee per creator and a number of creators (slots), and pays the whole budget
--     (fee x slots) upfront, by card through NardoPay, in US dollars converted at VIYO's rate. Campaign credit the
--     business already has is used first.
--   * Creators at level Rising or above apply; the business accepts (up to the slots) or declines.
--   * An accepted creator publishes a post and submits it; the post is labelled Sponsored. The business approves or
--     rejects it (a reason is required); a submission not decided within 7 days is approved automatically.
--   * On approval the creator earns 80% of the fee (VIYO keeps 20%), as a creator_earnings row of kind 'brand'
--     (earning_key 'brand:<application id>'), held like every new earning, once.
--   * Money not used (slots not filled, rejected posts, creators who didn't deliver by the deadline) goes back to the
--     business as campaign credit, exactly once, when the campaign is closed or its deadline passes. NardoPay has no
--     refund API, so a refund in money is done by an admin on request.

-- ---------------------------------------------------------------- campaigns
create table if not exists public.brand_campaigns (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  brand_name text not null check (char_length(brand_name) between 1 and 80),
  title text not null check (char_length(title) between 1 and 120),
  brief text not null check (char_length(brief) between 1 and 2000),
  deliverable text not null default '' check (char_length(deliverable) <= 500),
  link text check (link is null or (char_length(link) <= 300 and link like 'https://%')),
  fee_cents int not null check (fee_cents between 500 and 500000),          -- per creator: $5 to $5,000
  slots int not null check (slots between 1 and 50),
  budget_cents int not null check (budget_cents > 0),
  creator_share_rate numeric(5,4) not null default 0.80 check (creator_share_rate > 0 and creator_share_rate <= 1),
  credit_used_cents int not null default 0 check (credit_used_cents >= 0),
  pay_cents int not null default 0 check (pay_cents >= 0),                  -- paid through NardoPay
  status text not null default 'awaiting_payment'
    check (status in ('awaiting_payment', 'open', 'closed', 'cancelled')),
  deadline timestamptz not null,
  rate_id uuid,
  usd_rwf_rate numeric(12,4),
  price_rwf bigint,
  link_code text,
  payment_link text,
  transaction_id text unique,
  event_id uuid,
  amount_rwf numeric(14,2),
  credited_back_cents int not null default 0 check (credited_back_cents >= 0),
  failure_reason text not null default '' check (char_length(failure_reason) <= 400),
  created_at timestamptz not null default now(),
  paid_at timestamptz,
  closed_at timestamptz,
  check (budget_cents = fee_cents * slots),
  check (credit_used_cents + pay_cents = budget_cents),
  check (pay_cents = 0 or (price_rwf > 0 and usd_rwf_rate > 0))
);
create unique index if not exists brand_campaigns_link_code_key on public.brand_campaigns (link_code)
  where link_code is not null;
create index if not exists brand_campaigns_owner_idx on public.brand_campaigns (owner_id, created_at desc);
create index if not exists brand_campaigns_open_idx on public.brand_campaigns (status, created_at desc);

-- ---------------------------------------------------------------- applications (one per creator per campaign)
create table if not exists public.brand_campaign_applications (
  id uuid primary key default gen_random_uuid(),
  campaign_id uuid not null references public.brand_campaigns(id) on delete cascade,
  creator_id uuid not null references auth.users(id) on delete cascade,
  pitch text not null default '' check (char_length(pitch) <= 500),
  status text not null default 'applied'
    check (status in ('applied', 'accepted', 'declined', 'withdrawn', 'submitted', 'approved', 'rejected', 'expired')),
  post_id uuid,
  decision_note text not null default '' check (char_length(decision_note) <= 300),
  auto_approved boolean not null default false,
  paid_coins int,
  earning_id uuid,
  created_at timestamptz not null default now(),
  submitted_at timestamptz,
  decided_at timestamptz,
  unique (campaign_id, creator_id),
  check (status not in ('submitted', 'approved', 'rejected') or post_id is not null)
);
create index if not exists brand_campaign_applications_creator_idx
  on public.brand_campaign_applications (creator_id, created_at desc);
create index if not exists brand_campaign_applications_submitted_idx
  on public.brand_campaign_applications (status, submitted_at) where status = 'submitted';

-- ---------------------------------------------------------------- campaign credit (a ledger; balance = sum)
create table if not exists public.brand_credit_ledger (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  cents int not null check (cents <> 0),
  reason text not null check (reason in ('campaign_return', 'campaign_use', 'campaign_cancel', 'late_payment',
                                         'admin_refund', 'admin_adjustment')),
  campaign_id uuid references public.brand_campaigns(id) on delete set null,
  entry_key text not null unique,
  note text not null default '' check (char_length(note) <= 300),
  created_at timestamptz not null default now()
);
create index if not exists brand_credit_ledger_owner_idx on public.brand_credit_ledger (owner_id, created_at desc);

do $$ declare t text;
begin
  foreach t in array array['brand_campaigns', 'brand_campaign_applications', 'brand_credit_ledger'] loop
    execute format('alter table public.%I enable row level security', t);
    execute format('revoke all on public.%I from anon, authenticated', t);
    execute format('grant all on public.%I to service_role', t);
  end loop;
end $$;

-- ---------------------------------------------------------------- earnings of kind 'brand'; Sponsored label on posts
alter table public.creator_earnings add column if not exists brand_application_id uuid;
alter table public.creator_earnings drop constraint if exists creator_earnings_kind_check;
alter table public.creator_earnings add constraint creator_earnings_kind_check
  check (kind in ('unlock', 'donation', 'brand', 'reversal')) not valid;
do $$ begin
  if to_regclass('public.posts') is not null then
    alter table public.posts add column if not exists sponsored_campaign_id uuid;
    alter table public.posts add column if not exists sponsored_by text;
  end if;
end $$;

-- ---------------------------------------------------------------- credit balance
create or replace function public.brand_credit_balance(p_owner uuid)
returns int language sql stable set search_path = public as $$
  select coalesce(sum(cents), 0)::int from public.brand_credit_ledger where owner_id = p_owner $$;

-- ---------------------------------------------------------------- create (reserves credit atomically)
drop function if exists public.brand_campaign_create(uuid, text, text, text, text, text, int, int, int, text, text);
create or replace function public.brand_campaign_create(
  p_owner uuid, p_brand text, p_title text, p_brief text, p_deliverable text, p_link text, p_fee_cents int,
  p_slots int, p_days int, p_rate_id text, p_rate text)
returns jsonb language plpgsql set search_path = public as $$
declare
  v_budget int := p_fee_cents * p_slots;
  v_credit int;
  v_use int;
  v_pay int;
  v_rate numeric := nullif(p_rate, '')::numeric;
  c public.brand_campaigns;
begin
  perform pg_advisory_xact_lock(hashtext('brand_credit:' || p_owner::text));
  v_credit := greatest(public.brand_credit_balance(p_owner), 0);
  v_use := least(v_credit, v_budget);
  v_pay := v_budget - v_use;
  if v_pay > 0 and (v_rate is null or v_rate <= 0) then
    return jsonb_build_object('ok', false, 'error', 'no_rate');
  end if;
  insert into public.brand_campaigns (owner_id, brand_name, title, brief, deliverable, link, fee_cents, slots,
                                      budget_cents, credit_used_cents, pay_cents, status, deadline, rate_id,
                                      usd_rwf_rate, price_rwf, paid_at)
  values (p_owner, p_brand, p_title, p_brief, coalesce(p_deliverable, ''), nullif(p_link, ''), p_fee_cents, p_slots,
          v_budget, v_use, v_pay, case when v_pay = 0 then 'open' else 'awaiting_payment' end,
          now() + make_interval(days => greatest(least(coalesce(p_days, 14), 90), 1)),
          case when v_pay > 0 then nullif(p_rate_id, '')::uuid end,
          case when v_pay > 0 then v_rate end,
          case when v_pay > 0 then ceil(v_pay::numeric * v_rate / 100)::bigint end,
          case when v_pay = 0 then now() end)
  returning * into c;
  if v_use > 0 then
    insert into public.brand_credit_ledger (owner_id, cents, reason, campaign_id, entry_key)
    values (p_owner, -v_use, 'campaign_use', c.id, 'campaign:' || c.id::text || ':use');
  end if;
  return jsonb_build_object('ok', true, 'campaign', to_jsonb(c));
end $$;

-- ---------------------------------------------------------------- settle: unused money back as credit, once
create or replace function public.brand_campaign_settle(p_campaign uuid)
returns int language plpgsql set search_path = public as $$
declare
  c public.brand_campaigns;
  v_funded int;
  v_committed int;
  v_target int;
  v_delta int;
begin
  select * into c from public.brand_campaigns where id = p_campaign for update;
  if not found or c.status not in ('closed', 'cancelled') then
    return 0;
  end if;
  v_funded := c.credit_used_cents + case when c.transaction_id is not null then c.pay_cents else 0 end;
  select coalesce(count(*), 0) * c.fee_cents into v_committed from public.brand_campaign_applications
   where campaign_id = c.id and status in ('accepted', 'submitted', 'approved');
  v_target := greatest(v_funded - v_committed, 0);
  v_delta := v_target - c.credited_back_cents;
  if v_delta <= 0 then
    return 0;
  end if;
  insert into public.brand_credit_ledger (owner_id, cents, reason, campaign_id, entry_key)
  values (c.owner_id, v_delta, case when c.status = 'cancelled' then 'campaign_cancel' else 'campaign_return' end,
          c.id, 'campaign:' || c.id::text || ':return:' || v_target::text)
  on conflict (entry_key) do nothing;
  update public.brand_campaigns set credited_back_cents = v_target where id = c.id;
  return v_delta;
end $$;

-- ---------------------------------------------------------------- a verified NardoPay payment (atomic, once)
drop function if exists public.brand_campaign_record_payment(text, text, text, text, text);
create or replace function public.brand_campaign_record_payment(
  p_campaign text, p_transaction text, p_event text, p_amount text, p_currency text)
returns jsonb language plpgsql set search_path = public as $$
declare
  c public.brand_campaigns;
begin
  select * into c from public.brand_campaigns where id = p_campaign::uuid for update;
  if not found then
    return jsonb_build_object('result', 'not_found');
  end if;
  if c.transaction_id is not null then
    return jsonb_build_object('result', case when c.transaction_id = p_transaction then 'duplicate'
                                             else 'other_transaction' end);
  end if;
  if exists (select 1 from public.brand_campaigns where transaction_id = p_transaction) then
    return jsonb_build_object('result', 'transaction_used_elsewhere');
  end if;
  update public.brand_campaigns
     set transaction_id = p_transaction, event_id = nullif(p_event, '')::uuid, amount_rwf = p_amount::numeric,
         paid_at = now(), status = case when status = 'awaiting_payment' then 'open' else status end
   where id = c.id;
  if c.status = 'cancelled' then
    -- Paid after it was cancelled (e.g. the payment page was still open): the money becomes credit.
    perform public.brand_campaign_settle(c.id);
    return jsonb_build_object('result', 'credited', 'owner_id', c.owner_id);
  end if;
  return jsonb_build_object('result', 'recorded', 'owner_id', c.owner_id);
end $$;

-- ---------------------------------------------------------------- every change to an application
-- p_actor is the signed-in user (null only for 'auto_approve' / 'expire' run by the server's maintenance).
drop function if exists public.brand_campaign_act(uuid, uuid, text, uuid, text, int, text, text);
create or replace function public.brand_campaign_act(
  p_application uuid, p_actor uuid, p_action text, p_post uuid, p_note text, p_hold_days int, p_coin_usd text,
  p_sponsor text)
returns jsonb language plpgsql set search_path = public as $$
declare
  a public.brand_campaign_applications;
  c public.brand_campaigns;
  v_taken int;
  v_share_usd numeric;
  v_coins int;
  v_earning uuid;
  v_owner_ok boolean;
begin
  select * into a from public.brand_campaign_applications where id = p_application;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'not_found');
  end if;
  select * into c from public.brand_campaigns where id = a.campaign_id for update;   -- one change at a time
  select * into a from public.brand_campaign_applications where id = p_application for update;
  v_owner_ok := p_actor is not null and p_actor = c.owner_id;

  if p_action = 'accept' then
    if not v_owner_ok then return jsonb_build_object('ok', false, 'error', 'not_allowed'); end if;
    if c.status <> 'open' or now() > c.deadline then return jsonb_build_object('ok', false, 'error', 'campaign_not_open'); end if;
    if a.status <> 'applied' then return jsonb_build_object('ok', false, 'error', 'wrong_state'); end if;
    select count(*) into v_taken from public.brand_campaign_applications
     where campaign_id = c.id and status in ('accepted', 'submitted', 'approved');
    if v_taken >= c.slots then return jsonb_build_object('ok', false, 'error', 'no_slots'); end if;
    update public.brand_campaign_applications set status = 'accepted', decided_at = now() where id = a.id;

  elsif p_action = 'decline' then
    if not v_owner_ok then return jsonb_build_object('ok', false, 'error', 'not_allowed'); end if;
    if a.status <> 'applied' then return jsonb_build_object('ok', false, 'error', 'wrong_state'); end if;
    update public.brand_campaign_applications
       set status = 'declined', decided_at = now(), decision_note = left(coalesce(p_note, ''), 300) where id = a.id;

  elsif p_action = 'withdraw' then
    if p_actor is distinct from a.creator_id then return jsonb_build_object('ok', false, 'error', 'not_allowed'); end if;
    if a.status not in ('applied', 'accepted') then return jsonb_build_object('ok', false, 'error', 'wrong_state'); end if;
    update public.brand_campaign_applications set status = 'withdrawn', decided_at = now() where id = a.id;

  elsif p_action = 'submit' then
    if p_actor is distinct from a.creator_id then return jsonb_build_object('ok', false, 'error', 'not_allowed'); end if;
    if a.status <> 'accepted' then return jsonb_build_object('ok', false, 'error', 'wrong_state'); end if;
    if now() > c.deadline or c.status not in ('open', 'closed') then
      return jsonb_build_object('ok', false, 'error', 'deadline_passed');
    end if;
    if p_post is null then return jsonb_build_object('ok', false, 'error', 'no_post'); end if;
    if to_regclass('public.posts') is not null then
      if not exists (select 1 from public.posts where id = p_post and user_id = a.creator_id) then
        return jsonb_build_object('ok', false, 'error', 'not_your_post');
      end if;
      if exists (select 1 from public.brand_campaign_applications
                  where post_id = p_post and id <> a.id and status in ('submitted', 'approved')) then
        return jsonb_build_object('ok', false, 'error', 'post_used');
      end if;
      execute 'update public.posts set sponsored_campaign_id = $1, sponsored_by = $2 where id = $3'
        using c.id, left(coalesce(p_sponsor, c.brand_name), 80), p_post;
    end if;
    update public.brand_campaign_applications
       set status = 'submitted', post_id = p_post, submitted_at = now() where id = a.id;

  elsif p_action = 'reject' then
    if not v_owner_ok then return jsonb_build_object('ok', false, 'error', 'not_allowed'); end if;
    if a.status <> 'submitted' then return jsonb_build_object('ok', false, 'error', 'wrong_state'); end if;
    if char_length(trim(coalesce(p_note, ''))) < 3 then return jsonb_build_object('ok', false, 'error', 'reason_required'); end if;
    update public.brand_campaign_applications
       set status = 'rejected', decided_at = now(), decision_note = left(p_note, 300) where id = a.id;
    if to_regclass('public.posts') is not null then
      execute 'update public.posts set sponsored_campaign_id = null, sponsored_by = null
                where id = $1 and sponsored_campaign_id = $2' using a.post_id, c.id;
    end if;

  elsif p_action in ('approve', 'auto_approve') then
    if p_action = 'approve' and not v_owner_ok then return jsonb_build_object('ok', false, 'error', 'not_allowed'); end if;
    if p_action = 'auto_approve' and (p_actor is not null or a.submitted_at > now() - interval '7 days') then
      return jsonb_build_object('ok', false, 'error', 'not_due');
    end if;
    if a.status <> 'submitted' then return jsonb_build_object('ok', false, 'error', 'wrong_state'); end if;
    v_share_usd := round(c.fee_cents::numeric / 100 * c.creator_share_rate, 4);
    v_coins := floor(v_share_usd / p_coin_usd::numeric);
    if v_coins > 0 then
      insert into public.creator_earnings (creator_id, viewer_id, coins, kind, earning_key, status, available_at,
                                           brand_application_id, source_post_id, note)
      values (a.creator_id, c.owner_id, v_coins, 'brand', 'brand:' || a.id::text,
              case when coalesce(p_hold_days, 0) > 0 then 'pending' else 'available' end,
              now() + make_interval(days => greatest(coalesce(p_hold_days, 0), 0)), a.id, a.post_id,
              left('Brand deal: ' || c.brand_name, 400))
      on conflict (earning_key) where earning_key is not null do nothing
      returning id into v_earning;
    end if;
    update public.brand_campaign_applications
       set status = 'approved', decided_at = now(), auto_approved = (p_action = 'auto_approve'),
           paid_coins = v_coins, earning_id = v_earning
     where id = a.id;

  else
    return jsonb_build_object('ok', false, 'error', 'unknown_action');
  end if;

  perform public.brand_campaign_settle(c.id);       -- only does anything on a closed or cancelled campaign
  select * into a from public.brand_campaign_applications where id = a.id;
  return jsonb_build_object('ok', true, 'application', to_jsonb(a), 'campaign_id', c.id, 'owner_id', c.owner_id,
                            'creator_id', a.creator_id, 'brand_name', c.brand_name, 'title', c.title,
                            'share_usd', v_share_usd);
end $$;

-- ---------------------------------------------------------------- close (business) / expire (deadline) / cancel
-- 'close': no more applications or acceptances; accepted creators can still deliver until the deadline.
-- 'expire': the deadline has passed (server): accepted creators who didn't deliver lose the slot.
-- 'cancel': an unpaid campaign (business, or the server after 24 hours).
drop function if exists public.brand_campaign_close(uuid, uuid, text);
create or replace function public.brand_campaign_close(p_campaign uuid, p_actor uuid, p_mode text)
returns jsonb language plpgsql set search_path = public as $$
declare
  c public.brand_campaigns;
begin
  select * into c from public.brand_campaigns where id = p_campaign for update;
  if not found then return jsonb_build_object('ok', false, 'error', 'not_found'); end if;
  if p_actor is not null and p_actor <> c.owner_id then return jsonb_build_object('ok', false, 'error', 'not_allowed'); end if;
  if p_mode = 'cancel' then
    if c.status <> 'awaiting_payment' then return jsonb_build_object('ok', false, 'error', 'wrong_state'); end if;
    update public.brand_campaigns set status = 'cancelled', closed_at = now() where id = c.id;
  elsif p_mode = 'close' then
    if c.status not in ('open', 'closed') then return jsonb_build_object('ok', false, 'error', 'wrong_state'); end if;
    update public.brand_campaigns set status = 'closed', closed_at = coalesce(closed_at, now()) where id = c.id;
    update public.brand_campaign_applications set status = 'expired', decided_at = now()
     where campaign_id = c.id and status = 'applied';
  elsif p_mode = 'expire' then
    if p_actor is not null or now() <= c.deadline or c.status not in ('open', 'closed') then
      return jsonb_build_object('ok', false, 'error', 'not_due');
    end if;
    update public.brand_campaigns set status = 'closed', closed_at = coalesce(closed_at, now()) where id = c.id;
    update public.brand_campaign_applications set status = 'expired', decided_at = now()
     where campaign_id = c.id and status in ('applied', 'accepted');
  else
    return jsonb_build_object('ok', false, 'error', 'unknown_mode');
  end if;
  return jsonb_build_object('ok', true, 'credited_cents', public.brand_campaign_settle(c.id));
end $$;

-- ---------------------------------------------------------------- what the server's maintenance has to do
create or replace function public.brand_campaigns_due()
returns jsonb language sql stable set search_path = public as $$
  select jsonb_build_object(
    'auto_approve', coalesce((select jsonb_agg(id) from (select id from public.brand_campaign_applications
                               where status = 'submitted' and submitted_at <= now() - interval '7 days'
                               limit 200) t), '[]'::jsonb),
    'expire', coalesce((select jsonb_agg(id) from (select c.id from public.brand_campaigns c
                         where c.status in ('open', 'closed') and c.deadline < now()
                           and (c.status = 'open' or exists (select 1 from public.brand_campaign_applications a
                                where a.campaign_id = c.id and a.status in ('applied', 'accepted')))
                         limit 200) t), '[]'::jsonb),
    'abandon', coalesce((select jsonb_agg(id) from (select id from public.brand_campaigns
                          where status = 'awaiting_payment' and created_at < now() - interval '24 hours'
                          limit 200) t), '[]'::jsonb)) $$;

-- Reversals (refunds) also cover brand-deal earnings.
create or replace function public.viyo_reverse_earning(p_earning uuid, p_note text)
returns jsonb language plpgsql set search_path = public as $$
declare
  e public.creator_earnings;
  v_id uuid;
begin
  select * into e from public.creator_earnings where id = p_earning for update;
  if not found then
    return jsonb_build_object('ok', false, 'error', 'not_found');
  end if;
  if e.kind not in ('unlock', 'donation', 'brand') then
    return jsonb_build_object('ok', false, 'error', 'not_an_earning');
  end if;
  if e.status = 'reversed' or e.reversed_at is not null then
    return jsonb_build_object('ok', true, 'changed', false, 'status', e.status);
  end if;
  if e.status = 'pending_withdrawal' then
    return jsonb_build_object('ok', false, 'error', 'in_withdrawal');
  end if;
  if e.status in ('pending', 'available') then
    update public.creator_earnings
       set status = 'reversed', reversed_at = now(), note = left(coalesce(p_note, ''), 400)
     where id = e.id;
    return jsonb_build_object('ok', true, 'changed', true, 'status', 'reversed', 'creator_id', e.creator_id,
                              'coins', e.coins);
  end if;
  insert into public.creator_earnings (creator_id, source_post_id, viewer_id, coins, kind, earning_key, status,
                                       available_at, reversal_of, note, donation_id, brand_application_id)
  values (e.creator_id, e.source_post_id, e.viewer_id, -e.coins, 'reversal', 'reversal:' || e.id::text, 'available',
          now(), e.id, left(coalesce(p_note, ''), 400), e.donation_id, e.brand_application_id)
  returning id into v_id;
  update public.creator_earnings set reversed_at = now() where id = e.id;
  return jsonb_build_object('ok', true, 'changed', true, 'status', 'deducted', 'reversal_id', v_id,
                            'creator_id', e.creator_id, 'coins', e.coins);
end $$;

do $$
declare f text;
begin
  foreach f in array array[
    'brand_credit_balance(uuid)',
    'viyo_reverse_earning(uuid, text)',
    'brand_campaign_create(uuid, text, text, text, text, text, int, int, int, text, text)',
    'brand_campaign_settle(uuid)',
    'brand_campaign_record_payment(text, text, text, text, text)',
    'brand_campaign_act(uuid, uuid, text, uuid, text, int, text, text)',
    'brand_campaign_close(uuid, uuid, text)',
    'brand_campaigns_due()'] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;
