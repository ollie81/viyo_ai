-- Viyo Studio: the story canon, a permanent, time-aware memory of one story (a novel and every episode adapted
-- from it). Read by studio_canon.py. Run after creator_studio_schema.sql. Safe to run more than once. Only adds new
-- tables: no existing table or column is changed.
--
-- Time: a point in the story is (chapter, seq): the chapter and the beat inside it, in reading order. A record is
-- true from valid_from (inclusive) until valid_until (exclusive); an empty valid_until means "still true".
--
-- Status of every record: proposed (an AI suggestion, never canon) -> approved (the creator accepted it) ->
-- creator_locked (the creator fixed it: it beats any conflicting record) or retired (no longer used, kept for history).
-- Only approved and creator_locked records are canon, and the database refuses either without a decision on record.

create table if not exists public.studio_stories (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid references public.studio_projects(id) on delete set null,   -- the production it started from
  series_id uuid,                                                            -- = studio_projects.series_id of its episodes
  title text not null default '' check (char_length(title) <= 200),
  status text not null default 'draft' check (status in ('draft', 'archived')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists studio_stories_owner_idx on public.studio_stories (owner_id, updated_at desc);

-- Who and what the story is about: characters, places, objects, secrets and groups.
create table if not exists public.studio_story_entities (
  id uuid primary key default gen_random_uuid(),
  story_id uuid not null references public.studio_stories(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  kind text not null check (kind in ('character', 'place', 'object', 'secret', 'group')),
  name text not null check (char_length(name) between 1 and 120),
  aliases jsonb not null default '[]'::jsonb,              -- other names the text uses ("the old man", "Mama")
  profile jsonb not null default '{}'::jsonb,              -- a secret's truth, a place's description, ...
  character_id uuid,                                       -- studio_characters.id once cast (later stages)
  element_id uuid,                                         -- studio_elements.id once it has a picture (later stages)
  first_chapter int,
  source_quote text not null default '',
  origin text not null default 'ai' check (origin in ('creator', 'ai', 'import')),
  status text not null default 'proposed' check (status in ('proposed', 'approved', 'creator_locked', 'retired')),
  decided_by text check (decided_by in ('creator')),
  decided_at timestamptz,
  created_at timestamptz not null default now(),
  check (status not in ('approved', 'creator_locked') or decided_by is not null)
);
create index if not exists studio_story_entities_story_idx on public.studio_story_entities (story_id, kind);

-- Facts: what is true, about whom, from when until when, and the evidence for it.
--   status        subject is alive / dead / missing / unconscious / imprisoned ...   (one at a time)
--   location      where a character or object is (object_id = a place, or a character who carries it)  (one at a time)
--   relationship  subject -> object_id; attribute = which side of it ("family", "romance", "work", "loyalty"),
--                 value = what it is ("sister", "married", "enemy")  (one value per pair and side at a time)
--   knows         the character (subject) knows the secret (object_id)
--   trait         attribute = "eye colour", "age", "fear", ...; value = what it is  (one per attribute)
--   rule          a continuity or world rule (subject optional)
create table if not exists public.studio_story_facts (
  id uuid primary key default gen_random_uuid(),
  story_id uuid not null references public.studio_stories(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  kind text not null check (kind in ('status', 'location', 'relationship', 'knows', 'trait', 'rule')),
  subject_id uuid references public.studio_story_entities(id) on delete cascade,
  attribute text not null default '' check (char_length(attribute) <= 80),
  object_id uuid references public.studio_story_entities(id) on delete cascade,
  value text not null default '' check (char_length(value) <= 500),
  statement text not null default '' check (char_length(statement) <= 600),   -- the fact in one plain sentence
  valid_from_chapter int not null default 0 check (valid_from_chapter >= 0),
  valid_from_seq int not null default 0 check (valid_from_seq >= 0),
  valid_until_chapter int,
  valid_until_seq int,
  source_chapter int,
  source_ref text not null default '' check (char_length(source_ref) <= 200),   -- passage id, episode, "creator"
  source_quote text not null default '' check (char_length(source_quote) <= 1000),
  event_id uuid,
  origin text not null default 'ai' check (origin in ('creator', 'ai', 'import')),
  confidence real,
  status text not null default 'proposed' check (status in ('proposed', 'approved', 'creator_locked', 'retired')),
  decided_by text check (decided_by in ('creator')),
  decided_at timestamptz,
  supersedes uuid,
  retired_reason text,
  created_at timestamptz not null default now(),
  -- Never canon without the creator's decision.
  check (status not in ('approved', 'creator_locked') or decided_by is not null),
  -- Every fact keeps its evidence: a quote from the story, or it was stated by the creator.
  check (origin = 'creator' or char_length(source_quote) > 0),
  check (valid_until_chapter is null or (valid_until_chapter, coalesce(valid_until_seq, 0))
                                        > (valid_from_chapter, valid_from_seq))
);
create index if not exists studio_story_facts_story_idx on public.studio_story_facts (story_id, status);
create index if not exists studio_story_facts_subject_idx on public.studio_story_facts (subject_id, kind);

-- Events: what happens, where it sits in reading order (chapter, seq) and in story time (story_order: flashbacks
-- happen earlier in story time than where they are read), and what caused it.
create table if not exists public.studio_story_events (
  id uuid primary key default gen_random_uuid(),
  story_id uuid not null references public.studio_stories(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  chapter int not null check (chapter >= 0),
  seq int not null default 0 check (seq >= 0),
  story_order numeric,                       -- position in story time; empty = same as reading order
  story_time text not null default '' check (char_length(story_time) <= 120),   -- "the night of the wedding", "1994"
  summary text not null check (char_length(summary) between 1 and 600),
  participants jsonb not null default '[]'::jsonb,   -- entity ids
  location_id uuid references public.studio_story_entities(id) on delete set null,
  causes jsonb not null default '[]'::jsonb,         -- event ids
  consequences text not null default '' check (char_length(consequences) <= 600),
  source_quote text not null default '' check (char_length(source_quote) <= 1000),
  origin text not null default 'ai' check (origin in ('creator', 'ai', 'import')),
  status text not null default 'proposed' check (status in ('proposed', 'approved', 'creator_locked', 'retired')),
  decided_by text check (decided_by in ('creator')),
  decided_at timestamptz,
  created_at timestamptz not null default now(),
  check (status not in ('approved', 'creator_locked') or decided_by is not null),
  check (origin = 'creator' or char_length(source_quote) > 0)
);
create index if not exists studio_story_events_story_idx on public.studio_story_events (story_id, chapter, seq);

-- Threads: setups, foreshadowing, mysteries and promises the story owes the reader, and where they pay off.
create table if not exists public.studio_story_threads (
  id uuid primary key default gen_random_uuid(),
  story_id uuid not null references public.studio_stories(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  kind text not null check (kind in ('setup', 'foreshadowing', 'mystery', 'promise', 'question')),
  description text not null check (char_length(description) between 1 and 600),
  entities jsonb not null default '[]'::jsonb,
  planted_chapter int not null check (planted_chapter >= 0),
  planted_seq int not null default 0,
  planted_quote text not null default '' check (char_length(planted_quote) <= 1000),
  payoff_plan text not null default '' check (char_length(payoff_plan) <= 600),
  due_chapter int,
  due_episode int,
  state text not null default 'open' check (state in ('open', 'resolved', 'dropped')),
  resolved_chapter int,
  resolved_quote text not null default '' check (char_length(resolved_quote) <= 1000),
  origin text not null default 'ai' check (origin in ('creator', 'ai', 'import')),
  status text not null default 'proposed' check (status in ('proposed', 'approved', 'creator_locked', 'retired')),
  decided_by text check (decided_by in ('creator')),
  decided_at timestamptz,
  created_at timestamptz not null default now(),
  check (status not in ('approved', 'creator_locked') or decided_by is not null),
  check (origin = 'creator' or char_length(planted_quote) > 0)
);
create index if not exists studio_story_threads_story_idx on public.studio_story_threads (story_id, state);

-- Intentional exceptions the creator approved, so a known oddity isn't reported again (a ghost who appears after
-- death, a character who pretends to know). A check is silenced only for this subject and this stretch of story.
create table if not exists public.studio_story_exceptions (
  id uuid primary key default gen_random_uuid(),
  story_id uuid not null references public.studio_stories(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  rule text not null check (rule in ('dead_character_active', 'knowledge_leak', 'object_two_places',
                                     'object_location_conflict', 'character_two_places', 'cause_after_effect')),
  subject_id uuid references public.studio_story_entities(id) on delete cascade,
  from_chapter int not null default 0,
  until_chapter int,                         -- inclusive; empty = to the end
  reason text not null check (char_length(reason) between 1 and 400),
  created_at timestamptz not null default now()
);
create index if not exists studio_story_exceptions_story_idx on public.studio_story_exceptions (story_id);

-- Which chapters each adapted episode covers, so "the state at episode 30" is a real point in the story.
create table if not exists public.studio_story_episodes (
  id uuid primary key default gen_random_uuid(),
  story_id uuid not null references public.studio_stories(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  number int not null check (number >= 1),
  chapter_from int not null check (chapter_from >= 0),
  chapter_to int not null,
  project_id uuid references public.studio_projects(id) on delete set null,
  created_at timestamptz not null default now(),
  unique (story_id, number),
  check (chapter_to >= chapter_from)
);

-- Owner can read their own story memory; only the server writes it (service role).
do $$
declare t text;
begin
  foreach t in array array['studio_stories', 'studio_story_entities', 'studio_story_facts', 'studio_story_events',
                           'studio_story_threads', 'studio_story_exceptions', 'studio_story_episodes'] loop
    execute format('alter table public.%I enable row level security', t);
    execute format('drop policy if exists "owner reads" on public.%I', t);
    execute format('create policy "owner reads" on public.%I for select using (auth.uid() = owner_id)', t);
  end loop;
end $$;
