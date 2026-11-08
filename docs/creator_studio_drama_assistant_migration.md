# Creator Studio: bring the Drama Assistant into the creator product

## Audit date
2026-11-08

## Decision

Keep the existing AI Drama Assistant. Do not expose its current admin endpoints to customers and do not put an admin key in the browser. Move its useful story-development capability into the owner-scoped Creator Studio project model.

The intended promise is: a creator starts with an idea; VIYO helps develop the story and then guides the creator through a complete production that can be reviewed, rendered, and downloaded.

## What exists today

### 1. The existing AI Drama Assistant is an admin tool

Backend:
- `drama_creator.py` mounts at `/api/v1/admin/drama-creator`.
- Its endpoints use `Depends(_require_admin)`, which checks the shared `X-Admin-Key`.
- It works on legacy `series`, `series_characters`, and `series_scenes` rows.
- Idea generation creates a title, structured plot, and up to four characters; it can generate a cover, character reference sheets, and an episode storyboard.
- Story/storyboard writing uses OpenAI; cover and character reference images use Gemini.
- It relies on the external `drama_creator_schema.sql` migration.
- The existing assistant is a guided idea-to-story setup, not a persistent free-form assistant chat through all production stages.

Flutter:
- The screen is `lib/screens/admin/studio/new_drama_ai_screen.dart` in `ollie81/Viyo-`.
- It requires an `adminKey` and a legacy `seriesId`.
- `viyo_studio_home_screen.dart` presents the separate “New Drama — AI Assistant” admin entry point.
- That screen is under `screens/admin/studio`; it is not the customer Creator Studio experience.

### 2. A separate creator-facing backend already exists

Backend:
- `creator_studio.py` mounts at `/api/v1/studio`.
- It authenticates a signed-in, non-guest Supabase user through `get_current_user_id_no_guest`.
- Projects have `owner_id`; project reads and mutations call `_project(project_id, user_id)` / `_owned(...)`.
- It already has project creation/list/detail, characters, scenes, estimates, queued jobs, credit holds/capture/refunds, private storage and signed download URLs.
- Paid actions are estimated before jobs are queued; the job queue uses an idempotency key.
- Project kinds include short, episode, series, and film; target duration currently has a 600-second (10-minute) maximum.
- The assistant must use this project model and its existing pricing/queue system rather than creating another billing path.

Frontend gap:
- In the audited `ollie81/Viyo-` Flutter repo, the assistant screen is only wired to the admin Studio home flow.
- The creator-facing `/api/v1/studio` API is not represented by a discovered Creator Studio service/screen in that repo. A separate production web client, if any, must be located and audited before assuming this Flutter screen is the live web UI.

## Required target journey

1. **Start a production** — signed-in creator chooses drama/episode/film and enters a simple idea. VIYO must not ask them to write a finished script.
2. **Develop the story** — assistant proposes title, logline, premise, conflict, character motivations, secrets, opening hook, episode arc, and cliffhanger. It creates a cast draft with stable visual and voice descriptions.
3. **Review before spend** — creator can edit/approve story and cast. Show a clear estimate before any paid image/audio/video generation.
4. **Build the production plan** — turn the approved story into scenes and dialogue, reusing the existing Creator Studio character, voice, scene, and job models.
5. **Produce** — generate character references, scene visuals/video, dialogue/voices, captions, sound effects, and a suitable soundtrack using existing or separately approved generation jobs.
6. **Direct by conversation** — assistant accepts changes such as “make the opening stronger” or “change the final reveal,” previews the proposed changes, and asks for approval before regenerating paid assets. It must not silently replace completed work.
7. **Finish** — run technical quality checks, show the finished preview and total charge, then provide a creator-owned downloadable render.
8. **Continue a series** — preserve the approved story bible, character identity/voice settings, and episode continuity for subsequent episodes.

## Security and data rules

- Never expose `X-Admin-Key` to a browser, mobile app, or creator account.
- Every new assistant route must depend on the authenticated non-guest creator identity and resolve the project through `_project(project_id, user_id)` before reading or writing anything.
- Every character, scene, asset, job, and render lookup must be constrained to that verified project and owner. Return 404 for another user's IDs.
- Keep files private and return only the existing short-lived signed URLs.
- Use existing estimate -> confirmation -> idempotent job/credit hold -> worker -> capture/refund flow for every paid action. Never charge for a failed job.
- Separate free text/story edits from paid image, voice, video, and render operations; clearly label any action that incurs credits.
- Do not copy legacy `series_id` assumptions into the new project model. The Creator Studio uses `studio_projects`, `studio_characters`, and `studio_scenes`.
- Keep Watch coins separate from Studio credits.
- Do not change or remove the working admin tool until the creator version has passed tests and a real end-to-end check.

## Implementation plan

### Phase A — safe foundation
- Add creator-scoped assistant endpoints under `/api/v1/studio/projects/{project_id}/assistant`.
- Start with a story-development operation that validates ownership, uses the configured narrative model, validates structured output, and saves a versioned story bible/story draft on the project.
- Add a creator-edit/approve operation that records the approved story version. No image/video/audio generation should happen as a side effect of idea generation.
- Add rate limits, input/output bounds, structured error handling, cost logging, and tests for unauthenticated, guest, owner, and non-owner cases.
- Confirm the existing schema supports the story-bible fields; add a migration only if needed, and document it explicitly.

### Phase B — scene and production handoff
- Convert the approved story bible into the existing script/breakdown job workflow.
- Keep the creator's ability to inspect/edit scenes before paid generation.
- Add explicit change previews and replacement confirmation before regenerating already-produced scenes.

### Phase C — creator web experience
- Add the assistant as the first screen in the creator-facing Studio flow, not an admin-only menu item.
- Keep a persistent assistant panel/context alongside story, cast, scene and render views.
- Make it responsive for desktop and mobile web.
- Show estimates, current credit balance, job progress, failures/refunds, and download controls in the same flow.

### Phase D — release gate
- Run unit and API tests with fake providers.
- Test two different users and prove user A cannot read or mutate user B's project, characters, scenes, jobs, assets, or renders.
- Test guest rejection, duplicate job requests, insufficient credits, failed jobs/refunds, retries, and signed-link expiry.
- Run one real-provider end-to-end production only after provider billing/configuration is verified.
- Do not announce the feature as ready until a creator can complete and download a production through the web UI.

## First implementation boundary

The first code change should be the secure, owner-scoped story-development API and its tests. It should not be a direct conversion of the old admin router and should not start with a large UI rewrite. Once the API contract and tests are stable, connect the web client to it.

## Implementation status — 8 November 2026

Phase A has a first backend slice on branch `feat/creator-studio-assistant-api`:

- Added owner-scoped assistant state, generation, and story-approval endpoints under `/api/v1/studio/projects/{project_id}/assistant`.
- Generation routes into the existing `script` job, pricing estimate, idempotency, credit hold, worker, capture and refund flow. It requires the creator to echo the currently estimated credit amount; a changed estimate returns 409 without creating a job.
- Approval records an approval timestamp in the project's existing `story_bible`; it does not trigger asset generation.
- Added tests for owner isolation, guest/signed-out rejection, price mismatch without a credit hold, successful generation, and explicit story approval.

This is an API foundation only. It is not yet a conversational assistant, a web UI, a scene-planning UI, or a deployed feature. The repository tests have not been executed in this editing session; CI/review must verify them before merge. The existing story migration `creator_studio_story.sql` must be applied in the target environment for approval persistence.
