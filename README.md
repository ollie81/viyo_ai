VIYO AI Backend

Python FastAPI backend powering VIYO's creator platform, video features and AI-assisted production tools.

Repository: https://github.com/ollie81/viyo_ai

The VIYO AI Backend provides APIs for the VIYO mobile and web applications. It connects authentication, creator features, video processing, payment systems and AI-powered workflows.

Main capabilities

Creator and video platform

- Creator posts and content discovery
- Series and episode management
- Likes, comments and interactions
- Messaging and notifications
- Creator analytics and engagement tools
- Video upload and delivery integrations
- Creator-owned production downloads

AI creator tools

- AI content ideas and caption improvements
- AI Creator Coach with image-analysis capabilities
- Video repurposing workflows
- Automated transcription and clip selection
- Caption generation and video rendering
- Background processing with job queues

VIYO Creator Studio

The creator-facing Studio API supports:

- Production projects for dramas, episodes and films
- Script development and scene breakdown
- Character casting and reference images
- Character voice assignments and dialogue audio
- AI-generated scene images and video
- Scene rendering into finished MP4 productions
- Production cost estimates and credit accounting
- Job status tracking and failure handling

The backend separates Studio credits from VIYO Watch coins. Paid operations are estimated before submission, credits are held when jobs are queued, and unused credits are returned after completion. Failed jobs have a credit-return mechanism.

Payments and subscriptions

The backend contains integrations for:

- Stripe
- Paystack
- Flutterwave
- Lemon Squeezy
- Google Play billing
- Subscription handling and payment webhooks

Provider availability and live transaction flows depend on account eligibility and deployment configuration.

Architecture

Layer| Technology
API| Python, FastAPI
Database| Supabase PostgreSQL
Authentication| Supabase JWT verification
Storage| Supabase Storage
AI integrations| Google Gemini, OpenAI
Video processing| FFmpeg, MoviePy, OpenCV
Speech processing| Faster-Whisper
Video delivery| Bunny Stream
Background processing| Python workers
Deployment| Railway
Monitoring| Sentry integration

Background workers

The backend uses separate processing workflows for different types of jobs.

- Repurposer worker: Processes uploaded source videos, identifies useful clips and generates edited outputs.
- Creator Studio worker: Handles queued AI generation and video-rendering jobs.

The worker system includes job claiming, heartbeats, retry handling and recovery mechanisms to reduce duplicate processing.

Security and reliability

The codebase includes:

- Supabase JWT-based user authentication
- Ownership checks for creator resources
- Private file storage and signed download links
- Payment webhook verification
- Server-side credit calculations
- Idempotency protections for payment and credit operations
- Job failure and retry handling
- Rate-limiting and error-monitoring integrations

Local development

Requirements:

- Python 3.11
- FFmpeg
- Supabase project and credentials
- API keys for the AI services being tested

Install dependencies:

pip install -r requirements.txt

Run the API:

uvicorn main:app --reload --port 8000

The service uses environment variables for credentials and configuration. Consult the deployment configuration before running production services. Never commit secret keys or service-role credentials.

Database setup

Some VIYO features depend on SQL schema files that must be applied to the Supabase database.

These include Creator Studio tables, credit accounting and job-queue functions. The backend repository does not automatically apply every schema change.

Review and apply the relevant SQL files before enabling their associated production features.

Development status

Under active development.

The repository contains substantial API functionality, AI processing modules, creator workflows and automated tests.

However, passing tests with simulated providers does not prove that every real AI integration, payment provider or deployed workflow is ready for public use. Live provider testing, database setup and production configuration remain important validation steps.

---

Part of the VIYO platform, developed by the project owner.
