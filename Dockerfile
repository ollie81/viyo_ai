# Railway's default Python builder (railpack/nixpacks) does not include
# FFmpeg, which the video repurposing feature requires for cropping and
# caption burn-in. A Dockerfile is the reliable way to get FFmpeg into
# the container â€” Railway auto-detects and uses this instead of its
# default builder once it's present in the repo.
FROM python:3.11-slim

# fonts-dejavu-core is required, not optional: ffmpeg/libass depend on
# libfontconfig1 (the font-lookup *library*) but that pulls in zero
# actual font files with --no-install-recommends. Without a real font
# installed, both the existing subtitle burn-in (FontName=Arial-Bold,
# which doesn't exist here either — fontconfig just substitutes
# whatever default it can find) and the drawtext-based quote card
# feature fail to render any text at all.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Voice-embedding model for speaker identification (26 MB). Best effort: if the
# download fails the feature turns itself off and jobs still run.
RUN python -c "import speakers; speakers.ensure_model()" || true

# $PORT is injected by Railway at runtime â€” never hardcode a port number here.
# The same image runs two Railway services: the API (default) and the
# AI Repurposer worker, selected by setting VIYO_PROCESS=worker on that
# service. exec makes the chosen process PID 1 so Railway's SIGTERM
# reaches it directly (the worker hands its job back on SIGTERM).
CMD if [ "$VIYO_PROCESS" = "worker" ]; then exec python repurpose_worker.py; else exec uvicorn main:app --host 0.0.0.0 --port $PORT; fi
