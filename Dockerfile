# Railway's default Python builder (railpack/nixpacks) does not include
# FFmpeg, which the video repurposing feature requires for cropping and
# caption burn-in. A Dockerfile is the reliable way to get FFmpeg into
# the container â€” Railway auto-detects and uses this instead of its
# default builder once it's present in the repo.
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# $PORT is injected by Railway at runtime â€” never hardcode a port number here.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
