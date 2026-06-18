FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download Whisper model — avoids first-run internet fetch
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')"
COPY . .
RUN chmod +x entrypoint.sh
EXPOSE 8001
ENTRYPOINT ["/app/entrypoint.sh"]
