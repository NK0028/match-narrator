# --- build the React app ---
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- python runtime ---
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY commentator/ commentator/
COPY server/ server/
COPY models/spotter.pt models/thresholds.json models/
COPY --from=web /web/dist web/dist
# Hugging Face Spaces runs as uid 1000 and expects port 7860
RUN useradd -m -u 1000 user && mkdir -p outputs && chown -R user /app
USER user
ENV PORT=7860 DEMO_DIR=/app/web/dist/demo TTS_BACKEND=espeak YOLO_CONFIG_DIR=/tmp/ultralytics \
    MAX_UPLOAD_MB=60 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
