FROM python:3.11-slim

# curl: container health check. build tools: fallback for any dependency
# without a prebuilt wheel on this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ui.py binds 127.0.0.1 by default; inside the container it must listen on all
# interfaces so the published port works. docker-compose.yml publishes it on
# the host's loopback only.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    UI_HOST=0.0.0.0 \
    UI_PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user; chroma_db holds the vector index, the web-search
# cache and the (opt-in) trend store.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/chroma_db \
    && chown -R app:app /app/chroma_db
USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
  CMD curl -fsS http://127.0.0.1:7860/ >/dev/null || exit 1

CMD ["python", "ui.py"]
