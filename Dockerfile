FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/ml/model

WORKDIR /app

# Dependencies first, so this layer is cached while only application code changes.
# uv reads [tool.uv.sources] in pyproject.toml and installs nucleus-ml from TestPyPI.
COPY pyproject.toml ./
COPY ./app/ ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache .

# Production artefacts: preprocessing pipeline + promoted CCI and WHC bundles.
COPY ./ml/ ./ml/

EXPOSE 8080
# Render provides PORT; 8080 is the local default.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
