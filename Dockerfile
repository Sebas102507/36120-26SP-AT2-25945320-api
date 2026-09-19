FROM python:3.11-slim

# MODEL_DIR is kept outside /app: the application code has its own `models` package
# (app/models/, the response schemas), which must not be mixed with the model artefacts.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/models

WORKDIR /app

# Dependencies first, so this layer is cached while only application code changes.
# uv reads [tool.uv.sources] in pyproject.toml and installs nucleus-ml from TestPyPI.
COPY pyproject.toml ./
COPY ./app/ ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache .

# Production artefacts: preprocessing pipeline + promoted CCI and WHC bundles (make artefacts).
COPY ./models/ /models/

EXPOSE 8080
# Render provides PORT; 8080 is the local default.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
