FROM python:3.11-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY pyproject.toml ./
COPY ./app/ ./
COPY ./ml/ ./ml/
RUN pip install --no-cache-dir uv && \
    uv pip install --system .

ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV MODEL_DIR="/app/ml/model"

EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]