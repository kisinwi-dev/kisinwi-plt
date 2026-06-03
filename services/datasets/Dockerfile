FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev

COPY . /app

CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${DATASETS_SERVICE_PORT:-6500}"]