FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev

COPY . .

CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${AGENT_OFFICE_PORT:-6400}"]