FROM python:3.14-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev

COPY . /app

# CMD ["sleep", "infinity"]

CMD ["uv", "run", "-m", "main"]