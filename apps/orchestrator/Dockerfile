FROM python:3.11-slim as base

# set workdir
WORKDIR /app

# system deps for building some packages (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# copy poetry lock & pyproject first for better layer caching
COPY pyproject.toml poetry.lock* /app/

# copy code
COPY . /app

# install poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE="false" \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# install dependencies
RUN poetry install --no-interaction --no-ansi

# Expose port used by FastAPI
EXPOSE 8000

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    FASTAPI_HOST=0.0.0.0 \
    FASTAPI_PORT=8000

# Run uvicorn
CMD ["uvicorn", "apps.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--limit-max-requests", "1000"]
