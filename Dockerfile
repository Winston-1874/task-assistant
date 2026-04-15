FROM python:3.12-slim

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /usr/local/bin/uv

WORKDIR /app

# Copier les fichiers de dépendances en premier (cache Docker)
COPY pyproject.toml uv.lock ./

# Installer les dépendances sans le projet lui-même
RUN uv sync --frozen --no-dev --no-install-project

# Copier le code source
COPY app/ ./app/
COPY alembic.ini ./
COPY migrations/ ./migrations/

# Répertoire de données (monté en volume en prod)
RUN mkdir -p data

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Alembic appliqué au démarrage, puis uvicorn
CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"]
