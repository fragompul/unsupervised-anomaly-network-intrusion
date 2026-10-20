# syntax=docker/dockerfile:1
# Stage 1: build the web app (artifacts in app/public/data are committed to the repo)
FROM node:22-alpine AS web
WORKDIR /app
COPY app/package.json app/package-lock.json ./
RUN npm ci
COPY app/ ./
RUN npm run build

# Stage 2: API + static app on one port
FROM python:3.12-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NOCTUA_APP=/srv/app
WORKDIR /srv
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
COPY --from=web /app/dist /srv/app
ENV NOCTUA_DATA=/srv/app/data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "ids_anomaly.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
