#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "Нет .env — скопируйте .env.example и впишите GIGACHAT_CREDENTIALS"; exit 1; }
docker compose up --build -d
echo "UI: http://localhost:8501   Qdrant: http://localhost:6333/dashboard"
docker compose logs -f app
