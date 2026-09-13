$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path ".env")) { Write-Error "Нет .env — скопируйте .env.example и впишите GIGACHAT_CREDENTIALS" }
docker compose up --build -d
Write-Host "UI: http://localhost:8501   Qdrant: http://localhost:6333/dashboard"
docker compose logs -f app
