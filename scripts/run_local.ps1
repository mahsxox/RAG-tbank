$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt
docker compose up -d qdrant
python db_scripts/populate_db.py
streamlit run app.py
