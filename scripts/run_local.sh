#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
[ -d .venv ] || python3.11 -m venv .venv
source .venv/bin/activate
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt
docker compose up -d qdrant
python db_scripts/populate_db.py
streamlit run app.py
