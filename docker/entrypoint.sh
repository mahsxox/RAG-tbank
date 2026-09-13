#!/bin/sh
set -e

until curl -sf "${QDRANT_URL}/readyz" >/dev/null 2>&1; do
  echo "waiting for qdrant at ${QDRANT_URL}..."
  sleep 2
done

case "$1" in
  ui)
    if [ "${AUTO_INDEX:-1}" = "1" ] && ls /app/data/*.pdf >/dev/null 2>&1; then
      python db_scripts/populate_db.py
    fi
    exec streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
    ;;
  index)
    shift
    exec python db_scripts/populate_db.py "$@"
    ;;
  verify)
    shift
    exec python db_scripts/verify_retrieval.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
