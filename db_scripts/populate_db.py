import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.indexing import index_directory, points_count
from src.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("populate_db")


def main() -> int:
    recreate = "--recreate" in sys.argv
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        log.error("В %s нет PDF", config.DATA_DIR)
        return 1
    store = get_vector_store(recreate=recreate)
    existing = points_count(store)
    if existing and not recreate:
        log.info("Коллекция %s уже содержит %d точек, пропускаю (--recreate для полной переиндексации)",
                 store.collection, existing)
        return 0
    result = index_directory(store, config.DATA_DIR)
    log.info("Готово: %s, всего %d точек", result, points_count(store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
