import logging

from pathlib import Path

from typing import Dict, List


from qdrant_client import models


from src import config

from src.loader import load_pdf_with_fitz, split_documents

from src.vector_store import HybridStore, add_documents


log = logging.getLogger(__name__)


def index_pdf(store: HybridStore, path: Path) -> int:
    pages = load_pdf_with_fitz(str(path))

    for p in pages:
        p.metadata.setdefault("source", str(path))

    chunks = split_documents(pages)

    written = add_documents(store, chunks)

    log.info("%s: %d страниц, %d чанков, %d точек", path.name, len(pages), len(chunks), written)

    return written


def index_directory(store: HybridStore, directory: Path = config.DATA_DIR) -> Dict[str, int]:
    result = {}

    for pdf in sorted(directory.glob("*.pdf")):
        result[pdf.name] = index_pdf(store, pdf)

    return result


def remove_file(store: HybridStore, file_name: str) -> None:
    store.client.delete(

        collection_name=store.collection,

        points_selector=models.FilterSelector(

            filter=models.Filter(must=[models.FieldCondition(key="file", match=models.MatchValue(value=file_name))])

        ),

        wait=True,

    )


def indexed_files(store: HybridStore) -> Dict[str, Dict]:
    files: Dict[str, Dict] = {}

    offset = None

    while True:
        points, offset = store.client.scroll(

            collection_name=store.collection,

            limit=1000,

            offset=offset,

            with_payload=["file", "page", "report_year"],

            with_vectors=False,

        )

        for p in points:
            f = p.payload.get("file")

            entry = files.setdefault(f, {"chunks": 0, "pages": set(), "report_year": p.payload.get("report_year")})

            entry["chunks"] += 1

            entry["pages"].add(p.payload.get("page"))

        if offset is None:
            break

    for entry in files.values():
        entry["pages"] = len(entry["pages"])

    return files


def points_count(store: HybridStore) -> int:
    return store.client.count(collection_name=store.collection, exact=True).count
