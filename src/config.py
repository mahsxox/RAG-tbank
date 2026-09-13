import os

import re

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

QDRANT_TIMEOUT = 60

COLLECTION_BASE = "tbank_ifrs"

INDEX_VERSION = "v2"

LEGACY_COLLECTION = "my_documents"


DENSE_VECTOR_NAME = "dense"

SPARSE_VECTOR_NAME = "bm25"


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

EMBEDDING_DEVICE = "cpu"

EMBEDDING_QUERY_PREFIX = "query: " if "e5" in EMBEDDING_MODEL else ""

EMBEDDING_PASSAGE_PREFIX = "passage: " if "e5" in EMBEDDING_MODEL else ""

EMBEDDING_BATCH_SIZE = 64


SPARSE_MODEL = "Qdrant/bm25"

BM25_LANGUAGE = "russian"

BM25_AVG_LEN = 120.0


RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

RERANKER_DEVICE = "cpu"

RERANKER_MAX_LENGTH = 384

RERANKER_BATCH_SIZE = 16

RERANKER_QUANTIZE = os.getenv("RERANKER_QUANTIZE", "1") == "1"


CHUNK_SIZE = 800

CHUNK_OVERLAP = 150


DENSE_CANDIDATES = 25

SPARSE_CANDIDATES = 15

FUSION_CANDIDATES = 40

TOP_K = 8

CONTEXT_PAGES = 5

YEAR_FILTER_ENABLED = True


BOILERPLATE_PAGE_FRACTION = 0.30

BOILERPLATE_CHUNK_FRACTION = 0.08

BOILERPLATE_MIN_LEN = 20

MIN_CHUNK_CHARS = 30

PAGE_TITLE_MAX_LEN = 120


UPSERT_BATCH_SIZE = 64


def collection_name() -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", EMBEDDING_MODEL.lower()).strip("_")

    return f"{COLLECTION_BASE}__{slug}__{INDEX_VERSION}"
