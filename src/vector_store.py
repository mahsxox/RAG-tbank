from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from src import config
from src.embedding import get_embedding_dim, get_embeddings
from src.reranker import rerank

log = logging.getLogger(__name__)


_NUM_GROUP_RE = re.compile(r"(?<=\d)[ \u00a0\u202f\u2009](?=\d{3}(?!\d))")
_YEAR_RE = re.compile(r"(?<!\d)(20[0-9]{2})(?!\d)")
_WS_RE = re.compile(r"[ \t\u00a0\u202f\u2009]+")


def normalize_numbers(text: str) -> str:
    return _NUM_GROUP_RE.sub("", text)


def _norm_line(line: str) -> str:
    return _WS_RE.sub(" ", line).strip()


def extract_years(text: str) -> List[int]:
    return sorted({int(y) for y in _YEAR_RE.findall(text)})


def _report_year_from_source(source: str) -> Optional[int]:
    years = _YEAR_RE.findall(Path(str(source)).name)
    return int(years[-1]) if years else None


@dataclass
class HybridStore:
    client: QdrantClient
    collection: str
    dense: object
    sparse: SparseTextEmbedding


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=config.QDRANT_TIMEOUT, check_compatibility=False)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(
        model_name=config.SPARSE_MODEL,
        language=config.BM25_LANGUAGE,
        avg_len=config.BM25_AVG_LEN,
    )


def get_vector_store(recreate: bool = False) -> HybridStore:
    client = get_qdrant_client()
    name = config.collection_name()
    dim = get_embedding_dim()

    if recreate and client.collection_exists(name):
        log.info("Удаляю коллекцию %s", name)
        client.delete_collection(name)

    if not client.collection_exists(name):
        log.info("Создаю коллекцию %s (dense dim=%d, sparse=%s)", name, dim, config.SPARSE_VECTOR_NAME)
        client.create_collection(
            collection_name=name,
            vectors_config={
                config.DENSE_VECTOR_NAME: models.VectorParams(size=dim, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                config.SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )
        client.create_payload_index(name, "report_year", models.PayloadSchemaType.INTEGER)
        client.create_payload_index(name, "source", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "file", models.PayloadSchemaType.KEYWORD)
    else:
        existing = client.get_collection(name).config.params.vectors[config.DENSE_VECTOR_NAME].size
        if existing != dim:
            raise RuntimeError(
                f"Коллекция {name} имеет dense dim={existing}, модель {config.EMBEDDING_MODEL} даёт {dim}. "
                f"Запустите populate_db.py (recreate=True)."
            )

    return HybridStore(client=client, collection=name, dense=get_embeddings(), sparse=get_sparse_model())


def delete_legacy_collection() -> None:
    client = get_qdrant_client()
    if client.collection_exists(config.LEGACY_COLLECTION):
        client.delete_collection(config.LEGACY_COLLECTION)
        log.info("Удалена старая коллекция %s", config.LEGACY_COLLECTION)


def _find_boilerplate_lines(docs: List[Document]) -> Dict[str, Set[str]]:
    has_pages = all("page" in d.metadata for d in docs)
    units_by_source: Dict[str, Set] = defaultdict(set)
    line_units: Dict[str, Dict[str, Set]] = defaultdict(lambda: defaultdict(set))

    for idx, d in enumerate(docs):
        src = str(d.metadata.get("source", ""))
        unit = d.metadata["page"] if has_pages else idx
        units_by_source[src].add(unit)
        for raw in d.page_content.splitlines():
            line = _norm_line(raw)
            if len(line) >= config.BOILERPLATE_MIN_LEN:
                line_units[src][line].add(unit)

    threshold = config.BOILERPLATE_PAGE_FRACTION if has_pages else config.BOILERPLATE_CHUNK_FRACTION
    result: Dict[str, Set[str]] = {}
    for src, lines in line_units.items():
        n = max(1, len(units_by_source[src]))
        result[src] = {ln for ln, units in lines.items() if len(units) / n >= threshold}
        if result[src]:
            log.info("%s: удаляю %d строк-колонтитулов", Path(src).name, len(result[src]))
    return result


def _strip_boilerplate(text: str, boiler: Set[str]) -> str:
    kept = []
    for raw in text.splitlines():
        line = _norm_line(raw)
        if not line or line in boiler:
            continue
        kept.append(line)
    return "\n".join(kept)


def _page_titles(docs: List[Document], boiler: Dict[str, Set[str]]) -> Dict[Tuple[str, object], str]:
    first_chunk: Dict[Tuple[str, object], Tuple[int, Document]] = {}
    for idx, d in enumerate(docs):
        key = (str(d.metadata.get("source", "")), d.metadata.get("page"))
        if key not in first_chunk or idx < first_chunk[key][0]:
            first_chunk[key] = (idx, d)
    titles: Dict[Tuple[str, object], str] = {}
    for key, (_, d) in first_chunk.items():
        cleaned = _strip_boilerplate(d.page_content, boiler.get(key[0], set()))
        for line in cleaned.splitlines():
            if 4 <= len(line) <= config.PAGE_TITLE_MAX_LEN and not re.fullmatch(r"[\d\s.,()%-]+", line):
                titles[key] = line
                break
    return titles


def annotate_for_rerank(doc: Document) -> str:
    m = doc.metadata
    year = m.get("report_year")
    head = f"Отчёт МСФО за {year} год." if year else ""
    title = m.get("page_title") or ""
    return "\n".join(x for x in (head, title, doc.page_content) if x)


def _point_id(source: str, page, chunk_idx: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{page}|{chunk_idx}|{h}"))


def add_documents(store: HybridStore, docs: List[Document]) -> int:
    docs = list(docs)
    boiler = _find_boilerplate_lines(docs)
    titles = _page_titles(docs, boiler)

    prepared: List[Tuple[str, str, str, dict]] = []
    for idx, d in enumerate(docs):
        source = str(d.metadata.get("source", ""))
        page = d.metadata.get("page")
        text = _strip_boilerplate(d.page_content, boiler.get(source, set()))
        if len(text) < config.MIN_CHUNK_CHARS:
            continue
        year = _report_year_from_source(source)
        title = titles.get((source, page), "")

        prefix = " ".join(x for x in (f"Отчёт МСФО за {year} год." if year else "", title) if x)
        text_dense = f"{prefix}\n{text}" if prefix else text
        text_bm25 = normalize_numbers(f"{title}\n{text}" if title else text)
        page_text = d.metadata.get("page_text")
        page_text = _strip_boilerplate(page_text, boiler.get(source, set())) if page_text else text
        payload = {
            "text": text,
            "page_text": page_text,
            "source": source,
            "file": Path(source).name,
            "page": page,
            "page_title": title,
            "report_year": year,
            "chunk_idx": idx,
        }
        prepared.append((_point_id(source, page, idx, text), text_dense, text_bm25, payload))

    missing_year = {p[3]["file"] for p in prepared if p[3]["report_year"] is None}
    if missing_year:
        log.warning("Не удалось определить год отчёта из имени файла: %s — фильтр по году для них работать не будет",
                    sorted(missing_year))

    total = 0
    bs = config.UPSERT_BATCH_SIZE
    for start in range(0, len(prepared), bs):
        batch = prepared[start:start + bs]
        dense_vecs = store.dense.embed_documents([b[1] for b in batch])
        sparse_vecs = list(store.sparse.embed([b[2] for b in batch]))
        points = [
            models.PointStruct(
                id=pid,
                vector={
                    config.DENSE_VECTOR_NAME: dv,
                    config.SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sv.indices.tolist(), values=sv.values.tolist()
                    ),
                },
                payload=payload,
            )
            for (pid, _, _, payload), dv, sv in zip(batch, dense_vecs, sparse_vecs)
        ]
        store.client.upsert(collection_name=store.collection, points=points, wait=True)
        total += len(points)
        log.info("upsert %d/%d", total, len(prepared))
    return total


def _year_filter(query: str) -> Optional[models.Filter]:
    years = extract_years(query)
    if not years:
        return None
    allowed = sorted({y for yr in years for y in (yr, yr + 1)})
    return models.Filter(
        must=[models.FieldCondition(key="report_year", match=models.MatchAny(any=allowed))]
    )


def _branch_query(store: HybridStore, vector, using: str, limit: int, flt: Optional[models.Filter]):
    return store.client.query_points(
        collection_name=store.collection,
        query=vector,
        using=using,
        limit=limit,
        query_filter=flt,
        with_payload=True,
    ).points


def _rrf_query(store: HybridStore, q_dense, q_sparse_vec, flt: Optional[models.Filter], limit: int):
    return store.client.query_points(
        collection_name=store.collection,
        prefetch=[
            models.Prefetch(query=q_dense, using=config.DENSE_VECTOR_NAME,
                            limit=config.DENSE_CANDIDATES, filter=flt),
            models.Prefetch(query=q_sparse_vec, using=config.SPARSE_VECTOR_NAME,
                            limit=config.SPARSE_CANDIDATES, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points


def _to_document(point) -> Document:
    payload = dict(point.payload or {})
    text = payload.pop("text", "")
    return Document(page_content=text, metadata=payload)


def retrieve_candidates(store: HybridStore, query: str) -> Dict[str, List[Document]]:
    q_dense = store.dense.embed_query(query)
    sp = next(iter(store.sparse.query_embed(normalize_numbers(query))))
    q_sparse_vec = models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())

    flt = _year_filter(query) if config.YEAR_FILTER_ENABLED else None
    dense = _branch_query(store, q_dense, config.DENSE_VECTOR_NAME, config.DENSE_CANDIDATES, flt)
    sparse = _branch_query(store, q_sparse_vec, config.SPARSE_VECTOR_NAME, config.SPARSE_CANDIDATES, flt)
    if flt is not None and len(dense) + len(sparse) < config.TOP_K:
        log.info("Фильтр по году почти ничего не дал — повтор без фильтра")
        dense = _branch_query(store, q_dense, config.DENSE_VECTOR_NAME, config.DENSE_CANDIDATES, None)
        sparse = _branch_query(store, q_sparse_vec, config.SPARSE_VECTOR_NAME, config.SPARSE_CANDIDATES, None)

    seen = set()
    union = []
    for p in list(dense) + list(sparse):
        if p.id in seen:
            continue
        seen.add(p.id)
        union.append(p)

    return {
        "dense": [_to_document(p) for p in dense],
        "sparse": [_to_document(p) for p in sparse],
        "union": [_to_document(p) for p in union],
        "_rrf_args": (q_dense, q_sparse_vec, flt),
    }


def search_similar(
    store: HybridStore,
    query: str,
    top_k: Optional[int] = None,
    use_reranker: bool = True,
) -> List[Tuple[Document, float]]:
    top_k = top_k or config.TOP_K
    cands = retrieve_candidates(store, query)
    if use_reranker:
        docs = cands["union"]
        if not docs:
            return []
        return rerank(query, docs, top_k, texts=[annotate_for_rerank(d) for d in docs])

    q_dense, q_sparse_vec, flt = cands["_rrf_args"]
    points = _rrf_query(store, q_dense, q_sparse_vec, flt, top_k)
    if flt is not None and len(points) < top_k:
        points = _rrf_query(store, q_dense, q_sparse_vec, None, top_k)
    return [(_to_document(p), float(p.score)) for p in points]
