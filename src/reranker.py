import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src import config


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ce = CrossEncoder(
        config.RERANKER_MODEL,
        max_length=config.RERANKER_MAX_LENGTH,
        device=config.RERANKER_DEVICE,
    )
    if config.RERANKER_QUANTIZE and config.RERANKER_DEVICE == "cpu":
        torch.quantization.quantize_dynamic(ce.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return ce


def rerank(
    query: str,
    docs: List[Document],
    top_k: int,
    texts: Optional[List[str]] = None,
) -> List[Tuple[Document, float]]:
    if not docs:
        return []
    texts = texts or [d.page_content for d in docs]
    pairs = [(query, t) for t in texts]
    scores = get_reranker().predict(pairs, batch_size=config.RERANKER_BATCH_SIZE)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    order = np.argsort(-scores)
    return [(docs[i], float(scores[i])) for i in order[:top_k]]
