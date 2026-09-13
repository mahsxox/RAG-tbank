from functools import lru_cache

from typing import List


from langchain_huggingface import HuggingFaceEmbeddings


from src import config


class PrefixedEmbeddings:
    def __init__(self, base: HuggingFaceEmbeddings, query_prefix: str, passage_prefix: str):
        self._base = base

        self._qp = query_prefix

        self._pp = passage_prefix


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._base.embed_documents([self._pp + t for t in texts])


    def embed_query(self, text: str) -> List[float]:
        return self._base.embed_query(self._qp + text)


@lru_cache(maxsize=1)

def get_embeddings() -> PrefixedEmbeddings:
    base = HuggingFaceEmbeddings(

        model_name=config.EMBEDDING_MODEL,

        model_kwargs={"device": config.EMBEDDING_DEVICE},

        encode_kwargs={

            "normalize_embeddings": True,

            "batch_size": config.EMBEDDING_BATCH_SIZE,

        },

    )

    return PrefixedEmbeddings(base, config.EMBEDDING_QUERY_PREFIX, config.EMBEDDING_PASSAGE_PREFIX)


@lru_cache(maxsize=1)

def get_embedding_dim() -> int:
    return len(get_embeddings().embed_query("проверка размерности"))
