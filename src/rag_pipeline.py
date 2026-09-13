from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.generator import get_llm
from src.vector_store import HybridStore, get_vector_store, search_similar

_store: Optional[HybridStore] = None


def get_store() -> HybridStore:
    global _store
    if _store is None:
        _store = get_vector_store()
    return _store


def select_pages(results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    seen = set()
    pages: List[Tuple[Document, float]] = []
    for doc, score in results:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        pages.append((doc, score))
        if len(pages) >= config.CONTEXT_PAGES:
            break
    return pages


def build_context(results: List[Tuple[Document, float]]) -> str:
    parts = []
    for i, (doc, _score) in enumerate(select_pages(results), start=1):
        m = doc.metadata
        header = (
            f"[{i}] Отчёт МСФО за {m.get('report_year', '?')} год, "
            f"файл {m.get('file') or Path(str(m.get('source', ''))).name}, стр. {m.get('page', '?')}"
        )
        body = m.get("page_text") or doc.page_content
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


_SYSTEM = (
    "Ты отвечаешь на вопросы по финансовой отчётности Т-Банка по МСФО. "
    "Суммы в отчётности указаны в миллионах рублей. "
    "Отвечай только на основе приведённых фрагментов; если нужного значения в них нет, скажи об этом. "
    "В ответе укажи число как в источнике и номер фрагмента в квадратных скобках."
)


def ask_question(question: str, top_k: Optional[int] = None) -> Dict:
    results = search_similar(get_store(), question, top_k=top_k or config.TOP_K)
    context = build_context(results)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Фрагменты отчётности:\n\n{context}\n\nВопрос: {question}"),
    ]
    answer = get_llm().invoke(messages).content
    return {
        "answer": answer,
        "sources": [
            {
                "rank": i,
                "score": round(score, 4),
                "file": doc.metadata.get("file"),
                "page": doc.metadata.get("page"),
                "report_year": doc.metadata.get("report_year"),
                "text": doc.metadata.get("page_text") or doc.page_content,
            }
            for i, (doc, score) in enumerate(select_pages(results), start=1)
        ],
    }
