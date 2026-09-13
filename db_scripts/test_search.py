import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.vector_store import get_vector_store, search_similar

_WS = re.compile(r"\s+")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = args[0] if args else "Какая чистая прибыль Т-Банка за 2024 год?"
    use_reranker = "--no-rerank" not in sys.argv
    k = config.TOP_K if use_reranker else config.FUSION_CANDIDATES

    store = get_vector_store()
    results = search_similar(store, query, top_k=k, use_reranker=use_reranker)
    print(f"query: {query}\nreranker: {use_reranker}\n")
    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        print(f"{i:2d}. score={score:8.4f} year={m.get('report_year')} p.{m.get('page')} {m.get('file')}")
        print("    " + _WS.sub(" ", doc.page_content)[:220] + "…")


if __name__ == "__main__":
    main()
