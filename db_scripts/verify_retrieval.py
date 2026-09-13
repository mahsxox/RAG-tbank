import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.reranker import rerank
from src.vector_store import (annotate_for_rerank, get_vector_store, normalize_numbers,
                              retrieve_candidates)

HIT_AT = 3
TARGET_RATE = 0.80


TARGETS = [

    ("Какая чистая прибыль Т-Банка за 2024 год?",                                  "55 262",    True),
    ("Сколько активов у банка на 31 декабря 2024?",                               "3 791 332", True),
    ("Какой чистый процентный доход в 2024 году?",                                "321 673",   True),
    ("Сколько кредитов, предоставленных клиентам, было на балансе на конец 2024 года?", "1 618 711", True),
    ("Какие комиссионные доходы получил банк в 2024 году?",                       "166 173",   True),
    ("Каков итого совокупный доход за 2024 год?",                                 "47 924",    True),
    ("Какой размер субординированных займов на 31 декабря 2024 года?",           "69 097",    True),

    ("Сколько всего активов у банка на конец 2025 года?",                         "5 610 408", True),
    ("Какую прибыль за год получил банк в 2025 году?",                            "122 068",   True),
    ("Каков объём средств клиентов на 31 декабря 2025 года?",                     "4 404 492", True),
    ("Сколько составили расходы по налогу на прибыль в 2025 году?",               "40 238",    True),

    ("Каков итого капитал банка на 31 декабря 2023 года?",                        "218 833",   True),
    ("Какая прибыль до налогообложения за 2023 год?",                             "59 077",    True),
    ("Сколько денежных средств и их эквивалентов было у банка на конец 2023 года?", "686 972",  True),
    ("Каковы административные и прочие операционные расходы в 2023 году?",        "117 660",   True),

    ("Стоимость активов в 2025 году?",                                            "5 610 408", False),
    ("Как изменилась выручка в 2023 году по сравнению с 2022?",                   None,        False),
]

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub("", normalize_numbers(text))


def contains(text: str, number: str) -> bool:
    return _norm(number) in _norm(text)


def page_text(doc) -> str:
    return doc.metadata.get("page_text") or doc.page_content


def rank_of(docs, number, parent=False):
    for i, d in enumerate(docs, start=1):
        if contains(page_text(d) if parent else d.page_content, number):
            return i
    return None


def unique_pages(docs):
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def main() -> int:
    run_llm = "--llm" in sys.argv
    quick = "--quick" in sys.argv
    store = get_vector_store()
    print(f"collection={store.collection}  TOP_K={config.TOP_K}  HIT_AT={HIT_AT}  reranker={config.RERANKER_MODEL}\n")

    scored_total = scored_hits = hits_at_1 = 0
    rows = []
    for query, expected, scored in TARGETS:
        if quick and not scored:
            continue
        t0 = time.perf_counter()
        c = retrieve_candidates(store, query)
        t1 = time.perf_counter()
        final = rerank(query, c["union"], config.TOP_K, texts=[annotate_for_rerank(d) for d in c["union"]])
        t2 = time.perf_counter()
        final_docs = [d for d, _ in final]
        pages = unique_pages(final_docs)

        tag = "" if scored else "  [диагностический, не в зачёт]"
        print(f"Q: {query}{tag}")
        print(f"   candidates={len(c['union'])}  retrieval {t1 - t0:.2f}s  rerank {t2 - t1:.2f}s")
        if expected:
            r_cd, r_cs, r_cf = rank_of(c["dense"], expected), rank_of(c["sparse"], expected), rank_of(final_docs, expected)
            r_pd, r_ps = rank_of(c["dense"], expected, True), rank_of(c["sparse"], expected, True)
            r_pf = rank_of(pages, expected, True)
            hit = r_pf is not None and r_pf <= HIT_AT
            if scored:
                scored_total += 1
                scored_hits += hit
                hits_at_1 += (r_pf == 1)
            print(f"   expected={expected}   chunk d/s/f={r_cd}/{r_cs}/{r_cf}   page d/s={r_pd}/{r_ps}   "
                  f"page@final={r_pf}  -> {'HIT' if hit else 'MISS'}")
            rows.append((query, expected, r_pf, hit, scored))
        d0, s0 = final[0]
        print(f"   top1: year={d0.metadata.get('report_year')} p.{d0.metadata.get('page')} score={s0:.3f} "
              f"| {_WS.sub(' ', d0.page_content)[:140]}...")
        if run_llm:
            from src.rag_pipeline import ask_question
            print(f"   LLM: {_WS.sub(' ', ask_question(query)['answer'])[:300]}")
        print()

    rate = scored_hits / scored_total if scored_total else 0.0
    print("=" * 78)
    print(f"hit@1 = {hits_at_1}/{scored_total}   hit@{HIT_AT} = {scored_hits}/{scored_total} = {rate:.0%}   "
          f"цель >= {TARGET_RATE:.0%}  -> {'ДОСТИГНУТА' if rate >= TARGET_RATE else 'НЕ ДОСТИГНУТА'}")
    misses = [r for r in rows if r[4] and not r[3]]
    if misses:
        print("Промахи:")
        for q, e, r, _, _ in misses:
            print(f"  - {q}  (эталон {e}, page@final={r})")
    return 0 if rate >= TARGET_RATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
