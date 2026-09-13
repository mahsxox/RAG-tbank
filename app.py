import logging
from pathlib import Path

import streamlit as st

from src import config
from src.indexing import index_pdf, indexed_files, points_count, remove_file
from src.rag_pipeline import ask_question, get_store

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="T-Bank IFRS RAG", layout="wide")


@st.cache_resource(show_spinner="Загружаю модели и подключаюсь к Qdrant…")
def warmup():
    store = get_store()
    from src.reranker import get_reranker
    get_reranker()
    return store


store = warmup()

with st.sidebar:
    st.header("База документов")
    st.caption(f"Коллекция `{store.collection}`")
    st.caption(f"Эмбеддер `{config.EMBEDDING_MODEL}`")
    st.caption(f"Reranker `{config.RERANKER_MODEL}`")
    st.metric("Чанков в индексе", points_count(store))

    files = indexed_files(store)
    if files:
        st.subheader("Проиндексировано")
        for name, info in sorted(files.items()):
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{name}** · {info['pages']} стр. · {info['chunks']} чанков · год {info['report_year']}")
            if col2.button("✕", key=f"del_{name}", help="Удалить из индекса"):
                remove_file(store, name)
                st.rerun()
    else:
        st.info("Индекс пуст. Загрузите PDF или проиндексируйте папку data/.")

    st.subheader("Добавить файлы")
    uploads = st.file_uploader("PDF отчётности", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Загрузить и проиндексировать", type="primary"):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        for up in uploads:
            target = config.DATA_DIR / up.name
            target.write_bytes(up.getbuffer())
            with st.spinner(f"Индексирую {up.name}…"):
                if up.name in files:
                    remove_file(store, up.name)
                n = index_pdf(store, target)
            st.success(f"{up.name}: {n} чанков")
        st.rerun()

    local_pdfs = sorted(p.name for p in config.DATA_DIR.glob("*.pdf")) if config.DATA_DIR.exists() else []
    not_indexed = [n for n in local_pdfs if n not in files]
    if not_indexed:
        st.caption(f"В data/ есть непроиндексированные файлы: {', '.join(not_indexed)}")
        if st.button("Проиндексировать папку data/"):
            for name in not_indexed:
                with st.spinner(f"Индексирую {name}…"):
                    index_pdf(store, config.DATA_DIR / name)
            st.rerun()

    st.divider()
    top_k = st.slider("Чанков после reranker'а", 3, 15, config.TOP_K)

st.title("Вопросы по отчётности Т-Банка (МСФО)")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Источники"):
                for s in msg["sources"]:
                    st.markdown(f"**[{s['rank']}] {s['file']} · стр. {s['page']} · отчёт {s['report_year']} · score {s['score']}**")
                    st.text(s["text"][:1500])

question = st.chat_input("Например: Какая чистая прибыль Т-Банка за 2024 год?")
if question:
    if not files:
        st.warning("Индекс пуст — сначала загрузите документы.")
        st.stop()
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Ищу и формирую ответ…"):
            result = ask_question(question, top_k=top_k)
        st.write(result["answer"])
        with st.expander("Источники"):
            for s in result["sources"]:
                st.markdown(f"**[{s['rank']}] {s['file']} · стр. {s['page']} · отчёт {s['report_year']} · score {s['score']}**")
                st.text(s["text"][:1500])
    st.session_state.history.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
