import re

from pathlib import Path

from typing import List


import fitz

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter


from src import config


_SPACES_RE = re.compile(r"[ \t\u00a0\u202f\u2009]+")

_BLANK_LINES_RE = re.compile(r"\n{2,}")


def clean_page_text(text: str) -> str:
    lines = [_SPACES_RE.sub(" ", ln).strip() for ln in text.splitlines()]

    text = "\n".join(ln for ln in lines if ln)

    return _BLANK_LINES_RE.sub("\n", text).strip()


def load_pdf_with_fitz(path: str) -> List[Document]:
    path = str(path)

    docs: List[Document] = []

    with fitz.open(path) as pdf:
        for page_idx, page in enumerate(pdf):
            text = clean_page_text(page.get_text("text", sort=True))

            if not text:
                continue

            docs.append(

                Document(

                    page_content=text,

                    metadata={

                        "source": path,

                        "file": Path(path).name,

                        "page": page_idx + 1,

                        "page_text": text,

                    },

                )

            )

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(

        chunk_size=config.CHUNK_SIZE,

        chunk_overlap=config.CHUNK_OVERLAP,

        separators=["\n", " ", ""],

        keep_separator=False,

    )

    return splitter.split_documents(docs)
