import os

from functools import lru_cache


from dotenv import load_dotenv

from langchain_community.chat_models import GigaChat


from src import config


ENV_PATH = config.BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=False)


def get_credentials() -> str:
    value = (os.getenv("GIGACHAT_CREDENTIALS") or "").strip().strip('"').strip("'")

    if not value:
        raise RuntimeError(

            "GIGACHAT_CREDENTIALS не найдена.\n"

            f"  Искал файл: {ENV_PATH} (существует: {ENV_PATH.exists()})\n"

            "  Формат строки в .env:  GIGACHAT_CREDENTIALS=MDA4...==   (без 'export', без пробелов вокруг '=')\n"

            "  Проверка: python -c \"from src.generator import get_credentials as g; print(g()[:8]+'…')\""

        )

    return value


@lru_cache(maxsize=1)

def get_llm() -> GigaChat:
    return GigaChat(

        credentials=get_credentials(),

        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),

        model=os.getenv("GIGACHAT_MODEL", "GigaChat"),

        verify_ssl_certs=False,

        temperature=0.0,

        timeout=60,

    )
