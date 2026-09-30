"""Base vectorial persistente (ChromaDB) — Paso 4 del notebook.

El índice se construye offline con `ingest.py` y se commitea al repositorio;
en el servicio web (`app.py`) solo se abre en modo lectura/escritura perezosa.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma

from . import config
from .embeddings import get_embeddings


class EmptyVectorStoreError(RuntimeError):
    """La colección de Chroma existe pero no tiene fragmentos indexados."""


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=config.PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )


def count_fragments() -> int:
    return get_vector_store()._collection.count()


def ensure_ready() -> None:
    """Lanza un error claro si aún no se indexó ningún libro."""
    if count_fragments() == 0:
        raise EmptyVectorStoreError(
            "La base de conocimiento está vacía. Copia PDFs en la carpeta "
            f"'{config.PDF_DIR}' y ejecuta: python ingest.py"
        )
