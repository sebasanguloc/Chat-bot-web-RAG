"""Base vectorial ligera, sin ChromaDB.

ChromaDB arrastra una cadena de dependencias muy pesada (grpcio, un cliente
de Kubernetes, varios paquetes de OpenTelemetry, protobuf, un stack tipo
FastAPI/uvicorn, etc.) pensada para un servidor vectorial distribuido de
producción. Solo importarla puede costar varios cientos de MB de RAM — más
que suficiente para agotar los 512 MB del plan gratuito de Render, incluso
antes de cargar el modelo de embeddings.

Como el corpus de este proyecto son unos pocos miles de fragmentos (no
decenas de millones), una búsqueda por fuerza bruta con numpy es igual de
rápida en la práctica y no necesita nada de ese peso: los vectores ya están
normalizados (ver `embeddings.py`), así que "similitud coseno" es solo un
producto punto — una multiplicación de matrices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from . import config


class EmptyVectorStoreError(RuntimeError):
    """El índice existe pero no tiene fragmentos indexados."""


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, Any]
    score: float


class NumpyVectorStore:
    """Guarda vectores + texto/metadatos en disco (dos archivos simples) y
    busca por similitud coseno con una multiplicación de matrices."""

    def __init__(self, persist_dir: str):
        self._dir = Path(persist_dir)
        self._vectors_path = self._dir / "vectors.npy"
        self._docs_path = self._dir / "documents.json"
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._docs: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._vectors_path.exists() and self._docs_path.exists():
            self._vectors = np.load(self._vectors_path)
            self._docs = json.loads(self._docs_path.read_text(encoding="utf-8"))

    def count(self) -> int:
        return len(self._docs)

    def add(
        self,
        vectors: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not vectors:
            return
        new_vectors = np.array(vectors, dtype=np.float32)
        self._vectors = (
            new_vectors
            if self._vectors.size == 0
            else np.vstack([self._vectors, new_vectors])
        )
        for text, meta in zip(texts, metadatas):
            self._docs.append({"text": text, "metadata": meta})
        self._persist()

    def _persist(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(self._vectors_path, self._vectors)
        self._docs_path.write_text(
            json.dumps(self._docs, ensure_ascii=False), encoding="utf-8"
        )

    def search(self, query_vector: list[float], k: int) -> list[SearchResult]:
        if self.count() == 0:
            return []
        q = np.array(query_vector, dtype=np.float32)
        scores = self._vectors @ q  # cosine similarity: los vectores ya están normalizados
        k = min(k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            SearchResult(
                text=self._docs[i]["text"],
                metadata=self._docs[i]["metadata"],
                score=float(scores[i]),
            )
            for i in top_idx
        ]


@lru_cache(maxsize=1)
def get_vector_store() -> NumpyVectorStore:
    return NumpyVectorStore(config.PERSIST_DIR)


def count_fragments() -> int:
    return get_vector_store().count()


def ensure_ready() -> None:
    """Lanza un error claro si aún no se indexó ningún libro."""
    if count_fragments() == 0:
        raise EmptyVectorStoreError(
            "La base de conocimiento está vacía. Copia PDFs en la carpeta "
            f"'{config.PDF_DIR}' y ejecuta: python ingest.py"
        )
