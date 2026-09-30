"""Adaptador de fastembed (ONNX Runtime) a la interfaz `Embeddings` de LangChain.

Usa el mismo modelo del notebook (`paraphrase-multilingual-MiniLM-L12-v2`, 384
dimensiones) pero sin PyTorch: fastembed ejecuta el modelo cuantizado en ONNX
Runtime, lo que reduce el consumo de RAM de ~1-2 GB a ~250 MB. Esto es lo que
permite correr el servicio dentro del límite de 512 MB del plan gratuito de
Render (ver notas de arquitectura en README.md).

La normalización L2 se aplica a mano para igualar `encode_kwargs={"normalize_
embeddings": True}` del notebook (sentence-transformers normaliza por defecto
con ese flag; algunos modelos de fastembed ya devuelven vectores normalizados,
pero normalizamos explícitamente para no depender de ese detalle interno).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings

from . import config

# fastembed usa el nombre de modelo sin el prefijo "sentence-transformers/"
_FASTEMBED_MODEL_MAP = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
}


def _to_fastembed_name(model_name: str) -> str:
    return _FASTEMBED_MODEL_MAP.get(model_name, model_name)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FastEmbedEmbeddings(Embeddings):
    """Embeddings locales vía fastembed, compatibles con LangChain/Chroma."""

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding  # import perezoso: evita cargar ONNX al importar el módulo

        self._model = TextEmbedding(
            model_name=_to_fastembed_name(model_name),
            cache_dir=cache_dir,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = np.array(list(self._model.embed(texts)))
        vectors = _normalize(vectors)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = np.array(list(self._model.embed([text]))[0])
        vector = _normalize(vector.reshape(1, -1))[0]
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    """Instancia única y perezosa del modelo de embeddings (~250 MB en RAM)."""
    return FastEmbedEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        cache_dir=config.FASTEMBED_CACHE_PATH,
    )
