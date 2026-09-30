"""Adaptador de fastembed (ONNX Runtime) a la interfaz `Embeddings` de LangChain.

El notebook original usa `sentence-transformers` + PyTorch (~1-2 GB en RAM
solo para importarlo). Aquí se usa `fastembed` en su lugar, que ejecuta
modelos ONNX cuantizados sin depender de PyTorch. Aun así, se probó primero
el modelo multilingüe del notebook (`paraphrase-multilingual-MiniLM-L12-v2`)
y su sesión de ONNX Runtime por sí sola ya consume ~607 MB de RAM solo al
inicializarse — más que el límite de 512 MB del plan gratuito de Render. Por
eso el modelo por defecto es `BAAI/bge-small-en-v1.5` (inglés, 384d), cuya
sesión carga en ~224 MB. Como el corpus de este proyecto son libros técnicos
en inglés, no se pierde calidad de contenido; lo único que se sacrifica es
que una pregunta escrita en español puede emparejar un poco peor que una en
inglés (el modelo no fue entrenado para alinear ambos idiomas).

La normalización L2 se aplica a mano para que la similitud coseno en
`vector_store.py` (un simple producto punto) dé el resultado correcto,
sin depender de que el modelo ya normalice internamente.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings

from . import config


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FastEmbedEmbeddings(Embeddings):
    """Embeddings locales vía fastembed, compatibles con la interfaz de LangChain."""

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding  # import perezoso: evita cargar ONNX al importar el módulo

        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,  # evita que ONNX Runtime reserve un buffer por núcleo de CPU
        )

    def embed_documents(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        """Calcula embeddings en lotes pequeños en vez de todos de una vez.

        Un libro grande puede tener miles de fragmentos; pedirle a fastembed
        que los procese todos en una sola llamada genera un pico de memoria
        muy alto (se observó >3 GB con ~2700 fragmentos). Procesando de a
        `batch_size` fragmentos, el pico de memoria se mantiene acotado sin
        importar qué tan grande sea el libro.
        """
        if not texts:
            return []
        resultados: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            lote = texts[i : i + batch_size]
            vectores = np.array(list(self._model.embed(lote)))
            vectores = _normalize(vectores)
            resultados.extend(vectores.tolist())
        return resultados

    def embed_query(self, text: str) -> List[float]:
        vector = np.array(list(self._model.embed([text]))[0])
        vector = _normalize(vector.reshape(1, -1))[0]
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    """Instancia única y perezosa del modelo de embeddings (~225 MB en RAM)."""
    return FastEmbedEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        cache_dir=config.FASTEMBED_CACHE_PATH,
    )
