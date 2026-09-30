"""Configuración centralizada del sistema RAG, leída desde variables de entorno.

Los valores por defecto replican exactamente el pipeline validado en
`flujo_rag_groq.ipynb` (embeddings multilingües 384d, chunks de 500/50,
recuperación k=10, generación con Groq a temperatura 0).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


# --- Groq (LLM) ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE: float = _get_float("GROQ_TEMPERATURE", 0.0)

# --- Embeddings (fastembed / ONNX, sin torch) ---
# BAAI/bge-small-en-v1.5 (inglés, 384d) en vez del modelo multilingüe: el
# multilingüe carga ~607 MB en RAM solo para inicializar la sesión de ONNX
# Runtime (no cabe en los 512 MB de Render); este modelo carga ~224 MB. Los
# libros del corpus son en inglés, así que no se pierde calidad de contenido.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
FASTEMBED_CACHE_PATH: str = os.getenv("FASTEMBED_CACHE_PATH", "./.fastembed_cache")

# --- Chunking (idéntico al notebook) ---
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 50)

# --- Recuperación ---
RETRIEVER_K: int = _get_int("RETRIEVER_K", 10)

# --- Base vectorial (numpy: vectores + texto en disco, ver rag/vector_store.py) ---
PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./vector_index")

# --- Fuente de documentos ---
PDF_DIR: str = os.getenv("PDF_DIR", "./pdfs")

# --- Límites de la API del chat ---
MAX_MESSAGE_LENGTH: int = _get_int("MAX_MESSAGE_LENGTH", 2000)
