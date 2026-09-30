"""Pipeline RAG de extremo a extremo — equivalente a `rag_pipeline()` del
notebook (Paso 6 y 7), reempaquetado como módulo importable por Flask."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import TypedDict

from langchain_groq import ChatGroq

from . import config
from .embeddings import get_embeddings
from .prompt import prompt_template
from .vector_store import ensure_ready, get_vector_store


class FuenteRecuperada(TypedDict):
    libro: str
    pagina: int | str
    fragmento: str


class RagResultado(TypedDict):
    pregunta: str
    respuesta: str
    fuentes: list[FuenteRecuperada]
    tokens_contexto_aprox: int


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "Falta GROQ_API_KEY. Defínela en el archivo .env (ver .env.example)."
        )
    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=config.GROQ_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
    )


def _construir_contexto(results) -> str:
    return "\n\n---\n\n".join(
        f"[Fuente: {os.path.basename(r.metadata.get('libro', '?'))}"
        f" — Pág. {r.metadata.get('page', '?')}]\n{r.text}"
        for r in results
    )


def rag_pipeline(pregunta: str, k: int | None = None) -> RagResultado:
    """Consulta → recuperación (fastembed + búsqueda numpy) → prompt aumentado → Groq."""
    ensure_ready()

    k = k or config.RETRIEVER_K
    query_vector = get_embeddings().embed_query(pregunta)
    results = get_vector_store().search(query_vector, k=k)

    contexto = _construir_contexto(results)
    prompt = prompt_template.invoke({"context": contexto, "question": pregunta})
    respuesta = get_llm().invoke(prompt).content

    fuentes: list[FuenteRecuperada] = [
        {
            "libro": os.path.basename(r.metadata.get("libro", "?")),
            "pagina": r.metadata.get("page", "?"),
            "fragmento": r.text[:300],
        }
        for r in results
    ]

    return {
        "pregunta": pregunta,
        "respuesta": respuesta,
        "fuentes": fuentes,
        "tokens_contexto_aprox": len(contexto) // 4,
    }
