"""Chat-bot web con arquitectura RAG (Flask + Groq + fastembed + numpy).

Rutas:
    GET  /            -> interfaz de chat
    POST /api/chat     -> {"message": str} -> {"respuesta", "fuentes", "tokens_contexto_aprox"}
    GET  /api/health    -> estado del servicio (también sirve de "warm-up")
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

from rag import config
from rag.pipeline import rag_pipeline
from rag.vector_store import EmptyVectorStoreError, count_fragments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    try:
        fragmentos = count_fragments()
        return jsonify(
            {
                "status": "ok" if fragmentos > 0 else "sin_indice",
                "fragmentos": fragmentos,
                "modelo_llm": config.GROQ_MODEL,
                "modelo_embeddings": config.EMBEDDING_MODEL,
            }
        )
    except Exception as exc:  # primera carga del modelo puede fallar por config
        logger.exception("Fallo en /api/health")
        return jsonify({"status": "error", "detalle": str(exc)}), 503


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    mensaje = (data.get("message") or "").strip()

    if not mensaje:
        return jsonify({"error": "El mensaje no puede estar vacío."}), 400
    if len(mensaje) > config.MAX_MESSAGE_LENGTH:
        return jsonify(
            {"error": f"El mensaje supera el límite de {config.MAX_MESSAGE_LENGTH} caracteres."}
        ), 400

    try:
        resultado = rag_pipeline(mensaje)
        return jsonify(resultado)
    except EmptyVectorStoreError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Error procesando /api/chat")
        return jsonify({"error": "Ocurrió un error inesperado al generar la respuesta."}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
