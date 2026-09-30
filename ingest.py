"""Indexación offline de libros PDF a ChromaDB — Pasos 1-4 del notebook.

Uso:
    python ingest.py                  # incremental: solo indexa PDFs nuevos o modificados
    python ingest.py --rebuild        # borra el índice y reindexa todo desde cero
    python ingest.py --pdf-dir otra/carpeta

Requiere las dependencias de `requirements-ingest.txt` (incluye pypdf y
langchain-community, que NO se instalan en el servicio web de producción).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config
from rag.vector_store import get_vector_store

MANIFEST_NAME = "manifest.json"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(persist_dir: Path) -> dict:
    manifest_path = persist_dir / MANIFEST_NAME
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"libros": {}}


def _save_manifest(persist_dir: Path, manifest: dict) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa libros PDF en ChromaDB.")
    parser.add_argument(
        "--pdf-dir", default=config.PDF_DIR, help="Carpeta con los PDFs a indexar."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Borra el índice existente y reindexa todos los PDFs desde cero.",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    persist_dir = Path(config.PERSIST_DIR)

    if not pdf_dir.exists():
        print(f"[ERROR] La carpeta '{pdf_dir}' no existe.")
        sys.exit(1)

    pdf_files = sorted(p for p in pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No hay archivos .pdf en '{pdf_dir}'. Copia tus libros ahí y vuelve a correr este script.")
        sys.exit(0)

    if args.rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f"Índice anterior eliminado: {persist_dir}/")
        get_vector_store.cache_clear()

    manifest = _load_manifest(persist_dir)
    libros_indexados = manifest["libros"]

    pendientes = []
    for pdf_path in pdf_files:
        file_hash = _hash_file(pdf_path)
        registro = libros_indexados.get(pdf_path.name)
        if registro and registro.get("hash") == file_hash:
            print(f"  [=] {pdf_path.name}: sin cambios, se omite")
            continue
        pendientes.append((pdf_path, file_hash))

    if not pendientes:
        total = get_vector_store()._collection.count()
        print(f"\n[OK] No hay libros nuevos que indexar. Total en la colección: {total} fragmentos.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    print(f"Indexando {len(pendientes)} libro(s) nuevo(s) o modificado(s)...")
    vector_store = get_vector_store()

    for pdf_path, file_hash in pendientes:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        texto_total = sum(len(p.page_content.strip()) for p in pages)
        if texto_total == 0:
            print(
                f"  [!] {pdf_path.name}: no se extrajo texto (¿PDF escaneado sin OCR?). Se omite."
            )
            continue

        for page in pages:
            page.metadata["libro"] = pdf_path.name

        chunks = text_splitter.split_documents(pages)
        vector_store.add_documents(chunks)

        libros_indexados[pdf_path.name] = {
            "hash": file_hash,
            "paginas": len(pages),
            "fragmentos": len(chunks),
        }
        print(
            f"  [+] {pdf_path.name}: {len(pages)} páginas -> {len(chunks)} fragmentos "
            f"(factor {len(chunks)/max(len(pages),1):.1f}x)"
        )

    _save_manifest(persist_dir, manifest)

    total = vector_store._collection.count()
    print("\n[OK] Indexación completada.")
    print(f"  Libros en el manifiesto:  {len(libros_indexados)}")
    print(f"  Fragmentos en la colección: {total}")
    print(f"  Ubicación del índice:     {persist_dir}/")


if __name__ == "__main__":
    main()
