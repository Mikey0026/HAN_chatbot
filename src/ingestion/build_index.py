"""
Ingestion pipeline
Loads every PDF in data/raw/, splits each page into overlapping chunks,
embeds the chunks with a local Ollama embedding model, and persists the
result in a ChromaDB collection on disk.

Run with:
    python -m src.ingestion.build_index

Re-running is safe, the existing collection is wiped and rebuilt from
scratch so we never accumulate stale chunks during development.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.utils.config import load_config, project_root


def load_pdfs(raw_dir: Path) -> List[Document]:
    """Load every .pdf file in `raw_dir` as a list of LangChain Documents.

    Each Document corresponds to one page and carries `source` (file name)
    and `page` (0-indexed page number) in its metadata. We add a clean
    `source_file` field too, since the loader's default `source` is the
    full absolute path which is noisy in citations.
    """
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {raw_dir}. "
            "Drop the official HAN documents in there before running ingestion."
        )

    docs: List[Document] = []
    for pdf_path in pdf_paths:
        print(f"[ingestion] Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for p in pages:
            p.metadata["source_file"] = pdf_path.name
            p.metadata["source_type"] = "manual"
        docs.extend(pages)
        print(f"[ingestion]   → {len(pages)} pages")
    return docs


def chunk_documents(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Split page-level Documents into overlapping chunks.

    The separator hierarchy ("\\n\\n", "\\n", ". ", " ") tries paragraph
    breaks first and falls back to softer boundaries — this keeps related
    sentences together when possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingestion] Split {len(docs)} pages into {len(chunks)} chunks.")
    return chunks


def build_index() -> None:
    """End-to-end ingestion: PDFs → chunks → embeddings → persisted Chroma."""
    cfg = load_config()
    root = project_root()

    raw_dir = root / cfg["raw_pdf_dir"]
    store_dir = root / cfg["vector_store_dir"]

    # Clean rebuild — avoids stale chunks during iterative development.
    if store_dir.exists():
        print(f"[ingestion] Removing existing vector store at {store_dir}")
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    docs = load_pdfs(raw_dir)

    # 2. Chunk
    chunks = chunk_documents(
        docs,
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )

    # 3. Embed + persist
    print(f"[ingestion] Embedding with {cfg['embedding_model']} via Ollama ...")
    embeddings = OllamaEmbeddings(model=cfg["embedding_model"])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=cfg["collection_name"],
        persist_directory=str(store_dir),
    )

    count = vector_store._collection.count()
    print(f"[ingestion] Done. {count} chunks stored in {store_dir}.")


if __name__ == "__main__":
    build_index()
