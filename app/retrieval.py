from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from app.settings import settings

DOCNO_Q_RE = re.compile(r"(?:№|N)\s*([0-9]{1,7})", flags=re.IGNORECASE)

_model: SentenceTransformer | None = None
_collection = None


@dataclass
class RetrievedChunk:
    text: str
    title: str
    url: str
    distance: float | None = None
    page: int | None = None
    chunk_index: int | None = None
    unique_key: str | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_dir)
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _make_query(q: str) -> str:
    if "e5" in settings.embedding_model.lower():
        return "query: " + q
    return q


def _extract_docno_digits(question: str) -> str | None:
    m = DOCNO_Q_RE.search(question or "")
    if not m:
        return None
    return m.group(1)


def _exact_docno_retrieve(question: str, k: int) -> list[RetrievedChunk]:
    """
    Exact retrieval by doc number digits via metadata filter.
    Requires chunk metadata: doc_number_digits.
    """
    digits = _extract_docno_digits(question)
    if not digits:
        return []

    col = _get_collection()

    got: dict[str, Any] = col.get(
        where={"doc_number_digits": digits},
        include=["documents", "metadatas"],
    )

    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    if not docs or not metas:
        return []

    chunks: list[RetrievedChunk] = []
    for doc, meta in zip(docs, metas):
        if not doc or not meta:
            continue
        chunks.append(
            RetrievedChunk(
                text=doc,
                title=meta.get("title") or "Untitled",
                url=meta.get("url") or "",
                distance=None,
                page=None,
                chunk_index=meta.get("chunk_index"),
                unique_key=meta.get("uniqueKey"),
            )
        )

    # Prefer reading order if chunk_index exists
    chunks.sort(key=lambda c: (c.chunk_index is None, c.chunk_index or 0))

    # Return first k chunks
    return chunks[:k]


def retrieve(question: str, k: int = 6) -> list[RetrievedChunk]:
    # 1) Exact doc-number retrieval first
    exact = _exact_docno_retrieve(question, k=k)
    if exact:
        return exact

    # 2) Semantic fallback
    col = _get_collection()
    model = _get_model()

    q_emb = model.encode([_make_query(question)], normalize_embeddings=True)[0].tolist()

    res: dict[str, Any] = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if not doc or not meta:
            continue
        out.append(
            RetrievedChunk(
                text=doc,
                title=meta.get("title") or "Untitled",
                url=meta.get("url") or "",
                distance=dist,
                page=None,
                chunk_index=meta.get("chunk_index"),
                unique_key=meta.get("uniqueKey"),
            )
        )
    return out
