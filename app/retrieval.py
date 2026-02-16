from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from app.settings import settings


@dataclass
class RetrievedChunk:
    text: str
    title: str
    url: str
    distance: float | None = None
    page: int | None = None


_model: SentenceTransformer | None = None
_collection = None


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


def retrieve(question: str, k: int = 6) -> list[RetrievedChunk]:
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
            )
        )
    return out
