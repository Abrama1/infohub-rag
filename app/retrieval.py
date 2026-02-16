from __future__ import annotations

import re
from collections import defaultdict
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
    lexical_score: int = 0
    mode: str = "semantic"  # "docno_exact" | "semantic"


# --- basic tokenization/keyword logic (no extra deps) ---
_STOPWORDS = {
    # Georgian (tiny list, enough to reduce noise)
    "და", "თუ", "ან", "რომ", "რა", "როდის", "სად", "როგორ", "რამდენი", "ვის", "ვინ",
    "არის", "იყო", "იქნება", "ეს", "ის", "მათ", "ჩვენ", "თქვენ", "მის", "მისი", "მათი",
    "შესახებ", "მიხედვით", "დოკუმენტი", "დოკუმენტით", "მუხლი", "მუხლით", "კანონი", "კოდექსი",
}

_WORD_RE = re.compile(r"[0-9]+|[ა-ჰ]+|[A-Za-z]+", flags=re.UNICODE)


def _extract_keywords(q: str) -> list[str]:
    q = (q or "").lower()
    tokens = _WORD_RE.findall(q)
    out: list[str] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t in _STOPWORDS:
            continue
        # keep numbers and meaningful words
        if t.isdigit():
            out.append(t)
            continue
        if len(t) < 3:
            continue
        out.append(t)
    return out[:25]


def _make_stems(tokens: list[str]) -> list[str]:
    """
    Georgian is inflected; a cheap trick is using prefix stems for Georgian tokens.
    """
    stems: list[str] = []
    for t in tokens:
        if t.isdigit():
            stems.append(t)
        elif re.fullmatch(r"[ა-ჰ]+", t):
            stems.append(t[:5])  # prefix stem
        else:
            stems.append(t.lower())
    # unique while preserving order
    seen = set()
    uniq = []
    for s in stems:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _lexical_score(text: str, stems: list[str]) -> int:
    """
    Score by how many stems appear in the chunk text.
    """
    t = (text or "").lower()
    score = 0
    for s in stems:
        if not s:
            continue
        if s.isdigit():
            # number match
            if s in t:
                score += 2
        else:
            if s in t:
                score += 1
    return score


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


def _select_diverse(chunks: list[RetrievedChunk], k: int, per_doc: int = 2) -> list[RetrievedChunk]:
    """
    Avoid selecting 6 chunks from the same doc unless necessary.
    """
    selected: list[RetrievedChunk] = []
    counts: dict[str, int] = defaultdict(int)

    for c in chunks:
        url = c.url or ""
        if url and counts[url] >= per_doc:
            continue
        selected.append(c)
        if url:
            counts[url] += 1
        if len(selected) >= k:
            break

    return selected


def _relevance_gate(best_score: int, stems: list[str]) -> bool:
    """
    Decide whether retrieval is on-topic enough.
    - Short questions: require >=1 match
    - Longer questions: require >=2 matches
    """
    need = 1 if len(stems) <= 3 else 2
    return best_score >= need


def _exact_docno_retrieve(question: str, k: int) -> list[RetrievedChunk]:
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

    keywords = _extract_keywords(question)
    stems = _make_stems(keywords)

    chunks: list[RetrievedChunk] = []
    for doc, meta in zip(docs, metas):
        if not doc or not meta:
            continue
        score = _lexical_score(doc, stems)
        chunks.append(
            RetrievedChunk(
                text=doc,
                title=meta.get("title") or "Untitled",
                url=meta.get("url") or "",
                distance=None,
                chunk_index=meta.get("chunk_index"),
                unique_key=meta.get("uniqueKey"),
                lexical_score=score,
                mode="docno_exact",
            )
        )

    # rank within the doc-number matched document(s)
    chunks.sort(key=lambda c: (-c.lexical_score, c.chunk_index is None, c.chunk_index or 0))
    return _select_diverse(chunks, k=k, per_doc=3)


def retrieve(question: str, k: int = 6) -> list[RetrievedChunk]:
    # 1) Exact doc-number retrieval first
    exact = _exact_docno_retrieve(question, k=k)
    if exact:
        return exact

    # 2) Semantic retrieval + lexical rerank
    col = _get_collection()
    model = _get_model()

    keywords = _extract_keywords(question)
    stems = _make_stems(keywords)

    # retrieve more candidates than k
    n_candidates = max(40, k * 8)

    q_emb = model.encode([_make_query(question)], normalize_embeddings=True)[0].tolist()

    res: dict[str, Any] = col.query(
        query_embeddings=[q_emb],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    candidates: list[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if not doc or not meta:
            continue
        score = _lexical_score(doc, stems)
        candidates.append(
            RetrievedChunk(
                text=doc,
                title=meta.get("title") or "Untitled",
                url=meta.get("url") or "",
                distance=dist,
                chunk_index=meta.get("chunk_index"),
                unique_key=meta.get("uniqueKey"),
                lexical_score=score,
                mode="semantic",
            )
        )

    # Rerank: lexical first, then semantic distance
    candidates.sort(key=lambda c: (-c.lexical_score, c.distance is None, c.distance or 0.0))

    best_score = candidates[0].lexical_score if candidates else 0
    if not _relevance_gate(best_score, stems):
        # No on-topic evidence in retrieved text
        return []

    return _select_diverse(candidates, k=k, per_doc=2)
