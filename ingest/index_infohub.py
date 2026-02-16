from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ingest.infohub_client import InfoHubClient
from ingest.html_to_text import html_to_text
from ingest.chunking import chunk_text


def canonical_doc_url(unique_key: str) -> str:
    return f"https://infohub.rs.ge/ka/workspace/document/{unique_key}?openFromSearch=true"


def make_passage(text: str, model_name: str) -> str:
    # e5-style models work best with prefixes
    if "e5" in model_name.lower():
        return "passage: " + text
    return text


def make_query(text: str, model_name: str) -> str:
    if "e5" in model_name.lower():
        return "query: " + text
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", default="LegislativeNews")
    parser.add_argument("--take", type=int, default=99)
    parser.add_argument("--max-docs", type=int, default=500, help="limit for MVP; set 0 for no limit")
    parser.add_argument("--delay", type=float, default=0.2)

    parser.add_argument("--api-base", default="https://infohubapi.rs.ge/api")
    parser.add_argument("--lang", default="ka")
    parser.add_argument("--cookie", default="")

    parser.add_argument("--chroma-dir", default="./data/index")
    parser.add_argument("--collection", default="infohub_docs")
    parser.add_argument("--embed-model", default="intfloat/multilingual-e5-large")

    parser.add_argument("--raw-dir", default="./data/raw")
    parser.add_argument("--text-dir", default="./data/text")
    args = parser.parse_args()

    max_docs = None if args.max_docs == 0 else args.max_docs

    raw_dir = Path(args.raw_dir) / args.species
    text_dir = Path(args.text_dir) / args.species
    raw_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    client = InfoHubClient(
        base_url=args.api_base,
        language_code=args.lang,
        cookie=args.cookie.strip() or None,
        delay_sec=args.delay,
    )

    # Chroma
    chroma_path = Path(args.chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    # Embeddings
    model = SentenceTransformer(args.embed_model)

    processed = 0
    skip = 0

    pbar = tqdm(total=max_docs or 0, desc=f"Ingest {args.species}", unit="doc")
    pbar_total_unknown = max_docs is None

    while True:
        page = client.list_documents(species=args.species, skip=skip, take=args.take)
        items: list[dict[str, Any]] = page.get("data") or []

        if not items:
            break

        for item in items:
            if max_docs is not None and processed >= max_docs:
                pbar.close()
                return

            unique_key = item.get("uniqueKey")
            if not unique_key:
                continue

            raw_path = raw_dir / f"{unique_key}.json"
            text_path = text_dir / f"{unique_key}.txt"

            # Fetch details (cache raw JSON)
            if raw_path.exists():
                details = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                details = client.get_details_by_key(unique_key)
                raw_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

            title = details.get("name") or item.get("name") or f"InfoHub {unique_key}"
            url = canonical_doc_url(unique_key)

            description_html = details.get("description") or ""
            text = html_to_text(description_html)

            if text:
                text_path.write_text(text, encoding="utf-8")

                chunks = chunk_text(text, max_chars=1200, overlap=200)
                if chunks:
                    ids = [f"{unique_key}:{i}" for i in range(len(chunks))]
                    docs = chunks
                    metadatas = [
                        {
                            "uniqueKey": unique_key,
                            "title": title,
                            "url": url,
                            "species": args.species,
                            "chunk_index": i,
                            "publishDate": details.get("publishDate") or details.get("receiptDate"),
                        }
                        for i in range(len(chunks))
                    ]

                    embeddings = model.encode([make_passage(c, args.embed_model) for c in chunks], normalize_embeddings=True)
                    embeddings_list = [e.tolist() for e in embeddings]

                    # Upsert: safe for reruns
                    collection.upsert(
                        ids=ids,
                        documents=docs,
                        metadatas=metadatas,
                        embeddings=embeddings_list,
                    )

            processed += 1
            if pbar_total_unknown:
                pbar.update(1)
            else:
                pbar.update(1)

        skip += args.take

    pbar.close()


if __name__ == "__main__":
    main()
