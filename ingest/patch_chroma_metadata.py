from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from tqdm import tqdm

from ingest.doc_numbers import extract_doc_number_digits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", default="LegislativeNews")
    parser.add_argument("--raw-dir", default="./data/raw")
    parser.add_argument("--chroma-dir", default="./data/index")
    parser.add_argument("--collection", default="infohub_docs")
    args = parser.parse_args()

    raw_species_dir = Path(args.raw_dir) / args.species
    if not raw_species_dir.exists():
        raise RuntimeError(f"Raw dir not found: {raw_species_dir}")

    client = chromadb.PersistentClient(path=args.chroma_dir)
    col = client.get_or_create_collection(name=args.collection)

    files = sorted(raw_species_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No raw json files found in: {raw_species_dir}")

    updated_docs = 0
    skipped_docs = 0

    for fp in tqdm(files, desc=f"Patching metadata for {args.species}", unit="doc"):
        unique_key = fp.stem

        try:
            details = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            skipped_docs += 1
            continue

        # Prefer explicit documentNumber, otherwise try name/title
        doc_number_raw = details.get("documentNumber") or details.get("name") or ""
        doc_number_digits = extract_doc_number_digits(doc_number_raw)

        if not doc_number_digits:
            skipped_docs += 1
            continue

        # Fetch all chunks for this uniqueKey
        got = col.get(
            where={"uniqueKey": unique_key},
            include=["metadatas"],
        )

        ids = got.get("ids") or []
        metas = got.get("metadatas") or []

        if not ids or not metas:
            skipped_docs += 1
            continue

        # Update metadata for each chunk
        new_metas = []
        changed_any = False

        for m in metas:
            m = dict(m or {})
            if m.get("doc_number_digits") != doc_number_digits or m.get("doc_number_raw") != str(doc_number_raw):
                m["doc_number_digits"] = doc_number_digits
                m["doc_number_raw"] = str(doc_number_raw)
                changed_any = True
            new_metas.append(m)

        if not changed_any:
            continue

        # Use update if available; fallback to upsert
        try:
            col.update(ids=ids, metadatas=new_metas)
        except Exception:
            col.upsert(ids=ids, metadatas=new_metas)

        updated_docs += 1

    print(f"\nDone. Updated docs: {updated_docs}, skipped docs: {skipped_docs}")


if __name__ == "__main__":
    main()
