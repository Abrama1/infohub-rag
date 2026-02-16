from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def _find_chroma_root(extracted_dir: Path) -> Path | None:
    # Case 1: zip root contains chroma.sqlite3
    if (extracted_dir / "chroma.sqlite3").exists():
        return extracted_dir

    # Case 2: zip root contains a single folder that contains chroma.sqlite3
    for child in extracted_dir.iterdir():
        if child.is_dir() and (child / "chroma.sqlite3").exists():
            return child

    # Case 3: deeper search (safe fallback)
    for p in extracted_dir.rglob("chroma.sqlite3"):
        return p.parent

    return None


def ensure_chroma_index() -> bool:
    """
    Ensure CHROMA_DIR exists and contains chroma.sqlite3.
    If missing, download INDEX_URL (zip) and extract into CHROMA_DIR.
    """
    chroma_dir = Path(os.getenv("CHROMA_DIR", "./data/index"))
    index_url = os.getenv("INDEX_URL", "").strip()

    sqlite_file = chroma_dir / "chroma.sqlite3"
    MIN_SIZE = 5_000_000  # 5MB; your real one is ~82MB

    if sqlite_file.exists() and sqlite_file.stat().st_size >= MIN_SIZE:
        return True

    if not index_url:
        return False

    chroma_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "index.zip"

        urllib.request.urlretrieve(index_url, zip_path)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        chroma_root = _find_chroma_root(extract_dir)
        if chroma_root is None:
            return False

        if chroma_dir.exists():
            shutil.rmtree(chroma_dir, ignore_errors=True)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        for item in chroma_root.iterdir():
            dest = chroma_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    return sqlite_file.exists()
