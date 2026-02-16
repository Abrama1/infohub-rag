from __future__ import annotations

import re


DOCNO_RE = re.compile(r"(?:№|N)\s*([0-9]{1,7})", flags=re.IGNORECASE)


def extract_doc_number_digits(raw: str | None) -> str | None:
    """
    Extracts the numeric part from strings like:
      - "304"
      - "56/ნ"
      - "დადგენილება №304"
      - "N 56/ნ - 10/02/2026 - ..."
    Returns digits as string ("304", "56"), or None if not found.
    """
    if not raw:
        return None

    s = str(raw).strip()
    if not s:
        return None

    # If it already looks like a pure number
    if s.isdigit():
        return s

    # Try patterns containing № or N
    m = DOCNO_RE.search(s)
    if m:
        return m.group(1)

    # Fallback: first digit sequence
    m2 = re.search(r"([0-9]{1,7})", s)
    if m2:
        return m2.group(1)

    return None
