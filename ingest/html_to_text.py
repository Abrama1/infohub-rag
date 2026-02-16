from __future__ import annotations

import re
from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    """
    Converts InfoHub 'description' HTML to clean text for embedding.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # remove script/style
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)

    # normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
