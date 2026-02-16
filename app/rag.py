from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.prompts import SYSTEM_PROMPT, MANDATORY_CITATION_LINE
from app.llm import chat_with_meta
from app.settings import settings
from app.retrieval import retrieve as retrieve_chunks


@dataclass
class Source:
    title: str
    url: str
    page: int | None = None


# Strip any model-written sources section so we can always render canonical links.
_SOURCES_SPLIT_RE = re.compile(r"\n\s*(?:წყაროები|sources)\s*:\s*", flags=re.IGNORECASE)


def _strip_model_sources_block(text: str) -> str:
    """
    If the model wrote a 'წყაროები:' or 'sources:' section, remove it.
    We'll append our own deterministic sources block.
    """
    text = (text or "").strip()
    parts = _SOURCES_SPLIT_RE.split(text, maxsplit=1)
    return (parts[0] if parts else text).strip()


def _dedup_sources(sources: list[Source]) -> list[Source]:
    seen: set[tuple[str, str]] = set()
    out: list[Source] = []
    for s in sources:
        url = (s.url or "").strip()
        title = (s.title or "").strip()
        if not url:
            continue
        key = (url, title)
        if key in seen:
            continue
        seen.add(key)
        out.append(Source(title=title or "Untitled", url=url, page=s.page))
    return out


def _sources_block(sources: list[Source]) -> str:
    sources = _dedup_sources(sources)

    if not sources:
        return "წყაროები: (რელევანტური წყაროები ვერ მოიძებნა)"

    lines: list[str] = []
    for s in sources:
        page_part = f" — გვერდი {s.page}" if s.page is not None else ""
        lines.append(f"- {s.title} — {s.url}{page_part}")

    return "წყაროები:\n" + "\n".join(lines)


def _build_context(snippets: list[str], max_chars: int = 12000) -> str:
    """
    Keep prompt sizes under control for rate-limits/TPM.
    We retrieve k chunks, but only feed up to max_chars to the LLM.
    """
    if not snippets:
        return "(no context — retrieval returned empty)"

    buf: list[str] = []
    total = 0
    for s in snippets:
        s = (s or "").strip()
        if not s:
            continue
        # +2 for the \n\n join
        if total + len(s) + 2 > max_chars:
            break
        buf.append(s)
        total += len(s) + 2

    return "\n\n".join(buf) if buf else "(no context — all retrieved snippets were empty)"


def answer(question: str, k: int = 12) -> dict[str, Any]:
    # Retrieve from index (hybrid retrieval lives in app.retrieval)
    retrieved = retrieve_chunks(question, k=k)
    if not retrieved:
        content = (
            f"{MANDATORY_CITATION_LINE}\n\n"
            "InfoHub-ის ინდექსში ამ კითხვასთან დაკავშირებული სანდო ამონარიდები ვერ მოიძებნა, ამიტომ "
            "დოკუმენტებზე დაყრდნობით ზუსტი პასუხის გაცემას ვერ ვახერხებ."
        )
        content = _strip_model_sources_block(content)
        content = f"{content}\n\n{_sources_block([])}".strip()
        return {
            "answer": content,
            "sources": [],
            "meta": {
                "provider": settings.llm_provider,
                "model_used": None,
                "fallback_used": False,
                "k": k,
            },
        }

    snippets = [c.text for c in retrieved]
    sources = [Source(title=c.title, url=c.url, page=getattr(c, "page", None)) for c in retrieved]
    sources = _dedup_sources(sources)

    context_text = _build_context(snippets, max_chars=12000)

    user_prompt = f"""
Question:
{question}

Context snippets from InfoHub:
{context_text}

Return the final answer in Georgian following the rules.
Do NOT include a 'წყაროები:' section; it will be added separately.
""".strip()

    # Default meta (in case LLM fails and we fall back)
    llm_meta: dict[str, Any] = {
        "provider": settings.llm_provider,
        "model_used": None,
        "fallback_used": False,
    }

    try:
        content, llm_meta = chat_with_meta(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception:
        # Deterministic fallback (works with LLM_PROVIDER=none or temporary API failures)
        if not snippets:
            content = (
                f"{MANDATORY_CITATION_LINE}\n\n"
                "ამ ეტაპზე ინდექსში შესაბამისი ამონარიდები ვერ მოიძებნა, ამიტომ "
                "InfoHub-ის დოკუმენტებზე დაყრდნობით ზუსტი პასუხის გაცემას ვერ ვახერხებ."
            )
        else:
            content = (
                f"{MANDATORY_CITATION_LINE}\n\n"
                "ქვემოთ მოყვანილია ნაპოვნი ამონარიდები. ჩართეთ LLM_PROVIDER, რომ პასუხი უფრო ბუნებრივი იყოს.\n\n"
                + "\n\n".join(snippets[:3])
            )

    # ---- Compliance enforcement (always) ----
    content = (content or "").strip()

    # Ensure mandatory citation line
    if not content.startswith(MANDATORY_CITATION_LINE):
        content = f"{MANDATORY_CITATION_LINE}\n\n{content}".strip()

    # Remove any model-written sources block and ALWAYS append canonical sources.
    content = _strip_model_sources_block(content)
    content = f"{content}\n\n{_sources_block(sources)}".strip()

    return {
        "answer": content,
        "sources": [s.__dict__ for s in sources],
        "meta": {
            **llm_meta,
            "k": k,
        },
    }
