from __future__ import annotations

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


def _sources_block(sources: list[Source]) -> str:
    if not sources:
        return "წყაროები: (რელევანტური წყაროები ვერ მოიძებნა)"

    lines: list[str] = []
    for s in sources:
        page_part = f" — გვერდი {s.page}" if s.page is not None else ""
        lines.append(f"- {s.title} — {s.url}{page_part}")

    return "წყაროები:\n" + "\n".join(lines)


def answer(question: str, k: int = 6) -> dict[str, Any]:
    # Retrieve real chunks from the index
    retrieved = retrieve_chunks(question, k=k)
    snippets = [c.text for c in retrieved]
    sources = [Source(title=c.title, url=c.url, page=getattr(c, "page", None)) for c in retrieved]

    context_text = "(no context yet — retrieval returned empty)"
    if snippets:
        context_text = "\n\n".join(snippets)

    user_prompt = f"""
Question:
{question}

Context snippets from InfoHub:
{context_text}

Return the final answer in Georgian following the rules.
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

    if not content.startswith(MANDATORY_CITATION_LINE):
        content = f"{MANDATORY_CITATION_LINE}\n\n{content}".strip()

    if "წყაროები" not in content:
        content = f"{content}\n\n{_sources_block(sources)}".strip()

    return {
        "answer": content,
        "sources": [s.__dict__ for s in sources],
        "meta": {
            **llm_meta,
            "k": k,
        },
    }
