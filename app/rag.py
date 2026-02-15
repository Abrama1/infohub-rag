from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.prompts import SYSTEM_PROMPT, MANDATORY_CITATION_LINE
from app.llm import chat as llm_chat


@dataclass
class Source:
    title: str
    url: str
    page: int | None = None


def retrieve_stub(question: str, k: int = 6) -> tuple[list[str], list[Source]]:
    """
    Placeholder retrieval.
    We will replace this later with real vector DB retrieval from InfoHub documents.
    """
    return [], []


def answer(question: str, k: int = 6) -> dict[str, Any]:
    snippets, sources = retrieve_stub(question, k=k)

    context_text = "(no context yet — retrieval not implemented)"
    if snippets:
        context_text = "\n\n".join(snippets)

    user_prompt = f"""
    Question:
    {question}
    
    Context snippets from InfoHub:
    {context_text}
    
    Return the final answer in Georgian following the rules.
    """.strip()

    try:
        content = llm_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception:
        # Deterministic fallback (works with LLM_PROVIDER=none)
        if not snippets:
            content = (
                f"{MANDATORY_CITATION_LINE}\n\n"
                "ამ ეტაპზე დოკუმენტების მოძიება/ინდექსაცია ჯერ არ არის ჩართული, ამიტომ "
                "InfoHub-ის დოკუმენტებზე დაყრდნობით ზუსტი პასუხის გაცემას ვერ ვახერხებ.\n\n"
                "წყაროები: (ჯერ არ არის ხელმისაწვდომი)"
            )
        else:
            src_lines = []
            for s in sources:
                page_part = f" — გვერდი {s.page}" if s.page is not None else ""
                src_lines.append(f"- {s.title} — {s.url}{page_part}")

            content = (
                f"{MANDATORY_CITATION_LINE}\n\n"
                "ქვემოთ მოყვანილია ნაპოვნი ამონარიდები. ჩართეთ LLM_PROVIDER, რომ პასუხი უფრო ბუნებრივი იყოს.\n\n"
                + "\n\n".join(snippets[:3])
                + "\n\nწყაროები:\n"
                + "\n".join(src_lines)
            )

    return {
        "answer": content,
        "sources": [s.__dict__ for s in sources],
    }
