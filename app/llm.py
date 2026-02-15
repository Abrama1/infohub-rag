from __future__ import annotations

import requests
from app.settings import settings


def chat_with_meta(messages: list[dict]) -> tuple[str, dict]:
    """
    Returns (content, meta) where meta includes model_used and fallback_used.
    """
    provider = (settings.llm_provider or "none").lower().strip()

    if provider == "none":
        raise RuntimeError("LLM_PROVIDER is none")

    if provider == "ollama":
        content = _ollama_chat(messages)
        return content, {
            "provider": "ollama",
            "model_used": settings.ollama_model,
            "fallback_used": False,
        }

    if provider == "openai_compat":
        return _openai_compat_chat(messages)

    raise RuntimeError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def chat(messages: list[dict]) -> str:
    # Backwards-compatible wrapper
    return chat_with_meta(messages)[0]


def _openai_compat_chat(messages: list[dict]) -> tuple[str, dict]:
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set")

    def call_model(model: str) -> str:
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=120)

        # transient-ish statuses
        if r.status_code in (429, 500, 502, 503, 504):
            raise RuntimeError(f"Transient LLM error {r.status_code}: {r.text}")

        r.raise_for_status()
        data = r.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response format: {data}") from e

    # Try primary
    try:
        content = call_model(settings.llm_model)
        return content, {
            "provider": "openai_compat",
            "model_used": settings.llm_model,
            "fallback_used": False,
        }
    except Exception as primary_err:
        # Try fallback if configured
        if settings.llm_fallback_model:
            content = call_model(settings.llm_fallback_model)
            return content, {
                "provider": "openai_compat",
                "model_used": settings.llm_fallback_model,
                "fallback_used": True,
            }
        raise primary_err


def _ollama_chat(messages: list[dict]) -> str:
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    try:
        return data["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected Ollama response format: {data}") from e
