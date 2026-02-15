from __future__ import annotations

import requests
from app.settings import settings


def chat(messages: list[dict]) -> str:
    provider = (settings.llm_provider or "none").lower().strip()

    if provider == "none":
        raise RuntimeError("LLM_PROVIDER is none")

    if provider == "ollama":
        return _ollama_chat(messages)

    if provider == "openai_compat":
        return _openai_compat_chat(messages)

    raise RuntimeError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def _openai_compat_chat(messages: list[dict]) -> str:
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

        # If it’s rate-limited or temporarily failing, let caller decide fallback
        if r.status_code in (429, 500, 502, 503, 504):
            raise RuntimeError(f"Transient LLM error {r.status_code}: {r.text}")

        r.raise_for_status()
        data = r.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response format: {data}") from e

    # Try primary model
    try:
        return call_model(settings.llm_model)
    except Exception as primary_err:
        # Try fallback model if configured
        if settings.llm_fallback_model:
            return call_model(settings.llm_fallback_model)
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
