from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM provider: openai_compat | ollama | none
    llm_provider: str = "none"

    # OpenAI-compatible API settings (Groq/Mistral/etc)
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_fallback_model: str | None = None  # e.g. llama-3.1-8b-instant

    # Ollama settings (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    # Retrieval settings (later)
    embedding_model: str = "intfloat/multilingual-e5-large"
    chroma_dir: str = "./data/index"
    chroma_collection: str = "infohub_docs"

    # Optional cookie for authenticated InfoHub requests
    infohub_cookie: str | None = None


settings = Settings()
