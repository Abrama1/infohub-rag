from fastapi import FastAPI
from pydantic import BaseModel

from app.settings import settings
from app.version import __version__
from app.rag import answer

app = FastAPI(title="InfoHub RAG", version=__version__)


class AskRequest(BaseModel):
    question: str
    k: int = 6


@app.get("/")
def root():
    return {
        "name": "InfoHub RAG",
        "version": __version__,
        "endpoints": ["/health", "/info", "/ask", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    # Safe to expose: no API keys returned
    return {
        "version": __version__,
        "llm": {
            "provider": settings.llm_provider,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "fallback_model": settings.llm_fallback_model,
            "ollama_base_url": settings.ollama_base_url,
            "ollama_model": settings.ollama_model,
        },
    }


@app.post("/ask")
def ask(req: AskRequest):
    return answer(req.question, k=req.k)
