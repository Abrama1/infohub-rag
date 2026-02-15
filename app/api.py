from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import answer

app = FastAPI(title="InfoHub RAG")


class AskRequest(BaseModel):
    question: str
    k: int = 6


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    return answer(req.question, k=req.k)
