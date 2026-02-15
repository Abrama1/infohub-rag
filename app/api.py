from fastapi import FastAPI

app = FastAPI(title="InfoHub RAG")


@app.get("/health")
def health():
    return {"status": "ok"}
