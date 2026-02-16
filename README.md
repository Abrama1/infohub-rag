# InfoHub RAG (Georgian)

A Retrieval-Augmented Generation (RAG) demo that answers questions **in Georgian** and always includes the mandatory citation line:

> **InfoHub-ზე განთავსებული დოკუმენტების მიხედვით (according to the documents posted on the Information and Methodological Hub — Documents and information related to tax and customs administration in one space): https://infohub.rs.ge/ka**

Documents are sourced from the **Information and Methodological Hub** (RS InfoHub) and indexed for retrieval.  
The demo UI is built with **Streamlit** and the backend exposes a **FastAPI** endpoint.

---

## ✅ Requirements Covered

- **Answers in Georgian** ✅  
- **Always includes the mandatory InfoHub citation line** ✅  
- **Cites retrieved sources (document titles + direct URLs)** ✅  
- **RAG pipeline (retrieve → generate grounded answer)** ✅  
- **Demo link + GitHub repository** ✅


---

## How it works (high level)

1. **Ingestion / indexing** (one-time offline step)
   - Fetch metadata and content from the InfoHub API (`infohubapi.rs.ge`)
   - Clean HTML
   - Chunk text
   - Embed using `intfloat/multilingual-e5-large`
   - Store embeddings in **ChromaDB (persistent)**

2. **Runtime**
   - User asks a question
   - Retrieve Top-K relevant chunks from Chroma
   - Use LLM (Groq OpenAI-compatible API) to produce a grounded answer **in Georgian**
   - Enforce:
     - Mandatory citation line at the top
     - A `წყაროები:` block at the bottom

3. **Fast deploy**
   - The prebuilt Chroma index is packaged as a **GitHub Release zip**
   - On Streamlit Cloud (or any environment), the app downloads and extracts it automatically using `INDEX_URL`
   - This avoids re-indexing (which can take a long time)

---

## Project structure

```text
app/
  api.py                # FastAPI app
  bootstrap_index.py    # Downloads/extracts prebuilt Chroma index from INDEX_URL
  llm.py                # LLM call + retry/backoff + fallback
  prompts.py            # System prompt + mandatory citation line
  rag.py                # RAG pipeline (retrieve -> generate -> enforce compliance)
  retrieval.py          # Chroma retrieval
  settings.py           # Pydantic settings (env/.env/Streamlit secrets)

ingest/
  index_infohub.py      # Ingestion script (fetch from InfoHub API, chunk, embed, upsert into Chroma)
  infohub_client.py     # API client for InfoHub endpoints
  html_clean.py         # HTML -> text cleaning

ui/
  streamlit_app.py      # Streamlit UI demo
