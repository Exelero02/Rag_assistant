# RAG Assistant (Local Embeddings + FAISS + FastAPI)

A minimal **Retrieval-Augmented Generation** demo you can run fully **offline**.  
It ingests documents, builds a **FAISS** vector index with `sentence-transformers`, serves a **FastAPI** endpoint, and provides a **Streamlit** UI to ask questions over your own files.

## Run
```bash
pip install -r requirements.txt
python src/ingest.py --docs docs --index data/index.faiss --store data/store.json
uvicorn src.app:app --reload --port 8000  
streamlit run app_streamlit.py
```
## ✨ Features

- **Local embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers`
- **Vector search**: FAISS (`faiss-cpu`)
- **Local generation**: `google/flan-t5-base` (downloads on first run)
- **Two UX options**:
  - REST API (`FastAPI`) → `POST /query`
  - Simple web UI (`Streamlit`)
- **Plug-and-play docs**: drop `.txt` / `.md` / `.csv` / `.py` / `.json` into `docs/` and re-ingest

---


🧠 How It Works (MVP)

Ingestion (src/ingest.py)

Reads text-like files from docs/

Embeds them with sentence-transformers/all-MiniLM-L6-v2

Builds a FAISS index + stores raw texts to data/store.json

Querying (src/app.py)

Loads the index and sources

MVP retrieval: uses a simple keyword scoring fallback to keep code short

LLM answer: flan-t5-base generates an answer constrained by retrieved text
