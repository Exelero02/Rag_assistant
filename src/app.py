from fastapi import FastAPI
from pydantic import BaseModel
import json, faiss, numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Query(BaseModel):
    q: str
    k: int = 3

app = FastAPI(title="Local RAG API")


_index = None
_store = None
_tokenizer = None
_model = None

def get_index():
    global _index
    if _index is None:
        _index = faiss.read_index("data/index.faiss")
    return _index

def get_store():
    global _store
    if _store is None:
        with open("data/store.json","r",encoding="utf-8") as f:
            _store = json.load(f)
    return _store

def get_lm():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        name = "google/flan-t5-base"
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return _tokenizer, _model

@app.post("/query")
def query(q: Query):
    store = get_store()
    texts = store["texts"]
    #count the keyword occurrences
    scores = [sum(t.lower().count(w) for w in q.q.lower().split()) for t in texts]
    idx = np.argsort(scores)[::-1][:q.k]
    context = "\n\n".join(texts[i][:800] for i in idx)
    tok, lm = get_lm()
    prompt = f"Answer the question using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {q.q}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = lm.generate(**inputs, max_new_tokens=128)
    ans = tok.decode(outputs[0], skip_special_tokens=True)
    return {"answer": ans, "context_files": [store["files"][i] for i in idx]}
