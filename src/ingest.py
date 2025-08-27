import argparse, os, json, glob
from sentence_transformers import SentenceTransformer
import faiss

def read_texts(path: str):
    texts = []
    for fp in glob.glob(os.path.join(path, "**", "*"), recursive=True):
        if os.path.isfile(fp) and any(fp.endswith(ext) for ext in [".txt",".md",".csv",".py",".json"]):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    texts.append((fp, f.read()))
            except Exception:
                pass
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=str, default="docs")
    ap.add_argument("--index", type=str, default="data/index.faiss")
    ap.add_argument("--store", type=str, default="data/store.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.index), exist_ok=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = read_texts(args.docs)
    corpus = [t[1] for t in texts]
    embs = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    faiss.normalize_L2(embs)
    index.add(embs)
    faiss.write_index(index, args.index)
    with open(args.store, "w", encoding="utf-8") as f:
        json.dump({"files":[t[0] for t in texts], "texts": corpus}, f)
    print("Built index of", len(corpus), "chunks")

if __name__ == "__main__":
    main()
