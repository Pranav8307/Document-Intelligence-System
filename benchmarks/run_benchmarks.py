import os
import time
import requests
import uuid
import numpy as np
from services.document_service import chunk_text
from sentence_transformers import SentenceTransformer
import faiss

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def upload_text(text: str):
    files = {"file": ("test.txt", text.encode("utf-8"), "text/plain")}
    t0 = time.perf_counter()
    r = requests.post(f"{API_URL}/documents/upload", files=files)
    t = (time.perf_counter() - t0) * 1000
    return r, t


def ask_question(document_id: str, question: str, top_k: int = 3):
    payload = {"document_id": document_id, "question": question, "top_k": top_k}
    t0 = time.perf_counter()
    r = requests.post(f"{API_URL}/qa/ask", json=payload)
    t = (time.perf_counter() - t0) * 1000
    return r, t


def bench_cache_hit_miss():
    print("=== Cache hit/miss benchmark ===")
    text = ("The quick brown fox jumps over the lazy dog. \n" * 200)
    r, upload_ms = upload_text(text)
    print(f"upload: status={r.status_code} time_ms={upload_ms:.2f}")
    doc_id = r.json()["data"]["document_id"]

    q = "What jumps over the lazy dog?"
    r1, t1 = ask_question(doc_id, q)
    print(f"first ask (miss): status={r1.status_code} time_ms={t1:.2f}")

    r2, t2 = ask_question(doc_id, q)
    print(f"second ask (hit): status={r2.status_code} time_ms={t2:.2f}")


def bench_faiss_vs_brute(num_chunks: int):
    print(f"=== FAISS vs brute-force retrieval | chunks={num_chunks} ===")
    text = "\n".join([f"chunk {i} content" for i in range(num_chunks)])
    chunks = chunk_text(text, chunk_size=1000, overlap=0)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(chunks, convert_to_numpy=True).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = (vecs / norms)

    # FAISS build and search
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    t0 = time.perf_counter()
    index.add(vecs)
    t_add = (time.perf_counter() - t0) * 1000

    query = "chunk 7 content"
    qvec = model.encode([query], convert_to_numpy=True).astype("float32")
    qvec = qvec / np.linalg.norm(qvec, axis=1, keepdims=True)

    t0 = time.perf_counter()
    scores, inds = index.search(qvec, 3)
    faiss_ms = (time.perf_counter() - t0) * 1000

    # brute-force cosine (dot product on normalized vectors)
    t0 = time.perf_counter()
    sims = np.dot(vecs, qvec[0])
    top_idx = np.argsort(-sims)[:3]
    brute_ms = (time.perf_counter() - t0) * 1000

    print(f"faiss_add_ms={t_add:.2f} faiss_search_ms={faiss_ms:.4f} brute_search_ms={brute_ms:.4f}")


def bench_max_document_size():
    print("=== Max document size benchmark (tests up to 10MB limit) ===")
    sizes = [100_000, 500_000, 1_000_000, 5_000_000, 9_000_000]  # bytes
    for s in sizes:
        text = "A" * s
        r, t = upload_text(text)
        print(f"size={s} bytes -> status={r.status_code} time_ms={t:.2f}")


def bench_rate_limit():
    print("=== Rate limiting test (10/min configured) ===")
    text = ("rate test\n" * 100)
    r, u = upload_text(text)
    doc_id = r.json()["data"]["document_id"]
    q = "rate test"
    results = []
    for i in range(12):
        r, t = ask_question(doc_id, f"{q} {i}")
        results.append((i, r.status_code, t))
        time.sleep(0.1)
    for i, status, t in results:
        print(f"req={i} status={status} time_ms={t:.2f}")


def main():
    bench_cache_hit_miss()
    for n in (10, 50, 100):
        bench_faiss_vs_brute(n)
    bench_max_document_size()
    bench_rate_limit()


if __name__ == "__main__":
    main()
