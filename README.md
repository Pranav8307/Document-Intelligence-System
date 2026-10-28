# Document Intelligence System

A document Q&A API with a small React frontend. Upload a PDF or text file, ask questions in plain English, and get answers that are generated only from the retrieved parts of that document.

I built it to understand how a RAG (retrieval-augmented generation) pipeline works end to end: chunking, retrieval, grounded prompting, caching, rate limiting, metrics, and deployment. It is a learning project, not a production product.

**Live demo:** https://document-intelligence-system-29xr.onrender.com/ (Render free tier: the first request after idle can take about a minute to wake up.)

---

## How it works

1. **Upload** a PDF or TXT file (max 10 MB). Text is extracted with PyMuPDF.
2. **Chunk** the text into 500-character pieces with 50 characters of overlap, so a sentence cut at a boundary still appears whole in the neighbouring chunk.
3. **Index** the chunks with TF-IDF (scikit-learn, unigrams + bigrams, English stop words).
4. **Retrieve** the best-matching chunks for a question by cosine similarity, plus each match's two neighbouring chunks.
5. **Generate** an answer with an LLM (Groq in the deployed version), with a prompt that tells the model to answer only from the retrieved context and to say so when the answer is not there.
6. **Cache** the answer for 5 minutes (in memory), so a repeated question skips retrieval and the LLM call.

### Answer modes

| Mode | Retrieval limit | Output |
|---|---|---|
| `detailed` (default) | up to 20 matching chunks, plus neighbors | thorough answer with a bounded amount of context |
| `concise` | 8 chunks | 3 to 6 bullets or short paragraphs |

`top_k` (1 to 50) overrides the limit. The default keeps detailed requests small enough for reliable generation while still returning surrounding chunks for context.

---

## Why TF-IDF and not embeddings + FAISS?

The first version (March to September 2026) used SentenceTransformers (`all-MiniLM-L6-v2`) with a FAISS `IndexFlatIP` index on normalized vectors. When I deployed to Render's free tier, the image failed on a FAISS/NumPy compatibility error and then ran out of memory because of PyTorch. On 10 September 2026 I switched retrieval to TF-IDF so the service fits the free tier. The full history is in the git log.

Trade-off:

- TF-IDF matches **words**, not **meaning**. It misses paraphrases such as "forgot my login" vs "reset password".
- It is tiny, fast, needs no model download, and is easy to debug.
- For larger deployments I would go back to embeddings, plus keyword search (BM25) and a reranker.

`benchmarks/run_benchmarks.py` is a legacy script from the FAISS version (see "Benchmarks" below).

---

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/metrics` | Counters and timings (retrieval, indexing, LLM, total request) |
| POST | `/documents/upload` | Upload a PDF/TXT file. Returns `document_id` and `chunk_count` |
| POST | `/qa/ask` | Body: `document_id`, `question`, optional `answer_length`, `top_k` |

Interactive docs: `/docs`.

Limits: PDF or TXT only, 10 MB max, `/qa/ask` is rate limited to 10 requests per minute per IP (slowapi). Requests are validated with Pydantic.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| Retrieval | scikit-learn TF-IDF + cosine similarity |
| PDF parsing | PyMuPDF |
| LLM | Groq through the OpenAI-compatible SDK (model set by env var); mock mode for local runs |
| Storage | In-memory indexes. Optional PostgreSQL (SQLAlchemy) keeps documents and chunks, and indexes are rebuilt from it on startup |
| Frontend | React + Vite |
| Deploy | Docker, Render |

---

## Run locally

```bash
git clone https://github.com/Pranav8307/Document-Intelligence-System.git
cd Document-Intelligence-System
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit it
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs.

**No API key?** Keep `USE_MOCK_LLM=true` in `.env`. The mock returns an excerpt of a retrieved chunk labelled `[MOCK ANSWER]`. It lets you exercise upload, chunking, retrieval, caching and the API without calling a model. It does **not** measure retrieval accuracy, because it returns an excerpt of the first retrieved chunk in document order, not the best-scoring one.

**Frontend:**

```bash
cd frontend
npm install
npm run dev     # http://localhost:5173
```

### Environment variables

| Variable | Purpose |
|---|---|
| `USE_MOCK_LLM` | `true` to skip the LLM |
| `LLM_PROVIDER` | `groq` or `openai` |
| `LLM_MODEL` | model name from the provider |
| `LLM_BASE_URL` | provider base URL (Groq: `https://api.groq.com/openai/v1`) |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | key for the chosen provider |
| `DATABASE_URL` | optional PostgreSQL URL for persistence |
| `FRONTEND_ORIGINS` | comma-separated allowed CORS origins |

Deployment steps for Render are in `DEPLOY_RENDER.md`.

---

## Tests

```bash
pip install pytest
pytest -q
```

The tests cover chunking, retrieval ranking, cache expiry, and the upload/ask flow in mock mode.

---

## Benchmarks (legacy)

On the earlier FAISS version, in a local run in early April 2026 against 12 small synthetic documents, I measured about **94 ms retrieval** (which included encoding the question) and about **1.69 s end to end** (dominated by the LLM call). These numbers describe the old architecture, a tiny corpus, and a single machine. I have not re-measured them on the current TF-IDF version.

The best current numbers are your own: call `/metrics` after some real questions, and look at `retrieval_search` and `qa_request_total`.

To run the legacy script you need the extra packages in `requirements-bench.txt`.

---

## Known limitations

- TF-IDF cannot match synonyms or paraphrases.
- Fixed-size character chunking can split sentences and tables; scanned PDFs have no text layer and are not supported (no OCR).
- The answer cache and rate limiter are in memory, so they reset on restart and do not work across multiple instances.
- No authentication or per-user document isolation: anyone with a `document_id` can query it.
- There is no retrieval evaluation set yet; quality has only been checked by hand.

## What I would do next

1. Build a small question/answer evaluation set and measure recall@k.
2. Add embeddings + BM25 hybrid search and a reranker, compared against this baseline.
3. Move the cache and rate limiter to Redis; add per-user access control.

---

Pranav, [LinkedIn](https://linkedin.com/in/pranav2602), pranav260205@gmail.com
