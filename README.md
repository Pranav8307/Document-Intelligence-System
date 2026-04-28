# Document Intelligence System

A production-style document Q&A API. Upload a PDF or text file, ask questions in natural language, and get answers grounded in the document content — not hallucinated from model weights.

Built to explore what a real RAG pipeline looks like beyond the toy tutorial version: proper chunking, vector indexing, caching, rate limiting, and a fallback mock LLM mode that runs without any API keys.

---

## How it works

1. Upload a document (PDF or plain text)
2. The system chunks the text, generates sentence embeddings, and indexes them in FAISS
3. On query, the top-k most relevant chunks are retrieved by cosine similarity
4. Those chunks are passed as context to the LLM, which answers based only on what's in the document
5. Results are cached with TTL to avoid redundant embedding calls

---

## Features

- **RAG pipeline** — SentenceTransformers for embeddings, FAISS for vector retrieval
- **FastAPI backend** — clean REST endpoints, Pydantic models, proper error handling
- **TTL caching** — repeated queries on the same document don't re-embed
- **Rate limiting** — basic per-IP throttling on the query endpoint
- **Mock LLM mode** — runs without OpenAI keys, returns retrieved chunks directly (useful for testing retrieval quality independently)
- **Structured project layout** — models, routers, services, utils separated cleanly

---

## Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| LLM Integration | OpenAI API (optional) |
| Language | Python 3.10+ |

---

## Project structure

Document-Intelligence-System/
├── routers/        # FastAPI route handlers
├── services/       # Core logic: chunking, embedding, retrieval
├── models/         # Pydantic schemas
├── utils/          # Helpers: caching, rate limiting
├── main.py         # App entry point
└── requirements.txt

---

## Setup

```bash
git clone https://github.com/Pranav8307/Document-Intelligence-System.git
cd Document-Intelligence-System

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:
OPENAI_API_KEY=your_key_here   # Optional — omit to use mock LLM mode

Run the server:
```bash
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload and index a document |
| POST | `/query` | Ask a question against the indexed document |
| GET | `/health` | Health check |

---

## Design decisions worth noting

**Why mock LLM mode?** Separates retrieval quality from generation quality. You can test whether your chunking and vector search are finding the right passages before ever touching an LLM API.

**Why TTL cache on embeddings?** Embedding a multi-page document on every query would be slow and expensive. Cache the index, invalidate when the document changes.

**Why FAISS over a managed vector DB?** For a project at this scale, FAISS runs in-process, has no infrastructure cost, and is what ships in production at many companies before they hit scale limits.

---

## Author

Pranav · [LinkedIn](https://linkedin.com/in/pranav2602) · pranav260205@gmail.com
