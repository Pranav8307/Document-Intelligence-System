import io
import uuid
from typing import List
import fitz  # PyMuPDF
from utils.logger import get_logger
from services.embedding_service import EmbeddingService

logger = get_logger(__name__)

# In-memory store: document_id -> list of raw text chunks
_document_store: dict[str, List[str]] = {}

embedding_service = EmbeddingService()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace").strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple fixed-size character chunking with overlap.
    
    Token optimization note:
    - chunk_size=500 chars ≈ ~125 tokens (GPT tokenizer: ~4 chars/token)
    - top_k=3 chunks → max ~375 tokens of context per query
    - Full doc could be 100k+ tokens; we only send ~375 — massive saving
    - In production: use tiktoken to chunk by actual token count, not chars
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def process_and_store_document(
    file_bytes: bytes, filename: str
) -> tuple[str, int]:
    document_id = str(uuid.uuid4())
    logger.info(f"Processing document: {filename} | id={document_id}")

    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text:
        raise ValueError("Document appears to be empty or unreadable.")

    chunks = chunk_text(text)
    logger.info(f"Chunked into {len(chunks)} chunks | id={document_id}")

    _document_store[document_id] = chunks
    embedding_service.index_chunks(document_id, chunks)

    return document_id, len(chunks)


def document_exists(document_id: str) -> bool:
    return document_id in _document_store