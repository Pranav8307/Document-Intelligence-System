import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

# FAISS index store: document_id -> (faiss index, list of chunks)
_indexes: dict[str, tuple] = {}


class EmbeddingService:
    def __init__(self):
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Normalize for cosine similarity (FAISS IndexFlatIP = dot product on unit vecs)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / norms).astype("float32")

    def index_chunks(self, document_id: str, chunks: List[str]) -> None:
        vectors = self._embed(chunks)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
        index.add(vectors)
        _indexes[document_id] = (index, chunks)
        logger.info(f"Indexed {len(chunks)} chunks for doc {document_id}")

    def retrieve_top_k(
        self, document_id: str, question: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        if document_id not in _indexes:
            raise KeyError(f"No index found for document_id={document_id}")

        index, chunks = _indexes[document_id]
        query_vec = self._embed([question])
        scores, indices = index.search(query_vec, min(top_k, len(chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((chunks[idx], float(score)))
        return results