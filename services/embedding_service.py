import time
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import get_logger
from utils.metrics import metrics_store

logger = get_logger(__name__)

# Lightweight index store: document_id -> (vectorizer, matrix, chunks)
_indexes: dict[str, tuple] = {}


class EmbeddingService:
    def __init__(self):
        logger.info("Using lightweight TF-IDF retrieval.")

    def index_chunks(self, document_id: str, chunks: List[str]) -> None:
        start = time.perf_counter()
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(chunks)
        _indexes[document_id] = (vectorizer, matrix, chunks)
        metrics_store.record_timing(
            "embedding_encode", (time.perf_counter() - start) * 1000
        )
        logger.info(f"Indexed {len(chunks)} chunks for doc {document_id}")

    def retrieve_top_k(
        self, document_id: str, question: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        start = time.perf_counter()
        if document_id not in _indexes:
            raise KeyError(f"No index found for document_id={document_id}")

        vectorizer, matrix, chunks = _indexes[document_id]
        encode_start = time.perf_counter()
        query_vec = vectorizer.transform([question])
        metrics_store.record_timing(
            "embedding_encode", (time.perf_counter() - encode_start) * 1000
        )
        result_count = min(top_k, len(chunks))
        similarities = cosine_similarity(query_vec, matrix)[0]
        indices = similarities.argsort()[::-1][:result_count]

        # Include adjacent chunks so lists and bullet points split at a chunk
        # boundary arrive together in the LLM context.
        selected_indices = set()
        for idx in indices:
            selected_indices.add(int(idx))
            if idx > 0:
                selected_indices.add(int(idx) - 1)
            if idx + 1 < len(chunks):
                selected_indices.add(int(idx) + 1)

        score_by_index = {
            int(idx): float(similarities[idx]) for idx in indices
        }
        results = [
            (chunks[idx], score_by_index.get(idx, 0.0))
            for idx in sorted(selected_indices)
        ]
        metrics_store.record_timing(
            "retrieval_search", (time.perf_counter() - start) * 1000
        )
        return results
