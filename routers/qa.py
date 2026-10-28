import time
from fastapi import APIRouter, HTTPException, Request
from models.schemas import APIResponse, AskRequest, AskResponse
from services.document_service import document_exists, embedding_service
from services.llm_service import get_answer
from utils.cache import cache_get, cache_set, make_cache_key
from utils.logger import get_logger
from utils.metrics import metrics_store
from utils.rate_limiter import limiter

logger = get_logger(__name__)
router = APIRouter()

DEFAULT_RETRIEVAL_LIMITS = {"concise": 8, "detailed": 20}


def _resolve_retrieval_limit(answer_length: str, top_k: int | None) -> int:
    if top_k is not None:
        return top_k
    return DEFAULT_RETRIEVAL_LIMITS[answer_length]


def _answer_cache_key(
    document_id: str, question: str, answer_length: str, top_k: int
) -> str:
    return make_cache_key(
        document_id,
        f"v5-{answer_length}-{top_k}:{question}",
    )


def _cached_answer_details(cached) -> tuple[str, int]:
    if isinstance(cached, dict):
        return cached["answer"], cached.get("chunks_used", 0)
    return cached, 0


def _generate_answer(question: str, chunks, answer_length: str) -> str:
    try:
        return get_answer(question, chunks, answer_length)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "The answer service could not complete this request. "
                "Try concise mode or a smaller document."
            ),
        ) from exc


@router.post("/ask", response_model=APIResponse)
@limiter.limit("10/minute")
async def ask_question(request: Request, body: AskRequest):
    request_start = time.perf_counter()
    metrics_store.increment("qa_requests")

    if not document_exists(body.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{body.document_id}' not found. Upload it first.",
        )

    retrieval_limit = _resolve_retrieval_limit(body.answer_length, body.top_k)
    cache_key = _answer_cache_key(
        body.document_id, body.question, body.answer_length, retrieval_limit
    )
    cached = cache_get(cache_key)

    if cached:
        logger.info(f"Cache hit | key={cache_key}")
        metrics_store.increment("cache_hits")
        metrics_store.record_timing(
            "qa_request_total", (time.perf_counter() - request_start) * 1000
        )
        cached_answer, cached_chunks_used = _cached_answer_details(cached)
        return APIResponse(
            status="success",
            message="Answer retrieved from cache.",
            data=AskResponse(
                answer=cached_answer,
                chunks_used=cached_chunks_used,
                cached=True,
                document_id=body.document_id,
            ),
        )

    metrics_store.increment("cache_misses")

    try:
        chunks = embedding_service.retrieve_top_k(
            body.document_id, body.question, top_k=retrieval_limit
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=422, detail="No relevant chunks found.")

    logger.info(
        f"Retrieved {len(chunks)} chunks for question: '{body.question[:60]}...'"
    )

    answer = _generate_answer(body.question, chunks, body.answer_length)
    cache_set(cache_key, {"answer": answer, "chunks_used": len(chunks)})
    metrics_store.increment("answers_generated")
    metrics_store.record_timing(
        "qa_request_total", (time.perf_counter() - request_start) * 1000
    )

    return APIResponse(
        status="success",
        message="Answer generated successfully.",
        data=AskResponse(
            answer=answer,
            chunks_used=len(chunks),
            cached=False,
            document_id=body.document_id,
        ),
    )
