import time
from fastapi import APIRouter, HTTPException, Request
from openai import RateLimitError
from models.schemas import APIResponse, AskRequest, AskResponse
from services.document_service import document_exists, embedding_service
from services.llm_service import get_answer
from utils.cache import cache_get, cache_set, make_cache_key
from utils.logger import get_logger
from utils.metrics import metrics_store
from utils.rate_limiter import limiter

logger = get_logger(__name__)
router = APIRouter()

DEFAULT_RETRIEVAL_LIMITS = {"concise": 8, "detailed": 8}
MIN_SAFE_RETRIEVAL_LIMIT = 4
MAX_ADAPTIVE_RETRIEVAL_LIMIT = 12


def _resolve_retrieval_limit(answer_length: str, top_k: int | None) -> int:
    if top_k is not None:
        return min(top_k, MAX_ADAPTIVE_RETRIEVAL_LIMIT)
    return DEFAULT_RETRIEVAL_LIMITS[answer_length]


def _answer_cache_key(
    document_id: str, question: str, answer_length: str, top_k: int
) -> str:
    return make_cache_key(
        document_id,
        f"v6-{answer_length}-{top_k}:{question}",
    )


def _cached_answer_details(cached) -> tuple[str, int]:
    if isinstance(cached, dict):
        return cached["answer"], cached.get("chunks_used", 0)
    return cached, 0


def _generate_answer(question: str, chunks, answer_length: str) -> str:
    try:
        return get_answer(question, chunks, answer_length)
    except RateLimitError as exc:
        provider_error = str(exc).casefold()
        if "tokens per day" in provider_error or "(tpd)" in provider_error:
            detail = (
                "The LLM provider's daily token quota is exhausted. "
                "Wait for it to reset, then try again."
            )
        else:
            detail = "The LLM provider is rate-limiting requests. Please wait and retry."
        raise HTTPException(status_code=429, detail=detail) from exc
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


def _needs_more_context(answer: str) -> bool:
    return "i don't have enough information to answer that" in answer.casefold()


def _retrieve_chunks(document_id: str, question: str, top_k: int):
    try:
        chunks = embedding_service.retrieve_top_k(document_id, question, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not chunks:
        raise HTTPException(status_code=422, detail="No relevant chunks found.")
    return chunks


def _generate_answer_adaptively(
    question: str,
    chunks,
    answer_length: str,
    retrieval_limit: int,
    can_adapt: bool,
    retrieve_at_limit,
):
    try:
        answer = _generate_answer(question, chunks, answer_length)
    except HTTPException as initial_error:
        if (
            initial_error.status_code == 429
            or not can_adapt
            or retrieval_limit <= MIN_SAFE_RETRIEVAL_LIMIT
        ):
            raise
        fallback_chunks = retrieve_at_limit(MIN_SAFE_RETRIEVAL_LIMIT)
        if fallback_chunks == chunks:
            raise initial_error
        return (
            _generate_answer(question, fallback_chunks, answer_length),
            fallback_chunks,
        )

    if (
        can_adapt
        and retrieval_limit < MAX_ADAPTIVE_RETRIEVAL_LIMIT
        and _needs_more_context(answer)
    ):
        wider_chunks = retrieve_at_limit(MAX_ADAPTIVE_RETRIEVAL_LIMIT)
        if wider_chunks == chunks:
            return answer, chunks
        try:
            wider_answer = _generate_answer(question, wider_chunks, answer_length)
        except HTTPException as retry_error:
            if retry_error.status_code == 429:
                raise
            logger.warning(
                "Wider-context retry failed; returning the initial grounded response."
            )
        else:
            return wider_answer, wider_chunks

    return answer, chunks


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

    retrieve_at_limit = lambda limit: _retrieve_chunks(
        body.document_id, body.question, limit
    )
    chunks = retrieve_at_limit(retrieval_limit)

    logger.info(
        f"Retrieved {len(chunks)} chunks for question: '{body.question[:60]}...'"
    )

    answer, chunks = _generate_answer_adaptively(
        body.question,
        chunks,
        body.answer_length,
        retrieval_limit,
        can_adapt=body.top_k is None,
        retrieve_at_limit=retrieve_at_limit,
    )
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
