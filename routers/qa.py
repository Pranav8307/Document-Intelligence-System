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

    retrieval_limit = body.top_k or (50 if body.answer_length == "detailed" else 8)
    cache_key = make_cache_key(
        body.document_id,
        f"v4-{body.answer_length}-{retrieval_limit}:{body.question}",
    )
    cached = cache_get(cache_key)

    if cached:
        logger.info(f"Cache hit | key={cache_key}")
        metrics_store.increment("cache_hits")
        metrics_store.record_timing(
            "qa_request_total", (time.perf_counter() - request_start) * 1000
        )
        return APIResponse(
            status="success",
            message="Answer retrieved from cache.",
            data=AskResponse(
                answer=cached,
                chunks_used=body.top_k,
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

    try:
        answer = get_answer(body.question, chunks, body.answer_length)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    cache_set(cache_key, answer)
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
