from fastapi import APIRouter, HTTPException, Request
from models.schemas import APIResponse, AskRequest, AskResponse
from services.document_service import document_exists
from services.embedding_service import EmbeddingService
from services.llm_service import get_answer
from utils.cache import cache_get, cache_set, make_cache_key
from utils.logger import get_logger
from utils.rate_limiter import limiter

logger = get_logger(__name__)
router = APIRouter()
embedding_service = EmbeddingService()


@router.post("/ask", response_model=APIResponse)
@limiter.limit("10/minute")
async def ask_question(request: Request, body: AskRequest):
    if not document_exists(body.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{body.document_id}' not found. Upload it first.",
        )

    cache_key = make_cache_key(body.document_id, body.question)
    cached = cache_get(cache_key)

    if cached:
        logger.info(f"Cache hit | key={cache_key}")
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

    try:
        chunks = embedding_service.retrieve_top_k(
            body.document_id, body.question, top_k=body.top_k
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=422, detail="No relevant chunks found.")

    logger.info(
        f"Retrieved {len(chunks)} chunks for question: '{body.question[:60]}...'"
    )

    answer = get_answer(body.question, chunks)
    cache_set(cache_key, answer)

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