from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from routers import document, qa
from utils.logger import get_logger

logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Document Q&A API",
    description="Upload documents and ask questions using LLM-powered retrieval.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(document.router, prefix="/documents", tags=["Documents"])
app.include_router(qa.router, prefix="/qa", tags=["Q&A"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error", "data": None},
    )


@app.get("/health")
def health():
    return {"status": "ok", "message": "Service is running", "data": None}