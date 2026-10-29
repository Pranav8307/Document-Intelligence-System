from pydantic import BaseModel, Field
from typing import Any, Literal, Optional


class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None


class UploadResponse(BaseModel):
    document_id: str
    chunk_count: int
    filename: str


class AskRequest(BaseModel):
    document_id: str = Field(..., description="ID returned from /upload")
    question: str = Field(..., min_length=3, max_length=500)
    answer_length: Literal["concise", "detailed"] = Field(
        default="detailed", description="Controls context size and answer length"
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="Optional retrieval limit (1–12); omitted requests adapt between 4 and 12 matches",
    )


class AskResponse(BaseModel):
    answer: str
    chunks_used: int
    cached: bool
    document_id: str


class TimingStats(BaseModel):
    count: int
    total_ms: float
    avg_ms: float
    max_ms: float
    min_ms: float


class MetricsResponse(BaseModel):
    counters: dict[str, int]
    timings_ms: dict[str, TimingStats]
