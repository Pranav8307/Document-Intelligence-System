from pydantic import BaseModel, Field
from typing import Any, Optional


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
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    chunks_used: int
    cached: bool
    document_id: str