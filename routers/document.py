from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import APIResponse, UploadResponse
from services.document_service import process_and_store_document
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_TYPES = {"application/pdf", "text/plain"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload", response_model=APIResponse)
async def upload_document(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use PDF or TXT.",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        document_id, chunk_count = process_and_store_document(
            file_bytes, file.filename
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    logger.info(f"Upload success: {file.filename} → {document_id}")

    return APIResponse(
        status="success",
        message="Document uploaded and indexed successfully.",
        data=UploadResponse(
            document_id=document_id,
            chunk_count=chunk_count,
            filename=file.filename,
        ),
    )