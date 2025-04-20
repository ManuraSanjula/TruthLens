import asyncio
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.content_processor import process_content
from app.models import Content, ContentCreate
from app.utils.storage import save_uploaded_file
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/submit")
async def submit_content(
        content_type: str = Form(...),
        source_url: Optional[str] = Form(None),
        raw_text: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        user_id: str = Form(...)
):
    """Simplified endpoint without queueing"""
    try:
        # Validate and prepare content
        if content_type == "url" and source_url:
            content_data = ContentCreate(
                user_id=user_id,
                content_type="url",
                source_url=source_url
            )
        elif content_type == "text" and raw_text:
            content_data = ContentCreate(
                user_id=user_id,
                content_type="text",
                raw_text=raw_text
            )
        elif content_type in ["image", "video"] and file:
            file_path = await save_uploaded_file(file)
            content_data = ContentCreate(
                user_id=user_id,
                content_type=content_type,
                file_path=file_path
            )
        else:
            raise HTTPException(400, "Invalid content submission")

        # Create content record
        content = await Content.create(content_data)

        # Process immediately (no queue)
        asyncio.create_task(process_content(str(content.id)))

        return {
            "message": "Content processing started",
            "content_id": str(content.id),
            "status": "processing"
        }

    except Exception as e:
        logger.error(f"Submission error: {e}")
        raise HTTPException(500, "Processing failed")


@router.get("/status/{content_id}")
async def get_status(content_id: str):
    """Check processing status"""
    content = await Content.get(content_id)
    if not content:
        raise HTTPException(404, "Content not found")
    return {"status": content.status}