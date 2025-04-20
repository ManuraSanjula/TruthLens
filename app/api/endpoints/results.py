from fastapi import APIRouter, HTTPException
from typing import List
from app.models.results import Result
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/content/{content_id}", response_model=List[Result])
async def get_content_results(content_id: str):
    try:
        results = await Result.get_by_content(content_id)
        if not results:
            raise HTTPException(status_code=404, detail="No results found for this content")
        return results
    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        raise HTTPException(status_code=500, detail=str(e))