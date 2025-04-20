from app.models import Result, Content, ContentStatus
from app.services import (
    fake_news_detector,
    deepfake_detector,
    video_processor,
    url_processor
)
from app.utils import logger
from typing import Optional
import asyncio


async def process_content(content_id: str):
    """Directly process content without message queue"""
    try:
        content = await Content.get(content_id)
        if not content:
            logger.error(f"Content not found: {content_id}")
            return

        await Content.update_status(content_id, ContentStatus.PROCESSING)

        # Process based on content type
        if content.content_type == "text":
            results = await _process_text_content(content)
        elif content.content_type == "url":
            results = await _process_url_content(content)
        elif content.content_type == "image":
            results = await _process_image_content(content)
        elif content.content_type == "video":
            results = await _process_video_content(content)
        else:
            raise ValueError(f"Unknown content type: {content.content_type}")

        # Save results
        for result in results:
            await Result.create(result)

        await Content.update_status(content_id, ContentStatus.COMPLETED)
        logger.info(f"Successfully processed content {content_id}")

    except Exception as e:
        logger.error(f"Failed to process content {content_id}: {e}")
        await Content.update_status(content_id, ContentStatus.FAILED)
        raise


async def _process_text_content(content: Content) -> list[Result]:
    """Direct text processing"""
    analysis = await fake_news_detector.analyze_text(content.raw_text)
    return [Result(
        content_id=str(content.id),
        detection_type="fake_news",
        is_fake=analysis["is_fake"],
        confidence=analysis["confidence"],
        explanation=analysis.get("details", ""),
        model_used=analysis["model_used"],
        model_version="1.0"
    )]


async def _process_url_content(content: Content) -> list[Result]:
    """Direct URL processing"""
    extracted = await url_processor.process_url(content.source_url)
    analysis = await fake_news_detector.analyze_text(extracted["content"])
    return [Result(
        content_id=str(content.id),
        detection_type="fake_news",
        is_fake=analysis["is_fake"],
        confidence=analysis["confidence"],
        explanation=f"URL analysis: {analysis.get('details', '')}",
        model_used=f"URL+{analysis['model_used']}",
        model_version="1.0"
    )]


async def _process_image_content(content: Content) -> list[Result]:
    """Direct image processing"""
    analysis = await deepfake_detector.analyze_image(content.file_path)
    return [Result(
        content_id=str(content.id),
        detection_type="image_manipulation",
        is_fake=analysis["is_fake"],
        confidence=analysis["confidence"],
        explanation=str(analysis.get("details", "")),
        model_used=analysis["model_used"],
        model_version="1.0"
    )]


async def _process_video_content(content: Content) -> list[Result]:
    """Direct video processing"""
    return await video_processor.process_video(content.file_path, str(content.id))