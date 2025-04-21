from app.models import Result, Content, ContentStatus, ContentType
from app.services.fake_news_detector import fake_news_detector
from app.services.url_processor import url_processor
from app.services.deepfake_detector import deepfake_detector
from app.services.video_processor import video_processor
import logging

logger = logging.getLogger(__name__)
async def process_content(content_id: str):
    """Directly process content without message queue"""
    try:
        content = await Content.get(content_id)
        if not content:
            logger.error(f"Content not found: {content_id}")
            return

        await Content.update_status(content_id, ContentStatus.PROCESSING)

        # Process based on content type
        if content.content_type == ContentType.TEXT:
            results = await _process_text_content(content)
        elif content.content_type == ContentType.URL:
            results = await _process_url_content(content, content_id)
        elif content.content_type == ContentType.IMAGE:
            results = await _process_image_content(content)
        elif content.content_type == ContentType.VIDEO:
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



async def _process_url_content(content: Content, content_id: str) -> list[Result]:
    """Direct URL processing"""
    try:
        # Get URL content
        extracted = await url_processor.process_url(content.source_url, content_id)

        # Check if there was an error
        if "error" in extracted and not extracted.get("content"):
            # Return a result indicating failure
            return [Result(
                content_id=str(content.id),
                detection_type="fake_news",
                is_fake=False,  # We don't know, so default to false
                confidence=0.0,
                explanation=f"URL processing error: {extracted.get('error', 'Unknown error')}",
                model_used="URL_PROCESSING_FAILED",
                model_version="1.0"
            )]

        # Ensure content is not empty before analysis
        text_content = extracted.get("content", "").strip()
        if not text_content:
            return [Result(
                content_id=str(content.id),
                detection_type="fake_news",
                is_fake=False,
                confidence=0.0,
                explanation="No text content found in URL",
                model_used="URL_PROCESSING_NO_CONTENT",
                model_version="1.0"
            )]

        # Analyze the extracted content
        analysis = await fake_news_detector.analyze_text(text_content)

        # Return the analysis result
        return [Result(
            content_id=str(content.id),
            detection_type="fake_news",
            is_fake=analysis["is_fake"],
            confidence=analysis["confidence"],
            explanation=f"URL analysis: {str(analysis.get('details', ''))}",
            model_used=f"URL+{analysis['model_used']}",
            model_version="1.0"
        )]
    except Exception as e:
        logger.error(f"Error processing URL content: {e}")
        # Return a result indicating an exception occurred
        return [Result(
            content_id=str(content.id),
            detection_type="fake_news",
            is_fake=False,
            confidence=0.0,
            explanation=f"Exception during URL processing: {str(e)}",
            model_used="URL_PROCESSING_EXCEPTION",
            model_version="1.0"
        )]


async def _process_image_content(content: Content) -> list[Result]:
    try:
        analysis = await deepfake_detector.analyze_image(content.file_path)
        print(analysis)
        # Ensure required fields exist
        model_used = analysis.get("model_used", "unknown_model")
        explanation = str(analysis.get("details", ""))

        return [Result(
            content_id=str(content.id),
            detection_type="image_manipulation",
            is_fake=analysis.get("is_fake", False),
            confidence=analysis.get("confidence", 0.0),
            explanation=explanation,
            model_used=model_used,
            model_version="1.0"
        )]
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return [Result(
            content_id=str(content.id),
            detection_type="image_manipulation",
            is_fake=False,
            confidence=0.0,
            explanation=f"Processing error: {str(e)}",
            model_used="error_handling",
            model_version="1.0"
        )]


async def _process_video_content(content: Content) -> list[Result]:
    """Direct video processing"""
    return await video_processor.process_video(content.file_path, str(content.id))