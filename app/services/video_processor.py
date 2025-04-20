import cv2
import os
import numpy as np
from typing import List, Dict
from app.utils.config import settings
from app.utils.logger import logger
from app.services.deepfake_detector import DeepfakeDetector
from app.models.result import Result
import asyncio
from concurrent.futures import ThreadPoolExecutor


class VideoProcessor:
    def __init__(self):
        self.detector = DeepfakeDetector()
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process_video(self, video_path: str, content_id: str) -> List[Result]:
        """Main video processing pipeline"""
        try:
            # Step 1: Extract key frames
            frames = await self._extract_key_frames(video_path)

            # Step 2: Process frames in parallel
            loop = asyncio.get_event_loop()
            frame_results = await asyncio.gather(*[
                loop.run_in_executor(
                    self.executor,
                    self._process_frame,
                    frame,
                    i,
                    len(frames))
                for i, frame in enumerate(frames)
            ])

            # Step 3: Aggregate results
            return await self._generate_final_results(frame_results, content_id)

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise

    async def _extract_key_frames(self, video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """Extract representative frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()
            return frames
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise

    def _process_frame(self, frame: np.ndarray, frame_num: int, total_frames: int) -> Dict:
        """Process individual frame (runs in thread pool)"""
        try:
            # Save temporary frame for analysis
            frame_path = os.path.join(settings.MEDIA_STORAGE_PATH, f"frame_{frame_num}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Analyze frame
            result = self.detector.analyze_image(frame_path)
            os.remove(frame_path)

            return {
                "frame_num": frame_num,
                "total_frames": total_frames,
                "is_fake": result["is_fake"],
                "confidence": result["confidence"],
                "details": result.get("details", {})
            }
        except Exception as e:
            logger.error(f"Frame {frame_num} processing failed: {e}")
            return {
                "frame_num": frame_num,
                "error": str(e)
            }

    async def _generate_final_results(self, frame_results: List[Dict], content_id: str) -> List[Result]:
        """Generate final results from frame analyses"""
        valid_results = [r for r in frame_results if "error" not in r]

        if not valid_results:
            raise Exception("No valid frames processed")

        # Calculate aggregate confidence
        avg_confidence = sum(r["confidence"] for r in valid_results) / len(valid_results)
        is_fake = any(r["is_fake"] for r in valid_results)

        # Create main result
        main_result = Result(
            content_id=content_id,
            detection_type="deepfake",
            is_fake=is_fake,
            confidence=avg_confidence,
            explanation=f"Analyzed {len(valid_results)} frames. {sum(r['is_fake'] for r in valid_results)} frames detected as fake.",
            model_used="CNN+ELA",
            model_version="1.0"
        )

        return [main_result]


video_processor = VideoProcessor()