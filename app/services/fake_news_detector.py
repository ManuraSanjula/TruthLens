import os
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from app.models.result import Result, DetectionType
from app.utils.config import settings
import logging
import torch
import joblib

logger = logging.getLogger(__name__)


class FakeNewsDetector:
    def __init__(self):
        self.bert_model = None
        self.bert_tokenizer = None
        self.tfidf_vectorizer = None
        self.svm_model = None
        self.load_models()

    def load_models(self):
        """Load all required NLP models"""
        try:
            # Load BERT model
            bert_path = os.path.join(settings.MODEL_CACHE_PATH, "bert-fakenews")
            if not os.path.exists(bert_path):
                os.makedirs(bert_path, exist_ok=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    "yiyanghkust/finbert-tone",
                    cache_dir=bert_path
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    "yiyanghkust/finbert-tone",
                    cache_dir=bert_path
                )
                model.save_pretrained(bert_path)
                tokenizer.save_pretrained(bert_path)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(bert_path)
                tokenizer = AutoTokenizer.from_pretrained(bert_path)

            self.bert_model = model
            self.bert_tokenizer = tokenizer

            # Load TF-IDF and SVM model
            tfidf_path = os.path.join(settings.MODEL_CACHE_PATH, "tfidf.pkl")
            svm_path = os.path.join(settings.MODEL_CACHE_PATH, "svm_model.pkl")

            if os.path.exists(tfidf_path) and os.path.exists(svm_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                self.svm_model = joblib.load(svm_path)
            else:
                logger.warning("TF-IDF or SVM model not found")

            logger.info("All NLP models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    async def analyze_text(self, text: str) -> dict:
        """Analyze text using multiple techniques"""
        try:
            if not text.strip():
                return {"error": "Empty text provided"}

            # BERT Analysis
            bert_result = self._bert_analysis(text)

            # SVM Analysis (if available)
            svm_result = self._svm_analysis(text) if self.svm_model else None

            # Combine results
            is_fake = bert_result["is_fake"] or (svm_result["is_fake"] if svm_result else False)
            confidence = max(
                bert_result["confidence"],
                svm_result["confidence"] if svm_result else 0
            )

            return {
                "is_fake": is_fake,
                "confidence": confidence,
                "model_used": "BERT+SVM" if svm_result else "BERT",
                "details": {
                    "bert_result": bert_result,
                    "svm_result": svm_result
                }
            }
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise

    def _bert_analysis(self, text: str) -> dict:
        """Analyze text using BERT model"""
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs).item()

        # Map model's output to our labels
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        label = label_map.get(predicted_class, "Neutral")

        return {
            "is_fake": label == "Negative",
            "confidence": float(probs[0][predicted_class]),
            "sentiment": label,
            "model": "BERT"
        }

    def _svm_analysis(self, text: str) -> dict:
        """Analyze text using traditional ML approach"""
        features = self.tfidf_vectorizer.transform([text])
        prediction = self.svm_model.predict(features)[0]
        proba = self.svm_model.predict_proba(features)[0]

        return {
            "is_fake": bool(prediction),
            "confidence": float(max(proba)),
            "model": "SVM"
        }


fake_news_detector = FakeNewsDetector()


async def analyze_text_content(content_id: str, text: str) -> Result:
    """Full text analysis pipeline"""
    analysis = await fake_news_detector.analyze_text(text)
    return Result(
        content_id=content_id,
        detection_type=DetectionType.FAKE_NEWS,
        is_fake=analysis["is_fake"],
        confidence=analysis["confidence"],
        explanation=str(analysis.get("details", "")),
        model_used=analysis["model_used"],
        model_version="1.0"
    )