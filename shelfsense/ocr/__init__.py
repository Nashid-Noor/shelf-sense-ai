"""
OCR & Text Processing Module

Handles text extraction from book images:
- Multi-engine OCR (EasyOCR, Tesseract)
- Text normalization and cleaning
- Confidence estimation
- Multi-orientation support
"""

from shelfsense.ocr.ocr_engine import OCREngine, OCRResult
from shelfsense.ocr.text_normalizer import TextNormalizer
from shelfsense.ocr.confidence_estimator import ConfidenceEstimator

__all__ = [
    "OCREngine",
    "OCRResult",
    "TextNormalizer",
    "ConfidenceEstimator",
]
