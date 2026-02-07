"""
Computer Vision Module for ShelfSense AI

This module handles all visual processing:
- Image preprocessing and enhancement
- Layout classification (bookshelf/cover/mixed)
- Book spine detection
- Book cover detection
- ROI extraction and preprocessing
"""

from shelfsense.vision.preprocessing import ImagePreprocessor, PreprocessConfig
from shelfsense.vision.layout_classifier import LayoutClassifier, LayoutType
from shelfsense.vision.spine_detector import SpineDetector
from shelfsense.vision.cover_detector import CoverDetector
from shelfsense.vision.detector_ensemble import DetectorEnsemble
from shelfsense.vision.roi_extractor import ROIExtractor, ExtractedROI

__all__ = [
    "ImagePreprocessor",
    "PreprocessConfig",
    "LayoutClassifier",
    "LayoutType",
    "SpineDetector",
    "CoverDetector",
    "DetectorEnsemble",
    "ROIExtractor",
    "ExtractedROI",
]
