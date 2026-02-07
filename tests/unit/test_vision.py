"""
Unit tests for vision module components.
"""

import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from shelfsense.vision.preprocessing import ImagePreprocessor
from shelfsense.vision.layout_classifier import LayoutClassifier, LayoutType, LayoutPrediction
from shelfsense.vision.roi_extractor import ROIExtractor, ExtractedROI, BatchROIExtractor
from shelfsense.vision.detector_ensemble import DetectorEnsemble, EnsembleDetection
from shelfsense.vision.spine_detector import SpineDetector, Detection, DetectionResult


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return ImagePreprocessor()
    
    def test_preprocess_for_detector_resizing(self, preprocessor, sample_bookshelf_image):
        """Test resizing logic for detector input."""
        img_np = np.array(sample_bookshelf_image)
        
        processed, scale, pad = preprocessor.preprocess_for_detector(img_np)
        
        assert processed.shape == (640, 640, 3)
        assert scale > 0
        assert isinstance(pad, tuple)

    def test_preprocess_for_classifier(self, preprocessor, sample_bookshelf_image):
        """Test preprocessing for classifier."""
        img_np = np.array(sample_bookshelf_image)
        tensor = preprocessor.preprocess_for_classifier(img_np)
        
        assert tensor.shape == (1, 3, 224, 224)

    def test_enhance_for_ocr(self, preprocessor, sample_bookshelf_image):
        """Test enhancement for OCR."""
        img_np = np.array(sample_bookshelf_image)
        enhanced = preprocessor.enhance_for_ocr(img_np)
        
        assert enhanced.shape[-1] == 3
        assert enhanced.dtype == np.uint8


class TestLayoutClassifier:
    """Tests for LayoutClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create layout classifier with mocked model."""
        with patch("shelfsense.vision.layout_classifier.models.mobilenet_v3_small"), \
             patch("shelfsense.vision.layout_classifier.LayoutClassifierModel"):
            
            clf = LayoutClassifier()
            clf.model = Mock()
            return clf

    def test_predict_returns_prediction(self, classifier, sample_bookshelf_image):
        """Test predict method."""
        img_np = np.array(sample_bookshelf_image)
        
        # Determine tensor shape expected by model (B, C, H, W)
        # Mock model.predict_proba which is called by predict
        classifier.model.predict_proba.return_value = torch.tensor([[0.9, 0.1, 0.0]])
        
        result = classifier.predict(img_np)
        
        assert isinstance(result, LayoutPrediction)
        assert result.layout == LayoutType.BOOKSHELF
        assert result.confidence > 0.8


class TestROIExtractor:
    """Tests for ROIExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create ROI extractor instance."""
        return ROIExtractor()
    
    def test_extract_roi_from_detection(self, extractor, sample_bookshelf_image):
        """Test ROI extraction from Detection object."""
        img_np = np.array(sample_bookshelf_image)
        
        # Width > 50 (min_size)
        det = Detection(
            bbox=(50, 100, 150, 400),
            confidence=0.9,
            class_id=0,
            class_name="spine"
        )
        
        rois = extractor.extract(img_np, [det])
        
        assert len(rois) == 1
        assert isinstance(rois[0], ExtractedROI)
        assert rois[0].original_bbox == (50, 100, 150, 400)
    
    def test_extract_with_padding(self, extractor, sample_bookshelf_image):
        """Test ROI extraction includes padding."""
        img_np = np.array(sample_bookshelf_image)
        det = Detection((100, 100, 200, 300), 0.9, 0, "spine")
        
        extractor.padding_ratio = 0.1
        rois = extractor.extract(img_np, [det])
        
        assert rois[0].width >= 100
        assert rois[0].height >= 200

    def test_batch_extract(self, sample_bookshelf_image):
        """Test batch extraction via BatchROIExtractor."""
        batch_extractor = BatchROIExtractor()
        img_np = np.array(sample_bookshelf_image)
        
        det1 = Detection((50, 100, 150, 400), 0.9, 0, "spine")
        det2 = Detection((100, 100, 200, 400), 0.9, 0, "spine")
        
        images = [img_np]
        detections_list = [[det1, det2]]
        
        results = batch_extractor.extract_batch(images, detections_list)
        
        assert len(results) == 1
        assert len(results[0]) == 2


class TestDetectorEnsemble:
    """Tests for detector ensemble functionality."""
    
    def test_merge_detections_nms(self):
        """Test merging detections handles overlap (NMS-style)."""
        ensemble = DetectorEnsemble.__new__(DetectorEnsemble)
        ensemble.iou_merge_threshold = 0.5
        
        # Two overlapping detections (same object), sorted by confidence logic
        det1 = Detection((50, 100, 90, 400), 0.95, 0, "spine")
        det2 = Detection((55, 105, 95, 405), 0.80, 0, "cover")
        
        det3 = Detection((200, 100, 240, 400), 0.90, 0, "spine")
        
        spines = [det1, det3]
        covers = [det2]
        
        merged = ensemble._merge_detections(spines, covers)
        
        assert len(merged) == 2
        confs = [d.confidence for d in merged]
        assert 0.95 in confs
        assert 0.90 in confs
    
    def test_compute_iou(self):
        """Test IoU calculation."""
        ensemble = DetectorEnsemble.__new__(DetectorEnsemble)
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = ensemble._compute_iou(box1, box2)
        assert 0.14 < iou < 0.15


class TestSpineDetector:
    """Tests for spine-specific detection."""
    
    @pytest.fixture
    def detector(self):
        """Create spine detector with mocked model."""
        with patch("shelfsense.vision.spine_detector.YOLO"):
            det = SpineDetector(model_path="dummy.pt")
            det.confidence_threshold = 0.5
            return det
    
    def test_filter_for_spines(self, detector):
        """Test filtering logic."""
        d1 = Detection((0, 0, 20, 100), 0.9, 0, "spine")
        d2 = Detection((0, 0, 100, 100), 0.9, 0, "spine")
        d3 = Detection((0, 0, 5, 5), 0.9, 0, "spine")
        
        result = DetectionResult(detections=[d1, d2, d3], image_shape=(1000, 1000, 3))
        filtered_result = detector._filter_for_spines(result, (1000, 1000, 3))
        
        assert len(filtered_result.detections) == 1
        assert filtered_result.detections[0] == d1
