"""
Unit tests for OCR module components.
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from shelfsense.ocr.ocr_engine import OCREngine, OCRResult, TextBox
from shelfsense.ocr.text_normalizer import TextNormalizer, NormalizationResult
from shelfsense.ocr.confidence_estimator import ConfidenceEstimator, ConfidenceBreakdown


class TestOCREngine:
    """Tests for OCREngine class."""
    
    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine with mocked backends."""
        # Mock easyocr and pytesseract
        with patch("shelfsense.ocr.ocr_engine.easyocr") as mock_easyocr, \
             patch("shelfsense.ocr.ocr_engine.pytesseract") as mock_tesseract:
            
            # Configure mocks
            mock_reader = MagicMock()
            mock_easyocr.Reader.return_value = mock_reader
            mock_reader.readtext.return_value = []  # Default empty result
            
            # Initialize engine
            engine = OCREngine(use_gpu=False)
            engine.easyocr_reader = mock_reader  # Ensure explicit assignment
            return engine
    
    def test_process_returns_result(self, ocr_engine, sample_text_image):
        """Test text extraction returns OCRResult."""
        img_np = np.array(sample_text_image)
        
        ocr_engine.easyocr_reader.readtext.return_value = [
            ([[0, 0], [200, 0], [200, 50], [0, 50]], "The Great Gatsby", 0.95)
        ]
        
        # Run process (real implementation)
        result = ocr_engine.process(img_np, try_rotations=False)
        
        assert isinstance(result, OCRResult)
        assert len(result.text) > 0
    
    def test_confidence_filtering(self, ocr_engine, sample_text_image):
        """Test low confidence results are filtered."""
        img_np = np.array(sample_text_image)
        ocr_engine.confidence_threshold = 0.5
        
        ocr_engine.easyocr_reader.readtext.return_value = [
            ([[0, 0], [200, 0], [200, 50], [0, 50]], "Clear Text", 0.95),
            ([[0, 60], [200, 60], [200, 110], [0, 110]], "Blurry", 0.30),
        ]
        
        result = ocr_engine.process(img_np, try_rotations=False)
        
        # Should only include high-confidence text
        print(f"Result text: {result.text}")
        assert "Clear Text" in result.text
        # "Blurry" (0.30) < 0.5 threshold
        assert "Blurry" not in result.text
    
    def test_handles_empty_results(self, ocr_engine):
        """Test handling of images with no detected text."""
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        ocr_engine.easyocr_reader.readtext.return_value = []
        
        result = ocr_engine.process(blank_image, try_rotations=False)
        
        assert result.text == ""
        assert result.confidence == 0.0
    
    def test_multiple_text_pieces(self, ocr_engine, sample_text_image):
        """Test extraction of multiple text pieces."""
        img_np = np.array(sample_text_image)
        
        ocr_engine.easyocr_reader.readtext.return_value = [
            ([[0, 0], [200, 0], [200, 30], [0, 30]], "Title Line", 0.95),
            ([[0, 40], [150, 40], [150, 70], [0, 70]], "Author Name", 0.90),
            ([[0, 80], [100, 80], [100, 110], [0, 110]], "Publisher", 0.85),
        ]
        
        result = ocr_engine.process(img_np, try_rotations=False)
        
        assert len(result.text_boxes) == 3
        # Check that combined text contains all parts
        assert "Title Line" in result.text
        assert "Author Name" in result.text
        assert "Publisher" in result.text
    
    def test_bounding_box_extraction(self, ocr_engine, sample_text_image):
        """Test bounding box information is preserved."""
        img_np = np.array(sample_text_image)
        
        ocr_engine.easyocr_reader.readtext.return_value = [
            ([[10, 20], [100, 20], [100, 50], [10, 50]], "Test", 0.9)
        ]
        
        result = ocr_engine.process(img_np, try_rotations=False)
        
        assert result.text_boxes is not None
        assert len(result.text_boxes) == 1
        assert isinstance(result.text_boxes[0], TextBox)


class TestTextNormalizer:
    """Tests for TextNormalizer class."""
    
    @pytest.fixture
    def normalizer(self):
        """Create text normalizer instance."""
        return TextNormalizer()
    
    def test_remove_ocr_artifacts(self, normalizer):
        """Test removal of common OCR artifacts via normalize."""
        # Test noise removal
        dirty_text = "The Great Gatsby^&*"
        res = normalizer.normalize(dirty_text)
        # ^ and * are removed, & is kept
        assert "The Great Gatsby &" in res.normalized or "The Great Gatsby&" in res.normalized
        
        # Test vv -> w pattern (requires word boundaries in regex)
        # "vv" alone is w. "vvave" might not be if regex uses \b
        text_w = "a vv big wave" 
        res = normalizer.normalize(text_w)
        assert "a w big wave" in res.normalized
    
    def test_fix_common_ocr_errors(self, normalizer):
        """Test correction of common OCR misreadings."""
        # 0 -> O in caps
        text = "F. SC0TT"
        result = normalizer.normalize(text)
        assert "SCOTT" in result.normalized
        
        # B0OK -> BOOK
        text2 = "THE B0OK"
        res2 = normalizer.normalize(text2)
        assert "BOOK" in res2.normalized

    def test_normalize_whitespace(self, normalizer):
        """Test whitespace normalization."""
        text = "The\n\nGreat\t\tGatsby"
        result = normalizer.normalize(text)
        assert "\n" not in result.normalized
        assert "\t" not in result.normalized
        assert result.normalized == "The Great Gatsby"
    
    def test_handle_unicode(self, normalizer):
        """Test handling of unicode characters."""
        text = "Les Misérables"
        result = normalizer.normalize(text)
        # Check that it handles it (strips it in current impl)
        assert "Les" in result.normalized
        assert "é" not in result.normalized

    def test_normalize_case(self, normalizer):
        """Test case normalization if enabled."""
        norm = TextNormalizer(normalize_case=True)
        text = "the great gatsby"
        result = norm.normalize(text)
        assert result.normalized == "The Great Gatsby"


class TestConfidenceEstimator:
    """Tests for ConfidenceEstimator class."""
    
    @pytest.fixture
    def estimator(self):
        """Create confidence estimator instance."""
        return ConfidenceEstimator()
    
    def test_estimate_returns_breakdown(self, estimator):
        """Test confidence estimation returns ConfidenceBreakdown."""
        breakdown = estimator.estimate("The Great Gatsby", 0.9)
        assert isinstance(breakdown, ConfidenceBreakdown)
        assert 0.0 <= breakdown.final_confidence <= 1.0

    def test_penalize_short_text(self, estimator):
        """Test confidence penalty for very short text."""
        short_text = "AB"
        long_text = "The Great Gatsby by F. Scott Fitzgerald"
        
        bd_short = estimator.estimate(short_text, 0.9)
        bd_long = estimator.estimate(long_text, 0.9)
        
        assert "very_short" in bd_short.flags
        assert bd_short.final_confidence < bd_long.final_confidence
    
    def test_penalize_suspicious_patterns(self, estimator):
        """Test confidence penalty for suspicious patterns."""
        normal_text = "The Great Gatsby"
        suspicious_text = "ThGxrzPlmknnq" 
        
        bd_normal = estimator.estimate(normal_text, 0.9)
        bd_suspicious = estimator.estimate(suspicious_text, 0.9)
        
        assert bd_suspicious.final_confidence < bd_normal.final_confidence
    
    def test_boost_known_words(self, estimator):
        """Test implicit boost (via language score) for known words."""
        english = "The Great Gatsby"
        gibberish = "Xqz Plmk Rtyu"
        
        bd_eng = estimator.estimate(english, 0.8)
        bd_gib = estimator.estimate(gibberish, 0.8)
        
        assert bd_eng.language_score > bd_gib.language_score

    def test_estimate_batch(self, estimator):
        """Test batch estimation."""
        texts = ["Text A", "Text B"]
        confs = [0.9, 0.8]
        results = estimator.estimate_batch(texts, confs)
        
        assert len(results) == 2
        assert isinstance(results[0], ConfidenceBreakdown)


class TestOCRPipeline:
    """Integration tests for OCR pipeline."""
    
    def test_full_pipeline_flow(self, sample_text_image):
        """Test full flow utilizing OCREngine, Normalizer, and Estimator."""
        # 1. Setup OCR Engine
        with patch("shelfsense.ocr.ocr_engine.easyocr") as mock_easyocr, \
             patch("shelfsense.ocr.ocr_engine.pytesseract"):
            
            mock_reader = MagicMock()
            mock_easyocr.Reader.return_value = mock_reader
            
            engine = OCREngine()
            engine.languages = ["en"]
            engine.prefer_easyocr = True
            engine.easyocr_reader = mock_reader
            engine.easyocr_reader.readtext.return_value = [
                ([[0, 0], [200, 0], [200, 50], [0, 50]], "THE GREAT GATSBY", 0.95)
            ]
            engine.confidence_threshold = 0.5
            
            # 2. Setup others
            normalizer = TextNormalizer(normalize_case=True)
            estimator = ConfidenceEstimator()
            
            # 3. Process
            img_np = np.array(sample_text_image)
            ocr_result = engine.process(img_np, try_rotations=False)
            
            # 4. Normalize
            norm_result = normalizer.normalize(ocr_result.text)
            
            # 5. Estimate
            confidence = estimator.estimate(
                norm_result.normalized, 
                ocr_result.confidence + norm_result.confidence_adjustment
            )
            
            assert "The Great Gatsby" in norm_result.normalized
            assert confidence.final_confidence > 0.6
