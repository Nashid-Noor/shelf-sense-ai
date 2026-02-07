"""
ROI Extractor for ShelfSense AI

Extracts and normalizes detected book regions (ROIs) for downstream
processing (OCR, embedding generation).
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image
from loguru import logger

from shelfsense.vision.spine_detector import Detection
from shelfsense.vision.detector_ensemble import EnsembleDetection


@dataclass
class ExtractedROI:
    """An extracted region of interest."""
    image: np.ndarray  # Cropped BGR image
    original_bbox: Tuple[int, int, int, int]  # Original bounding box
    source: str  # "spine" or "cover"
    detection_confidence: float
    roi_id: int
    
    @property
    def width(self) -> int:
        return self.image.shape[1]
    
    @property
    def height(self) -> int:
        return self.image.shape[0]
    
    @property
    def aspect_ratio(self) -> float:
        return self.height / self.width if self.width > 0 else 0
    
    def to_pil(self) -> Image.Image:
        """Convert to PIL Image (RGB)."""
        return Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
    
    def resize(self, size: Tuple[int, int]) -> 'ExtractedROI':
        """Return resized ROI."""
        resized = cv2.resize(self.image, size, interpolation=cv2.INTER_LINEAR)
        return ExtractedROI(
            image=resized,
            original_bbox=self.original_bbox,
            source=self.source,
            detection_confidence=self.detection_confidence,
            roi_id=self.roi_id
        )


class ROIExtractor:
    """
    Extract and preprocess detected book regions.
    
    Handles:
    - Cropping with padding
    - Aspect ratio normalization
    - Quality enhancement for OCR
    - Rotation correction for spine text
    
    Usage:
        extractor = ROIExtractor()
        rois = extractor.extract(image, detections)
        
        for roi in rois:
            # Process ROI for OCR or embedding
            pass
    """
    
    def __init__(
        self,
        padding_ratio: float = 0.1,
        min_size: int = 10,
        target_size: Optional[Tuple[int, int]] = None,
        enhance_for_ocr: bool = True
    ):
        """
        Initialize the ROI extractor.
        
        Args:
            padding_ratio: Padding around detection as ratio of box size
            min_size: Minimum dimension for extracted ROIs
            target_size: Optional target size for normalization
            enhance_for_ocr: Apply enhancement for OCR
        """
        self.padding_ratio = padding_ratio
        self.min_size = min_size
        self.target_size = target_size
        self.enhance_for_ocr = enhance_for_ocr
    
    def extract(
        self,
        image: np.ndarray,
        detections: List[Union[Detection, EnsembleDetection]]
    ) -> List[ExtractedROI]:
        """
        Extract ROIs from detected regions.
        
        Args:
            image: Source BGR image
            detections: List of detections
            
        Returns:
            List of extracted ROIs
        """
        rois = []
        
        for i, det in enumerate(detections):
            roi = self._extract_single(image, det, roi_id=i)
            if roi is not None:
                rois.append(roi)
        
        logger.debug(f"Extracted {len(rois)} ROIs from {len(detections)} detections")
        return rois
    
    def _extract_single(
        self,
        image: np.ndarray,
        detection: Union[Detection, EnsembleDetection],
        roi_id: int
    ) -> Optional[ExtractedROI]:
        """Extract a single ROI with padding and validation."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        
        # Calculate padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * self.padding_ratio)
        pad_y = int(box_h * self.padding_ratio)
        
        # Apply padding with bounds checking
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(w, x2 + pad_x)
        y2_padded = min(h, y2 + pad_y)
        
        # Crop
        crop = image[y1_padded:y2_padded, x1_padded:x2_padded].copy()
        
        # Validate size
        if crop.shape[0] < self.min_size or crop.shape[1] < self.min_size:
            logger.debug(f"ROI {roi_id} too small: {crop.shape[:2]}")
            return None
        
        # Get source if available
        source = getattr(detection, 'source', 'unknown')
        
        return ExtractedROI(
            image=crop,
            original_bbox=(x1, y1, x2, y2),
            source=source,
            detection_confidence=detection.confidence,
            roi_id=roi_id
        )
    
    def prepare_for_ocr(self, roi: ExtractedROI) -> np.ndarray:
        """
        Prepare ROI for OCR processing.
        
        Applies:
        - Contrast enhancement
        - Noise reduction
        """
        image = roi.image.copy()
        
        # Disabled forced rotation - let OCREngine handle it
        # if roi.source == "spine" and roi.aspect_ratio > 2.0:
        #     image = self._smart_rotate_spine(image)
        
        if self.enhance_for_ocr:
            image = self._enhance_image(image)
        
        return image
    
    def prepare_for_embedding(
        self,
        roi: ExtractedROI,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Prepare ROI for embedding model input.
        
        Args:
            roi: Extracted ROI
            target_size: Target size for embedding model
            
        Returns:
            Resized and normalized image
        """
        # Resize maintaining aspect ratio with padding
        image = roi.image.copy()
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        padded = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return padded
    
    def _smart_rotate_spine(self, image: np.ndarray) -> np.ndarray:
        """
        Intelligently rotate spine image for better OCR.
        
        Spine text can be:
        - Top-to-bottom (most common in US)
        - Bottom-to-top (common in UK/Europe)
        
        We try both rotations and use text detection to choose.
        """
        # Default: rotate 90 degrees clockwise
        rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Use simple heuristic: check edge density in each direction
        # More sophisticated: run OCR on both and compare confidence
        
        # For now, return clockwise rotation (most common)
        return rotated_cw
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image enhancement for OCR."""
        h, w = image.shape[:2]
        
        # 1. Upscale if small (especially for thin spines)
        # Check both dimensions - for vertical spines, width determines text height
        min_dim = min(h, w)
        if min_dim < 150:
            scale = 150 / min_dim
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 2. Sharpen
        # Gaussian unsharp mask
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # CLAHE for adaptive contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Sharpening (Unsharp Mask)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Convert back to BGR
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    def extract_and_prepare(
        self,
        image: np.ndarray,
        detections: List[Union[Detection, EnsembleDetection]],
        for_ocr: bool = True,
        for_embedding: bool = True,
        embedding_size: Tuple[int, int] = (224, 224)
    ) -> List[dict]:
        """
        Extract ROIs and prepare both OCR and embedding versions.
        
        Returns list of dicts with keys:
        - roi: ExtractedROI
        - ocr_image: Enhanced image for OCR (if for_ocr=True)
        - embedding_image: Normalized image for embedding (if for_embedding=True)
        """
        rois = self.extract(image, detections)
        
        results = []
        for roi in rois:
            result = {"roi": roi}
            
            if for_ocr:
                result["ocr_image"] = self.prepare_for_ocr(roi)
            
            if for_embedding:
                result["embedding_image"] = self.prepare_for_embedding(roi, embedding_size)
            
            results.append(result)
        
        return results


class BatchROIExtractor:
    """
    Batch extraction for multiple images.
    
    More efficient for processing multiple images at once.
    """
    
    def __init__(self, extractor: Optional[ROIExtractor] = None):
        self.extractor = extractor or ROIExtractor()
    
    def extract_batch(
        self,
        images: List[np.ndarray],
        detections_list: List[List[Detection]]
    ) -> List[List[ExtractedROI]]:
        """
        Extract ROIs from multiple images.
        
        Args:
            images: List of source images
            detections_list: List of detection lists, one per image
            
        Returns:
            List of ROI lists, one per image
        """
        if len(images) != len(detections_list):
            raise ValueError("Number of images must match number of detection lists")
        
        results = []
        for image, detections in zip(images, detections_list):
            rois = self.extractor.extract(image, detections)
            results.append(rois)
        
        return results
