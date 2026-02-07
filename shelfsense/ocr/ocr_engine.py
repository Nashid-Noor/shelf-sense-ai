"""
OCR Engine

Multi-strategy OCR with EasyOCR and Tesseract.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import cv2
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import time

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available. Install with: pip install pytesseract")


@dataclass
class TextBox:
    """A single detected text region."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    
    def __str__(self) -> str:
        return f"'{self.text}' ({self.confidence:.2f})"


@dataclass
class OCRResult:
    """Result of OCR on an image."""
    text: str  # Combined text
    confidence: float  # Average confidence
    text_boxes: List[TextBox] = field(default_factory=list)
    rotation_applied: int = 0  # Degrees rotated
    engine_used: str = "unknown"
    processing_time_ms: float = 0.0
    raw_output: Any = None
    
    @property
    def has_text(self) -> bool:
        return len(self.text.strip()) > 0
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    def get_high_confidence_text(self, threshold: float = 0.5) -> str:
        """Get only high-confidence text."""
        high_conf = [tb.text for tb in self.text_boxes if tb.confidence >= threshold]
        return " ".join(high_conf)


class OCREngine:
    """
    Multi-engine OCR wrapper.
    """
    
    DEFAULT_LANGUAGES = ['en']
    ROTATION_ANGLES = [0, 90, 180, 270]  # Try all orientations
    SPINE_ROTATION_ANGLES = [0, 90, 270]  # Skip 180 for spines
    
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = True,
        confidence_threshold: float = 0.2,
        prefer_easyocr: bool = True
    ):
        """
        Initialize the OCR engine.
        
        Args:
            languages: List of language codes
            use_gpu: Use GPU acceleration if available
            confidence_threshold: Minimum confidence to include text
            prefer_easyocr: Prefer EasyOCR over Tesseract
        """
        self.languages = languages or self.DEFAULT_LANGUAGES
        self.confidence_threshold = confidence_threshold
        self.prefer_easyocr = prefer_easyocr
        
        # Initialize EasyOCR
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE and prefer_easyocr:
            try:
                self.easyocr_reader = easyocr.Reader(
                    self.languages,
                    gpu=use_gpu,
                    verbose=False
                )
                logger.info(f"EasyOCR initialized (GPU: {use_gpu})")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
        
        # Tesseract config
        self.tesseract_config = '--oem 3 --psm 6'
        
        if not EASYOCR_AVAILABLE and not TESSERACT_AVAILABLE:
            raise RuntimeError("No OCR engine available. Install easyocr or pytesseract.")
        
        logger.info("OCREngine initialized")
    
    def process(
        self,
        image: np.ndarray,
        is_spine: bool = False,
        try_rotations: bool = True
    ) -> OCRResult:
        """
        Process an image and extract text.
        
        Args:
            image: BGR numpy array
            is_spine: Whether this is a book spine (affects rotation strategy)
            try_rotations: Whether to try multiple orientations
            
        Returns:
            OCRResult with extracted text
        """
        start_time = time.time()
        
        # Determine rotation angles to try
        if try_rotations:
            # For spines, prioritize 90 and 270 degrees (vertical text)
            if is_spine:
                 angles = [90, 270, 0] 
            else:
                 angles = self.ROTATION_ANGLES
        else:
            angles = [0]
        
        best_result = None
        best_confidence = -1
        
        for angle in angles:
            # Rotate image
            if angle != 0:
                rotated = self._rotate_image(image, angle)
            else:
                rotated = image
            
            # Try OCR
            result = self._process_single(rotated)
            result.rotation_applied = angle
            
            # Track best result
            if result.confidence > best_confidence:
                best_confidence = result.confidence
                best_result = result
            
            # Early exit if we find high confidence result
            if best_confidence > 0.8:
                break
        
        if best_result is None:
            best_result = OCRResult(
                text="",
                confidence=0.0,
                engine_used="none"
            )
        
        best_result.processing_time_ms = (time.time() - start_time) * 1000
        
        return best_result
    
    def _process_single(self, image: np.ndarray) -> OCRResult:
        """Process a single orientation."""
        # Try EasyOCR first
        if self.easyocr_reader is not None:
            result = self._process_easyocr(image)
            if result.confidence > 0.3 or not TESSERACT_AVAILABLE:
                return result
        
        # Fall back to Tesseract ONLY if available
        if TESSERACT_AVAILABLE:
            return self._process_tesseract(image)
        
        return OCRResult(text="", confidence=0.0, engine_used="none")
    
    def _process_easyocr(self, image: np.ndarray) -> OCRResult:
        """Process with EasyOCR."""
        try:
            # EasyOCR expects RGB or BGR
            results = self.easyocr_reader.readtext(
                image,
                detail=1,
                paragraph=False,
                adjust_contrast=0.5,
                text_threshold=0.6,
                low_text=0.3
            )
            
            text_boxes = []
            texts = []
            confidences = []
            
            for detection in results:
                bbox_points, text, confidence = detection
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Convert polygon to bbox
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                bbox = (
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                )
                
                text_boxes.append(TextBox(
                    text=text,
                    confidence=confidence,
                    bbox=bbox
                ))
                texts.append(text)
                confidences.append(confidence)
            
            # Combine text
            combined_text = " ".join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                text=combined_text,
                confidence=float(avg_confidence),
                text_boxes=text_boxes,
                engine_used="easyocr",
                raw_output=results
            )
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return OCRResult(text="", confidence=0.0, engine_used="easyocr_error")
    
    def _process_tesseract(self, image: np.ndarray) -> OCRResult:
        """Process with Tesseract."""
        try:
            # Convert to grayscale for Tesseract
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Get detailed output
            data = pytesseract.image_to_data(
                gray,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            text_boxes = []
            texts = []
            confidences = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if not text or conf < self.confidence_threshold * 100:
                    continue
                
                # Get bounding box
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                confidence = conf / 100.0
                
                text_boxes.append(TextBox(
                    text=text,
                    confidence=confidence,
                    bbox=(x, y, x + w, y + h)
                ))
                texts.append(text)
                confidences.append(confidence)
            
            combined_text = " ".join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                text=combined_text,
                confidence=float(avg_confidence),
                text_boxes=text_boxes,
                engine_used="tesseract",
                raw_output=data
            )
            
        except Exception as e:
            # Check for "not installed" or "not in PATH" error
            error_str = str(e).lower()
            if "not installed" in error_str or "not in your path" in error_str:
                logger.warning("Tesseract not found. Disabling Tesseract fallback for this session.")
                global TESSERACT_AVAILABLE
                TESSERACT_AVAILABLE = False
            else:
                logger.error(f"Tesseract error: {e}")
            
            return OCRResult(text="", confidence=0.0, engine_used="tesseract_error")
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle."""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    
    def process_batch(
        self,
        images: List[np.ndarray],
        is_spine: List[bool] = None
    ) -> List[OCRResult]:
        """
        Process multiple images.
        
        For efficiency, processes sequentially but with early rotation exit.
        """
        if is_spine is None:
            is_spine = [False] * len(images)
        
        results = []
        for img, spine in zip(images, is_spine):
            result = self.process(img, is_spine=spine)
            results.append(result)
        
        return results


class AsyncOCREngine:
    """
    Async wrapper for OCR processing.
    
    Allows concurrent processing of multiple ROIs.
    """
    
    def __init__(self, engine: Optional[OCREngine] = None, max_workers: int = 4):
        self.engine = engine or OCREngine()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_async(
        self,
        image: np.ndarray,
        is_spine: bool = False
    ) -> OCRResult:
        """Process image asynchronously."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.engine.process,
            image,
            is_spine
        )
    
    async def process_batch_async(
        self,
        images: List[np.ndarray],
        is_spine: List[bool] = None
    ) -> List[OCRResult]:
        """Process batch asynchronously."""
        import asyncio
        
        if is_spine is None:
            is_spine = [False] * len(images)
        
        tasks = [
            self.process_async(img, spine)
            for img, spine in zip(images, is_spine)
        ]
        
        return await asyncio.gather(*tasks)
