"""
Detector Ensemble for ShelfSense AI

Orchestrates layout classification and detection routing.
Combines results from spine and cover detectors based on scene type.
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import cv2
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

from shelfsense.vision.layout_classifier import LayoutClassifier, LayoutType, LayoutPrediction
from shelfsense.vision.spine_detector import SpineDetector, Detection, DetectionResult
from shelfsense.vision.cover_detector import CoverDetector


@dataclass
class EnsembleDetection(Detection):
    """Detection with source information."""
    source: str = "unknown"  # "spine" or "cover"
    
    def to_dict(self):
        d = super().to_dict()
        d["source"] = self.source
        return d


@dataclass
class EnsembleResult:
    """Combined result from the detector ensemble."""
    layout: LayoutPrediction
    detections: List[EnsembleDetection] = field(default_factory=list)
    spine_detections: List[Detection] = field(default_factory=list)
    cover_detections: List[Detection] = field(default_factory=list)
    image_shape: Tuple[int, int, int] = (0, 0, 0)
    total_inference_time_ms: float = 0.0
    
    def __len__(self) -> int:
        return len(self.detections)
    
    @property
    def has_spines(self) -> bool:
        return len(self.spine_detections) > 0
    
    @property
    def has_covers(self) -> bool:
        return len(self.cover_detections) > 0
    
    def filter_by_confidence(self, threshold: float) -> 'EnsembleResult':
        """Filter detections by confidence threshold."""
        return EnsembleResult(
            layout=self.layout,
            detections=[d for d in self.detections if d.confidence >= threshold],
            spine_detections=[d for d in self.spine_detections if d.confidence >= threshold],
            cover_detections=[d for d in self.cover_detections if d.confidence >= threshold],
            image_shape=self.image_shape,
            total_inference_time_ms=self.total_inference_time_ms
        )


class DetectorEnsemble:
    """
    Unified interface for book detection.
    
    Orchestrates:
    1. Layout classification to determine scene type
    2. Routing to appropriate detector(s)
    3. Merging and deduplicating results
    
    Usage:
        ensemble = DetectorEnsemble()
        result = ensemble.detect(image)
        
        print(f"Layout: {result.layout.layout}")
        print(f"Found {len(result.detections)} books")
    """
    
    def __init__(
        self,
        layout_classifier: Optional[LayoutClassifier] = None,
        spine_detector: Optional[SpineDetector] = None,
        cover_detector: Optional[CoverDetector] = None,
        layout_model_path: Optional[str] = None,
        spine_model_path: Optional[str] = None,
        cover_model_path: Optional[str] = None,
        device: Optional[str] = None,
        iou_merge_threshold: float = 0.5
    ):
        """
        Initialize the detector ensemble.
        
        Args:
            layout_classifier: Pre-initialized layout classifier
            spine_detector: Pre-initialized spine detector
            cover_detector: Pre-initialized cover detector
            layout_model_path: Path to layout classifier weights
            spine_model_path: Path to spine detector weights
            cover_model_path: Path to cover detector weights
            device: Device for inference
            iou_merge_threshold: IoU threshold for merging overlapping detections
        """
        self.device = device
        self.iou_merge_threshold = iou_merge_threshold
        
        # Initialize components
        self.layout_classifier = layout_classifier or LayoutClassifier(
            model_path=layout_model_path,
            device=device
        )
        
        self.spine_detector = spine_detector or SpineDetector(
            model_path=spine_model_path,
            device=device
        )
        
        self.cover_detector = cover_detector or CoverDetector(
            model_path=cover_model_path,
            device=device
        )
        
        # Thread pool for parallel detection
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info("DetectorEnsemble initialized")
    
    def detect(
        self,
        image: np.ndarray,
        force_layout: Optional[LayoutType] = None,
        confidence: Optional[float] = None
    ) -> EnsembleResult:
        """
        Detect books in an image.
        
        Args:
            image: BGR numpy array
            force_layout: Override layout classification (for testing)
            
        Returns:
            EnsembleResult with all detections
        """
        import time
        start_time = time.time()
        
        # Step 1: Classify layout
        if force_layout:
            layout = LayoutPrediction(
                layout=force_layout,
                confidence=1.0,
                probabilities={force_layout.value: 1.0}
            )
        else:
            layout = self.layout_classifier.predict(image)
        
        logger.debug(f"Layout: {layout.layout.value} (conf: {layout.confidence:.2f})")
        
        # Step 2: Run appropriate detector(s)
        spine_result = DetectionResult()
        cover_result = DetectionResult()
        
        if layout.should_run_spine_detector():
            spine_result = self.spine_detector.detect(image, confidence=confidence)
            logger.debug(f"Spine detector found {len(spine_result)} detections")
        
        if layout.should_run_cover_detector():
            cover_result = self.cover_detector.detect(image, confidence=confidence)
            logger.debug(f"Cover detector found {len(cover_result)} detections")
        
        # Step 3: Merge and deduplicate
        all_detections = self._merge_detections(
            spine_result.detections,
            cover_result.detections
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return EnsembleResult(
            layout=layout,
            detections=all_detections,
            spine_detections=spine_result.detections,
            cover_detections=cover_result.detections,
            image_shape=image.shape,
            total_inference_time_ms=total_time
        )
    
    async def detect_async(
        self,
        image: np.ndarray,
        force_layout: Optional[LayoutType] = None,
        confidence: Optional[float] = None
    ) -> EnsembleResult:
        """
        Async version of detect() for better throughput.
        
        Runs spine and cover detection in parallel when needed.
        """
        import time
        start_time = time.time()
        
        # Layout classification
        loop = asyncio.get_event_loop()
        
        if force_layout:
            layout = LayoutPrediction(
                layout=force_layout,
                confidence=1.0,
                probabilities={force_layout.value: 1.0}
            )
        else:
            layout = await loop.run_in_executor(
                self._executor,
                self.layout_classifier.predict,
                image
            )
        
        # Parallel detection
        spine_result = DetectionResult()
        cover_result = DetectionResult()
        
        tasks = []
        
        if layout.should_run_spine_detector():
            tasks.append(("spine", loop.run_in_executor(
                self._executor,
                lambda img: self.spine_detector.detect(img, confidence=confidence),
                image
            )))
        
        if layout.should_run_cover_detector():
            tasks.append(("cover", loop.run_in_executor(
                self._executor,
                lambda img: self.cover_detector.detect(img, confidence=confidence),
                image
            )))
        
        # Wait for all detections
        for name, task in tasks:
            result = await task
            if name == "spine":
                spine_result = result
            else:
                cover_result = result
        
        # Merge
        all_detections = self._merge_detections(
            spine_result.detections,
            cover_result.detections
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return EnsembleResult(
            layout=layout,
            detections=all_detections,
            spine_detections=spine_result.detections,
            cover_detections=cover_result.detections,
            image_shape=image.shape,
            total_inference_time_ms=total_time
        )
    
    def _merge_detections(
        self,
        spine_detections: List[Detection],
        cover_detections: List[Detection]
    ) -> List[EnsembleDetection]:
        """
        Merge detections from both sources, removing duplicates.
        
        Uses IoU-based matching to identify overlapping detections.
        When overlap is found, keeps the higher confidence one.
        """
        # Convert to ensemble detections with source tag
        all_dets = []
        
        for det in spine_detections:
            all_dets.append(EnsembleDetection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name="book_spine",
                source="spine"
            ))
        
        for det in cover_detections:
            all_dets.append(EnsembleDetection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name="book_cover",
                source="cover"
            ))
        
        if len(all_dets) <= 1:
            return all_dets
        
        # NMS-style deduplication
        # Sort by confidence descending
        all_dets.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, det_i in enumerate(all_dets):
            if i in suppressed:
                continue
            
            keep.append(det_i)
            
            for j, det_j in enumerate(all_dets[i+1:], start=i+1):
                if j in suppressed:
                    continue
                
                should_merge = self._should_merge(det_i.bbox, det_j.bbox)
                if should_merge:
                    suppressed.add(j)
        
        return keep

    def _should_merge(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
        """
        Check if two boxes should be merged based on IoU or Containment (IoA).
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # IoU
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        
        # IoA (Intersection over Area of smaller box) - Containment check
        min_area = min(area1, area2)
        ioa = intersection / min_area if min_area > 0 else 0
        
        # Merge if standard overlapping OR meaningful containment
        return iou > self.iou_merge_threshold or ioa > 0.8
    
    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def visualize(
        self,
        image: np.ndarray,
        result: EnsembleResult,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Visualize ensemble detection results.
        
        Uses different colors for spine vs cover detections.
        """
        vis_image = image.copy()
        
        # Color scheme
        colors = {
            "spine": (0, 255, 0),   # Green for spines
            "cover": (255, 0, 0),   # Blue for covers
            "unknown": (0, 255, 255)  # Yellow for unknown
        }
        
        for det in result.detections:
            color = colors.get(det.source, colors["unknown"])
            
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                2
            )
            
            if show_confidence:
                label = f"{det.source} {det.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Background
                cv2.rectangle(
                    vis_image,
                    (det.x1, det.y1 - label_size[1] - 5),
                    (det.x1 + label_size[0], det.y1),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    vis_image,
                    label,
                    (det.x1, det.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        # Add layout info
        layout_text = f"Layout: {result.layout.layout.value} ({result.layout.confidence:.2f})"
        cv2.putText(
            vis_image,
            layout_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return vis_image
