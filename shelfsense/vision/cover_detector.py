"""
Book Cover Detector for ShelfSense AI

YOLOv8-based detection model optimized for detecting book front covers.
Handles:
- Various orientations (flat, tilted, held)
- Perspective distortion
- Partial occlusion
- Reflections and glare
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import cv2
import torch
from pathlib import Path
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from shelfsense.vision.spine_detector import Detection, DetectionResult


class CoverDetector:
    """
    YOLOv8-based book cover detector.
    
    Optimized for detecting book front covers:
    - Rectangular book-like aspect ratios
    - Handles perspective and rotation
    - Post-processing for quality filtering
    
    Usage:
        detector = CoverDetector("path/to/weights.pt")
        result = detector.detect(image)
        
        for det in result.detections:
            print(f"Book cover at {det.bbox}")
    """
    
    # Cover-specific parameters
    DEFAULT_CONFIDENCE = 0.3
    DEFAULT_IOU_THRESHOLD = 0.5
    MIN_ASPECT_RATIO = 0.5  # Can be wider or taller
    MAX_ASPECT_RATIO = 3.0  # But within reasonable book proportions
    MIN_AREA_RATIO = 0.005  # Larger minimum area than spines
    MAX_AREA_RATIO = 0.9    # Not the entire image
    
    # Typical book cover aspect ratios
    BOOK_ASPECT_RATIOS = [
        (1.0, 1.5),   # Mass market paperback
        (1.0, 1.33),  # Trade paperback
        (1.0, 1.6),   # Hardcover
        (1.0, 1.29),  # A4-ish
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: Optional[str] = None
    ):
        """
        Initialize the cover detector.
        
        Args:
            model_path: Path to trained YOLOv8 weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        if YOLO is None:
            raise ImportError("ultralytics required. Install with: pip install ultralytics")
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            logger.info(f"Loaded cover detector from {model_path}")
        else:
            # Use pretrained as fallback
            self.model = YOLO("yolov8n.pt")
            logger.warning(
                "No custom cover model found. Using base YOLOv8n. "
                "Train a custom model for best results."
            )
        
        self.model.to(self.device)
        logger.info(f"CoverDetector initialized on {self.device}")
    
    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        filter_covers: bool = True
    ) -> DetectionResult:
        """
        Detect book covers in an image.
        
        Args:
            image: BGR numpy array
            confidence: Override confidence threshold
            filter_covers: Apply cover-specific filtering
            
        Returns:
            DetectionResult with detected book covers
        """
        conf = confidence or self.confidence_threshold
        
        # Run YOLO inference
        results = self.model.predict(
            image,
            conf=conf,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                conf_score = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names.get(class_id, f"class_{class_id}")
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf_score,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)
        
        result = DetectionResult(
            detections=detections,
            image_shape=image.shape,
            inference_time_ms=results[0].speed.get('inference', 0) if results else 0
        )
        
        # Apply cover-specific filtering
        if filter_covers:
            result = self._filter_for_covers(result, image.shape)
            
        # Fallback: If no detections found, assume the whole image is a cover
        # This is useful for "single cover" mode or when the generic model fails
        if len(result.detections) == 0:
            h, w = image.shape[:2]
            fallback_det = Detection(
                bbox=(0, 0, w, h),
                confidence=0.5, # Artificial confidence
                class_id=73, # "book" in COCO
                class_name="book_fallback"
            )
            logger.info("No objects detected. Using full image as fallback cover.")
            result.detections.append(fallback_det)
        
        return result
    
    def _filter_for_covers(
        self,
        result: DetectionResult,
        image_shape: Tuple[int, ...]
    ) -> DetectionResult:
        """
        Apply cover-specific post-processing.
        
        Filters based on:
        - Aspect ratio suitable for book covers
        - Area constraints
        - Removes likely false positives
        """
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * self.MIN_AREA_RATIO
        max_area = image_area * self.MAX_AREA_RATIO
        
        filtered = []
        for det in result.detections:
            # Check aspect ratio
            aspect = det.aspect_ratio
            if not (self.MIN_ASPECT_RATIO <= aspect <= self.MAX_ASPECT_RATIO):
                # Also check inverted aspect ratio (width/height)
                inv_aspect = 1.0 / aspect if aspect > 0 else float('inf')
                if not (self.MIN_ASPECT_RATIO <= inv_aspect <= self.MAX_ASPECT_RATIO):
                    continue
            
            # Check area
            if not (min_area <= det.area <= max_area):
                continue
            
            filtered.append(det)
        
        # Score detections by how "book-like" they are
        scored = []
        for det in filtered:
            book_score = self._compute_book_likeness(det)
            scored.append((det, book_score))
        
        # Sort by book-likeness score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return DetectionResult(
            detections=[d for d, _ in scored],
            image_shape=result.image_shape,
            inference_time_ms=result.inference_time_ms
        )
    
    def _compute_book_likeness(self, det: Detection) -> float:
        """
        Score how likely this detection is a book cover.
        
        Based on:
        - Aspect ratio similarity to common book formats
        - Confidence score
        - Size (not too small or too large)
        """
        # Aspect ratio score
        aspect = det.height / det.width if det.width > 0 else 1.0
        
        # Find best matching book format
        best_match_score = 0.0
        for w_ratio, h_ratio in self.BOOK_ASPECT_RATIOS:
            expected_aspect = h_ratio / w_ratio
            
            # Score based on closeness to expected aspect ratio
            diff = abs(aspect - expected_aspect)
            inv_diff = abs((1.0/aspect if aspect > 0 else 0) - expected_aspect)
            
            score = 1.0 / (1.0 + min(diff, inv_diff))
            best_match_score = max(best_match_score, score)
        
        # Combine with confidence
        final_score = det.confidence * 0.5 + best_match_score * 0.5
        
        return final_score
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        confidence: Optional[float] = None
    ) -> List[DetectionResult]:
        """Batch detection for efficiency."""
        conf = confidence or self.confidence_threshold
        
        results = self.model.predict(
            images,
            conf=conf,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detection_results = []
        for i, res in enumerate(results):
            detections = []
            
            if res.boxes is not None:
                boxes = res.boxes
                
                for j in range(len(boxes)):
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf_score = float(boxes.conf[j].cpu().numpy())
                    class_id = int(boxes.cls[j].cpu().numpy())
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                    
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf_score,
                        class_id=class_id,
                        class_name=class_name
                    )
                    detections.append(detection)
            
            result = DetectionResult(
                detections=detections,
                image_shape=images[i].shape,
                inference_time_ms=res.speed.get('inference', 0)
            )
            
            result = self._filter_for_covers(result, images[i].shape)
            detection_results.append(result)
        
        return detection_results
    
    def visualize(
        self,
        image: np.ndarray,
        result: DetectionResult,
        show_confidence: bool = True,
        color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """Draw detections on image."""
        vis_image = image.copy()
        
        for det in result.detections:
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                2
            )
            
            if show_confidence:
                label = f"cover {det.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                cv2.rectangle(
                    vis_image,
                    (det.x1, det.y1 - label_size[1] - 5),
                    (det.x1 + label_size[0], det.y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    vis_image,
                    label,
                    (det.x1, det.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return vis_image


class CoverDetectorTrainer:
    """Training utilities for the cover detector."""
    
    # Cover-optimized anchors
    COVER_ANCHORS = [
        [40, 60], [60, 80], [80, 120],      # Small covers
        [100, 150], [140, 200], [180, 260],  # Medium covers
        [220, 320], [280, 400], [350, 500]   # Large covers
    ]
    
    def __init__(
        self,
        base_model: str = "yolov8n.pt",
        project: str = "cover_detector",
        name: str = "train"
    ):
        self.model = YOLO(base_model)
        self.project = project
        self.name = name
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 20,
        **kwargs
    ):
        """Train with cover-specific augmentations."""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=self.project,
            name=self.name,
            # Cover-specific augmentations (more aggressive)
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.5,
            degrees=15,  # More rotation for covers
            translate=0.1,
            scale=0.5,
            shear=5,
            perspective=0.001,
            flipud=0.0,
            fliplr=0.5,  # Can flip covers horizontally
            mosaic=0.7,
            **kwargs
        )
        
        return results
