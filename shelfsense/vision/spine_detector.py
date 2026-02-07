"""
Book Spine Detector

YOLOv8-based detection model for identifying book spines in images.
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
    logger.warning("ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None


@dataclass
class Detection:
    """A single detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def x1(self) -> int:
        return self.bbox[0]
    
    @property
    def y1(self) -> int:
        return self.bbox[1]
    
    @property
    def x2(self) -> int:
        return self.bbox[2]
    
    @property
    def y2(self) -> int:
        return self.bbox[3]
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def aspect_ratio(self) -> float:
        """Height / Width ratio. Spines typically have high aspect ratio."""
        if self.width == 0:
            return float('inf')
        return self.height / self.width
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "area": self.area,
            "aspect_ratio": self.aspect_ratio
        }


@dataclass
class DetectionResult:
    """Result of detection on a single image."""
    detections: List[Detection] = field(default_factory=list)
    image_shape: Tuple[int, int, int] = (0, 0, 0)
    inference_time_ms: float = 0.0
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def filter_by_confidence(self, threshold: float) -> 'DetectionResult':
        """Return new result with only high-confidence detections."""
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionResult(
            detections=filtered,
            image_shape=self.image_shape,
            inference_time_ms=self.inference_time_ms
        )
    
    def filter_by_aspect_ratio(self, min_ratio: float, max_ratio: float) -> 'DetectionResult':
        """Filter detections by aspect ratio."""
        filtered = [
            d for d in self.detections
            if min_ratio <= d.aspect_ratio <= max_ratio
        ]
        return DetectionResult(
            detections=filtered,
            image_shape=self.image_shape,
            inference_time_ms=self.inference_time_ms
        )
    
    def sort_by_position(self, by: str = "x") -> 'DetectionResult':
        """Sort detections by position (x for left-to-right, y for top-to-bottom)."""
        key_func = (lambda d: d.x1) if by == "x" else (lambda d: d.y1)
        sorted_dets = sorted(self.detections, key=key_func)
        return DetectionResult(
            detections=sorted_dets,
            image_shape=self.image_shape,
            inference_time_ms=self.inference_time_ms
        )


class SpineDetector:
    """
    YOLOv8-based book spine detector.
    """
    
    # Spine-specific parameters
    DEFAULT_CONFIDENCE = 0.25
    DEFAULT_IOU_THRESHOLD = 0.45
    MIN_ASPECT_RATIO = 0.1  # Allow horizontal spines (flat books)

    MAX_ASPECT_RATIO = 20.0  # But not infinitely thin
    MIN_AREA_RATIO = 0.001  # Minimum area relative to image
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: Optional[str] = None
    ):
        """
        Initialize the spine detector.
        
        Args:
            model_path: Path to trained YOLOv8 weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference ('cuda', 'cpu', or auto)
        """
        if YOLO is None:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            logger.info(f"Loaded spine detector from {model_path}")
        else:
            # Use pretrained YOLOv8n as base (for demonstration)
            # In production, this would be a fine-tuned model
            self.model = YOLO("yolov8n.pt")
            logger.warning(
                "No custom spine model found. Using base YOLOv8n. "
                "For best results, train a custom model on book spine data."
            )
        
        self.model.to(self.device)
        logger.info(f"SpineDetector initialized on {self.device}")
    
    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        filter_spines: bool = True
    ) -> DetectionResult:
        """
        Detect book spines in an image.
        
        Args:
            image: BGR numpy array
            confidence: Override confidence threshold
            filter_spines: Apply aspect ratio filtering for spines
            
        Returns:
            DetectionResult with detected book spines
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
                # Get bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Get confidence and class
                conf_score = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
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
        
        # Apply spine-specific filtering
        if filter_spines:
            result = self._filter_for_spines(result, image.shape)
        
        return result
    
    def _filter_for_spines(
        self,
        result: DetectionResult,
        image_shape: Tuple[int, ...]
    ) -> DetectionResult:
        """
        Apply spine-specific post-processing.
        
        Filters out:
        - Detections with wrong aspect ratio
        - Very small detections
        - Overlapping detections (keep highest confidence)
        """
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * self.MIN_AREA_RATIO
        
        filtered = []
        for det in result.detections:
            # Check aspect ratio (height/width should be > 1.5 for spines)
            if not (self.MIN_ASPECT_RATIO <= det.aspect_ratio <= self.MAX_ASPECT_RATIO):
                continue
            
            # Check minimum area
            if det.area < min_area:
                continue
            
            filtered.append(det)
        
        # Sort by x-position (left to right) for natural reading order
        filtered.sort(key=lambda d: d.x1)
        
        return DetectionResult(
            detections=filtered,
            image_shape=result.image_shape,
            inference_time_ms=result.inference_time_ms
        )
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        confidence: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        Batch detection for multiple images.
        
        More efficient than calling detect() multiple times.
        """
        conf = confidence or self.confidence_threshold
        
        # Run batch inference
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
            
            result = self._filter_for_spines(result, images[i].shape)
            detection_results.append(result)
        
        return detection_results
    
    def visualize(
        self,
        image: np.ndarray,
        result: DetectionResult,
        show_confidence: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw detections on image for visualization.
        
        Args:
            image: BGR numpy array
            result: Detection result
            show_confidence: Show confidence scores
            color: BGR color for bounding boxes
            
        Returns:
            Image with drawn detections
        """
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
            
            # Draw label
            if show_confidence:
                label = f"spine {det.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Background for text
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
        
        return vis_image


class SpineDetectorTrainer:
    """
    Training utilities for the spine detector.
    
    Wraps YOLOv8 training with spine-specific configurations.
    """
    
    # Spine-optimized anchors (tall and narrow)
    SPINE_ANCHORS = [
        [5, 20], [8, 35], [12, 50],      # Small spines
        [15, 70], [20, 100], [25, 140],  # Medium spines
        [30, 180], [40, 220], [50, 280]  # Large spines
    ]
    
    def __init__(
        self,
        base_model: str = "yolov8n.pt",
        project: str = "spine_detector",
        name: str = "train"
    ):
        """
        Initialize trainer.
        
        Args:
            base_model: Base YOLO model to fine-tune
            project: Project name for saving runs
            name: Run name
        """
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
        """
        Train the spine detector.
        
        Args:
            data_yaml: Path to data configuration YAML
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            patience: Early stopping patience
            **kwargs: Additional training arguments
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=self.project,
            name=self.name,
            # Spine-specific augmentations
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5,  # Limited rotation for spines
            translate=0.1,
            scale=0.5,
            shear=2,
            perspective=0.0001,
            flipud=0.0,  # Don't flip vertically
            fliplr=0.0,  # Don't flip horizontally (text direction matters)
            mosaic=0.5,  # Less mosaic for spine detection
            **kwargs
        )
        
        return results
    
    def export(self, format: str = "onnx") -> str:
        """Export trained model to specified format."""
        return self.model.export(format=format)
