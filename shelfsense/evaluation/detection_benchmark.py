"""Object detection evaluation metrics (mAP, Precision, Recall)."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box representation."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_name: str = "book"
    
    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class DetectionResult:
    """Single detection result."""
    image_id: str
    predictions: list[BoundingBox] = field(default_factory=list)
    ground_truth: list[BoundingBox] = field(default_factory=list)
    inference_time_ms: float = 0.0


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics."""
    map_50: float = 0.0  # mAP @ IoU=0.50
    map_75: float = 0.0  # mAP @ IoU=0.75
    map_50_95: float = 0.0  # mAP @ IoU=0.50:0.95
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    avg_inference_time_ms: float = 0.0
    total_images: int = 0
    total_ground_truth: int = 0
    total_predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    pr_curve: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "map_50": round(self.map_50, 4),
            "map_75": round(self.map_75, 4),
            "map_50_95": round(self.map_50_95, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "avg_inference_time_ms": round(self.avg_inference_time_ms, 2),
            "total_images": self.total_images,
            "total_ground_truth": self.total_ground_truth,
            "total_predictions": self.total_predictions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "per_class_metrics": self.per_class_metrics,
        }


class DetectionBenchmark:
    """
    Benchmark suite for book detection evaluation.
    """
    
    def __init__(
        self,
        iou_thresholds: list[float] | None = None,
        confidence_threshold: float = 0.5,
    ):
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.confidence_threshold = confidence_threshold
        self.results: list[DetectionResult] = []
    
    def add_result(self, result: DetectionResult) -> None:
        """Add a detection result for evaluation."""
        self.results.append(result)
    
    def load_dataset(
        self,
        images_dir: Path,
        annotations_file: Path,
        detector: Any | None = None,
    ) -> None:
        """
        Load evaluation dataset and optionally run detection.
        
        Annotations should be in COCO format:
        {
            "images": [{"id": 1, "file_name": "img1.jpg", ...}],
            "annotations": [{"image_id": 1, "bbox": [x, y, w, h], "category_id": 1}],
            "categories": [{"id": 1, "name": "spine"}, {"id": 2, "name": "cover"}]
        }
        """
        with open(annotations_file) as f:
            coco_data = json.load(f)
        
        # Build lookup maps
        images = {img["id"]: img for img in coco_data["images"]}
        categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        
        # Group annotations by image
        annotations_by_image: dict[int, list] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Process each image
        for img_id, img_info in images.items():
            image_path = images_dir / img_info["file_name"]
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Ground truth boxes
            gt_boxes = []
            for ann in annotations_by_image.get(img_id, []):
                x, y, w, h = ann["bbox"]
                gt_boxes.append(BoundingBox(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    class_name=categories.get(ann["category_id"], "book"),
                ))
            
            # Run detection if detector provided
            pred_boxes = []
            inference_time = 0.0
            if detector:
                import time
                image = Image.open(image_path)
                start = time.perf_counter()
                detections = detector.detect(image)
                inference_time = (time.perf_counter() - start) * 1000
                
                for det in detections:
                    if det.get("confidence", 1.0) >= self.confidence_threshold:
                        bbox = det.get("bbox", det.get("bounding_box", {}))
                        pred_boxes.append(BoundingBox(
                            x1=bbox.get("x1", bbox.get("x", 0)),
                            y1=bbox.get("y1", bbox.get("y", 0)),
                            x2=bbox.get("x2", bbox.get("x", 0) + bbox.get("width", 0)),
                            y2=bbox.get("y2", bbox.get("y", 0) + bbox.get("height", 0)),
                            confidence=det.get("confidence", 1.0),
                            class_name=det.get("class", det.get("type", "book")),
                        ))
            
            self.results.append(DetectionResult(
                image_id=str(img_id),
                predictions=pred_boxes,
                ground_truth=gt_boxes,
                inference_time_ms=inference_time,
            ))
    
    def evaluate(self) -> DetectionMetrics:
        """Run full evaluation and compute all metrics."""
        if not self.results:
            logger.warning("No results to evaluate")
            return DetectionMetrics()
        
        metrics = DetectionMetrics()
        metrics.total_images = len(self.results)
        
        # Collect all predictions and ground truth
        all_predictions: list[tuple[BoundingBox, str]] = []  # (box, image_id)
        all_ground_truth: dict[str, list[BoundingBox]] = {}
        
        for result in self.results:
            all_ground_truth[result.image_id] = result.ground_truth
            metrics.total_ground_truth += len(result.ground_truth)
            
            for pred in result.predictions:
                all_predictions.append((pred, result.image_id))
                metrics.total_predictions += 1
        
        # Sort predictions by confidence (descending)
        all_predictions.sort(key=lambda x: x[0].confidence, reverse=True)
        
        # Compute mAP at different IoU thresholds
        ap_per_threshold = []
        for iou_thresh in self.iou_thresholds:
            ap, pr_data = self._compute_ap(all_predictions, all_ground_truth, iou_thresh)
            ap_per_threshold.append(ap)
            if iou_thresh == 0.5:
                metrics.map_50 = ap
                metrics.pr_curve["iou_0.50"] = pr_data
            elif iou_thresh == 0.75:
                metrics.map_75 = ap
        
        metrics.map_50_95 = np.mean(ap_per_threshold) if ap_per_threshold else 0.0
        
        # Compute precision/recall at default threshold (IoU=0.5)
        tp, fp, fn = self._match_predictions(all_predictions, all_ground_truth, 0.5)
        metrics.true_positives = tp
        metrics.false_positives = fp
        metrics.false_negatives = fn
        
        if tp + fp > 0:
            metrics.precision = tp / (tp + fp)
        if tp + fn > 0:
            metrics.recall = tp / (tp + fn)
        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        
        # Compute per-class metrics
        metrics.per_class_metrics = self._compute_per_class_metrics(all_predictions, all_ground_truth)
        
        # Average inference time
        times = [r.inference_time_ms for r in self.results if r.inference_time_ms > 0]
        if times:
            metrics.avg_inference_time_ms = np.mean(times)
        
        return metrics
    
    def _compute_ap(
        self,
        predictions: list[tuple[BoundingBox, str]],
        ground_truth: dict[str, list[BoundingBox]],
        iou_threshold: float,
    ) -> tuple[float, list[tuple[float, float]]]:
        """Compute Average Precision at given IoU threshold."""
        # Track which GT boxes have been matched
        gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in ground_truth.items()}
        
        total_gt = sum(len(boxes) for boxes in ground_truth.values())
        if total_gt == 0:
            return 0.0, []
        
        # Compute TP/FP for each prediction
        tp_list = []
        fp_list = []
        
        for pred_box, img_id in predictions:
            gt_boxes = ground_truth.get(img_id, [])
            matched_gt_idx = -1
            best_iou = iou_threshold
            
            # Find best matching GT box
            for idx, gt_box in enumerate(gt_boxes):
                if gt_matched[img_id][idx]:
                    continue
                if pred_box.class_name != gt_box.class_name:
                    continue
                    
                iou = pred_box.iou(gt_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_idx = idx
            
            if matched_gt_idx >= 0:
                gt_matched[img_id][matched_gt_idx] = True
                tp_list.append(1)
                fp_list.append(0)
            else:
                tp_list.append(0)
                fp_list.append(1)
        
        # Compute cumulative TP/FP
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        
        # Compute precision/recall at each point
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / total_gt
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        pr_curve = []
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if mask.any():
                p = precisions[mask].max()
            else:
                p = 0
            ap += p / 11
            pr_curve.append((float(t), float(p)))
        
        return ap, pr_curve
    
    def _match_predictions(
        self,
        predictions: list[tuple[BoundingBox, str]],
        ground_truth: dict[str, list[BoundingBox]],
        iou_threshold: float,
    ) -> tuple[int, int, int]:
        """Match predictions to ground truth and return TP, FP, FN counts."""
        gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in ground_truth.items()}
        
        tp = 0
        fp = 0
        
        for pred_box, img_id in predictions:
            gt_boxes = ground_truth.get(img_id, [])
            matched = False
            
            for idx, gt_box in enumerate(gt_boxes):
                if gt_matched[img_id][idx]:
                    continue
                    
                iou = pred_box.iou(gt_box)
                if iou >= iou_threshold:
                    gt_matched[img_id][idx] = True
                    matched = True
                    break
            
            if matched:
                tp += 1
            else:
                fp += 1
        
        fn = sum(
            sum(1 for m in matches if not m)
            for matches in gt_matched.values()
        )
        
        return tp, fp, fn
    
    def _compute_per_class_metrics(
        self,
        predictions: list[tuple[BoundingBox, str]],
        ground_truth: dict[str, list[BoundingBox]],
    ) -> dict[str, dict[str, float]]:
        """Compute metrics broken down by class."""
        # Collect classes
        classes = set()
        for boxes in ground_truth.values():
            for box in boxes:
                classes.add(box.class_name)
        for pred, _ in predictions:
            classes.add(pred.class_name)
        
        per_class = {}
        
        for cls in classes:
            # Filter predictions and GT for this class
            cls_predictions = [(p, img) for p, img in predictions if p.class_name == cls]
            cls_gt = {
                img_id: [b for b in boxes if b.class_name == cls]
                for img_id, boxes in ground_truth.items()
            }
            
            if not cls_predictions and all(len(boxes) == 0 for boxes in cls_gt.values()):
                continue
            
            ap, _ = self._compute_ap(cls_predictions, cls_gt, 0.5)
            tp, fp, fn = self._match_predictions(cls_predictions, cls_gt, 0.5)
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            per_class[cls] = {
                "ap_50": round(ap, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "count_gt": sum(len(boxes) for boxes in cls_gt.values()),
                "count_pred": len(cls_predictions),
            }
        
        return per_class
    
    def generate_report(self, output_path: Path | None = None) -> str:
        """Generate human-readable evaluation report."""
        metrics = self.evaluate()
        
        lines = [
            "=" * 60,
            "DETECTION BENCHMARK REPORT",
            "=" * 60,
            "",
            "Overall Metrics:",
            f"  mAP@50:     {metrics.map_50:.4f}",
            f"  mAP@75:     {metrics.map_75:.4f}",
            f"  mAP@50:95:  {metrics.map_50_95:.4f}",
            "",
            f"  Precision:  {metrics.precision:.4f}",
            f"  Recall:     {metrics.recall:.4f}",
            f"  F1 Score:   {metrics.f1_score:.4f}",
            "",
            "Counts:",
            f"  Total Images:        {metrics.total_images}",
            f"  Ground Truth Boxes:  {metrics.total_ground_truth}",
            f"  Predicted Boxes:     {metrics.total_predictions}",
            f"  True Positives:      {metrics.true_positives}",
            f"  False Positives:     {metrics.false_positives}",
            f"  False Negatives:     {metrics.false_negatives}",
            "",
            f"Avg Inference Time: {metrics.avg_inference_time_ms:.2f} ms",
            "",
        ]
        
        if metrics.per_class_metrics:
            lines.extend([
                "Per-Class Metrics:",
                "-" * 40,
            ])
            for cls, cls_metrics in metrics.per_class_metrics.items():
                lines.extend([
                    f"  {cls}:",
                    f"    AP@50:     {cls_metrics['ap_50']:.4f}",
                    f"    Precision: {cls_metrics['precision']:.4f}",
                    f"    Recall:    {cls_metrics['recall']:.4f}",
                    f"    F1:        {cls_metrics['f1_score']:.4f}",
                    f"    GT/Pred:   {cls_metrics['count_gt']}/{cls_metrics['count_pred']}",
                ])
        
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report)
            # Also save JSON metrics
            json_path = output_path.with_suffix(".json")
            json_path.write_text(json.dumps(metrics.to_dict(), indent=2))
            logger.info(f"Report saved to {output_path}")
        
        return report


def create_synthetic_benchmark(
    num_images: int = 100,
    boxes_per_image: tuple[int, int] = (3, 8),
    noise_level: float = 0.1,
) -> DetectionBenchmark:
    """
    Create a synthetic benchmark for testing the evaluation pipeline.
    
    Generates fake ground truth and predictions with controlled noise.
    """
    import random
    
    benchmark = DetectionBenchmark()
    
    classes = ["spine", "cover"]
    
    for i in range(num_images):
        num_boxes = random.randint(boxes_per_image[0], boxes_per_image[1])
        
        gt_boxes = []
        pred_boxes = []
        
        for j in range(num_boxes):
            # Random ground truth box
            x1 = random.uniform(0, 800)
            y1 = random.uniform(0, 600)
            w = random.uniform(50, 200)
            h = random.uniform(100, 400)
            cls = random.choice(classes)
            
            gt_boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x1 + w, y2=y1 + h,
                class_name=cls,
            ))
            
            # Prediction with noise
            if random.random() > 0.1:  # 90% detection rate
                noise_x = random.gauss(0, w * noise_level)
                noise_y = random.gauss(0, h * noise_level)
                noise_w = random.gauss(0, w * noise_level)
                noise_h = random.gauss(0, h * noise_level)
                
                pred_boxes.append(BoundingBox(
                    x1=x1 + noise_x,
                    y1=y1 + noise_y,
                    x2=x1 + w + noise_w,
                    y2=y1 + h + noise_h,
                    confidence=random.uniform(0.5, 1.0),
                    class_name=cls,
                ))
        
        # Add some false positives
        for _ in range(random.randint(0, 2)):
            pred_boxes.append(BoundingBox(
                x1=random.uniform(0, 800),
                y1=random.uniform(0, 600),
                x2=random.uniform(50, 200),
                y2=random.uniform(100, 400),
                confidence=random.uniform(0.3, 0.7),
                class_name=random.choice(classes),
            ))
        
        benchmark.add_result(DetectionResult(
            image_id=f"img_{i:04d}",
            predictions=pred_boxes,
            ground_truth=gt_boxes,
            inference_time_ms=random.uniform(20, 100),
        ))
    
    return benchmark
