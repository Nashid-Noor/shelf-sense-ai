"""
Evaluation Metrics

Comprehensive metrics for evaluating object detection, OCR accuracy, and book identification.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import Levenshtein
import re


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoundingBox:
    """Bounding box representation."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = 0
    
    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    """Single detection evaluation result."""
    predicted: List[BoundingBox]
    ground_truth: List[BoundingBox]
    image_id: str = ""


@dataclass
class OCRResult:
    """OCR evaluation result."""
    predicted: str
    ground_truth: str
    book_id: str = ""


@dataclass
class IdentificationResult:
    """Book identification evaluation result."""
    predicted_ids: List[str]  # Ranked list of predicted book IDs
    ground_truth_id: str
    query_id: str = ""
    scores: List[float] = field(default_factory=list)


@dataclass
class MetricsSummary:
    """Summary of evaluation metrics."""
    detection_map: float = 0.0
    detection_map_50: float = 0.0
    detection_map_75: float = 0.0
    ocr_cer: float = 0.0
    ocr_wer: float = 0.0
    ocr_accuracy: float = 0.0
    identification_p_at_1: float = 0.0
    identification_p_at_5: float = 0.0
    identification_mrr: float = 0.0
    identification_ndcg: float = 0.0
    e2e_accuracy: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0


# =============================================================================
# Detection Metrics
# =============================================================================

class DetectionMetrics:
    """
    Object detection evaluation metrics.
    
    Implements:
    - IoU (Intersection over Union)
    - Precision/Recall at various IoU thresholds
    - mAP (mean Average Precision)
    - AP50, AP75 (AP at specific IoU thresholds)
    """
    
    @staticmethod
    def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: First bounding box.
            box2: Second bounding box.
        
        Returns:
            IoU score between 0 and 1.
        """
        # Calculate intersection
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union
        union = box1.area + box2.area - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def match_detections(
        predictions: List[BoundingBox],
        ground_truth: List[BoundingBox],
        iou_threshold: float = 0.5,
    ) -> Tuple[List[bool], List[bool]]:
        """
        Match predictions to ground truth boxes.
        
        Uses greedy matching: highest confidence predictions matched first.
        
        Args:
            predictions: List of predicted bounding boxes.
            ground_truth: List of ground truth bounding boxes.
            iou_threshold: Minimum IoU for a valid match.
        
        Returns:
            Tuple of (prediction_matched, gt_matched) boolean lists.
        """
        # Sort predictions by confidence (highest first)
        sorted_preds = sorted(
            enumerate(predictions),
            key=lambda x: x[1].confidence,
            reverse=True,
        )
        
        pred_matched = [False] * len(predictions)
        gt_matched = [False] * len(ground_truth)
        
        for pred_idx, pred_box in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                # Check class match if classes are specified
                if pred_box.class_id != gt_box.class_id:
                    continue
                
                iou = DetectionMetrics.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                pred_matched[pred_idx] = True
                gt_matched[best_gt_idx] = True
        
        return pred_matched, gt_matched
    
    @staticmethod
    def precision_recall_curve(
        results: List[DetectionResult],
        iou_threshold: float = 0.5,
        num_points: int = 101,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve across all images.
        
        Args:
            results: List of detection results per image.
            iou_threshold: IoU threshold for matching.
            num_points: Number of recall points to evaluate.
        
        Returns:
            Tuple of (precisions, recalls) arrays.
        """
        # Collect all predictions with their match status
        all_predictions = []
        total_gt = 0
        
        for result in results:
            pred_matched, _ = DetectionMetrics.match_detections(
                result.predicted,
                result.ground_truth,
                iou_threshold,
            )
            
            for i, pred in enumerate(result.predicted):
                all_predictions.append({
                    "confidence": pred.confidence,
                    "is_tp": pred_matched[i],
                })
            
            total_gt += len(result.ground_truth)
        
        if total_gt == 0:
            return np.array([1.0]), np.array([0.0])
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Compute precision/recall at each point
        tp_cumsum = 0
        precisions = []
        recalls = []
        
        for i, pred in enumerate(all_predictions):
            if pred["is_tp"]:
                tp_cumsum += 1
            
            precision = tp_cumsum / (i + 1)
            recall = tp_cumsum / total_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Interpolate to fixed recall points
        recall_points = np.linspace(0, 1, num_points)
        interpolated_precisions = np.zeros(num_points)
        
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        for i, r in enumerate(recall_points):
            mask = recalls >= r
            if mask.any():
                interpolated_precisions[i] = precisions[mask].max()
        
        return interpolated_precisions, recall_points
    
    @staticmethod
    def calculate_ap(
        results: List[DetectionResult],
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Calculate Average Precision at a specific IoU threshold.
        
        Uses 101-point interpolation (COCO-style).
        
        Args:
            results: List of detection results.
            iou_threshold: IoU threshold for matching.
        
        Returns:
            AP score between 0 and 1.
        """
        precisions, _ = DetectionMetrics.precision_recall_curve(
            results, iou_threshold
        )
        return precisions.mean()
    
    @staticmethod
    def calculate_map(
        results: List[DetectionResult],
        iou_thresholds: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate mean Average Precision across IoU thresholds.
        
        Args:
            results: List of detection results.
            iou_thresholds: List of IoU thresholds. Default: COCO thresholds.
        
        Returns:
            Dictionary with mAP, AP50, AP75.
        """
        if iou_thresholds is None:
            # COCO-style: 0.5 to 0.95 in 0.05 steps
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        aps = []
        ap_by_threshold = {}
        
        for thresh in iou_thresholds:
            ap = DetectionMetrics.calculate_ap(results, thresh)
            aps.append(ap)
            ap_by_threshold[f"AP{int(thresh * 100)}"] = ap
        
        return {
            "mAP": np.mean(aps),
            "AP50": ap_by_threshold.get("AP50", 0.0),
            "AP75": ap_by_threshold.get("AP75", 0.0),
            **ap_by_threshold,
        }


# =============================================================================
# OCR Metrics
# =============================================================================

class OCRMetrics:
    """
    OCR evaluation metrics.
    
    Implements:
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - Exact match accuracy
    - Normalized edit distance
    """
    
    @staticmethod
    def normalize_text(text: str, lowercase: bool = True) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Input text.
            lowercase: Whether to convert to lowercase.
        
        Returns:
            Normalized text.
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Optionally lowercase
        if lowercase:
            text = text.lower()
        
        return text
    
    @staticmethod
    def character_error_rate(
        predicted: str,
        ground_truth: str,
        normalize: bool = True,
    ) -> float:
        """
        Calculate Character Error Rate (CER).
        
        CER = (insertions + deletions + substitutions) / len(ground_truth)
        
        Args:
            predicted: Predicted text.
            ground_truth: Ground truth text.
            normalize: Whether to normalize texts first.
        
        Returns:
            CER score (0 = perfect, higher = worse).
        """
        if normalize:
            predicted = OCRMetrics.normalize_text(predicted)
            ground_truth = OCRMetrics.normalize_text(ground_truth)
        
        if len(ground_truth) == 0:
            return 0.0 if len(predicted) == 0 else 1.0
        
        distance = Levenshtein.distance(predicted, ground_truth)
        return distance / len(ground_truth)
    
    @staticmethod
    def word_error_rate(
        predicted: str,
        ground_truth: str,
        normalize: bool = True,
    ) -> float:
        """
        Calculate Word Error Rate (WER).
        
        WER = (insertions + deletions + substitutions) / num_words(ground_truth)
        
        Args:
            predicted: Predicted text.
            ground_truth: Ground truth text.
            normalize: Whether to normalize texts first.
        
        Returns:
            WER score (0 = perfect, higher = worse).
        """
        if normalize:
            predicted = OCRMetrics.normalize_text(predicted)
            ground_truth = OCRMetrics.normalize_text(ground_truth)
        
        pred_words = predicted.split()
        gt_words = ground_truth.split()
        
        if len(gt_words) == 0:
            return 0.0 if len(pred_words) == 0 else 1.0
        
        # Use Levenshtein on word sequences
        distance = Levenshtein.distance(
            " ".join(pred_words),
            " ".join(gt_words),
        )
        
        # Approximate word-level distance
        # For more accurate WER, use dynamic programming on word sequences
        return min(1.0, distance / max(len(gt_words), len(pred_words)))
    
    @staticmethod
    def exact_match(
        predicted: str,
        ground_truth: str,
        normalize: bool = True,
    ) -> bool:
        """
        Check for exact match after normalization.
        
        Args:
            predicted: Predicted text.
            ground_truth: Ground truth text.
            normalize: Whether to normalize texts first.
        
        Returns:
            True if texts match exactly.
        """
        if normalize:
            predicted = OCRMetrics.normalize_text(predicted)
            ground_truth = OCRMetrics.normalize_text(ground_truth)
        
        return predicted == ground_truth
    
    @staticmethod
    def evaluate_batch(
        results: List[OCRResult],
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate OCR on a batch of results.
        
        Args:
            results: List of OCR results.
            normalize: Whether to normalize texts.
        
        Returns:
            Dictionary with CER, WER, and accuracy metrics.
        """
        if not results:
            return {
                "cer": 0.0,
                "wer": 0.0,
                "exact_match_accuracy": 0.0,
                "total_samples": 0,
            }
        
        cers = []
        wers = []
        exact_matches = 0
        
        for result in results:
            cer = OCRMetrics.character_error_rate(
                result.predicted, result.ground_truth, normalize
            )
            wer = OCRMetrics.word_error_rate(
                result.predicted, result.ground_truth, normalize
            )
            is_match = OCRMetrics.exact_match(
                result.predicted, result.ground_truth, normalize
            )
            
            cers.append(cer)
            wers.append(wer)
            if is_match:
                exact_matches += 1
        
        return {
            "cer": np.mean(cers),
            "cer_std": np.std(cers),
            "wer": np.mean(wers),
            "wer_std": np.std(wers),
            "exact_match_accuracy": exact_matches / len(results),
            "total_samples": len(results),
        }


# =============================================================================
# Identification Metrics
# =============================================================================

class IdentificationMetrics:
    """
    Book identification/retrieval metrics.
    
    Implements:
    - Precision@k (P@k)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG)
    - Hit Rate
    """
    
    @staticmethod
    def precision_at_k(
        predicted_ids: List[str],
        ground_truth_id: str,
        k: int,
    ) -> float:
        """
        Calculate Precision@k for single query.
        
        For identification (single correct answer), P@k is 1/k if correct
        answer is in top k, else 0.
        
        Args:
            predicted_ids: Ranked list of predicted IDs.
            ground_truth_id: Correct ID.
            k: Number of top results to consider.
        
        Returns:
            P@k score.
        """
        top_k = predicted_ids[:k]
        
        if ground_truth_id in top_k:
            return 1.0 / k
        return 0.0
    
    @staticmethod
    def hit_at_k(
        predicted_ids: List[str],
        ground_truth_id: str,
        k: int,
    ) -> bool:
        """
        Check if correct answer is in top k results.
        
        Args:
            predicted_ids: Ranked list of predicted IDs.
            ground_truth_id: Correct ID.
            k: Number of top results to consider.
        
        Returns:
            True if ground truth is in top k.
        """
        return ground_truth_id in predicted_ids[:k]
    
    @staticmethod
    def reciprocal_rank(
        predicted_ids: List[str],
        ground_truth_id: str,
    ) -> float:
        """
        Calculate reciprocal rank for single query.
        
        RR = 1/rank of correct answer, or 0 if not found.
        
        Args:
            predicted_ids: Ranked list of predicted IDs.
            ground_truth_id: Correct ID.
        
        Returns:
            Reciprocal rank score.
        """
        try:
            rank = predicted_ids.index(ground_truth_id) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0
    
    @staticmethod
    def dcg_at_k(
        predicted_ids: List[str],
        ground_truth_id: str,
        k: int,
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at k.
        
        For identification, relevance is binary (1 if correct, 0 otherwise).
        
        Args:
            predicted_ids: Ranked list of predicted IDs.
            ground_truth_id: Correct ID.
            k: Number of results to consider.
        
        Returns:
            DCG@k score.
        """
        dcg = 0.0
        
        for i, pred_id in enumerate(predicted_ids[:k]):
            rel = 1.0 if pred_id == ground_truth_id else 0.0
            # DCG formula: rel / log2(rank + 1)
            dcg += rel / np.log2(i + 2)  # +2 because rank is 1-indexed
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(
        predicted_ids: List[str],
        ground_truth_id: str,
        k: int,
    ) -> float:
        """
        Calculate Normalized DCG at k.
        
        NDCG = DCG / IDCG (ideal DCG).
        
        Args:
            predicted_ids: Ranked list of predicted IDs.
            ground_truth_id: Correct ID.
            k: Number of results to consider.
        
        Returns:
            NDCG@k score between 0 and 1.
        """
        dcg = IdentificationMetrics.dcg_at_k(predicted_ids, ground_truth_id, k)
        
        # Ideal DCG: correct answer at position 1
        idcg = 1.0 / np.log2(2)  # rel=1 at rank 1
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def evaluate_batch(
        results: List[IdentificationResult],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, float]:
        """
        Evaluate identification on a batch of results.
        
        Args:
            results: List of identification results.
            k_values: List of k values for P@k and NDCG@k.
        
        Returns:
            Dictionary with all metrics.
        """
        if not results:
            return {
                "mrr": 0.0,
                "total_samples": 0,
            }
        
        mrr_scores = []
        hit_rates = {k: [] for k in k_values}
        ndcg_scores = {k: [] for k in k_values}
        
        for result in results:
            # MRR
            rr = IdentificationMetrics.reciprocal_rank(
                result.predicted_ids, result.ground_truth_id
            )
            mrr_scores.append(rr)
            
            # Hit@k and NDCG@k
            for k in k_values:
                hit = IdentificationMetrics.hit_at_k(
                    result.predicted_ids, result.ground_truth_id, k
                )
                ndcg = IdentificationMetrics.ndcg_at_k(
                    result.predicted_ids, result.ground_truth_id, k
                )
                
                hit_rates[k].append(1.0 if hit else 0.0)
                ndcg_scores[k].append(ndcg)
        
        metrics = {
            "mrr": np.mean(mrr_scores),
            "mrr_std": np.std(mrr_scores),
            "total_samples": len(results),
        }
        
        for k in k_values:
            metrics[f"hit_rate_at_{k}"] = np.mean(hit_rates[k])
            metrics[f"ndcg_at_{k}"] = np.mean(ndcg_scores[k])
        
        return metrics


# =============================================================================
# Latency Metrics
# =============================================================================

class LatencyMetrics:
    """
    Latency evaluation metrics.
    
    Tracks timing across pipeline stages.
    """
    
    def __init__(self):
        """Initialize latency tracker."""
        self.measurements: Dict[str, List[float]] = defaultdict(list)
    
    def record(self, stage: str, latency_ms: float) -> None:
        """
        Record a latency measurement.
        
        Args:
            stage: Pipeline stage name.
            latency_ms: Latency in milliseconds.
        """
        self.measurements[stage].append(latency_ms)
    
    def get_percentiles(
        self,
        stage: str,
        percentiles: List[int] = [50, 90, 95, 99],
    ) -> Dict[str, float]:
        """
        Get latency percentiles for a stage.
        
        Args:
            stage: Pipeline stage name.
            percentiles: Percentiles to calculate.
        
        Returns:
            Dictionary with percentile values.
        """
        if stage not in self.measurements:
            return {}
        
        data = np.array(self.measurements[stage])
        
        result = {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "count": len(data),
        }
        
        for p in percentiles:
            result[f"p{p}"] = np.percentile(data, p)
        
        return result
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of all stages.
        
        Returns:
            Dictionary mapping stage names to their percentile stats.
        """
        return {
            stage: self.get_percentiles(stage)
            for stage in self.measurements
        }
    
    def get_e2e_latency(self) -> Dict[str, float]:
        """
        Get end-to-end latency (sum of all stages).
        
        Returns:
            Percentile stats for total pipeline latency.
        """
        if not self.measurements:
            return {}
        
        # Find the stage that represents total time, or sum all stages
        if "total" in self.measurements:
            return self.get_percentiles("total")
        
        # Sum all stage latencies per sample (assumes aligned measurements)
        min_samples = min(len(v) for v in self.measurements.values())
        
        totals = []
        for i in range(min_samples):
            total = sum(
                self.measurements[stage][i]
                for stage in self.measurements
            )
            totals.append(total)
        
        data = np.array(totals)
        
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "p50": np.percentile(data, 50),
            "p90": np.percentile(data, 90),
            "p95": np.percentile(data, 95),
            "p99": np.percentile(data, 99),
            "count": len(data),
        }


# =============================================================================
# Aggregate Evaluator
# =============================================================================

class PipelineEvaluator:
    """
    Comprehensive pipeline evaluation.
    
    Aggregates all metrics into a single evaluation report.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.detection_results: List[DetectionResult] = []
        self.ocr_results: List[OCRResult] = []
        self.identification_results: List[IdentificationResult] = []
        self.latency_metrics = LatencyMetrics()
    
    def add_detection_result(
        self,
        predicted: List[BoundingBox],
        ground_truth: List[BoundingBox],
        image_id: str = "",
    ) -> None:
        """Add a detection evaluation sample."""
        self.detection_results.append(DetectionResult(
            predicted=predicted,
            ground_truth=ground_truth,
            image_id=image_id,
        ))
    
    def add_ocr_result(
        self,
        predicted: str,
        ground_truth: str,
        book_id: str = "",
    ) -> None:
        """Add an OCR evaluation sample."""
        self.ocr_results.append(OCRResult(
            predicted=predicted,
            ground_truth=ground_truth,
            book_id=book_id,
        ))
    
    def add_identification_result(
        self,
        predicted_ids: List[str],
        ground_truth_id: str,
        query_id: str = "",
        scores: List[float] = None,
    ) -> None:
        """Add an identification evaluation sample."""
        self.identification_results.append(IdentificationResult(
            predicted_ids=predicted_ids,
            ground_truth_id=ground_truth_id,
            query_id=query_id,
            scores=scores or [],
        ))
    
    def record_latency(self, stage: str, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.latency_metrics.record(stage, latency_ms)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run full evaluation and return comprehensive report.
        
        Returns:
            Dictionary with all metrics by category.
        """
        report = {
            "detection": {},
            "ocr": {},
            "identification": {},
            "latency": {},
            "summary": {},
        }
        
        # Detection metrics
        if self.detection_results:
            detection_metrics = DetectionMetrics.calculate_map(
                self.detection_results
            )
            report["detection"] = {
                **detection_metrics,
                "num_images": len(self.detection_results),
                "total_predictions": sum(
                    len(r.predicted) for r in self.detection_results
                ),
                "total_ground_truth": sum(
                    len(r.ground_truth) for r in self.detection_results
                ),
            }
        
        # OCR metrics
        if self.ocr_results:
            report["ocr"] = OCRMetrics.evaluate_batch(self.ocr_results)
        
        # Identification metrics
        if self.identification_results:
            report["identification"] = IdentificationMetrics.evaluate_batch(
                self.identification_results
            )
        
        # Latency metrics
        report["latency"] = {
            "by_stage": self.latency_metrics.get_summary(),
            "e2e": self.latency_metrics.get_e2e_latency(),
        }
        
        # Summary
        report["summary"] = MetricsSummary(
            detection_map=report["detection"].get("mAP", 0.0),
            detection_map_50=report["detection"].get("AP50", 0.0),
            detection_map_75=report["detection"].get("AP75", 0.0),
            ocr_cer=report["ocr"].get("cer", 0.0),
            ocr_wer=report["ocr"].get("wer", 0.0),
            ocr_accuracy=report["ocr"].get("exact_match_accuracy", 0.0),
            identification_p_at_1=report["identification"].get("hit_rate_at_1", 0.0),
            identification_p_at_5=report["identification"].get("hit_rate_at_5", 0.0),
            identification_mrr=report["identification"].get("mrr", 0.0),
            identification_ndcg=report["identification"].get("ndcg_at_10", 0.0),
            latency_p50_ms=report["latency"].get("e2e", {}).get("p50", 0.0),
            latency_p95_ms=report["latency"].get("e2e", {}).get("p95", 0.0),
            latency_p99_ms=report["latency"].get("e2e", {}).get("p99", 0.0),
        )
        
        return report
    
    def reset(self) -> None:
        """Clear all recorded results."""
        self.detection_results.clear()
        self.ocr_results.clear()
        self.identification_results.clear()
        self.latency_metrics = LatencyMetrics()
