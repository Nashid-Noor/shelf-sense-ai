"""Benchmark runner for the full pipeline (Detection, OCR, ID, E2E)."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .metrics import (
    PipelineEvaluator,
    BoundingBox,
    DetectionResult,
    OCRResult,
    IdentificationResult,
    MetricsSummary,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkSample:
    """Single benchmark sample with ground truth."""
    id: str
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    
    # Ground truth
    gt_bboxes: List[Dict] = field(default_factory=list)
    gt_texts: List[str] = field(default_factory=list)
    gt_book_ids: List[str] = field(default_factory=list)
    gt_metadata: Dict = field(default_factory=dict)
    
    def get_bboxes(self) -> List[BoundingBox]:
        """Convert ground truth bboxes to BoundingBox objects."""
        return [
            BoundingBox(
                x1=bb["x1"],
                y1=bb["y1"],
                x2=bb["x2"],
                y2=bb["y2"],
                class_id=bb.get("class_id", 0),
            )
            for bb in self.gt_bboxes
        ]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    name: str = "default"
    description: str = ""
    
    # What to evaluate
    run_detection: bool = True
    run_ocr: bool = True
    run_identification: bool = True
    run_e2e: bool = True
    
    # Settings
    batch_size: int = 8
    num_workers: int = 4
    warmup_samples: int = 5
    max_samples: Optional[int] = None
    
    # Detection settings
    detection_iou_thresholds: List[float] = field(
        default_factory=lambda: [0.5, 0.75]
    )
    
    # Identification settings
    identification_k_values: List[int] = field(
        default_factory=lambda: [1, 3, 5, 10]
    )
    
    # Output
    output_dir: str = "./benchmark_results"
    save_predictions: bool = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    metrics: Dict[str, Any]
    summary: MetricsSummary
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    duration_seconds: float = 0.0
    num_samples: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "config": asdict(self.config),
            "metrics": self.metrics,
            "summary": asdict(self.summary),
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "num_samples": self.num_samples,
            "errors": self.errors,
        }
    
    def save(self, path: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"benchmark_{self.config.name}_{timestamp}.json"
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {path}")
        return str(path)


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Runs systematic benchmarks on the ShelfSense pipeline.
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        detector: Optional[Any] = None,
        ocr_engine: Optional[Any] = None,
        identifier: Optional[Any] = None,
        pipeline: Optional[Any] = None,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration.
            detector: Detection model instance.
            ocr_engine: OCR engine instance.
            identifier: Book identifier instance.
            pipeline: Full pipeline instance for e2e tests.
        """
        self.config = config or BenchmarkConfig()
        self.detector = detector
        self.ocr_engine = ocr_engine
        self.identifier = identifier
        self.pipeline = pipeline
        
        self.evaluator = PipelineEvaluator()
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
    
    def _load_image(self, sample: BenchmarkSample) -> Optional[np.ndarray]:
        """Load image from sample."""
        import cv2
        
        if sample.image_data:
            nparr = np.frombuffer(sample.image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif sample.image_path:
            return cv2.imread(sample.image_path)
        return None
    
    async def run_detection_benchmark(
        self,
        samples: List[BenchmarkSample],
    ) -> Dict[str, Any]:
        """
        Run detection benchmark.
        
        Args:
            samples: List of samples with ground truth bboxes.
        
        Returns:
            Detection metrics dictionary.
        """
        if not self.detector:
            logger.warning("No detector provided, skipping detection benchmark")
            return {}
        
        logger.info(f"Running detection benchmark on {len(samples)} samples")
        
        results = []
        latencies = []
        
        for sample in samples:
            try:
                image = self._load_image(sample)
                if image is None:
                    logger.warning(f"Could not load image for sample {sample.id}")
                    continue
                
                # Time detection
                start = time.perf_counter()
                predictions = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.detector.detect,
                    image,
                )
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                
                # Convert predictions to BoundingBox format
                pred_bboxes = [
                    BoundingBox(
                        x1=p["bbox"][0],
                        y1=p["bbox"][1],
                        x2=p["bbox"][2],
                        y2=p["bbox"][3],
                        confidence=p.get("confidence", 1.0),
                        class_id=p.get("class_id", 0),
                    )
                    for p in predictions
                ]
                
                # Record result
                self.evaluator.add_detection_result(
                    predicted=pred_bboxes,
                    ground_truth=sample.get_bboxes(),
                    image_id=sample.id,
                )
                
                self.evaluator.record_latency("detection", latency)
                
            except Exception as e:
                logger.error(f"Detection failed for sample {sample.id}: {e}")
        
        return {
            "num_samples": len(samples),
            "latency_mean_ms": np.mean(latencies) if latencies else 0,
            "latency_p95_ms": np.percentile(latencies, 95) if latencies else 0,
        }
    
    async def run_ocr_benchmark(
        self,
        samples: List[BenchmarkSample],
    ) -> Dict[str, Any]:
        """
        Run OCR benchmark.
        
        Args:
            samples: List of samples with ground truth texts.
        
        Returns:
            OCR metrics dictionary.
        """
        if not self.ocr_engine:
            logger.warning("No OCR engine provided, skipping OCR benchmark")
            return {}
        
        logger.info(f"Running OCR benchmark on {len(samples)} samples")
        
        latencies = []
        
        for sample in samples:
            try:
                image = self._load_image(sample)
                if image is None:
                    continue
                
                # For each ground truth text (each book region)
                for i, gt_text in enumerate(sample.gt_texts):
                    # If we have bboxes, crop to region
                    if i < len(sample.gt_bboxes):
                        bbox = sample.gt_bboxes[i]
                        roi = image[
                            int(bbox["y1"]):int(bbox["y2"]),
                            int(bbox["x1"]):int(bbox["x2"]),
                        ]
                    else:
                        roi = image
                    
                    # Time OCR
                    start = time.perf_counter()
                    predicted_text = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.ocr_engine.extract_text,
                        roi,
                    )
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                    
                    # Record result
                    self.evaluator.add_ocr_result(
                        predicted=predicted_text,
                        ground_truth=gt_text,
                        book_id=f"{sample.id}_{i}",
                    )
                    
                    self.evaluator.record_latency("ocr", latency)
                    
            except Exception as e:
                logger.error(f"OCR failed for sample {sample.id}: {e}")
        
        return {
            "num_samples": sum(len(s.gt_texts) for s in samples),
            "latency_mean_ms": np.mean(latencies) if latencies else 0,
        }
    
    async def run_identification_benchmark(
        self,
        samples: List[BenchmarkSample],
    ) -> Dict[str, Any]:
        """
        Run book identification benchmark.
        
        Args:
            samples: List of samples with ground truth book IDs.
        
        Returns:
            Identification metrics dictionary.
        """
        if not self.identifier:
            logger.warning("No identifier provided, skipping identification benchmark")
            return {}
        
        logger.info(f"Running identification benchmark on {len(samples)} samples")
        
        latencies = []
        
        for sample in samples:
            try:
                # Use OCR text if available, otherwise extract
                for i, (gt_text, gt_id) in enumerate(
                    zip(sample.gt_texts, sample.gt_book_ids)
                ):
                    # Time identification
                    start = time.perf_counter()
                    candidates = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.identifier.identify,
                        gt_text,  # Use ground truth text for pure ID eval
                    )
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                    
                    # Extract ranked IDs
                    predicted_ids = [c["id"] for c in candidates]
                    scores = [c.get("score", 0) for c in candidates]
                    
                    # Record result
                    self.evaluator.add_identification_result(
                        predicted_ids=predicted_ids,
                        ground_truth_id=gt_id,
                        query_id=f"{sample.id}_{i}",
                        scores=scores,
                    )
                    
                    self.evaluator.record_latency("identification", latency)
                    
            except Exception as e:
                logger.error(f"Identification failed for sample {sample.id}: {e}")
        
        return {
            "num_samples": sum(len(s.gt_book_ids) for s in samples),
            "latency_mean_ms": np.mean(latencies) if latencies else 0,
        }
    
    async def run_e2e_benchmark(
        self,
        samples: List[BenchmarkSample],
    ) -> Dict[str, Any]:
        """
        Run end-to-end pipeline benchmark.
        
        Args:
            samples: List of samples with full ground truth.
        
        Returns:
            E2E metrics dictionary.
        """
        if not self.pipeline:
            logger.warning("No pipeline provided, skipping e2e benchmark")
            return {}
        
        logger.info(f"Running e2e benchmark on {len(samples)} samples")
        
        latencies = []
        correct = 0
        total = 0
        
        for sample in samples:
            try:
                image = self._load_image(sample)
                if image is None:
                    continue
                
                # Time full pipeline
                start = time.perf_counter()
                results = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.pipeline.process_image,
                    image,
                )
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                
                self.evaluator.record_latency("e2e", latency)
                
                # Evaluate accuracy
                predicted_ids = set(r["book_id"] for r in results if r.get("book_id"))
                gt_ids = set(sample.gt_book_ids)
                
                # Count correct identifications
                correct += len(predicted_ids & gt_ids)
                total += len(gt_ids)
                
            except Exception as e:
                logger.error(f"E2E failed for sample {sample.id}: {e}")
        
        return {
            "num_samples": len(samples),
            "accuracy": correct / total if total > 0 else 0,
            "latency_mean_ms": np.mean(latencies) if latencies else 0,
            "latency_p50_ms": np.percentile(latencies, 50) if latencies else 0,
            "latency_p95_ms": np.percentile(latencies, 95) if latencies else 0,
            "latency_p99_ms": np.percentile(latencies, 99) if latencies else 0,
        }
    
    async def run(
        self,
        samples: List[BenchmarkSample],
        warmup: bool = True,
    ) -> BenchmarkResult:
        """
        Run full benchmark suite.
        
        Args:
            samples: List of benchmark samples.
            warmup: Whether to run warmup iterations.
        
        Returns:
            Complete benchmark results.
        """
        start_time = time.perf_counter()
        errors = []
        
        # Apply sample limit
        if self.config.max_samples:
            samples = samples[:self.config.max_samples]
        
        logger.info(f"Starting benchmark '{self.config.name}' with {len(samples)} samples")
        
        # Warmup
        if warmup and self.config.warmup_samples > 0:
            logger.info(f"Running warmup with {self.config.warmup_samples} samples")
            warmup_samples = samples[:self.config.warmup_samples]
            
            if self.config.run_detection and self.detector:
                await self.run_detection_benchmark(warmup_samples)
            if self.config.run_ocr and self.ocr_engine:
                await self.run_ocr_benchmark(warmup_samples)
            
            # Reset evaluator after warmup
            self.evaluator.reset()
        
        # Run benchmarks
        component_results = {}
        
        try:
            if self.config.run_detection:
                component_results["detection"] = await self.run_detection_benchmark(
                    samples
                )
        except Exception as e:
            errors.append(f"Detection benchmark error: {e}")
            logger.error(f"Detection benchmark failed: {e}")
        
        try:
            if self.config.run_ocr:
                component_results["ocr"] = await self.run_ocr_benchmark(samples)
        except Exception as e:
            errors.append(f"OCR benchmark error: {e}")
            logger.error(f"OCR benchmark failed: {e}")
        
        try:
            if self.config.run_identification:
                component_results["identification"] = await self.run_identification_benchmark(
                    samples
                )
        except Exception as e:
            errors.append(f"Identification benchmark error: {e}")
            logger.error(f"Identification benchmark failed: {e}")
        
        try:
            if self.config.run_e2e:
                component_results["e2e"] = await self.run_e2e_benchmark(samples)
        except Exception as e:
            errors.append(f"E2E benchmark error: {e}")
            logger.error(f"E2E benchmark failed: {e}")
        
        # Get full evaluation report
        evaluation = self.evaluator.evaluate()
        
        # Merge results
        metrics = {
            **evaluation,
            "component_results": component_results,
        }
        
        duration = time.perf_counter() - start_time
        
        result = BenchmarkResult(
            config=self.config,
            metrics=metrics,
            summary=evaluation["summary"],
            duration_seconds=duration,
            num_samples=len(samples),
            errors=errors,
        )
        
        logger.info(f"Benchmark completed in {duration:.2f}s")
        logger.info(f"Summary: mAP={result.summary.detection_map:.3f}, "
                   f"CER={result.summary.ocr_cer:.3f}, "
                   f"MRR={result.summary.identification_mrr:.3f}")
        
        return result


# =============================================================================
# Dataset Loaders
# =============================================================================

class BenchmarkDataset:
    """
    Load benchmark datasets in various formats.
    
    Supports:
    - COCO format (detection)
    - Custom JSON format
    - Directory of images with annotation files
    """
    
    @staticmethod
    def load_coco(
        annotation_path: str,
        image_dir: str,
        max_samples: Optional[int] = None,
    ) -> List[BenchmarkSample]:
        """
        Load dataset in COCO format.
        
        Args:
            annotation_path: Path to COCO annotations JSON.
            image_dir: Directory containing images.
            max_samples: Maximum samples to load.
        
        Returns:
            List of BenchmarkSample objects.
        """
        with open(annotation_path) as f:
            coco = json.load(f)
        
        # Build image ID to annotations mapping
        img_to_anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        samples = []
        
        for img_info in coco["images"]:
            if max_samples and len(samples) >= max_samples:
                break
            
            img_id = img_info["id"]
            img_path = Path(image_dir) / img_info["file_name"]
            
            # Get annotations for this image
            anns = img_to_anns.get(img_id, [])
            
            # Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]
            bboxes = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                bboxes.append({
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "class_id": ann.get("category_id", 0),
                })
            
            samples.append(BenchmarkSample(
                id=str(img_id),
                image_path=str(img_path),
                gt_bboxes=bboxes,
                gt_metadata=img_info,
            ))
        
        return samples
    
    @staticmethod
    def load_custom(
        annotation_path: str,
        image_dir: Optional[str] = None,
    ) -> List[BenchmarkSample]:
        """
        Load dataset in custom ShelfSense format.
        
        Expected JSON structure:
        {
            "samples": [
                {
                    "id": "sample_001",
                    "image": "path/to/image.jpg",
                    "books": [
                        {
                            "bbox": [x1, y1, x2, y2],
                            "title": "Book Title",
                            "author": "Author Name",
                            "isbn": "1234567890",
                            "id": "book_id"
                        }
                    ]
                }
            ]
        }
        
        Args:
            annotation_path: Path to annotations JSON.
            image_dir: Base directory for image paths.
        
        Returns:
            List of BenchmarkSample objects.
        """
        with open(annotation_path) as f:
            data = json.load(f)
        
        samples = []
        
        for item in data.get("samples", []):
            img_path = item["image"]
            if image_dir:
                img_path = str(Path(image_dir) / img_path)
            
            bboxes = []
            texts = []
            book_ids = []
            
            for book in item.get("books", []):
                if "bbox" in book:
                    bboxes.append({
                        "x1": book["bbox"][0],
                        "y1": book["bbox"][1],
                        "x2": book["bbox"][2],
                        "y2": book["bbox"][3],
                        "class_id": 0,
                    })
                
                # Build expected text from title/author
                text_parts = []
                if book.get("title"):
                    text_parts.append(book["title"])
                if book.get("author"):
                    text_parts.append(book["author"])
                texts.append(" ".join(text_parts))
                
                if book.get("id"):
                    book_ids.append(book["id"])
                elif book.get("isbn"):
                    book_ids.append(book["isbn"])
            
            samples.append(BenchmarkSample(
                id=item["id"],
                image_path=img_path,
                gt_bboxes=bboxes,
                gt_texts=texts,
                gt_book_ids=book_ids,
                gt_metadata=item,
            ))
        
        return samples
    
    @staticmethod
    def create_synthetic(
        num_samples: int = 100,
        books_per_image: int = 5,
        image_size: Tuple[int, int] = (1920, 1080),
    ) -> List[BenchmarkSample]:
        """
        Create synthetic benchmark dataset for testing.
        
        Args:
            num_samples: Number of synthetic samples.
            books_per_image: Number of books per image.
            image_size: Image dimensions (width, height).
        
        Returns:
            List of synthetic BenchmarkSample objects.
        """
        samples = []
        
        sample_titles = [
            "The Great Gatsby", "To Kill a Mockingbird", "1984",
            "Pride and Prejudice", "The Catcher in the Rye",
            "Lord of the Flies", "The Grapes of Wrath",
            "Brave New World", "Animal Farm", "Fahrenheit 451",
        ]
        
        sample_authors = [
            "F. Scott Fitzgerald", "Harper Lee", "George Orwell",
            "Jane Austen", "J.D. Salinger", "William Golding",
            "John Steinbeck", "Aldous Huxley", "George Orwell",
            "Ray Bradbury",
        ]
        
        for i in range(num_samples):
            bboxes = []
            texts = []
            book_ids = []
            
            # Generate random non-overlapping bboxes
            w, h = image_size
            spine_width = w // (books_per_image + 2)
            
            for j in range(books_per_image):
                x1 = (j + 1) * spine_width
                x2 = x1 + spine_width - 10
                y1 = 50
                y2 = h - 50
                
                bboxes.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_id": 0,
                })
                
                title_idx = (i * books_per_image + j) % len(sample_titles)
                texts.append(f"{sample_titles[title_idx]} {sample_authors[title_idx]}")
                book_ids.append(f"book_{i}_{j}")
            
            samples.append(BenchmarkSample(
                id=f"synthetic_{i:04d}",
                image_path=None,  # No actual image
                gt_bboxes=bboxes,
                gt_texts=texts,
                gt_book_ids=book_ids,
                gt_metadata={"synthetic": True},
            ))
        
        return samples


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_quick_benchmark(
    pipeline: Any,
    samples: List[BenchmarkSample],
    name: str = "quick",
) -> BenchmarkResult:
    """
    Run a quick benchmark with default settings.
    
    Args:
        pipeline: Full pipeline instance.
        samples: Benchmark samples.
        name: Benchmark name.
    
    Returns:
        Benchmark results.
    """
    config = BenchmarkConfig(
        name=name,
        max_samples=50,
        warmup_samples=5,
    )
    
    runner = BenchmarkRunner(config=config, pipeline=pipeline)
    return await runner.run(samples)


def compare_results(
    results: List[BenchmarkResult],
    metrics: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple benchmark results.
    
    Args:
        results: List of benchmark results.
        metrics: Specific metrics to compare.
    
    Returns:
        Comparison dictionary mapping metric to result values.
    """
    if metrics is None:
        metrics = [
            "detection_map", "ocr_cer", "identification_mrr",
            "latency_p95_ms",
        ]
    
    comparison = {}
    
    for metric in metrics:
        comparison[metric] = {}
        for result in results:
            value = getattr(result.summary, metric, None)
            comparison[metric][result.config.name] = value
    
    return comparison
