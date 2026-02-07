"""
ShelfSense AI Evaluation Module.

Comprehensive evaluation toolkit for:
- Object detection (mAP, IoU, precision/recall)
- OCR accuracy (CER, WER, exact match)
- Book identification (P@k, MRR, NDCG)
- Latency profiling
- End-to-end benchmarks
"""

from .metrics import (
    # Data classes
    BoundingBox,
    DetectionResult,
    OCRResult,
    IdentificationResult,
    MetricsSummary,
    
    # Metric calculators
    DetectionMetrics,
    OCRMetrics,
    IdentificationMetrics,
    LatencyMetrics,
    
    # Aggregator
    PipelineEvaluator,
)

from .benchmark import (
    # Configuration
    BenchmarkSample,
    BenchmarkConfig,
    BenchmarkResult,
    
    # Runner
    BenchmarkRunner,
    
    # Dataset utilities
    BenchmarkDataset,
    
    # Convenience functions
    compare_results,
)

__all__ = [
    # Data classes
    "BoundingBox",
    "DetectionResult",
    "OCRResult",
    "IdentificationResult",
    "MetricsSummary",
    "BenchmarkSample",
    "BenchmarkConfig",
    "BenchmarkResult",
    
    # Metric calculators
    "DetectionMetrics",
    "OCRMetrics",
    "IdentificationMetrics",
    "LatencyMetrics",
    
    # Evaluators
    "PipelineEvaluator",
    "BenchmarkRunner",
    
    # Utilities
    "BenchmarkDataset",
    "compare_results",
]
