"""
OCR Benchmark Module

Evaluates OCR performance using standard text recognition metrics.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER).
    
    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference words
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER).
    
    CER = edit_distance / len(reference)
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    
    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def normalize_text(text: str, lowercase: bool = True, remove_punctuation: bool = False) -> str:
    """Normalize text for comparison."""
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    return text


@dataclass
class OCRSample:
    """Single OCR evaluation sample."""
    sample_id: str
    ground_truth: str
    prediction: str
    confidence: float = 1.0
    image_path: str | None = None
    category: str = "general"  # spine, cover, title, author, etc.
    
    @property
    def cer(self) -> float:
        return character_error_rate(
            normalize_text(self.ground_truth),
            normalize_text(self.prediction)
        )
    
    @property
    def wer(self) -> float:
        return word_error_rate(
            normalize_text(self.ground_truth),
            normalize_text(self.prediction)
        )
    
    @property
    def exact_match(self) -> bool:
        return normalize_text(self.ground_truth) == normalize_text(self.prediction)
    
    @property
    def edit_distance(self) -> int:
        return levenshtein_distance(
            normalize_text(self.ground_truth),
            normalize_text(self.prediction)
        )


@dataclass
class OCRMetrics:
    """OCR evaluation metrics."""
    avg_cer: float = 0.0
    avg_wer: float = 0.0
    exact_match_rate: float = 0.0
    avg_edit_distance: float = 0.0
    median_cer: float = 0.0
    median_wer: float = 0.0
    cer_std: float = 0.0
    wer_std: float = 0.0
    confidence_correlation: float = 0.0  # Correlation between confidence and accuracy
    total_samples: int = 0
    total_characters: int = 0
    total_words: int = 0
    per_category_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    error_distribution: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_cer": round(self.avg_cer, 4),
            "avg_wer": round(self.avg_wer, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "avg_edit_distance": round(self.avg_edit_distance, 2),
            "median_cer": round(self.median_cer, 4),
            "median_wer": round(self.median_wer, 4),
            "cer_std": round(self.cer_std, 4),
            "wer_std": round(self.wer_std, 4),
            "confidence_correlation": round(self.confidence_correlation, 4),
            "total_samples": self.total_samples,
            "total_characters": self.total_characters,
            "total_words": self.total_words,
            "per_category_metrics": self.per_category_metrics,
            "error_distribution": self.error_distribution,
        }


class OCRBenchmark:
    """
    Benchmark suite for OCR evaluation.
    """
    
    def __init__(
        self,
        normalize_lowercase: bool = True,
        normalize_punctuation: bool = False,
    ):
        self.normalize_lowercase = normalize_lowercase
        self.normalize_punctuation = normalize_punctuation
        self.samples: list[OCRSample] = []
    
    def add_sample(
        self,
        sample_id: str,
        ground_truth: str,
        prediction: str,
        confidence: float = 1.0,
        category: str = "general",
        image_path: str | None = None,
    ) -> None:
        """Add an OCR sample for evaluation."""
        self.samples.append(OCRSample(
            sample_id=sample_id,
            ground_truth=ground_truth,
            prediction=prediction,
            confidence=confidence,
            category=category,
            image_path=image_path,
        ))
    
    def load_dataset(
        self,
        annotations_file: Path,
        images_dir: Path | None = None,
        ocr_engine: Any | None = None,
    ) -> None:
        """
        Load evaluation dataset and optionally run OCR.
        
        Expected annotation format:
        [
            {
                "id": "sample_001",
                "image": "spine_001.jpg",
                "text": "The Great Gatsby",
                "category": "spine"
            },
            ...
        ]
        """
        with open(annotations_file) as f:
            data = json.load(f)
        
        for sample in data:
            sample_id = sample.get("id", str(len(self.samples)))
            ground_truth = sample.get("text", "")
            category = sample.get("category", "general")
            image_name = sample.get("image")
            
            prediction = ""
            confidence = 1.0
            
            if ocr_engine and images_dir and image_name:
                image_path = images_dir / image_name
                if image_path.exists():
                    from PIL import Image
                    image = Image.open(image_path)
                    result = ocr_engine.extract_text(image)
                    prediction = result.get("text", "")
                    confidence = result.get("confidence", 1.0)
            else:
                prediction = sample.get("prediction", "")
                confidence = sample.get("confidence", 1.0)
            
            self.add_sample(
                sample_id=sample_id,
                ground_truth=ground_truth,
                prediction=prediction,
                confidence=confidence,
                category=category,
                image_path=str(images_dir / image_name) if images_dir and image_name else None,
            )
    
    def evaluate(self) -> OCRMetrics:
        """Run full evaluation and compute all metrics."""
        if not self.samples:
            logger.warning("No samples to evaluate")
            return OCRMetrics()
        
        metrics = OCRMetrics()
        metrics.total_samples = len(self.samples)
        
        cer_values = []
        wer_values = []
        edit_distances = []
        exact_matches = 0
        confidences = []
        accuracies = []  # 1 - CER for correlation
        
        # Collect per-category data
        category_data: dict[str, list[OCRSample]] = {}
        
        for sample in self.samples:
            cer = sample.cer
            wer = sample.wer
            
            cer_values.append(cer)
            wer_values.append(wer)
            edit_distances.append(sample.edit_distance)
            
            if sample.exact_match:
                exact_matches += 1
            
            confidences.append(sample.confidence)
            accuracies.append(1 - cer)
            
            metrics.total_characters += len(sample.ground_truth)
            metrics.total_words += len(sample.ground_truth.split())
            
            # Group by category
            if sample.category not in category_data:
                category_data[sample.category] = []
            category_data[sample.category].append(sample)
        
        # Compute aggregate metrics
        metrics.avg_cer = np.mean(cer_values)
        metrics.avg_wer = np.mean(wer_values)
        metrics.median_cer = np.median(cer_values)
        metrics.median_wer = np.median(wer_values)
        metrics.cer_std = np.std(cer_values)
        metrics.wer_std = np.std(wer_values)
        metrics.avg_edit_distance = np.mean(edit_distances)
        metrics.exact_match_rate = exact_matches / len(self.samples)
        
        # Confidence correlation
        if len(set(confidences)) > 1:  # Need variance
            metrics.confidence_correlation = float(np.corrcoef(confidences, accuracies)[0, 1])
        
        # Error distribution (binned CER)
        metrics.error_distribution = {
            "perfect (0%)": sum(1 for c in cer_values if c == 0),
            "excellent (<5%)": sum(1 for c in cer_values if 0 < c < 0.05),
            "good (5-10%)": sum(1 for c in cer_values if 0.05 <= c < 0.10),
            "fair (10-20%)": sum(1 for c in cer_values if 0.10 <= c < 0.20),
            "poor (20-50%)": sum(1 for c in cer_values if 0.20 <= c < 0.50),
            "failed (>50%)": sum(1 for c in cer_values if c >= 0.50),
        }
        
        # Per-category metrics
        for category, samples in category_data.items():
            cat_cer = [s.cer for s in samples]
            cat_wer = [s.wer for s in samples]
            cat_exact = sum(1 for s in samples if s.exact_match)
            
            metrics.per_category_metrics[category] = {
                "avg_cer": round(np.mean(cat_cer), 4),
                "avg_wer": round(np.mean(cat_wer), 4),
                "exact_match_rate": round(cat_exact / len(samples), 4),
                "count": len(samples),
            }
        
        return metrics
    
    def analyze_errors(self, top_n: int = 10) -> list[dict[str, Any]]:
        """
        Analyze worst performing samples.
        
        Returns list of samples with highest error rates.
        """
        sorted_samples = sorted(self.samples, key=lambda s: s.cer, reverse=True)
        
        return [
            {
                "sample_id": s.sample_id,
                "ground_truth": s.ground_truth,
                "prediction": s.prediction,
                "cer": round(s.cer, 4),
                "wer": round(s.wer, 4),
                "edit_distance": s.edit_distance,
                "confidence": s.confidence,
                "category": s.category,
            }
            for s in sorted_samples[:top_n]
        ]
    
    def analyze_confusion(self) -> dict[str, dict[str, int]]:
        """
        Analyze common character confusions.
        
        Returns confusion matrix of character substitutions.
        """
        confusion: dict[str, dict[str, int]] = {}
        
        for sample in self.samples:
            gt = normalize_text(sample.ground_truth, self.normalize_lowercase, self.normalize_punctuation)
            pred = normalize_text(sample.prediction, self.normalize_lowercase, self.normalize_punctuation)
            
            # Align characters using edit distance alignment
            aligned_gt, aligned_pred = self._align_strings(gt, pred)
            
            for g, p in zip(aligned_gt, aligned_pred):
                if g != p and g != "-" and p != "-":
                    if g not in confusion:
                        confusion[g] = {}
                    confusion[g][p] = confusion[g].get(p, 0) + 1
        
        # Sort by frequency
        for char in confusion:
            confusion[char] = dict(sorted(
                confusion[char].items(),
                key=lambda x: x[1],
                reverse=True
            ))
        
        return confusion
    
    def _align_strings(self, s1: str, s2: str) -> tuple[str, str]:
        """Align two strings using dynamic programming."""
        m, n = len(s1), len(s2)
        
        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Backtrack to get alignment
        aligned1, aligned2 = [], []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
                aligned1.append(s1[i-1])
                aligned2.append(s2[j-1])
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or dp[i-1][j] <= dp[i][j-1]):
                aligned1.append(s1[i-1])
                aligned2.append("-")
                i -= 1
            else:
                aligned1.append("-")
                aligned2.append(s2[j-1])
                j -= 1
        
        return "".join(reversed(aligned1)), "".join(reversed(aligned2))
    
    def generate_report(self, output_path: Path | None = None) -> str:
        """Generate human-readable evaluation report."""
        metrics = self.evaluate()
        errors = self.analyze_errors(5)
        
        lines = [
            "=" * 60,
            "OCR BENCHMARK REPORT",
            "=" * 60,
            "",
            "Overall Metrics:",
            f"  Avg CER:           {metrics.avg_cer:.4f} (±{metrics.cer_std:.4f})",
            f"  Avg WER:           {metrics.avg_wer:.4f} (±{metrics.wer_std:.4f})",
            f"  Median CER:        {metrics.median_cer:.4f}",
            f"  Median WER:        {metrics.median_wer:.4f}",
            f"  Exact Match Rate:  {metrics.exact_match_rate:.4f}",
            f"  Avg Edit Distance: {metrics.avg_edit_distance:.2f}",
            "",
            f"  Confidence ↔ Accuracy Correlation: {metrics.confidence_correlation:.4f}",
            "",
            "Counts:",
            f"  Total Samples:    {metrics.total_samples}",
            f"  Total Characters: {metrics.total_characters}",
            f"  Total Words:      {metrics.total_words}",
            "",
            "Error Distribution:",
        ]
        
        for bucket, count in metrics.error_distribution.items():
            pct = count / metrics.total_samples * 100
            bar = "█" * int(pct / 5)
            lines.append(f"  {bucket:16s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        if metrics.per_category_metrics:
            lines.extend([
                "",
                "Per-Category Metrics:",
                "-" * 40,
            ])
            for cat, cat_metrics in metrics.per_category_metrics.items():
                lines.extend([
                    f"  {cat} (n={cat_metrics['count']}):",
                    f"    CER: {cat_metrics['avg_cer']:.4f}  WER: {cat_metrics['avg_wer']:.4f}  "
                    f"Exact: {cat_metrics['exact_match_rate']:.4f}",
                ])
        
        if errors:
            lines.extend([
                "",
                "Top Errors:",
                "-" * 40,
            ])
            for err in errors:
                lines.extend([
                    f"  [{err['sample_id']}] CER={err['cer']:.3f}",
                    f"    GT:   {err['ground_truth'][:50]}{'...' if len(err['ground_truth']) > 50 else ''}",
                    f"    Pred: {err['prediction'][:50]}{'...' if len(err['prediction']) > 50 else ''}",
                ])
        
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report)
            json_path = output_path.with_suffix(".json")
            json_path.write_text(json.dumps(metrics.to_dict(), indent=2))
            logger.info(f"Report saved to {output_path}")
        
        return report


def create_synthetic_benchmark(
    num_samples: int = 200,
    error_rate: float = 0.1,
    categories: list[str] | None = None,
) -> OCRBenchmark:
    """
    Create a synthetic benchmark for testing the evaluation pipeline.
    
    Generates fake OCR predictions with controlled error rates.
    """
    import random
    import string
    
    benchmark = OCRBenchmark()
    categories = categories or ["spine", "cover", "title", "author"]
    
    # Sample texts
    sample_texts = [
        "The Great Gatsby",
        "To Kill a Mockingbird",
        "1984 George Orwell",
        "Pride and Prejudice",
        "The Catcher in the Rye",
        "Lord of the Flies",
        "Animal Farm",
        "Brave New World",
        "The Hobbit",
        "Fahrenheit 451",
        "Jane Eyre",
        "Wuthering Heights",
        "Great Expectations",
        "Crime and Punishment",
        "War and Peace",
    ]
    
    def add_noise(text: str, rate: float) -> str:
        """Add OCR-like noise to text."""
        result = list(text)
        for i in range(len(result)):
            if random.random() < rate:
                action = random.choice(["substitute", "delete", "insert"])
                if action == "substitute":
                    # Common OCR confusions
                    confusions = {"0": "O", "O": "0", "l": "1", "1": "l", "I": "l", "m": "rn", "rn": "m"}
                    result[i] = confusions.get(result[i], random.choice(string.ascii_letters))
                elif action == "delete":
                    result[i] = ""
                else:  # insert
                    result[i] = result[i] + random.choice(string.ascii_letters)
        return "".join(result)
    
    for i in range(num_samples):
        gt = random.choice(sample_texts)
        category = random.choice(categories)
        
        # Vary error rate by category
        cat_error_rate = error_rate * (1.5 if category == "spine" else 1.0)
        prediction = add_noise(gt, cat_error_rate)
        
        # Confidence inversely related to actual error
        actual_cer = character_error_rate(gt.lower(), prediction.lower())
        confidence = max(0.3, 1.0 - actual_cer + random.gauss(0, 0.1))
        
        benchmark.add_sample(
            sample_id=f"ocr_{i:04d}",
            ground_truth=gt,
            prediction=prediction,
            confidence=min(1.0, confidence),
            category=category,
        )
    
    return benchmark
