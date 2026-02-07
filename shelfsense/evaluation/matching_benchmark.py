"""
Matching Benchmark Module

Evaluates book identification/matching performance:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Hit Rate
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchCandidate:
    """A candidate match result."""
    book_id: str
    title: str
    author: str = ""
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchQuery:
    """A single matching query with ground truth."""
    query_id: str
    query_text: str  # OCR text or search query
    query_image_embedding: list[float] | None = None
    ground_truth_id: str = ""  # Correct book ID
    ground_truth_title: str = ""
    ground_truth_author: str = ""
    candidates: list[MatchCandidate] = field(default_factory=list)
    relevant_ids: set[str] = field(default_factory=set)  # For multi-relevant scenarios
    
    @property
    def hit_at_1(self) -> bool:
        """Check if top candidate is correct."""
        if not self.candidates:
            return False
        return self.candidates[0].book_id == self.ground_truth_id or \
               self.candidates[0].book_id in self.relevant_ids
    
    def hit_at_k(self, k: int) -> bool:
        """Check if correct answer appears in top-k."""
        top_k_ids = {c.book_id for c in self.candidates[:k]}
        return self.ground_truth_id in top_k_ids or bool(top_k_ids & self.relevant_ids)
    
    def reciprocal_rank(self) -> float:
        """Compute reciprocal rank (1/position of first correct result)."""
        for i, candidate in enumerate(self.candidates):
            if candidate.book_id == self.ground_truth_id or candidate.book_id in self.relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def precision_at_k(self, k: int) -> float:
        """Compute precision@k."""
        if k == 0:
            return 0.0
        top_k = self.candidates[:k]
        relevant_in_top_k = sum(
            1 for c in top_k
            if c.book_id == self.ground_truth_id or c.book_id in self.relevant_ids
        )
        return relevant_in_top_k / k
    
    def recall_at_k(self, k: int) -> float:
        """Compute recall@k."""
        total_relevant = 1 if self.ground_truth_id else 0
        total_relevant += len(self.relevant_ids)
        if total_relevant == 0:
            return 0.0
        
        top_k_ids = {c.book_id for c in self.candidates[:k]}
        found = 0
        if self.ground_truth_id in top_k_ids:
            found += 1
        found += len(top_k_ids & self.relevant_ids)
        
        return found / total_relevant
    
    def ndcg_at_k(self, k: int) -> float:
        """Compute NDCG@k (assuming binary relevance)."""
        def dcg(relevances: list[int], k: int) -> float:
            relevances = relevances[:k]
            return sum(
                rel / np.log2(i + 2)
                for i, rel in enumerate(relevances)
            )
        
        # Get relevance scores for candidates
        relevances = [
            1 if (c.book_id == self.ground_truth_id or c.book_id in self.relevant_ids)
            else 0
            for c in self.candidates
        ]
        
        # Ideal DCG (all relevant items at top)
        total_relevant = 1 if self.ground_truth_id else 0
        total_relevant += len(self.relevant_ids)
        ideal_relevances = [1] * total_relevant + [0] * (len(relevances) - total_relevant)
        
        dcg_score = dcg(relevances, k)
        idcg_score = dcg(ideal_relevances, k)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0


@dataclass
class MatchingMetrics:
    """Matching evaluation metrics."""
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    total_queries: int = 0
    queries_with_match: int = 0
    avg_candidates_per_query: float = 0.0
    avg_first_correct_rank: float = 0.0  # Average rank of first correct result
    per_category_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": {
                "p@1": round(self.precision_at_1, 4),
                "p@3": round(self.precision_at_3, 4),
                "p@5": round(self.precision_at_5, 4),
                "p@10": round(self.precision_at_10, 4),
            },
            "recall": {
                "r@1": round(self.recall_at_1, 4),
                "r@3": round(self.recall_at_3, 4),
                "r@5": round(self.recall_at_5, 4),
                "r@10": round(self.recall_at_10, 4),
            },
            "mrr": round(self.mrr, 4),
            "ndcg": {
                "ndcg@5": round(self.ndcg_at_5, 4),
                "ndcg@10": round(self.ndcg_at_10, 4),
            },
            "hit_rate": {
                "hr@1": round(self.hit_rate_at_1, 4),
                "hr@3": round(self.hit_rate_at_3, 4),
                "hr@5": round(self.hit_rate_at_5, 4),
                "hr@10": round(self.hit_rate_at_10, 4),
            },
            "total_queries": self.total_queries,
            "queries_with_match": self.queries_with_match,
            "avg_candidates_per_query": round(self.avg_candidates_per_query, 2),
            "avg_first_correct_rank": round(self.avg_first_correct_rank, 2),
            "per_category_metrics": self.per_category_metrics,
        }


class MatchingBenchmark:
    """
    Benchmark suite for book identification/matching evaluation.
    
    Computes retrieval metrics including precision@k,

    recall@k, MRR, NDCG, and hit rates.
    """
    
    def __init__(self):
        self.queries: list[MatchQuery] = []
        self.query_categories: dict[str, str] = {}  # query_id -> category
    
    def add_query(
        self,
        query_id: str,
        query_text: str,
        ground_truth_id: str,
        candidates: list[dict[str, Any]],
        ground_truth_title: str = "",
        ground_truth_author: str = "",
        relevant_ids: set[str] | None = None,
        category: str = "general",
    ) -> None:
        """Add a matching query for evaluation."""
        match_candidates = [
            MatchCandidate(
                book_id=c.get("id", c.get("book_id", "")),
                title=c.get("title", ""),
                author=c.get("author", ""),
                score=c.get("score", c.get("confidence", 0.0)),
                rank=i + 1,
                metadata=c.get("metadata", {}),
            )
            for i, c in enumerate(candidates)
        ]
        
        self.queries.append(MatchQuery(
            query_id=query_id,
            query_text=query_text,
            ground_truth_id=ground_truth_id,
            ground_truth_title=ground_truth_title,
            ground_truth_author=ground_truth_author,
            candidates=match_candidates,
            relevant_ids=relevant_ids or set(),
        ))
        
        self.query_categories[query_id] = category
    
    def load_dataset(
        self,
        queries_file: Path,
        matcher: Any | None = None,
    ) -> None:
        """
        Load evaluation dataset and optionally run matching.
        
        Expected format:
        [
            {
                "query_id": "q_001",
                "query_text": "The Great Gatsby F Scott Fitzgerald",
                "ground_truth_id": "isbn:9780743273565",
                "ground_truth_title": "The Great Gatsby",
                "ground_truth_author": "F. Scott Fitzgerald",
                "category": "spine_ocr",
                "candidates": [...]  // Optional if matcher provided
            },
            ...
        ]
        """
        with open(queries_file) as f:
            data = json.load(f)
        
        for item in data:
            query_id = item["query_id"]
            query_text = item["query_text"]
            ground_truth_id = item["ground_truth_id"]
            
            if matcher:
                # Run matcher to get candidates
                results = matcher.match(query_text)
                candidates = [
                    {
                        "id": r.get("id"),
                        "title": r.get("title"),
                        "author": r.get("author"),
                        "score": r.get("score"),
                    }
                    for r in results
                ]
            else:
                candidates = item.get("candidates", [])
            
            self.add_query(
                query_id=query_id,
                query_text=query_text,
                ground_truth_id=ground_truth_id,
                candidates=candidates,
                ground_truth_title=item.get("ground_truth_title", ""),
                ground_truth_author=item.get("ground_truth_author", ""),
                relevant_ids=set(item.get("relevant_ids", [])),
                category=item.get("category", "general"),
            )
    
    def evaluate(self) -> MatchingMetrics:
        """Run full evaluation and compute all metrics."""
        if not self.queries:
            logger.warning("No queries to evaluate")
            return MatchingMetrics()
        
        metrics = MatchingMetrics()
        metrics.total_queries = len(self.queries)
        
        # Collect per-query metrics
        p1_scores, p3_scores, p5_scores, p10_scores = [], [], [], []
        r1_scores, r3_scores, r5_scores, r10_scores = [], [], [], []
        mrr_scores = []
        ndcg5_scores, ndcg10_scores = [], []
        h1_scores, h3_scores, h5_scores, h10_scores = [], [], [], []
        first_correct_ranks = []
        candidate_counts = []
        
        # Per-category tracking
        category_data: dict[str, list[MatchQuery]] = {}
        
        for query in self.queries:
            # Precision@k
            p1_scores.append(query.precision_at_k(1))
            p3_scores.append(query.precision_at_k(3))
            p5_scores.append(query.precision_at_k(5))
            p10_scores.append(query.precision_at_k(10))
            
            # Recall@k
            r1_scores.append(query.recall_at_k(1))
            r3_scores.append(query.recall_at_k(3))
            r5_scores.append(query.recall_at_k(5))
            r10_scores.append(query.recall_at_k(10))
            
            # MRR
            rr = query.reciprocal_rank()
            mrr_scores.append(rr)
            
            if rr > 0:
                first_correct_ranks.append(1.0 / rr)
                metrics.queries_with_match += 1
            
            # NDCG@k
            ndcg5_scores.append(query.ndcg_at_k(5))
            ndcg10_scores.append(query.ndcg_at_k(10))
            
            # Hit rate@k
            h1_scores.append(1.0 if query.hit_at_k(1) else 0.0)
            h3_scores.append(1.0 if query.hit_at_k(3) else 0.0)
            h5_scores.append(1.0 if query.hit_at_k(5) else 0.0)
            h10_scores.append(1.0 if query.hit_at_k(10) else 0.0)
            
            candidate_counts.append(len(query.candidates))
            
            # Group by category
            category = self.query_categories.get(query.query_id, "general")
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(query)
        
        # Compute aggregates
        metrics.precision_at_1 = np.mean(p1_scores)
        metrics.precision_at_3 = np.mean(p3_scores)
        metrics.precision_at_5 = np.mean(p5_scores)
        metrics.precision_at_10 = np.mean(p10_scores)
        
        metrics.recall_at_1 = np.mean(r1_scores)
        metrics.recall_at_3 = np.mean(r3_scores)
        metrics.recall_at_5 = np.mean(r5_scores)
        metrics.recall_at_10 = np.mean(r10_scores)
        
        metrics.mrr = np.mean(mrr_scores)
        
        metrics.ndcg_at_5 = np.mean(ndcg5_scores)
        metrics.ndcg_at_10 = np.mean(ndcg10_scores)
        
        metrics.hit_rate_at_1 = np.mean(h1_scores)
        metrics.hit_rate_at_3 = np.mean(h3_scores)
        metrics.hit_rate_at_5 = np.mean(h5_scores)
        metrics.hit_rate_at_10 = np.mean(h10_scores)
        
        metrics.avg_candidates_per_query = np.mean(candidate_counts)
        
        if first_correct_ranks:
            metrics.avg_first_correct_rank = np.mean(first_correct_ranks)
        
        # Per-category metrics
        for category, cat_queries in category_data.items():
            cat_mrr = np.mean([q.reciprocal_rank() for q in cat_queries])
            cat_p1 = np.mean([q.precision_at_k(1) for q in cat_queries])
            cat_hr5 = np.mean([1.0 if q.hit_at_k(5) else 0.0 for q in cat_queries])
            
            metrics.per_category_metrics[category] = {
                "mrr": round(cat_mrr, 4),
                "p@1": round(cat_p1, 4),
                "hr@5": round(cat_hr5, 4),
                "count": len(cat_queries),
            }
        
        return metrics
    
    def analyze_failures(self, top_n: int = 10) -> list[dict[str, Any]]:
        """
        Analyze queries where matching failed (no correct result in top-k).
        """
        failures = []
        
        for query in self.queries:
            if not query.hit_at_k(5):  # Failed to find in top-5
                failures.append({
                    "query_id": query.query_id,
                    "query_text": query.query_text,
                    "ground_truth": {
                        "id": query.ground_truth_id,
                        "title": query.ground_truth_title,
                        "author": query.ground_truth_author,
                    },
                    "top_3_candidates": [
                        {
                            "id": c.book_id,
                            "title": c.title,
                            "author": c.author,
                            "score": c.score,
                        }
                        for c in query.candidates[:3]
                    ],
                    "reciprocal_rank": query.reciprocal_rank(),
                    "category": self.query_categories.get(query.query_id, "general"),
                })
        
        return failures[:top_n]
    
    def score_distribution(self) -> dict[str, list[float]]:
        """
        Get score distribution for correct vs incorrect matches.
        """
        correct_scores = []
        incorrect_scores = []
        
        for query in self.queries:
            for candidate in query.candidates:
                is_correct = (
                    candidate.book_id == query.ground_truth_id or
                    candidate.book_id in query.relevant_ids
                )
                if is_correct:
                    correct_scores.append(candidate.score)
                else:
                    incorrect_scores.append(candidate.score)
        
        return {
            "correct": correct_scores,
            "incorrect": incorrect_scores,
        }
    
    def generate_report(self, output_path: Path | None = None) -> str:
        """Generate human-readable evaluation report."""
        metrics = self.evaluate()
        failures = self.analyze_failures(5)
        
        lines = [
            "=" * 60,
            "MATCHING BENCHMARK REPORT",
            "=" * 60,
            "",
            "Precision@K:",
            f"  P@1:  {metrics.precision_at_1:.4f}",
            f"  P@3:  {metrics.precision_at_3:.4f}",
            f"  P@5:  {metrics.precision_at_5:.4f}",
            f"  P@10: {metrics.precision_at_10:.4f}",
            "",
            "Recall@K:",
            f"  R@1:  {metrics.recall_at_1:.4f}",
            f"  R@3:  {metrics.recall_at_3:.4f}",
            f"  R@5:  {metrics.recall_at_5:.4f}",
            f"  R@10: {metrics.recall_at_10:.4f}",
            "",
            "Hit Rate@K:",
            f"  HR@1:  {metrics.hit_rate_at_1:.4f}",
            f"  HR@3:  {metrics.hit_rate_at_3:.4f}",
            f"  HR@5:  {metrics.hit_rate_at_5:.4f}",
            f"  HR@10: {metrics.hit_rate_at_10:.4f}",
            "",
            "Ranking Metrics:",
            f"  MRR:     {metrics.mrr:.4f}",
            f"  NDCG@5:  {metrics.ndcg_at_5:.4f}",
            f"  NDCG@10: {metrics.ndcg_at_10:.4f}",
            "",
            "Counts:",
            f"  Total Queries:        {metrics.total_queries}",
            f"  Queries with Match:   {metrics.queries_with_match}",
            f"  Avg Candidates/Query: {metrics.avg_candidates_per_query:.1f}",
            f"  Avg First Correct Rank: {metrics.avg_first_correct_rank:.2f}",
            "",
        ]
        
        if metrics.per_category_metrics:
            lines.extend([
                "Per-Category Metrics:",
                "-" * 40,
            ])
            for cat, cat_metrics in metrics.per_category_metrics.items():
                lines.append(
                    f"  {cat} (n={cat_metrics['count']}): "
                    f"MRR={cat_metrics['mrr']:.3f}, P@1={cat_metrics['p@1']:.3f}, HR@5={cat_metrics['hr@5']:.3f}"
                )
        
        if failures:
            lines.extend([
                "",
                "Sample Failures (missed in top-5):",
                "-" * 40,
            ])
            for fail in failures:
                lines.extend([
                    f"  Query: {fail['query_text'][:40]}...",
                    f"    Expected: {fail['ground_truth']['title']}",
                    f"    Got Top-1: {fail['top_3_candidates'][0]['title'] if fail['top_3_candidates'] else 'None'}",
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
    num_queries: int = 100,
    candidates_per_query: int = 10,
    accuracy: float = 0.8,
) -> MatchingBenchmark:
    """
    Create a synthetic benchmark for testing the evaluation pipeline.
    """
    import random
    
    benchmark = MatchingBenchmark()
    
    books = [
        ("isbn:001", "The Great Gatsby", "F. Scott Fitzgerald"),
        ("isbn:002", "To Kill a Mockingbird", "Harper Lee"),
        ("isbn:003", "1984", "George Orwell"),
        ("isbn:004", "Pride and Prejudice", "Jane Austen"),
        ("isbn:005", "The Catcher in the Rye", "J.D. Salinger"),
        ("isbn:006", "Lord of the Flies", "William Golding"),
        ("isbn:007", "Animal Farm", "George Orwell"),
        ("isbn:008", "Brave New World", "Aldous Huxley"),
        ("isbn:009", "The Hobbit", "J.R.R. Tolkien"),
        ("isbn:010", "Fahrenheit 451", "Ray Bradbury"),
    ]
    
    categories = ["spine_ocr", "cover_ocr", "title_search", "author_search"]
    
    for i in range(num_queries):
        # Pick ground truth
        gt_id, gt_title, gt_author = random.choice(books)
        query_text = f"{gt_title} {gt_author}"
        
        # Generate candidates
        candidates = []
        correct_rank = 1 if random.random() < accuracy else random.randint(2, candidates_per_query)
        
        for j in range(candidates_per_query):
            if j + 1 == correct_rank:
                # Correct answer
                candidates.append({
                    "id": gt_id,
                    "title": gt_title,
                    "author": gt_author,
                    "score": 0.95 - j * 0.05 + random.gauss(0, 0.02),
                })
            else:
                # Wrong answer
                wrong = random.choice([b for b in books if b[0] != gt_id])
                candidates.append({
                    "id": wrong[0],
                    "title": wrong[1],
                    "author": wrong[2],
                    "score": 0.9 - j * 0.08 + random.gauss(0, 0.05),
                })
        
        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        benchmark.add_query(
            query_id=f"q_{i:04d}",
            query_text=query_text,
            ground_truth_id=gt_id,
            candidates=candidates,
            ground_truth_title=gt_title,
            ground_truth_author=gt_author,
            category=random.choice(categories),
        )
    
    return benchmark
