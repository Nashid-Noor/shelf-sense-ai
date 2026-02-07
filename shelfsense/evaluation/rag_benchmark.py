"""
RAG Benchmark Module

Evaluates RAG (Retrieval-Augmented Generation) performance:
- Retrieval metrics (relevance, coverage)
- Generation quality (faithfulness, groundedness)
- Answer relevance
- Citation accuracy
- Hallucination detection
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RelevanceLevel(Enum):
    """Document relevance levels."""
    HIGHLY_RELEVANT = 3
    RELEVANT = 2
    PARTIALLY_RELEVANT = 1
    NOT_RELEVANT = 0


@dataclass
class RetrievedDocument:
    """A document retrieved by the RAG system."""
    doc_id: str
    title: str
    content: str
    score: float = 0.0
    rank: int = 0
    relevance: RelevanceLevel = RelevanceLevel.NOT_RELEVANT
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGQuery:
    """A single RAG evaluation query."""
    query_id: str
    question: str
    retrieved_docs: list[RetrievedDocument] = field(default_factory=list)
    generated_answer: str = ""
    reference_answer: str = ""  # Gold standard answer
    expected_doc_ids: set[str] = field(default_factory=set)  # Docs that should be retrieved
    citations: list[str] = field(default_factory=list)  # Cited doc IDs in answer
    category: str = "general"
    
    # Human/LLM evaluation scores (0-1 scale)
    answer_relevance_score: float | None = None
    faithfulness_score: float | None = None
    groundedness_score: float | None = None


@dataclass
class RAGMetrics:
    """RAG evaluation metrics."""
    # Retrieval metrics
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    retrieval_f1: float = 0.0
    retrieval_mrr: float = 0.0
    retrieval_ndcg: float = 0.0
    
    # Generation metrics
    avg_answer_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    avg_groundedness: float = 0.0
    
    # Citation metrics
    citation_precision: float = 0.0  # Cited docs that are relevant
    citation_recall: float = 0.0  # Relevant docs that are cited
    citation_f1: float = 0.0
    
    # Answer quality
    answer_coverage: float = 0.0  # How much of reference answer is covered
    hallucination_rate: float = 0.0  # Claims not supported by retrieved docs
    
    # Aggregates
    total_queries: int = 0
    avg_docs_retrieved: float = 0.0
    avg_answer_length: float = 0.0
    
    per_category_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval": {
                "precision": round(self.retrieval_precision, 4),
                "recall": round(self.retrieval_recall, 4),
                "f1": round(self.retrieval_f1, 4),
                "mrr": round(self.retrieval_mrr, 4),
                "ndcg": round(self.retrieval_ndcg, 4),
            },
            "generation": {
                "answer_relevance": round(self.avg_answer_relevance, 4),
                "faithfulness": round(self.avg_faithfulness, 4),
                "groundedness": round(self.avg_groundedness, 4),
            },
            "citation": {
                "precision": round(self.citation_precision, 4),
                "recall": round(self.citation_recall, 4),
                "f1": round(self.citation_f1, 4),
            },
            "quality": {
                "answer_coverage": round(self.answer_coverage, 4),
                "hallucination_rate": round(self.hallucination_rate, 4),
            },
            "counts": {
                "total_queries": self.total_queries,
                "avg_docs_retrieved": round(self.avg_docs_retrieved, 2),
                "avg_answer_length": round(self.avg_answer_length, 2),
            },
            "per_category_metrics": self.per_category_metrics,
        }


class RAGBenchmark:
    """
    Benchmark suite for RAG evaluation.
    
    Evaluates both retrieval quality and generation quality,
    including faithfulness, groundedness, and citation accuracy.
    """
    
    def __init__(
        self,
        relevance_threshold: int = 2,  # Min relevance level to count as relevant
        use_llm_judge: bool = False,
        llm_judge: Any | None = None,
    ):
        self.relevance_threshold = relevance_threshold
        self.use_llm_judge = use_llm_judge
        self.llm_judge = llm_judge
        self.queries: list[RAGQuery] = []
    
    def add_query(
        self,
        query_id: str,
        question: str,
        retrieved_docs: list[dict[str, Any]],
        generated_answer: str,
        reference_answer: str = "",
        expected_doc_ids: set[str] | None = None,
        citations: list[str] | None = None,
        category: str = "general",
        relevance_judgments: dict[str, int] | None = None,
    ) -> None:
        """
        Add a RAG query for evaluation.
        
        Args:
            relevance_judgments: Dict mapping doc_id to relevance level (0-3)
        """
        docs = []
        for i, doc in enumerate(retrieved_docs):
            doc_id = doc.get("id", doc.get("doc_id", f"doc_{i}"))
            relevance = RelevanceLevel.NOT_RELEVANT
            if relevance_judgments and doc_id in relevance_judgments:
                relevance = RelevanceLevel(relevance_judgments[doc_id])
            
            docs.append(RetrievedDocument(
                doc_id=doc_id,
                title=doc.get("title", ""),
                content=doc.get("content", doc.get("text", "")),
                score=doc.get("score", 0.0),
                rank=i + 1,
                relevance=relevance,
                metadata=doc.get("metadata", {}),
            ))
        
        # Extract citations from answer if not provided
        if citations is None:
            citations = self._extract_citations(generated_answer, [d.doc_id for d in docs])
        
        self.queries.append(RAGQuery(
            query_id=query_id,
            question=question,
            retrieved_docs=docs,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
            expected_doc_ids=expected_doc_ids or set(),
            citations=citations,
            category=category,
        ))
    
    def _extract_citations(self, answer: str, doc_ids: list[str]) -> list[str]:
        """Extract cited document IDs from answer text."""
        citations = []
        
        # Look for [1], [2], etc. style citations
        bracket_refs = re.findall(r'\[(\d+)\]', answer)
        for ref in bracket_refs:
            idx = int(ref) - 1
            if 0 <= idx < len(doc_ids):
                citations.append(doc_ids[idx])
        
        # Look for explicit doc ID references
        for doc_id in doc_ids:
            if doc_id in answer:
                citations.append(doc_id)
        
        return list(set(citations))
    
    def load_dataset(
        self,
        queries_file: Path,
        rag_system: Any | None = None,
    ) -> None:
        """
        Load evaluation dataset and optionally run RAG.
        
        Expected format:
        [
            {
                "query_id": "q_001",
                "question": "What themes are explored in 1984?",
                "reference_answer": "The novel explores themes of...",
                "expected_doc_ids": ["book_1984"],
                "relevance_judgments": {"book_1984": 3, "book_brave_new_world": 2},
                "category": "theme_analysis"
            },
            ...
        ]
        """
        with open(queries_file) as f:
            data = json.load(f)
        
        for item in data:
            if rag_system:
                # Run RAG system
                result = rag_system.query(item["question"])
                retrieved_docs = result.get("retrieved_docs", [])
                generated_answer = result.get("answer", "")
            else:
                retrieved_docs = item.get("retrieved_docs", [])
                generated_answer = item.get("generated_answer", "")
            
            self.add_query(
                query_id=item["query_id"],
                question=item["question"],
                retrieved_docs=retrieved_docs,
                generated_answer=generated_answer,
                reference_answer=item.get("reference_answer", ""),
                expected_doc_ids=set(item.get("expected_doc_ids", [])),
                citations=item.get("citations"),
                category=item.get("category", "general"),
                relevance_judgments=item.get("relevance_judgments"),
            )
    
    def evaluate(self) -> RAGMetrics:
        """Run full evaluation and compute all metrics."""
        if not self.queries:
            logger.warning("No queries to evaluate")
            return RAGMetrics()
        
        metrics = RAGMetrics()
        metrics.total_queries = len(self.queries)
        
        # Collect per-query metrics
        retrieval_precisions = []
        retrieval_recalls = []
        retrieval_mrrs = []
        retrieval_ndcgs = []
        
        citation_precisions = []
        citation_recalls = []
        
        answer_lengths = []
        doc_counts = []
        
        answer_relevances = []
        faithfulness_scores = []
        groundedness_scores = []
        hallucination_counts = []
        
        category_data: dict[str, list[RAGQuery]] = {}
        
        for query in self.queries:
            # Retrieval metrics
            relevant_docs = [d for d in query.retrieved_docs if d.relevance.value >= self.relevance_threshold]
            retrieved_ids = {d.doc_id for d in query.retrieved_docs}
            relevant_ids = {d.doc_id for d in relevant_docs}
            
            # Precision = relevant retrieved / total retrieved
            if query.retrieved_docs:
                r_precision = len(relevant_docs) / len(query.retrieved_docs)
            else:
                r_precision = 0.0
            retrieval_precisions.append(r_precision)
            
            # Recall = relevant retrieved / total relevant (expected)
            if query.expected_doc_ids:
                r_recall = len(retrieved_ids & query.expected_doc_ids) / len(query.expected_doc_ids)
            else:
                r_recall = 1.0 if relevant_docs else 0.0  # If no expected docs defined
            retrieval_recalls.append(r_recall)
            
            # MRR
            mrr = 0.0
            for i, doc in enumerate(query.retrieved_docs):
                if doc.relevance.value >= self.relevance_threshold:
                    mrr = 1.0 / (i + 1)
                    break
            retrieval_mrrs.append(mrr)
            
            # NDCG
            ndcg = self._compute_ndcg(query.retrieved_docs)
            retrieval_ndcgs.append(ndcg)
            
            # Citation metrics
            cited_ids = set(query.citations)
            if cited_ids:
                cited_relevant = len(cited_ids & relevant_ids)
                c_precision = cited_relevant / len(cited_ids)
            else:
                c_precision = 0.0
            citation_precisions.append(c_precision)
            
            if relevant_ids:
                c_recall = len(cited_ids & relevant_ids) / len(relevant_ids)
            else:
                c_recall = 1.0 if not cited_ids else 0.0
            citation_recalls.append(c_recall)
            
            # Answer stats
            answer_lengths.append(len(query.generated_answer.split()))
            doc_counts.append(len(query.retrieved_docs))
            
            # LLM judge scores (if available)
            if query.answer_relevance_score is not None:
                answer_relevances.append(query.answer_relevance_score)
            if query.faithfulness_score is not None:
                faithfulness_scores.append(query.faithfulness_score)
            if query.groundedness_score is not None:
                groundedness_scores.append(query.groundedness_score)
            
            # Simple hallucination detection (claims not in retrieved docs)
            hallucination_rate = self._estimate_hallucination_rate(
                query.generated_answer,
                [d.content for d in query.retrieved_docs]
            )
            hallucination_counts.append(hallucination_rate)
            
            # Category grouping
            if query.category not in category_data:
                category_data[query.category] = []
            category_data[query.category].append(query)
        
        # Aggregate metrics
        metrics.retrieval_precision = np.mean(retrieval_precisions)
        metrics.retrieval_recall = np.mean(retrieval_recalls)
        if metrics.retrieval_precision + metrics.retrieval_recall > 0:
            metrics.retrieval_f1 = 2 * metrics.retrieval_precision * metrics.retrieval_recall / \
                                   (metrics.retrieval_precision + metrics.retrieval_recall)
        metrics.retrieval_mrr = np.mean(retrieval_mrrs)
        metrics.retrieval_ndcg = np.mean(retrieval_ndcgs)
        
        if citation_precisions:
            metrics.citation_precision = np.mean(citation_precisions)
        if citation_recalls:
            metrics.citation_recall = np.mean(citation_recalls)
        if metrics.citation_precision + metrics.citation_recall > 0:
            metrics.citation_f1 = 2 * metrics.citation_precision * metrics.citation_recall / \
                                  (metrics.citation_precision + metrics.citation_recall)
        
        if answer_relevances:
            metrics.avg_answer_relevance = np.mean(answer_relevances)
        if faithfulness_scores:
            metrics.avg_faithfulness = np.mean(faithfulness_scores)
        if groundedness_scores:
            metrics.avg_groundedness = np.mean(groundedness_scores)
        
        metrics.hallucination_rate = np.mean(hallucination_counts)
        metrics.avg_docs_retrieved = np.mean(doc_counts)
        metrics.avg_answer_length = np.mean(answer_lengths)
        
        # Per-category metrics
        for category, cat_queries in category_data.items():
            cat_relevant = []
            for q in cat_queries:
                rel_count = sum(1 for d in q.retrieved_docs if d.relevance.value >= self.relevance_threshold)
                cat_relevant.append(rel_count / len(q.retrieved_docs) if q.retrieved_docs else 0)
            
            metrics.per_category_metrics[category] = {
                "retrieval_precision": round(np.mean(cat_relevant), 4),
                "count": len(cat_queries),
            }
        
        return metrics
    
    def _compute_ndcg(self, docs: list[RetrievedDocument], k: int = 10) -> float:
        """Compute NDCG for retrieved documents."""
        docs = docs[:k]
        
        def dcg(relevances: list[int]) -> float:
            return sum(
                rel / np.log2(i + 2)
                for i, rel in enumerate(relevances)
            )
        
        relevances = [d.relevance.value for d in docs]
        ideal_relevances = sorted(relevances, reverse=True)
        
        dcg_score = dcg(relevances)
        idcg_score = dcg(ideal_relevances)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0
    
    def _estimate_hallucination_rate(
        self,
        answer: str,
        doc_contents: list[str],
    ) -> float:
        """
        Estimate hallucination rate using simple heuristics.
        
        # Ideally use an LLM judge here

        """
        if not answer or not doc_contents:
            return 0.0
        
        # Combine all retrieved content
        all_content = " ".join(doc_contents).lower()
        
        # Extract "factual" claims from answer (simple heuristic: sentences with entities)
        sentences = re.split(r'[.!?]', answer)
        unsupported_count = 0
        total_claims = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Check if key words from sentence appear in context
            words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
            common_words = {'this', 'that', 'which', 'where', 'when', 'what', 'have', 'been', 'were', 'will', 'would', 'could', 'should', 'there', 'their', 'about'}
            content_words = words - common_words
            
            if content_words:
                total_claims += 1
                # Check overlap with context
                context_words = set(re.findall(r'\b\w{4,}\b', all_content))
                overlap = len(content_words & context_words) / len(content_words)
                
                if overlap < 0.3:  # Less than 30% overlap
                    unsupported_count += 1
        
        return unsupported_count / total_claims if total_claims > 0 else 0.0
    
    async def run_llm_evaluation(self) -> None:
        """
        Run LLM-based evaluation for answer quality metrics.
        
        Requires llm_judge to be set.
        """
        if not self.llm_judge:
            logger.warning("No LLM judge configured")
            return
        
        for query in self.queries:
            # Evaluate answer relevance
            relevance_prompt = f"""
Rate how well this answer addresses the question on a scale of 0-1.

Question: {query.question}
Answer: {query.generated_answer}

Provide only a number between 0 and 1.
"""
            relevance_response = await self.llm_judge.generate(relevance_prompt)
            try:
                query.answer_relevance_score = float(relevance_response.strip())
            except ValueError:
                pass
            
            # Evaluate faithfulness
            context = "\n\n".join([d.content for d in query.retrieved_docs])
            faithfulness_prompt = f"""
Rate how faithful this answer is to the provided context on a scale of 0-1.
0 = contains claims not supported by context
1 = all claims are supported by context

Context:
{context[:2000]}

Answer: {query.generated_answer}

Provide only a number between 0 and 1.
"""
            faithfulness_response = await self.llm_judge.generate(faithfulness_prompt)
            try:
                query.faithfulness_score = float(faithfulness_response.strip())
            except ValueError:
                pass
    
    def generate_report(self, output_path: Path | None = None) -> str:
        """Generate human-readable evaluation report."""
        metrics = self.evaluate()
        
        lines = [
            "=" * 60,
            "RAG BENCHMARK REPORT",
            "=" * 60,
            "",
            "Retrieval Metrics:",
            f"  Precision:  {metrics.retrieval_precision:.4f}",
            f"  Recall:     {metrics.retrieval_recall:.4f}",
            f"  F1:         {metrics.retrieval_f1:.4f}",
            f"  MRR:        {metrics.retrieval_mrr:.4f}",
            f"  NDCG:       {metrics.retrieval_ndcg:.4f}",
            "",
            "Citation Metrics:",
            f"  Precision:  {metrics.citation_precision:.4f}",
            f"  Recall:     {metrics.citation_recall:.4f}",
            f"  F1:         {metrics.citation_f1:.4f}",
            "",
            "Generation Quality:",
            f"  Answer Relevance:  {metrics.avg_answer_relevance:.4f}",
            f"  Faithfulness:      {metrics.avg_faithfulness:.4f}",
            f"  Groundedness:      {metrics.avg_groundedness:.4f}",
            f"  Hallucination Rate: {metrics.hallucination_rate:.4f}",
            "",
            "Statistics:",
            f"  Total Queries:       {metrics.total_queries}",
            f"  Avg Docs Retrieved:  {metrics.avg_docs_retrieved:.1f}",
            f"  Avg Answer Length:   {metrics.avg_answer_length:.0f} words",
            "",
        ]
        
        if metrics.per_category_metrics:
            lines.extend([
                "Per-Category Retrieval Precision:",
                "-" * 40,
            ])
            for cat, cat_metrics in metrics.per_category_metrics.items():
                lines.append(f"  {cat} (n={cat_metrics['count']}): {cat_metrics['retrieval_precision']:.4f}")
        
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report)
            json_path = output_path.with_suffix(".json")
            json_path.write_text(json.dumps(metrics.to_dict(), indent=2))
            logger.info(f"Report saved to {output_path}")
        
        return report


def create_synthetic_benchmark(
    num_queries: int = 50,
    docs_per_query: int = 5,
    relevance_rate: float = 0.6,
) -> RAGBenchmark:
    """
    Create a synthetic benchmark for testing the evaluation pipeline.
    """
    import random
    
    benchmark = RAGBenchmark()
    
    categories = ["theme_analysis", "character_discussion", "plot_summary", "recommendation", "comparison"]
    
    sample_questions = [
        "What are the main themes in this book?",
        "How does the protagonist change throughout the story?",
        "What is the significance of the title?",
        "How does this book compare to other works by the same author?",
        "What makes this book a classic?",
    ]
    
    sample_contents = [
        "The novel explores themes of identity, belonging, and the search for meaning in a chaotic world.",
        "Through careful character development, the author shows how experiences shape personality.",
        "The setting plays a crucial role in establishing the mood and atmosphere of the narrative.",
        "Critics have praised the book for its innovative structure and compelling prose.",
        "The historical context provides important background for understanding the characters' motivations.",
    ]
    
    for i in range(num_queries):
        question = random.choice(sample_questions)
        
        # Generate retrieved docs
        docs = []
        relevance_judgments = {}
        expected_doc_ids = set()
        
        for j in range(docs_per_query):
            doc_id = f"doc_{i}_{j}"
            is_relevant = random.random() < relevance_rate
            
            relevance = random.choice([2, 3]) if is_relevant else random.choice([0, 1])
            relevance_judgments[doc_id] = relevance
            
            if relevance >= 2:
                expected_doc_ids.add(doc_id)
            
            docs.append({
                "id": doc_id,
                "title": f"Book {j + 1}",
                "content": random.choice(sample_contents),
                "score": 0.9 - j * 0.1 + random.gauss(0, 0.05),
            })
        
        # Generate answer
        answer = "Based on the retrieved information, " + random.choice(sample_contents)
        citations = [docs[0]["id"]] if docs else []
        
        benchmark.add_query(
            query_id=f"q_{i:04d}",
            question=question,
            retrieved_docs=docs,
            generated_answer=answer,
            reference_answer=random.choice(sample_contents),
            expected_doc_ids=expected_doc_ids,
            citations=citations,
            category=random.choice(categories),
            relevance_judgments=relevance_judgments,
        )
    
    return benchmark
