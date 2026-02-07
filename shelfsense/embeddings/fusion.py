"""
Embedding Fusion

Late fusion strategy for combining text and visual embeddings:
- Adaptive weighting based on OCR confidence
- Score-level fusion
- Fallback handling
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union
from loguru import logger

from shelfsense.embeddings.text_embedder import TextEmbeddingResult
from shelfsense.embeddings.visual_embedder import VisualEmbeddingResult


@dataclass
class FusedEmbedding:
    """
    Result of multimodal embedding fusion.
    
    Contains both individual embeddings and fusion metadata
    for interpretable matching.
    """
    
    # Individual embeddings
    text_embedding: Optional[np.ndarray] = None
    visual_embedding: Optional[np.ndarray] = None
    
    # Fusion weights used
    text_weight: float = 0.5
    visual_weight: float = 0.5
    
    # Source metadata
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    source_type: str = "unknown"  # "spine", "cover", "mixed"
    
    # Quality indicators
    has_text: bool = False
    has_visual: bool = True
    fusion_method: str = "late"  # "late", "text_only", "visual_only"
    
    # Derived metrics
    text_quality_score: float = 0.0
    visual_quality_score: float = 0.0
    
    @property
    def is_multimodal(self) -> bool:
        """Check if both modalities are present."""
        return self.has_text and self.has_visual
    
    @property
    def dominant_modality(self) -> str:
        """Get the dominant modality based on weights."""
        if not self.has_text:
            return "visual"
        if not self.has_visual:
            return "text"
        return "text" if self.text_weight >= self.visual_weight else "visual"
    
    def get_embedding(self, modality: str = "auto") -> Optional[np.ndarray]:
        """
        Get embedding for specified modality.
        
        Args:
            modality: "text", "visual", or "auto" (returns dominant)
            
        Returns:
            Embedding array or None
        """
        if modality == "auto":
            modality = self.dominant_modality
        
        if modality == "text":
            return self.text_embedding
        elif modality == "visual":
            return self.visual_embedding
        
        return None


@dataclass
class FusionScore:
    """Score from fused similarity computation."""
    
    combined_score: float
    text_score: float
    visual_score: float
    
    # Weights used
    text_weight: float
    visual_weight: float
    
    # Breakdown
    components: dict = field(default_factory=dict)
    
    @property
    def confidence(self) -> float:
        """
        Confidence in the combined score.
        
        Higher when both modalities agree.
        """
        if self.text_score == 0 or self.visual_score == 0:
            return min(self.combined_score, 0.7)  # Cap if single modality
        
        # Agreement bonus
        agreement = 1.0 - abs(self.text_score - self.visual_score)
        return self.combined_score * (0.8 + 0.2 * agreement)


class EmbeddingFusion:
    """
    Late fusion strategy for multimodal book matching.
    
    Design Decision: Late Fusion vs Early Fusion
    
    We use LATE FUSION (score combination) rather than early fusion
    (embedding concatenation) because:
    
    1. Independent embedding spaces: Text (Sentence-BERT) and visual (CLIP)
       embeddings live in different semantic spaces. Concatenation would
       require additional training to learn cross-modal relationships.
    
    2. Adaptive weighting: OCR confidence varies significantly. When OCR
       fails, we can gracefully fall back to visual-only matching without
       needing a separate model.
    
    3. Interpretability: We can examine text vs visual scores separately
       for debugging and explanation.
    
    4. Flexibility: Easy to adjust weights per-query or per-collection.
    
    Fusion Formula:
        score = α * sim_text + β * sim_visual
        
    Where:
        α = f(OCR_confidence), clamped to [0.3, 0.8]
        β = 1 - α
    
    Usage:
        fusion = EmbeddingFusion()
        fused = fusion.fuse(
            text_result, 
            visual_result, 
            ocr_confidence=0.85
        )
        
        # Match against candidates
        score = fusion.compute_similarity(
            fused, 
            candidate_text_emb, 
            candidate_visual_emb
        )
    """
    
    # Weight bounds
    MIN_TEXT_WEIGHT = 0.3
    MAX_TEXT_WEIGHT = 0.8
    
    # Confidence thresholds
    HIGH_OCR_CONFIDENCE = 0.8
    LOW_OCR_CONFIDENCE = 0.4
    
    def __init__(
        self,
        default_text_weight: float = 0.6,
        default_visual_weight: float = 0.4,
        adaptive_weighting: bool = True,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize embedding fusion.
        
        Args:
            default_text_weight: Default weight for text similarity
            default_visual_weight: Default weight for visual similarity
            adaptive_weighting: Adjust weights based on OCR confidence
            confidence_threshold: Below this, mark text as unreliable
        """
        self.default_text_weight = default_text_weight
        self.default_visual_weight = default_visual_weight
        self.adaptive_weighting = adaptive_weighting
        self.confidence_threshold = confidence_threshold
        
        logger.info(
            f"EmbeddingFusion initialized: "
            f"weights=[{default_text_weight}, {default_visual_weight}], "
            f"adaptive={adaptive_weighting}"
        )
    
    def fuse(
        self,
        text_result: Optional[TextEmbeddingResult],
        visual_result: Optional[VisualEmbeddingResult],
        ocr_confidence: float = 1.0,
        ocr_text: str = "",
        source_type: str = "unknown",
    ) -> FusedEmbedding:
        """
        Fuse text and visual embeddings.
        
        Args:
            text_result: Text embedding result (from OCR)
            visual_result: Visual embedding result (from image)
            ocr_confidence: OCR confidence score (0-1)
            ocr_text: Original OCR text for debugging
            source_type: "spine", "cover", or "unknown"
            
        Returns:
            FusedEmbedding with combined representation
        """
        # Determine what we have
        has_text = (
            text_result is not None 
            and text_result.embedding is not None
            and np.any(text_result.embedding != 0)
        )
        
        has_visual = (
            visual_result is not None
            and visual_result.embedding is not None
            and np.any(visual_result.embedding != 0)
        )
        
        # Calculate adaptive weights
        text_weight, visual_weight = self._compute_weights(
            ocr_confidence, has_text, has_visual, source_type
        )
        
        # Determine fusion method
        if has_text and has_visual:
            fusion_method = "late"
        elif has_text:
            fusion_method = "text_only"
            text_weight = 1.0
            visual_weight = 0.0
        elif has_visual:
            fusion_method = "visual_only"
            text_weight = 0.0
            visual_weight = 1.0
        else:
            fusion_method = "none"
        
        # Compute quality scores
        text_quality = self._compute_text_quality(text_result, ocr_confidence)
        visual_quality = self._compute_visual_quality(visual_result)
        
        return FusedEmbedding(
            text_embedding=text_result.embedding if has_text else None,
            visual_embedding=visual_result.embedding if has_visual else None,
            text_weight=text_weight,
            visual_weight=visual_weight,
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            source_type=source_type,
            has_text=has_text,
            has_visual=has_visual,
            fusion_method=fusion_method,
            text_quality_score=text_quality,
            visual_quality_score=visual_quality,
        )
    
    def _compute_weights(
        self,
        ocr_confidence: float,
        has_text: bool,
        has_visual: bool,
        source_type: str,
    ) -> tuple[float, float]:
        """
        Compute adaptive weights based on OCR confidence.
        
        Returns:
            Tuple of (text_weight, visual_weight)
        """
        if not self.adaptive_weighting:
            return self.default_text_weight, self.default_visual_weight
        
        # Base case: no text
        if not has_text:
            return 0.0, 1.0
        
        # Base case: no visual
        if not has_visual:
            return 1.0, 0.0
        
        # Adaptive weighting based on OCR confidence
        # High confidence -> more text weight
        # Low confidence -> more visual weight
        
        if ocr_confidence >= self.HIGH_OCR_CONFIDENCE:
            text_weight = self.MAX_TEXT_WEIGHT
        elif ocr_confidence <= self.LOW_OCR_CONFIDENCE:
            text_weight = self.MIN_TEXT_WEIGHT
        else:
            # Linear interpolation
            ratio = (ocr_confidence - self.LOW_OCR_CONFIDENCE) / (
                self.HIGH_OCR_CONFIDENCE - self.LOW_OCR_CONFIDENCE
            )
            text_weight = (
                self.MIN_TEXT_WEIGHT + 
                ratio * (self.MAX_TEXT_WEIGHT - self.MIN_TEXT_WEIGHT)
            )
        
        # Source-type adjustments
        if source_type == "spine":
            # Spines often have hard-to-read text
            text_weight *= 0.9
        elif source_type == "cover":
            # Covers have more visual information
            text_weight *= 0.95
        
        # Clamp to bounds
        text_weight = max(self.MIN_TEXT_WEIGHT, min(self.MAX_TEXT_WEIGHT, text_weight))
        visual_weight = 1.0 - text_weight
        
        return text_weight, visual_weight
    
    def _compute_text_quality(
        self,
        text_result: Optional[TextEmbeddingResult],
        ocr_confidence: float,
    ) -> float:
        """Compute quality score for text embedding."""
        if text_result is None:
            return 0.0
        
        quality = ocr_confidence
        
        # Penalty for truncation
        if text_result.was_truncated:
            quality *= 0.9
        
        # Penalty for short text
        if text_result.text and len(text_result.text) < 5:
            quality *= 0.7
        
        return quality
    
    def _compute_visual_quality(
        self,
        visual_result: Optional[VisualEmbeddingResult],
    ) -> float:
        """Compute quality score for visual embedding."""
        if visual_result is None:
            return 0.0
        
        quality = 1.0
        
        # Small images might have less information
        if visual_result.original_size:
            w, h = visual_result.original_size
            if min(w, h) < 50:
                quality *= 0.7
            elif min(w, h) < 100:
                quality *= 0.85
        
        return quality
    
    def compute_similarity(
        self,
        query: FusedEmbedding,
        candidate_text_emb: Optional[np.ndarray],
        candidate_visual_emb: Optional[np.ndarray],
    ) -> FusionScore:
        """
        Compute fused similarity between query and candidate.
        
        Args:
            query: Query fused embedding
            candidate_text_emb: Candidate text embedding
            candidate_visual_emb: Candidate visual embedding
            
        Returns:
            FusionScore with combined and component scores
        """
        # Compute individual similarities
        text_score = 0.0
        visual_score = 0.0
        
        if query.has_text and candidate_text_emb is not None:
            text_score = self._cosine_similarity(
                query.text_embedding, candidate_text_emb
            )
        
        if query.has_visual and candidate_visual_emb is not None:
            visual_score = self._cosine_similarity(
                query.visual_embedding, candidate_visual_emb
            )
        
        # Compute combined score
        combined = (
            query.text_weight * text_score + 
            query.visual_weight * visual_score
        )
        
        return FusionScore(
            combined_score=combined,
            text_score=text_score,
            visual_score=visual_score,
            text_weight=query.text_weight,
            visual_weight=query.visual_weight,
            components={
                "text_contribution": query.text_weight * text_score,
                "visual_contribution": query.visual_weight * visual_score,
            }
        )
    
    def compute_similarity_batch(
        self,
        query: FusedEmbedding,
        candidate_text_embs: Optional[np.ndarray],  # Shape: (N, D_text)
        candidate_visual_embs: Optional[np.ndarray],  # Shape: (N, D_visual)
    ) -> list[FusionScore]:
        """
        Compute fused similarities for multiple candidates efficiently.
        
        Args:
            query: Query fused embedding
            candidate_text_embs: Array of candidate text embeddings
            candidate_visual_embs: Array of candidate visual embeddings
            
        Returns:
            List of FusionScore objects
        """
        n_candidates = 0
        if candidate_text_embs is not None:
            n_candidates = len(candidate_text_embs)
        elif candidate_visual_embs is not None:
            n_candidates = len(candidate_visual_embs)
        
        if n_candidates == 0:
            return []
        
        # Compute batch similarities
        text_scores = np.zeros(n_candidates)
        visual_scores = np.zeros(n_candidates)
        
        if query.has_text and candidate_text_embs is not None:
            # Batch cosine similarity
            text_scores = self._batch_cosine_similarity(
                query.text_embedding, candidate_text_embs
            )
        
        if query.has_visual and candidate_visual_embs is not None:
            visual_scores = self._batch_cosine_similarity(
                query.visual_embedding, candidate_visual_embs
            )
        
        # Compute combined scores
        combined_scores = (
            query.text_weight * text_scores + 
            query.visual_weight * visual_scores
        )
        
        # Build results
        results = []
        for i in range(n_candidates):
            results.append(FusionScore(
                combined_score=float(combined_scores[i]),
                text_score=float(text_scores[i]),
                visual_score=float(visual_scores[i]),
                text_weight=query.text_weight,
                visual_weight=query.visual_weight,
                components={
                    "text_contribution": query.text_weight * float(text_scores[i]),
                    "visual_contribution": query.visual_weight * float(visual_scores[i]),
                }
            ))
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # Assuming normalized embeddings
        return float(np.dot(a, b))
    
    def _batch_cosine_similarity(
        self, 
        query: np.ndarray, 
        candidates: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarities for batch of candidates."""
        # query: (D,), candidates: (N, D)
        # Assuming normalized embeddings
        return np.dot(candidates, query)
    
    def rank_candidates(
        self,
        query: FusedEmbedding,
        candidates: list[tuple[np.ndarray, np.ndarray]],  # List of (text_emb, visual_emb)
        top_k: int = 10,
    ) -> list[tuple[int, FusionScore]]:
        """
        Rank candidates by fused similarity.
        
        Args:
            query: Query fused embedding
            candidates: List of (text_emb, visual_emb) tuples
            top_k: Number of results to return
            
        Returns:
            List of (index, FusionScore) tuples, sorted by score
        """
        # Compute all scores
        scores = []
        for i, (text_emb, visual_emb) in enumerate(candidates):
            score = self.compute_similarity(query, text_emb, visual_emb)
            scores.append((i, score))
        
        # Sort by combined score
        scores.sort(key=lambda x: x[1].combined_score, reverse=True)
        
        return scores[:top_k]


class CrossModalMatcher:
    """
    Specialized matcher for cross-modal scenarios.
    
    Handles cases where:
    - We have only visual (cover image) and need to match to text database
    - We have only text (OCR) and need to match to visual database
    """
    
    def __init__(
        self,
        visual_embedder=None,  # VisualEmbedder
        text_embedder=None,    # TextEmbedder
    ):
        """
        Initialize cross-modal matcher.
        
        Args:
            visual_embedder: VisualEmbedder instance (for text-to-visual via CLIP)
            text_embedder: TextEmbedder instance (for text matching)
        """
        self.visual_embedder = visual_embedder
        self.text_embedder = text_embedder
    
    def match_image_to_text_db(
        self,
        image_embedding: np.ndarray,
        text_descriptions: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Match image against text descriptions using CLIP.
        
        CLIP's text encoder produces embeddings in the same space
        as image embeddings, enabling direct comparison.
        
        Args:
            image_embedding: Visual embedding from CLIP
            text_descriptions: List of text descriptions
            top_k: Number of results
            
        Returns:
            List of (index, similarity) tuples
        """
        if self.visual_embedder is None:
            raise ValueError("VisualEmbedder required for cross-modal matching")
        
        # Embed texts using CLIP's text encoder
        text_embeddings = [
            self.visual_embedder.embed_text(text)
            for text in text_descriptions
        ]
        
        # Compute similarities
        similarities = []
        for i, text_emb in enumerate(text_embeddings):
            sim = float(np.dot(image_embedding, text_emb))
            similarities.append((i, sim))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def match_text_to_image_db(
        self,
        text: str,
        image_embeddings: np.ndarray,  # Shape: (N, D)
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Match text against image database using CLIP.
        
        Args:
            text: Query text (book title, description)
            image_embeddings: Array of visual embeddings
            top_k: Number of results
            
        Returns:
            List of (index, similarity) tuples
        """
        if self.visual_embedder is None:
            raise ValueError("VisualEmbedder required for cross-modal matching")
        
        # Embed text using CLIP's text encoder
        text_embedding = self.visual_embedder.embed_text(text)
        
        # Batch similarity
        similarities = np.dot(image_embeddings, text_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(i), float(similarities[i])) for i in top_indices]
