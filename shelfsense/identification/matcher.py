"""
Book Matcher

Semantic matching of detected books against reference database:
- Vector similarity search (FAISS)
- Multi-modal score fusion
- Candidate verification
- Confidence-based ranking
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union
from loguru import logger

from shelfsense.embeddings.fusion import FusedEmbedding, EmbeddingFusion, FusionScore


@dataclass
class MatchCandidate:
    """Single match candidate."""
    
    # Identifiers
    book_id: str
    title: str
    author: str
    
    # Scores
    similarity_score: float
    text_score: float
    visual_score: float
    
    # Confidence
    confidence: float
    match_quality: str  # "high", "medium", "low", "uncertain"
    
    # Additional metadata
    isbn: Optional[str] = None
    cover_url: Optional[str] = None
    publication_year: Optional[int] = None
    
    # Match details
    matched_via: str = "fusion"  # "fusion", "text_only", "visual_only"
    rank: int = 0


@dataclass
class MatchResult:
    """
    Result of book matching operation.
    
    Contains ranked candidates and match metadata.
    """
    
    # Query info
    query_text: str
    query_source: str  # "spine", "cover", "mixed"
    
    # Results
    candidates: list[MatchCandidate] = field(default_factory=list)
    
    # Best match (convenience)
    @property
    def best_match(self) -> Optional[MatchCandidate]:
        if self.candidates:
            return self.candidates[0]
        return None
    
    @property
    def is_confident(self) -> bool:
        """Check if we have a confident match."""
        if not self.candidates:
            return False
        return self.candidates[0].confidence >= 0.7
    
    @property
    def needs_verification(self) -> bool:
        """Check if match needs human verification."""
        if not self.candidates:
            return True
        best = self.candidates[0]
        # Needs verification if low confidence or close second match
        if best.confidence < 0.6:
            return True
        if len(self.candidates) > 1:
            gap = best.similarity_score - self.candidates[1].similarity_score
            if gap < 0.1:
                return True
        return False
    
    # Diagnostics
    search_time_ms: float = 0.0
    total_candidates_searched: int = 0


class BookMatcher:
    """
    Production book matcher using vector similarity search.
    
    Architecture:
    1. Query embedding is compared against pre-indexed book embeddings
    2. Top-k candidates retrieved via FAISS approximate search
    3. Candidates re-ranked using multi-modal fusion scores
    4. Confidence scoring determines match quality
    
    Usage:
        matcher = BookMatcher(index_path="books.faiss")
        result = matcher.match(fused_embedding, top_k=10)
        print(result.best_match.title)
    """
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.65
    LOW_CONFIDENCE = 0.45
    
    def __init__(
        self,
        text_index=None,      # FAISS index for text embeddings
        visual_index=None,    # FAISS index for visual embeddings  
        book_metadata=None,   # Mapping of book_id -> metadata
        fusion: Optional[EmbeddingFusion] = None,
        rerank_factor: int = 3,  # Retrieve rerank_factor * top_k candidates
    ):
        """
        Initialize book matcher.
        
        Args:
            text_index: FAISS index for text embeddings
            visual_index: FAISS index for visual embeddings
            book_metadata: Dict or callable returning book metadata
            fusion: EmbeddingFusion instance for score combination
            rerank_factor: Over-retrieve by this factor before reranking
        """
        self.text_index = text_index
        self.visual_index = visual_index
        self.book_metadata = book_metadata or {}
        self.fusion = fusion or EmbeddingFusion()
        self.rerank_factor = rerank_factor
        
        logger.info("BookMatcher initialized")
    
    def match(
        self,
        query: FusedEmbedding,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> MatchResult:
        """
        Match query embedding against book database.
        
        Args:
            query: Fused embedding from detection
            top_k: Number of results to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            MatchResult with ranked candidates
        """
        import time
        start = time.perf_counter()
        
        # Determine search strategy based on available modalities
        if query.fusion_method == "text_only":
            candidates = self._search_text_only(query, top_k)
        elif query.fusion_method == "visual_only":
            candidates = self._search_visual_only(query, top_k)
        else:
            candidates = self._search_multimodal(query, top_k)
        
        # Filter by confidence
        if min_confidence > 0:
            candidates = [c for c in candidates if c.confidence >= min_confidence]
        
        # Assign ranks
        for i, candidate in enumerate(candidates):
            candidate.rank = i + 1
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return MatchResult(
            query_text=query.ocr_text,
            query_source=query.source_type,
            candidates=candidates,
            search_time_ms=elapsed,
            total_candidates_searched=len(candidates) * self.rerank_factor,
        )
    
    def _search_text_only(
        self,
        query: FusedEmbedding,
        top_k: int,
    ) -> list[MatchCandidate]:
        """Search using only text embedding."""
        if self.text_index is None or query.text_embedding is None:
            return []
        
        # Search FAISS index
        k = min(top_k * self.rerank_factor, self.text_index.ntotal)
        
        distances, indices = self.text_index.search(
            query.text_embedding.reshape(1, -1).astype(np.float32),
            k
        )
        
        # Build candidates
        candidates = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # Invalid index
                continue
            
            # Convert distance to similarity (for L2 normalized vectors)
            similarity = 1 - dist / 2  # Convert L2 to cosine-like
            
            metadata = self._get_metadata(idx)
            confidence = self._compute_confidence(
                similarity, 0.0, "text_only", query
            )
            
            candidates.append(MatchCandidate(
                book_id=str(idx),
                title=metadata.get("title", f"Book {idx}"),
                author=metadata.get("author", "Unknown"),
                similarity_score=similarity,
                text_score=similarity,
                visual_score=0.0,
                confidence=confidence,
                match_quality=self._quality_label(confidence),
                isbn=metadata.get("isbn"),
                cover_url=metadata.get("cover_url"),
                publication_year=metadata.get("year"),
                matched_via="text_only",
            ))
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates[:top_k]
    
    def _search_visual_only(
        self,
        query: FusedEmbedding,
        top_k: int,
    ) -> list[MatchCandidate]:
        """Search using only visual embedding."""
        if self.visual_index is None or query.visual_embedding is None:
            return []
        
        k = min(top_k * self.rerank_factor, self.visual_index.ntotal)
        
        distances, indices = self.visual_index.search(
            query.visual_embedding.reshape(1, -1).astype(np.float32),
            k
        )
        
        candidates = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                continue
            
            similarity = 1 - dist / 2
            metadata = self._get_metadata(idx)
            confidence = self._compute_confidence(
                0.0, similarity, "visual_only", query
            )
            
            candidates.append(MatchCandidate(
                book_id=str(idx),
                title=metadata.get("title", f"Book {idx}"),
                author=metadata.get("author", "Unknown"),
                similarity_score=similarity,
                text_score=0.0,
                visual_score=similarity,
                confidence=confidence,
                match_quality=self._quality_label(confidence),
                isbn=metadata.get("isbn"),
                cover_url=metadata.get("cover_url"),
                publication_year=metadata.get("year"),
                matched_via="visual_only",
            ))
        
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[:top_k]
    
    def _search_multimodal(
        self,
        query: FusedEmbedding,
        top_k: int,
    ) -> list[MatchCandidate]:
        """
        Search using both modalities with fusion.
        
        Strategy:
        1. Retrieve candidates from both indices
        2. Combine and deduplicate
        3. Rerank using fusion scores
        """
        # Get candidates from both modalities
        text_candidates = {}
        visual_candidates = {}
        
        if self.text_index is not None and query.text_embedding is not None:
            k = min(top_k * self.rerank_factor, self.text_index.ntotal)
            distances, indices = self.text_index.search(
                query.text_embedding.reshape(1, -1).astype(np.float32),
                k
            )
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    text_candidates[int(idx)] = 1 - dist / 2
        
        if self.visual_index is not None and query.visual_embedding is not None:
            k = min(top_k * self.rerank_factor, self.visual_index.ntotal)
            distances, indices = self.visual_index.search(
                query.visual_embedding.reshape(1, -1).astype(np.float32),
                k
            )
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    visual_candidates[int(idx)] = 1 - dist / 2
        
        # Merge candidates
        all_ids = set(text_candidates.keys()) | set(visual_candidates.keys())
        
        candidates = []
        for idx in all_ids:
            text_score = text_candidates.get(idx, 0.0)
            visual_score = visual_candidates.get(idx, 0.0)
            
            # Compute fused score
            fused_score = (
                query.text_weight * text_score +
                query.visual_weight * visual_score
            )
            
            metadata = self._get_metadata(idx)
            confidence = self._compute_confidence(
                text_score, visual_score, "fusion", query
            )
            
            candidates.append(MatchCandidate(
                book_id=str(idx),
                title=metadata.get("title", f"Book {idx}"),
                author=metadata.get("author", "Unknown"),
                similarity_score=fused_score,
                text_score=text_score,
                visual_score=visual_score,
                confidence=confidence,
                match_quality=self._quality_label(confidence),
                isbn=metadata.get("isbn"),
                cover_url=metadata.get("cover_url"),
                publication_year=metadata.get("year"),
                matched_via="fusion",
            ))
        
        # Sort by fused score
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        
        return candidates[:top_k]
    
    def _get_metadata(self, idx: int) -> dict:
        """Get metadata for book index."""
        if callable(self.book_metadata):
            return self.book_metadata(idx)
        return self.book_metadata.get(idx, {})
    
    def _compute_confidence(
        self,
        text_score: float,
        visual_score: float,
        method: str,
        query: FusedEmbedding,
    ) -> float:
        """
        Compute confidence score for a match.
        
        Factors:
        - Raw similarity scores
        - OCR confidence (for text matches)
        - Modal agreement (for fusion)
        """
        # Base confidence from primary score
        if method == "text_only":
            base = text_score
            # Adjust by OCR confidence
            base *= (0.5 + 0.5 * query.ocr_confidence)
        elif method == "visual_only":
            base = visual_score
            # Visual-only is less confident than multimodal
            base *= 0.85
        else:
            # Fusion - weighted combination
            base = query.text_weight * text_score + query.visual_weight * visual_score
            
            # Bonus for agreement between modalities
            if text_score > 0 and visual_score > 0:
                agreement = 1 - abs(text_score - visual_score)
                base *= (0.9 + 0.1 * agreement)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, base))
    
    def _quality_label(self, confidence: float) -> str:
        """Convert confidence to quality label."""
        if confidence >= self.HIGH_CONFIDENCE:
            return "high"
        elif confidence >= self.MEDIUM_CONFIDENCE:
            return "medium"
        elif confidence >= self.LOW_CONFIDENCE:
            return "low"
        else:
            return "uncertain"
    
    def batch_match(
        self,
        queries: list[FusedEmbedding],
        top_k: int = 5,
    ) -> list[MatchResult]:
        """
        Match multiple queries.
        
        Args:
            queries: List of fused embeddings
            top_k: Results per query
            
        Returns:
            List of MatchResult objects
        """
        return [self.match(q, top_k) for q in queries]


class ExactMatcher:
    """
    Exact matching using ISBN or other identifiers.
    
    Complements semantic matching with precise lookups.
    """
    
    def __init__(self, isbn_index: Optional[dict] = None):
        """
        Initialize exact matcher.
        
        Args:
            isbn_index: Mapping of ISBN -> book metadata
        """
        self.isbn_index = isbn_index or {}
    
    def match_isbn(self, isbn: str) -> Optional[dict]:
        """
        Match by ISBN.
        
        Args:
            isbn: ISBN-10 or ISBN-13
            
        Returns:
            Book metadata or None
        """
        # Normalize ISBN
        isbn = isbn.replace("-", "").replace(" ", "").upper()
        
        # Try direct lookup
        if isbn in self.isbn_index:
            return self.isbn_index[isbn]
        
        # Try converting between ISBN-10 and ISBN-13
        if len(isbn) == 10:
            isbn13 = self._isbn10_to_isbn13(isbn)
            if isbn13 in self.isbn_index:
                return self.isbn_index[isbn13]
        elif len(isbn) == 13:
            isbn10 = self._isbn13_to_isbn10(isbn)
            if isbn10 in self.isbn_index:
                return self.isbn_index[isbn10]
        
        return None
    
    def _isbn10_to_isbn13(self, isbn10: str) -> str:
        """Convert ISBN-10 to ISBN-13."""
        if len(isbn10) != 10:
            return isbn10
        
        prefix = "978" + isbn10[:-1]
        
        # Calculate check digit
        total = sum(
            int(d) * (1 if i % 2 == 0 else 3)
            for i, d in enumerate(prefix)
        )
        check = (10 - (total % 10)) % 10
        
        return prefix + str(check)
    
    def _isbn13_to_isbn10(self, isbn13: str) -> str:
        """Convert ISBN-13 to ISBN-10."""
        if len(isbn13) != 13 or not isbn13.startswith("978"):
            return isbn13
        
        body = isbn13[3:-1]
        
        # Calculate check digit
        total = sum(
            int(d) * (10 - i)
            for i, d in enumerate(body)
        )
        check = (11 - (total % 11)) % 11
        check_char = "X" if check == 10 else str(check)
        
        return body + check_char
