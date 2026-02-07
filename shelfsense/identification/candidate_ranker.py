"""
Candidate Ranker for ShelfSense AI

Advanced ranking and scoring of book match candidates:
- Multi-signal score fusion
- String similarity verification
- Confidence calibration
- Disambiguation for similar titles

Design Decisions:
1. Hybrid scoring: Combine embedding similarity with fuzzy string matching
2. Author verification: Boost matches where author names align
3. Title normalization: Handle subtitle variations, articles, punctuation
4. Edition handling: Group different editions of same work
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional
from loguru import logger


@dataclass
class ScoringComponents:
    """Breakdown of scoring components for transparency."""
    
    embedding_score: float = 0.0
    text_similarity: float = 0.0
    author_match: float = 0.0
    title_match: float = 0.0
    
    # Penalty factors
    length_penalty: float = 0.0
    edition_bonus: float = 0.0
    
    # Final computed score
    final_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "embedding_score": self.embedding_score,
            "text_similarity": self.text_similarity,
            "author_match": self.author_match,
            "title_match": self.title_match,
            "length_penalty": self.length_penalty,
            "edition_bonus": self.edition_bonus,
            "final_score": self.final_score,
        }


@dataclass
class RankedCandidate:
    """Candidate with full ranking details."""
    
    # Original match info
    book_id: str
    title: str
    author: str
    
    # Scores
    original_score: float
    reranked_score: float
    scoring_components: ScoringComponents
    
    # Ranking
    original_rank: int
    final_rank: int
    
    # Confidence
    confidence: float
    quality: str  # "high", "medium", "low", "uncertain"
    
    # Metadata
    isbn: Optional[str] = None
    cover_url: Optional[str] = None
    
    # Disambiguation
    is_ambiguous: bool = False
    similar_candidates: list[str] = field(default_factory=list)


@dataclass
class RankingResult:
    """Result of ranking operation."""
    
    query_text: str
    candidates: list[RankedCandidate]
    
    # Statistics
    total_considered: int = 0
    reranking_changed_order: bool = False
    ambiguous_matches: int = 0
    
    @property
    def best_match(self) -> Optional[RankedCandidate]:
        if self.candidates:
            return self.candidates[0]
        return None
    
    @property
    def is_confident(self) -> bool:
        if not self.candidates:
            return False
        return self.candidates[0].confidence >= 0.75


class TextNormalizer:
    """Normalize text for comparison."""
    
    # Articles to strip from beginning
    ARTICLES = {"the", "a", "an", "el", "la", "le", "der", "die", "das"}
    
    # Common subtitle separators
    SUBTITLE_PATTERNS = [
        r":\s+.+$",  # ": subtitle"
        r"\s+-\s+.+$",  # " - subtitle"
        r"\s+\(.+\)$",  # " (subtitle)"
    ]
    
    @classmethod
    def normalize_title(cls, title: str, strip_subtitle: bool = False) -> str:
        """
        Normalize book title for comparison.
        
        Args:
            title: Original title
            strip_subtitle: Remove subtitle portion
            
        Returns:
            Normalized title
        """
        if not title:
            return ""
        
        text = title.lower().strip()
        
        # Strip subtitle if requested
        if strip_subtitle:
            for pattern in cls.SUBTITLE_PATTERNS:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Remove leading articles
        words = text.split()
        if words and words[0] in cls.ARTICLES:
            text = " ".join(words[1:])
        
        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r"[^\w\s]", "", text)
        
        # Collapse whitespace
        text = " ".join(text.split())
        
        return text
    
    @classmethod
    def normalize_author(cls, author: str) -> str:
        """
        Normalize author name for comparison.
        
        Args:
            author: Original author name
            
        Returns:
            Normalized name
        """
        if not author:
            return ""
        
        text = author.lower().strip()
        
        # Remove common suffixes
        for suffix in ["jr", "jr.", "sr", "sr.", "iii", "ii", "phd", "ph.d."]:
            text = re.sub(rf"\s+{re.escape(suffix)}$", "", text, flags=re.IGNORECASE)
        
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        
        # Collapse whitespace
        text = " ".join(text.split())
        
        return text
    
    @classmethod
    def extract_author_lastname(cls, author: str) -> str:
        """Extract likely last name from author."""
        normalized = cls.normalize_author(author)
        parts = normalized.split()
        
        if not parts:
            return ""
        
        # Usually last word is surname
        return parts[-1]


class StringSimilarity:
    """String similarity calculations."""
    
    @staticmethod
    def ratio(s1: str, s2: str) -> float:
        """
        Compute similarity ratio between two strings.
        
        Uses SequenceMatcher for robust fuzzy matching.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score 0-1
        """
        if not s1 or not s2:
            return 0.0
        
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    @staticmethod
    def partial_ratio(s1: str, s2: str) -> float:
        """
        Compute partial match ratio.
        
        Handles cases where one string is substring of other.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Partial similarity score 0-1
        """
        if not s1 or not s2:
            return 0.0
        
        s1, s2 = s1.lower(), s2.lower()
        
        # Direct substring check
        if s1 in s2 or s2 in s1:
            shorter = min(len(s1), len(s2))
            longer = max(len(s1), len(s2))
            return shorter / longer
        
        # Use SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
    
    @staticmethod
    def token_sort_ratio(s1: str, s2: str) -> float:
        """
        Compute similarity after sorting tokens.
        
        Handles word order differences:
        "Rowling JK" vs "JK Rowling"
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Token-sorted similarity score 0-1
        """
        if not s1 or not s2:
            return 0.0
        
        # Sort tokens alphabetically
        tokens1 = sorted(s1.lower().split())
        tokens2 = sorted(s2.lower().split())
        
        sorted_s1 = " ".join(tokens1)
        sorted_s2 = " ".join(tokens2)
        
        return SequenceMatcher(None, sorted_s1, sorted_s2).ratio()


class CandidateRanker:
    """
    Advanced candidate ranking with multi-signal scoring.
    
    Features:
    - Embedding similarity as base score
    - Fuzzy string matching for title/author verification
    - Author name extraction and matching
    - Subtitle handling
    - Edition grouping
    - Confidence calibration
    
    Usage:
        ranker = CandidateRanker()
        
        # Rerank candidates from initial retrieval
        result = ranker.rank(
            query_text="Harry Potter Philosopher Stone",
            candidates=initial_candidates,
        )
        
        print(result.best_match.title)
    """
    
    # Scoring weights
    EMBEDDING_WEIGHT = 0.5
    TITLE_WEIGHT = 0.3
    AUTHOR_WEIGHT = 0.2
    
    # Quality thresholds
    HIGH_QUALITY_THRESHOLD = 0.85
    MEDIUM_QUALITY_THRESHOLD = 0.65
    LOW_QUALITY_THRESHOLD = 0.45
    
    # Ambiguity threshold (gap to second place)
    AMBIGUITY_GAP = 0.05
    
    def __init__(
        self,
        embedding_weight: float = 0.5,
        title_weight: float = 0.3,
        author_weight: float = 0.2,
        use_subtitle_stripping: bool = True,
    ):
        """
        Initialize ranker.
        
        Args:
            embedding_weight: Weight for embedding similarity
            title_weight: Weight for title string match
            author_weight: Weight for author string match
            use_subtitle_stripping: Strip subtitles for comparison
        """
        self.embedding_weight = embedding_weight
        self.title_weight = title_weight
        self.author_weight = author_weight
        self.use_subtitle_stripping = use_subtitle_stripping
        
        # Validate weights sum to 1
        total = embedding_weight + title_weight + author_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Scoring weights sum to {total}, not 1.0")
    
    def rank(
        self,
        query_text: str,
        candidates: list,  # List of MatchCandidate or similar
        query_author: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RankingResult:
        """
        Rerank candidates using multi-signal scoring.
        
        Args:
            query_text: OCR text from detection
            candidates: Initial candidates from retrieval
            query_author: Extracted author (if available)
            top_k: Limit results (None = all)
            
        Returns:
            RankingResult with reranked candidates
        """
        if not candidates:
            return RankingResult(
                query_text=query_text,
                candidates=[],
                total_considered=0,
            )
        
        # Parse query
        query_title, parsed_author = self._parse_query(query_text)
        if query_author:
            parsed_author = query_author
        
        # Normalize query
        norm_query_title = TextNormalizer.normalize_title(
            query_title, 
            strip_subtitle=self.use_subtitle_stripping
        )
        norm_query_author = TextNormalizer.normalize_author(parsed_author) if parsed_author else ""
        
        # Score each candidate
        ranked = []
        for i, candidate in enumerate(candidates):
            scored = self._score_candidate(
                candidate=candidate,
                norm_query_title=norm_query_title,
                norm_query_author=norm_query_author,
                original_rank=i + 1,
            )
            ranked.append(scored)
        
        # Sort by reranked score
        ranked.sort(key=lambda c: c.reranked_score, reverse=True)
        
        # Assign final ranks
        for i, candidate in enumerate(ranked):
            candidate.final_rank = i + 1
        
        # Detect ambiguous matches
        ambiguous_count = self._detect_ambiguity(ranked)
        
        # Check if order changed
        order_changed = any(
            c.final_rank != c.original_rank for c in ranked
        )
        
        # Limit results
        if top_k:
            ranked = ranked[:top_k]
        
        return RankingResult(
            query_text=query_text,
            candidates=ranked,
            total_considered=len(candidates),
            reranking_changed_order=order_changed,
            ambiguous_matches=ambiguous_count,
        )
    
    def _parse_query(self, query_text: str) -> tuple[str, Optional[str]]:
        """
        Parse query text into title and author.
        
        Heuristic patterns:
        - "Title by Author"
        - "Author - Title"
        - "Title Author" (author usually last)
        
        Args:
            query_text: Raw query text
            
        Returns:
            (title, author) tuple
        """
        text = query_text.strip()
        
        # Pattern: "Title by Author"
        if " by " in text.lower():
            parts = text.lower().split(" by ", 1)
            title = text[:len(parts[0])]
            author = text[len(parts[0]) + 4:]  # Skip " by "
            return title.strip(), author.strip()
        
        # Pattern: "Author - Title" or "Title - Author"
        if " - " in text:
            parts = text.split(" - ", 1)
            # Heuristic: shorter part is usually author
            if len(parts[0]) < len(parts[1]):
                return parts[1].strip(), parts[0].strip()
            else:
                return parts[0].strip(), parts[1].strip()
        
        # No clear separator - return as title
        return text, None
    
    def _score_candidate(
        self,
        candidate,
        norm_query_title: str,
        norm_query_author: str,
        original_rank: int,
    ) -> RankedCandidate:
        """
        Score a single candidate.
        
        Args:
            candidate: Match candidate
            norm_query_title: Normalized query title
            norm_query_author: Normalized query author
            original_rank: Original retrieval rank
            
        Returns:
            RankedCandidate with scores
        """
        # Get candidate info
        cand_title = getattr(candidate, "title", "")
        cand_author = getattr(candidate, "author", "")
        cand_score = getattr(candidate, "similarity_score", 0.0)
        cand_isbn = getattr(candidate, "isbn", None)
        cand_cover = getattr(candidate, "cover_url", None)
        
        # Normalize candidate
        norm_cand_title = TextNormalizer.normalize_title(
            cand_title,
            strip_subtitle=self.use_subtitle_stripping
        )
        norm_cand_author = TextNormalizer.normalize_author(cand_author)
        
        # Calculate component scores
        components = ScoringComponents()
        
        # 1. Embedding score (from retrieval)
        components.embedding_score = cand_score
        
        # 2. Title similarity
        title_ratio = StringSimilarity.ratio(norm_query_title, norm_cand_title)
        title_partial = StringSimilarity.partial_ratio(norm_query_title, norm_cand_title)
        components.title_match = max(title_ratio, title_partial * 0.9)
        
        # 3. Author similarity
        if norm_query_author and norm_cand_author:
            author_ratio = StringSimilarity.ratio(norm_query_author, norm_cand_author)
            author_token = StringSimilarity.token_sort_ratio(norm_query_author, norm_cand_author)
            
            # Also try last name only
            query_lastname = TextNormalizer.extract_author_lastname(norm_query_author)
            cand_lastname = TextNormalizer.extract_author_lastname(norm_cand_author)
            lastname_match = StringSimilarity.ratio(query_lastname, cand_lastname)
            
            components.author_match = max(author_ratio, author_token, lastname_match)
        else:
            # No author to compare - neutral score
            components.author_match = 0.5
        
        # 4. Length penalty (very short or very long mismatches)
        len_ratio = len(norm_query_title) / max(len(norm_cand_title), 1)
        if len_ratio < 0.5 or len_ratio > 2.0:
            components.length_penalty = -0.1
        
        # Compute final score
        components.final_score = (
            self.embedding_weight * components.embedding_score +
            self.title_weight * components.title_match +
            self.author_weight * components.author_match +
            components.length_penalty +
            components.edition_bonus
        )
        
        # Calibrate confidence
        confidence = self._calibrate_confidence(components)
        quality = self._quality_label(confidence)
        
        return RankedCandidate(
            book_id=getattr(candidate, "book_id", ""),
            title=cand_title,
            author=cand_author,
            original_score=cand_score,
            reranked_score=components.final_score,
            scoring_components=components,
            original_rank=original_rank,
            final_rank=0,  # Set later
            confidence=confidence,
            quality=quality,
            isbn=cand_isbn,
            cover_url=cand_cover,
        )
    
    def _calibrate_confidence(self, components: ScoringComponents) -> float:
        """
        Calibrate confidence based on scoring components.
        
        High confidence requires:
        - Good embedding score
        - Good title match
        - Reasonable author match (if available)
        """
        # Base from final score
        base = components.final_score
        
        # Boost for strong agreement
        if (components.embedding_score > 0.8 and 
            components.title_match > 0.8):
            base *= 1.1
        
        # Penalty for disagreement
        if abs(components.embedding_score - components.title_match) > 0.3:
            base *= 0.9
        
        # Clamp
        return max(0.0, min(1.0, base))
    
    def _quality_label(self, confidence: float) -> str:
        """Convert confidence to quality label."""
        if confidence >= self.HIGH_QUALITY_THRESHOLD:
            return "high"
        elif confidence >= self.MEDIUM_QUALITY_THRESHOLD:
            return "medium"
        elif confidence >= self.LOW_QUALITY_THRESHOLD:
            return "low"
        else:
            return "uncertain"
    
    def _detect_ambiguity(self, ranked: list[RankedCandidate]) -> int:
        """
        Detect ambiguous matches.
        
        Marks candidates as ambiguous if:
        - Score gap to next candidate is small
        - Multiple candidates have very similar scores
        
        Returns count of ambiguous matches.
        """
        if len(ranked) < 2:
            return 0
        
        ambiguous_count = 0
        
        for i in range(len(ranked) - 1):
            gap = ranked[i].reranked_score - ranked[i + 1].reranked_score
            
            if gap < self.AMBIGUITY_GAP:
                ranked[i].is_ambiguous = True
                ranked[i].similar_candidates.append(ranked[i + 1].book_id)
                ambiguous_count += 1
        
        return ambiguous_count
    
    def explain_ranking(self, result: RankingResult) -> str:
        """
        Generate human-readable ranking explanation.
        
        Args:
            result: Ranking result
            
        Returns:
            Explanation string
        """
        lines = [
            f"Query: {result.query_text}",
            f"Candidates considered: {result.total_considered}",
            f"Order changed by reranking: {result.reranking_changed_order}",
            f"Ambiguous matches: {result.ambiguous_matches}",
            "",
        ]
        
        for cand in result.candidates[:5]:
            lines.append(f"Rank {cand.final_rank}: {cand.title} by {cand.author}")
            lines.append(f"  Score: {cand.reranked_score:.3f} (was {cand.original_score:.3f})")
            lines.append(f"  Confidence: {cand.confidence:.2f} ({cand.quality})")
            
            sc = cand.scoring_components
            lines.append(
                f"  Components: emb={sc.embedding_score:.2f}, "
                f"title={sc.title_match:.2f}, "
                f"author={sc.author_match:.2f}"
            )
            
            if cand.is_ambiguous:
                lines.append(f"  ⚠️ Ambiguous - similar to: {cand.similar_candidates}")
            
            lines.append("")
        
        return "\n".join(lines)
