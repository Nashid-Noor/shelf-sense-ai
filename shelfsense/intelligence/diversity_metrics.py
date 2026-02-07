"""
Diversity Metrics for ShelfSense AI

Quantify library diversity across multiple dimensions:
- Genre diversity (Shannon entropy, Simpson's index)
- Author diversity (unique authors, concentration)
- Temporal diversity (publication year spread)
- Geographic/cultural diversity (if available)

Design Decisions:
1. Multiple metrics: Different indices capture different aspects
2. Normalized scores: All metrics scaled to 0-1 for comparison
3. Benchmarking: Compare against "ideal" diverse library
4. Actionable insights: Specific recommendations for improvement
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from loguru import logger


@dataclass
class DiversityScore:
    """A single diversity metric."""
    
    name: str
    value: float  # 0-1 normalized
    raw_value: float  # Original calculation
    interpretation: str
    percentile: Optional[float] = None  # vs benchmark
    
    def __str__(self) -> str:
        return f"{self.name}: {self.value:.2f} ({self.interpretation})"


@dataclass
class DiversityReport:
    """Complete diversity analysis."""
    
    # Overall scores
    overall_score: float
    overall_grade: str  # A, B, C, D, F
    
    # Individual dimension scores
    genre_diversity: DiversityScore
    author_diversity: DiversityScore
    temporal_diversity: DiversityScore
    
    # Optional dimensions
    subject_diversity: Optional[DiversityScore] = None
    
    # Breakdown
    total_books: int = 0
    unique_genres: int = 0
    unique_authors: int = 0
    year_range: tuple[int, int] = (0, 0)
    
    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "genre_diversity": {
                "score": self.genre_diversity.value,
                "interpretation": self.genre_diversity.interpretation,
            },
            "author_diversity": {
                "score": self.author_diversity.value,
                "interpretation": self.author_diversity.interpretation,
            },
            "temporal_diversity": {
                "score": self.temporal_diversity.value,
                "interpretation": self.temporal_diversity.interpretation,
            },
            "stats": {
                "total_books": self.total_books,
                "unique_genres": self.unique_genres,
                "unique_authors": self.unique_authors,
                "year_range": self.year_range,
            },
            "recommendations": self.recommendations,
        }


@dataclass
class BookMetrics:
    """Book data for diversity calculation."""
    
    id: str
    genres: list[str] = field(default_factory=list)
    author: str = ""
    publication_year: Optional[int] = None
    subjects: list[str] = field(default_factory=list)


class DiversityCalculator:
    """
    Calculate library diversity metrics.
    
    Uses information-theoretic and ecological diversity indices:
    - Shannon entropy (information diversity)
    - Simpson's index (probability-based)
    - Gini-Simpson (complement of Simpson's)
    """
    
    # Benchmarks for "highly diverse" libraries
    BENCHMARK_GENRES = 12  # Ideal unique genre count
    BENCHMARK_AUTHORS_RATIO = 0.7  # Unique authors / total books
    BENCHMARK_YEAR_SPREAD = 100  # Years between oldest and newest
    
    def __init__(self):
        """Initialize calculator."""
        logger.info("DiversityCalculator initialized")
    
    def calculate(self, books: list[BookMetrics]) -> DiversityReport:
        """
        Calculate diversity metrics.
        
        Args:
            books: List of books with metadata
            
        Returns:
            Complete diversity report
        """
        if not books:
            return self._empty_report()
        
        # Calculate individual dimensions
        genre_score = self._genre_diversity(books)
        author_score = self._author_diversity(books)
        temporal_score = self._temporal_diversity(books)
        
        # Subject diversity if available
        subject_score = None
        has_subjects = any(b.subjects for b in books)
        if has_subjects:
            subject_score = self._subject_diversity(books)
        
        # Overall score (weighted average)
        weights = [0.35, 0.35, 0.30]  # genre, author, temporal
        scores = [genre_score.value, author_score.value, temporal_score.value]
        
        if subject_score:
            weights.append(0.15)
            scores.append(subject_score.value)
            # Renormalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        overall = sum(w * s for w, s in zip(weights, scores))
        
        # Generate grade
        grade = self._score_to_grade(overall)
        
        # Collect stats
        all_genres = set()
        for book in books:
            all_genres.update(g.lower() for g in book.genres)
        
        authors = set(book.author for book in books)
        
        years = [b.publication_year for b in books if b.publication_year]
        year_range = (min(years), max(years)) if years else (0, 0)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            genre_score, author_score, temporal_score, books
        )
        
        report = DiversityReport(
            overall_score=overall,
            overall_grade=grade,
            genre_diversity=genre_score,
            author_diversity=author_score,
            temporal_diversity=temporal_score,
            subject_diversity=subject_score,
            total_books=len(books),
            unique_genres=len(all_genres),
            unique_authors=len(authors),
            year_range=year_range,
            recommendations=recommendations,
        )
        
        logger.info(
            f"Diversity calculated: overall={overall:.2f} ({grade}), "
            f"genres={genre_score.value:.2f}, "
            f"authors={author_score.value:.2f}, "
            f"temporal={temporal_score.value:.2f}"
        )
        
        return report
    
    def _genre_diversity(self, books: list[BookMetrics]) -> DiversityScore:
        """Calculate genre diversity using Shannon entropy."""
        # Flatten all genres
        all_genres = []
        for book in books:
            all_genres.extend(g.lower() for g in book.genres)
        
        if not all_genres:
            return DiversityScore(
                name="Genre Diversity",
                value=0.0,
                raw_value=0.0,
                interpretation="No genre data available",
            )
        
        # Count frequencies
        counts = Counter(all_genres)
        total = len(all_genres)
        
        # Shannon entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize by maximum possible entropy (log2 of unique genres)
        unique_count = len(counts)
        max_entropy = math.log2(unique_count) if unique_count > 1 else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        
        # Adjust for number of genres (reward having more categories)
        genre_bonus = min(unique_count / self.BENCHMARK_GENRES, 1.0) * 0.2
        final_score = min(normalized * 0.8 + genre_bonus, 1.0)
        
        interpretation = self._interpret_score(final_score, "genre")
        
        return DiversityScore(
            name="Genre Diversity",
            value=final_score,
            raw_value=entropy,
            interpretation=interpretation,
        )
    
    def _author_diversity(self, books: list[BookMetrics]) -> DiversityScore:
        """Calculate author diversity."""
        if not books:
            return DiversityScore(
                name="Author Diversity",
                value=0.0,
                raw_value=0.0,
                interpretation="No books to analyze",
            )
        
        authors = [book.author for book in books if book.author]
        if not authors:
            return DiversityScore(
                name="Author Diversity",
                value=0.0,
                raw_value=0.0,
                interpretation="No author data available",
            )
        
        # Unique author ratio
        unique_authors = len(set(authors))
        total = len(authors)
        ratio = unique_authors / total
        
        # Gini-Simpson index for author concentration
        counts = Counter(authors)
        simpson = sum((c / total) ** 2 for c in counts.values())
        gini_simpson = 1 - simpson
        
        # Combine ratio and Gini-Simpson
        # Penalize having many books by few authors
        score = (ratio * 0.6 + gini_simpson * 0.4)
        
        # Benchmark comparison
        benchmark_ratio = min(ratio / self.BENCHMARK_AUTHORS_RATIO, 1.0)
        final_score = score * 0.7 + benchmark_ratio * 0.3
        
        interpretation = self._interpret_score(final_score, "author")
        
        return DiversityScore(
            name="Author Diversity",
            value=final_score,
            raw_value=ratio,
            interpretation=interpretation,
        )
    
    def _temporal_diversity(self, books: list[BookMetrics]) -> DiversityScore:
        """Calculate publication year diversity."""
        years = [b.publication_year for b in books if b.publication_year]
        
        if not years:
            return DiversityScore(
                name="Temporal Diversity",
                value=0.5,  # Neutral if no data
                raw_value=0.0,
                interpretation="No publication year data available",
            )
        
        # Year spread
        min_year = min(years)
        max_year = max(years)
        spread = max_year - min_year
        
        # Standard deviation (normalized by benchmark)
        mean_year = sum(years) / len(years)
        variance = sum((y - mean_year) ** 2 for y in years) / len(years)
        std_dev = math.sqrt(variance)
        
        # Decade distribution (how many decades represented)
        decades = set(y // 10 for y in years)
        decade_coverage = len(decades) / 15  # Assume 15 decades max (1870-2020s)
        
        # Combine metrics
        spread_score = min(spread / self.BENCHMARK_YEAR_SPREAD, 1.0)
        std_score = min(std_dev / 30, 1.0)  # Normalize by expected std
        
        final_score = (
            spread_score * 0.4 +
            std_score * 0.3 +
            decade_coverage * 0.3
        )
        
        interpretation = self._interpret_score(final_score, "temporal")
        
        return DiversityScore(
            name="Temporal Diversity",
            value=final_score,
            raw_value=spread,
            interpretation=interpretation,
        )
    
    def _subject_diversity(self, books: list[BookMetrics]) -> DiversityScore:
        """Calculate subject/topic diversity."""
        all_subjects = []
        for book in books:
            all_subjects.extend(s.lower() for s in book.subjects)
        
        if not all_subjects:
            return DiversityScore(
                name="Subject Diversity",
                value=0.5,
                raw_value=0.0,
                interpretation="No subject data available",
            )
        
        # Shannon entropy for subjects
        counts = Counter(all_subjects)
        total = len(all_subjects)
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        unique_count = len(counts)
        max_entropy = math.log2(unique_count) if unique_count > 1 else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        
        interpretation = self._interpret_score(normalized, "subject")
        
        return DiversityScore(
            name="Subject Diversity",
            value=normalized,
            raw_value=entropy,
            interpretation=interpretation,
        )
    
    def _interpret_score(self, score: float, dimension: str) -> str:
        """Generate human-readable interpretation."""
        if score >= 0.8:
            level = "Excellent"
            detail = {
                "genre": "Your library spans a rich variety of genres",
                "author": "You read widely across many different authors",
                "temporal": "Your collection spans many eras of literature",
                "subject": "Your interests cover diverse topics",
            }
        elif score >= 0.6:
            level = "Good"
            detail = {
                "genre": "Your library has good genre variety",
                "author": "You have a healthy mix of authors",
                "temporal": "Your collection includes books from different periods",
                "subject": "You explore various subject areas",
            }
        elif score >= 0.4:
            level = "Moderate"
            detail = {
                "genre": "Your library focuses on a few main genres",
                "author": "You tend to read multiple books by the same authors",
                "temporal": "Your books cluster around certain time periods",
                "subject": "Your reading focuses on specific topics",
            }
        else:
            level = "Limited"
            detail = {
                "genre": "Your library is heavily concentrated in few genres",
                "author": "Your collection is dominated by few authors",
                "temporal": "Your books are from a narrow time range",
                "subject": "Your reading is highly specialized",
            }
        
        return f"{level}: {detail.get(dimension, 'See detailed metrics')}"
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        elif score >= 0.4:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(
        self,
        genre: DiversityScore,
        author: DiversityScore,
        temporal: DiversityScore,
        books: list[BookMetrics],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs = []
        
        if genre.value < 0.5:
            # Find dominant genres
            all_genres = []
            for book in books:
                all_genres.extend(g.lower() for g in book.genres)
            top_genres = [g for g, _ in Counter(all_genres).most_common(2)]
            
            missing = ["science fiction", "mystery", "biography", "history"]
            suggestions = [g for g in missing if g not in top_genres][:2]
            
            recs.append(
                f"Consider exploring new genres like {' or '.join(suggestions)} "
                f"to diversify beyond {', '.join(top_genres)}"
            )
        
        if author.value < 0.5:
            # Find dominant authors
            authors = Counter(b.author for b in books if b.author)
            top_authors = [a for a, c in authors.most_common(3) if c >= 3]
            
            if top_authors:
                recs.append(
                    f"You have many books by {', '.join(top_authors[:2])}. "
                    "Try discovering new authors in similar genres"
                )
        
        if temporal.value < 0.5:
            years = [b.publication_year for b in books if b.publication_year]
            if years:
                avg_year = sum(years) / len(years)
                if avg_year > 2010:
                    recs.append(
                        "Your library focuses on recent releases. "
                        "Consider exploring classics from earlier decades"
                    )
                elif avg_year < 1990:
                    recs.append(
                        "Many of your books are older publications. "
                        "Try some contemporary authors"
                    )
        
        if not recs:
            recs.append(
                "Great diversity! Keep exploring different genres, "
                "authors, and eras to maintain variety"
            )
        
        return recs[:3]  # Max 3 recommendations
    
    def _empty_report(self) -> DiversityReport:
        """Generate empty report for no books."""
        return DiversityReport(
            overall_score=0.0,
            overall_grade="N/A",
            genre_diversity=DiversityScore(
                name="Genre Diversity",
                value=0.0,
                raw_value=0.0,
                interpretation="No books to analyze",
            ),
            author_diversity=DiversityScore(
                name="Author Diversity",
                value=0.0,
                raw_value=0.0,
                interpretation="No books to analyze",
            ),
            temporal_diversity=DiversityScore(
                name="Temporal Diversity",
                value=0.0,
                raw_value=0.0,
                interpretation="No books to analyze",
            ),
            total_books=0,
            recommendations=["Add some books to get diversity insights!"],
        )


def calculate_quick_diversity(
    genre_counts: dict[str, int],
    author_counts: dict[str, int],
) -> float:
    """
    Quick diversity calculation without full BookMetrics.
    
    Useful for API endpoints that just need a score.
    
    Returns:
        Diversity score 0-1
    """
    # Genre Shannon entropy
    total_genres = sum(genre_counts.values()) or 1
    genre_entropy = 0.0
    for count in genre_counts.values():
        if count > 0:
            p = count / total_genres
            genre_entropy -= p * math.log2(p)
    
    max_genre_entropy = math.log2(len(genre_counts)) if len(genre_counts) > 1 else 1
    genre_score = genre_entropy / max_genre_entropy if max_genre_entropy > 0 else 0
    
    # Author Gini-Simpson
    total_authors = sum(author_counts.values()) or 1
    simpson = sum((c / total_authors) ** 2 for c in author_counts.values())
    author_score = 1 - simpson
    
    # Combined
    return (genre_score + author_score) / 2
