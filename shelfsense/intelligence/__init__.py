"""
Shelf Intelligence Module for ShelfSense AI

Analytics and recommendations for personal book libraries:
- Genre analysis and distribution
- Library diversity metrics
- Personalized recommendations
- Reading pattern insights

Components:
- GenreAnalyzer: Analyze genre composition and trends
- DiversityCalculator: Quantify library diversity
- BookRecommender: Generate personalized recommendations
"""

from shelfsense.intelligence.genre_analyzer import (
    GenreAnalyzer,
    GenreClassifier,
    GenreInfo,
    GenreClassification,
    GenreTaxonomy,
)
from shelfsense.intelligence.diversity_metrics import (
    DiversityCalculator,
    DiversityReport,
    DiversityScore,
    BookMetrics,
    calculate_quick_diversity,
)
from shelfsense.intelligence.recommender import (
    BookRecommender,
    Recommendation,
    RecommendationRequest,
    RecommendationType,
    LibraryProfile,
    BookData,
)

__all__ = [
    # Genre Analysis
    "GenreAnalyzer",
    "GenreClassifier",
    "GenreInfo",
    "GenreClassification",
    "GenreTaxonomy",
    # Diversity Metrics
    "DiversityCalculator",
    "DiversityReport",
    "DiversityScore",
    "BookMetrics",
    "calculate_quick_diversity",
    # Recommendations
    "BookRecommender",
    "Recommendation",
    "RecommendationRequest",
    "RecommendationType",
    "LibraryProfile",
    "BookData",
]
