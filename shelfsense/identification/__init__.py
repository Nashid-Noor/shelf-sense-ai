"""
Book Identification Module

Matches detected books against external databases and enriches metadata.
"""

from shelfsense.identification.matcher import (
    BookMatcher,
    MatchResult,
    MatchCandidate,
    ExactMatcher,
)
from shelfsense.identification.metadata_enricher import (
    MetadataEnricher,
    BookMetadata,
    OpenLibraryClient,
    GoogleBooksClient,
)
from shelfsense.identification.candidate_ranker import (
    CandidateRanker,
    RankingResult,
    RankedCandidate,
)

__all__ = [
    # Matcher
    "BookMatcher",
    "MatchResult",
    "MatchCandidate",
    "ExactMatcher",
    # Metadata
    "MetadataEnricher",
    "BookMetadata",
    "OpenLibraryClient",
    "GoogleBooksClient",
    # Ranking
    "CandidateRanker",
    "RankingResult",
    "RankedCandidate",
]
