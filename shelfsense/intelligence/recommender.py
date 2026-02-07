"""
Book Recommender for ShelfSense AI

Smart recommendations based on library analysis:
- Collaborative filtering (what similar readers enjoy)
- Content-based filtering (similar to books you own)
- Gap-based recommendations (genres to explore)
- Serendipity recommendations (unexpected discoveries)

Design Decisions:
1. Hybrid approach: Combine multiple recommendation strategies
2. Explainability: Every recommendation has a reason
3. Diversity: Balance between similar and exploratory
4. Context-aware: Consider reading history and preferences
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
from enum import Enum

from loguru import logger


class RecommendationType(str, Enum):
    """Types of recommendations."""
    
    SIMILAR = "similar"  # Similar to books you own
    AUTHOR = "author"  # More by authors you like
    GENRE = "genre"  # Popular in genres you read
    EXPLORATION = "exploration"  # Expand your horizons
    TRENDING = "trending"  # Currently popular
    CLASSIC = "classic"  # Timeless classics
    COMPLEMENT = "complement"  # Fills a gap in collection


@dataclass
class Recommendation:
    """A single book recommendation."""
    
    title: str
    author: str
    
    # Recommendation metadata
    type: RecommendationType
    reason: str
    confidence: float  # 0-1
    
    # Optional book metadata
    genres: list[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    description: Optional[str] = None
    
    # Reference (if based on existing book)
    based_on: Optional[str] = None  # Book ID
    based_on_title: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "title": self.title,
            "author": self.author,
            "type": self.type.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "genres": self.genres,
            "publication_year": self.publication_year,
            "based_on": self.based_on_title,
        }


@dataclass
class RecommendationRequest:
    """Request parameters for recommendations."""
    
    # What to base recommendations on
    based_on_book_id: Optional[str] = None
    based_on_genre: Optional[str] = None
    based_on_author: Optional[str] = None
    
    # Filters
    exclude_owned: bool = True
    genres: Optional[list[str]] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    
    # Diversity settings
    include_exploration: bool = True
    exploration_ratio: float = 0.2  # 20% exploratory
    
    # Limits
    count: int = 5


@dataclass
class LibraryProfile:
    """Profile of user's library for recommendations."""
    
    # Genre preferences
    top_genres: list[str] = field(default_factory=list)
    genre_distribution: dict[str, float] = field(default_factory=dict)
    
    # Author preferences
    favorite_authors: list[str] = field(default_factory=list)
    author_counts: dict[str, int] = field(default_factory=dict)
    
    # Temporal preferences
    preferred_era: str = "contemporary"
    avg_publication_year: int = 2000
    
    # Collection characteristics
    total_books: int = 0
    diversity_score: float = 0.5
    
    # Missing categories (gaps)
    underrepresented_genres: list[str] = field(default_factory=list)
    
    # Owned book IDs (for filtering)
    owned_book_ids: set[str] = field(default_factory=set)


@dataclass
class BookData:
    """Book data for recommendation matching."""
    
    id: str
    title: str
    author: str
    genres: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    description: Optional[str] = None
    popularity_score: float = 0.0


class BookRecommender:
    """
    Generate personalized book recommendations.
    
    Strategies:
    1. Content-based: Similar to owned books
    2. Author-based: More from favorite authors
    3. Genre-based: Popular in preferred genres
    4. Gap-filling: Explore underrepresented areas
    """
    
    # Genre exploration suggestions
    GENRE_EXPLORATION_MAP = {
        "science fiction": ["fantasy", "thriller", "popular science"],
        "fantasy": ["science fiction", "historical fiction", "mythology"],
        "mystery": ["thriller", "crime fiction", "spy novels"],
        "romance": ["contemporary fiction", "historical fiction"],
        "literary fiction": ["classics", "short stories", "memoir"],
        "history": ["biography", "historical fiction", "politics"],
        "science": ["philosophy", "technology", "nature"],
        "self-help": ["psychology", "philosophy", "biography"],
        "biography": ["memoir", "history", "journalism"],
    }
    
    # Classic recommendations by genre
    CLASSICS = {
        "science fiction": [
            BookData(id="classic_1", title="Dune", author="Frank Herbert", genres=["science fiction"]),
            BookData(id="classic_2", title="1984", author="George Orwell", genres=["science fiction", "dystopian"]),
            BookData(id="classic_3", title="Foundation", author="Isaac Asimov", genres=["science fiction"]),
        ],
        "fantasy": [
            BookData(id="classic_4", title="The Lord of the Rings", author="J.R.R. Tolkien", genres=["fantasy"]),
            BookData(id="classic_5", title="A Wizard of Earthsea", author="Ursula K. Le Guin", genres=["fantasy"]),
        ],
        "mystery": [
            BookData(id="classic_6", title="The Maltese Falcon", author="Dashiell Hammett", genres=["mystery"]),
            BookData(id="classic_7", title="Murder on the Orient Express", author="Agatha Christie", genres=["mystery"]),
        ],
        "literary fiction": [
            BookData(id="classic_8", title="To Kill a Mockingbird", author="Harper Lee", genres=["literary fiction"]),
            BookData(id="classic_9", title="The Great Gatsby", author="F. Scott Fitzgerald", genres=["literary fiction"]),
        ],
    }
    
    def __init__(
        self,
        catalog: Optional[list[BookData]] = None,
    ):
        """
        Initialize recommender.
        
        Args:
            catalog: Optional book catalog for recommendations
                    (if not provided, uses built-in classics)
        """
        self.catalog = catalog or []
        self._build_indices()
        
        logger.info(
            f"BookRecommender initialized with {len(self.catalog)} books in catalog"
        )
    
    def _build_indices(self):
        """Build lookup indices for efficient retrieval."""
        self.books_by_genre: dict[str, list[BookData]] = {}
        self.books_by_author: dict[str, list[BookData]] = {}
        
        for book in self.catalog:
            # Genre index
            for genre in book.genres:
                genre_lower = genre.lower()
                if genre_lower not in self.books_by_genre:
                    self.books_by_genre[genre_lower] = []
                self.books_by_genre[genre_lower].append(book)
            
            # Author index
            author_lower = book.author.lower()
            if author_lower not in self.books_by_author:
                self.books_by_author[author_lower] = []
            self.books_by_author[author_lower].append(book)
    
    def build_profile(
        self,
        owned_books: list[BookData],
    ) -> LibraryProfile:
        """
        Build library profile from owned books.
        
        Args:
            owned_books: User's book collection
            
        Returns:
            Library profile for recommendations
        """
        if not owned_books:
            # Continue with empty lists to populate defaults (like underrepresented genres)
            pass
        
        # Genre analysis
        genre_counts = Counter()
        for book in owned_books:
            for genre in book.genres:
                genre_counts[genre.lower()] += 1
        
        total_genres = sum(genre_counts.values()) or 1
        genre_distribution = {
            g: c / total_genres for g, c in genre_counts.items()
        }
        top_genres = [g for g, _ in genre_counts.most_common(5)]
        
        # Author analysis
        author_counts = Counter(book.author for book in owned_books)
        favorite_authors = [
            a for a, c in author_counts.most_common(5) if c >= 2
        ]
        
        # Temporal analysis
        years = [b.publication_year for b in owned_books if b.publication_year]
        avg_year = int(sum(years) / len(years)) if years else 2000
        
        if avg_year >= 2010:
            era = "contemporary"
        elif avg_year >= 1990:
            era = "modern"
        elif avg_year >= 1950:
            era = "mid-century"
        else:
            era = "classic"
        
        # Find underrepresented genres
        all_major_genres = [
            "science fiction", "fantasy", "mystery", "romance",
            "literary fiction", "biography", "history", "science",
        ]
        underrepresented = [
            g for g in all_major_genres
            if genre_distribution.get(g, 0) < 0.05
        ]
        
        # Owned book IDs
        owned_ids = {book.id for book in owned_books}
        
        return LibraryProfile(
            top_genres=top_genres,
            genre_distribution=genre_distribution,
            favorite_authors=favorite_authors,
            author_counts=dict(author_counts),
            preferred_era=era,
            avg_publication_year=avg_year,
            total_books=len(owned_books),
            underrepresented_genres=underrepresented,
            owned_book_ids=owned_ids,
        )
    
    def recommend(
        self,
        profile: LibraryProfile,
        request: Optional[RecommendationRequest] = None,
    ) -> list[Recommendation]:
        """
        Generate personalized recommendations.
        
        Args:
            profile: User's library profile
            request: Optional request parameters
            
        Returns:
            List of recommendations
        """
        request = request or RecommendationRequest()
        recommendations = []
        
        # Calculate how many of each type
        total_count = request.count
        exploration_count = int(total_count * request.exploration_ratio) if request.include_exploration else 0
        main_count = total_count - exploration_count
        
        # 1. Similar to specific book
        if request.based_on_book_id:
            similar = self._recommend_similar(
                request.based_on_book_id,
                profile,
                count=min(main_count, 3),
            )
            recommendations.extend(similar)
            main_count -= len(similar)
        
        # 2. More from favorite authors
        if main_count > 0 and profile.favorite_authors:
            author_recs = self._recommend_by_author(
                profile,
                count=min(main_count, 2),
            )
            recommendations.extend(author_recs)
            main_count -= len(author_recs)
        
        # 3. Genre-based recommendations
        if main_count > 0 and (request.based_on_genre or profile.top_genres):
            genre = request.based_on_genre or profile.top_genres[0]
            genre_recs = self._recommend_by_genre(
                genre,
                profile,
                count=main_count,
            )
            recommendations.extend(genre_recs)
        
        # 4. Exploration recommendations
        if exploration_count > 0:
            exploration_recs = self._recommend_exploration(
                profile,
                count=exploration_count,
            )
            recommendations.extend(exploration_recs)
        
        # Filter duplicates
        seen_titles = set()
        unique_recs = []
        for rec in recommendations:
            key = (rec.title.lower(), rec.author.lower())
            if key not in seen_titles:
                seen_titles.add(key)
                unique_recs.append(rec)
        
        # Sort by confidence
        unique_recs.sort(key=lambda r: r.confidence, reverse=True)
        
        logger.info(
            f"Generated {len(unique_recs)} recommendations "
            f"(similar: {sum(1 for r in unique_recs if r.type == RecommendationType.SIMILAR)}, "
            f"author: {sum(1 for r in unique_recs if r.type == RecommendationType.AUTHOR)}, "
            f"genre: {sum(1 for r in unique_recs if r.type == RecommendationType.GENRE)}, "
            f"exploration: {sum(1 for r in unique_recs if r.type == RecommendationType.EXPLORATION)})"
        )
        
        return unique_recs[:request.count]
    
    def _recommend_similar(
        self,
        book_id: str,
        profile: LibraryProfile,
        count: int = 3,
    ) -> list[Recommendation]:
        """Find books similar to a specific book."""
        # Find source book
        source_book = None
        for book in self.catalog:
            if book.id == book_id:
                source_book = book
                break
        
        if not source_book:
            return []
        
        recommendations = []
        
        # Find books with overlapping genres
        candidates = []
        for genre in source_book.genres:
            genre_lower = genre.lower()
            if genre_lower in self.books_by_genre:
                for book in self.books_by_genre[genre_lower]:
                    if book.id != book_id and book.id not in profile.owned_book_ids:
                        # Score by genre overlap
                        overlap = len(
                            set(g.lower() for g in book.genres) &
                            set(g.lower() for g in source_book.genres)
                        )
                        candidates.append((overlap, book))
        
        # Sort by overlap and take top
        candidates.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        
        for overlap, book in candidates:
            if book.id in seen:
                continue
            seen.add(book.id)
            
            confidence = min(0.5 + overlap * 0.15, 0.95)
            recommendations.append(Recommendation(
                title=book.title,
                author=book.author,
                type=RecommendationType.SIMILAR,
                reason=f"Similar themes and genres to {source_book.title}",
                confidence=confidence,
                genres=book.genres,
                publication_year=book.publication_year,
                based_on=book_id,
                based_on_title=source_book.title,
            ))
            
            if len(recommendations) >= count:
                break
        
        return recommendations
    
    def _recommend_by_author(
        self,
        profile: LibraryProfile,
        count: int = 2,
    ) -> list[Recommendation]:
        """Recommend more books by favorite authors."""
        recommendations = []
        
        for author in profile.favorite_authors:
            author_lower = author.lower()
            if author_lower in self.books_by_author:
                for book in self.books_by_author[author_lower]:
                    if book.id not in profile.owned_book_ids:
                        owned_count = profile.author_counts.get(author, 0)
                        recommendations.append(Recommendation(
                            title=book.title,
                            author=book.author,
                            type=RecommendationType.AUTHOR,
                            reason=f"You've enjoyed {owned_count} other books by {author}",
                            confidence=min(0.6 + owned_count * 0.1, 0.9),
                            genres=book.genres,
                            publication_year=book.publication_year,
                        ))
                        
                        if len(recommendations) >= count:
                            return recommendations
        
        return recommendations
    
    def _recommend_by_genre(
        self,
        genre: str,
        profile: LibraryProfile,
        count: int = 3,
    ) -> list[Recommendation]:
        """Recommend popular books in a genre."""
        recommendations = []
        genre_lower = genre.lower()
        
        # Try catalog first
        if genre_lower in self.books_by_genre:
            for book in self.books_by_genre[genre_lower]:
                if book.id not in profile.owned_book_ids:
                    genre_pct = profile.genre_distribution.get(genre_lower, 0)
                    recommendations.append(Recommendation(
                        title=book.title,
                        author=book.author,
                        type=RecommendationType.GENRE,
                        reason=f"Highly regarded in {genre}, which makes up {genre_pct*100:.0f}% of your library",
                        confidence=0.7,
                        genres=book.genres,
                        publication_year=book.publication_year,
                    ))
                    
                    if len(recommendations) >= count:
                        return recommendations
        
        # Fall back to classics
        if genre_lower in self.CLASSICS:
            for book in self.CLASSICS[genre_lower]:
                if book.id not in profile.owned_book_ids:
                    recommendations.append(Recommendation(
                        title=book.title,
                        author=book.author,
                        type=RecommendationType.CLASSIC,
                        reason=f"A classic of {genre} that every fan should read",
                        confidence=0.85,
                        genres=book.genres,
                    ))
                    
                    if len(recommendations) >= count:
                        return recommendations
        
        return recommendations
    
    def _recommend_exploration(
        self,
        profile: LibraryProfile,
        count: int = 1,
    ) -> list[Recommendation]:
        """Recommend books outside comfort zone."""
        recommendations = []
        
        # Suggest from underrepresented genres
        for genre in profile.underrepresented_genres:
            genre_lower = genre.lower()
            
            # Try catalog
            if genre_lower in self.books_by_genre:
                book = random.choice(self.books_by_genre[genre_lower])
                if book.id not in profile.owned_book_ids:
                    recommendations.append(Recommendation(
                        title=book.title,
                        author=book.author,
                        type=RecommendationType.EXPLORATION,
                        reason=f"Expand your horizons with {genre} - only {profile.genre_distribution.get(genre_lower, 0)*100:.0f}% of your library",
                        confidence=0.5,
                        genres=book.genres,
                        publication_year=book.publication_year,
                    ))
                    
                    if len(recommendations) >= count:
                        return recommendations
            
            # Fall back to classics
            if genre_lower in self.CLASSICS:
                book = random.choice(self.CLASSICS[genre_lower])
                if book.id not in profile.owned_book_ids:
                    recommendations.append(Recommendation(
                        title=book.title,
                        author=book.author,
                        type=RecommendationType.EXPLORATION,
                        reason=f"A great entry point into {genre}",
                        confidence=0.6,
                        genres=book.genres,
                    ))
                    
                    if len(recommendations) >= count:
                        return recommendations
        
        # Suggest genre bridges
        if profile.top_genres:
            top_genre = profile.top_genres[0].lower()
            if top_genre in self.GENRE_EXPLORATION_MAP:
                bridge_genres = self.GENRE_EXPLORATION_MAP[top_genre]
                for bridge in bridge_genres:
                    if bridge.lower() in self.CLASSICS:
                        book = random.choice(self.CLASSICS[bridge.lower()])
                        if book.id not in profile.owned_book_ids:
                            recommendations.append(Recommendation(
                                title=book.title,
                                author=book.author,
                                type=RecommendationType.COMPLEMENT,
                                reason=f"Fans of {top_genre} often enjoy {bridge}",
                                confidence=0.55,
                                genres=book.genres,
                            ))
                            
                            if len(recommendations) >= count:
                                return recommendations
        
        return recommendations
    
    def get_quick_recommendations(
        self,
        genres: list[str],
        count: int = 3,
    ) -> list[Recommendation]:
        """
        Quick recommendations without full profile.
        
        Useful for new users or quick suggestions.
        """
        recommendations = []
        
        for genre in genres:
            genre_lower = genre.lower()
            if genre_lower in self.CLASSICS:
                for book in self.CLASSICS[genre_lower]:
                    recommendations.append(Recommendation(
                        title=book.title,
                        author=book.author,
                        type=RecommendationType.CLASSIC,
                        reason=f"Essential reading for {genre} fans",
                        confidence=0.85,
                        genres=book.genres,
                    ))
                    
                    if len(recommendations) >= count:
                        return recommendations
        
        return recommendations
