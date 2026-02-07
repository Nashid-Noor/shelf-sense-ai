"""
Genre Analyzer for ShelfSense AI

Genre analysis for book collections:
- Genre classification from text/metadata
- Genre hierarchy and relationships
- Collection genre distribution
- Genre trend detection

Design Decisions:
1. Hierarchical genres: Main genres with sub-genres
2. Multi-label: Books can belong to multiple genres
3. Confidence scoring: Probabilistic genre assignment
4. Rule + ML hybrid: Fast rules with ML fallback
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter, defaultdict
import re

from loguru import logger


@dataclass
class GenreInfo:
    """Information about a genre."""
    
    name: str
    parent: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    
    @property
    def full_path(self) -> str:
        """Get full genre path (e.g., 'Fiction > Science Fiction')."""
        if self.parent:
            return f"{self.parent} > {self.name}"
        return self.name


@dataclass
class GenreClassification:
    """Genre classification result for a book."""
    
    primary_genre: str
    confidence: float
    secondary_genres: list[tuple[str, float]] = field(default_factory=list)
    
    # All detected genres with scores
    all_genres: dict[str, float] = field(default_factory=dict)
    
    # Metadata
    classification_method: str = "rule_based"


class GenreTaxonomy:
    """
    Hierarchical genre taxonomy.
    
    Defines the genre structure and relationships
    used for classification.
    """
    
    # Main genre categories
    GENRES = {
        # Fiction
        "Fiction": GenreInfo(
            name="Fiction",
            aliases=["novel", "fiction"],
            keywords=["novel", "story", "tale"],
        ),
        "Science Fiction": GenreInfo(
            name="Science Fiction",
            parent="Fiction",
            aliases=["sci-fi", "sf", "scifi"],
            keywords=[
                "space", "robot", "alien", "future", "dystopia",
                "technology", "artificial", "mars", "galaxy", "cyberpunk",
                "android", "spaceship", "interstellar", "quantum",
            ],
        ),
        "Fantasy": GenreInfo(
            name="Fantasy",
            parent="Fiction",
            aliases=["fantastical"],
            keywords=[
                "magic", "wizard", "dragon", "elf", "kingdom",
                "sword", "quest", "mythical", "prophecy", "enchanted",
                "sorcerer", "medieval", "realm", "throne",
            ],
        ),
        "Mystery": GenreInfo(
            name="Mystery",
            parent="Fiction",
            aliases=["detective", "whodunit"],
            keywords=[
                "detective", "murder", "crime", "investigation",
                "suspect", "clue", "sleuth", "forensic", "alibi",
                "victim", "witness", "police",
            ],
        ),
        "Thriller": GenreInfo(
            name="Thriller",
            parent="Fiction",
            aliases=["suspense"],
            keywords=[
                "thriller", "suspense", "chase", "conspiracy",
                "spy", "assassin", "danger", "escape", "pursuit",
                "hostage", "terrorist", "agent",
            ],
        ),
        "Romance": GenreInfo(
            name="Romance",
            parent="Fiction",
            aliases=["love story", "romantic"],
            keywords=[
                "love", "romance", "heart", "passion", "desire",
                "relationship", "wedding", "marriage", "couple",
                "attraction", "soulmate",
            ],
        ),
        "Horror": GenreInfo(
            name="Horror",
            parent="Fiction",
            aliases=["scary", "terror"],
            keywords=[
                "horror", "scary", "ghost", "monster", "haunted",
                "nightmare", "terror", "undead", "vampire", "zombie",
                "supernatural", "evil", "dark",
            ],
        ),
        "Literary Fiction": GenreInfo(
            name="Literary Fiction",
            parent="Fiction",
            aliases=["literature", "literary"],
            keywords=[
                "literary", "contemporary", "character", "society",
                "human condition", "introspective", "prose",
            ],
        ),
        "Historical Fiction": GenreInfo(
            name="Historical Fiction",
            parent="Fiction",
            aliases=["historical novel"],
            keywords=[
                "war", "century", "historical", "period", "era",
                "ancient", "medieval", "victorian", "civil war",
                "world war", "revolutionary",
            ],
        ),
        
        # Non-Fiction
        "Non-Fiction": GenreInfo(
            name="Non-Fiction",
            aliases=["nonfiction", "non fiction"],
            keywords=["true", "factual", "real"],
        ),
        "Biography": GenreInfo(
            name="Biography",
            parent="Non-Fiction",
            aliases=["memoir", "autobiography", "life story"],
            keywords=[
                "biography", "memoir", "life", "autobiography",
                "personal", "journey", "story of",
            ],
        ),
        "History": GenreInfo(
            name="History",
            parent="Non-Fiction",
            aliases=["historical"],
            keywords=[
                "history", "historical", "civilization", "empire",
                "revolution", "ancient", "war", "period",
            ],
        ),
        "Science": GenreInfo(
            name="Science",
            parent="Non-Fiction",
            aliases=["scientific"],
            keywords=[
                "science", "scientific", "physics", "biology",
                "chemistry", "evolution", "research", "experiment",
                "theory", "discovery", "nature",
            ],
        ),
        "Self-Help": GenreInfo(
            name="Self-Help",
            parent="Non-Fiction",
            aliases=["personal development", "self improvement"],
            keywords=[
                "self-help", "improve", "success", "habit", "mindset",
                "productivity", "motivation", "goal", "growth",
                "wellness", "happiness",
            ],
        ),
        "Business": GenreInfo(
            name="Business",
            parent="Non-Fiction",
            aliases=["finance", "economics"],
            keywords=[
                "business", "management", "leadership", "startup",
                "entrepreneur", "company", "market", "strategy",
                "investing", "money", "career",
            ],
        ),
        "Philosophy": GenreInfo(
            name="Philosophy",
            parent="Non-Fiction",
            aliases=["philosophical"],
            keywords=[
                "philosophy", "ethics", "moral", "existence",
                "consciousness", "meaning", "truth", "wisdom",
            ],
        ),
        "Psychology": GenreInfo(
            name="Psychology",
            parent="Non-Fiction",
            aliases=["psychological"],
            keywords=[
                "psychology", "mind", "behavior", "cognitive",
                "mental", "therapy", "brain", "emotion",
            ],
        ),
        
        # Other categories
        "Young Adult": GenreInfo(
            name="Young Adult",
            aliases=["YA", "teen"],
            keywords=[
                "young adult", "teen", "coming of age", "high school",
                "adolescent",
            ],
        ),
        "Children's": GenreInfo(
            name="Children's",
            aliases=["kids", "children"],
            keywords=[
                "children", "kids", "picture book", "fairy tale",
                "adventure", "animal", "bedtime",
            ],
        ),
        "Poetry": GenreInfo(
            name="Poetry",
            aliases=["poems", "verse"],
            keywords=[
                "poetry", "poem", "verse", "lyrical", "stanza",
            ],
        ),
        "Graphic Novel": GenreInfo(
            name="Graphic Novel",
            aliases=["comic", "manga"],
            keywords=[
                "graphic novel", "comic", "manga", "illustrated",
            ],
        ),
    }
    
    def __init__(self):
        """Initialize taxonomy with lookup indices."""
        self._build_indices()
    
    def _build_indices(self):
        """Build lookup indices for fast matching."""
        # Keyword to genre mapping
        self._keyword_to_genres: dict[str, list[str]] = defaultdict(list)
        
        for genre_name, info in self.GENRES.items():
            for keyword in info.keywords:
                self._keyword_to_genres[keyword.lower()].append(genre_name)
            
            # Also index aliases
            for alias in info.aliases:
                self._keyword_to_genres[alias.lower()].append(genre_name)
        
        # Build parent-child relationships
        self._children: dict[str, list[str]] = defaultdict(list)
        for genre_name, info in self.GENRES.items():
            if info.parent:
                self._children[info.parent].append(genre_name)
    
    def get_genre(self, name: str) -> Optional[GenreInfo]:
        """Get genre info by name or alias."""
        # Direct match
        if name in self.GENRES:
            return self.GENRES[name]
        
        # Alias match
        name_lower = name.lower()
        for genre_name, info in self.GENRES.items():
            if name_lower in [a.lower() for a in info.aliases]:
                return info
        
        return None
    
    def get_children(self, parent: str) -> list[str]:
        """Get child genres of a parent."""
        return self._children.get(parent, [])
    
    def get_parent(self, genre: str) -> Optional[str]:
        """Get parent genre."""
        info = self.GENRES.get(genre)
        return info.parent if info else None
    
    def normalize_genre(self, genre: str) -> Optional[str]:
        """Normalize genre name to canonical form."""
        info = self.get_genre(genre)
        return info.name if info else None


class GenreClassifier:
    """
    Rule-based genre classifier.
    
    Uses keyword matching and heuristics for
    fast, interpretable classification.
    """
    
    def __init__(self, taxonomy: Optional[GenreTaxonomy] = None):
        """
        Initialize classifier.
        
        Args:
            taxonomy: Genre taxonomy to use
        """
        self.taxonomy = taxonomy or GenreTaxonomy()
    
    def classify(
        self,
        title: str,
        description: Optional[str] = None,
        subjects: Optional[list[str]] = None,
        existing_genres: Optional[list[str]] = None,
    ) -> GenreClassification:
        """
        Classify a book into genres.
        
        Args:
            title: Book title
            description: Book description
            subjects: Subject headings (e.g., from Open Library)
            existing_genres: Pre-assigned genres to normalize
            
        Returns:
            GenreClassification result
        """
        genre_scores: Counter = Counter()
        
        # Process existing genres
        if existing_genres:
            for genre in existing_genres:
                normalized = self.taxonomy.normalize_genre(genre)
                if normalized:
                    genre_scores[normalized] += 2.0  # High weight
                else:
                    # Try keyword matching
                    matched = self._match_keywords(genre)
                    for g, score in matched.items():
                        genre_scores[g] += score * 1.5
        
        # Process subjects
        if subjects:
            for subject in subjects:
                matched = self._match_keywords(subject)
                for genre, score in matched.items():
                    genre_scores[genre] += score
        
        # Process title
        title_matches = self._match_keywords(title)
        for genre, score in title_matches.items():
            genre_scores[genre] += score * 0.5
        
        # Process description
        if description:
            desc_matches = self._match_keywords(description)
            for genre, score in desc_matches.items():
                genre_scores[genre] += score * 0.3
        
        # Normalize scores
        if genre_scores:
            max_score = max(genre_scores.values())
            normalized_scores = {
                genre: score / max_score
                for genre, score in genre_scores.items()
            }
        else:
            # Default to Fiction
            normalized_scores = {"Fiction": 0.5}
        
        # Get top genres
        sorted_genres = sorted(
            normalized_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        primary_genre = sorted_genres[0][0]
        primary_confidence = sorted_genres[0][1]
        
        secondary_genres = [
            (genre, score)
            for genre, score in sorted_genres[1:5]
            if score > 0.3
        ]
        
        return GenreClassification(
            primary_genre=primary_genre,
            confidence=primary_confidence,
            secondary_genres=secondary_genres,
            all_genres=normalized_scores,
            classification_method="rule_based",
        )
    
    def _match_keywords(self, text: str) -> dict[str, float]:
        """
        Match keywords in text to genres.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict mapping genre to match score
        """
        text_lower = text.lower()
        matches: Counter = Counter()
        
        for keyword, genres in self.taxonomy._keyword_to_genres.items():
            if keyword in text_lower:
                for genre in genres:
                    matches[genre] += 1.0
        
        return dict(matches)


class GenreAnalyzer:
    """
    High-level genre analysis for book collections.
    
    Provides:
    - Collection-wide genre statistics
    - Genre distribution visualization
    - Genre trend detection
    - Gap analysis (missing genres)
    """
    
    def __init__(
        self,
        classifier: Optional[GenreClassifier] = None,
        taxonomy: Optional[GenreTaxonomy] = None,
    ):
        """
        Initialize analyzer.
        
        Args:
            classifier: Genre classifier
            taxonomy: Genre taxonomy
        """
        self.taxonomy = taxonomy or GenreTaxonomy()
        self.classifier = classifier or GenreClassifier(self.taxonomy)
    
    def analyze_collection(
        self,
        books: list[dict],
    ) -> dict:
        """
        Analyze genre distribution of a collection.
        
        Args:
            books: List of book dicts with title, description, genres
            
        Returns:
            Analysis results dict
        """
        genre_counts: Counter = Counter()
        primary_genres: Counter = Counter()
        books_per_genre: dict[str, list[str]] = defaultdict(list)
        
        for book in books:
            # Classify if needed
            classification = self.classifier.classify(
                title=book.get("title", ""),
                description=book.get("description"),
                subjects=book.get("subjects"),
                existing_genres=book.get("genres"),
            )
            
            # Count primary genre
            primary_genres[classification.primary_genre] += 1
            books_per_genre[classification.primary_genre].append(
                book.get("id", book.get("title", "unknown"))
            )
            
            # Count all genres
            for genre in classification.all_genres:
                genre_counts[genre] += 1
        
        total_books = len(books)
        
        # Calculate percentages
        genre_distribution = {
            genre: {
                "count": count,
                "percentage": (count / total_books * 100) if total_books > 0 else 0,
                "books": books_per_genre.get(genre, [])[:5],  # Sample books
            }
            for genre, count in primary_genres.most_common()
        }
        
        # Find dominant and underrepresented genres
        dominant_genres = [
            genre for genre, count in primary_genres.most_common(3)
            if count >= total_books * 0.15
        ]
        
        # Identify gaps
        major_genres = {"Fiction", "Non-Fiction", "Science Fiction", "Fantasy",
                       "Mystery", "Biography", "History", "Science", "Self-Help"}
        missing_genres = major_genres - set(primary_genres.keys())
        
        return {
            "total_books": total_books,
            "unique_genres": len(genre_counts),
            "distribution": genre_distribution,
            "dominant_genres": dominant_genres,
            "missing_genres": list(missing_genres),
            "top_5_genres": primary_genres.most_common(5),
            "fiction_vs_nonfiction": self._fiction_ratio(books, primary_genres),
        }
    
    def _fiction_ratio(
        self,
        books: list[dict],
        primary_genres: Counter,
    ) -> dict:
        """Calculate fiction vs non-fiction ratio."""
        fiction_genres = {
            "Fiction", "Science Fiction", "Fantasy", "Mystery",
            "Thriller", "Romance", "Horror", "Literary Fiction",
            "Historical Fiction", "Young Adult",
        }
        nonfiction_genres = {
            "Non-Fiction", "Biography", "History", "Science",
            "Self-Help", "Business", "Philosophy", "Psychology",
        }
        
        fiction_count = sum(
            count for genre, count in primary_genres.items()
            if genre in fiction_genres
        )
        nonfiction_count = sum(
            count for genre, count in primary_genres.items()
            if genre in nonfiction_genres
        )
        
        total = fiction_count + nonfiction_count
        
        return {
            "fiction_count": fiction_count,
            "nonfiction_count": nonfiction_count,
            "fiction_percentage": (fiction_count / total * 100) if total > 0 else 0,
            "nonfiction_percentage": (nonfiction_count / total * 100) if total > 0 else 0,
        }
    
    def suggest_genres(
        self,
        current_genres: list[str],
        collection_analysis: dict,
    ) -> list[str]:
        """
        Suggest genres to explore based on collection.
        
        Args:
            current_genres: User's current preferred genres
            collection_analysis: Output of analyze_collection
            
        Returns:
            List of suggested genres to explore
        """
        suggestions = []
        
        # Suggest from missing major genres
        for genre in collection_analysis.get("missing_genres", []):
            # Find related genres user might like
            related = self._find_related_genres(genre, current_genres)
            if related:
                suggestions.append(genre)
        
        # Suggest sub-genres of dominant genres
        for dominant in collection_analysis.get("dominant_genres", []):
            children = self.taxonomy.get_children(dominant)
            for child in children:
                if child not in current_genres:
                    suggestions.append(child)
        
        return suggestions[:5]
    
    def _find_related_genres(
        self,
        genre: str,
        liked_genres: list[str],
    ) -> bool:
        """Check if genre is related to liked genres."""
        genre_info = self.taxonomy.GENRES.get(genre)
        if not genre_info:
            return False
        
        # Check if same parent
        for liked in liked_genres:
            liked_info = self.taxonomy.GENRES.get(liked)
            if liked_info:
                if genre_info.parent == liked_info.parent:
                    return True
                if genre_info.parent == liked:
                    return True
        
        return False
