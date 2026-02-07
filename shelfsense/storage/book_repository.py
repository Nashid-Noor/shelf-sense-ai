"""
Book Repository for ShelfSense AI

Structured storage for book metadata using SQLAlchemy:
- PostgreSQL for production
- SQLite for development/testing
- Full-text search support
- Relationship management (authors, genres, shelves)

Design Decisions:
1. SQLAlchemy ORM: Portable across databases
2. Soft deletes: Preserve history
3. Full-text search: PostgreSQL tsvector or SQLite FTS5
4. Denormalized fields: Avoid joins for common queries
"""

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
from loguru import logger

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    Index,
    func,
    or_,
)
from sqlalchemy.orm import (
    relationship,
    sessionmaker,
    Session,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from .models import Base, User


class BookModel(Base):
    """SQLAlchemy model for books."""
    
    __tablename__ = "books"
    
    # Primary key
    id = Column(String(64), primary_key=True)
    
    # Core fields
    title = Column(String(500), nullable=False, index=True)
    author = Column(String(500), nullable=False, index=True)
    
    # ISBNs
    isbn_10 = Column(String(10), index=True)
    isbn_13 = Column(String(13), index=True)
    
    # Publication
    publisher = Column(String(200))
    publication_year = Column(Integer, index=True)
    publication_date = Column(String(50))
    edition = Column(String(100))
    
    # Content
    page_count = Column(Integer)
    language = Column(String(10))
    description = Column(Text)
    
    # Classification (JSON array)
    genres = Column(JSON, default=list)
    subjects = Column(JSON, default=list)
    
    # Additional authors (JSON array)
    authors = Column(JSON, default=list)
    
    # Cover images
    cover_url_small = Column(String(500))
    cover_url_medium = Column(String(500))
    cover_url_large = Column(String(500))
    
    # External IDs
    openlibrary_id = Column(String(50))
    google_books_id = Column(String(50))
    goodreads_id = Column(String(50))
    
    # ShelfSense metadata
    added_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source = Column(String(50))  # "scan", "manual", "import"
    
    # Detection metadata
    detection_confidence = Column(Float)
    ocr_text = Column(Text)
    
    # Embedding flags
    has_text_embedding = Column(Boolean, default=False)
    has_visual_embedding = Column(Boolean, default=False)
    
    # User data
    shelf_location = Column(String(100))  # User's physical shelf
    user_rating = Column(Float)
    user_notes = Column(Text)
    read_status = Column(String(20))  # "unread", "reading", "read", "dnf"
    
    # Soft delete
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime)
    
    # Full-text search (PostgreSQL)
    # search_vector = Column(TSVECTOR)  # Uncomment for PostgreSQL
    
    # Indexes
    __table_args__ = (
        Index("idx_books_title_author", "title", "author"),
        Index("idx_books_isbn", "isbn_10", "isbn_13"),
        Index("idx_books_added", "added_at"),
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "isbn_10": self.isbn_10,
            "isbn_13": self.isbn_13,
            "publisher": self.publisher,
            "publication_year": self.publication_year,
            "publication_date": self.publication_date,
            "edition": self.edition,
            "page_count": self.page_count,
            "language": self.language,
            "description": self.description,
            "genres": self.genres or [],
            "subjects": self.subjects or [],
            "authors": self.authors or [],
            "cover_url_small": self.cover_url_small,
            "cover_url_medium": self.cover_url_medium,
            "cover_url_large": self.cover_url_large,
            "openlibrary_id": self.openlibrary_id,
            "google_books_id": self.google_books_id,
            "goodreads_id": self.goodreads_id,
            "added_at": self.added_at.isoformat() if self.added_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source": self.source,
            "detection_confidence": self.detection_confidence,
            "shelf_location": self.shelf_location,
            "user_rating": self.user_rating,
            "user_notes": self.user_notes,
            "read_status": self.read_status,
        }


@dataclass
class StoredBook:
    """Data class for book data transfer."""
    
    id: str
    title: str
    author: str
    
    # Optional fields
    isbn_10: Optional[str] = None
    isbn_13: Optional[str] = None
    publisher: Optional[str] = None
    publication_year: Optional[int] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    description: Optional[str] = None
    
    genres: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    
    cover_url: Optional[str] = None
    
    # Metadata
    added_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    source: Optional[str] = None
    detection_confidence: Optional[float] = None
    
    # User data
    shelf_location: Optional[str] = None
    user_rating: Optional[float] = None
    read_status: Optional[str] = None
    
    @classmethod
    def from_model(cls, model: BookModel) -> "StoredBook":
        """Create from SQLAlchemy model."""
        return cls(
            id=model.id,
            title=model.title,
            author=model.author,
            isbn_10=model.isbn_10,
            isbn_13=model.isbn_13,
            publisher=model.publisher,
            publication_year=model.publication_year,
            page_count=model.page_count,
            language=model.language,
            description=model.description,
            genres=model.genres or [],
            subjects=model.subjects or [],
            authors=model.authors or [],
            cover_url=model.cover_url_medium or model.cover_url_small,
            added_at=model.added_at,
            created_at=model.added_at,
            updated_at=model.updated_at,
            source=model.source,
            detection_confidence=model.detection_confidence,
            shelf_location=model.shelf_location,
            user_rating=model.user_rating,
            read_status=model.read_status,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "isbn_10": self.isbn_10,
            "isbn_13": self.isbn_13,
            "publisher": self.publisher,
            "publication_year": self.publication_year,
            "page_count": self.page_count,
            "language": self.language,
            "description": self.description,
            "genres": self.genres,
            "subjects": self.subjects,
            "authors": self.authors,
            "cover_url": self.cover_url,
            "added_at": self.added_at.isoformat() if self.added_at else None,
            "source": self.source,
            "detection_confidence": self.detection_confidence,
            "shelf_location": self.shelf_location,
            "user_rating": self.user_rating,
            "read_status": self.read_status,
        }


class BookRepository:
    """
    Repository for book metadata CRUD operations.
    
    Supports:
    - PostgreSQL (production)
    - SQLite (development/testing)
    - Full-text search
    - Filtering and pagination
    
    Usage:
        repo = BookRepository("postgresql://user:pass@localhost/shelfsense")
        
        # Add book
        book = repo.create(
            id="abc123",
            title="Harry Potter",
            author="J.K. Rowling",
        )
        
        # Search
        results = repo.search("harry potter")
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        sqlite_path: Optional[Path] = None,
    ):
        """
        Initialize repository.
        
        Args:
            database_url: SQLAlchemy database URL
            sqlite_path: Path for SQLite database
        """
        if database_url:
            # Strip async drivers for sync engine
            self.database_url = database_url.replace("+aiosqlite", "").replace("+asyncpg", "")
        elif sqlite_path:
            self.database_url = f"sqlite:///{sqlite_path}"
        else:
            # Default to in-memory SQLite
            self.database_url = "sqlite:///:memory:"
        
        # Create engine
        self.engine = create_engine(
            self.database_url,
            echo=False,  # Set to True for SQL logging
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info(f"BookRepository initialized: {self.database_url[:50]}...")
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def create(
        self,
        id: str,
        title: str,
        author: str,
        **kwargs,
    ) -> StoredBook:
        """
        Create a new book.
        
        Args:
            id: Unique book ID
            title: Book title
            author: Primary author
            **kwargs: Additional fields
            
        Returns:
            Created StoredBook
        """
        with self.get_session() as session:
            book = BookModel(
                id=id,
                title=title,
                author=author,
                **kwargs,
            )
            session.add(book)
            session.commit()
            session.refresh(book)
            
            return StoredBook.from_model(book)
    
    def get(self, book_id: str) -> Optional[StoredBook]:
        """
        Get book by ID.
        
        Args:
            book_id: Book ID
            
        Returns:
            StoredBook or None
        """
        with self.get_session() as session:
            book = session.query(BookModel).filter(
                BookModel.id == book_id,
                BookModel.is_deleted == False,
            ).first()
            
            if book:
                return StoredBook.from_model(book)
            return None
    
    def get_by_isbn(self, isbn: str) -> Optional[StoredBook]:
        """
        Get book by ISBN.
        
        Args:
            isbn: ISBN-10 or ISBN-13
            
        Returns:
            StoredBook or None
        """
        isbn = isbn.replace("-", "").replace(" ", "")
        
        with self.get_session() as session:
            book = session.query(BookModel).filter(
                or_(
                    BookModel.isbn_10 == isbn,
                    BookModel.isbn_13 == isbn,
                ),
                BookModel.is_deleted == False,
            ).first()
            
            if book:
                return StoredBook.from_model(book)
            return None
    
    def update(
        self,
        book_id: str,
        **updates,
    ) -> Optional[StoredBook]:
        """
        Update book fields.
        
        Args:
            book_id: Book ID
            **updates: Fields to update
            
        Returns:
            Updated StoredBook or None
        """
        with self.get_session() as session:
            book = session.query(BookModel).filter(
                BookModel.id == book_id,
            ).first()
            
            if not book:
                return None
            
            for key, value in updates.items():
                if hasattr(book, key):
                    setattr(book, key, value)
            
            book.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(book)
            
            return StoredBook.from_model(book)
    
    def delete(self, book_id: str, soft: bool = True) -> bool:
        """
        Delete a book.
        
        Args:
            book_id: Book ID
            soft: Soft delete (mark as deleted)
            
        Returns:
            True if deleted
        """
        with self.get_session() as session:
            book = session.query(BookModel).filter(
                BookModel.id == book_id,
            ).first()
            
            if not book:
                return False
            
            if soft:
                book.is_deleted = True
                book.deleted_at = datetime.utcnow()
            else:
                session.delete(book)
            
            session.commit()
            return True
    
    
    def list_books(
        self,
        page: int = 1,
        limit: int = 20,
        filters: dict = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[StoredBook], int]:
        """
        List books with filtering and pagination.
        
        Args:
            page: Page number (1-based)
            limit: Items per page
            filters: Filter dict (genre, author, read_status)
            sort_by: Field to sort by
            sort_order: "asc" or "desc"
            
        Returns:
            (List of StoredBooks, total_count)
        """
        filters = filters or {}
        offset = (page - 1) * limit
        
        with self.get_session() as session:
            query = session.query(BookModel).filter(
                BookModel.is_deleted == False,
            )
            
            # Apply filters
            if filters.get("genre"):
                # SQLite JSON filter
                query = query.filter(func.json_extract(BookModel.genres, "$").contains(filters["genre"]))
            
            if filters.get("author"):
                query = query.filter(BookModel.author.ilike(f"%{filters['author']}%"))
                
            if filters.get("read_status"):
                query = query.filter(BookModel.read_status == filters["read_status"])
            
            # Get total count before pagination
            total = query.count()
            
            # Apply sorting
            sort_field = getattr(BookModel, sort_by, BookModel.added_at)
            if sort_order == "desc":
                query = query.order_by(sort_field.desc())
            else:
                query = query.order_by(sort_field.asc())
            
            # Apply pagination
            books = query.offset(offset).limit(limit).all()
            
            return [StoredBook.from_model(b) for b in books], total

    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "added_at",
        descending: bool = True,
    ) -> list[StoredBook]:
        """
        List all books with pagination.
        
        Args:
            limit: Max results
            offset: Skip count
            sort_by: Field to sort by
            descending: Sort order
            
        Returns:
            List of StoredBooks
        """
        with self.get_session() as session:
            query = session.query(BookModel).filter(
                BookModel.is_deleted == False,
            )
            
            # Sort
            sort_col = getattr(BookModel, sort_by, BookModel.added_at)
            if descending:
                query = query.order_by(sort_col.desc())
            else:
                query = query.order_by(sort_col.asc())
            
            # Paginate
            books = query.offset(offset).limit(limit).all()
            
            return [StoredBook.from_model(b) for b in books]
    
    def search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[StoredBook]:
        """
        Search books by title/author.
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            Matching books
        """
        with self.get_session() as session:
            # Simple LIKE search (use full-text for production)
            pattern = f"%{query}%"
            
            books = session.query(BookModel).filter(
                BookModel.is_deleted == False,
                or_(
                    BookModel.title.ilike(pattern),
                    BookModel.author.ilike(pattern),
                    BookModel.isbn_10.ilike(pattern),
                    BookModel.isbn_13.ilike(pattern),
                ),
            ).limit(limit).all()
            
            return [StoredBook.from_model(b) for b in books]
    
    def filter_by_genre(
        self,
        genre: str,
        limit: int = 100,
    ) -> list[StoredBook]:
        """
        Filter books by genre.
        
        Args:
            genre: Genre to filter
            limit: Max results
            
        Returns:
            Matching books
        """
        with self.get_session() as session:
            # JSON array contains (SQLite-compatible)
            books = session.query(BookModel).filter(
                BookModel.is_deleted == False,
                func.json_extract(BookModel.genres, "$").contains(genre),
            ).limit(limit).all()
            
            return [StoredBook.from_model(b) for b in books]
    
    def filter_by_year(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        limit: int = 100,
    ) -> list[StoredBook]:
        """
        Filter books by publication year.
        
        Args:
            start_year: Minimum year (inclusive)
            end_year: Maximum year (inclusive)
            limit: Max results
            
        Returns:
            Matching books
        """
        with self.get_session() as session:
            query = session.query(BookModel).filter(
                BookModel.is_deleted == False,
            )
            
            if start_year:
                query = query.filter(BookModel.publication_year >= start_year)
            if end_year:
                query = query.filter(BookModel.publication_year <= end_year)
            
            books = query.limit(limit).all()
            
            return [StoredBook.from_model(b) for b in books]
    
    def get_stats(self) -> dict:
        """
        Get library statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.get_session() as session:
            total = session.query(func.count(BookModel.id)).filter(
                BookModel.is_deleted == False,
            ).scalar()
            
            with_isbn = session.query(func.count(BookModel.id)).filter(
                BookModel.is_deleted == False,
                or_(
                    BookModel.isbn_10.isnot(None),
                    BookModel.isbn_13.isnot(None),
                ),
            ).scalar()
            
            read = session.query(func.count(BookModel.id)).filter(
                BookModel.is_deleted == False,
                or_(
                    BookModel.read_status == "read",
                    BookModel.read_status == "completed"
                )
            ).scalar()
            
            unread = session.query(func.count(BookModel.id)).filter(
                BookModel.is_deleted == False,
                BookModel.read_status == "unread",
            ).scalar()
            
            reading = session.query(func.count(BookModel.id)).filter(
                BookModel.is_deleted == False,
                BookModel.read_status == "reading",
            ).scalar()
            
            return {
                "total_books": total,
                "books_with_isbn": with_isbn,
                "books_read": read,
                "books_unread": unread,
                "books_reading": reading,
            }
    
    def get_genre_distribution(self) -> dict[str, int]:
        """
        Get genre distribution.
        
        Returns:
            Dict of genre -> count
        """
        with self.get_session() as session:
            books = session.query(BookModel.genres).filter(
                BookModel.is_deleted == False,
                BookModel.genres.isnot(None),
            ).all()
            
            counts: dict[str, int] = {}
            for (genres,) in books:
                if genres:
                    for genre in genres:
                        counts[genre] = counts.get(genre, 0) + 1
            
            # Sort by count
            return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_author_stats(self, limit: int = 20) -> list[tuple[str, int]]:
        """
        Get most common authors.
        
        Args:
            limit: Max authors to return
            
        Returns:
            List of (author, count) tuples
        """
        with self.get_session() as session:
            results = session.query(
                BookModel.author,
                func.count(BookModel.id).label("count"),
            ).filter(
                BookModel.is_deleted == False,
            ).group_by(
                BookModel.author,
            ).order_by(
                func.count(BookModel.id).desc(),
            ).limit(limit).all()
            
            return [(author, count) for author, count in results]
    
    def bulk_create(self, books: list[dict]) -> int:
        """
        Bulk create books.
        
        Args:
            books: List of book dicts
            
        Returns:
            Number created
        """
        with self.get_session() as session:
            models = [BookModel(**book) for book in books]
            session.bulk_save_objects(models)
            session.commit()
            return len(models)
    
    def export_all(self, format: str = "json") -> str:
        """
        Export all books.
        
        Args:
            format: "json" or "csv"
            
        Returns:
            Exported data string
        """
        books = self.list_all(limit=10000)
        
        if format == "json":
            return json.dumps([b.to_dict() for b in books], indent=2)
        
        # CSV export would go here
        raise ValueError(f"Unsupported format: {format}")

    def get_publication_year_range(self) -> tuple[Optional[int], Optional[int]]:
        """
        Get min and max publication years.
        
        Returns:
            (min_year, max_year)
        """
        with self.get_session() as session:
            result = session.query(
                func.min(BookModel.publication_year),
                func.max(BookModel.publication_year)
            ).filter(
                BookModel.is_deleted == False,
                BookModel.publication_year.isnot(None),
                BookModel.publication_year > 1000, # Basic sanity check
                BookModel.publication_year <= datetime.now().year + 1
            ).first()
            
            return result if result else (None, None)

    def get_reading_trends_data(self) -> dict:
        """
        Get reading trend data (additions per month).
        
        Returns:
            Dictionary with genre trends and timeline
        """
        with self.get_session() as session:
            # 1. Timeline of additions (last 12 months)
            # SQLite specific date truncation for grouping
            # For cross-database compatibility, we'll fetch dates and group in Python
            books = session.query(
                BookModel.added_at,
                BookModel.genres
            ).filter(
                BookModel.is_deleted == False,
                BookModel.added_at.isnot(None)
            ).all()

            # Group by Month-Year
            from collections import defaultdict
            timeline = defaultdict(int)
            genre_trends = defaultdict(lambda: {"count": 0, "recent_count": 0})
            
            now = datetime.now()
            recent_cutoff = now.replace(month=now.month-3 if now.month > 3 else 12 + now.month - 3, year=now.year if now.month > 3 else now.year - 1)

            for added_at, genres in books:
                if not added_at: continue
                
                # Timeline
                key = added_at.strftime("%Y-%m")
                timeline[key] += 1
                
                # Genre trends
                if genres:
                    for genre in genres:
                        genre_trends[genre]["count"] += 1
                        if added_at >= recent_cutoff:
                            genre_trends[genre]["recent_count"] += 1

            # Determine trends (increasing/decreasing)
            formatted_trends = {}
            for genre, data in genre_trends.items():
                total = data["count"]
                if total < 2: continue # Ignore small samples
                
                # Check if disproportionately recent
                # heuristic: if > 40% of books in this genre were added in last 3 months
                recent_ratio = data["recent_count"] / total
                
                if recent_ratio > 0.4:
                    trend = "increasing"
                    change = int(data["recent_count"] * 10) # arbitrary score
                elif recent_ratio < 0.1:
                    trend = "decreasing"
                    change = -10
                else:
                    trend = "stable"
                    change = 0
                    
                formatted_trends[genre] = {
                    "trend": trend,
                    "change_percent": change,
                    "count": total
                }
            
            return {
                "timeline": dict(timeline),
                "genre_trends": dict(sorted(formatted_trends.items(), key=lambda x: x[1]['count'], reverse=True)[:10])
            }

    def get_diversity_stats(self) -> dict:
        """
        Calculate diversity statistics for library.
        
        Returns:
            Dictionary with entropy scores for genres, authors, etc.
        """
        import math
        
        with self.get_session() as session:
            # 1. Author Diversity
            total_books = session.query(func.count(BookModel.id)).filter(BookModel.is_deleted == False).scalar() or 1
            unique_authors = session.query(func.count(func.distinct(BookModel.author))).filter(BookModel.is_deleted == False).scalar() or 0
            
            # Author concentration (simple metric: unique / total)
            author_score = min(1.0, unique_authors / (total_books * 0.8)) # Expecting some repetition is normal
            
            # 2. Genre Diversity (Entropy)
            genre_counts = self.get_genre_distribution()
            total_genres_assigned = sum(genre_counts.values()) or 1
            
            genre_entropy = 0
            for count in genre_counts.values():
                p = count / total_genres_assigned
                if p > 0:
                    genre_entropy -= p * math.log2(p)
            
            # Normalize entropy (assuming max 10 reasonable genres -> log2(10) ≈ 3.32)
            # A score of 1.0 would mean perfect even distribution across ~8-10 genres
            genre_score = min(1.0, genre_entropy / 3.0)
            
            # 3. Temporal Diversity
            years = session.query(BookModel.publication_year).filter(
                BookModel.is_deleted == False,
                BookModel.publication_year.isnot(None)
            ).all()
            
            years = [y[0] for y in years if y[0] and 1800 < y[0] < 2030]
            
            if len(years) > 1:
                # Standard deviation
                mean_year = sum(years) / len(years)
                variance = sum((y - mean_year) ** 2 for y in years) / len(years)
                std_dev = math.sqrt(variance)
                
                # Score: 1.0 if std_dev >= 30 years, 0.0 if 0
                temporal_score = min(1.0, std_dev / 30.0)
            else:
                temporal_score = 0
                
            # Overall Score
            overall = (author_score * 0.3) + (genre_score * 0.4) + (temporal_score * 0.3)
            
            def get_grade(score):
                if score >= 0.8: return "A"
                if score >= 0.6: return "B"
                if score >= 0.4: return "C"
                return "D"

            return {
                "overall_score": round(overall, 2),
                "overall_grade": get_grade(overall),
                "author_diversity": {
                    "score": round(author_score, 2),
                    "grade": get_grade(author_score),
                    "unique_authors": unique_authors,
                    "interpretation": "High author variety" if author_score > 0.7 else "Frequent author repetition"
                },
                "genre_diversity": {
                    "score": round(genre_score, 2),
                    "grade": get_grade(genre_score),
                    "unique_genres": len(genre_counts),
                    "interpretation": "Wide genre coverage" if genre_score > 0.7 else "Focused on few genres"
                },
                "temporal_diversity": {
                    "score": round(temporal_score, 2),
                    "grade": get_grade(temporal_score),
                    "interpretation": "Spans many eras" if temporal_score > 0.6 else "Concentrated in one period"
                }
            }
