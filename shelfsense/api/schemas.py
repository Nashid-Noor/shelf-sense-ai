"""
API Schemas for ShelfSense AI

Pydantic models for request validation and response serialization:
- Book models
- Detection models
- Chat/RAG models
- Analytics models

Design Decisions:
1. Strict validation: Use Pydantic's validation for all inputs
2. Separate Request/Response: Clear distinction between inputs and outputs
3. Optional fields: Gracefully handle partial data
4. Examples: OpenAPI documentation with realistic examples
"""

from datetime import datetime
from typing import Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict


# =============================================================================
# Enums
# =============================================================================

class ReadStatus(str, Enum):
    """Book reading status."""
    UNREAD = "unread"
    READING = "reading"
    COMPLETED = "completed"
    DNF = "did_not_finish"


class DetectionMode(str, Enum):
    """Image detection mode."""
    AUTO = "auto"
    SHELF = "shelf"
    COVER = "cover"
    SPINE = "spine"


class SearchMode(str, Enum):
    """Search mode for retrieval."""
    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"


# =============================================================================
# Book Schemas
# =============================================================================

class BookBase(BaseModel):
    """Base book fields."""
    
    title: str = Field(..., min_length=1, max_length=500)
    author: str = Field(..., min_length=1, max_length=200)
    
    isbn_10: Optional[str] = Field(None, pattern=r"^\d{10}$")
    isbn_13: Optional[str] = Field(None, pattern=r"^\d{13}$")
    
    genres: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    
    publication_year: Optional[int] = Field(None, ge=1000, le=2100)
    publisher: Optional[str] = None
    language: Optional[str] = Field(None, max_length=50)
    page_count: Optional[int] = Field(None, ge=1)
    
    description: Optional[str] = None
    cover_url: Optional[str] = None


class BookCreate(BookBase):
    """Book creation request."""
    
    # User-specific fields
    shelf_location: Optional[str] = None
    user_notes: Optional[str] = None
    read_status: ReadStatus = ReadStatus.UNREAD
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Dune",
                "author": "Frank Herbert",
                "isbn_13": "9780441172719",
                "genres": ["Science Fiction", "Space Opera"],
                "publication_year": 1965,
                "shelf_location": "Living Room - Shelf 2",
            }
        }
    )


class BookUpdate(BaseModel):
    """Book update request (partial)."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    author: Optional[str] = Field(None, min_length=1, max_length=200)
    
    genres: Optional[list[str]] = None
    subjects: Optional[list[str]] = None
    
    publication_year: Optional[int] = Field(None, ge=1000, le=2100)
    description: Optional[str] = None
    
    shelf_location: Optional[str] = None
    user_notes: Optional[str] = None
    read_status: Optional[ReadStatus] = None
    user_rating: Optional[int] = Field(None, ge=1, le=5)


class BookResponse(BookBase):
    """Book response model."""
    
    id: str
    created_at: datetime
    updated_at: datetime
    
    # User-specific
    shelf_location: Optional[str] = None
    user_notes: Optional[str] = None
    read_status: ReadStatus = ReadStatus.UNREAD
    user_rating: Optional[int] = None
    
    # Detection metadata
    detection_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)


class BookListResponse(BaseModel):
    """Paginated book list response."""
    
    books: list[BookResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class BookSearchRequest(BaseModel):
    """Book search request."""
    
    query: str = Field(..., min_length=1, max_length=500)
    mode: SearchMode = SearchMode.HYBRID
    
    # Filters
    genres: Optional[list[str]] = None
    authors: Optional[list[str]] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    read_status: Optional[ReadStatus] = None
    
    # Pagination
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class BookSearchResult(BaseModel):
    """Single search result."""
    
    book: BookResponse
    score: float
    match_type: str  # "title", "author", "content", "hybrid"


class BookSearchResponse(BaseModel):
    """Book search response."""
    
    results: list[BookSearchResult]
    query: str
    total_results: int
    search_time_ms: float


# =============================================================================
# Detection Schemas
# =============================================================================

class DetectionRequest(BaseModel):
    """Image detection configuration."""
    
    mode: DetectionMode = DetectionMode.AUTO
    
    # Detection settings
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_detections: int = Field(50, ge=1, le=200)
    
    # Processing options
    auto_identify: bool = True
    enrich_metadata: bool = True
    
    # OCR settings
    ocr_languages: list[str] = Field(default=["en"])


class DetectedBook(BaseModel):
    """A book detected in an image."""
    
    # Bounding box (normalized 0-1)
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    
    # Detection info
    detection_type: str  # "spine", "cover"
    detection_confidence: float
    
    # OCR results
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    
    # Identification (if matched)
    identified: bool = False
    book_id: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    identification_confidence: Optional[float] = None
    
    # Metadata for enrichment
    genres: list[str] = Field(default_factory=list)
    publication_year: Optional[int] = None
    isbn_13: Optional[str] = None
    description: Optional[str] = None
    publisher: Optional[str] = None
    cover_url: Optional[str] = None
    
    # Cropped image (base64)
    crop_image: Optional[str] = None


class DetectionResponse(BaseModel):
    """Image detection response."""
    
    # Detection results
    detected_books: list[DetectedBook]
    
    # Metadata
    image_width: int
    image_height: int
    layout_type: str  # "shelf", "single_book", "stack", "unknown"
    
    # Timing
    detection_time_ms: float
    ocr_time_ms: float
    identification_time_ms: float
    total_time_ms: float
    
    # Summary
    total_detected: int
    total_identified: int
    average_confidence: float


class BatchDetectionRequest(BaseModel):
    """Batch image detection request."""
    
    images: list[str] = Field(..., min_length=1, max_length=10)  # Base64 or URLs
    settings: DetectionRequest = Field(default_factory=DetectionRequest)


class BatchDetectionResponse(BaseModel):
    """Batch detection response."""
    
    results: list[DetectionResponse]
    total_images: int
    total_books_detected: int
    total_time_ms: float


# =============================================================================
# Chat/RAG Schemas
# =============================================================================

class ChatMessage(BaseModel):
    """A single chat message."""
    
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Chat request."""
    
    message: str = Field(..., min_length=1, max_length=5000)
    
    # Conversation context
    conversation_id: Optional[str] = None
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    
    # RAG settings
    include_context: bool = True
    max_context_books: int = Field(10, ge=1, le=20)
    
    # Generation settings
    stream: bool = False
    max_tokens: int = Field(1000, ge=100, le=4000)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "What science fiction books do I have?",
                "include_context": True,
                "stream": False,
            }
        }
    )


class Citation(BaseModel):
    """A citation in a response."""
    
    title: str
    author: str
    book_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response."""
    
    response: str
    citations: list[Citation] = Field(default_factory=list)
    
    # Context used
    books_referenced: int
    
    # Conversation
    conversation_id: str
    
    # Metadata
    response_time_ms: float
    tokens_used: int


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    
    type: str  # "content", "citation", "done", "error"
    content: Optional[str] = None
    citation: Optional[Citation] = None
    error: Optional[str] = None


# =============================================================================
# Analytics Schemas
# =============================================================================

class GenreDistribution(BaseModel):
    """Genre distribution statistics."""
    
    genre: str
    count: int
    percentage: float


class LibraryStatsResponse(BaseModel):
    """Library statistics response."""
    
    total_books: int
    total_read: int
    total_unread: int
    
    # Genre breakdown
    genre_distribution: list[GenreDistribution]
    top_genres: list[str]
    
    # Author stats
    unique_authors: int
    top_authors: list[dict[str, Any]]  # {"author": str, "count": int}
    
    # Temporal
    oldest_book_year: Optional[int]
    newest_book_year: Optional[int]
    average_publication_year: Optional[int]
    
    # Diversity
    diversity_score: float
    diversity_grade: str


class DiversityResponse(BaseModel):
    """Detailed diversity analysis."""
    
    overall_score: float
    overall_grade: str
    
    genre_diversity: dict[str, Any]
    author_diversity: dict[str, Any]
    temporal_diversity: dict[str, Any]
    
    recommendations: list[str]


class RecommendationResponse(BaseModel):
    """Book recommendations."""
    
    recommendations: list[dict[str, Any]]
    based_on: Optional[str] = None  # Book ID or "library_profile"
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendations": [
                    {
                        "title": "Neuromancer",
                        "author": "William Gibson",
                        "type": "similar",
                        "reason": "Similar themes to Dune",
                        "confidence": 0.85,
                    }
                ],
                "based_on": "library_profile",
            }
        }
    )


# =============================================================================
# Error Schemas
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str
    detail: Optional[str] = None
    code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Book not found",
                "detail": "No book with ID 'abc123' exists",
                "code": "NOT_FOUND",
                "timestamp": "2025-01-20T12:00:00Z",
            }
        }
    )


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    
    error: str = "Validation Error"
    detail: list[dict[str, Any]]
    code: str = "VALIDATION_ERROR"


# =============================================================================
# Health Check
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    version: str
    uptime_seconds: float
    
    # Component status
    database: str = "connected"
    vector_store: str = "connected"
    llm_provider: str = "connected"
    
    # Resource usage
    memory_mb: Optional[float] = None
    gpu_available: bool = False
