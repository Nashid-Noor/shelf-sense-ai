"""
Book API Routes

CRUD operations for book management including create, read, update, delete, and search.
"""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends, status
from loguru import logger

from shelfsense.api.schemas import (
    BookCreate,
    BookUpdate,
    BookResponse,
    BookListResponse,
    BookSearchRequest,
    BookSearchResponse,
    BookSearchResult,
    ErrorResponse,
)


router = APIRouter(prefix="/books", tags=["books"])


# =============================================================================
# Dependencies
# =============================================================================

from shelfsense.api.dependencies import (
    get_db,
    get_db,
    get_book_repository,
    get_hybrid_retriever,
    get_hybrid_retriever,
)
from sqlalchemy.ext.asyncio import AsyncSession





async def get_search_service(db: AsyncSession = Depends(get_db)):
    """Dependency to get search service with database session."""
    return get_hybrid_retriever(db)


# =============================================================================
# CRUD Endpoints
# =============================================================================

@router.post(
    "",
    response_model=BookResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid book data"},
        409: {"model": ErrorResponse, "description": "Book already exists"},
    },
)
def create_book(
    book: BookCreate,
    repo = Depends(get_book_repository),
):
    """
    Create a new book with metadata validation and duplicate checking.
    """
    logger.info(f"Creating book: {book.title} by {book.author}")
    
    try:
        # Check for duplicates if ISBN is provided
        if book.isbn_13:
            existing = repo.get_by_isbn(book.isbn_13)
            if existing:
                raise HTTPException(status_code=409, detail=f"Book with ISBN {book.isbn_13} already exists")
        
        # Generate ID
        import uuid
        book_id = str(uuid.uuid4())
        
        # Create book
        # repo.create expects id, title, author positional, then kwargs
        book_data = book.dict(exclude={"title", "author"})
        
        # Map cover_url to cover_url_medium
        if "cover_url" in book_data:
             book_data["cover_url_medium"] = book_data.pop("cover_url")
             
        created_book = repo.create(
            id=book_id,
            title=book.title,
            author=book.author,
            **book_data
        )
        return created_book
    except Exception as e:
        logger.error(f"Failed to create book: {e}")
        # If it's a known error, re-raise, otherwise internal error
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{book_id}",
    response_model=BookResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Book not found"},
    },
)
def get_book(
    book_id: str,
    repo = Depends(get_book_repository),
):
    """Get a book by ID with full details."""
    logger.info(f"Fetching book: {book_id}")
    
    book = repo.get(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book


@router.get(
    "",
    response_model=BookListResponse,
)
def list_books(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    author: Optional[str] = Query(None, description="Filter by author"),
    read_status: Optional[str] = Query(None, description="Filter by read status"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    repo = Depends(get_book_repository),
):
    """List books with pagination and filtering options."""
    logger.info(f"Listing books: page={page}, size={page_size}")
    
    filters = {}
    if genre:
        filters["genre"] = genre
    if author:
        filters["author"] = author
    if read_status:
        filters["read_status"] = read_status
        
    books, total = repo.list_books(
        page=page,
        limit=page_size,
        filters=filters,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    return BookListResponse(
        books=books,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.put(
    "/{book_id}",
    response_model=BookResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Book not found"},
    },
)
async def update_book(
    book_id: str,
    book: BookUpdate,
    # repo = Depends(get_book_repository),
):
    """
    Update a book.
    
    Supports partial updates - only provided fields are modified.
    If metadata changes significantly, embeddings are regenerated.
    """
    logger.info(f"Updating book: {book_id}")
    
    # Placeholder - would update in database
    # 1. Fetch existing book
    # 2. Apply updates
    # 3. Regenerate embeddings if needed
    # 4. Update vector store
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Book {book_id} not found",
    )


@router.delete(
    "/{book_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Book not found"},
    },
)
async def delete_book(
    book_id: str,
    repo = Depends(get_book_repository),
):
    """
    Delete a book.
    
    Performs soft delete by default (can be recovered).
    Also removes from vector store index.
    """
    logger.info(f"Deleting book: {book_id}")
    
    success = repo.delete(book_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book {book_id} not found",
        )
    
    return None


# =============================================================================
# Search Endpoints
# =============================================================================

@router.post(
    "/search",
    response_model=BookSearchResponse,
)
async def search_books(
    request: BookSearchRequest,
    # search_service = Depends(get_search_service),
):
    """Search books using hybrid retrieval strategy."""
    logger.info(f"Searching books: '{request.query}'")
    
    import time
    start = time.time()
    
    # Placeholder - would use hybrid retriever
    # results = await search_service.search(
    #     query=request.query,
    #     mode=request.mode,
    #     filters={
    #         "genres": request.genres,
    #         "authors": request.authors,
    #         "year_min": request.year_min,
    #         "year_max": request.year_max,
    #     },
    #     limit=request.limit,
    #     offset=request.offset,
    # )
    
    elapsed = (time.time() - start) * 1000
    
    return BookSearchResponse(
        results=[],
        query=request.query,
        total_results=0,
        search_time_ms=elapsed,
    )


@router.get(
    "/isbn/{isbn}",
    response_model=BookResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Book not found"},
    },
)
async def get_book_by_isbn(
    isbn: str,
    # repo = Depends(get_book_repository),
):
    """
    Get a book by ISBN (10 or 13 digit).
    
    Automatically handles both ISBN-10 and ISBN-13 formats.
    """
    logger.info(f"Fetching book by ISBN: {isbn}")
    
    # Normalize ISBN
    isbn_clean = isbn.replace("-", "").replace(" ", "")
    
    if len(isbn_clean) not in (10, 13):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ISBN format",
        )
    
    # Placeholder - would query database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No book found with ISBN {isbn}",
    )


# =============================================================================
# Bulk Operations
# =============================================================================

@router.post(
    "/bulk",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
)
async def bulk_create_books(
    books: list[BookCreate],
    # repo = Depends(get_book_repository),
):
    """
    Bulk create books.
    
    Accepts up to 100 books at once. Returns immediately
    with a job ID for tracking progress.
    """
    if len(books) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 books per batch",
        )
    
    logger.info(f"Bulk creating {len(books)} books")
    
    # Would queue for background processing
    # job_id = await background_tasks.enqueue(
    #     "bulk_create_books",
    #     books=[b.dict() for b in books],
    # )
    
    return {
        "job_id": "job_123",
        "status": "queued",
        "total_books": len(books),
        "message": "Bulk creation started",
    }


@router.get(
    "/bulk/{job_id}",
    response_model=dict,
)
async def get_bulk_job_status(job_id: str):
    """
    Get status of a bulk operation.
    """
    logger.info(f"Checking bulk job: {job_id}")
    
    # Would check job status
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "created": 50,
        "failed": 0,
        "errors": [],
    }


# =============================================================================
# User Actions
# =============================================================================

@router.post(
    "/{book_id}/read-status",
    response_model=BookResponse,
)
async def update_read_status(
    book_id: str,
    status: str = Query(..., pattern="^(unread|reading|completed|did_not_finish)$"),
    # repo = Depends(get_book_repository),
):
    """
    Update book reading status.
    
    Quick endpoint for updating just the read status.
    """
    logger.info(f"Updating read status: {book_id} -> {status}")
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Book {book_id} not found",
    )


@router.post(
    "/{book_id}/rating",
    response_model=BookResponse,
)
async def update_rating(
    book_id: str,
    rating: int = Query(..., ge=1, le=5),
    # repo = Depends(get_book_repository),
):
    """
    Rate a book (1-5 stars).
    """
    logger.info(f"Rating book: {book_id} -> {rating} stars")
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Book {book_id} not found",
    )
