"""
Analytics API Routes for ShelfSense AI

Library insights and statistics:
- Library statistics
- Diversity analysis
- Recommendations
- Reading trends
"""

from fastapi import APIRouter, Query, Depends
from loguru import logger
from shelfsense.api.dependencies import get_book_repository
from shelfsense.storage.book_repository import BookRepository

from shelfsense.api.schemas import (
    LibraryStatsResponse,
    GenreDistribution,
    DiversityResponse,
    RecommendationResponse,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# =============================================================================
# Library Statistics
# =============================================================================

@router.get(
    "/stats",
    response_model=LibraryStatsResponse,
)
def get_library_stats(
    repo: BookRepository = Depends(get_book_repository),
):
    """
    Get library statistics overview.
    
    Includes:
    - Total books and read/unread counts
    - Genre distribution
    - Top authors
    - Temporal spread
    - Overall diversity score
    """
    logger.info("Fetching library stats")
    
    # Fetch real stats from database
    stats = repo.get_stats()
    genre_counts = repo.get_genre_distribution()
    top_authors_list = repo.get_author_stats(limit=5)
    
    # Calculate genre distribution objects
    total_books = stats["total_books"] or 1 # Avoid division by zero
    genre_dist = []
    
    for genre, count in list(genre_counts.items())[:6]:  # Top 6
        genre_dist.append(
            GenreDistribution(
                genre=genre, 
                count=count, 
                percentage=round((count / total_books) * 100, 1)
            )
        )
        
    # Format authors
    formatted_authors = [
        {"author": author, "count": count} 
        for author, count in top_authors_list
    ]
    
    
    # Fetch year range
    min_year, max_year = repo.get_publication_year_range()
    
    # Calculate diversity score (simplified)
    diversity = repo.get_diversity_stats()
    
    return LibraryStatsResponse(
        total_books=stats["total_books"],
        total_read=stats["books_read"],
        total_unread=stats["books_unread"],
        genre_distribution=genre_dist,
        top_genres=list(genre_counts.keys())[:3],
        unique_authors=len(top_authors_list), 
        top_authors=formatted_authors,
        oldest_book_year=min_year,
        newest_book_year=max_year,
        average_publication_year=None, # Todo: implementations
        diversity_score=diversity["overall_score"],
        diversity_grade=diversity["overall_grade"],
    )



@router.get(
    "/diversity",
    response_model=DiversityResponse,
)
def get_diversity_analysis(
    repo: BookRepository = Depends(get_book_repository),
):
    """
    Get detailed diversity analysis.
    
    Analyzes your library across multiple dimensions:
    - Genre diversity (Shannon entropy)
    - Author diversity (concentration index)
    - Temporal diversity (publication year spread)
    
    Provides actionable recommendations for improvement.
    """
    logger.info("Calculating diversity metrics")
    
    # Fetch real diversity stats
    stats = repo.get_diversity_stats()
    
    # Generate dynamic recommendations
    recommendations = []
    
    # Genre gaps
    if stats["genre_diversity"]["score"] < 0.5:
        recommendations.append("Consider exploring new genres to broaden your horizons")
    
    # Temporal gaps
    if stats["temporal_diversity"]["score"] < 0.4:
         recommendations.append("Your collection is concentrated in a specific era - try older classics or modern releases")
         
    # Default if doing well
    if not recommendations:
        recommendations.append("Great diversity! Keep exploring new authors.")

    return DiversityResponse(
        overall_score=stats["overall_score"],
        overall_grade=stats["overall_grade"],
        genre_diversity=stats["genre_diversity"],
        author_diversity=stats["author_diversity"],
        temporal_diversity=stats["temporal_diversity"],
        recommendations=recommendations,
    )


from shelfsense.api.dependencies import get_recommender
from shelfsense.intelligence.recommender import BookRecommender, BookData, RecommendationRequest

@router.get(
    "/recommendations",
    response_model=RecommendationResponse,
)
def get_recommendations(
    count: int = Query(5, ge=1, le=20),
    include_exploration: bool = Query(True, description="Include genre-expanding suggestions"),
    based_on_book_id: str = Query(None, description="Get recommendations similar to a specific book"),
    repo: BookRepository = Depends(get_book_repository),
    recommender: BookRecommender = Depends(get_recommender),
):
    """
    Get personalized book recommendations.
    
    Combines multiple strategies:
    - Similar to books you own
    - More from authors you like
    - Genre favorites
    - Exploration (expand your horizons)
    
    Each recommendation includes an explanation.
    """
    logger.info(f"Generating {count} recommendations")
    
    # 1. Fetch user's library
    # We fetch enough books to build a good profile
    owned_books = repo.list_all(limit=1000)
    
    # 2. Convert to BookData for recommender
    book_data_list = []
    for book in owned_books:
        book_data_list.append(
            BookData(
                id=book.id,
                title=book.title,
                author=book.author,
                genres=book.genres,
                subjects=book.subjects,
                publication_year=book.publication_year,
                description=book.description,
            )
        )
        
    # 3. Build library profile
    profile = recommender.build_profile(book_data_list)
    
    # 4. Create request
    req = RecommendationRequest(
        count=count,
        include_exploration=include_exploration,
        based_on_book_id=based_on_book_id,
        exclude_owned=True,
    )
    
    # 5. Generate recommendations
    recs = recommender.recommend(profile, req)
    
    # 6. Format response
    formatted_recs = [rec.to_dict() for rec in recs]
    
    return RecommendationResponse(
        recommendations=formatted_recs,
        based_on="library_profile" if not based_on_book_id else based_on_book_id,
    )


# =============================================================================
# Reading Trends
# =============================================================================

@router.get(
    "/trends",
    response_model=dict,
)
def get_reading_trends(
    repo: BookRepository = Depends(get_book_repository),
):
    """
    Analyze reading trends over time.
    
    Shows how your reading habits have evolved.
    """
    logger.info("Analyzing reading trends")
    
    trends_data = repo.get_reading_trends_data()
    
    return {
        "monthly_additions": trends_data["timeline"],
        "genre_trends": trends_data["genre_trends"],
        "era_preferences": {
            "2020s": "28%", # Still placeholder until era analysis implemented
            "2010s": "22%",
            "2000s": "18%",
            "1990s and earlier": "32%",
        },
        "reading_pace": {
            "books_per_month": 4.2,
            "average_page_count": 312,
        },
    }


@router.get(
    "/genre/{genre}",
    response_model=dict,
)
async def get_genre_details(genre: str):
    """
    Get detailed analysis of a specific genre in your library.
    """
    logger.info(f"Analyzing genre: {genre}")
    
    return {
        "genre": genre,
        "book_count": 44,
        "percentage": 34.6,
        "authors": [
            {"name": "Isaac Asimov", "count": 7},
            {"name": "Philip K. Dick", "count": 4},
            {"name": "Ursula K. Le Guin", "count": 5},
        ],
        "sub_genres": [
            {"name": "Space Opera", "count": 12},
            {"name": "Cyberpunk", "count": 8},
            {"name": "Dystopian", "count": 6},
        ],
        "era_distribution": {
            "Pre-1970": 15,
            "1970-1990": 12,
            "1990-2010": 10,
            "2010+": 7,
        },
        "recommendations": [
            {
                "title": "Children of Time",
                "author": "Adrian Tchaikovsky",
                "reason": "Modern SF you might enjoy",
            },
        ],
    }


@router.get(
    "/author/{author_name}",
    response_model=dict,
)
async def get_author_details(author_name: str):
    """
    Get details about books by a specific author in your library.
    """
    logger.info(f"Fetching author details: {author_name}")
    
    return {
        "author": author_name,
        "books_owned": 7,
        "books": [
            {
                "title": "Foundation",
                "publication_year": 1951,
                "read_status": "completed",
                "user_rating": 5,
            },
            {
                "title": "I, Robot",
                "publication_year": 1950,
                "read_status": "completed",
                "user_rating": 4,
            },
        ],
        "genres": ["Science Fiction"],
        "total_available": 506,  # Total books by author
        "missing_notable": [
            {"title": "The Caves of Steel", "publication_year": 1954},
            {"title": "The Gods Themselves", "publication_year": 1972},
        ],
    }
