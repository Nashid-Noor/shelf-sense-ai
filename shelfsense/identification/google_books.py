"""
Google Books API Client

Handles searching for books using the Google Books Volume API.
"""

import aiohttp
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger
import urllib.parse
import os

@dataclass
class BookMetadata:
    """Standardized book metadata from external source."""
    title: str
    authors: List[str]
    description: Optional[str]
    publisher: Optional[str]
    published_date: Optional[str]
    page_count: Optional[int]
    categories: List[str]
    average_rating: Optional[float]
    ratings_count: Optional[int]
    thumbnail_url: Optional[str]
    language: Optional[str]
    isbn_10: Optional[str] = None
    isbn_13: Optional[str] = None
    google_books_id: Optional[str] = None

class GoogleBooksClient:
    """Client for Google Books API."""
    
    BASE_URL = "https://www.googleapis.com/books/v1/volumes"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize client.
        
        Args:
            api_key: Optional API key. If not provided, tries GOOGLE_BOOKS_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_BOOKS_API_KEY")
        if not self.api_key:
            logger.warning("No Google Books API key provided. Rate limits will be lower.")
            
    async def search(self, query: str, max_results: int = 5) -> List[BookMetadata]:
        """
        Search for books.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of BookMetadata objects
        """
        if not query or not query.strip():
            return []
            
        params = {
            "q": query,
            "maxResults": min(max_results, 40),
            "printType": "books",
        }
        
        if self.api_key:
            params["key"] = self.api_key
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Google Books API error {resp.status}: {error_text}")
                        return []
                        
                    data = await resp.json()
                    
                    if "items" not in data:
                        return []
                        
                    results = []
                    for item in data["items"]:
                        metadata = self._parse_volume(item)
                        if metadata:
                            results.append(metadata)
                            
                    return results
                    
        except Exception as e:
            logger.error(f"Failed to search Google Books: {e}")
            return []
            
    def _parse_volume(self, item: Dict[str, Any]) -> Optional[BookMetadata]:
        """Parse raw API response into BookMetadata."""
        try:
            volume_info = item.get("volumeInfo", {})
            
            # Extract identifiers
            isbn_10 = None
            isbn_13 = None
            for identifier in volume_info.get("industryIdentifiers", []):
                if identifier["type"] == "ISBN_10":
                    isbn_10 = identifier["identifier"]
                elif identifier["type"] == "ISBN_13":
                    isbn_13 = identifier["identifier"]
            
            # Extract thumbnail
            image_links = volume_info.get("imageLinks", {})
            thumbnail = image_links.get("thumbnail") or image_links.get("smallThumbnail")
            if thumbnail:
                thumbnail = thumbnail.replace("http://", "https://")
            
            return BookMetadata(
                title=volume_info.get("title", "Unknown Title"),
                authors=volume_info.get("authors", []),
                description=volume_info.get("description"),
                publisher=volume_info.get("publisher"),
                published_date=volume_info.get("publishedDate"),
                page_count=volume_info.get("pageCount"),
                categories=volume_info.get("categories", []),
                average_rating=volume_info.get("averageRating"),
                ratings_count=volume_info.get("ratingsCount"),
                thumbnail_url=thumbnail,
                language=volume_info.get("language"),
                isbn_10=isbn_10,
                isbn_13=isbn_13,
                google_books_id=item.get("id")
            )
            
        except Exception as e:
            logger.warning(f"Error parsing volume data: {e}")
            return None
