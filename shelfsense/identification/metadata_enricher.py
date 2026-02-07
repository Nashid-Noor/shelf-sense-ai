"""
Metadata Enricher

Fetches and enriches book metadata from Open Library and Google Books.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import httpx
from loguru import logger


@dataclass
class BookMetadata:
    """
    Book metadata from external sources.
    
    Combines data from multiple sources.
    """
    
    # Core identifiers
    title: str
    authors: list[str] = field(default_factory=list)
    isbn_10: Optional[str] = None
    isbn_13: Optional[str] = None
    
    # Publication info
    publisher: Optional[str] = None
    publish_date: Optional[str] = None
    publish_year: Optional[int] = None
    
    # Description
    description: Optional[str] = None
    subjects: list[str] = field(default_factory=list)
    
    # Physical
    page_count: Optional[int] = None
    language: str = "en"
    
    # Media
    cover_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    
    # External IDs
    open_library_id: Optional[str] = None
    google_books_id: Optional[str] = None
    
    # Metadata about the metadata
    source: str = "unknown"
    fetched_at: Optional[datetime] = None
    confidence: float = 1.0
    
    @property
    def primary_author(self) -> str:
        """Get primary author name."""
        if self.authors:
            return self.authors[0]
        return "Unknown"
    
    @property
    def primary_isbn(self) -> Optional[str]:
        """Get primary ISBN (prefer ISBN-13)."""
        return self.isbn_13 or self.isbn_10
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "isbn_10": self.isbn_10,
            "isbn_13": self.isbn_13,
            "publisher": self.publisher,
            "publish_date": self.publish_date,
            "publish_year": self.publish_year,
            "description": self.description,
            "subjects": self.subjects,
            "page_count": self.page_count,
            "language": self.language,
            "cover_url": self.cover_url,
            "thumbnail_url": self.thumbnail_url,
            "open_library_id": self.open_library_id,
            "google_books_id": self.google_books_id,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BookMetadata':
        """Create from dictionary."""
        return cls(
            title=data.get("title", "Unknown"),
            authors=data.get("authors", []),
            isbn_10=data.get("isbn_10"),
            isbn_13=data.get("isbn_13"),
            publisher=data.get("publisher"),
            publish_date=data.get("publish_date"),
            publish_year=data.get("publish_year"),
            description=data.get("description"),
            subjects=data.get("subjects", []),
            page_count=data.get("page_count"),
            language=data.get("language", "en"),
            cover_url=data.get("cover_url"),
            thumbnail_url=data.get("thumbnail_url"),
            open_library_id=data.get("open_library_id"),
            google_books_id=data.get("google_books_id"),
            source=data.get("source", "dict"),
        )


class OpenLibraryClient:
    """
    Client for Open Library API.
    
    Open Library is a free, open-source library catalog.
    Rate limits: Be respectful, no official limit but don't abuse.
    """
    
    BASE_URL = "https://openlibrary.org"
    COVERS_URL = "https://covers.openlibrary.org"
    
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def search_by_isbn(self, isbn: str) -> Optional[BookMetadata]:
        """
        Search by ISBN.
        
        Args:
            isbn: ISBN-10 or ISBN-13
            
        Returns:
            BookMetadata or None
        """
        client = await self._get_client()
        
        # Clean ISBN
        isbn = isbn.replace("-", "").replace(" ", "")
        
        try:
            # Use ISBN API
            url = f"{self.BASE_URL}/isbn/{isbn}.json"
            response = await client.get(url)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            return await self._parse_work_data(data, isbn)
            
        except Exception as e:
            logger.warning(f"OpenLibrary ISBN lookup failed: {e}")
            return None
    
    async def search_by_title_author(
        self,
        title: str,
        author: Optional[str] = None,
        limit: int = 5,
    ) -> list[BookMetadata]:
        """
        Search by title and optionally author.
        
        Args:
            title: Book title
            author: Author name (optional)
            limit: Maximum results
            
        Returns:
            List of BookMetadata
        """
        client = await self._get_client()
        
        try:
            # Build query
            query = f"title:{title}"
            if author:
                query += f" author:{author}"
            
            url = f"{self.BASE_URL}/search.json"
            params = {
                "q": query,
                "limit": limit,
                "fields": "key,title,author_name,isbn,publisher,publish_year,cover_i,number_of_pages_median,subject",
            }
            
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            results = []
            
            for doc in data.get("docs", []):
                metadata = self._parse_search_result(doc)
                if metadata:
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            logger.warning(f"OpenLibrary search failed: {e}")
            return []
    
    async def _parse_work_data(self, data: dict, isbn: str) -> Optional[BookMetadata]:
        """Parse work/edition data from API."""
        try:
            # Get title
            title = data.get("title", "Unknown")
            
            # Get authors (need separate lookup)
            authors = []
            if "authors" in data:
                for author_ref in data["authors"]:
                    if isinstance(author_ref, dict) and "key" in author_ref:
                        author_name = await self._get_author_name(author_ref["key"])
                        if author_name:
                            authors.append(author_name)
            
            # Get ISBNs
            isbn_10 = None
            isbn_13 = None
            if len(isbn) == 10:
                isbn_10 = isbn
            elif len(isbn) == 13:
                isbn_13 = isbn
            
            # Additional ISBNs from data
            if "isbn_10" in data:
                isbn_10 = data["isbn_10"][0] if isinstance(data["isbn_10"], list) else data["isbn_10"]
            if "isbn_13" in data:
                isbn_13 = data["isbn_13"][0] if isinstance(data["isbn_13"], list) else data["isbn_13"]
            
            # Cover URL
            cover_url = None
            cover_id = data.get("covers", [None])[0] if data.get("covers") else None
            if cover_id:
                cover_url = f"{self.COVERS_URL}/b/id/{cover_id}-L.jpg"
            
            return BookMetadata(
                title=title,
                authors=authors,
                isbn_10=isbn_10,
                isbn_13=isbn_13,
                publisher=data.get("publishers", [None])[0] if data.get("publishers") else None,
                publish_date=data.get("publish_date"),
                page_count=data.get("number_of_pages"),
                cover_url=cover_url,
                open_library_id=data.get("key", "").replace("/books/", ""),
                source="openlibrary",
                fetched_at=datetime.now(),
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse OpenLibrary data: {e}")
            return None
    
    def _parse_search_result(self, doc: dict) -> Optional[BookMetadata]:
        """Parse search result document."""
        try:
            title = doc.get("title")
            if not title:
                return None
            
            authors = doc.get("author_name", [])
            
            # Get ISBNs
            isbns = doc.get("isbn", [])
            isbn_10 = None
            isbn_13 = None
            for isbn in isbns:
                if len(isbn) == 10 and isbn_10 is None:
                    isbn_10 = isbn
                elif len(isbn) == 13 and isbn_13 is None:
                    isbn_13 = isbn
            
            # Cover
            cover_url = None
            cover_id = doc.get("cover_i")
            if cover_id:
                cover_url = f"{self.COVERS_URL}/b/id/{cover_id}-L.jpg"
            
            # Year
            publish_year = None
            if doc.get("publish_year"):
                years = doc["publish_year"]
                publish_year = min(years) if years else None
            
            return BookMetadata(
                title=title,
                authors=authors,
                isbn_10=isbn_10,
                isbn_13=isbn_13,
                publisher=doc.get("publisher", [None])[0] if doc.get("publisher") else None,
                publish_year=publish_year,
                page_count=doc.get("number_of_pages_median"),
                subjects=doc.get("subject", [])[:10],  # Limit subjects
                cover_url=cover_url,
                open_library_id=doc.get("key", "").replace("/works/", ""),
                source="openlibrary",
                fetched_at=datetime.now(),
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse search result: {e}")
            return None
    
    async def _get_author_name(self, author_key: str) -> Optional[str]:
        """Fetch author name from author key."""
        client = await self._get_client()
        
        try:
            url = f"{self.BASE_URL}{author_key}.json"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("name")
                
        except Exception:
            pass
        
        return None
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class GoogleBooksClient:
    """
    Client for Google Books API.
    
    Provides high-quality cover images and descriptions.
    Rate limit: 1000 requests/day without API key.
    """
    
    BASE_URL = "https://www.googleapis.com/books/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def search_by_isbn(self, isbn: str) -> Optional[BookMetadata]:
        """Search by ISBN."""
        client = await self._get_client()
        
        isbn = isbn.replace("-", "").replace(" ", "")
        
        try:
            params = {"q": f"isbn:{isbn}"}
            if self.api_key:
                params["key"] = self.api_key
            
            response = await client.get(
                f"{self.BASE_URL}/volumes",
                params=params,
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if data.get("totalItems", 0) == 0:
                return None
            
            return self._parse_volume(data["items"][0])
            
        except Exception as e:
            logger.warning(f"Google Books ISBN lookup failed: {e}")
            return None
    
    async def search_by_title_author(
        self,
        title: str,
        author: Optional[str] = None,
        limit: int = 5,
    ) -> list[BookMetadata]:
        """Search by title and author."""
        client = await self._get_client()
        
        try:
            # Build query
            query = f"intitle:{title}"
            if author:
                query += f"+inauthor:{author}"
            
            params = {
                "q": query,
                "maxResults": limit,
            }
            if self.api_key:
                params["key"] = self.api_key
            
            response = await client.get(
                f"{self.BASE_URL}/volumes",
                params=params,
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            results = []
            
            for item in data.get("items", []):
                metadata = self._parse_volume(item)
                if metadata:
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            logger.warning(f"Google Books search failed: {e}")
            return []
    
    def _parse_volume(self, item: dict) -> Optional[BookMetadata]:
        """Parse volume data."""
        try:
            info = item.get("volumeInfo", {})
            
            title = info.get("title")
            if not title:
                return None
            
            # ISBNs
            isbn_10 = None
            isbn_13 = None
            for identifier in info.get("industryIdentifiers", []):
                if identifier.get("type") == "ISBN_10":
                    isbn_10 = identifier.get("identifier")
                elif identifier.get("type") == "ISBN_13":
                    isbn_13 = identifier.get("identifier")
            
            # Cover images
            images = info.get("imageLinks", {})
            cover_url = images.get("large") or images.get("medium") or images.get("small")
            thumbnail_url = images.get("thumbnail") or images.get("smallThumbnail")
            
            # Fix HTTP URLs
            if cover_url and cover_url.startswith("http:"):
                cover_url = cover_url.replace("http:", "https:")
            if thumbnail_url and thumbnail_url.startswith("http:"):
                thumbnail_url = thumbnail_url.replace("http:", "https:")
            
            # Extract year from publish date
            publish_year = None
            publish_date = info.get("publishedDate")
            if publish_date:
                try:
                    publish_year = int(publish_date[:4])
                except (ValueError, IndexError):
                    pass
            
            return BookMetadata(
                title=title,
                authors=info.get("authors", []),
                isbn_10=isbn_10,
                isbn_13=isbn_13,
                publisher=info.get("publisher"),
                publish_date=publish_date,
                publish_year=publish_year,
                description=info.get("description"),
                subjects=info.get("categories", []),
                page_count=info.get("pageCount"),
                language=info.get("language", "en"),
                cover_url=cover_url,
                thumbnail_url=thumbnail_url,
                google_books_id=item.get("id"),
                source="google_books",
                fetched_at=datetime.now(),
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse Google Books volume: {e}")
            return None
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class MetadataEnricher:
    """
    Unified metadata enrichment service with caching.
    """
    
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        cache_path: Optional[Union[str, Path]] = None,
        cache_ttl_days: int = 30,
    ):
        """
        Initialize metadata enricher.
        
        Args:
            google_api_key: Google Books API key (optional)
            cache_path: Path for cache storage
            cache_ttl_days: Cache entry TTL in days
        """
        self.openlibrary = OpenLibraryClient()
        self.google_books = GoogleBooksClient(api_key=google_api_key)
        
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_ttl = timedelta(days=cache_ttl_days)
        self._cache: dict[str, tuple[BookMetadata, datetime]] = {}
        
        # Load persistent cache
        if self.cache_path:
            self._load_cache()
        
        logger.info("MetadataEnricher initialized")
    
    async def enrich_by_isbn(self, isbn: str) -> Optional[BookMetadata]:
        """
        Enrich metadata by ISBN.
        
        Args:
            isbn: ISBN-10 or ISBN-13
            
        Returns:
            BookMetadata with combined information
        """
        # Check cache
        cache_key = f"isbn:{isbn.replace('-', '')}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Try Google Books first (better covers)
        google_result = await self.google_books.search_by_isbn(isbn)
        
        # Try Open Library
        openlibrary_result = await self.openlibrary.search_by_isbn(isbn)
        
        # Merge results
        metadata = self._merge_metadata(google_result, openlibrary_result)
        
        if metadata:
            self._set_cached(cache_key, metadata)
        
        return metadata
    
    async def enrich_by_title_author(
        self,
        title: str,
        author: Optional[str] = None,
    ) -> Optional[BookMetadata]:
        """
        Enrich metadata by title and author.
        
        Args:
            title: Book title
            author: Author name (optional)
            
        Returns:
            Best matching BookMetadata
        """
        # Check cache
        cache_key = f"title:{title}|author:{author or ''}"
        cache_key = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Search both sources
        google_results = await self.google_books.search_by_title_author(
            title, author, limit=3
        )
        openlibrary_results = await self.openlibrary.search_by_title_author(
            title, author, limit=3
        )
        
        # Get best match from each source
        google_best = google_results[0] if google_results else None
        openlibrary_best = openlibrary_results[0] if openlibrary_results else None
        
        # Merge
        metadata = self._merge_metadata(google_best, openlibrary_best)
        
        if metadata:
            self._set_cached(cache_key, metadata)
        
        return metadata
    
    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[BookMetadata]:
        """
        General search for books.
        
        Args:
            query: Search query (title, author, keywords)
            limit: Maximum results
            
        Returns:
            List of matching BookMetadata
        """
        # Search both sources
        tasks = [
            self.google_books.search_by_title_author(query, limit=limit // 2),
            self.openlibrary.search_by_title_author(query, limit=limit // 2),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_results = []
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
        
        # Deduplicate by ISBN
        seen_isbns = set()
        unique_results = []
        
        for metadata in all_results:
            isbn = metadata.primary_isbn
            if isbn and isbn in seen_isbns:
                continue
            if isbn:
                seen_isbns.add(isbn)
            unique_results.append(metadata)
        
        return unique_results[:limit]
    
    def _merge_metadata(
        self,
        primary: Optional[BookMetadata],
        secondary: Optional[BookMetadata],
    ) -> Optional[BookMetadata]:
        """
        Merge metadata from two sources.
        
        Primary source takes precedence for most fields.
        Secondary fills in missing data.
        """
        if not primary and not secondary:
            return None
        
        if not primary:
            return secondary
        
        if not secondary:
            return primary
        
        # Use primary as base, fill in missing from secondary
        return BookMetadata(
            title=primary.title or secondary.title,
            authors=primary.authors or secondary.authors,
            isbn_10=primary.isbn_10 or secondary.isbn_10,
            isbn_13=primary.isbn_13 or secondary.isbn_13,
            publisher=primary.publisher or secondary.publisher,
            publish_date=primary.publish_date or secondary.publish_date,
            publish_year=primary.publish_year or secondary.publish_year,
            description=primary.description or secondary.description,
            subjects=primary.subjects or secondary.subjects,
            page_count=primary.page_count or secondary.page_count,
            language=primary.language or secondary.language,
            cover_url=primary.cover_url or secondary.cover_url,
            thumbnail_url=primary.thumbnail_url or secondary.thumbnail_url,
            open_library_id=secondary.open_library_id if secondary.source == "openlibrary" else primary.open_library_id,
            google_books_id=primary.google_books_id if primary.source == "google_books" else secondary.google_books_id,
            source="merged",
            fetched_at=datetime.now(),
        )
    
    def _get_cached(self, key: str) -> Optional[BookMetadata]:
        """Get from cache if not expired."""
        if key in self._cache:
            metadata, cached_at = self._cache[key]
            if datetime.now() - cached_at < self.cache_ttl:
                return metadata
            else:
                del self._cache[key]
        return None
    
    def _set_cached(self, key: str, metadata: BookMetadata):
        """Set cache entry."""
        self._cache[key] = (metadata, datetime.now())
        
        # Persist if path configured
        if self.cache_path:
            self._save_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        if not self.cache_path or not self.cache_path.exists():
            return
        
        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)
            
            for key, entry in data.items():
                metadata = BookMetadata.from_dict(entry["metadata"])
                cached_at = datetime.fromisoformat(entry["cached_at"])
                self._cache[key] = (metadata, cached_at)
            
            logger.info(f"Loaded {len(self._cache)} cached metadata entries")
            
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        if not self.cache_path:
            return
        
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for key, (metadata, cached_at) in self._cache.items():
                data[key] = {
                    "metadata": metadata.to_dict(),
                    "cached_at": cached_at.isoformat(),
                }
            
            with open(self.cache_path, "w") as f:
                json.dump(data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save metadata cache: {e}")
    
    async def close(self):
        """Close all clients."""
        await self.openlibrary.close()
        await self.google_books.close()
