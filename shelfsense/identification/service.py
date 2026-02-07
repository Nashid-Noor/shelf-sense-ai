"""
Identification Service

Runs book identification from OCR text.
"""

import re
from typing import Optional, List, Dict, Any
from loguru import logger
from dataclasses import asdict

from shelfsense.identification.google_books import GoogleBooksClient, BookMetadata

class IdentificationService:
    """Service for identifying books from OCR text."""
    
    def __init__(self, google_books_client: Optional[GoogleBooksClient] = None):
        """
        Initialize service.
        
        Args:
            google_books_client: Client for Google Books API
        """
        self.client = google_books_client or GoogleBooksClient()
        
    async def identify(self, ocr_text: str, similarity_threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """
        Identify a book from OCR text.
        
        Args:
            ocr_text: Raw text from OCR
            similarity_threshold: Minimum similarity to accept a match (not used yet, placeholder)
            
        Returns:
            Dictionary with book details or None if not identified
        """
        if not ocr_text or len(ocr_text.strip()) < 3:
            return None
            
        # 1. Clean query
        query = self._clean_query(ocr_text)
        if not query:
            return None
            
        logger.info(f"Searching for book with query: '{query}'")
        
        # 2. Search
        results = await self.client.search(query, max_results=3)
        
        if not results:
            logger.info("No results found")
            return None
            
        # 3. Select best match using fuzzy similarity
        # We want to match the OCR query against the Book Title + Author
        
        best_match = None
        best_score = -1.0
        
        from difflib import SequenceMatcher
        
        for book in results:
            # Check similarity with title
            title_sim = SequenceMatcher(None, query.lower(), book.title.lower()).ratio()
            
            # Check similarity with title + author (if available)
            full_text = f"{book.title}"
            if book.authors:
                 full_text += f" {' '.join(book.authors)}"
            
            full_sim = SequenceMatcher(None, query.lower(), full_text.lower()).ratio()
            
            # Use the better of the two scores
            score = max(title_sim, full_sim)
            
            if score > best_score:
                best_score = score
                best_match = book

        # 4. Verify threshold
        if best_match and best_score >= similarity_threshold:
            logger.info(f"Identified '{best_match.title}' with similarity {best_score:.2f}")
            return {
                "identified": True,
                "title": best_match.title,
                "author": best_match.authors[0] if best_match.authors else "Unknown",
                "book_id": best_match.google_books_id,
                "isbn_13": best_match.isbn_13,
                "isbn_10": best_match.isbn_10,
                "cover_url": best_match.thumbnail_url,
                "identification_confidence": best_score,
                
                # Rich Metadata
                "genres": best_match.categories or [],
                "publication_year": int(best_match.published_date[:4]) if best_match.published_date and best_match.published_date[:4].isdigit() else None,
                "publisher": best_match.publisher,
                "description": best_match.description,
            }
        else:
            if best_match:
                logger.warning(f"Rejected best match '{best_match.title}' - score {best_score:.2f} < {similarity_threshold}")
            return None
        
    def _clean_query(self, text: str) -> str:
        """Clean OCR text for better search results."""
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove single characters (often noise)
        text = ' '.join(word for word in text.split() if len(word) > 1)
        
        # Collapse whitespace
        text = ' '.join(text.split())
        
        return text
