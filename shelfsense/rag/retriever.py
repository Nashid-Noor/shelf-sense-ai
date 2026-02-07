"""
RAG Retriever for ShelfSense AI

Intelligent retrieval for conversational book queries:
- Query understanding and expansion
- Multi-modal hybrid retrieval
- Re-ranking for relevance
- Conversation-aware context

Design Decisions:
1. Query expansion: Use synonyms and related terms to improve recall
2. Multi-stage retrieval: Broad recall → precise re-ranking
3. Conversation context: Consider recent turns for better relevance
4. Diversity: Avoid returning too-similar books
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import numpy as np
from loguru import logger


@dataclass
class RetrievedBook:
    """A book retrieved for RAG context."""
    
    book_id: str
    title: str
    author: str
    score: float
    retrieval_method: str  # "dense", "sparse", "hybrid"
    
    # Optional metadata
    genres: list[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    description: Optional[str] = None
    subjects: list[str] = field(default_factory=list)
    
    # For re-ranking
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for prompt formatting."""
        return {
            "id": self.book_id,
            "title": self.title,
            "author": self.author,
            "genres": self.genres,
            "publication_year": self.publication_year,
            "description": self.description,
        }


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    
    query: str
    expanded_query: Optional[str]
    books: list[RetrievedBook]
    total_candidates: int
    retrieval_time_ms: float
    
    # Metadata
    methods_used: list[str] = field(default_factory=list)
    diversity_score: float = 0.0


class QueryExpander:
    """
    Expand queries with synonyms and related terms.
    
    Improves recall for semantic search by adding:
    - Genre synonyms
    - Author name variants
    - Topic expansions
    """
    
    # Genre synonym mappings
    GENRE_SYNONYMS = {
        "sci-fi": ["science fiction", "sf", "speculative fiction"],
        "science fiction": ["sci-fi", "sf", "speculative fiction"],
        "fantasy": ["magical", "mythical", "epic fantasy"],
        "mystery": ["detective", "crime", "thriller", "whodunit"],
        "thriller": ["suspense", "mystery", "crime"],
        "romance": ["love story", "romantic", "love"],
        "horror": ["scary", "supernatural", "gothic"],
        "non-fiction": ["nonfiction", "non fiction", "factual"],
        "biography": ["memoir", "life story", "autobiography"],
        "history": ["historical", "past", "ancient"],
        "self-help": ["self help", "personal development", "self improvement"],
    }
    
    # Common query pattern expansions
    PATTERN_EXPANSIONS = {
        "books like": "similar to",
        "something like": "similar to",
        "similar to": "related to",
        "written by": "author",
        "about": "on the topic of",
    }
    
    def __init__(
        self,
        use_synonyms: bool = True,
        use_patterns: bool = True,
        max_expansion_terms: int = 3,
    ):
        """
        Initialize query expander.
        
        Args:
            use_synonyms: Enable genre synonym expansion
            use_patterns: Enable query pattern expansion
            max_expansion_terms: Maximum terms to add
        """
        self.use_synonyms = use_synonyms
        self.use_patterns = use_patterns
        self.max_expansion_terms = max_expansion_terms
    
    def expand(self, query: str) -> str:
        """
        Expand a query with related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query string
        """
        query_lower = query.lower()
        expansions = []
        
        # Genre synonym expansion
        if self.use_synonyms:
            for genre, synonyms in self.GENRE_SYNONYMS.items():
                if genre in query_lower:
                    # Add first few synonyms
                    expansions.extend(synonyms[:self.max_expansion_terms])
                    break
        
        # Pattern expansion (for specific query types)
        if self.use_patterns:
            for pattern, expansion in self.PATTERN_EXPANSIONS.items():
                if pattern in query_lower:
                    expansions.append(expansion)
        
        if not expansions:
            return query
        
        # Combine original query with expansions
        expansion_str = " ".join(expansions[:self.max_expansion_terms])
        expanded = f"{query} {expansion_str}"
        
        logger.debug(f"Query expanded: '{query}' → '{expanded}'")
        return expanded
    
    def extract_entities(self, query: str) -> dict:
        """
        Extract entities from query for structured search.
        
        Args:
            query: User query
            
        Returns:
            Dict with extracted entities
        """
        entities = {
            "genres": [],
            "authors": [],
            "titles": [],
            "years": [],
        }
        
        query_lower = query.lower()
        
        # Extract genres
        for genre in self.GENRE_SYNONYMS.keys():
            if genre in query_lower:
                entities["genres"].append(genre)
        
        # Extract years (simple pattern matching)
        import re
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        entities["years"] = [int(y) for y in years]
        
        return entities


class ReRanker:
    """
    Re-rank retrieved candidates for improved relevance.
    
    Uses multiple signals:
    - Query-document similarity
    - Diversity penalty (avoid similar books)
    - Recency boost (conversation context)
    """
    
    def __init__(
        self,
        diversity_weight: float = 0.2,
        recency_weight: float = 0.1,
        min_diversity_distance: float = 0.3,
    ):
        """
        Initialize re-ranker.
        
        Args:
            diversity_weight: Weight for diversity penalty
            recency_weight: Weight for recency boost
            min_diversity_distance: Minimum distance between results
        """
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
        self.min_diversity_distance = min_diversity_distance
    
    def rerank(
        self,
        candidates: list[RetrievedBook],
        query: str,
        conversation_context: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[RetrievedBook]:
        """
        Re-rank candidates using multiple signals.
        
        Args:
            candidates: Initial candidates
            query: Original query
            conversation_context: Recent book IDs mentioned
            top_k: Number of results to return
            
        Returns:
            Re-ranked list of books
        """
        if not candidates:
            return []
        
        # Start with retrieval scores
        for book in candidates:
            book.rerank_score = book.score
        
        # Apply recency boost for conversation context
        if conversation_context:
            context_set = set(conversation_context)
            for book in candidates:
                if book.book_id in context_set:
                    # Slight boost for recently discussed books
                    book.rerank_score += self.recency_weight
        
        # Sort by rerank score
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Apply diversity filtering using MMR-style approach
        selected = []
        remaining = candidates.copy()
        
        while remaining and len(selected) < top_k:
            if not selected:
                # First item: highest score
                best = remaining[0]
            else:
                # Balance relevance and diversity
                best = None
                best_mmr = float('-inf')
                
                for candidate in remaining:
                    # Calculate diversity: min distance to selected
                    min_sim = self._calculate_similarity(candidate, selected)
                    
                    # MMR: λ * relevance - (1-λ) * max_similarity
                    mmr = (
                        (1 - self.diversity_weight) * candidate.rerank_score -
                        self.diversity_weight * min_sim
                    )
                    
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best = candidate
                
                if best is None:
                    break
            
            selected.append(best)
            remaining.remove(best)
        
        return selected
    
    def _calculate_similarity(
        self,
        candidate: RetrievedBook,
        selected: list[RetrievedBook],
    ) -> float:
        """
        Calculate max similarity to already selected books.
        
        Uses simple heuristics based on metadata.
        """
        max_sim = 0.0
        
        for book in selected:
            sim = 0.0
            
            # Same author: high similarity
            if candidate.author.lower() == book.author.lower():
                sim += 0.5
            
            # Shared genres
            if candidate.genres and book.genres:
                shared = set(candidate.genres) & set(book.genres)
                if shared:
                    sim += 0.3 * len(shared) / max(len(candidate.genres), len(book.genres))
            
            # Similar year
            if candidate.publication_year and book.publication_year:
                year_diff = abs(candidate.publication_year - book.publication_year)
                if year_diff <= 5:
                    sim += 0.2
            
            max_sim = max(max_sim, sim)
        
        return max_sim


class RAGRetriever:
    """
    Main retriever for RAG conversations.
    
    Orchestrates:
    1. Query expansion
    2. Hybrid retrieval (dense + sparse)
    3. Re-ranking
    4. Context building
    """
    
    def __init__(
        self,
        hybrid_retriever,  # HybridRetriever from storage module
        book_repository,   # BookRepository for metadata
        text_embedder,     # TextEmbedder for query embedding
        query_expander: Optional[QueryExpander] = None,
        reranker: Optional[ReRanker] = None,
        default_top_k: int = 10,
        candidate_multiplier: int = 3,
    ):
        """
        Initialize RAG retriever.
        
        Args:
            hybrid_retriever: Hybrid search backend
            book_repository: Book metadata store
            text_embedder: For query embedding
            query_expander: Query expansion (optional)
            reranker: Re-ranking (optional)
            default_top_k: Default results to return
            candidate_multiplier: Candidates to fetch for re-ranking
        """
        self.hybrid_retriever = hybrid_retriever
        self.book_repository = book_repository
        self.text_embedder = text_embedder
        self.query_expander = query_expander or QueryExpander()
        self.reranker = reranker or ReRanker()
        self.default_top_k = default_top_k
        self.candidate_multiplier = candidate_multiplier
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        conversation_context: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        expand_query: bool = True,
        rerank: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant books for a query.
        
        Args:
            query: User query
            top_k: Number of results
            conversation_context: Recent book IDs from conversation
            filters: Optional filters (genre, year, etc.)
            expand_query: Whether to expand query
            rerank: Whether to re-rank results
            
        Returns:
            RetrievalResult with books and metadata
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.default_top_k
        
        # Step 1: Query expansion
        expanded_query = None
        search_query = query
        
        if expand_query:
            expanded_query = self.query_expander.expand(query)
            if expanded_query != query:
                search_query = expanded_query
        
        # Step 2: Generate query embedding
        query_embedding = await asyncio.to_thread(
            self.text_embedder.embed,
            search_query,
        )
        
        # Step 3: Hybrid retrieval (fetch more for re-ranking)
        num_candidates = top_k * self.candidate_multiplier if rerank else top_k
        
        search_results = await self.hybrid_retriever.search(
            query=search_query,
            k=num_candidates,
        )
        
        # Step 4: Enrich with metadata
        candidates = await self._enrich_results(search_results)
        
        # Step 5: Apply filters
        if filters:
            candidates = self._apply_filters(candidates, filters)
        
        # Step 6: Re-rank
        if rerank and len(candidates) > top_k:
            candidates = self.reranker.rerank(
                candidates,
                query,
                conversation_context,
                top_k,
            )
        else:
            candidates = candidates[:top_k]
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query=query,
            expanded_query=expanded_query if expanded_query != query else None,
            books=candidates,
            total_candidates=len(search_results),
            retrieval_time_ms=elapsed_ms,
            methods_used=["dense", "sparse", "hybrid"],
        )
    
    async def retrieve_by_similarity(
        self,
        book_id: str,
        top_k: int = 5,
    ) -> list[RetrievedBook]:
        """
        Find books similar to a given book.
        
        Args:
            book_id: Reference book ID
            top_k: Number of similar books
            
        Returns:
            List of similar books (excluding the reference)
        """
        # Get reference book
        reference = await asyncio.to_thread(
            self.book_repository.get_by_id,
            book_id,
        )
        
        if not reference:
            return []
        
        # Build query from book metadata
        query = f"{reference.title} {reference.author}"
        if reference.genres:
            query += f" {' '.join(reference.genres[:3])}"
        
        # Retrieve similar
        result = await self.retrieve(
            query=query,
            top_k=top_k + 1,  # +1 to exclude self
            rerank=True,
        )
        
        # Filter out the reference book
        similar = [
            book for book in result.books
            if book.book_id != book_id
        ]
        
        return similar[:top_k]
    
    async def retrieve_by_genre(
        self,
        genres: list[str],
        top_k: int = 10,
    ) -> list[RetrievedBook]:
        """
        Retrieve books by genre.
        
        Args:
            genres: List of genres
            top_k: Number of results
            
        Returns:
            Books matching genres
        """
        # Query with genres
        query = " ".join(genres)
        
        result = await self.retrieve(
            query=query,
            top_k=top_k,
            filters={"genres": genres},
            expand_query=True,
        )
        
        return result.books
    
    async def _enrich_results(
        self,
        search_results: list,
    ) -> list[RetrievedBook]:
        """
        Enrich search results with book metadata.
        
        Args:
            search_results: List of RetrievedDocument objects
            
        Returns:
            Enriched RetrievedBook list
        """
        if not search_results:
            return []
        
        # Fetch all books in batch
        book_ids = [doc.id for doc in search_results]
        books_data = await asyncio.to_thread(
            self._batch_get_books,
            book_ids,
        )
        
        # Build retrieved book objects
        enriched = []
        for doc in search_results:
            book_id = doc.id
            score = doc.score
            book_data = books_data.get(book_id)
            
            if book_data:
                enriched.append(RetrievedBook(
                    book_id=book_id,
                    title=book_data.get("title", "Unknown"),
                    author=book_data.get("author", "Unknown"),
                    score=score,
                    retrieval_method="hybrid",
                    genres=book_data.get("genres", []),
                    publication_year=book_data.get("publication_year"),
                    description=book_data.get("description"),
                    subjects=book_data.get("subjects", []),
                ))
        
        return enriched
    
    def _batch_get_books(self, book_ids: list[str]) -> dict:
        """
        Batch fetch books from repository.
        
        Returns dict mapping book_id to book data.
        """
        result = {}
        
        for book_id in book_ids:
            book = self.book_repository.get(book_id)
            if book:
                result[book_id] = {
                    "title": book.title,
                    "author": book.author,
                    "genres": book.genres or [],
                    "publication_year": book.publication_year,
                    "description": book.description,
                    "subjects": book.subjects or [],
                }
        
        return result
    
    def _apply_filters(
        self,
        candidates: list[RetrievedBook],
        filters: dict,
    ) -> list[RetrievedBook]:
        """
        Apply filters to candidates.
        
        Supported filters:
        - genres: List of required genres
        - min_year: Minimum publication year
        - max_year: Maximum publication year
        - authors: List of authors
        """
        filtered = candidates
        
        # Genre filter
        if "genres" in filters and filters["genres"]:
            required_genres = set(g.lower() for g in filters["genres"])
            filtered = [
                book for book in filtered
                if any(g.lower() in required_genres for g in book.genres)
            ]
        
        # Year range filter
        if "min_year" in filters:
            filtered = [
                book for book in filtered
                if book.publication_year and book.publication_year >= filters["min_year"]
            ]
        
        if "max_year" in filters:
            filtered = [
                book for book in filtered
                if book.publication_year and book.publication_year <= filters["max_year"]
            ]
        
        # Author filter
        if "authors" in filters and filters["authors"]:
            required_authors = set(a.lower() for a in filters["authors"])
            filtered = [
                book for book in filtered
                if book.author.lower() in required_authors
            ]
        
        return filtered


class ConversationAwareRetriever(RAGRetriever):
    """
    Extended retriever that tracks conversation context.
    
    Maintains a sliding window of mentioned books
    and uses them for improved retrieval.
    """
    
    def __init__(
        self,
        *args,
        context_window_size: int = 10,
        **kwargs,
    ):
        """
        Initialize with conversation tracking.
        
        Args:
            context_window_size: Books to track in conversation
            *args, **kwargs: Passed to RAGRetriever
        """
        super().__init__(*args, **kwargs)
        self.context_window_size = context_window_size
        
        # Track mentioned books per conversation
        self._conversation_contexts: dict[str, list[str]] = defaultdict(list)
    
    def update_context(
        self,
        conversation_id: str,
        book_ids: list[str],
    ):
        """
        Update conversation context with mentioned books.
        
        Args:
            conversation_id: Conversation identifier
            book_ids: Books mentioned in recent turn
        """
        context = self._conversation_contexts[conversation_id]
        
        for book_id in book_ids:
            if book_id in context:
                context.remove(book_id)
            context.append(book_id)
        
        # Trim to window size
        if len(context) > self.context_window_size:
            self._conversation_contexts[conversation_id] = context[-self.context_window_size:]
    
    def get_context(self, conversation_id: str) -> list[str]:
        """Get current conversation context."""
        return self._conversation_contexts.get(conversation_id, [])
    
    def clear_context(self, conversation_id: str):
        """Clear conversation context."""
        if conversation_id in self._conversation_contexts:
            del self._conversation_contexts[conversation_id]
    
    async def retrieve_with_context(
        self,
        query: str,
        conversation_id: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve with automatic conversation context.
        
        Args:
            query: User query
            conversation_id: Conversation identifier
            top_k: Number of results
            **kwargs: Additional retrieval options
            
        Returns:
            RetrievalResult with conversation-aware ranking
        """
        context = self.get_context(conversation_id)
        
        return await self.retrieve(
            query=query,
            top_k=top_k,
            conversation_context=context,
            **kwargs,
        )
