"""
Storage Module for ShelfSense AI

Persistent storage for embeddings and book metadata:
- FAISS vector index for similarity search
- PostgreSQL for structured metadata
- Hybrid retrieval (dense + sparse)
- Index management and optimization
"""

from shelfsense.storage.vector_store import (
    VectorStore,
    VectorIndex,
    SearchResult,
)
from shelfsense.storage.book_repository import (
    BookRepository,
    StoredBook,
)
from shelfsense.storage.hybrid_retriever import (
    HybridRetriever,
    RetrievalResult,
)

__all__ = [
    # Vector Store
    "VectorStore",
    "VectorIndex",
    "SearchResult",
    # Book Repository
    "BookRepository",
    "StoredBook",
    # Hybrid Retriever
    "HybridRetriever",
    "RetrievalResult",
]
