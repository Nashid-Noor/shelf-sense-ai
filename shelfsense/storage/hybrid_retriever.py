"""
Hybrid Retriever

Combines dense vector search with sparse BM25.
"""

import asyncio
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from loguru import logger


@dataclass
class RetrievedDocument:
    """Single retrieved document."""
    
    id: str
    score: float
    
    # Score breakdown
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int = 0
    sparse_rank: int = 0
    
    # Source tracking
    retrieved_via: str = "hybrid"  # "dense", "sparse", "hybrid"
    
    # Document data
    title: Optional[str] = None
    author: Optional[str] = None
    content: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of retrieval operation."""
    
    query: str
    documents: list[RetrievedDocument]
    
    # Statistics
    total_candidates: int = 0
    dense_retrieval_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    fusion_ms: float = 0.0
    
    @property
    def total_ms(self) -> float:
        return self.dense_retrieval_ms + self.sparse_retrieval_ms + self.fusion_ms
    
    def __iter__(self):
        return iter(self.documents)
    
    def __len__(self):
        return len(self.documents)


class BM25Index:
    """
    BM25 sparse retrieval index.
    """
    
    # BM25 parameters
    K1 = 1.5  # Term frequency saturation
    B = 0.75  # Length normalization
    
    def __init__(self):
        """Initialize empty BM25 index."""
        # Document storage
        self._documents: dict[str, str] = {}  # id -> text
        self._doc_metadata: dict[str, dict] = {}
        
        # Index structures
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._doc_freqs: dict[str, int] = {}  # term -> doc count
        self._term_freqs: dict[str, dict[str, int]] = {}  # doc_id -> term -> count
        
        self._is_built = False
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add document to index.
        
        Args:
            doc_id: Document ID
            text: Document text
            metadata: Optional metadata
        """
        self._documents[doc_id] = text
        if metadata:
            self._doc_metadata[doc_id] = metadata
        self._is_built = False
    
    def add_documents(
        self,
        documents: list[tuple[str, str, Optional[dict]]],
    ) -> None:
        """
        Batch add documents.
        
        Args:
            documents: List of (id, text, metadata) tuples
        """
        for doc_id, text, metadata in documents:
            self.add_document(doc_id, text, metadata)
    
    def build(self) -> None:
        """Build the BM25 index."""
        if not self._documents:
            logger.warning("No documents to index")
            return
        
        # Tokenize all documents
        for doc_id, text in self._documents.items():
            tokens = self._tokenize(text)
            self._doc_lengths[doc_id] = len(tokens)
            
            # Term frequencies for this document
            term_freqs = Counter(tokens)
            self._term_freqs[doc_id] = dict(term_freqs)
            
            # Document frequencies
            for term in set(tokens):
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1
        
        # Average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths)
        
        self._is_built = True
        logger.info(f"BM25 index built: {len(self._documents)} documents")
    
    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (doc_id, score) tuples
        """
        if not self._is_built:
            self.build()
        
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        n_docs = len(self._documents)
        scores: dict[str, float] = {}
        
        for doc_id in self._documents:
            score = 0.0
            doc_length = self._doc_lengths[doc_id]
            
            for term in query_tokens:
                if term not in self._doc_freqs:
                    continue
                
                # Term frequency in document
                tf = self._term_freqs[doc_id].get(term, 0)
                if tf == 0:
                    continue
                
                # Document frequency
                df = self._doc_freqs[term]
                
                # IDF component
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                
                # BM25 score for term
                numerator = tf * (self.K1 + 1)
                denominator = tf + self.K1 * (
                    1 - self.B + self.B * doc_length / self._avg_doc_length
                )
                
                score += idf * (numerator / denominator)
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:k]
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for indexing/search.
        
        Simple tokenization: lowercase, alphanumeric only.
        For production, consider using nltk or spacy.
        """
        text = text.lower()
        # Keep alphanumeric and spaces
        text = re.sub(r"[^\w\s]", " ", text)
        # Split and filter
        tokens = [t.strip() for t in text.split() if len(t.strip()) > 1]
        return tokens
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get document text by ID."""
        return self._documents.get(doc_id)
    
    def get_metadata(self, doc_id: str) -> dict:
        """Get document metadata."""
        return self._doc_metadata.get(doc_id, {})
    
    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse search.
    """
    
    # RRF parameter (higher = less emphasis on top ranks)
    RRF_K = 60
    
    def __init__(
        self,
        vector_store=None,
        text_embedder=None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: VectorStore for dense retrieval
            text_embedder: TextEmbedder for query embedding
            dense_weight: Weight for dense scores (for weighted combination)
            sparse_weight: Weight for sparse scores
        """
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # BM25 index
        self.bm25_index = BM25Index()
        
        # Document cache
        self._documents: dict[str, dict] = {}
        
        logger.info(
            f"HybridRetriever initialized: "
            f"dense_weight={dense_weight}, sparse_weight={sparse_weight}"
        )
    
    def index_document(
        self,
        doc_id: str,
        text: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Index a document for both dense and sparse retrieval.
        
        Args:
            doc_id: Document ID
            text: Document text for BM25
            embedding: Precomputed embedding (optional)
            metadata: Document metadata
        """
        # Store document
        self._documents[doc_id] = {
            "text": text,
            "metadata": metadata or {},
        }
        
        # Add to BM25
        self.bm25_index.add_document(doc_id, text, metadata)
        
        # Add to vector store if embedding provided
        if embedding is not None and self.vector_store is not None:
            self.vector_store.text_index.add(
                ids=[doc_id],
                embeddings=embedding.reshape(1, -1),
                metadata=[metadata] if metadata else None,
            )
    
    def index_documents(
        self,
        documents: list[dict],
        batch_embed: bool = True,
    ) -> int:
        """
        Batch index documents.
        
        Args:
            documents: List of dicts with 'id', 'text', and optional 'embedding'
            batch_embed: Compute embeddings for documents without them
            
        Returns:
            Number indexed
        """
        # Add to BM25
        for doc in documents:
            doc_id = doc["id"]
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            self._documents[doc_id] = {
                "text": text,
                "metadata": metadata,
            }
            
            self.bm25_index.add_document(doc_id, text, metadata)
        
        # Build BM25 index
        self.bm25_index.build()
        
        # Add embeddings to vector store
        if self.vector_store is not None:
            ids = []
            embeddings = []
            metadata_list = []
            
            for doc in documents:
                if "embedding" in doc:
                    ids.append(doc["id"])
                    embeddings.append(doc["embedding"])
                    metadata_list.append(doc.get("metadata", {}))
                elif batch_embed and self.text_embedder is not None:
                    # Compute embedding
                    emb_result = self.text_embedder.embed(doc.get("text", ""))
                    emb = emb_result.embedding
                    ids.append(doc["id"])
                    embeddings.append(emb)
                    metadata_list.append(doc.get("metadata", {}))
            
            if ids:
                self.vector_store.text_index.add(
                    ids=ids,
                    embeddings=np.array(embeddings),
                    metadata=metadata_list,
                )
        
        return len(documents)
    
    async def search(
        self,
        query: str,
        k: int = 10,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
        use_rrf: bool = True,
    ) -> RetrievalResult:
        """
        Search using hybrid retrieval.
        
        Args:
            query: Search query
            k: Final number of results
            dense_k: Candidates from dense search (default: k * 3)
            sparse_k: Candidates from sparse search (default: k * 3)
            use_rrf: Use RRF fusion (vs weighted combination)
            
        Returns:
            RetrievalResult with fused documents
        """
        import time
        
        dense_k = dense_k or k * 3
        sparse_k = sparse_k or k * 3
        
        # Run both searches
        dense_results = []
        sparse_results = []
        dense_time = 0.0
        sparse_time = 0.0
        
        # Dense search
        if self.vector_store is not None and self.text_embedder is not None:
            start = time.perf_counter()
            
            # Embed query
            query_embedding = self.text_embedder.embed(query).embedding
            
            # Search
            search_results = self.vector_store.search_text(query_embedding, k=dense_k)
            
            dense_results = [
                (r.id, r.score, r.metadata)
                for r in search_results
            ]
            
            dense_time = (time.perf_counter() - start) * 1000
        
        # Sparse search
        start = time.perf_counter()
        sparse_raw = self.bm25_index.search(query, k=sparse_k)
        sparse_results = [
            (doc_id, score, self.bm25_index.get_metadata(doc_id))
            for doc_id, score in sparse_raw
        ]
        sparse_time = (time.perf_counter() - start) * 1000
        
        # Fusion
        start = time.perf_counter()
        
        if use_rrf:
            fused = self._rrf_fusion(dense_results, sparse_results, k)
        else:
            fused = self._weighted_fusion(dense_results, sparse_results, k)
        
        fusion_time = (time.perf_counter() - start) * 1000
        
        # Build result documents
        documents = []
        for rank, (doc_id, score, d_score, s_score, d_rank, s_rank, metadata) in enumerate(fused):
            doc_data = self._documents.get(doc_id, {})
            
            documents.append(RetrievedDocument(
                id=doc_id,
                score=score,
                dense_score=d_score,
                sparse_score=s_score,
                dense_rank=d_rank,
                sparse_rank=s_rank,
                retrieved_via="hybrid" if d_score > 0 and s_score > 0 else (
                    "dense" if d_score > 0 else "sparse"
                ),
                title=metadata.get("title"),
                author=metadata.get("author"),
                content=doc_data.get("text"),
                metadata=metadata,
            ))
        
        return RetrievalResult(
            query=query,
            documents=documents,
            total_candidates=len(dense_results) + len(sparse_results),
            dense_retrieval_ms=dense_time,
            sparse_retrieval_ms=sparse_time,
            fusion_ms=fusion_time,
        )
    
    def _rrf_fusion(
        self,
        dense_results: list[tuple[str, float, dict]],
        sparse_results: list[tuple[str, float, dict]],
        k: int,
    ) -> list[tuple]:
        """
        Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (K + rank))
        
        Returns list of (doc_id, fused_score, dense_score, sparse_score,
                        dense_rank, sparse_rank, metadata)
        """
        # Build rank maps
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _, _) in enumerate(dense_results)}
        sparse_ranks = {doc_id: rank + 1 for rank, (doc_id, _, _) in enumerate(sparse_results)}
        
        # Score maps
        dense_scores = {doc_id: score for doc_id, score, _ in dense_results}
        sparse_scores = {doc_id: score for doc_id, score, _ in sparse_results}
        
        # Metadata maps
        metadata_map = {}
        for doc_id, _, metadata in dense_results:
            metadata_map[doc_id] = metadata
        for doc_id, _, metadata in sparse_results:
            if doc_id not in metadata_map:
                metadata_map[doc_id] = metadata
        
        # Compute RRF scores
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        fused = []
        for doc_id in all_ids:
            d_rank = dense_ranks.get(doc_id, len(dense_results) + 100)
            s_rank = sparse_ranks.get(doc_id, len(sparse_results) + 100)
            
            # RRF score
            rrf_score = 0.0
            if doc_id in dense_ranks:
                rrf_score += 1 / (self.RRF_K + d_rank)
            if doc_id in sparse_ranks:
                rrf_score += 1 / (self.RRF_K + s_rank)
            
            fused.append((
                doc_id,
                rrf_score,
                dense_scores.get(doc_id, 0.0),
                sparse_scores.get(doc_id, 0.0),
                d_rank if doc_id in dense_ranks else 0,
                s_rank if doc_id in sparse_ranks else 0,
                metadata_map.get(doc_id, {}),
            ))
        
        # Sort by RRF score
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused[:k]
    
    def _weighted_fusion(
        self,
        dense_results: list[tuple[str, float, dict]],
        sparse_results: list[tuple[str, float, dict]],
        k: int,
    ) -> list[tuple]:
        """
        Weighted score combination.
        
        Normalizes scores to [0, 1] and combines with weights.
        """
        # Normalize dense scores
        dense_scores = {doc_id: score for doc_id, score, _ in dense_results}
        if dense_scores:
            max_d = max(dense_scores.values())
            min_d = min(dense_scores.values())
            range_d = max_d - min_d if max_d > min_d else 1.0
            dense_scores = {k: (v - min_d) / range_d for k, v in dense_scores.items()}
        
        # Normalize sparse scores  
        sparse_scores = {doc_id: score for doc_id, score, _ in sparse_results}
        if sparse_scores:
            max_s = max(sparse_scores.values())
            min_s = min(sparse_scores.values())
            range_s = max_s - min_s if max_s > min_s else 1.0
            sparse_scores = {k: (v - min_s) / range_s for k, v in sparse_scores.items()}
        
        # Metadata
        metadata_map = {}
        for doc_id, _, metadata in dense_results + sparse_results:
            if doc_id not in metadata_map:
                metadata_map[doc_id] = metadata
        
        # Combine
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        fused = []
        for doc_id in all_ids:
            d_score = dense_scores.get(doc_id, 0.0)
            s_score = sparse_scores.get(doc_id, 0.0)
            
            combined = self.dense_weight * d_score + self.sparse_weight * s_score
            
            fused.append((
                doc_id,
                combined,
                d_score,
                s_score,
                0,  # Ranks not meaningful for weighted
                0,
                metadata_map.get(doc_id, {}),
            ))
        
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused[:k]
    
    @property
    def stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "bm25_documents": self.bm25_index.size,
            "vector_store_connected": self.vector_store is not None,
            "text_embedder_connected": self.text_embedder is not None,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
        }
