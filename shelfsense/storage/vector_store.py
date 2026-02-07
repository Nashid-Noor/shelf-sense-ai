"""
Vector Store for ShelfSense AI

FAISS-based vector storage and similarity search:
- Multiple index types (Flat, IVF, HNSW)
- Automatic index selection based on scale
- Batch operations for efficiency
- Index persistence and versioning

Design Decisions:
1. FAISS over pgvector: Better performance for pure similarity search
2. Separate indices for text/visual: Different embedding spaces
3. IVF for scale: O(n) → O(sqrt(n)) search time
4. Index sharding: Support for very large collections
"""

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - using numpy fallback")


@dataclass
class SearchResult:
    """Single search result."""
    
    id: str
    score: float
    distance: float
    rank: int
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResults:
    """Collection of search results."""
    
    results: list[SearchResult]
    query_time_ms: float
    total_searched: int
    index_type: str
    
    def __iter__(self):
        return iter(self.results)
    
    def __len__(self):
        return len(self.results)
    
    @property
    def ids(self) -> list[str]:
        return [r.id for r in self.results]
    
    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.results]


class VectorIndex:
    """
    Single FAISS index with metadata.
    
    Supports:
    - Multiple index types (Flat, IVF, HNSW)
    - ID mapping (FAISS uses int64, we use strings)
    - Persistence to disk
    - Incremental updates
    """
    
    # Index type configurations
    INDEX_CONFIGS = {
        "flat": {
            "description": "Exact search (brute force)",
            "max_vectors": 100_000,
            "search_time": "O(n)",
        },
        "ivf": {
            "description": "Inverted file index (approximate)",
            "max_vectors": 10_000_000,
            "search_time": "O(sqrt(n))",
            "nlist": 100,  # Number of clusters
            "nprobe": 10,  # Clusters to search
        },
        "hnsw": {
            "description": "Hierarchical NSW (approximate)",
            "max_vectors": 10_000_000,
            "search_time": "O(log(n))",
            "M": 32,  # Connections per layer
            "ef_construction": 200,
            "ef_search": 64,
        },
    }
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "cosine",
        name: str = "default",
    ):
        """
        Initialize vector index.
        
        Args:
            dimension: Embedding dimension
            index_type: "flat", "ivf", or "hnsw"
            metric: "cosine" or "l2"
            name: Index name for logging
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required for VectorIndex")
        
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.name = name
        
        # ID mapping (FAISS idx -> string ID)
        self._id_map: dict[int, str] = {}
        self._reverse_map: dict[str, int] = {}
        self._next_idx = 0
        
        # Metadata storage
        self._metadata: dict[str, dict] = {}
        
        # Create FAISS index
        self._index = self._create_index()
        
        # Track if trained (for IVF)
        self._is_trained = index_type == "flat"  # Flat doesn't need training
        
        logger.info(
            f"VectorIndex '{name}' created: "
            f"dim={dimension}, type={index_type}, metric={metric}"
        )
    
    def _create_index(self) -> "faiss.Index":
        """Create FAISS index based on type."""
        # Normalize for cosine similarity
        if self.metric == "cosine":
            # We'll L2-normalize vectors before adding
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2
        
        if self.index_type == "flat":
            if self.metric == "cosine":
                return faiss.IndexFlatIP(self.dimension)
            else:
                return faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "ivf":
            config = self.INDEX_CONFIGS["ivf"]
            nlist = config["nlist"]
            
            # Quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            
            # IVF index
            index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                nlist,
                metric,
            )
            index.nprobe = config["nprobe"]
            return index
        
        elif self.index_type == "hnsw":
            config = self.INDEX_CONFIGS["hnsw"]
            
            index = faiss.IndexHNSWFlat(
                self.dimension,
                config["M"],
                metric,
            )
            index.hnsw.efConstruction = config["ef_construction"]
            index.hnsw.efSearch = config["ef_search"]
            return index
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """
        Add vectors to index.
        
        Args:
            ids: String IDs for vectors
            embeddings: Vector array (n, dimension)
            metadata: Optional metadata per vector
            
        Returns:
            Number of vectors added
        """
        if len(ids) != len(embeddings):
            raise ValueError("IDs and embeddings must have same length")
        
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != index dimension {self.dimension}"
            )
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        
        # Check if IVF needs training
        if self.index_type == "ivf" and not self._is_trained:
            if len(embeddings) >= self.INDEX_CONFIGS["ivf"]["nlist"]:
                logger.info(f"Training IVF index with {len(embeddings)} vectors")
                self._index.train(embeddings)
                self._is_trained = True
            else:
                logger.warning(
                    f"Not enough vectors ({len(embeddings)}) to train IVF, "
                    f"need {self.INDEX_CONFIGS['ivf']['nlist']}"
                )
                return 0
        
        # Map IDs
        faiss_ids = []
        for str_id in ids:
            if str_id in self._reverse_map:
                # Update existing
                faiss_idx = self._reverse_map[str_id]
            else:
                # New ID
                faiss_idx = self._next_idx
                self._id_map[faiss_idx] = str_id
                self._reverse_map[str_id] = faiss_idx
                self._next_idx += 1
            faiss_ids.append(faiss_idx)
        
        # Add to FAISS
        self._index.add(embeddings)
        
        # Store metadata
        if metadata:
            for str_id, meta in zip(ids, metadata):
                self._metadata[str_id] = meta
        
        return len(ids)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> SearchResults:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector(s)
            k: Number of results
            
        Returns:
            SearchResults object
        """
        start = time.perf_counter()
        
        query = np.asarray(query, dtype=np.float32)
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize for cosine
        if self.metric == "cosine":
            norms = np.linalg.norm(query, axis=1, keepdims=True)
            query = query / np.maximum(norms, 1e-8)
        
        # Limit k to index size
        k = min(k, self._index.ntotal)
        
        if k == 0:
            return SearchResults(
                results=[],
                query_time_ms=0,
                total_searched=0,
                index_type=self.index_type,
            )
        
        # Search
        distances, indices = self._index.search(query, k)
        
        # Build results
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # Invalid index
                continue
            
            str_id = self._id_map.get(idx, str(idx))
            
            # Convert distance to similarity score
            if self.metric == "cosine":
                score = float(dist)  # IP is already similarity
            else:
                # L2 distance to similarity
                score = 1.0 / (1.0 + float(dist))
            
            results.append(SearchResult(
                id=str_id,
                score=score,
                distance=float(dist),
                rank=rank + 1,
                metadata=self._metadata.get(str_id, {}),
            ))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return SearchResults(
            results=results,
            query_time_ms=elapsed,
            total_searched=self._index.ntotal,
            index_type=self.index_type,
        )
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> list[SearchResults]:
        """
        Batch search for efficiency.
        
        Args:
            queries: Query vectors (n_queries, dimension)
            k: Results per query
            
        Returns:
            List of SearchResults
        """
        queries = np.asarray(queries, dtype=np.float32)
        
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        
        # Normalize
        if self.metric == "cosine":
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / np.maximum(norms, 1e-8)
        
        k = min(k, self._index.ntotal)
        
        if k == 0:
            return [SearchResults([], 0, 0, self.index_type) for _ in range(len(queries))]
        
        start = time.perf_counter()
        distances, indices = self._index.search(queries, k)
        elapsed = (time.perf_counter() - start) * 1000
        
        # Build results per query
        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for rank, (dist, idx) in enumerate(zip(distances[q_idx], indices[q_idx])):
                if idx < 0:
                    continue
                
                str_id = self._id_map.get(idx, str(idx))
                
                if self.metric == "cosine":
                    score = float(dist)
                else:
                    score = 1.0 / (1.0 + float(dist))
                
                results.append(SearchResult(
                    id=str_id,
                    score=score,
                    distance=float(dist),
                    rank=rank + 1,
                    metadata=self._metadata.get(str_id, {}),
                ))
            
            all_results.append(SearchResults(
                results=results,
                query_time_ms=elapsed / len(queries),
                total_searched=self._index.ntotal,
                index_type=self.index_type,
            ))
        
        return all_results
    
    def remove(self, ids: list[str]) -> int:
        """
        Remove vectors by ID.
        
        Note: FAISS removal is expensive for some index types.
        Consider rebuilding for bulk removals.
        
        Args:
            ids: IDs to remove
            
        Returns:
            Number removed
        """
        # Get FAISS indices
        faiss_ids = []
        for str_id in ids:
            if str_id in self._reverse_map:
                faiss_ids.append(self._reverse_map[str_id])
        
        if not faiss_ids:
            return 0
        
        # FAISS removal (creates new index without those IDs)
        # This is a simplified approach - real implementation would use IDMap
        logger.warning("Vector removal not fully implemented - consider rebuilding index")
        
        # Remove from mappings
        for str_id in ids:
            if str_id in self._reverse_map:
                faiss_idx = self._reverse_map.pop(str_id)
                self._id_map.pop(faiss_idx, None)
                self._metadata.pop(str_id, None)
        
        return len(ids)
    
    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return self._index.ntotal
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save index to disk.
        
        Args:
            path: Directory path for index files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self._index, str(path / "index.faiss"))
        
        # Save mappings and metadata
        state = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "name": self.name,
            "id_map": self._id_map,
            "reverse_map": self._reverse_map,
            "next_idx": self._next_idx,
            "metadata": self._metadata,
            "is_trained": self._is_trained,
        }
        
        with open(path / "state.pkl", "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Index saved to {path} ({self.size} vectors)")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorIndex":
        """
        Load index from disk.
        
        Args:
            path: Directory path containing index files
            
        Returns:
            Loaded VectorIndex
        """
        path = Path(path)
        
        # Load state
        with open(path / "state.pkl", "rb") as f:
            state = pickle.load(f)
        
        # Create instance
        instance = cls.__new__(cls)
        instance.dimension = state["dimension"]
        instance.index_type = state["index_type"]
        instance.metric = state["metric"]
        instance.name = state["name"]
        instance._id_map = state["id_map"]
        instance._reverse_map = state["reverse_map"]
        instance._next_idx = state["next_idx"]
        instance._metadata = state["metadata"]
        instance._is_trained = state["is_trained"]
        
        # Load FAISS index
        instance._index = faiss.read_index(str(path / "index.faiss"))
        
        logger.info(f"Index loaded from {path} ({instance.size} vectors)")
        
        return instance


class VectorStore:
    """
    Multi-index vector store for ShelfSense AI.
    
    Manages separate indices for:
    - Text embeddings (from OCR)
    - Visual embeddings (from CLIP)
    - Fused embeddings (optional)
    
    Usage:
        store = VectorStore(
            text_dimension=384,
            visual_dimension=512,
        )
        
        # Add book
        store.add_book(
            book_id="123",
            text_embedding=text_emb,
            visual_embedding=visual_emb,
            metadata={"title": "Harry Potter"},
        )
        
        # Search
        results = store.search_text(query_embedding, k=10)
    """
    
    def __init__(
        self,
        text_dimension: int = 384,
        visual_dimension: int = 512,
        index_type: str = "flat",
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize vector store.
        
        Args:
            text_dimension: Dimension of text embeddings
            visual_dimension: Dimension of visual embeddings
            index_type: Index type for all indices
            data_dir: Directory for persistence
        """
        self.text_dimension = text_dimension
        self.visual_dimension = visual_dimension
        self.index_type = index_type
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Create indices
        self.text_index = VectorIndex(
            dimension=text_dimension,
            index_type=index_type,
            metric="cosine",
            name="text",
        )
        
        self.visual_index = VectorIndex(
            dimension=visual_dimension,
            index_type=index_type,
            metric="cosine",
            name="visual",
        )
        
        logger.info(
            f"VectorStore initialized: "
            f"text_dim={text_dimension}, visual_dim={visual_dimension}"
        )
    
    def add_book(
        self,
        book_id: str,
        text_embedding: Optional[np.ndarray] = None,
        visual_embedding: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a book to the store.
        
        Args:
            book_id: Unique book identifier
            text_embedding: Text embedding (from OCR)
            visual_embedding: Visual embedding (from CLIP)
            metadata: Book metadata
        """
        if text_embedding is not None:
            self.text_index.add(
                ids=[book_id],
                embeddings=text_embedding.reshape(1, -1),
                metadata=[metadata] if metadata else None,
            )
        
        if visual_embedding is not None:
            self.visual_index.add(
                ids=[book_id],
                embeddings=visual_embedding.reshape(1, -1),
                metadata=[metadata] if metadata else None,
            )
    
    def add_books_batch(
        self,
        book_ids: list[str],
        text_embeddings: Optional[np.ndarray] = None,
        visual_embeddings: Optional[np.ndarray] = None,
        metadata_list: Optional[list[dict]] = None,
    ) -> int:
        """
        Batch add books.
        
        Args:
            book_ids: List of book IDs
            text_embeddings: Array of text embeddings
            visual_embeddings: Array of visual embeddings
            metadata_list: List of metadata dicts
            
        Returns:
            Number of books added
        """
        added = 0
        
        if text_embeddings is not None:
            added = self.text_index.add(
                ids=book_ids,
                embeddings=text_embeddings,
                metadata=metadata_list,
            )
        
        if visual_embeddings is not None:
            self.visual_index.add(
                ids=book_ids,
                embeddings=visual_embeddings,
                metadata=metadata_list,
            )
        
        return added
    
    def search_text(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> SearchResults:
        """Search using text embedding."""
        return self.text_index.search(query, k)
    
    def search_visual(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> SearchResults:
        """Search using visual embedding."""
        return self.visual_index.search(query, k)
    
    def search_multimodal(
        self,
        text_query: Optional[np.ndarray] = None,
        visual_query: Optional[np.ndarray] = None,
        text_weight: float = 0.6,
        visual_weight: float = 0.4,
        k: int = 10,
    ) -> SearchResults:
        """
        Search using both modalities with fusion.
        
        Args:
            text_query: Text embedding query
            visual_query: Visual embedding query
            text_weight: Weight for text scores
            visual_weight: Weight for visual scores
            k: Number of results
            
        Returns:
            Fused search results
        """
        start = time.perf_counter()
        
        # Retrieve more candidates for fusion
        retrieve_k = k * 3
        
        text_results = {}
        visual_results = {}
        
        if text_query is not None:
            text_search = self.text_index.search(text_query, retrieve_k)
            for r in text_search:
                text_results[r.id] = r.score
        
        if visual_query is not None:
            visual_search = self.visual_index.search(visual_query, retrieve_k)
            for r in visual_search:
                visual_results[r.id] = r.score
        
        # Fuse scores
        all_ids = set(text_results.keys()) | set(visual_results.keys())
        
        fused = []
        for book_id in all_ids:
            t_score = text_results.get(book_id, 0.0)
            v_score = visual_results.get(book_id, 0.0)
            
            fused_score = text_weight * t_score + visual_weight * v_score
            
            fused.append((book_id, fused_score, t_score, v_score))
        
        # Sort by fused score
        fused.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for rank, (book_id, score, t_score, v_score) in enumerate(fused[:k]):
            # Get metadata from either index
            metadata = self.text_index._metadata.get(book_id, {})
            if not metadata:
                metadata = self.visual_index._metadata.get(book_id, {})
            
            results.append(SearchResult(
                id=book_id,
                score=score,
                distance=1.0 - score,  # Convert similarity to distance
                rank=rank + 1,
                metadata={
                    **metadata,
                    "text_score": t_score,
                    "visual_score": v_score,
                },
            ))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return SearchResults(
            results=results,
            query_time_ms=elapsed,
            total_searched=self.text_index.size + self.visual_index.size,
            index_type=f"multimodal:{self.index_type}",
        )
    
    @property
    def stats(self) -> dict:
        """Get store statistics."""
        return {
            "text_index_size": self.text_index.size,
            "visual_index_size": self.visual_index.size,
            "text_dimension": self.text_dimension,
            "visual_dimension": self.visual_dimension,
            "index_type": self.index_type,
        }
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save all indices to disk."""
        path = Path(path) if path else self.data_dir
        if not path:
            raise ValueError("No save path specified")
        
        path.mkdir(parents=True, exist_ok=True)
        
        self.text_index.save(path / "text")
        self.visual_index.save(path / "visual")
        
        # Save config
        config = {
            "text_dimension": self.text_dimension,
            "visual_dimension": self.visual_dimension,
            "index_type": self.index_type,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"VectorStore saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorStore":
        """Load store from disk."""
        path = Path(path)
        
        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)
        
        # Create instance
        store = cls(
            text_dimension=config["text_dimension"],
            visual_dimension=config["visual_dimension"],
            index_type=config["index_type"],
            data_dir=path,
        )
        
        # Load indices
        store.text_index = VectorIndex.load(path / "text")
        store.visual_index = VectorIndex.load(path / "visual")
        
        logger.info(f"VectorStore loaded from {path}")
        
        return store
