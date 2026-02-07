"""
Text Embedder

Sentence-BERT based text embedding for semantic book matching:
- Multiple model options
- Normalized embeddings
- Batch processing
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger


@dataclass
class TextEmbeddingResult:
    """Result of text embedding."""
    
    text: str
    embedding: np.ndarray
    model_name: str
    dimension: int
    
    # Metadata
    processing_time_ms: float = 0.0
    was_truncated: bool = False
    
    def to_list(self) -> list[float]:
        """Convert embedding to Python list."""
        return self.embedding.tolist()
    
    @property
    def norm(self) -> float:
        """L2 norm of embedding."""
        return float(np.linalg.norm(self.embedding))


class TextEmbedder:
    """
    Production text embedder using Sentence-BERT models.
    
    Converts OCR text and book metadata into dense vector embeddings
    for semantic similarity search. Optimized for:
    - Short to medium text (titles, authors, descriptions)
    - Book-domain vocabulary
    - Fast inference with batching
    
    Model Options:
    - 'all-MiniLM-L6-v2': Fast, 384d, good for most cases
    - 'all-mpnet-base-v2': Higher quality, 768d, slower
    - 'multi-qa-MiniLM-L6-cos-v1': Optimized for Q&A/retrieval
    
    Usage:
        embedder = TextEmbedder()
        result = embedder.embed("The Great Gatsby by F. Scott Fitzgerald")
        print(result.embedding.shape)  # (384,)
    """
    
    # Recommended models for book matching
    MODELS = {
        'fast': 'sentence-transformers/all-MiniLM-L6-v2',
        'balanced': 'sentence-transformers/all-mpnet-base-v2',
        'retrieval': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    }
    
    def __init__(
        self,
        model_name: str = 'fast',
        device: Optional[str] = None,
        max_seq_length: int = 256,
        normalize_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize text embedder.
        
        Args:
            model_name: One of 'fast', 'balanced', 'retrieval' or HuggingFace model ID
            device: 'cuda', 'cpu', or None for auto-detect
            max_seq_length: Maximum token length (truncate longer texts)
            normalize_embeddings: L2 normalize embeddings (recommended)
            cache_dir: Directory for model cache
        """
        # Resolve model name
        if model_name in self.MODELS:
            self.model_id = self.MODELS[model_name]
        else:
            self.model_id = model_name
        
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        logger.info(f"Loading text embedding model: {self.model_id} on {device}")
        
        # Load model
        self.model = SentenceTransformer(
            self.model_id,
            device=device,
            cache_folder=str(cache_dir) if cache_dir else None,
        )
        self.model.max_seq_length = max_seq_length
        
        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(
            f"TextEmbedder ready: dim={self.dimension}, "
            f"max_seq={max_seq_length}, device={device}"
        )
    
    def embed(self, text: str) -> TextEmbeddingResult:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed (OCR output, title, etc.)
            
        Returns:
            TextEmbeddingResult with embedding vector
        """
        import time
        start = time.perf_counter()
        
        # Handle empty text
        if not text or not text.strip():
            return TextEmbeddingResult(
                text=text,
                embedding=np.zeros(self.dimension, dtype=np.float32),
                model_name=self.model_name,
                dimension=self.dimension,
                processing_time_ms=0.0,
            )
        
        # Check if text will be truncated
        # Rough estimate: 1 token ≈ 4 characters
        was_truncated = len(text) > self.max_seq_length * 4
        
        # Generate embedding
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return TextEmbeddingResult(
            text=text,
            embedding=embedding.astype(np.float32),
            model_name=self.model_name,
            dimension=self.dimension,
            processing_time_ms=elapsed,
            was_truncated=was_truncated,
        )
    
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[TextEmbeddingResult]:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for GPU processing
            show_progress: Show progress bar
            
        Returns:
            List of TextEmbeddingResult objects
        """
        import time
        start = time.perf_counter()
        
        if not texts:
            return []
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        # Generate embeddings for valid texts
        if valid_texts:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
        else:
            embeddings = np.array([])
        
        elapsed = (time.perf_counter() - start) * 1000
        per_text = elapsed / len(texts) if texts else 0
        
        # Build results, inserting zeros for empty texts
        results = []
        valid_idx = 0
        
        for i, text in enumerate(texts):
            if i in valid_indices:
                results.append(TextEmbeddingResult(
                    text=text,
                    embedding=embeddings[valid_idx].astype(np.float32),
                    model_name=self.model_name,
                    dimension=self.dimension,
                    processing_time_ms=per_text,
                    was_truncated=len(text) > self.max_seq_length * 4,
                ))
                valid_idx += 1
            else:
                results.append(TextEmbeddingResult(
                    text=text,
                    embedding=np.zeros(self.dimension, dtype=np.float32),
                    model_name=self.model_name,
                    dimension=self.dimension,
                    processing_time_ms=0.0,
                ))
        
        logger.debug(
            f"Embedded {len(texts)} texts in {elapsed:.1f}ms "
            f"({per_text:.1f}ms/text)"
        )
        
        return results
    
    def embed_query(self, query: str) -> TextEmbeddingResult:
        """
        Embed a search query.
        
        Some models distinguish query vs document embeddings.
        This method handles that automatically.
        
        Args:
            query: Search query string
            
        Returns:
            TextEmbeddingResult optimized for query matching
        """
        # For retrieval models, prepend query instruction
        if 'qa' in self.model_id.lower() or 'retrieval' in self.model_id.lower():
            # Some models use instruction prefixes
            formatted_query = f"query: {query}"
        else:
            formatted_query = query
        
        return self.embed(formatted_query)
    
    def embed_document(self, document: str) -> TextEmbeddingResult:
        """
        Embed a document/passage for retrieval.
        
        Args:
            document: Document text (book metadata, description)
            
        Returns:
            TextEmbeddingResult optimized for document indexing
        """
        # For retrieval models, prepend passage instruction
        if 'qa' in self.model_id.lower() or 'retrieval' in self.model_id.lower():
            formatted_doc = f"passage: {document}"
        else:
            formatted_doc = document
        
        return self.embed(formatted_doc)
    
    def similarity(
        self,
        embedding1: Union[np.ndarray, TextEmbeddingResult],
        embedding2: Union[np.ndarray, TextEmbeddingResult],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (array or result)
            embedding2: Second embedding (array or result)
            
        Returns:
            Similarity score (0-1 for normalized embeddings)
        """
        # Extract arrays
        vec1 = embedding1.embedding if isinstance(embedding1, TextEmbeddingResult) else embedding1
        vec2 = embedding2.embedding if isinstance(embedding2, TextEmbeddingResult) else embedding2
        
        # Cosine similarity
        if self.normalize_embeddings:
            # For normalized vectors, dot product = cosine similarity
            return float(np.dot(vec1, vec2))
        else:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def find_similar(
        self,
        query_embedding: Union[np.ndarray, TextEmbeddingResult],
        candidate_embeddings: list[Union[np.ndarray, TextEmbeddingResult]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity) tuples, sorted by similarity
        """
        query_vec = (
            query_embedding.embedding 
            if isinstance(query_embedding, TextEmbeddingResult) 
            else query_embedding
        )
        
        # Compute all similarities
        similarities = []
        for i, cand in enumerate(candidate_embeddings):
            cand_vec = cand.embedding if isinstance(cand, TextEmbeddingResult) else cand
            sim = self.similarity(query_vec, cand_vec)
            similarities.append((i, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def to(self, device: str) -> 'TextEmbedder':
        """
        Move model to device.
        
        Args:
            device: 'cuda' or 'cpu'
            
        Returns:
            Self for chaining
        """
        self.model = self.model.to(device)
        self.device = device
        return self


class BookTextEmbedder(TextEmbedder):
    """
    Text embedder specialized for book metadata.
    
    Adds book-specific preprocessing:
    - Combines title and author intelligently
    - Handles missing fields gracefully
    - Optimized prompt formatting for books
    """
    
    def __init__(self, **kwargs):
        # Default to balanced model for better quality
        if 'model_name' not in kwargs:
            kwargs['model_name'] = 'balanced'
        super().__init__(**kwargs)
    
    def embed_book_metadata(
        self,
        title: Optional[str] = None,
        author: Optional[str] = None,
        description: Optional[str] = None,
        isbn: Optional[str] = None,
    ) -> TextEmbeddingResult:
        """
        Embed structured book metadata.
        
        Combines fields into an optimized format for matching.
        
        Args:
            title: Book title
            author: Author name(s)
            description: Book description/summary
            isbn: ISBN (included for exact matching)
            
        Returns:
            TextEmbeddingResult for the combined metadata
        """
        parts = []
        
        # Build composite text
        if title:
            parts.append(title)
        
        if author:
            parts.append(f"by {author}")
        
        if description:
            # Truncate long descriptions
            desc = description[:500] if len(description) > 500 else description
            parts.append(desc)
        
        # Combine
        combined = " ".join(parts)
        
        if not combined.strip():
            # Return zero embedding for empty metadata
            return TextEmbeddingResult(
                text="",
                embedding=np.zeros(self.dimension, dtype=np.float32),
                model_name=self.model_name,
                dimension=self.dimension,
            )
        
        return self.embed_document(combined)
    
    def embed_ocr_text(
        self,
        ocr_text: str,
        is_spine: bool = False,
        normalized_text: Optional[str] = None,
    ) -> TextEmbeddingResult:
        """
        Embed OCR-extracted text.
        
        Handles the messiness of OCR output.
        
        Args:
            ocr_text: Raw OCR text
            is_spine: Whether from spine (may affect preprocessing)
            normalized_text: Pre-normalized text if available
            
        Returns:
            TextEmbeddingResult for the OCR text
        """
        # Use normalized if available, otherwise use raw
        text = normalized_text if normalized_text else ocr_text
        
        # For spines, text is often just title + author
        if is_spine:
            return self.embed_query(text)  # Treat as query-like
        else:
            return self.embed(text)  # Standard embedding
