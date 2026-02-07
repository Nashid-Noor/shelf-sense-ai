"""
Embedding Cache

Efficient caching of computed embeddings:
- LRU cache for in-memory caching
- Persistent storage with SQLite/pickle
- Content-addressed storage (hash-based keys)
- Batch operations for efficiency
"""

import hashlib
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from collections import OrderedDict
import numpy as np
from loguru import logger


@dataclass
class CacheEntry:
    """Single cache entry."""
    
    key: str
    embedding: np.ndarray
    embedding_type: str  # "text" or "visual"
    model_name: str
    created_at: float
    
    # Optional metadata
    source_hash: Optional[str] = None
    metadata: Optional[dict] = None


class LRUEmbeddingCache:
    """
    In-memory LRU cache for embeddings.
    
    Fast access with automatic eviction of least recently used entries.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries to cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        
        logger.info(f"LRUEmbeddingCache initialized: max_size={max_size}")
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Embedding array or None if not found
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key].embedding
        
        self._misses += 1
        return None
    
    def put(
        self,
        key: str,
        embedding: np.ndarray,
        embedding_type: str = "unknown",
        model_name: str = "unknown",
        metadata: Optional[dict] = None,
    ):
        """
        Store embedding in cache.
        
        Args:
            key: Cache key
            embedding: Embedding vector
            embedding_type: "text" or "visual"
            model_name: Model that generated the embedding
            metadata: Optional metadata
        """
        import time
        
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest
        
        entry = CacheEntry(
            key=key,
            embedding=embedding,
            embedding_type=embedding_type,
            model_name=model_name,
            created_at=time.time(),
            metadata=metadata,
        )
        
        self._cache[key] = entry
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache
    
    def remove(self, key: str):
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
    
    def clear(self):
        """Clear all entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class PersistentEmbeddingCache:
    """
    Persistent cache using SQLite for embeddings.
    
    Stores embeddings on disk for recovery across restarts.
    Uses content-addressed storage with hash-based keys.
    """
    
    def __init__(
        self,
        db_path: Union[str, Path],
        table_name: str = "embeddings",
        use_memory_cache: bool = True,
        memory_cache_size: int = 5000,
    ):
        """
        Initialize persistent cache.
        
        Args:
            db_path: Path to SQLite database
            table_name: Table name for embeddings
            use_memory_cache: Also use in-memory LRU cache
            memory_cache_size: Size of memory cache
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        # Optional memory cache layer
        self._memory_cache: Optional[LRUEmbeddingCache] = None
        if use_memory_cache:
            self._memory_cache = LRUEmbeddingCache(max_size=memory_cache_size)
        
        logger.info(
            f"PersistentEmbeddingCache initialized: "
            f"db={db_path}, memory_cache={use_memory_cache}"
        )
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                embedding_type TEXT,
                model_name TEXT,
                dimension INTEGER,
                source_hash TEXT,
                metadata TEXT,
                created_at REAL,
                accessed_at REAL
            )
        """)
        
        # Index for faster lookups
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source_hash
            ON {self.table_name}(source_hash)
        """)
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Embedding array or None
        """
        # Check memory cache first
        if self._memory_cache:
            result = self._memory_cache.get(key)
            if result is not None:
                return result
        
        # Query database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            f"SELECT embedding FROM {self.table_name} WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        # Update access time
        if row:
            import time
            cursor.execute(
                f"UPDATE {self.table_name} SET accessed_at = ? WHERE key = ?",
                (time.time(), key)
            )
            conn.commit()
        
        conn.close()
        
        if row is None:
            return None
        
        # Deserialize embedding
        embedding = pickle.loads(row[0])
        
        # Populate memory cache
        if self._memory_cache:
            self._memory_cache.put(key, embedding)
        
        return embedding
    
    def put(
        self,
        key: str,
        embedding: np.ndarray,
        embedding_type: str = "unknown",
        model_name: str = "unknown",
        source_hash: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Store embedding in cache.
        
        Args:
            key: Cache key
            embedding: Embedding vector
            embedding_type: "text" or "visual"
            model_name: Model that generated embedding
            source_hash: Hash of source content
            metadata: Optional metadata dict
        """
        import time
        import json
        
        # Serialize embedding
        embedding_blob = pickle.dumps(embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            INSERT OR REPLACE INTO {self.table_name}
            (key, embedding, embedding_type, model_name, dimension, 
             source_hash, metadata, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            key,
            embedding_blob,
            embedding_type,
            model_name,
            len(embedding),
            source_hash,
            json.dumps(metadata) if metadata else None,
            time.time(),
            time.time(),
        ))
        
        conn.commit()
        conn.close()
        
        # Update memory cache
        if self._memory_cache:
            self._memory_cache.put(
                key, embedding, embedding_type, model_name, metadata
            )
    
    def contains(self, key: str) -> bool:
        """Check if key exists."""
        if self._memory_cache and self._memory_cache.contains(key):
            return True
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT 1 FROM {self.table_name} WHERE key = ? LIMIT 1",
            (key,)
        )
        result = cursor.fetchone() is not None
        conn.close()
        
        return result
    
    def get_by_source_hash(self, source_hash: str) -> Optional[np.ndarray]:
        """
        Get embedding by source content hash.
        
        Useful when the same content might have different keys.
        
        Args:
            source_hash: Hash of source content
            
        Returns:
            Embedding or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            f"SELECT key, embedding FROM {self.table_name} WHERE source_hash = ? LIMIT 1",
            (source_hash,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        key, embedding_blob = row
        embedding = pickle.loads(embedding_blob)
        
        # Populate memory cache
        if self._memory_cache:
            self._memory_cache.put(key, embedding)
        
        return embedding
    
    def remove(self, key: str):
        """Remove entry from cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM {self.table_name} WHERE key = ?",
            (key,)
        )
        conn.commit()
        conn.close()
        
        if self._memory_cache:
            self._memory_cache.remove(key)
    
    def clear(self):
        """Clear all entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {self.table_name}")
        conn.commit()
        conn.close()
        
        if self._memory_cache:
            self._memory_cache.clear()
    
    def count(self) -> int:
        """Get total number of cached entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def cleanup(self, max_age_days: int = 30, max_entries: int = 100000):
        """
        Clean up old entries.
        
        Args:
            max_age_days: Remove entries older than this
            max_entries: Maximum entries to keep
        """
        import time
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Remove old entries
        cutoff = time.time() - (max_age_days * 24 * 60 * 60)
        cursor.execute(
            f"DELETE FROM {self.table_name} WHERE accessed_at < ?",
            (cutoff,)
        )
        deleted_old = cursor.rowcount
        
        # Remove excess entries (keep most recently accessed)
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        current_count = cursor.fetchone()[0]
        
        deleted_excess = 0
        if current_count > max_entries:
            to_delete = current_count - max_entries
            cursor.execute(f"""
                DELETE FROM {self.table_name}
                WHERE key IN (
                    SELECT key FROM {self.table_name}
                    ORDER BY accessed_at ASC
                    LIMIT ?
                )
            """, (to_delete,))
            deleted_excess = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(
            f"Cache cleanup: removed {deleted_old} old, "
            f"{deleted_excess} excess entries"
        )
    
    def stats(self) -> dict:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        total_count = cursor.fetchone()[0]
        
        cursor.execute(f"""
            SELECT embedding_type, COUNT(*) 
            FROM {self.table_name} 
            GROUP BY embedding_type
        """)
        type_counts = dict(cursor.fetchall())
        
        cursor.execute(f"""
            SELECT model_name, COUNT(*) 
            FROM {self.table_name} 
            GROUP BY model_name
        """)
        model_counts = dict(cursor.fetchall())
        
        conn.close()
        
        memory_stats = {}
        if self._memory_cache:
            memory_stats = self._memory_cache.stats()
        
        return {
            "total_entries": total_count,
            "by_type": type_counts,
            "by_model": model_counts,
            "db_path": str(self.db_path),
            "memory_cache": memory_stats,
        }


class EmbeddingCache:
    """
    Unified embedding cache interface.
    
    Combines memory and persistent caching with
    content-addressed storage.
    """
    
    def __init__(
        self,
        persistent_path: Optional[Union[str, Path]] = None,
        memory_only: bool = False,
        memory_cache_size: int = 5000,
    ):
        """
        Initialize unified cache.
        
        Args:
            persistent_path: Path for persistent storage (None for memory only)
            memory_only: Force memory-only mode
            memory_cache_size: Size of memory cache
        """
        if memory_only or persistent_path is None:
            self._cache = LRUEmbeddingCache(max_size=memory_cache_size)
            self._persistent = False
        else:
            self._cache = PersistentEmbeddingCache(
                db_path=persistent_path,
                memory_cache_size=memory_cache_size,
            )
            self._persistent = True
        
        logger.info(
            f"EmbeddingCache initialized: "
            f"persistent={self._persistent}"
        )
    
    @staticmethod
    def compute_text_key(text: str, model_name: str) -> str:
        """
        Compute cache key for text.
        
        Args:
            text: Input text
            model_name: Embedding model name
            
        Returns:
            Cache key string
        """
        content = f"text:{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    @staticmethod
    def compute_image_key(
        image: Union[np.ndarray, bytes],
        model_name: str,
    ) -> str:
        """
        Compute cache key for image.
        
        Args:
            image: Image array or bytes
            model_name: Embedding model name
            
        Returns:
            Cache key string
        """
        if isinstance(image, np.ndarray):
            # Hash image data
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        else:
            image_hash = hashlib.sha256(image).hexdigest()[:16]
        
        return f"img:{model_name}:{image_hash}"
    
    def get_text_embedding(
        self,
        text: str,
        model_name: str,
    ) -> Optional[np.ndarray]:
        """Get cached text embedding."""
        key = self.compute_text_key(text, model_name)
        return self._cache.get(key)
    
    def put_text_embedding(
        self,
        text: str,
        model_name: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
    ):
        """Cache text embedding."""
        key = self.compute_text_key(text, model_name)
        
        if self._persistent:
            self._cache.put(
                key, embedding,
                embedding_type="text",
                model_name=model_name,
                source_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
                metadata=metadata,
            )
        else:
            self._cache.put(
                key, embedding,
                embedding_type="text",
                model_name=model_name,
                metadata=metadata,
            )
    
    def get_image_embedding(
        self,
        image: Union[np.ndarray, bytes],
        model_name: str,
    ) -> Optional[np.ndarray]:
        """Get cached image embedding."""
        key = self.compute_image_key(image, model_name)
        return self._cache.get(key)
    
    def put_image_embedding(
        self,
        image: Union[np.ndarray, bytes],
        model_name: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
    ):
        """Cache image embedding."""
        key = self.compute_image_key(image, model_name)
        
        if isinstance(image, np.ndarray):
            source_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        else:
            source_hash = hashlib.sha256(image).hexdigest()[:16]
        
        if self._persistent:
            self._cache.put(
                key, embedding,
                embedding_type="visual",
                model_name=model_name,
                source_hash=source_hash,
                metadata=metadata,
            )
        else:
            self._cache.put(
                key, embedding,
                embedding_type="visual",
                model_name=model_name,
                metadata=metadata,
            )
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.stats()
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
