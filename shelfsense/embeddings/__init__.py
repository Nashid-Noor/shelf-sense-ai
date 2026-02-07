"""
Multimodal Embeddings Module

Generates and fuses embeddings for book identification:
- Text embeddings (Sentence-BERT)
- Visual embeddings (CLIP)
- Late fusion with adaptive weighting
- Embedding caching and batch processing
"""

from shelfsense.embeddings.text_embedder import TextEmbedder
from shelfsense.embeddings.visual_embedder import VisualEmbedder
from shelfsense.embeddings.fusion import EmbeddingFusion, FusedEmbedding
from shelfsense.embeddings.cache import EmbeddingCache

__all__ = [
    "TextEmbedder",
    "VisualEmbedder", 
    "EmbeddingFusion",
    "FusedEmbedding",
    "EmbeddingCache",
]
