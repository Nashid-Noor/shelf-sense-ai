"""
Visual Embedder for ShelfSense AI

CLIP-based visual embedding for book image matching:
- Cover image recognition
- Spine visual features
- Cross-modal text-image matching
- Efficient batch processing
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import torch
from PIL import Image
from loguru import logger

# CLIP imports - supports multiple backends
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_BACKEND = 'transformers'
except ImportError:
    try:
        import clip
        CLIP_BACKEND = 'openai'
    except ImportError:
        CLIP_BACKEND = None
        logger.warning("No CLIP backend available. Install transformers or openai-clip.")


@dataclass
class VisualEmbeddingResult:
    """Result of visual embedding."""
    
    embedding: np.ndarray
    model_name: str
    dimension: int
    
    # Source info
    source_path: Optional[str] = None
    source_type: str = "unknown"  # "cover", "spine", "roi"
    
    # Processing metadata
    processing_time_ms: float = 0.0
    original_size: Optional[tuple[int, int]] = None
    
    def to_list(self) -> list[float]:
        """Convert embedding to Python list."""
        return self.embedding.tolist()
    
    @property
    def norm(self) -> float:
        """L2 norm of embedding."""
        return float(np.linalg.norm(self.embedding))


class VisualEmbedder:
    """
    Production visual embedder using CLIP models.
    
    Generates visual embeddings for book images that can be:
    - Compared to other visual embeddings (visual similarity)
    - Compared to text embeddings (cross-modal matching)
    
    CLIP models jointly learn image and text representations,
    enabling powerful book cover matching even when OCR fails.
    
    Model Options:
    - 'openai/clip-vit-base-patch32': Fast, good quality
    - 'openai/clip-vit-large-patch14': Higher quality, slower
    - 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K': Trained on larger dataset
    
    Usage:
        embedder = VisualEmbedder()
        result = embedder.embed(image)
        print(result.embedding.shape)  # (512,)
    """
    
    MODELS = {
        'fast': 'openai/clip-vit-base-patch32',
        'balanced': 'openai/clip-vit-base-patch16',
        'quality': 'openai/clip-vit-large-patch14',
    }
    
    def __init__(
        self,
        model_name: str = 'fast',
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize visual embedder.
        
        Args:
            model_name: One of 'fast', 'balanced', 'quality' or HuggingFace model ID
            device: 'cuda', 'cpu', or None for auto-detect
            normalize_embeddings: L2 normalize embeddings
            cache_dir: Directory for model cache
        """
        if CLIP_BACKEND is None:
            raise ImportError(
                "No CLIP backend available. "
                "Install with: pip install transformers torch"
            )
        
        # Resolve model name
        if model_name in self.MODELS:
            self.model_id = self.MODELS[model_name]
        else:
            self.model_id = model_name
        
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        logger.info(f"Loading CLIP model: {self.model_id} on {device}")
        
        # Load model based on backend
        if CLIP_BACKEND == 'transformers':
            self._load_transformers_model(cache_dir)
        else:
            self._load_openai_model()
        
        logger.info(
            f"VisualEmbedder ready: dim={self.dimension}, device={device}"
        )
    
    def _load_transformers_model(self, cache_dir: Optional[Path]):
        """Load CLIP using HuggingFace transformers."""
        self.model = CLIPModel.from_pretrained(
            self.model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
        ).to(self.device)
        self.model.eval()
        
        self.processor = CLIPProcessor.from_pretrained(
            self.model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        
        # Get embedding dimension
        self.dimension = self.model.config.projection_dim
        self._backend = 'transformers'
    
    def _load_openai_model(self):
        """Load CLIP using OpenAI's implementation."""
        model_name = self.model_name.replace('openai/', '').replace('-', '_')
        self.model, self.processor = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Get embedding dimension from model
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            self.dimension = self.model.encode_image(dummy).shape[-1]
        
        self._backend = 'openai'
    
    def embed(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        source_type: str = "unknown",
    ) -> VisualEmbeddingResult:
        """
        Embed a single image.
        
        Args:
            image: PIL Image, numpy array, or path to image
            source_type: "cover", "spine", or "roi"
            
        Returns:
            VisualEmbeddingResult with embedding vector
        """
        import time
        start = time.perf_counter()
        
        # Load image if path
        source_path = None
        if isinstance(image, (str, Path)):
            source_path = str(image)
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        original_size = image.size
        
        # Generate embedding
        with torch.no_grad():
            if self._backend == 'transformers':
                embedding = self._embed_transformers(image)
            else:
                embedding = self._embed_openai(image)
        
        # Normalize if requested
        if self.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return VisualEmbeddingResult(
            embedding=embedding.astype(np.float32),
            model_name=self.model_name,
            dimension=self.dimension,
            source_path=source_path,
            source_type=source_type,
            processing_time_ms=elapsed,
            original_size=original_size,
        )
    
    def _embed_transformers(self, image: Image.Image) -> np.ndarray:
        """Generate embedding using transformers backend."""
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)
        
        image_features = self.model.get_image_features(**inputs)
        
        return image_features.cpu().numpy().squeeze()
    
    def _embed_openai(self, image: Image.Image) -> np.ndarray:
        """Generate embedding using OpenAI CLIP backend."""
        image_input = self.processor(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_input)
        
        return image_features.cpu().numpy().squeeze()
    
    def embed_batch(
        self,
        images: list[Union[Image.Image, np.ndarray, str, Path]],
        batch_size: int = 16,
        source_types: Optional[list[str]] = None,
        show_progress: bool = False,
    ) -> list[VisualEmbeddingResult]:
        """
        Embed multiple images efficiently.
        
        Args:
            images: List of images (PIL, numpy, or paths)
            batch_size: Batch size for GPU processing
            source_types: Optional list of source types
            show_progress: Show progress bar
            
        Returns:
            List of VisualEmbeddingResult objects
        """
        import time
        from tqdm import tqdm
        
        if not images:
            return []
        
        if source_types is None:
            source_types = ["unknown"] * len(images)
        
        start = time.perf_counter()
        
        # Load all images
        pil_images = []
        source_paths = []
        original_sizes = []
        
        for img in images:
            if isinstance(img, (str, Path)):
                source_paths.append(str(img))
                img = Image.open(img).convert('RGB')
            else:
                source_paths.append(None)
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img).convert('RGB')
            
            original_sizes.append(img.size)
            pil_images.append(img)
        
        # Process in batches
        all_embeddings = []
        
        iterator = range(0, len(pil_images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding images")
        
        for i in iterator:
            batch = pil_images[i:i + batch_size]
            
            with torch.no_grad():
                if self._backend == 'transformers':
                    inputs = self.processor(
                        images=batch,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)
                    features = self.model.get_image_features(**inputs)
                else:
                    batch_tensors = torch.stack([
                        self.processor(img) for img in batch
                    ]).to(self.device)
                    features = self.model.encode_image(batch_tensors)
            
            batch_embeddings = features.cpu().numpy()
            
            # Normalize
            if self.normalize_embeddings:
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1)
                batch_embeddings = batch_embeddings / norms
            
            all_embeddings.extend(batch_embeddings)
        
        elapsed = (time.perf_counter() - start) * 1000
        per_image = elapsed / len(images)
        
        # Build results
        results = []
        for i, embedding in enumerate(all_embeddings):
            results.append(VisualEmbeddingResult(
                embedding=embedding.astype(np.float32),
                model_name=self.model_name,
                dimension=self.dimension,
                source_path=source_paths[i],
                source_type=source_types[i],
                processing_time_ms=per_image,
                original_size=original_sizes[i],
            ))
        
        logger.debug(
            f"Embedded {len(images)} images in {elapsed:.1f}ms "
            f"({per_image:.1f}ms/image)"
        )
        
        return results
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text using CLIP's text encoder.
        
        This enables cross-modal matching between:
        - Visual embeddings of book covers
        - Text descriptions of books
        
        Args:
            text: Text to embed (title, description, etc.)
            
        Returns:
            Embedding vector compatible with visual embeddings
        """
        with torch.no_grad():
            if self._backend == 'transformers':
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,  # CLIP's max
                ).to(self.device)
                text_features = self.model.get_text_features(**inputs)
            else:
                text_input = clip.tokenize([text]).to(self.device)
                text_features = self.model.encode_text(text_input)
        
        embedding = text_features.cpu().numpy().squeeze()
        
        if self.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def similarity(
        self,
        embedding1: Union[np.ndarray, VisualEmbeddingResult],
        embedding2: Union[np.ndarray, VisualEmbeddingResult],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Works for:
        - Image vs Image
        - Image vs Text (cross-modal)
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1 for normalized embeddings)
        """
        vec1 = embedding1.embedding if isinstance(embedding1, VisualEmbeddingResult) else embedding1
        vec2 = embedding2.embedding if isinstance(embedding2, VisualEmbeddingResult) else embedding2
        
        if self.normalize_embeddings:
            return float(np.dot(vec1, vec2))
        else:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def match_image_to_texts(
        self,
        image: Union[Image.Image, np.ndarray, VisualEmbeddingResult],
        texts: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, str, float]]:
        """
        Match an image against text descriptions.
        
        Useful for finding which book title best matches a cover image.
        
        Args:
            image: Image or pre-computed embedding
            texts: List of text candidates (titles, descriptions)
            top_k: Number of matches to return
            
        Returns:
            List of (index, text, similarity) tuples
        """
        # Get image embedding
        if isinstance(image, VisualEmbeddingResult):
            img_emb = image.embedding
        else:
            img_emb = self.embed(image).embedding
        
        # Get text embeddings
        text_embeddings = [self.embed_text(t) for t in texts]
        
        # Compute similarities
        similarities = []
        for i, (text, text_emb) in enumerate(zip(texts, text_embeddings)):
            sim = self.similarity(img_emb, text_emb)
            similarities.append((i, text, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        return similarities[:top_k]
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def to(self, device: str) -> 'VisualEmbedder':
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device
        return self


class BookCoverEmbedder(VisualEmbedder):
    """
    Visual embedder specialized for book covers.
    
    Adds book-specific preprocessing:
    - Aspect ratio normalization for covers
    - Edge enhancement for spine detection
    - Color normalization for faded covers
    """
    
    def __init__(self, **kwargs):
        # Default to balanced model
        if 'model_name' not in kwargs:
            kwargs['model_name'] = 'balanced'
        super().__init__(**kwargs)
    
    def preprocess_cover(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Image.Image:
        """
        Preprocess a book cover image before embedding.
        
        Args:
            image: Cover image
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.convert('RGB')
        
        # Resize while maintaining aspect ratio
        # Book covers are typically 2:3 or similar
        max_size = 512
        w, h = image.size
        
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def embed_cover(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
    ) -> VisualEmbeddingResult:
        """
        Embed a book cover with preprocessing.
        
        Args:
            image: Cover image
            
        Returns:
            VisualEmbeddingResult optimized for cover matching
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Apply cover-specific preprocessing
        processed = self.preprocess_cover(image)
        
        return self.embed(processed, source_type="cover")
    
    def embed_spine(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> VisualEmbeddingResult:
        """
        Embed a book spine image.
        
        Spines have different characteristics than covers:
        - Typically narrow and tall
        - Often rotated text
        - Less visual information
        
        Args:
            image: Spine image (already cropped)
            
        Returns:
            VisualEmbeddingResult for spine
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Spines might need rotation
        w, h = image.size
        
        # If wider than tall, rotate
        if w > h:
            image = image.rotate(90, expand=True)
        
        return self.embed(image, source_type="spine")
