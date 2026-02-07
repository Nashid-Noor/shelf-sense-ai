"""
Dependency injection for FastAPI routes.

Provides injectable dependencies for:
- Database sessions
- Service instances (detector, OCR, RAG, etc.)
- Configuration
- Authentication
"""

import os
from typing import AsyncGenerator, Optional
from functools import lru_cache
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Settings:
    """Application settings loaded from environment."""
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./shelfsense.db"
    database_echo: bool = False
    
    # Vector store
    vector_store_path: str = "./data/vectors"
    vector_store_dimension: int = 768
    
    # Models
    yolo_model_path: str = "./models/yolov8n.pt"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    sentence_model_name: str = "all-MiniLM-L6-v2"
    
    # OCR
    ocr_engine: str = "easyocr"  # or "tesseract"
    ocr_languages: str = "en"
    
    # LLM
    llm_provider: str = "anthropic"  # anthropic, openai, google
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    llm_model: str = "claude-3-sonnet-20240229"
    
    # External APIs
    google_books_api_key: Optional[str] = None
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    
    # File uploads
    max_upload_size_mb: int = 10
    allowed_image_types: str = "image/jpeg,image/png,image/webp"
    
    # Conversation
    max_conversation_history: int = 10
    conversation_ttl_hours: int = 24
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            database_echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            vector_store_path=os.getenv("VECTOR_STORE_PATH", cls.vector_store_path),
            vector_store_dimension=int(os.getenv("VECTOR_DIMENSION", cls.vector_store_dimension)),
            yolo_model_path=os.getenv("YOLO_MODEL_PATH", cls.yolo_model_path),
            clip_model_name=os.getenv("CLIP_MODEL", cls.clip_model_name),
            sentence_model_name=os.getenv("SENTENCE_MODEL", cls.sentence_model_name),
            ocr_engine=os.getenv("OCR_ENGINE", cls.ocr_engine),
            ocr_languages=os.getenv("OCR_LANGUAGES", cls.ocr_languages),
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            google_books_api_key=os.getenv("GOOGLE_BOOKS_API_KEY"),
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", cls.rate_limit_requests_per_minute)),
            max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", cls.max_upload_size_mb)),
            allowed_image_types=os.getenv("ALLOWED_IMAGE_TYPES", cls.allowed_image_types),
            max_conversation_history=int(os.getenv("MAX_CONVERSATION_HISTORY", cls.max_conversation_history)),
            conversation_ttl_hours=int(os.getenv("CONVERSATION_TTL_HOURS", cls.conversation_ttl_hours)),
            environment=os.getenv("SHELFSENSE_ENV", cls.environment),
            debug=os.getenv("DEBUG", "true").lower() == "true",
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings.from_env()


# =============================================================================
# Database
# =============================================================================

# Global engine and session factory (initialized in lifespan)
_engine = None
_async_session_factory = None


def init_database(settings: Settings) -> None:
    """Initialize database engine and session factory."""
    global _engine, _async_session_factory
    
    _engine = create_async_engine(
        settings.database_url,
        echo=settings.database_echo,
        pool_pre_ping=True,
    )
    
    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.
    
    Yields:
        AsyncSession for database operations.
    """
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database first.")
    
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Create database tables."""
    from ..storage.models import Base
    if _engine is None:
        raise RuntimeError("Database not initialized.")
        
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# =============================================================================
# Service Dependencies (Lazy Loading)
# =============================================================================

# Service singletons (initialized on first use)
_services = {}


class ServiceContainer:
    """
    Container for lazy-loaded service instances.
    
    Services are initialized on first access to avoid startup delays.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._book_repository = None
        self._vector_store = None
        self._hybrid_retriever = None
        self._text_embedder = None
        self._visual_embedder = None
        self._detector_ensemble = None
        self._ocr_engine = None
        self._matcher = None
        self._metadata_enricher = None
        self._rag_retriever = None
        self._generator = None
        self._conversation_manager = None
        self._recommender = None
        self._genre_analyzer = None
        self._diversity_calculator = None
        self._genre_analyzer = None
        self._diversity_calculator = None
        self._identification_service = None
        self._orchestrator = None
    
    @property
    def book_repository(self):
        """Get book repository instance."""
        if self._book_repository is None:
            from ..storage.book_repository import BookRepository
            self._book_repository = BookRepository(self.settings.database_url)
        return self._book_repository
    
    @property
    def vector_store(self):
        """Get vector store instance."""
        if self._vector_store is None:
            from ..storage.vector_store import VectorStore
            self._vector_store = VectorStore(
                text_dimension=self.settings.vector_store_dimension,
                data_dir=self.settings.vector_store_path,
            )
        return self._vector_store
    
    @property
    def hybrid_retriever(self):
        """Get hybrid retriever instance."""
        if self._hybrid_retriever is None:
            from ..storage.hybrid_retriever import HybridRetriever
            self._hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                text_embedder=self.text_embedder,
            )
        return self._hybrid_retriever
    
    @property
    def text_embedder(self):
        """Get text embedder instance."""
        if self._text_embedder is None:
            from ..embeddings.text_embedder import TextEmbedder
            self._text_embedder = TextEmbedder(
                model_name=self.settings.sentence_model_name
            )
        return self._text_embedder
    
    @property
    def visual_embedder(self):
        """Get visual embedder instance."""
        if self._visual_embedder is None:
            from ..embeddings.visual_embedder import VisualEmbedder
            self._visual_embedder = VisualEmbedder(
                model_name=self.settings.clip_model_name
            )
        return self._visual_embedder
    
    @property
    def detector_ensemble(self):
        """Get detector ensemble instance."""
        if self._detector_ensemble is None:
            from ..vision.detector_ensemble import DetectorEnsemble
            from ..vision.layout_classifier import create_synthetic_classifier
            
            # Use synthetic classifier since we don't have trained layout weights
            layout_classifier = create_synthetic_classifier()
            
            self._detector_ensemble = DetectorEnsemble(
                layout_classifier=layout_classifier,
                spine_model_path=self.settings.yolo_model_path,
                cover_model_path=self.settings.yolo_model_path,
            )
        return self._detector_ensemble
    
    @property
    def ocr_engine(self):
        """Get OCR engine instance."""
        if self._ocr_engine is None:
            from ..ocr.ocr_engine import OCREngine
            self._ocr_engine = OCREngine(
                languages=self.settings.ocr_languages.split(","),
                prefer_easyocr=(self.settings.ocr_engine == "easyocr"),
            )
        return self._ocr_engine
    
    @property
    def matcher(self):
        """Get book matcher instance."""
        if self._matcher is None:
            from ..identification.matcher import BookMatcher
            self._matcher = BookMatcher(
                text_embedder=self.text_embedder,
                vector_store=self.vector_store,
            )
        return self._matcher
    
    @property
    def metadata_enricher(self):
        """Get metadata enricher instance."""
        if self._metadata_enricher is None:
            from ..identification.metadata_enricher import MetadataEnricher
            self._metadata_enricher = MetadataEnricher(
                google_books_api_key=self.settings.google_books_api_key,
            )
        return self._metadata_enricher
    
    @property
    def rag_retriever(self):
        """Get RAG retriever instance."""
        if self._rag_retriever is None:
            from ..rag.retriever import RAGRetriever
            self._rag_retriever = RAGRetriever(
                hybrid_retriever=self.hybrid_retriever,
                book_repository=self.book_repository,
                text_embedder=self.text_embedder,
            )
        return self._rag_retriever
    
    @property
    def generator(self):
        """Get response generator instance."""
        if self._generator is None:
            from ..rag.generator import create_generator, LLMProvider
            
            provider = self.settings.llm_provider
            api_key = None
            
            if provider == "anthropic":
                api_key = self.settings.anthropic_api_key
                provider_enum = LLMProvider.ANTHROPIC
            elif provider == "openai":
                api_key = self.settings.openai_api_key
                provider_enum = LLMProvider.OPENAI
            elif provider == "google":
                api_key = self.settings.google_api_key
                provider_enum = LLMProvider.GOOGLE
            else:
                provider_enum = LLMProvider.ANTHROPIC
            
            self._generator = create_generator(
                provider=provider_enum,
                api_key=api_key,
                model=self.settings.llm_model,
            )
        return self._generator
    
    @property
    def conversation_manager(self):
        """Get conversation manager instance."""
        if self._conversation_manager is None:
            from ..rag.conversation import ConversationManager, ConversationStore
            from pathlib import Path
            
            # Use a persistent store in the data directory
            store_path = Path(self.settings.vector_store_path).parent / "conversations"
            store = ConversationStore(storage_path=store_path)
            
            self._conversation_manager = ConversationManager(
                store=store,
                max_turns=50, # Keep reasonable history
                context_window_turns=self.settings.max_conversation_history,
            )
        return self._conversation_manager
    
    @property
    def orchestrator(self):
        """Get conversation orchestrator instance."""
        if self._orchestrator is None:
            from ..rag.conversation import ConversationOrchestrator
            self._orchestrator = ConversationOrchestrator(
                retriever=self.rag_retriever,
                generator=self.generator,
                manager=self.conversation_manager,
            )
        return self._orchestrator
    
    @property
    def recommender(self):
        """Get book recommender instance."""
        if self._recommender is None:
            from ..intelligence.recommender import BookRecommender
            self._recommender = BookRecommender()
        return self._recommender
    
    @property
    def genre_analyzer(self):
        """Get genre analyzer instance."""
        if self._genre_analyzer is None:
            from ..intelligence.genre_analyzer import GenreAnalyzer
            self._genre_analyzer = GenreAnalyzer()
        return self._genre_analyzer
    
    @property
    def identification_service(self):
        """Get identification service instance."""
        if self._identification_service is None:
            from ..identification.service import IdentificationService
            from ..identification.google_books import GoogleBooksClient
            
            client = GoogleBooksClient(api_key=self.settings.google_books_api_key)
            self._identification_service = IdentificationService(google_books_client=client)
        return self._identification_service

    @property
    def diversity_calculator(self):
        """Get diversity calculator instance."""
        if self._diversity_calculator is None:
            from ..intelligence.diversity_metrics import DiversityCalculator
            self._diversity_calculator = DiversityCalculator()
        return self._diversity_calculator


# Global service container
_service_container: Optional[ServiceContainer] = None


def init_services(settings: Settings) -> ServiceContainer:
    """Initialize service container."""
    global _service_container
    _service_container = ServiceContainer(settings)
    return _service_container


def get_service_container() -> ServiceContainer:
    """Get service container instance."""
    if _service_container is None:
        # Auto-initialize with default settings if not explicitly initialized
        return init_services(get_settings())
    return _service_container


# =============================================================================
# Individual Service Dependencies
# =============================================================================

def get_identification_service(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for identification service."""
    return container.identification_service

def get_book_repository(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for book repository."""
    return container.book_repository


def get_vector_store(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for vector store."""
    return container.vector_store


def get_hybrid_retriever(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for hybrid retriever."""
    return container.hybrid_retriever


def get_detector_ensemble(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for detector ensemble."""
    return container.detector_ensemble


def get_ocr_engine(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for OCR engine."""
    return container.ocr_engine


def get_rag_retriever(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for RAG retriever."""
    return container.rag_retriever


def get_generator(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for response generator."""
    return container.generator


def get_conversation_manager(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for conversation manager."""
    return container.conversation_manager


def get_rag_orchestrator(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for conversation orchestrator."""
    return container.orchestrator


def get_recommender(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for book recommender."""
    return container.recommender


def get_genre_analyzer(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for genre analyzer."""
    return container.genre_analyzer


def get_diversity_calculator(
    container: ServiceContainer = Depends(get_service_container),
):
    """Dependency for diversity calculator."""
    return container.diversity_calculator


# =============================================================================
# Authentication Dependencies
# =============================================================================

async def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """
    Extract API key from header.
    
    Returns None if no key provided (for public endpoints).
    """
    return x_api_key


async def require_api_key(
    api_key: Optional[str] = Depends(get_api_key),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Require valid API key for protected endpoints.
    
    Raises:
        HTTPException: If API key is missing or invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # In production, validate against stored keys
    # For now, just check it exists
    return api_key


# =============================================================================
# Request Context Dependencies
# =============================================================================

async def get_client_ip(request: Request) -> str:
    """Extract client IP from request, considering proxies."""
    # Check X-Forwarded-For header
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct client
    if request.client:
        return request.client.host
    
    return "unknown"


async def get_user_agent(request: Request) -> str:
    """Extract user agent from request."""
    return request.headers.get("User-Agent", "unknown")
