"""
ShelfSense API

FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .schemas import HealthResponse
from .routes import books, detection, chat, analytics, auth
from .middleware import (
    setup_cors,
    setup_rate_limiting,
    setup_logging,
    setup_exception_handlers,
    RateLimitConfig,
    LoggingConfig,
    get_cors_config,
)
from .dependencies import (
    get_settings,
    init_database,
    create_tables,
    init_services,
    Settings,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks:
    - Initialize database connections
    - Load ML models
    - Create indexes
    - Cleanup on shutdown
    """
    settings = get_settings()
    logger.info(f"Starting ShelfSense in {settings.environment} mode")
    
    # Startup
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database(settings)
        await create_tables()
        
        # Initialize services (lazy loading, but validate config)
        logger.info("Initializing services...")
        services = init_services(settings)
        
        # Store services in app state for access in routes
        app.state.services = services
        app.state.settings = settings
        
        # Sync search index
        try:
            book_repo = services.book_repository
            retriever = services.hybrid_retriever
            
            books = book_repo.list_all(limit=10000)
            if books:
                logger.info(f"Syncing {len(books)} books to search index...")
                documents = []
                for book in books:
                    # Construct search text
                    text_parts = [book.title, book.author]
                    if book.description:
                        text_parts.append(book.description)
                    if hasattr(book, 'subjects') and book.subjects:
                        text_parts.extend(book.subjects)
                    
                    full_text = " ".join(str(p) for p in text_parts)
                    
                    documents.append({
                        "id": book.id,
                        "text": full_text,
                        "metadata": {
                            "title": book.title,
                            "author": book.author,
                        }
                    })
                
                retriever.index_documents(documents)
                logger.info("Index sync complete.")
        except Exception as e:
            logger.error(f"Failed to sync index: {e}")
        
        # Pre-warm critical services in production
        if settings.environment == "production":
            logger.info("Pre-warming services...")
            # Touch services to trigger lazy loading
            _ = services.text_embedder
            _ = services.detector_ensemble
        
        logger.info("ShelfSense started successfully")
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down ShelfSense...")
        
        # Cleanup resources
        # Vector store persistence
        if hasattr(app.state, "services") and app.state.services._vector_store:
            logger.info("Persisting vector store...")
            app.state.services.vector_store.save()
        
        logger.info("Shutdown complete")


# =============================================================================
# Application Factory
# =============================================================================

def create_app(settings: Settings = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Application settings. If None, loads from environment.
    
    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title="ShelfSense",
        description="Personal library intelligence system.",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # ==========================================================================
    # Middleware (order matters - first added = outermost)
    # ==========================================================================
    
    # 1. Logging (outermost - captures everything)
    setup_logging(
        app,
        config=LoggingConfig(
            enabled=True,
            log_request_body=settings.debug,
            log_response_body=False,
        ),
        structured=settings.environment != "development",
    )
    
    # 2. Exception handling
    setup_exception_handlers(app)
    
    # 3. Rate limiting
    if settings.rate_limit_enabled:
        setup_rate_limiting(
            app,
            config=RateLimitConfig(
                requests_per_minute=settings.rate_limit_requests_per_minute,
                enabled=settings.rate_limit_enabled,
            ),
        )
    
    # 4. CORS (innermost - only affects actual responses)
    setup_cors(app, config=get_cors_config(settings.environment))
    
    # ==========================================================================
    # Routers
    # ==========================================================================
    
    # API version prefix
    api_prefix = "/api/v1"
    
    # Register route handlers
    app.include_router(
        auth.router,
        prefix=api_prefix,
    )

    app.include_router(
        books.router,
        prefix=api_prefix,
    )
    
    app.include_router(
        detection.router,
        prefix=api_prefix,
    )
    
    app.include_router(
        chat.router,
        prefix=api_prefix,
    )
    
    app.include_router(
        analytics.router,
        prefix=api_prefix,
    )
    
    # ==========================================================================
    # Root Routes
    # ==========================================================================
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "ShelfSense",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs" if settings.debug else None,
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check(request: Request) -> HealthResponse:
        """
        Health check endpoint.
        
        Returns status of all system components.
        """
        services = request.app.state.services
        
        # Check component health
        components = {}
        overall_healthy = True
        
        # Database
        try:
            # Quick DB check would go here
            components["database"] = "healthy"
        except Exception as e:
            components["database"] = f"unhealthy: {str(e)}"
            overall_healthy = False
        
        # Vector store
        try:
            if services._vector_store:
                components["vector_store"] = "healthy"
            else:
                components["vector_store"] = "not_initialized"
        except Exception as e:
            components["vector_store"] = f"unhealthy: {str(e)}"
            overall_healthy = False
        
        # ML models (check if loadable)
        components["ml_models"] = "healthy"  # Lazy loaded
        
        # LLM API
        if settings.anthropic_api_key or settings.openai_api_key:
            components["llm_api"] = "configured"
        else:
            components["llm_api"] = "not_configured"
        
        return HealthResponse(
            status="healthy" if overall_healthy else "degraded",
            version="1.0.0",
            components=components,
        )
    
    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request):
        """
        Basic metrics endpoint.
        
        For production, integrate with Prometheus or similar.
        """
        # Placeholder - would integrate with actual metrics
        return {
            "requests_total": 0,
            "active_connections": 0,
            "books_indexed": 0,
            "conversations_active": 0,
        }
    
    return app


# =============================================================================
# Application Instance
# =============================================================================

# Create application instance
app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the application using uvicorn."""
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "shelfsense.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
