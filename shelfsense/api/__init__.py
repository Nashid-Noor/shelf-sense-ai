"""
ShelfSense AI - FastAPI Backend.

Production-ready API for personal library intelligence.
"""

from .main import app, create_app, main
from .dependencies import (
    Settings,
    get_settings,
    get_db,
    get_service_container,
    ServiceContainer,
)
from .schemas import (
    BookBase,
    BookCreate,
    BookUpdate,
    BookResponse,
    BookListResponse,
    BookSearchRequest,
    BookSearchResponse,
    DetectionRequest,
    DetectionResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ErrorResponse,
)

__all__ = [
    # Application
    "app",
    "create_app",
    "main",
    # Dependencies
    "Settings",
    "get_settings",
    "get_db",
    "get_service_container",
    "ServiceContainer",
    # Schemas
    "BookBase",
    "BookCreate",
    "BookUpdate",
    "BookResponse",
    "BookListResponse",
    "BookSearchRequest",
    "BookSearchResponse",
    "DetectionRequest",
    "DetectionResponse",
    "ChatRequest",
    "ChatResponse",
    "HealthResponse",
    "ErrorResponse",
]
