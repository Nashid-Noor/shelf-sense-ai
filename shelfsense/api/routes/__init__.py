"""
API Routes for ShelfSense AI

Route modules:
- books: Book CRUD and search
- detection: Image upload and book detection
- chat: RAG conversational interface
- analytics: Library insights and statistics
"""

from shelfsense.api.routes.books import router as books_router
from shelfsense.api.routes.detection import router as detection_router
from shelfsense.api.routes.chat import router as chat_router
from shelfsense.api.routes.analytics import router as analytics_router

__all__ = [
    "books_router",
    "detection_router", 
    "chat_router",
    "analytics_router",
]
