"""
Pytest configuration and fixtures for ShelfSense AI tests.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shelfsense.api.main import create_app
from shelfsense.api.dependencies import Settings, get_settings, get_db


# =============================================================================
# Test Settings
# =============================================================================

def get_test_settings() -> Settings:
    """Return settings configured for testing."""
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        database_echo=False,
        vector_store_path="./test_data/vectors",
        environment="test",
        debug=True,
        rate_limit_enabled=False,
        llm_provider="mock",
    )


# =============================================================================
# Event Loop
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create async database engine for tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    
    # Import models and create tables
    from shelfsense.storage.book_repository import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide database session for tests."""
    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with session_factory() as session:
        yield session
        await session.rollback()


# =============================================================================
# Application Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def app(async_engine):
    """Create FastAPI application for testing."""
    application = create_app()
    
    # Override dependencies
    application.dependency_overrides[get_settings] = get_test_settings
    
    yield application
    
    application.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Provide async HTTP client for API tests."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_bookshelf_image() -> Image.Image:
    """Generate synthetic bookshelf image for testing."""
    # Create a 640x480 image with colored rectangles representing book spines
    img = Image.new("RGB", (640, 480), color=(240, 240, 240))
    pixels = np.array(img)
    
    # Add some "book spines" as colored vertical rectangles
    spine_colors = [
        (150, 50, 50),    # Red
        (50, 150, 50),    # Green
        (50, 50, 150),    # Blue
        (150, 150, 50),   # Yellow
        (150, 50, 150),   # Purple
    ]
    
    x_start = 50
    for i, color in enumerate(spine_colors):
        width = 40 + (i * 5)
        pixels[100:400, x_start:x_start+width] = color
        x_start += width + 10
    
    return Image.fromarray(pixels)


@pytest.fixture
def sample_book_cover_image() -> Image.Image:
    """Generate synthetic book cover image for testing."""
    img = Image.new("RGB", (300, 450), color=(200, 180, 160))
    pixels = np.array(img)
    
    # Add a "title area" at top
    pixels[30:80, 30:270] = (50, 50, 50)
    
    # Add "author area" at bottom
    pixels[380:410, 30:200] = (80, 80, 80)
    
    return Image.fromarray(pixels)


@pytest.fixture
def sample_text_image() -> Image.Image:
    """Generate image with text for OCR testing."""
    from PIL import ImageDraw, ImageFont
    
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    
    draw.text((20, 30), "The Great Gatsby", fill=(0, 0, 0), font=font)
    
    return img


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_book_data() -> dict:
    """Sample book data for testing."""
    return {
        "title": "The Great Gatsby",
        "authors": ["F. Scott Fitzgerald"],
        "isbn_13": "9780743273565",
        "isbn_10": "0743273567",
        "publisher": "Scribner",
        "published_date": "1925-04-10",
        "page_count": 180,
        "genres": ["Fiction", "Classic", "Literary Fiction"],
        "description": "A story of decadence and excess in the Jazz Age.",
        "cover_url": "https://example.com/cover.jpg",
        "language": "en",
    }


@pytest.fixture
def sample_books_batch() -> list[dict]:
    """Multiple sample books for batch testing."""
    return [
        {
            "title": "1984",
            "authors": ["George Orwell"],
            "isbn_13": "9780451524935",
            "genres": ["Fiction", "Dystopian", "Science Fiction"],
        },
        {
            "title": "To Kill a Mockingbird",
            "authors": ["Harper Lee"],
            "isbn_13": "9780061120084",
            "genres": ["Fiction", "Classic", "Southern Gothic"],
        },
        {
            "title": "Pride and Prejudice",
            "authors": ["Jane Austen"],
            "isbn_13": "9780141439518",
            "genres": ["Fiction", "Classic", "Romance"],
        },
        {
            "title": "The Catcher in the Rye",
            "authors": ["J.D. Salinger"],
            "isbn_13": "9780316769488",
            "genres": ["Fiction", "Classic", "Coming-of-age"],
        },
    ]


@pytest.fixture
def sample_detection_result() -> dict:
    """Sample detection result for testing."""
    return {
        "boxes": [
            {"x1": 50, "y1": 100, "x2": 90, "y2": 400, "confidence": 0.95, "class": "spine"},
            {"x1": 100, "y1": 100, "x2": 145, "y2": 400, "confidence": 0.88, "class": "spine"},
            {"x1": 155, "y1": 100, "x2": 205, "y2": 400, "confidence": 0.92, "class": "spine"},
        ],
        "image_size": {"width": 640, "height": 480},
        "processing_time_ms": 150,
    }


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response for RAG testing."""
    return """Based on your library, I'd recommend "The Road" by Cormac McCarthy.

Given your interest in dystopian fiction like 1984, you might enjoy McCarthy's post-apocalyptic 
narrative style. The sparse prose and survival themes would complement your existing collection.

Other recommendations:
- "Brave New World" by Aldous Huxley
- "The Handmaid's Tale" by Margaret Atwood"""


@pytest.fixture
def mock_embedding() -> np.ndarray:
    """Generate mock embedding vector."""
    np.random.seed(42)
    return np.random.randn(768).astype(np.float32)


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield
    
    # Clean up any test files
    test_dirs = [
        Path("./test_data"),
        Path("./test_uploads"),
        Path("./test_outputs"),
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
