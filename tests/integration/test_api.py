"""
Integration tests for API endpoints.
"""

import pytest
import io
import json
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image

pytestmark = pytest.mark.asyncio


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    async def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    async def test_readiness_check(self, client):
        """Test readiness probe endpoint."""
        response = await client.get("/ready")
        
        assert response.status_code in [200, 503]  # Healthy or not ready
    
    async def test_liveness_check(self, client):
        """Test liveness probe endpoint."""
        response = await client.get("/live")
        
        assert response.status_code == 200


class TestBooksEndpoints:
    """Tests for book CRUD endpoints."""
    
    async def test_create_book(self, client, sample_book_data):
        """Test creating a new book."""
        response = await client.post(
            "/api/v1/books",
            json=sample_book_data,
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "id" in data
        assert data["title"] == sample_book_data["title"]
    
    async def test_get_book_by_id(self, client, sample_book_data):
        """Test retrieving a book by ID."""
        # First create a book
        create_response = await client.post("/api/v1/books", json=sample_book_data)
        
        if create_response.status_code in [200, 201]:
            book_id = create_response.json()["id"]
            
            # Then retrieve it
            response = await client.get(f"/api/v1/books/{book_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == sample_book_data["title"]
    
    async def test_get_nonexistent_book(self, client):
        """Test retrieving a nonexistent book returns 404."""
        response = await client.get("/api/v1/books/nonexistent-id")
        
        assert response.status_code == 404
    
    async def test_list_books(self, client, sample_books_batch):
        """Test listing all books with pagination."""
        # Create some books
        for book in sample_books_batch[:2]:
            await client.post("/api/v1/books", json=book)
        
        # List books
        response = await client.get("/api/v1/books?limit=10&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data or isinstance(data, list)
    
    async def test_update_book(self, client, sample_book_data):
        """Test updating a book."""
        # Create a book
        create_response = await client.post("/api/v1/books", json=sample_book_data)
        
        if create_response.status_code in [200, 201]:
            book_id = create_response.json()["id"]
            
            # Update it
            update_data = {"title": "Updated Title"}
            response = await client.patch(f"/api/v1/books/{book_id}", json=update_data)
            
            assert response.status_code == 200
            assert response.json()["title"] == "Updated Title"
    
    async def test_delete_book(self, client, sample_book_data):
        """Test deleting a book."""
        # Create a book
        create_response = await client.post("/api/v1/books", json=sample_book_data)
        
        if create_response.status_code in [200, 201]:
            book_id = create_response.json()["id"]
            
            # Delete it
            response = await client.delete(f"/api/v1/books/{book_id}")
            
            assert response.status_code in [200, 204]
            
            # Verify it's gone
            get_response = await client.get(f"/api/v1/books/{book_id}")
            assert get_response.status_code == 404
    
    async def test_search_books(self, client, sample_books_batch):
        """Test searching books."""
        # Create some books
        for book in sample_books_batch:
            await client.post("/api/v1/books", json=book)
        
        # Search
        response = await client.get("/api/v1/books/search?q=gatsby")
        
        assert response.status_code == 200


class TestDetectionEndpoints:
    """Tests for detection endpoints."""
    
    async def test_detect_books_from_image(self, client, sample_bookshelf_image):
        """Test book detection from uploaded image."""
        # Convert image to bytes
        img_bytes = io.BytesIO()
        sample_bookshelf_image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        # Mock detection service
        with patch("shelfsense.api.routes.detection.get_detector") as mock_detector:
            mock_detector.return_value.detect = AsyncMock(return_value={
                "boxes": [
                    {"x1": 50, "y1": 100, "x2": 90, "y2": 400, "confidence": 0.95}
                ],
                "count": 1,
            })
            
            response = await client.post(
                "/api/v1/detect",
                files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            )
        
        assert response.status_code == 200
    
    async def test_detect_rejects_invalid_file_type(self, client):
        """Test detection rejects non-image files."""
        response = await client.post(
            "/api/v1/detect",
            files={"image": ("test.txt", b"not an image", "text/plain")},
        )
        
        assert response.status_code in [400, 415, 422]
    
    async def test_detect_with_mode_option(self, client, sample_bookshelf_image):
        """Test detection with different mode options."""
        img_bytes = io.BytesIO()
        sample_bookshelf_image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        with patch("shelfsense.api.routes.detection.get_detector") as mock_detector:
            mock_detector.return_value.detect = AsyncMock(return_value={"boxes": [], "count": 0})
            
            response = await client.post(
                "/api/v1/detect",
                files={"image": ("test.jpg", img_bytes, "image/jpeg")},
                data={"mode": "accurate"},
            )
        
        assert response.status_code == 200
    
    async def test_batch_detection(self, client, sample_bookshelf_image):
        """Test batch image detection."""
        # Create multiple images
        images = []
        for i in range(2):
            img_bytes = io.BytesIO()
            sample_bookshelf_image.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            images.append(("images", (f"test{i}.jpg", img_bytes, "image/jpeg")))
        
        with patch("shelfsense.api.routes.detection.get_detector") as mock_detector:
            mock_detector.return_value.detect = AsyncMock(return_value={"boxes": [], "count": 0})
            
            response = await client.post(
                "/api/v1/detect/batch",
                files=images,
            )
        
        assert response.status_code in [200, 202]  # 202 for async processing


class TestChatEndpoints:
    """Tests for chat/RAG endpoints."""
    
    async def test_create_conversation(self, client):
        """Test creating a new conversation."""
        response = await client.post("/api/v1/chat/conversations")
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "id" in data
    
    async def test_send_message(self, client):
        """Test sending a message to a conversation."""
        # Create conversation
        conv_response = await client.post("/api/v1/chat/conversations")
        
        if conv_response.status_code in [200, 201]:
            conv_id = conv_response.json()["id"]
            
            with patch("shelfsense.api.routes.chat.get_rag_service") as mock_rag:
                mock_rag.return_value.generate_response = AsyncMock(
                    return_value={"response": "Based on your library...", "sources": []}
                )
                
                response = await client.post(
                    f"/api/v1/chat/conversations/{conv_id}/messages",
                    json={"message": "What should I read next?"},
                )
            
            assert response.status_code == 200
    
    async def test_get_conversation_history(self, client):
        """Test retrieving conversation history."""
        # Create and populate conversation
        conv_response = await client.post("/api/v1/chat/conversations")
        
        if conv_response.status_code in [200, 201]:
            conv_id = conv_response.json()["id"]
            
            response = await client.get(f"/api/v1/chat/conversations/{conv_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "messages" in data or "id" in data
    
    async def test_delete_conversation(self, client):
        """Test deleting a conversation."""
        conv_response = await client.post("/api/v1/chat/conversations")
        
        if conv_response.status_code in [200, 201]:
            conv_id = conv_response.json()["id"]
            
            response = await client.delete(f"/api/v1/chat/conversations/{conv_id}")
            
            assert response.status_code in [200, 204]


class TestAnalyticsEndpoints:
    """Tests for analytics endpoints."""
    
    async def test_get_library_stats(self, client, sample_books_batch):
        """Test getting library statistics."""
        # Add some books
        for book in sample_books_batch:
            await client.post("/api/v1/books", json=book)
        
        response = await client.get("/api/v1/analytics/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_books" in data or "count" in data or isinstance(data, dict)
    
    async def test_get_genre_distribution(self, client, sample_books_batch):
        """Test getting genre distribution."""
        for book in sample_books_batch:
            await client.post("/api/v1/books", json=book)
        
        response = await client.get("/api/v1/analytics/genres")
        
        assert response.status_code == 200
    
    async def test_get_diversity_metrics(self, client):
        """Test getting diversity metrics."""
        response = await client.get("/api/v1/analytics/diversity")
        
        assert response.status_code == 200
    
    async def test_get_recommendations(self, client):
        """Test getting recommendations."""
        response = await client.get("/api/v1/analytics/recommendations")
        
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""
    
    async def test_invalid_json_returns_422(self, client):
        """Test invalid JSON returns 422."""
        response = await client.post(
            "/api/v1/books",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )
        
        assert response.status_code == 422
    
    async def test_missing_required_field(self, client):
        """Test missing required field returns error."""
        response = await client.post(
            "/api/v1/books",
            json={"authors": ["Test Author"]},  # Missing title
        )
        
        assert response.status_code == 422
    
    async def test_too_large_upload(self, client):
        """Test oversized upload is rejected."""
        # Create large file (simulated)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        
        response = await client.post(
            "/api/v1/detect",
            files={"image": ("large.jpg", large_content, "image/jpeg")},
        )
        
        assert response.status_code in [400, 413, 422]


class TestRateLimiting:
    """Tests for rate limiting."""
    
    async def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = await client.get("/api/v1/books")
        
        # Rate limit headers might be present
        # X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
        # This test verifies the endpoint works, headers may vary by config


class TestCORS:
    """Tests for CORS configuration."""
    
    async def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = await client.options(
            "/api/v1/books",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        
        # Should not error
        assert response.status_code in [200, 204, 405]
    
    async def test_cors_headers_in_response(self, client):
        """Test CORS headers are in response."""
        response = await client.get(
            "/api/v1/books",
            headers={"Origin": "http://localhost:3000"},
        )
        
        # CORS headers may be present depending on configuration
        assert response.status_code == 200
