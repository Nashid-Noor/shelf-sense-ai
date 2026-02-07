"""
Rate limiting middleware for API protection.

Implements multiple rate limiting strategies:
- Token bucket for burst handling
- Sliding window for smooth rate limiting
- Per-endpoint and per-user limits

Supports Redis backend for distributed deployments or in-memory for single instance.
"""

import time
import asyncio
from typing import Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import hashlib
import logging

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting algorithm strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    # Default requests per minute
    requests_per_minute: int = 60
    
    # Burst allowance (for token bucket)
    burst_size: int = 10
    
    # Strategy to use
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    
    # Enable rate limiting
    enabled: bool = True
    
    # Paths to exclude from rate limiting
    excluded_paths: list = field(default_factory=lambda: [
        "/health",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    ])
    
    # Per-endpoint limits (path pattern -> requests per minute)
    endpoint_limits: Dict[str, int] = field(default_factory=lambda: {
        "/api/v1/detect": 20,           # Image processing is expensive
        "/api/v1/detect/batch": 5,      # Batch processing even more so
        "/api/v1/chat": 30,             # LLM calls are rate-limited
        "/api/v1/chat/stream": 30,
        "/api/v1/books/bulk": 10,       # Bulk operations
    })
    
    # Header to check for API key (for per-key limits)
    api_key_header: str = "X-API-Key"
    
    # Use IP-based limiting as fallback
    use_ip_fallback: bool = True
    
    # Trusted proxy headers for real IP
    trusted_proxy_headers: list = field(default_factory=lambda: [
        "X-Forwarded-For",
        "X-Real-IP",
    ])


@dataclass
class RateLimitState:
    """State for a single rate limit bucket."""
    tokens: float
    last_update: float
    request_count: int = 0
    window_start: float = 0.0


class InMemoryRateLimiter:
    """
    In-memory rate limiter implementation.
    
    Suitable for single-instance deployments.
    For distributed deployments, use Redis-backed implementation.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiting configuration.
        """
        self.config = config
        self._buckets: Dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = 300  # Clean up every 5 minutes
        self._last_cleanup = time.time()
    
    def _get_bucket_key(self, identifier: str, endpoint: Optional[str] = None) -> str:
        """Generate bucket key for identifier and optional endpoint."""
        if endpoint:
            return f"{identifier}:{endpoint}"
        return identifier
    
    def _get_limit_for_endpoint(self, path: str) -> int:
        """Get rate limit for specific endpoint."""
        for pattern, limit in self.config.endpoint_limits.items():
            if path.startswith(pattern):
                return limit
        return self.config.requests_per_minute
    
    async def _cleanup_old_buckets(self) -> None:
        """Remove expired bucket entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        async with self._lock:
            # Remove buckets not updated in the last hour
            expired_keys = [
                key for key, state in self._buckets.items()
                if now - state.last_update > 3600
            ]
            for key in expired_keys:
                del self._buckets[key]
            
            self._last_cleanup = now
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit buckets")
    
    async def check_token_bucket(
        self,
        identifier: str,
        endpoint: Optional[str] = None,
    ) -> Tuple[bool, int, float]:
        """
        Check rate limit using token bucket algorithm.
        
        Args:
            identifier: Client identifier (API key or IP).
            endpoint: Optional endpoint path for per-endpoint limits.
        
        Returns:
            Tuple of (allowed, remaining_tokens, reset_time).
        """
        await self._cleanup_old_buckets()
        
        bucket_key = self._get_bucket_key(identifier, endpoint)
        limit = self._get_limit_for_endpoint(endpoint) if endpoint else self.config.requests_per_minute
        
        # Tokens per second
        refill_rate = limit / 60.0
        max_tokens = min(self.config.burst_size, limit)
        
        async with self._lock:
            now = time.time()
            
            if bucket_key not in self._buckets:
                self._buckets[bucket_key] = RateLimitState(
                    tokens=max_tokens,
                    last_update=now,
                )
            
            state = self._buckets[bucket_key]
            
            # Refill tokens based on elapsed time
            elapsed = now - state.last_update
            state.tokens = min(max_tokens, state.tokens + elapsed * refill_rate)
            state.last_update = now
            
            if state.tokens >= 1:
                state.tokens -= 1
                remaining = int(state.tokens)
                # Time until next token
                reset_time = (1 - (state.tokens % 1)) / refill_rate if state.tokens < max_tokens else 0
                return True, remaining, reset_time
            else:
                # Calculate time until next token available
                reset_time = (1 - state.tokens) / refill_rate
                return False, 0, reset_time
    
    async def check_sliding_window(
        self,
        identifier: str,
        endpoint: Optional[str] = None,
    ) -> Tuple[bool, int, float]:
        """
        Check rate limit using sliding window algorithm.
        
        Args:
            identifier: Client identifier (API key or IP).
            endpoint: Optional endpoint path for per-endpoint limits.
        
        Returns:
            Tuple of (allowed, remaining_requests, reset_time).
        """
        await self._cleanup_old_buckets()
        
        bucket_key = self._get_bucket_key(identifier, endpoint)
        limit = self._get_limit_for_endpoint(endpoint) if endpoint else self.config.requests_per_minute
        window_size = 60.0  # 1 minute window
        
        async with self._lock:
            now = time.time()
            
            if bucket_key not in self._buckets:
                self._buckets[bucket_key] = RateLimitState(
                    tokens=0,
                    last_update=now,
                    request_count=0,
                    window_start=now,
                )
            
            state = self._buckets[bucket_key]
            
            # Check if we're in a new window
            elapsed = now - state.window_start
            if elapsed >= window_size:
                # Reset window
                windows_passed = int(elapsed / window_size)
                state.window_start += windows_passed * window_size
                state.request_count = 0
            
            # Calculate weighted count (sliding window approximation)
            window_progress = (now - state.window_start) / window_size
            effective_count = state.request_count * (1 - window_progress)
            
            if effective_count < limit:
                state.request_count += 1
                state.last_update = now
                remaining = int(limit - effective_count - 1)
                reset_time = window_size - (now - state.window_start)
                return True, max(0, remaining), reset_time
            else:
                reset_time = window_size - (now - state.window_start)
                return False, 0, reset_time
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: Optional[str] = None,
    ) -> Tuple[bool, int, float]:
        """
        Check rate limit using configured strategy.
        
        Args:
            identifier: Client identifier (API key or IP).
            endpoint: Optional endpoint path for per-endpoint limits.
        
        Returns:
            Tuple of (allowed, remaining, reset_time).
        """
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self.check_token_bucket(identifier, endpoint)
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self.check_sliding_window(identifier, endpoint)
        else:
            # Fixed window (simpler version of sliding window)
            return await self.check_sliding_window(identifier, endpoint)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting requests.
    """
    
    def __init__(
        self,
        app: FastAPI,
        config: Optional[RateLimitConfig] = None,
        limiter: Optional[InMemoryRateLimiter] = None,
    ):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application.
            config: Rate limiting configuration.
            limiter: Rate limiter instance (creates default if None).
        """
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.limiter = limiter or InMemoryRateLimiter(self.config)
    
    def _get_client_identifier(self, request: Request) -> str:
        """
        Extract client identifier from request.
        
        Checks for API key first, then falls back to IP address.
        """
        # Check for API key
        api_key = request.headers.get(self.config.api_key_header)
        if api_key:
            # Hash API key for privacy
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        if not self.config.use_ip_fallback:
            return "anonymous"
        
        # Try to get real IP from proxy headers
        for header in self.config.trusted_proxy_headers:
            forwarded = request.headers.get(header)
            if forwarded:
                # Take first IP in chain (client IP)
                ip = forwarded.split(",")[0].strip()
                return f"ip:{ip}"
        
        # Fall back to direct client IP
        client = request.client
        if client:
            return f"ip:{client.host}"
        
        return "unknown"
    
    def _is_excluded(self, path: str) -> bool:
        """Check if path is excluded from rate limiting."""
        return any(path.startswith(excluded) for excluded in self.config.excluded_paths)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip if disabled or excluded path
        if not self.config.enabled or self._is_excluded(request.url.path):
            return await call_next(request)
        
        identifier = self._get_client_identifier(request)
        endpoint = request.url.path
        
        allowed, remaining, reset_time = await self.limiter.check_rate_limit(
            identifier, endpoint
        )
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {identifier} on {endpoint}",
                extra={
                    "identifier": identifier,
                    "endpoint": endpoint,
                    "reset_time": reset_time,
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": int(reset_time) + 1,
                    }
                },
                headers={
                    "Retry-After": str(int(reset_time) + 1),
                    "X-Rate-Limit-Remaining": "0",
                    "X-Rate-Limit-Reset": str(int(time.time() + reset_time)),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-Rate-Limit-Remaining"] = str(remaining)
        response.headers["X-Rate-Limit-Reset"] = str(int(time.time() + reset_time))
        
        return response


def setup_rate_limiting(
    app: FastAPI,
    config: Optional[RateLimitConfig] = None,
) -> InMemoryRateLimiter:
    """
    Configure rate limiting middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance.
        config: Rate limiting configuration.
    
    Returns:
        The rate limiter instance for potential external use.
    """
    if config is None:
        config = RateLimitConfig()
    
    limiter = InMemoryRateLimiter(config)
    app.add_middleware(RateLimitMiddleware, config=config, limiter=limiter)
    
    return limiter


# Decorator for custom per-route rate limits
def rate_limit(
    requests_per_minute: int,
    key_func: Optional[Callable[[Request], str]] = None,
):
    """
    Decorator for custom rate limiting on specific routes.
    
    Usage:
        @app.get("/expensive-operation")
        @rate_limit(requests_per_minute=5)
        async def expensive_operation():
            ...
    
    Args:
        requests_per_minute: Maximum requests per minute.
        key_func: Optional function to extract rate limit key from request.
    """
    _limiter = InMemoryRateLimiter(RateLimitConfig(requests_per_minute=requests_per_minute))
    
    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args: Any, **kwargs: Any) -> Any:
            # Get identifier
            if key_func:
                identifier = key_func(request)
            else:
                client = request.client
                identifier = client.host if client else "unknown"
            
            allowed, remaining, reset_time = await _limiter.check_rate_limit(identifier)
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": int(reset_time) + 1,
                    },
                    headers={
                        "Retry-After": str(int(reset_time) + 1),
                    },
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator
