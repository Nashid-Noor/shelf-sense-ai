"""
API middleware components.

Provides cross-cutting concerns for the API:
- Error handling
- CORS configuration
- Rate limiting
- Request/response logging
"""

from .error_handler import (
    ShelfSenseException,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ProcessingError,
    ExternalServiceError,
    setup_exception_handlers,
    error_handler_middleware,
    create_error_response,
)

from .cors import (
    CORSConfig,
    OriginValidator,
    get_cors_config,
    setup_cors,
    create_cors_validator,
)

from .rate_limit import (
    RateLimitConfig,
    RateLimitStrategy,
    RateLimitState,
    InMemoryRateLimiter,
    RateLimitMiddleware,
    setup_rate_limiting,
    rate_limit,
)

from .logging import (
    LoggingConfig,
    StructuredLogFormatter,
    RequestLoggingMiddleware,
    setup_logging,
    get_request_id,
    get_logger,
    redact_sensitive_data,
)


__all__ = [
    # Error handling
    "ShelfSenseException",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ProcessingError",
    "ExternalServiceError",
    "setup_exception_handlers",
    "error_handler_middleware",
    "create_error_response",
    # CORS
    "CORSConfig",
    "OriginValidator",
    "get_cors_config",
    "setup_cors",
    "create_cors_validator",
    # Rate limiting
    "RateLimitConfig",
    "RateLimitStrategy",
    "RateLimitState",
    "InMemoryRateLimiter",
    "RateLimitMiddleware",
    "setup_rate_limiting",
    "rate_limit",
    # Logging
    "LoggingConfig",
    "StructuredLogFormatter",
    "RequestLoggingMiddleware",
    "setup_logging",
    "get_request_id",
    "get_logger",
    "redact_sensitive_data",
]
