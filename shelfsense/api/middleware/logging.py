"""
Request/Response logging middleware.

Provides structured logging for all API requests with:
- Request timing and performance metrics
- Request/response body logging (configurable)
- Correlation IDs for tracing
- Log filtering for sensitive data
"""

import time
import uuid
import json
import logging
from typing import Optional, Callable, Set, Any, Dict
from dataclasses import dataclass, field
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message

# Context variable for request ID (accessible throughout request lifecycle)
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

logger = logging.getLogger("shelfsense.api")


@dataclass
class LoggingConfig:
    """Configuration for request logging."""
    
    # Enable request logging
    enabled: bool = True
    
    # Log request bodies
    log_request_body: bool = False
    
    # Log response bodies
    log_response_body: bool = False
    
    # Maximum body size to log (bytes)
    max_body_log_size: int = 10000
    
    # Paths to exclude from logging
    excluded_paths: Set[str] = field(default_factory=lambda: {
        "/health",
        "/metrics",
        "/favicon.ico",
    })
    
    # Headers to exclude from logging (sensitive)
    excluded_headers: Set[str] = field(default_factory=lambda: {
        "authorization",
        "x-api-key",
        "cookie",
        "set-cookie",
    })
    
    # Fields to redact in request/response bodies
    redacted_fields: Set[str] = field(default_factory=lambda: {
        "password",
        "token",
        "secret",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "credit_card",
        "ssn",
    })
    
    # Log level for successful requests
    success_log_level: int = logging.INFO
    
    # Log level for errors (4xx, 5xx)
    error_log_level: int = logging.WARNING
    
    # Slow request threshold (seconds)
    slow_request_threshold: float = 2.0
    
    # Header name for request ID
    request_id_header: str = "X-Request-ID"


class StructuredLogFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs in JSON format for easy parsing by log aggregators.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id
        
        # Add extra fields
        if hasattr(record, "request_data"):
            log_data["request"] = record.request_data
        if hasattr(record, "response_data"):
            log_data["response"] = record.response_data
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def redact_sensitive_data(
    data: Any,
    redacted_fields: Set[str],
    replacement: str = "[REDACTED]",
) -> Any:
    """
    Recursively redact sensitive fields from data structure.
    
    Args:
        data: Data to redact (dict, list, or primitive).
        redacted_fields: Set of field names to redact.
        replacement: Replacement string for redacted values.
    
    Returns:
        Data with sensitive fields redacted.
    """
    if isinstance(data, dict):
        return {
            key: replacement if key.lower() in redacted_fields else redact_sensitive_data(value, redacted_fields, replacement)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [redact_sensitive_data(item, redacted_fields, replacement) for item in data]
    else:
        return data


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request/response logging.
    """
    
    def __init__(self, app: FastAPI, config: Optional[LoggingConfig] = None):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application.
            config: Logging configuration.
        """
        super().__init__(app)
        self.config = config or LoggingConfig()
    
    def _should_log(self, path: str) -> bool:
        """Check if path should be logged."""
        return path not in self.config.excluded_paths
    
    def _filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter sensitive headers from logging."""
        return {
            key: value if key.lower() not in self.config.excluded_headers else "[REDACTED]"
            for key, value in headers.items()
        }
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """Safely get request body for logging."""
        if not self.config.log_request_body:
            return None
        
        try:
            body = await request.body()
            if len(body) > self.config.max_body_log_size:
                return f"[BODY TOO LARGE: {len(body)} bytes]"
            
            # Try to parse as JSON for redaction
            try:
                body_json = json.loads(body)
                body_json = redact_sensitive_data(body_json, self.config.redacted_fields)
                return json.dumps(body_json)
            except json.JSONDecodeError:
                return body.decode("utf-8", errors="replace")
        except Exception:
            return "[FAILED TO READ BODY]"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging."""
        # Generate or extract request ID
        request_id = request.headers.get(
            self.config.request_id_header,
            str(uuid.uuid4())[:8]
        )
        request_id_var.set(request_id)
        
        # Skip logging for excluded paths
        if not self.config.enabled or not self._should_log(request.url.path):
            response = await call_next(request)
            response.headers[self.config.request_id_header] = request_id
            return response
        
        # Capture request details
        start_time = time.time()
        
        request_data = {
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query) if request.url.query else None,
            "headers": self._filter_headers(dict(request.headers)),
            "client_ip": request.client.host if request.client else None,
        }
        
        # Get request body (need to read before processing)
        if self.config.log_request_body:
            body = await self._get_request_body(request)
            if body:
                request_data["body"] = body
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        duration_ms = round(duration * 1000, 2)
        
        # Capture response details
        response_data = {
            "status_code": response.status_code,
            "headers": self._filter_headers(dict(response.headers)),
        }
        
        # Add request ID to response
        response.headers[self.config.request_id_header] = request_id
        
        # Determine log level
        if response.status_code >= 500:
            log_level = logging.ERROR
        elif response.status_code >= 400:
            log_level = self.config.error_log_level
        elif duration > self.config.slow_request_threshold:
            log_level = logging.WARNING
        else:
            log_level = self.config.success_log_level
        
        # Create log message
        message = f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)"
        
        if duration > self.config.slow_request_threshold:
            message = f"[SLOW] {message}"
        
        # Log with structured data
        extra = {
            "request_data": request_data,
            "response_data": response_data,
            "duration_ms": duration_ms,
        }
        
        logger.log(log_level, message, extra=extra)
        
        return response


def setup_logging(
    app: FastAPI,
    config: Optional[LoggingConfig] = None,
    structured: bool = True,
) -> None:
    """
    Configure logging middleware and formatters.
    
    Args:
        app: FastAPI application instance.
        config: Logging configuration.
        structured: Use JSON structured logging format.
    """
    if config is None:
        config = LoggingConfig()
    
    # Configure root logger for structured output
    if structured:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredLogFormatter())
        
        # Configure shelfsense logger
        shelfsense_logger = logging.getLogger("shelfsense")
        shelfsense_logger.addHandler(handler)
        shelfsense_logger.setLevel(logging.INFO)
    
    # Add middleware
    app.add_middleware(RequestLoggingMiddleware, config=config)


class RequestLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes request ID.
    
    Usage:
        logger = RequestLoggerAdapter(logging.getLogger(__name__))
        logger.info("Processing book", extra={"book_id": "123"})
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add request ID to log record."""
        request_id = request_id_var.get()
        
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        
        kwargs["extra"]["request_id"] = request_id
        
        return msg, kwargs


# Convenience function to get a request-aware logger
def get_logger(name: str) -> RequestLoggerAdapter:
    """
    Get a logger that automatically includes request context.
    
    Args:
        name: Logger name (typically __name__).
    
    Returns:
        Logger adapter with request context.
    """
    return RequestLoggerAdapter(logging.getLogger(name), {})
