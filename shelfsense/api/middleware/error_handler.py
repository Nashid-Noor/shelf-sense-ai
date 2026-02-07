"""
Error Handling Middleware for ShelfSense AI

Centralized error handling:
- Structured error responses
- Logging of errors
- Exception translation
"""

import traceback
from datetime import datetime
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError


class ShelfSenseException(Exception):
    """Base exception for ShelfSense errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        detail: str = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class NotFoundError(ShelfSenseException):
    """Resource not found."""
    
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} not found",
            code="NOT_FOUND",
            status_code=404,
            detail=f"No {resource} with identifier '{identifier}' exists",
        )


class ValidationError(ShelfSenseException):
    """Input validation failed."""
    
    def __init__(self, message: str, detail: str = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
            detail=detail,
        )


class RateLimitError(ShelfSenseException):
    """Rate limit exceeded."""
    
    def __init__(self, limit: int, window: str):
        super().__init__(
            message="Rate limit exceeded",
            code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            detail=f"Maximum {limit} requests per {window}",
        )


class ProcessingError(ShelfSenseException):
    """Error during processing."""
    
    def __init__(self, message: str, detail: str = None):
        super().__init__(
            message=message,
            code="PROCESSING_ERROR",
            status_code=500,
            detail=detail,
        )


class ExternalServiceError(ShelfSenseException):
    """External service failure."""
    
    def __init__(self, service: str, detail: str = None):
        super().__init__(
            message=f"{service} service unavailable",
            code="EXTERNAL_SERVICE_ERROR",
            status_code=503,
            detail=detail,
        )


def create_error_response(
    error: str,
    code: str,
    status_code: int,
    detail: str = None,
) -> JSONResponse:
    """Create standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "code": code,
            "detail": detail,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def error_handler_middleware(
    request: Request,
    call_next: Callable,
) -> Response:
    """
    Global error handling middleware.
    
    Catches all exceptions and returns structured error responses.
    """
    try:
        return await call_next(request)
    
    except ShelfSenseException as e:
        logger.warning(
            f"ShelfSense error: {e.code} - {e.message}",
            extra={"path": request.url.path, "detail": e.detail},
        )
        return create_error_response(
            error=e.message,
            code=e.code,
            status_code=e.status_code,
            detail=e.detail,
        )
    
    except ValidationError as e:
        logger.warning(
            f"Validation error: {str(e)}",
            extra={"path": request.url.path},
        )
        return create_error_response(
            error="Validation Error",
            code="VALIDATION_ERROR",
            status_code=400,
            detail=str(e),
        )
    
    except Exception as e:
        # Log full traceback for unexpected errors
        logger.error(
            f"Unexpected error: {type(e).__name__}: {str(e)}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "traceback": traceback.format_exc(),
            },
        )
        
        # Don't expose internal error details in production
        return create_error_response(
            error="Internal Server Error",
            code="INTERNAL_ERROR",
            status_code=500,
            detail="An unexpected error occurred",
        )


def setup_exception_handlers(app):
    """Register exception handlers with FastAPI app."""
    
    @app.exception_handler(ShelfSenseException)
    async def shelfsense_exception_handler(request: Request, exc: ShelfSenseException):
        logger.warning(f"ShelfSense error: {exc.code} - {exc.message}")
        return create_error_response(
            error=exc.message,
            code=exc.code,
            status_code=exc.status_code,
            detail=exc.detail,
        )
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        logger.warning(f"Validation error: {str(exc)}")
        return create_error_response(
            error="Validation Error",
            code="VALIDATION_ERROR",
            status_code=400,
            detail=str(exc),
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
            extra={"traceback": traceback.format_exc()},
        )
        return create_error_response(
            error="Internal Server Error",
            code="INTERNAL_ERROR",
            status_code=500,
            detail="An unexpected error occurred",
        )
