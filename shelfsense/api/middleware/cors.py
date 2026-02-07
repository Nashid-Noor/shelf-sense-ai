"""
CORS Configuration

Configures Cross-Origin Resource Sharing settings.
"""

from typing import List, Optional
from dataclasses import dataclass, field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import re


@dataclass
class CORSConfig:
    """CORS configuration settings."""
    
    # Allowed origins (can be specific URLs or patterns)
    allowed_origins: List[str] = field(default_factory=list)
    
    # Allow credentials (cookies, authorization headers)
    allow_credentials: bool = True
    
    # Allowed HTTP methods
    allowed_methods: List[str] = field(default_factory=lambda: [
        "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"
    ])
    
    # Allowed headers
    allowed_headers: List[str] = field(default_factory=lambda: [
        "Accept",
        "Accept-Language",
        "Content-Type",
        "Content-Language",
        "Authorization",
        "X-Request-ID",
        "X-Conversation-ID",
        "X-API-Key",
    ])
    
    # Headers to expose to the browser
    expose_headers: List[str] = field(default_factory=lambda: [
        "X-Request-ID",
        "X-Conversation-ID",
        "X-Rate-Limit-Remaining",
        "X-Rate-Limit-Reset",
        "Content-Disposition",
    ])
    
    # Max age for preflight cache (in seconds)
    max_age: int = 3600
    
    # Allow all origins (development only!)
    allow_all_origins: bool = False


# Environment-specific configurations
CORS_CONFIGS = {
    "development": CORSConfig(
        allowed_origins=[
            "http://localhost:3000",      # React dev server
            "http://localhost:5173",      # Vite dev server
            "http://localhost:8080",      # Vue dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ],
        allow_all_origins=True,  # Allow all in development
        allow_credentials=True,
    ),
    "staging": CORSConfig(
        allowed_origins=[
            "https://staging.shelfsense.example.com",
            "https://staging-app.shelfsense.example.com",
        ],
        allow_credentials=True,
    ),
    "production": CORSConfig(
        allowed_origins=[
            "https://shelfsense.example.com",
            "https://app.shelfsense.example.com",
            "https://www.shelfsense.example.com",
        ],
        allow_credentials=True,
        max_age=7200,  # Longer cache in production
    ),
}


def get_cors_config(environment: Optional[str] = None) -> CORSConfig:
    """Get CORS configuration for the environment."""
    if environment is None:
        environment = os.getenv("SHELFSENSE_ENV", "development")
    
    config = CORS_CONFIGS.get(environment, CORS_CONFIGS["development"])
    
    # Allow additional origins from environment variable
    extra_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if extra_origins:
        config.allowed_origins.extend(
            origin.strip() for origin in extra_origins.split(",") if origin.strip()
        )
    
    return config


class OriginValidator:
    """Validates origins against allowed patterns."""
    
    def __init__(self, allowed_origins: List[str], allow_all: bool = False):
        """
        Initialize validator.
        
        Args:
            allowed_origins: List of allowed origin URLs or patterns.
            allow_all: If True, allows all origins.
        """
        self.allow_all = allow_all
        self.exact_origins = set()
        self.patterns = []
        
        for origin in allowed_origins:
            if "*" in origin:
                # Convert wildcard pattern to regex
                pattern = origin.replace(".", r"\.").replace("*", r".*")
                self.patterns.append(re.compile(f"^{pattern}$"))
            else:
                self.exact_origins.add(origin.lower())
    
    def is_allowed(self, origin: str) -> bool:
        """
        Check if origin is allowed.
        
        Args:
            origin: Origin URL to check.
        
        Returns:
            True if origin is allowed.
        """
        if self.allow_all:
            return True
        
        origin_lower = origin.lower()
        
        # Check exact match
        if origin_lower in self.exact_origins:
            return True
        
        # Check patterns
        for pattern in self.patterns:
            if pattern.match(origin_lower):
                return True
        
        return False


def setup_cors(app: FastAPI, config: Optional[CORSConfig] = None) -> None:
    """
    Configure CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance.
        config: CORS configuration. If None, loads from environment.
    """
    if config is None:
        config = get_cors_config()
    
    # Determine origins to allow
    if config.allow_all_origins:
        allow_origins = ["*"]
    else:
        allow_origins = config.allowed_origins
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=config.allow_credentials if not config.allow_all_origins else False,
        allow_methods=config.allowed_methods,
        allow_headers=config.allowed_headers,
        expose_headers=config.expose_headers,
        max_age=config.max_age,
    )


def create_cors_validator(config: Optional[CORSConfig] = None) -> OriginValidator:
    """
    Create an origin validator for manual CORS checks.
    
    Useful for WebSocket connections which don't use CORSMiddleware.
    
    Args:
        config: CORS configuration. If None, loads from environment.
    
    Returns:
        OriginValidator instance.
    """
    if config is None:
        config = get_cors_config()
    
    return OriginValidator(
        allowed_origins=config.allowed_origins,
        allow_all=config.allow_all_origins,
    )


# Example usage and testing
if __name__ == "__main__":
    # Test origin validator
    validator = OriginValidator(
        allowed_origins=[
            "https://shelfsense.example.com",
            "https://*.shelfsense.example.com",
        ],
        allow_all=False,
    )
    
    test_origins = [
        "https://shelfsense.example.com",
        "https://app.shelfsense.example.com",
        "https://staging.shelfsense.example.com",
        "https://malicious.com",
        "http://shelfsense.example.com",  # Wrong scheme
    ]
    
    for origin in test_origins:
        allowed = validator.is_allowed(origin)
        print(f"{origin}: {'✓' if allowed else '✗'}")
