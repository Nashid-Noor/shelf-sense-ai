"""
Database models for ShelfSense AI.
"""

from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, CheckConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True)  # UUID
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Ideally we'd separate BookModel here too, but for minimal changes we'll import it or keep it separate
    # For now, this is just User. book_repository.py handles BookModel.
    # To avoid circular imports, we define Base here and import it in BookRepository.
