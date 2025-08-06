"""
Database configuration and session management.
Uses SQLAlchemy 2.0 with async support for better performance.
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import os

from config import settings

# Create database engine
if settings.database_url.startswith("sqlite"):
    # SQLite configuration for development
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.debug
    )
else:
    # PostgreSQL/MySQL configuration for production
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=settings.debug
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    Ensures proper session cleanup after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    # Ensure instance directory exists
    os.makedirs("instance", exist_ok=True)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)


def close_db() -> None:
    """Close database connections."""
    engine.dispose() 