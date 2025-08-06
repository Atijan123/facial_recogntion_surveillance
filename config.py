"""
Configuration management for the Face Recognition System.
Uses Pydantic Settings for type-safe environment variable handling.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Face Recognition System"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Database
    database_url: str = "sqlite:///./instance/database.db"
    
    # Redis (for caching)
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Face Recognition
    recognition_threshold: float = 0.4
    detection_confidence: float = 0.9
    model_name: str = "ArcFace"  # ArcFace, FaceNet, VGG-Face
    distance_metric: str = "cosine"  # cosine, euclidean, euclidean_l2
    detector_backend: str = "retinaface"  # retinaface, mtcnn, opencv
    
    # File Storage
    upload_folder: str = "static/faces"
    logs_folder: str = "static/logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: set = {".jpg", ".jpeg", ".png"}
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings 