"""
Database models for Flask-SQLAlchemy.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    """User model for registered individuals."""
    __tablename__ = "users"
    
    id = db.Column(db.Integer, primary_key=True, index=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, index=True, nullable=True)
    image_path = db.Column(db.String(500), nullable=False)
    embedding_path = db.Column(db.String(500), nullable=True)  # Path to stored face embedding
    password_hash = db.Column(db.String(255), nullable=True)  # For authentication
    full_name = db.Column(db.String(200), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    logs = db.relationship("RecognitionLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}', email='{self.email}')>"


class RecognitionLog(db.Model):
    """Log model for face recognition events."""
    __tablename__ = "recognition_logs"
    
    id = db.Column(db.Integer, primary_key=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    snapshot_path = db.Column(db.String(500), nullable=False)
    confidence = db.Column(db.Float, nullable=False, index=True)
    processing_time = db.Column(db.Float, nullable=True)  # Time taken for recognition in seconds
    location = db.Column(db.String(255), nullable=True)  # Location metadata
    device_id = db.Column(db.String(100), nullable=True)  # Device identifier
    ip_address = db.Column(db.String(45), nullable=True)  # Client IP address
    user_agent = db.Column(db.Text, nullable=True)  # Browser/client information
    success = db.Column(db.Boolean, default=True)  # Whether recognition was successful
    error_message = db.Column(db.Text, nullable=True)  # Error details if recognition failed
    
    # Relationships
    user = db.relationship("User", back_populates="logs")
    
    def __repr__(self):
        return f"<RecognitionLog(id={self.id}, name='{self.name}', confidence={self.confidence:.2f})>"


# Alias for backward compatibility
Log = RecognitionLog