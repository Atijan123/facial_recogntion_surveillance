#!/usr/bin/env python3
"""
Startup script for the Face Recognition System.
Handles initialization and launches the FastAPI application.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main startup function."""
    print("🚀 Starting Face Recognition System...")
    
    # Ensure required directories exist
    directories = [
        "static/faces",
        "static/logs", 
        "instance"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Initialize database
    try:
        from database import init_db
        init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
    
    # Start the application
    try:
        import uvicorn
        from config import settings
        
        print(f"🌐 Starting server on {settings.host}:{settings.port}")
        print(f"📖 API Documentation: http://{settings.host}:{settings.port}/docs")
        print(f"🏥 Health Check: http://{settings.host}:{settings.port}/health")
        print("=" * 50)
        
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level=settings.log_level.lower()
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 