"""
Main FastAPI application for the Face Recognition System.
Configures middleware, CORS, logging, and routes.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
import structlog
from datetime import datetime
import time
import os
from pathlib import Path

from config import settings
from database import init_db, close_db
from routes import api_router, web_router
from models import SystemLog
from database import get_db

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise-grade Face Recognition System with FastAPI",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing information."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    # Add processing time to response headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.middleware("http")
async def error_handling(request: Request, call_next):
    """Global error handling middleware."""
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        # Log the error
        logger.error(
            "Unhandled exception",
            method=request.method,
            url=str(request.url),
            error=str(exc),
            exc_info=True
        )
        
        # Log to database if possible
        try:
            db = next(get_db())
            system_log = SystemLog(
                level="ERROR",
                message=f"Unhandled exception: {str(exc)}",
                module="main",
                function="error_handling",
                traceback=str(exc)
            )
            db.add(system_log)
            db.commit()
        except Exception as db_error:
            logger.error(f"Failed to log error to database: {str(db_error)}")
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else "An unexpected error occurred"
            }
        )


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Face Recognition System", version=settings.app_version)
    
    # Ensure directories exist
    for directory in [settings.upload_folder, settings.logs_folder, "instance"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Log startup
    try:
        db = next(get_db())
        system_log = SystemLog(
            level="INFO",
            message="Application started",
            module="main",
            function="startup_event"
        )
        db.add(system_log)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log startup: {str(e)}")
    
    logger.info("Application startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Face Recognition System")
    
    # Log shutdown
    try:
        db = next(get_db())
        system_log = SystemLog(
            level="INFO",
            message="Application shutdown",
            module="main",
            function="shutdown_event"
        )
        db.add(system_log)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log shutdown: {str(e)}")
    
    # Close database connections
    close_db()
    logger.info("Application shutdown completed")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check database connection
        db = next(get_db())
        db.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version,
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Include routers
app.include_router(api_router)
app.include_router(web_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Enterprise-grade Face Recognition System",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers,
        log_level=settings.log_level.lower()
    ) 