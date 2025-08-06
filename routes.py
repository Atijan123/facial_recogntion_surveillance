"""
FastAPI routes for the Face Recognition System.
Implements RESTful API endpoints with proper validation and error handling.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
import cv2
import numpy as np
from datetime import datetime
import io
import logging
from pathlib import Path

from config import settings
from database import get_db
from models import User, RecognitionLog
from recognition import FaceRecognitionService, RecognitionResult
from auth import auth_service, rate_limit, require_auth, get_current_user
from preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)

# Initialize services
recognition_service = FaceRecognitionService()
preprocessor = ImagePreprocessor()

# Create routers
api_router = APIRouter(prefix="/api/v1", tags=["API"])
web_router = APIRouter(tags=["Web"])

# Templates
templates = Jinja2Templates(directory="templates")


# ============================================================================
# API Routes
# ============================================================================

@api_router.post("/auth/register", response_model=dict)
@rate_limit(max_requests=10, window_seconds=3600)
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    full_name: Optional[str] = Form(None)
):
    """Register a new user account."""
    try:
        from auth import UserCreate
        user_data = UserCreate(
            username=username,
            email=email,
            password=password,
            full_name=full_name
        )
        
        user = auth_service.create_user(user_data)
        if user:
            return {
                "success": True,
                "message": "User registered successfully",
                "user_id": user.id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in user registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@api_router.post("/auth/login", response_model=dict)
@rate_limit(max_requests=20, window_seconds=3600)
async def login_user(
    username: str = Form(...),
    password: str = Form(...)
):
    """Authenticate user and return access token."""
    try:
        from auth import UserLogin
        user_data = UserLogin(username=username, password=password)
        
        token = auth_service.login_user(user_data)
        if token:
            return {
                "success": True,
                "access_token": token.access_token,
                "token_type": token.token_type,
                "expires_in": token.expires_in
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in user login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@api_router.post("/faces/register", response_model=dict)
@require_auth
@rate_limit(max_requests=50, window_seconds=3600)
async def register_face(
    name: str = Form(...),
    email: Optional[str] = Form(None),
    face_image: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Register a new face in the system."""
    try:
        # Validate file
        if not face_image.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file uploaded"
            )
        
        # Check file extension
        file_ext = Path(face_image.filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Save uploaded file temporarily
        temp_path = Path(settings.upload_folder) / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await face_image.read()
            buffer.write(content)
        
        # Register face
        success = recognition_service.register_face(temp_path, name, email)
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        
        if success:
            return {
                "success": True,
                "message": f"Face registered successfully for {name}",
                "name": name
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to register face. Please ensure the image contains a clear face."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering face: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@api_router.post("/faces/recognize", response_model=dict)
@rate_limit(max_requests=100, window_seconds=3600)
async def recognize_face(
    face_image: UploadFile = File(...),
    request: Request = None
):
    """Recognize a face in the uploaded image."""
    try:
        # Validate file
        if not face_image.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file uploaded"
            )
        
        # Check file extension
        file_ext = Path(face_image.filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Save uploaded file temporarily
        temp_path = Path(settings.logs_folder) / f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await face_image.read()
            buffer.write(content)
        
        # Prepare metadata for logging
        metadata = {
            "ip_address": request.client.host if request else None,
            "user_agent": request.headers.get("user-agent") if request else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Recognize face
        result = recognition_service.recognize_face(temp_path, metadata)
        
        if result:
            # Log recognition
            recognition_service.log_recognition(result, str(temp_path), metadata)
            
            return {
                "success": True,
                "name": result.name,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "quality_score": result.quality_score,
                "cache_hit": result.cache_hit,
                "image_path": str(temp_path)
            }
        else:
            # Log failed recognition
            if request:
                metadata["success"] = False
                metadata["error_message"] = "No face detected or recognition failed"
                recognition_service.log_recognition(
                    RecognitionResult(name="Unknown", confidence=0.0),
                    str(temp_path),
                    metadata
                )
            
            return {
                "success": False,
                "message": "No face detected or recognition failed",
                "image_path": str(temp_path)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recognizing face: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@api_router.get("/faces/list", response_model=dict)
@require_auth
async def list_registered_faces(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of all registered faces."""
    try:
        users = db.query(User).filter(User.is_active == True).all()
        
        faces = []
        for user in users:
            faces.append({
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "created_at": user.created_at.isoformat(),
                "image_path": user.image_path
            })
        
        return {
            "success": True,
            "faces": faces,
            "total_count": len(faces)
        }
        
    except Exception as e:
        logger.error(f"Error listing faces: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@api_router.get("/logs", response_model=dict)
@require_auth
async def get_recognition_logs(
    limit: int = 100,
    offset: int = 0,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get recognition logs with pagination and search."""
    try:
        query = db.query(RecognitionLog)
        
        if search:
            query = query.filter(RecognitionLog.name.ilike(f"%{search}%"))
        
        total_count = query.count()
        logs = query.order_by(RecognitionLog.timestamp.desc()).offset(offset).limit(limit).all()
        
        log_data = []
        for log in logs:
            log_data.append({
                "id": log.id,
                "name": log.name,
                "confidence": log.confidence,
                "timestamp": log.timestamp.isoformat(),
                "processing_time": log.processing_time,
                "success": log.success,
                "snapshot_path": log.snapshot_path
            })
        
        return {
            "success": True,
            "logs": log_data,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@api_router.get("/stats", response_model=dict)
@require_auth
async def get_system_stats(
    current_user: User = Depends(get_current_user)
):
    """Get system statistics."""
    try:
        stats = recognition_service.get_recognition_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@api_router.delete("/cache/clear", response_model=dict)
@require_auth
async def clear_cache(
    current_user: User = Depends(get_current_user)
):
    """Clear recognition cache."""
    try:
        success = recognition_service.clear_cache()
        
        return {
            "success": success,
            "message": "Cache cleared successfully" if success else "Failed to clear cache"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ============================================================================
# Web Routes (HTML Templates)
# ============================================================================

@web_router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@web_router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Face registration page."""
    return templates.TemplateResponse("register.html", {"request": request})


@web_router.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request):
    """Manual face scanning page."""
    return templates.TemplateResponse("scan.html", {"request": request})


@web_router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    """Recognition logs page."""
    return templates.TemplateResponse("logs.html", {"request": request})


@web_router.get("/video_feed")
async def video_feed():
    """Live video feed for face recognition."""
    def generate_frames():
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame for face recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                processed_faces = preprocessor.preprocess_image(rgb_frame)
                
                # Draw recognition results on frame
                for processed_face in processed_faces:
                    face_region = processed_face.face_region
                    x, y, w, h = face_region.x, face_region.y, face_region.w, face_region.h
                    
                    # Recognize face
                    temp_path = Path(settings.logs_folder) / f"temp_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(temp_path), frame)
                    
                    result = recognition_service.recognize_face(temp_path)
                    
                    if result and result.confidence >= settings.recognition_threshold:
                        # Draw green rectangle for recognized face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"{result.name} ({result.confidence:.2f})"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # Draw red rectangle for unrecognized face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    ) 