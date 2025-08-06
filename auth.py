"""
Authentication and authorization module.
Implements JWT-based authentication with rate limiting and security features.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import redis
import logging
from functools import wraps
import time

from config import settings
from models import User
from database import get_db

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Redis for rate limiting
redis_client = None
try:
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.ping()
except Exception as e:
    logger.warning(f"Redis not available for rate limiting: {str(e)}")


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    user_id: Optional[int] = None


class UserCreate(BaseModel):
    """User creation model."""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str


class AuthService:
    """Authentication service with JWT and security features."""
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        try:
            db = next(get_db())
            user = db.query(User).filter(User.name == username).first()
            
            if not user:
                return None
            
            if not self.verify_password(password, user.password_hash):
                return None
            
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return None
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            
            if username is None:
                return None
            
            token_data = TokenData(username=username, user_id=user_id)
            return token_data
            
        except JWTError as e:
            logger.error(f"JWT verification error: {str(e)}")
            return None
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
        """Get current authenticated user."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            token = credentials.credentials
            token_data = self.verify_token(token)
            
            if token_data is None:
                raise credentials_exception
            
            db = next(get_db())
            user = db.query(User).filter(User.id == token_data.user_id).first()
            
            if user is None:
                raise credentials_exception
            
            return user
            
        except Exception as e:
            logger.error(f"Error getting current user: {str(e)}")
            raise credentials_exception
    
    def create_user(self, user_data: UserCreate) -> Optional[User]:
        """Create a new user account."""
        try:
            db = next(get_db())
            
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.name == user_data.username) | (User.email == user_data.email)
            ).first()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already registered"
                )
            
            # Create new user
            hashed_password = self.get_password_hash(user_data.password)
            
            user = User(
                name=user_data.username,
                email=user_data.email,
                password_hash=hashed_password,
                full_name=user_data.full_name
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"Created new user: {user_data.username}")
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return None
    
    def login_user(self, user_data: UserLogin) -> Optional[Token]:
        """Authenticate user and return access token."""
        try:
            user = self.authenticate_user(user_data.username, user_data.password)
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Create access token
            access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
            access_token = self.create_access_token(
                data={"sub": user.name, "user_id": user.id},
                expires_delta=access_token_expires
            )
            
            logger.info(f"User logged in: {user.name}")
            
            return Token(
                access_token=access_token,
                token_type="bearer",
                expires_in=self.access_token_expire_minutes * 60
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            return None


# Global auth service instance
auth_service = AuthService()


def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """
    Rate limiting decorator.
    
    Args:
        max_requests: Maximum requests allowed in the time window
        window_seconds: Time window in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not redis_client:
                # Skip rate limiting if Redis is not available
                return await func(*args, **kwargs)
            
            # Get client IP (you might need to adjust this based on your setup)
            client_ip = kwargs.get('client_ip', 'unknown')
            key = f"rate_limit:{client_ip}:{func.__name__}"
            
            try:
                # Check current request count
                current_requests = redis_client.get(key)
                
                if current_requests is None:
                    # First request in the window
                    redis_client.setex(key, window_seconds, 1)
                else:
                    current_requests = int(current_requests)
                    
                    if current_requests >= max_requests:
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds."
                        )
                    
                    # Increment request count
                    redis_client.incr(key)
                
                return await func(*args, **kwargs)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Rate limiting error: {str(e)}")
                # Continue without rate limiting on error
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_auth(func):
    """Decorator to require authentication for endpoints."""
    @wraps(func)
    async def wrapper(*args, current_user: User = Depends(auth_service.get_current_user), **kwargs):
        return await func(*args, current_user=current_user, **kwargs)
    return wrapper


def require_admin(func):
    """Decorator to require admin privileges."""
    @wraps(func)
    async def wrapper(*args, current_user: User = Depends(auth_service.get_current_user), **kwargs):
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        return await func(*args, current_user=current_user, **kwargs)
    return wrapper


# Dependency functions for FastAPI
def get_current_user(current_user: User = Depends(auth_service.get_current_user)) -> User:
    """FastAPI dependency for getting current user."""
    return current_user


def get_current_active_user(current_user: User = Depends(auth_service.get_current_user)) -> User:
    """FastAPI dependency for getting current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user 