"""
Face recognition service using advanced models and caching.
Implements ArcFace for high-accuracy recognition with optimized performance.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import json
import pickle
from datetime import datetime, timedelta
import hashlib
from deepface import DeepFace
import redis
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings
from preprocessing import ImagePreprocessor, ProcessedFace
from models import User, RecognitionLog, CacheEntry
from database import get_db

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Data class for recognition results."""
    name: str
    confidence: float
    user_id: Optional[int] = None
    processing_time: float = 0.0
    quality_score: float = 0.0
    cache_hit: bool = False


@dataclass
class FaceEmbedding:
    """Data class for face embeddings."""
    embedding: np.ndarray
    user_id: int
    name: str
    created_at: datetime
    quality_score: float


class FaceRecognitionService:
    """Advanced face recognition service with caching and optimization."""
    
    def __init__(self):
        self.model_name = settings.model_name
        self.distance_metric = settings.distance_metric
        self.recognition_threshold = settings.recognition_threshold
        self.preprocessor = ImagePreprocessor()
        
        # Initialize Redis cache
        self.redis_client = None
        self._init_redis()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Load known embeddings cache
        self.known_embeddings: Dict[str, FaceEmbedding] = {}
        self._load_known_embeddings()
        
        logger.info(f"Face recognition service initialized with model: {self.model_name}")
    
    def _init_redis(self) -> None:
        """Initialize Redis connection for caching."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                db=settings.redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis cache not available: {str(e)}")
            self.redis_client = None
    
    def _load_known_embeddings(self) -> None:
        """Load known face embeddings from database."""
        try:
            db = next(get_db())
            users = db.query(User).filter(User.is_active == True).all()
            
            for user in users:
                if user.embedding_path and Path(user.embedding_path).exists():
                    try:
                        embedding = self._load_embedding(user.embedding_path)
                        if embedding is not None:
                            face_embedding = FaceEmbedding(
                                embedding=embedding,
                                user_id=user.id,
                                name=user.name,
                                created_at=user.created_at,
                                quality_score=0.8  # Default quality score
                            )
                            self.known_embeddings[user.name] = face_embedding
                    except Exception as e:
                        logger.error(f"Error loading embedding for user {user.name}: {str(e)}")
            
            logger.info(f"Loaded {len(self.known_embeddings)} known embeddings")
            
        except Exception as e:
            logger.error(f"Error loading known embeddings: {str(e)}")
    
    def _load_embedding(self, embedding_path: str) -> Optional[np.ndarray]:
        """Load embedding from file."""
        try:
            with open(embedding_path, 'rb') as f:
                embedding = pickle.load(f)
            return embedding
        except Exception as e:
            logger.error(f"Error loading embedding from {embedding_path}: {str(e)}")
            return None
    
    def _save_embedding(self, embedding: np.ndarray, embedding_path: str) -> bool:
        """Save embedding to file."""
        try:
            Path(embedding_path).parent.mkdir(parents=True, exist_ok=True)
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding, f)
            return True
        except Exception as e:
            logger.error(f"Error saving embedding to {embedding_path}: {str(e)}")
            return False
    
    def _generate_cache_key(self, image_hash: str) -> str:
        """Generate cache key for image."""
        return f"face_recognition:{image_hash}"
    
    def _get_image_hash(self, image_path: Union[str, Path]) -> str:
        """Generate hash for image file."""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating image hash: {str(e)}")
            return str(image_path)
    
    def register_face(self, image_path: Union[str, Path], name: str, email: Optional[str] = None) -> bool:
        """
        Register a new face in the system.
        
        Args:
            image_path: Path to the face image
            name: Name of the person
            email: Email address (optional)
            
        Returns:
            True if registration successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_faces = self.preprocessor.preprocess_image(image_path)
            if not processed_faces:
                logger.error(f"No faces detected in image: {image_path}")
                return False
            
            # Use the best quality face
            best_face = max(processed_faces, key=lambda x: x.quality_score)
            
            if best_face.quality_score < 0.3:
                logger.warning(f"Face quality too low: {best_face.quality_score}")
                return False
            
            # Generate embedding
            embedding = self._generate_embedding(best_face.face_image)
            if embedding is None:
                logger.error("Failed to generate embedding")
                return False
            
            # Save processed image
            processed_image_path = Path(settings.upload_folder) / f"{name}.jpg"
            self.preprocessor.save_processed_face(best_face, processed_image_path)
            
            # Save embedding
            embedding_path = Path(settings.upload_folder) / f"{name}_embedding.pkl"
            if not self._save_embedding(embedding, str(embedding_path)):
                return False
            
            # Save to database
            db = next(get_db())
            
            # Check if user already exists
            existing_user = db.query(User).filter(User.name == name).first()
            if existing_user:
                # Update existing user
                existing_user.image_path = str(processed_image_path)
                existing_user.embedding_path = str(embedding_path)
                if email:
                    existing_user.email = email
                existing_user.updated_at = datetime.utcnow()
                user = existing_user
            else:
                # Create new user
                user = User(
                    name=name,
                    email=email,
                    image_path=str(processed_image_path),
                    embedding_path=str(embedding_path)
                )
                db.add(user)
            
            db.commit()
            
            # Update in-memory cache
            face_embedding = FaceEmbedding(
                embedding=embedding,
                user_id=user.id,
                name=name,
                created_at=user.created_at,
                quality_score=best_face.quality_score
            )
            self.known_embeddings[name] = face_embedding
            
            processing_time = time.time() - start_time
            logger.info(f"Face registered successfully for {name} in {processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering face: {str(e)}")
            return False
    
    def recognize_face(self, image_path: Union[str, Path], 
                      request_metadata: Optional[Dict] = None) -> Optional[RecognitionResult]:
        """
        Recognize a face in the given image.
        
        Args:
            image_path: Path to the image containing the face
            request_metadata: Additional metadata for logging
            
        Returns:
            RecognitionResult if face recognized, None otherwise
        """
        start_time = time.time()
        
        try:
            # Check cache first
            image_hash = self._get_image_hash(image_path)
            cache_key = self._generate_cache_key(image_hash)
            
            if self.redis_client:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    result_data = json.loads(cached_result)
                    logger.info(f"Cache hit for image: {image_path}")
                    return RecognitionResult(
                        name=result_data['name'],
                        confidence=result_data['confidence'],
                        user_id=result_data.get('user_id'),
                        processing_time=time.time() - start_time,
                        cache_hit=True
                    )
            
            # Preprocess image
            processed_faces = self.preprocessor.preprocess_image(image_path)
            if not processed_faces:
                logger.warning(f"No faces detected in image: {image_path}")
                return None
            
            # Use the best quality face
            best_face = max(processed_faces, key=lambda x: x.quality_score)
            
            if best_face.quality_score < 0.2:
                logger.warning(f"Face quality too low for recognition: {best_face.quality_score}")
                return None
            
            # Generate embedding
            embedding = self._generate_embedding(best_face.face_image)
            if embedding is None:
                logger.error("Failed to generate embedding for recognition")
                return None
            
            # Find best match
            best_match = self._find_best_match(embedding)
            
            processing_time = time.time() - start_time
            
            if best_match and best_match['confidence'] >= self.recognition_threshold:
                result = RecognitionResult(
                    name=best_match['name'],
                    confidence=best_match['confidence'],
                    user_id=best_match.get('user_id'),
                    processing_time=processing_time,
                    quality_score=best_face.quality_score
                )
                
                # Cache result
                if self.redis_client:
                    cache_data = {
                        'name': result.name,
                        'confidence': result.confidence,
                        'user_id': result.user_id,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    self.redis_client.setex(
                        cache_key, 
                        3600,  # Cache for 1 hour
                        json.dumps(cache_data)
                    )
                
                return result
            else:
                logger.info(f"No match found above threshold {self.recognition_threshold}")
                return None
                
        except Exception as e:
            logger.error(f"Error recognizing face: {str(e)}")
            return None
    
    def _generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding for face image."""
        try:
            # Convert to uint8 for DeepFace
            face_image_uint8 = (face_image * 255).astype(np.uint8)
            
            # Generate embedding using DeepFace
            embedding = DeepFace.represent(
                img_path=face_image_uint8,
                model_name=self.model_name,
                detector_backend=settings.detector_backend,
                enforce_detection=False
            )
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            else:
                logger.error("Failed to generate embedding")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def _find_best_match(self, query_embedding: np.ndarray) -> Optional[Dict]:
        """Find best matching face in known embeddings."""
        try:
            best_match = None
            highest_confidence = 0
            
            # Compare with all known embeddings
            for name, face_embedding in self.known_embeddings.items():
                try:
                    # Calculate distance
                    distance = self._calculate_distance(query_embedding, face_embedding.embedding)
                    
                    # Convert distance to confidence
                    confidence = 1 - distance
                    
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_match = {
                            'name': name,
                            'confidence': confidence,
                            'user_id': face_embedding.user_id
                        }
                        
                except Exception as e:
                    logger.warning(f"Error comparing with {name}: {str(e)}")
                    continue
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding best match: {str(e)}")
            return None
    
    def _calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate distance between two embeddings."""
        try:
            if self.distance_metric == 'cosine':
                # Cosine distance
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                cosine_similarity = dot_product / (norm1 * norm2)
                return 1 - cosine_similarity
            elif self.distance_metric == 'euclidean':
                # Euclidean distance
                return np.linalg.norm(embedding1 - embedding2)
            elif self.distance_metric == 'euclidean_l2':
                # L2-normalized Euclidean distance
                embedding1_norm = embedding1 / np.linalg.norm(embedding1)
                embedding2_norm = embedding2 / np.linalg.norm(embedding2)
                return np.linalg.norm(embedding1_norm - embedding2_norm)
            else:
                # Default to cosine distance
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                cosine_similarity = dot_product / (norm1 * norm2)
                return 1 - cosine_similarity
                
        except Exception as e:
            logger.error(f"Error calculating distance: {str(e)}")
            return 1.0  # Maximum distance on error
    
    def log_recognition(self, result: RecognitionResult, 
                       image_path: str, 
                       metadata: Optional[Dict] = None) -> None:
        """Log recognition result to database."""
        try:
            db = next(get_db())
            
            log = RecognitionLog(
                user_id=result.user_id,
                name=result.name,
                snapshot_path=image_path,
                confidence=result.confidence,
                processing_time=result.processing_time,
                success=True,
                **metadata or {}
            )
            
            db.add(log)
            db.commit()
            
        except Exception as e:
            logger.error(f"Error logging recognition: {str(e)}")
    
    def get_recognition_stats(self) -> Dict:
        """Get recognition service statistics."""
        try:
            db = next(get_db())
            
            total_logs = db.query(RecognitionLog).count()
            successful_logs = db.query(RecognitionLog).filter(RecognitionLog.success == True).count()
            total_users = db.query(User).filter(User.is_active == True).count()
            
            # Calculate average confidence
            avg_confidence = db.query(RecognitionLog.confidence).filter(
                RecognitionLog.success == True
            ).scalar()
            
            return {
                'total_recognitions': total_logs,
                'successful_recognitions': successful_logs,
                'success_rate': successful_logs / total_logs if total_logs > 0 else 0,
                'total_users': total_users,
                'known_embeddings': len(self.known_embeddings),
                'average_confidence': float(avg_confidence) if avg_confidence else 0,
                'cache_enabled': self.redis_client is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting recognition stats: {str(e)}")
            return {}
    
    def clear_cache(self) -> bool:
        """Clear all cached recognition results."""
        try:
            if self.redis_client:
                # Clear all face recognition cache keys
                keys = self.redis_client.keys("face_recognition:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached recognition results")
                return True
            return False
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False 