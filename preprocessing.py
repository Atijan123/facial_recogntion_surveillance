"""
Image preprocessing module for face recognition.
Handles face detection, alignment, normalization, and quality enhancement.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from deepface import DeepFace
from PIL import Image, ImageEnhance, ImageFilter
import os

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class FaceRegion:
    """Data class for face region information."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: Optional[Dict] = None


@dataclass
class ProcessedFace:
    """Data class for processed face data."""
    face_image: np.ndarray
    face_region: FaceRegion
    quality_score: float
    is_aligned: bool


class ImagePreprocessor:
    """Advanced image preprocessing for face recognition."""
    
    def __init__(self):
        self.detector_backend = settings.detector_backend
        self.min_face_size = 80
        self.target_size = (224, 224)  # Standard size for most face recognition models
        self.quality_threshold = 0.5
        
    def preprocess_image(self, image_path: Union[str, Path]) -> List[ProcessedFace]:
        """
        Main preprocessing pipeline for face images.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of processed faces with metadata
        """
        try:
            # Load and validate image
            image = self._load_image(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []
            
            # Detect faces
            faces = self._detect_faces(image)
            if not faces:
                logger.warning(f"No faces detected in image: {image_path}")
                return []
            
            processed_faces = []
            for face_data in faces:
                try:
                    # Extract face region
                    face_region = self._extract_face_region(face_data)
                    
                    # Extract and process face
                    face_image = face_data['face']
                    
                    # Enhance face quality
                    enhanced_face = self._enhance_face_quality(face_image)
                    
                    # Align face
                    aligned_face = self._align_face(enhanced_face, face_data.get('landmarks'))
                    
                    # Normalize face
                    normalized_face = self._normalize_face(aligned_face)
                    
                    # Calculate quality score
                    quality_score = self._calculate_quality_score(normalized_face)
                    
                    # Create processed face object
                    processed_face = ProcessedFace(
                        face_image=normalized_face,
                        face_region=face_region,
                        quality_score=quality_score,
                        is_aligned=face_data.get('landmarks') is not None
                    )
                    
                    processed_faces.append(processed_face)
                    
                except Exception as e:
                    logger.error(f"Error processing face: {str(e)}")
                    continue
            
            return processed_faces
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            return []
    
    def _load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load and validate image."""
        try:
            if isinstance(image_path, str):
                image_path = Path(image_path)
            
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Load with OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image with OpenCV: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in image using DeepFace."""
        try:
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=False  # We'll handle alignment separately
            )
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def _extract_face_region(self, face_data: Dict) -> FaceRegion:
        """Extract face region information."""
        facial_area = face_data.get('facial_area', {})
        
        return FaceRegion(
            x=facial_area.get('x', 0),
            y=facial_area.get('y', 0),
            w=facial_area.get('w', 0),
            h=facial_area.get('h', 0),
            confidence=face_data.get('confidence', 0.0),
            landmarks=face_data.get('landmarks')
        )
    
    def _enhance_face_quality(self, face_image: np.ndarray) -> np.ndarray:
        """Enhance face image quality using various techniques."""
        try:
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(face_image)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.1)
            
            # Enhance brightness if needed
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(1.05)
            
            # Convert back to numpy
            enhanced_image = np.array(pil_image)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Error enhancing face quality: {str(e)}")
            return face_image
    
    def _align_face(self, face_image: np.ndarray, landmarks: Optional[Dict]) -> np.ndarray:
        """Align face using facial landmarks."""
        if landmarks is None:
            return face_image
        
        try:
            # Get eye landmarks for alignment
            left_eye = landmarks.get('left_eye')
            right_eye = landmarks.get('right_eye')
            
            if left_eye and right_eye:
                # Calculate eye angle
                eye_angle = np.degrees(np.arctan2(
                    right_eye[1] - left_eye[1],
                    right_eye[0] - left_eye[0]
                ))
                
                # Rotate image to align eyes horizontally
                if abs(eye_angle) > 1.0:  # Only rotate if angle is significant
                    height, width = face_image.shape[:2]
                    center = (width // 2, height // 2)
                    
                    # Create rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(center, eye_angle, 1.0)
                    
                    # Apply rotation
                    aligned_face = cv2.warpAffine(
                        face_image, rotation_matrix, (width, height),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                    )
                    
                    return aligned_face
            
            return face_image
            
        except Exception as e:
            logger.warning(f"Error aligning face: {str(e)}")
            return face_image
    
    def _normalize_face(self, face_image: np.ndarray) -> np.ndarray:
        """Normalize face image for recognition."""
        try:
            # Resize to target size
            resized_face = cv2.resize(
                face_image, self.target_size,
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Convert to float and normalize to [0, 1]
            normalized_face = resized_face.astype(np.float32) / 255.0
            
            # Apply histogram equalization for better contrast
            if len(normalized_face.shape) == 3:
                # Convert to LAB color space for better equalization
                lab = cv2.cvtColor(normalized_face, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = cv2.equalizeHist((lab[:, :, 0] * 255).astype(np.uint8)) / 255.0
                normalized_face = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return normalized_face
            
        except Exception as e:
            logger.error(f"Error normalizing face: {str(e)}")
            return face_image
    
    def _calculate_quality_score(self, face_image: np.ndarray) -> float:
        """Calculate quality score for face image."""
        try:
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Calculate various quality metrics
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 100.0, 1.0)
            
            # 2. Brightness
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 0.5) / 0.5
            
            # 3. Contrast
            contrast = np.std(gray)
            contrast_score = min(contrast / 50.0, 1.0)
            
            # 4. Face size (larger faces are generally better)
            face_size_score = min(face_image.shape[0] * face_image.shape[1] / (224 * 224), 1.0)
            
            # Combine scores (weighted average)
            quality_score = (
                0.4 * sharpness_score +
                0.2 * brightness_score +
                0.2 * contrast_score +
                0.2 * face_size_score
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {str(e)}")
            return 0.5
    
    def save_processed_face(self, processed_face: ProcessedFace, output_path: Union[str, Path]) -> bool:
        """Save processed face image."""
        try:
            # Convert back to uint8 for saving
            face_image = (processed_face.face_image * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            success = cv2.imwrite(str(output_path), face_image_bgr)
            
            if success:
                logger.info(f"Saved processed face to: {output_path}")
                return True
            else:
                logger.error(f"Failed to save processed face to: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving processed face: {str(e)}")
            return False 