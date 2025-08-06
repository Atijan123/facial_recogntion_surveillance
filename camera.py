import cv2
import os
from datetime import datetime
from deepface import DeepFace
import numpy as np

class Camera:
    def __init__(self, faces_dir='static/faces'):
        self.video = None
        self.faces_dir = faces_dir
        self.known_faces = self._load_known_faces()
        self.detection_confidence = 0.9
        self.recognition_threshold = 0.4
        self._initialize_camera()

    def _initialize_camera(self):
        if self.video is None or not self.video.isOpened():
            self.video = cv2.VideoCapture(0)
            if not self.video.isOpened():
                print("Error: Could not open video capture device")
                return False
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_FPS, 30)
        return True

    def _load_known_faces(self):
        known_faces = {}
        if os.path.exists(self.faces_dir):
            for filename in os.listdir(self.faces_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(self.faces_dir, filename)
                    name = os.path.splitext(filename)[0]
                    known_faces[name] = path
        return known_faces

    def get_frame(self):
        if not self._initialize_camera():
            return None, None

        success, frame = self.video.read()
        if not success:
            print("Error: Failed to read frame from camera")
            self.video.release()
            self.video = None
            return None, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            detected_faces = DeepFace.extract_faces(
                rgb_frame,
                detector_backend='retinaface',
                enforce_detection=True,
                align=True
            )
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            return frame, []

        face_instances = []
        for face_data in detected_faces:
            if face_data.get('confidence', 0) < self.detection_confidence:
                continue
                
            face_region = face_data['face']
            facial_area = face_data['facial_area']
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']

            name = "Unknown"
            max_confidence = 0
            
            # Fix face_region data type and range for OpenCV
            if face_region.dtype != 'uint8':
                # Convert from float (0-1) to uint8 (0-255)
                if face_region.max() <= 1.0:
                    face_region = (face_region * 255).astype('uint8')
                else:
                    face_region = face_region.astype('uint8')
            
            temp_path = 'temp_camera_face.jpg'
            cv2.imwrite(temp_path, cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR))
            
            for known_name, known_path in self.known_faces.items():
                try:
                    result = DeepFace.verify(
                        img1_path=known_path,
                        img2_path=temp_path,
                        model_name='VGG-Face',
                        distance_metric="cosine",
                        enforce_detection=False
                    )
                    
                    distance = result.get("distance", 1.0)
                    confidence = 1 - distance
                    
                    if confidence > max_confidence and confidence > self.recognition_threshold:
                        max_confidence = confidence
                        name = known_name
                        
                except Exception as e:
                    continue

            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Draw rectangle and name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{name} ({max_confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            face_instances.append({
                'name': name,
                'confidence': max_confidence,
                'frame': frame[y:y+h, x:x+w]
            })

        return frame, face_instances

    def __del__(self):
        if self.video and self.video.isOpened():
            self.video.release()

    def save_snapshot(self, face_frame, logs_dir='static/logs'):
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'snapshot_{timestamp}.jpg'
        filepath = os.path.join(logs_dir, filename)
        cv2.imwrite(filepath, face_frame)
        return filepath