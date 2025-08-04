from deepface import DeepFace
import cv2
import os
from datetime import datetime

class FaceRecognizer:
    def __init__(self, faces_dir='static/faces'):
        self.faces_dir = faces_dir
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)

    def register_face(self, image_path, name):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(
                rgb_img,
                detector_backend='retinaface',
                enforce_detection=True,
                align=True
            )
            
            if faces and len(faces) > 0:
                processed_img = cv2.cvtColor(faces[0]['face'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_path, processed_img)
                return True
            return False
            
        except Exception as e:
            print(f"Error registering face: {str(e)}")
            return False

    def scan_image(self, image_path):
        if not os.path.exists(self.faces_dir):
            return None, 0

        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, 0
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            best_match = None
            highest_confidence = 0
            recognition_threshold = 0.4

            faces = DeepFace.extract_faces(
                rgb_img,
                detector_backend='retinaface',
                enforce_detection=True,
                align=True
            )
            
            if not faces:
                return None, 0
                
            face_region = faces[0]['face']
            temp_path = 'temp_scan_face.jpg'
            cv2.imwrite(temp_path, cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR))

            for filename in os.listdir(self.faces_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    known_face_path = os.path.join(self.faces_dir, filename)
                    try:
                        result = DeepFace.verify(
                            img1_path=known_face_path,
                            img2_path=temp_path,
                            model_name='VGG-Face',
                            distance_metric='cosine',
                            enforce_detection=False
                        )
                        
                        distance = result.get("distance", 1.0)
                        confidence = 1 - distance
                        
                        if confidence > highest_confidence and confidence > recognition_threshold:
                            highest_confidence = confidence
                            best_match = os.path.splitext(filename)[0]
                            
                    except Exception as e:
                        print(f"Error comparing with {filename}: {str(e)}")
                        continue

            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return best_match, highest_confidence

        except Exception as e:
            print(f"Error scanning image: {str(e)}")
            return None, 0

    @staticmethod
    def save_upload(file, upload_dir, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"scan_{timestamp}.jpg"
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        return filepath