"""
Proper Face Recognition Pipeline:
1. Face Detection (MTCNN with landmarks)
2. Face Alignment (using landmarks)
3. Face Embedding (CNN that outputs feature vectors)
4. Cosine Similarity Matching
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import json
from pathlib import Path

# Try to import MTCNN, fallback to OpenCV if not available
# Disabled MTCNN on macOS due to mutex issues - using OpenCV only
MTCNN_AVAILABLE = False
try:
    # Uncomment below if you want to try MTCNN (may have issues on macOS)
    # from mtcnn import MTCNN
    # MTCNN_AVAILABLE = True
    pass
except ImportError:
    MTCNN_AVAILABLE = False
    pass

if not MTCNN_AVAILABLE:
    print("Using OpenCV Haar Cascade for face detection")
    print("(MTCNN disabled to avoid macOS threading issues)")

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class FaceEmbeddingModel(nn.Module):
    """
    Face embedding model - outputs 512D feature vectors
    Based on MobileFaceNet architecture (lightweight, good for CPU/GPU)
    """
    def __init__(self, embedding_size=512):
        super(FaceEmbeddingModel, self).__init__()
        
        # MobileFaceNet-style architecture
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Depthwise separable convolutions
        self.dw_conv1 = nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False)
        self.bn_dw1 = nn.BatchNorm2d(64)
        self.pw_conv1 = nn.Conv2d(64, 128, 1, bias=False)
        self.bn_pw1 = nn.BatchNorm2d(128)
        
        self.dw_conv2 = nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False)
        self.bn_dw2 = nn.BatchNorm2d(128)
        self.pw_conv2 = nn.Conv2d(128, 256, 1, bias=False)
        self.bn_pw2 = nn.BatchNorm2d(256)
        
        self.dw_conv3 = nn.Conv2d(256, 256, 3, 2, 1, groups=256, bias=False)
        self.bn_dw3 = nn.BatchNorm2d(256)
        self.pw_conv3 = nn.Conv2d(256, 512, 1, bias=False)
        self.bn_pw3 = nn.BatchNorm2d(512)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding layer (use LayerNorm instead of BatchNorm for small batches)
        self.fc = nn.Linear(512, embedding_size)
        self.ln_fc = nn.LayerNorm(embedding_size)
        
    def forward(self, x):
        # Input: (B, 3, 112, 112)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Depthwise separable conv 1
        x = F.relu(self.bn_dw1(self.dw_conv1(x)))
        x = F.relu(self.bn_pw1(self.pw_conv1(x)))
        
        # Depthwise separable conv 2
        x = F.relu(self.bn_dw2(self.dw_conv2(x)))
        x = F.relu(self.bn_pw2(self.pw_conv2(x)))
        
        # Depthwise separable conv 3
        x = F.relu(self.bn_dw3(self.dw_conv3(x)))
        x = F.relu(self.bn_pw3(self.pw_conv3(x)))
        
        # Global pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Embedding
        x = self.fc(x)
        x = self.ln_fc(x)
        
        # L2 normalize (important for cosine similarity)
        x = F.normalize(x, p=2, dim=1)
        
        return x

class FaceDetector:
    """Face detector with landmark detection"""
    def __init__(self):
        if MTCNN_AVAILABLE:
            try:
                # Disable threading to avoid mutex issues on macOS
                import os
                os.environ['OMP_NUM_THREADS'] = '1'
                self.detector = MTCNN()
                self.use_mtcnn = True
                print("Using MTCNN for face detection (with landmarks)")
            except Exception as e:
                print(f"MTCNN initialization failed: {e}. Falling back to OpenCV.")
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.use_mtcnn = False
        else:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_mtcnn = False
            print("Using OpenCV Haar Cascade (no landmarks - less accurate)")
    
    def detect(self, image):
        """
        Detect face and return bounding box + landmarks
        Returns: {
            'box': [x, y, w, h],
            'landmarks': {
                'left_eye': [x, y],
                'right_eye': [x, y],
                'nose': [x, y],
                'mouth_left': [x, y],
                'mouth_right': [x, y]
            }
        } or None
        """
        if self.use_mtcnn:
            # MTCNN returns results in format:
            # {'box': [x, y, w, h], 'confidence': float, 'keypoints': {...}}
            results = self.detector.detect_faces(image)
            if len(results) == 0:
                return None
            
            # Get largest face
            largest = max(results, key=lambda x: x['box'][2] * x['box'][3])
            
            # Extract landmarks
            keypoints = largest.get('keypoints', {})
            if keypoints:
                return {
                    'box': largest['box'],
                    'landmarks': {
                        'left_eye': keypoints.get('left_eye', None),
                        'right_eye': keypoints.get('right_eye', None),
                        'nose': keypoints.get('nose', None),
                        'mouth_left': keypoints.get('mouth_left', None),
                        'mouth_right': keypoints.get('mouth_right', None)
                    }
                }
            else:
                return {'box': largest['box'], 'landmarks': None}
        else:
            # OpenCV fallback (no landmarks)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) == 0:
                return None
            
            largest = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest
            return {'box': [x, y, w, h], 'landmarks': None}

def align_face(image, detection_result, output_size=(112, 112)):
    """
    Align face using landmarks (if available) or center crop
    
    Args:
        image: BGR image
        detection_result: Result from FaceDetector.detect()
        output_size: Target size (width, height)
    
    Returns:
        Aligned face image (RGB, output_size)
    """
    landmarks = detection_result.get('landmarks')
    box = detection_result['box']
    x, y, w, h = box
    
    # Extract face region with padding
    padding = max(w, h) // 4
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    face_roi = image[y1:y2, x1:x2].copy()
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    # If we have landmarks, do proper alignment
    if landmarks and landmarks.get('left_eye') and landmarks.get('right_eye'):
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        # Adjust coordinates relative to face ROI
        left_eye = (left_eye[0] - x1, left_eye[1] - y1)
        right_eye = (right_eye[0] - x1, right_eye[1] - y1)
        
        # Calculate angle to rotate
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate desired eye positions (for 112x112 output)
        # Standard positions: left eye at (35.5, 40.5), right eye at (76.5, 40.5)
        desired_left_eye = (output_size[0] * 0.317, output_size[1] * 0.362)
        desired_right_eye = (output_size[0] * 0.683, output_size[1] * 0.362)
        
        # Calculate scale
        eye_distance = np.sqrt((dx**2) + (dy**2))
        desired_eye_distance = desired_right_eye[0] - desired_left_eye[0]
        scale = desired_eye_distance / eye_distance
        
        # Calculate center point between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                      (left_eye[1] + right_eye[1]) / 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Calculate translation
        tx = desired_left_eye[0] - eyes_center[0]
        ty = desired_left_eye[1] - eyes_center[1]
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformation
        aligned = cv2.warpAffine(face_rgb, M, output_size, flags=cv2.INTER_CUBIC)
    else:
        # No landmarks - just resize maintaining aspect ratio
        h_face, w_face = face_rgb.shape[:2]
        scale = min(output_size[0] / w_face, output_size[1] / h_face)
        new_w = int(w_face * scale)
        new_h = int(h_face * scale)
        
        resized = cv2.resize(face_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Center in output size
        aligned = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        y_offset = (output_size[1] - new_h) // 2
        x_offset = (output_size[0] - new_w) // 2
        aligned[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return aligned

def get_face_transform():
    """Get transform for face embedding model (112x112 input)"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
    ])

class FaceRecognitionSystem:
    """
    Complete face recognition system using embeddings
    """
    def __init__(self, model_path=None, embedding_size=512):
        self.detector = FaceDetector()
        self.embedding_model = FaceEmbeddingModel(embedding_size=embedding_size).to(device)
        self.transform = get_face_transform()
        self.reference_embeddings = {}  # {person_name: [embeddings]}
        self.similarity_threshold = 0.6  # Cosine similarity threshold
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No model loaded. You need to train the embedding model first.")
    
    def load_model(self, model_path):
        """Load trained embedding model"""
        try:
            self.embedding_model.load_state_dict(torch.load(model_path, map_location=device))
            self.embedding_model.eval()
            print(f"Loaded embedding model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def extract_embedding(self, image):
        """
        Extract face embedding from image
        
        Args:
            image: BGR image (OpenCV format)
        
        Returns:
            embedding vector (512D) or None if no face detected
        """
        # Detect face
        detection = self.detector.detect(image)
        if detection is None:
            return None
        
        # Align face
        aligned_face = align_face(image, detection, output_size=(112, 112))
        
        # Transform
        face_tensor = self.transform(aligned_face).unsqueeze(0).to(device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.embedding_model(face_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def register_person(self, person_name, image_paths):
        """
        Register a person by computing embeddings from multiple images
        
        Args:
            person_name: Name of the person
            image_paths: List of image paths for this person
        """
        embeddings = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            embedding = self.extract_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) > 0:
            # Average embeddings for robustness
            avg_embedding = np.mean(embeddings, axis=0)
            # L2 normalize
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            self.reference_embeddings[person_name] = avg_embedding
            print(f"Registered {person_name} with {len(embeddings)} images")
            return True
        return False
    
    def recognize(self, image):
        """
        Recognize person in image
        
        Returns:
            (person_name, confidence) or (None, 0.0) if no match
        """
        embedding = self.extract_embedding(image)
        if embedding is None:
            return None, 0.0
        
        if len(self.reference_embeddings) == 0:
            return None, 0.0
        
        # Compute cosine similarity with all registered persons
        best_match = None
        best_similarity = -1.0
        
        for person_name, ref_embedding in self.reference_embeddings.items():
            # Cosine similarity (dot product since both are normalized)
            similarity = np.dot(embedding, ref_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
        
        # Check threshold
        if best_similarity >= self.similarity_threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity
    
    def save_reference_embeddings(self, filepath):
        """Save reference embeddings to file"""
        data = {
            name: emb.tolist() 
            for name, emb in self.reference_embeddings.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Saved reference embeddings to {filepath}")
    
    def load_reference_embeddings(self, filepath):
        """Load reference embeddings from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.reference_embeddings = {
            name: np.array(emb) 
            for name, emb in data.items()
        }
        print(f"Loaded reference embeddings from {filepath}")





