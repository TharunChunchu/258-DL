"""
Advanced PyTorch Face Recognition Flask App
With enhanced UI, progress tracking, and better error handling
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime
import time
import base64
from io import BytesIO
from PIL import Image

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class AdvancedCNN(nn.Module):
    """Advanced CNN with attention mechanisms"""
    def __init__(self, num_classes=3):
        super(AdvancedCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # 512 * 2 for avg + max pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        
        # Concatenate features
        features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        output = self.classifier(features)
        return output

class FaceRecognitionApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.model = None
        self.class_names = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.setup_routes()
        self.load_model()
    
    def load_model(self):
        """Load the trained PyTorch model"""
        try:
            model_path = 'models/pytorch_face_model.pth'
            class_names_path = 'models/pytorch_class_names.json'
            
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found")
                return False
            
            if not os.path.exists(class_names_path):
                print(f"Class names file {class_names_path} not found")
                return False
            
            # Load class names
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            
            # Create model
            self.model = AdvancedCNN(num_classes=len(self.class_names)).to(device)
            
            # Load weights
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            
            print(f"PyTorch model loaded successfully! Classes: {self.class_names}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image_bytes):
        """Preprocess uploaded image"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                raise ValueError("No face detected in the image")
            
            # Extract the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Add padding
            padding = max(w, h) // 4
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_img = image[y1:y2, x1:x2]
            
            # Resize to model input size
            face_img = cv2.resize(face_img, (224, 224))
            
            # Convert to tensor
            face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float() / 255.0
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            face_tensor = (face_tensor - mean) / std
            
            return face_tensor, len(faces)
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def predict(self, image_bytes):
        """Make prediction on uploaded image"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess image
            face_tensor, faces_detected = self.preprocess_image(image_bytes)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction_idx = predicted.item()
                confidence_score = confidence.item()
                
                if prediction_idx < len(self.class_names):
                    prediction = self.class_names[prediction_idx]
                else:
                    prediction = "Unknown"
                    confidence_score = 0.0
            
            return {
                'prediction': prediction,
                'confidence': confidence_score,
                'faces_detected': faces_detected,
                'all_probabilities': probabilities.cpu().numpy().tolist()[0]
            }
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('advanced_ui.html')
        
        @self.app.route('/predict', methods=['POST'])
        def predict_route():
            try:
                start_time = time.time()
                
                # Check if image is uploaded
                if 'image' not in request.files:
                    return jsonify({'error': 'No image uploaded'})
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': 'No image selected'})
                
                # Read image bytes
                image_bytes = file.read()
                
                # Make prediction
                result = self.predict(image_bytes)
                
                # Add processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                result['processing_time'] = round(processing_time, 2)
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/model_info')
        def model_info():
            return jsonify({
                'model_loaded': self.model is not None,
                'class_names': self.class_names,
                'device': str(device),
                'architecture': 'Advanced CNN with Attention',
                'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
            })
        
        @self.app.route('/health')
        def health():
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'timestamp': datetime.now().isoformat()
            })
    
    def run(self, host='0.0.0.0', port=5001, debug=True):
        """Run the Flask app"""
        print("Starting Advanced PyTorch Face Recognition System...")
        if self.model is not None:
            print("PyTorch model loaded successfully!")
        else:
            print("Warning: Model not loaded!")
        
        print(f"Access the web interface at: http://localhost:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    app = FaceRecognitionApp()
    app.run()


































