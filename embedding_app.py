"""
Face Recognition App using Proper Embedding Pipeline
- MTCNN for face detection with landmarks
- Face alignment
- Embedding extraction
- Cosine similarity matching
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import json

from face_embedding_model import FaceRecognitionSystem

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global face recognition system
face_system = None

def initialize_system():
    """Initialize face recognition system"""
    global face_system
    
    model_path = 'models/face_embedding_model.pth'
    embeddings_path = 'models/reference_embeddings.json'
    
    # Create system
    face_system = FaceRecognitionSystem(model_path=model_path if os.path.exists(model_path) else None)
    
    # Load or create reference embeddings
    if os.path.exists(embeddings_path):
        face_system.load_reference_embeddings(embeddings_path)
    else:
        # Register persons from training data
        print("No reference embeddings found. Registering from training data...")
        data_path = 'data/train'
        if os.path.exists(data_path):
            for person_name in os.listdir(data_path):
                person_path = os.path.join(data_path, person_name)
                if not os.path.isdir(person_path):
                    continue
                
                # Get all images for this person
                image_paths = [
                    os.path.join(person_path, f) 
                    for f in os.listdir(person_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ]
                
                if len(image_paths) > 0:
                    face_system.register_person(person_name, image_paths)
            
            # Save embeddings
            face_system.save_reference_embeddings(embeddings_path)
    
    print(f"Face recognition system initialized!")
    print(f"Registered persons: {list(face_system.reference_embeddings.keys())}")

@app.route('/predict', methods=['POST'])
def predict():
    """Predict person from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if face_system is None:
            return jsonify({'error': 'Face recognition system not initialized'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not load image'}), 400
        
        # Recognize
        person_name, confidence = face_system.recognize(image)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if person_name is None:
            return jsonify({
                'success': False,
                'prediction': 'Unknown',
                'confidence': float(confidence),
                'message': 'No matching person found or face not detected'
            })
        
        return jsonify({
            'success': True,
            'prediction': person_name,
            'confidence': float(confidence),
            'similarity': float(confidence)
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in predict: {error_trace}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/register', methods=['POST'])
def register():
    """Register a new person"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        person_name = request.form.get('name', '').strip()
        if not person_name:
            return jsonify({'error': 'Person name required'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if face_system is None:
            return jsonify({'error': 'Face recognition system not initialized'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Register person
        success = face_system.register_person(person_name, [filepath])
        
        if success:
            # Save updated embeddings
            face_system.save_reference_embeddings('models/reference_embeddings.json')
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Person {person_name} registered successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not detect face in image'
            }), 400
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in register: {error_trace}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if face_system is None:
            return jsonify({'error': 'System not initialized'}), 500
        
        return jsonify({
            'model_loaded': os.path.exists('models/face_embedding_model.pth'),
            'registered_persons': list(face_system.reference_embeddings.keys()),
            'num_registered': len(face_system.reference_embeddings),
            'similarity_threshold': face_system.similarity_threshold,
            'embedding_size': 512
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Initializing Face Recognition System...")
    initialize_system()
    print("Starting Flask server...")
    print("Access the API at: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)









