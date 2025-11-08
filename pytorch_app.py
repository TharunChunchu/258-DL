"""
Advanced PyTorch Face Recognition App
Using Proper Embedding-Based Pipeline:
1. Face Detection (MTCNN/OpenCV)
2. Face Alignment
3. Face Embedding (512D vectors)
4. Cosine Similarity Matching
"""

import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import training_api
import threading
import queue
from werkzeug.utils import secure_filename
from datetime import datetime

# Import the new embedding-based system
from face_embedding_model import FaceRecognitionSystem

# Set device (use CPU to avoid threading issues)
device = torch.device('cpu')
print(f"Using device: {device}")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables - using new embedding system
face_system = None

# Old model classes removed - now using embedding-based system from face_embedding_model.py

def load_trained_model():
    """Load the trained embedding model and initialize face recognition system"""
    global face_system
    
    try:
        model_path = 'models/face_embedding_model.pth'
        embeddings_path = 'models/reference_embeddings.json'
        data_path = 'data/train'
        
        # Check if embedding model exists
        if not os.path.exists(model_path):
            print("Embedding model not found. Please run train_embedding_model.py first.")
            return False
        
        # Create face recognition system
        global face_system
        face_system = FaceRecognitionSystem(model_path=model_path)
        
        # Load or create reference embeddings
        if os.path.exists(embeddings_path):
            face_system.load_reference_embeddings(embeddings_path)
            print(f"Loaded reference embeddings. Registered persons: {list(face_system.reference_embeddings.keys())}")
        else:
            # Register persons from training data
            print("No reference embeddings found. Registering from training data...")
            if os.path.exists(data_path):
                for person_name in sorted(os.listdir(data_path)):
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
                print(f"Registered persons: {list(face_system.reference_embeddings.keys())}")
            else:
                print(f"Data directory {data_path} not found.")
                return False
        
        print(f"Face recognition system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def detect_face(image):
    """Detect face in image with multiple strategies for better accuracy"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Strategy 1: Primary detection with standard parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Strategy 2: More lenient parameters if no face found
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # Strategy 3: Even more lenient with different scale
        if len(faces) == 0:
            # Enhance image contrast for better detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            faces = face_cascade.detectMultiScale(
                enhanced_gray, 
                scaleFactor=1.3, 
                minNeighbors=2,
                minSize=(15, 15),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        if len(faces) == 0:
            return None
        
        # Select the largest face (matching training approach)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        
        # Minimum size check - match training requirement
        x, y, w, h = largest_face
        if w < 60 or h < 60:
            return None
        
        return largest_face
    except Exception as e:
        print(f"Error detecting face: {str(e)}")
        return None

def preprocess_face(image, face_box):
    """
    Preprocess face for PyTorch model - Handles different face sizes and aspect ratios
    
    Key improvements:
    1. Maintains aspect ratio to avoid distortion
    2. Uses adaptive padding based on face size
    3. Better interpolation for small faces (LANCZOS4 for upscaling)
    4. Adds padding to make square instead of stretching
    """
    try:
        x, y, w, h = face_box
        
        # Convert BGR to RGB first (before cropping) - match training
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # STEP 1: Adaptive padding based on face size
        # For small faces, use more padding to preserve quality
        # For large faces, use standard padding
        face_size = max(w, h)
        if face_size < 100:
            # Small face: use 30% padding to preserve more context
            padding_ratio = 0.3
        elif face_size < 200:
            # Medium face: use 25% padding (standard)
            padding_ratio = 0.25
        else:
            # Large face: use 20% padding (less needed)
            padding_ratio = 0.2
        
        padding = int(max(w, h) * padding_ratio)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_rgb.shape[1], x + w + padding)
        y2 = min(image_rgb.shape[0], y + h + padding)
        
        # STEP 2: Extract face region with padding
        face_img = image_rgb[y1:y2, x1:x2]
        face_h, face_w = face_img.shape[:2]
        
        # STEP 3: Maintain aspect ratio and add padding to make square
        # This prevents distortion from stretching non-square faces
        target_size = 224
        max_dim = max(face_h, face_w)
        
        # Calculate scale to fit within target_size while maintaining aspect ratio
        scale = target_size / max_dim
        new_w = int(face_w * scale)
        new_h = int(face_h * scale)
        
        # Choose best interpolation based on whether we're upscaling or downscaling
        if max_dim < target_size:
            # Upscaling: Use LANCZOS4 for best quality (better than CUBIC)
            # LANCZOS4 is slower but produces much better results for small faces
            interpolation = cv2.INTER_LANCZOS4
        else:
            # Downscaling: Use INTER_AREA for best quality
            interpolation = cv2.INTER_AREA
        
        # Resize maintaining aspect ratio
        face_img_resized = cv2.resize(face_img, (new_w, new_h), interpolation=interpolation)
        
        # STEP 4: Add padding to make it square (224x224) without distortion
        # Calculate padding needed to center the face
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        # Create square image with padding (use edge color for padding)
        face_img_square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate placement (centered)
        start_h = pad_h
        start_w = pad_w
        end_h = start_h + new_h
        end_w = start_w + new_w
        
        # Place resized face in center
        face_img_square[start_h:end_h, start_w:end_w] = face_img_resized
        
        # Alternative: Use border replication for padding (can be better than black)
        # face_img_square = cv2.copyMakeBorder(
        #     face_img_resized,
        #     pad_h, target_size - new_h - pad_h,
        #     pad_w, target_size - new_w - pad_w,
        #     cv2.BORDER_REPLICATE
        # )
        
        # STEP 5: Apply transform - MUST match validation transform from training exactly
        # Training validation transform: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
        face_tensor = transform(face_img_square)  # transform handles: ToPILImage -> Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor.to(device)
        
    except Exception as e:
        print(f"Error preprocessing face: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict_person(image):
    """
    Predict person using embedding-based recognition
    
    Args:
        image: Input image (BGR format from OpenCV)
    
    Returns:
        (person_name, confidence, all_similarities)
    """
    try:
        if face_system is None:
            return "Model not loaded. Please train a model first.", 0.0, {}
        
        # Recognize using embedding system
        person_name, confidence = face_system.recognize(image)
        
        # Get all similarities for display
        all_similarities = {}
        if face_system.reference_embeddings:
            # Extract embedding from image
            embedding = face_system.extract_embedding(image)
            if embedding is not None:
                for ref_name, ref_emb in face_system.reference_embeddings.items():
                    similarity = np.dot(embedding, ref_emb)
                    all_similarities[ref_name] = float(similarity)
        
        # Debug output
        print(f"\n=== Prediction Debug ===")
        print(f"Predicted: {person_name}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All similarities:")
        for name, sim in sorted(all_similarities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {sim:.4f} ({sim*100:.2f}%)")
        print("=" * 30)
        
        if person_name is None:
            return "Unknown", confidence, all_similarities
        
        return person_name, confidence, all_similarities
        
    except Exception as e:
        print(f"Error predicting person: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error", 0.0, {}

def save_preprocessed_face_debug(image, face_box, output_path):
    """Save preprocessed face for debugging - see what model receives"""
    try:
        x, y, w, h = face_box
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Adaptive padding
        face_size = max(w, h)
        if face_size < 100:
            padding_ratio = 0.3
        elif face_size < 200:
            padding_ratio = 0.25
        else:
            padding_ratio = 0.2
        
        padding = int(max(w, h) * padding_ratio)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_rgb.shape[1], x + w + padding)
        y2 = min(image_rgb.shape[0], y + h + padding)
        
        face_img = image_rgb[y1:y2, x1:x2]
        face_h, face_w = face_img.shape[:2]
        
        target_size = 224
        max_dim = max(face_h, face_w)
        scale = target_size / max_dim
        new_w = int(face_w * scale)
        new_h = int(face_h * scale)
        
        if max_dim < target_size:
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_AREA
        
        face_img_resized = cv2.resize(face_img, (new_w, new_h), interpolation=interpolation)
        
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        face_img_square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        face_img_square[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = face_img_resized
        
        # Save for debugging
        face_img_bgr = cv2.cvtColor(face_img_square, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, face_img_bgr)
        print(f"Debug: Saved preprocessed face to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving debug face: {str(e)}")
        return False

def process_uploaded_image(image_path, save_extracted_face=False, extracted_face_path=None):
    """Process uploaded image with optional face extraction"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not load image"
        
        # Make prediction using embedding system
        prediction, confidence, all_similarities = predict_person(image)
        
        # Check if face was detected (if confidence is 0, likely no face)
        faces_detected = 1 if confidence > 0 else 0
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'faces_detected': faces_detected
        }
        
        # Add all probabilities (similarities) for frontend display
        if all_similarities:
            # Convert similarities to probabilities (softmax-like)
            # Normalize to make them sum to 1 for display
            max_sim = max(all_similarities.values()) if all_similarities.values() else 1.0
            if max_sim > 0:
                # Scale similarities to [0, 1] range and normalize
                scaled = {k: max(0, v / max_sim) for k, v in all_similarities.items()}
                total = sum(scaled.values())
                if total > 0:
                    result['all_probabilities'] = {k: v / total for k, v in scaled.items()}
                else:
                    result['all_probabilities'] = all_similarities
            else:
                result['all_probabilities'] = all_similarities
        
        return result, None
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

@app.route('/')
def index():
    """Main page - serve React app in production, or redirect in development"""
    # In production, serve built React app
    build_path = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
    if os.path.exists(build_path):
        return send_from_directory(build_path, 'index.html')
    # In development, React dev server runs on port 3000
    return jsonify({
        'message': 'React frontend not built. Run "npm run build" in frontend folder, or use React dev server on port 3000'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image
            result, error = process_uploaded_image(filepath)
            
            if error:
                return jsonify({'error': error}), 400
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            # All probabilities are already in result from process_uploaded_image
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'faces_detected': result['faces_detected'],
                'all_probabilities': result.get('all_probabilities', {})
            })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Training start error: {error_trace}")
        return jsonify({'error': f'Server error: {str(e)}', 'trace': error_trace}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training via API"""
    try:
        if training_api.training_state['is_training']:
            return jsonify({'error': 'Training already in progress'}), 400
        
        data = request.get_json() or {}
        num_epochs = data.get('epochs', 50)
        data_path = 'data/train'
        
        # Create new progress queue for this training session
        progress_queue = queue.Queue()
        training_api.training_state['progress_queue'] = progress_queue
        
        # Start training in background thread
        thread = threading.Thread(
            target=training_api.training_thread,
            args=(data_path, num_epochs, progress_queue)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'epochs': num_epochs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/status')
def training_status():
    """Get current training status"""
    state = training_api.training_state
    return jsonify({
        'is_training': state['is_training'],
        'current_epoch': state['current_epoch'],
        'total_epochs': state['total_epochs'],
        'train_loss': state['train_loss'],
        'train_acc': state['train_acc'],
        'val_loss': state['val_loss'],
        'val_acc': state['val_acc'],
        'history': state['history']
    })

@app.route('/api/training/progress')
def training_progress():
    """Server-Sent Events stream for training progress"""
    def generate():
        queue = training_api.training_state['progress_queue']
        while True:
            try:
                # Get update from queue (with timeout)
                try:
                    update = queue.get(timeout=1)
                    yield f"data: {json.dumps(update)}\n\n"
                    if update.get('status') == 'completed' or update.get('error'):
                        break
                except:
                    # Send heartbeat
                    yield f"data: {json.dumps({'heartbeat': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training"""
    training_api.training_state['is_training'] = False
    return jsonify({'success': True, 'message': 'Training stop requested'})

@app.route('/model_info')
def model_info():
    """Get model information"""
    try:
        if face_system is None:
            return jsonify({'error': 'Model not loaded'}), 400
        
        return jsonify({
            'model_loaded': True,
            'device': str(device),
            'registered_persons': list(face_system.reference_embeddings.keys()) if face_system.reference_embeddings else [],
            'num_registered': len(face_system.reference_embeddings) if face_system.reference_embeddings else 0,
            'similarity_threshold': face_system.similarity_threshold,
            'model_architecture': 'Embedding-Based Face Recognition (MobileFaceNet-style)',
            'embedding_size': 512
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Face Recognition System (Embedding-Based)...")
    
    # Load trained embedding model
    model_loaded = load_trained_model()
    if not model_loaded:
        print("Warning: Model not loaded. Please run: python train_embedding_model.py")
        print("Face recognition will not work until a model is trained.")
    else:
        print(f"✓ System ready! Registered {len(face_system.reference_embeddings)} persons")
    
    print("Access the web interface at: http://localhost:5001")
    print("React frontend should be at: http://localhost:3000")
    app.run(debug=True, host='0.0.0.0', port=5001)









