"""
Test the complete face recognition system
"""
import os
import cv2
from face_embedding_model import FaceRecognitionSystem

def test_recognition():
    """Test face recognition on sample images"""
    print("Testing Face Recognition System...")
    print("=" * 60)
    
    # Initialize system
    model_path = 'models/face_embedding_model.pth'
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first: python train_embedding_model.py")
        return
    
    system = FaceRecognitionSystem(model_path=model_path)
    
    # Register persons from training data
    data_path = 'data/train'
    print("\n1. Registering persons...")
    for person_name in ['tharun', 'sai', 'mohan']:
        person_path = os.path.join(data_path, person_name)
        if os.path.exists(person_path):
            image_paths = [
                os.path.join(person_path, f) 
                for f in os.listdir(person_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if len(image_paths) > 0:
                success = system.register_person(person_name, image_paths[:3])  # Use first 3 images
                if success:
                    print(f"   ✓ Registered {person_name}")
    
    print(f"\n2. Registered persons: {list(system.reference_embeddings.keys())}")
    
    # Test recognition on sample images
    print("\n3. Testing recognition...")
    test_cases = []
    for person_name in ['tharun', 'sai', 'mohan']:
        person_path = os.path.join(data_path, person_name)
        if os.path.exists(person_path):
            # Get an image we didn't use for registration
            all_images = [
                os.path.join(person_path, f) 
                for f in os.listdir(person_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if len(all_images) > 3:
                test_cases.append((person_name, all_images[3]))  # Use 4th image
    
    correct = 0
    total = 0
    
    for expected_name, img_path in test_cases:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        predicted_name, confidence = system.recognize(image)
        total += 1
        
        status = "✓" if predicted_name == expected_name else "✗"
        print(f"   {status} Expected: {expected_name:8s} | Predicted: {str(predicted_name):8s} | Confidence: {confidence:.3f}")
        
        if predicted_name == expected_name:
            correct += 1
    
    print(f"\n4. Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("=" * 60)

if __name__ == '__main__':
    test_recognition()
















