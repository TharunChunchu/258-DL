"""
Quick test of the embedding pipeline
"""
import os
import cv2
import numpy as np
from face_embedding_model import FaceDetector, align_face, FaceRecognitionSystem

def test_pipeline():
    """Test the face detection and alignment pipeline"""
    print("Testing Face Recognition Pipeline...")
    print("=" * 60)
    
    # Initialize detector
    print("\n1. Initializing face detector...")
    detector = FaceDetector()
    
    # Test on a sample image
    data_path = 'data/train'
    test_images = []
    
    for person_name in ['tharun', 'sai', 'mohan']:
        person_path = os.path.join(data_path, person_name)
        if os.path.exists(person_path):
            for img_file in os.listdir(person_path)[:1]:  # Just first image
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append((person_name, os.path.join(person_path, img_file)))
                    break
    
    print(f"\n2. Testing on {len(test_images)} images...")
    
    for person_name, img_path in test_images:
        print(f"\n   Testing: {person_name} - {os.path.basename(img_path)}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"      ❌ Could not load image")
            continue
        
        # Detect face
        detection = detector.detect(image)
        if detection is None:
            print(f"      ❌ No face detected")
            continue
        
        print(f"      ✓ Face detected: {detection['box']}")
        
        # Align face
        try:
            aligned = align_face(image, detection, output_size=(112, 112))
            print(f"      ✓ Face aligned: {aligned.shape}")
            
            # Save aligned face for inspection
            output_dir = 'test_output'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{person_name}_aligned.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            print(f"      ✓ Saved aligned face to: {output_path}")
        except Exception as e:
            print(f"      ❌ Alignment failed: {e}")
    
    print("\n" + "=" * 60)
    print("Pipeline test completed!")
    print(f"Check 'test_output/' folder for aligned faces")

if __name__ == '__main__':
    test_pipeline()





