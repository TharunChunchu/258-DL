"""
Extract and display sample faces from training images
"""

import cv2
import os
import numpy as np

def extract_sample_faces():
    """Extract sample faces from each class and save them"""
    data_path = 'data/train'
    output_path = 'sample_extracted_faces'
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get class names
    class_names = sorted([d for d in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, d))])
    
    print(f"Found classes: {class_names}")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\nProcessing {class_name}...")
        
        # Process first image from each class
        for image_file in image_files[:1]:  # Just get first image
            image_path = os.path.join(class_path, image_file)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) == 0:
                    # Try with more lenient parameters
                    faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.2, 
                        minNeighbors=3,
                        minSize=(20, 20)
                    )
                
                if len(faces) == 0:
                    print(f"  No face detected in {image_file}")
                    continue
                
                # Select largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Add padding
                padding = max(w, h) // 4
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image_rgb.shape[1], x + w + padding)
                y2 = min(image_rgb.shape[0], y + h + padding)
                
                # Extract face with padding
                face_img = image_rgb[y1:y2, x1:x2]
                
                # Resize to 224x224 (same as training)
                face_img_resized = cv2.resize(face_img, (224, 224))
                
                # Save original extracted face
                face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                output_file = os.path.join(output_path, f"{class_name}_extracted.jpg")
                cv2.imwrite(output_file, face_bgr)
                print(f"  ✓ Saved: {output_file} (Original size: {face_img.shape[1]}x{face_img.shape[0]})")
                
                # Save resized face (224x224)
                face_resized_bgr = cv2.cvtColor(face_img_resized, cv2.COLOR_RGB2BGR)
                output_file_resized = os.path.join(output_path, f"{class_name}_resized_224x224.jpg")
                cv2.imwrite(output_file_resized, face_resized_bgr)
                print(f"  ✓ Saved: {output_file_resized} (Resized: 224x224)")
                
                # Also save with bounding box on original image
                image_with_box = image.copy()
                cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (255, 0, 0), 2)
                output_file_box = os.path.join(output_path, f"{class_name}_with_detection.jpg")
                cv2.imwrite(output_file_box, image_with_box)
                print(f"  ✓ Saved: {output_file_box} (Green: face, Blue: with padding)")
                
                break  # Only process first image per class
                
            except Exception as e:
                print(f"  Error processing {image_file}: {str(e)}")
                continue
    
    print(f"\n✅ Sample faces extracted to: {output_path}/")
    print("\nFiles created:")
    for class_name in class_names:
        print(f"  - {class_name}_extracted.jpg (original extracted face)")
        print(f"  - {class_name}_resized_224x224.jpg (resized for model)")
        print(f"  - {class_name}_with_detection.jpg (original with detection box)")

if __name__ == '__main__':
    extract_sample_faces()













