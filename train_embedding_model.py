"""
Train face embedding model using proper pipeline:
1. Detect faces with MTCNN (landmarks)
2. Align faces
3. Train embedding model with triplet loss or ArcFace
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

from face_embedding_model import (
    FaceDetector, align_face, FaceEmbeddingModel, get_face_transform
)

# Use CPU to avoid MPS threading issues on macOS
device = torch.device('cpu')  # Changed from MPS to avoid mutex issues
print(f"Using device: {device}")

class FaceEmbeddingDataset(Dataset):
    """Dataset for training face embedding model"""
    def __init__(self, image_paths, labels, detector, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.detector = detector
        self.transform = transform or get_face_transform()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return dummy if image can't be loaded
            dummy = np.zeros((112, 112, 3), dtype=np.uint8)
            return self.transform(dummy), label
        
        # Detect and align face
        detection = self.detector.detect(image)
        if detection is None:
            dummy = np.zeros((112, 112, 3), dtype=np.uint8)
            return self.transform(dummy), label
        
        aligned_face = align_face(image, detection, output_size=(112, 112))
        
        # Transform
        face_tensor = self.transform(aligned_face)
        
        return face_tensor, label

def prepare_dataset(data_path):
    """Prepare dataset from folder structure"""
    detector = FaceDetector()
    image_paths = []
    labels = []
    class_names = []
    
    # Get class names from folders
    for class_name in sorted(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        class_names.append(class_name)
        class_idx = len(class_names) - 1
        
        # Get all images
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            img_path = os.path.join(class_path, img_file)
            image_paths.append(img_path)
            labels.append(class_idx)
    
    return image_paths, labels, class_names

def train_embedding_model(data_path, num_epochs=100, embedding_size=512):
    """Train face embedding model"""
    print("Preparing dataset...")
    image_paths, labels, class_names = prepare_dataset(data_path)
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    print(f"Found {len(image_paths)} images from {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create detector
    detector = FaceDetector()
    
    # Create datasets
    train_dataset = FaceEmbeddingDataset(X_train, y_train, detector)
    test_dataset = FaceEmbeddingDataset(X_test, y_test, detector)
    
    # Create data loaders (num_workers=0 to avoid threading issues on macOS)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
    
    # Create model
    model = FaceEmbeddingModel(embedding_size=embedding_size).to(device)
    
    # Use ArcFace loss (better than triplet loss for small datasets)
    # For simplicity, we'll use CrossEntropy with a linear layer for classification
    # Then extract embeddings from the penultimate layer
    num_classes = len(class_names)
    
    # Add classification head for training
    classifier = nn.Linear(embedding_size, num_classes).to(device)
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 0.001},
        {'params': classifier.parameters(), 'lr': 0.01}
    ], weight_decay=0.0001)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels_tensor in pbar:
            images = images.to(device)
            labels_tensor = labels_tensor.to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings
            embeddings = model(images)
            
            # Classify
            logits = classifier(embeddings)
            loss = criterion(logits, labels_tensor)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels_tensor.size(0)
            train_correct += (predicted == labels_tensor).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        classifier.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels_tensor in test_loader:
                images = images.to(device)
                labels_tensor = labels_tensor.to(device)
                
                embeddings = model(images)
                logits = classifier(embeddings)
                _, predicted = torch.max(logits.data, 1)
                
                test_total += labels_tensor.size(0)
                test_correct += (predicted == labels_tensor).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        train_acc = 100.0 * train_correct / train_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'models/face_embedding_model.pth')
            print(f"  → New best model saved! (Test Acc: {test_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: models/face_embedding_model.pth")
    
    # Save class names
    with open('models/class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    return model, class_names

if __name__ == '__main__':
    data_path = 'data/train'
    train_embedding_model(data_path, num_epochs=100)







