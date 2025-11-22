"""
Training API with real-time progress updates
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import threading
import queue

# Import model architecture from pytorch_train
import sys
sys.path.append(os.path.dirname(__file__))
from pytorch_train import AdvancedFaceCNN, AttentionBlock, ResidualBlock, get_transforms, prepare_dataset, FaceDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Global training state
training_state = {
    'is_training': False,
    'progress_queue': queue.Queue(),
    'current_epoch': 0,
    'total_epochs': 0,
    'train_loss': 0.0,
    'train_acc': 0.0,
    'val_loss': 0.0,
    'val_acc': 0.0,
    'history': {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': []
    }
}

def train_model_with_progress(model, train_loader, val_loader, num_epochs=50, progress_callback=None, class_weights=None):
    """Train the PyTorch model with progress updates"""
    # Use weighted CrossEntropyLoss - more stable for small datasets
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    # Different learning rates for pretrained vs new layers
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'fc'):
        # Transfer learning: lower LR for pretrained, higher for new layers
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad]
        classifier_params = [p for n, p in model.named_parameters() if 'fc' in n]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 0.0001},  # Lower LR for pretrained
            {'params': classifier_params, 'lr': 0.001}   # Higher LR for new layers
        ], weight_decay=0.0001)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        training_state['current_epoch'] = epoch + 1
        training_state['total_epochs'] = num_epochs
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update state
        training_state['train_loss'] = train_loss
        training_state['train_acc'] = train_acc
        training_state['val_loss'] = val_loss
        training_state['val_acc'] = val_acc
        training_state['history']['train_losses'] = train_losses.copy()
        training_state['history']['train_accuracies'] = train_accuracies.copy()
        training_state['history']['val_losses'] = val_losses.copy()
        training_state['history']['val_accuracies'] = val_accuracies.copy()
        
        # Send progress update
        if progress_callback:
            progress_callback({
                'epoch': epoch + 1,
                'total_epochs': num_epochs,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/pytorch_face_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if progress_callback:
                    progress_callback({
                        'epoch': epoch + 1,
                        'status': 'early_stopping',
                        'best_val_acc': best_val_acc
                    })
                break
        
        scheduler.step()  # CosineAnnealingWarmRestarts doesn't need loss
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def training_thread(data_path, num_epochs, progress_queue):
    """Training thread function"""
    try:
        training_state['is_training'] = True
        # Reset training state
        training_state['current_epoch'] = 0
        training_state['total_epochs'] = 0
        training_state['train_loss'] = 0.0
        training_state['train_acc'] = 0.0
        training_state['val_loss'] = 0.0
        training_state['val_acc'] = 0.0
        training_state['history'] = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
        
        def progress_callback(data):
            progress_queue.put(data)
        
        # Prepare dataset
        images, labels, class_names = prepare_dataset(data_path, save_extracted_faces=False)
        
        if len(images) == 0:
            progress_queue.put({'error': 'No images found!'})
            return
        
        # Split data with stratification to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Calculate class distribution and weights
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        total_samples = len(y_train)
        
        class_weights = []
        for idx in range(len(class_names)):
            class_count = class_counts.get(idx, 1)
            weight = total_samples / (len(class_names) * class_count)
            class_weights.append(weight)
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        progress_queue.put({
            'status': 'data_loaded',
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'classes': class_names,
            'class_distribution': {class_names[idx]: int(class_counts.get(idx, 0)) for idx in range(len(class_names))},
            'class_weights': {class_names[idx]: f'{w:.3f}' for idx, w in enumerate(class_weights)}
        })
        
        # Get transforms
        train_transform, val_transform = get_transforms()
        
        # Create datasets
        train_dataset = FaceDataset(X_train, y_train, train_transform)
        val_dataset = FaceDataset(X_val, y_val, val_transform)
        
        # Create weighted sampler for balanced training
        sample_weights = []
        for label in y_train:
            sample_weights.append(class_weights[label])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders with weighted sampling
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Create model
        print("Using Transfer Learning with Pretrained ResNet50...")
        model = AdvancedFaceCNN(num_classes=len(class_names), use_pretrained=True).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        progress_queue.put({
            'status': 'model_created',
            'total_parameters': total_params,
            'num_classes': len(class_names)
        })
        
        # Train model
        progress_queue.put({'status': 'training_started'})
        train_losses, train_accuracies, val_losses, val_accuracies = train_model_with_progress(
            model, train_loader, val_loader, num_epochs=num_epochs, 
            progress_callback=progress_callback, class_weights=class_weights_tensor
        )
        
        # Ensure models directory exists before loading
        os.makedirs('models', exist_ok=True)
        
        # Load best model and evaluate
        if os.path.exists('models/pytorch_face_model.pth'):
            model.load_state_dict(torch.load('models/pytorch_face_model.pth'))
        
        # Test evaluation
        test_dataset = FaceDataset(X_test, y_test, val_transform)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        
        progress_queue.put({
            'status': 'completed',
            'test_accuracy': test_accuracy,
            'final_train_acc': train_accuracies[-1] if train_accuracies else 0,
            'final_val_acc': val_accuracies[-1] if val_accuracies else 0
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Training error: {error_trace}")
        progress_queue.put({'error': str(e), 'trace': error_trace})
    finally:
        training_state['is_training'] = False






