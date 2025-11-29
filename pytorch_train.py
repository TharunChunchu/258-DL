"""
Advanced PyTorch Face Recognition Training
Using sophisticated CNN architecture with attention mechanisms
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class AttentionBlock(nn.Module):
    """Attention mechanism for better feature focus"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x

class ResidualBlock(nn.Module):
    """Residual block with attention"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.attention = AttentionBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += residual
        out = F.relu(out)
        
        return out

class AdvancedFaceCNN(nn.Module):
    """Transfer Learning Model using Pretrained ResNet50"""
    def __init__(self, num_classes=3, use_pretrained=True):
        super(AdvancedFaceCNN, self).__init__()
        
        # Use pretrained ResNet50 as backbone
        from torchvision import models
        resnet = models.resnet50(pretrained=use_pretrained)
        
        # Freeze early layers, fine-tune later layers
        if use_pretrained:
            # Freeze first few layers
            for i, param in enumerate(resnet.parameters()):
                if i < 100:  # Freeze first ~100 layers
                    param.requires_grad = False
        
        # Replace the final fully connected layer
        # Use LayerNorm instead of BatchNorm1d to avoid issues with batch size=1
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.backbone = resnet
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetFaceCNN(nn.Module):
    """Better model using EfficientNet-B3"""
    def __init__(self, num_classes=3, use_pretrained=True):
        super(EfficientNetFaceCNN, self).__init__()
        
        try:
            from torchvision import models
            # Try to use EfficientNet (requires torchvision 0.13+)
            efficientnet = models.efficientnet_b3(pretrained=use_pretrained)
            
            # Freeze early layers
            if use_pretrained:
                for i, param in enumerate(efficientnet.features.parameters()):
                    if i < len(list(efficientnet.features.children())) // 2:
                        param.requires_grad = False
            
            # Replace classifier
            num_features = efficientnet.classifier[1].in_features
            efficientnet.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            
            self.backbone = efficientnet
        except Exception as e:
            print(f"EfficientNet not available: {e}. Falling back to ResNet50.")
            # Fallback to ResNet50
            self.backbone = AdvancedFaceCNN(num_classes, use_pretrained).backbone
    
    def forward(self, x):
        return self.backbone(x)

class VisionTransformerFaceCNN(nn.Module):
    """Best model using Vision Transformer"""
    def __init__(self, num_classes=3, use_pretrained=True):
        super(VisionTransformerFaceCNN, self).__init__()
        
        try:
            from torchvision import models
            # Use ViT-B/16
            vit = models.vit_b_16(pretrained=use_pretrained)
            
            # Freeze encoder blocks
            if use_pretrained:
                for param in vit.encoder.layers[:6].parameters():  # Freeze first 6 layers
                    param.requires_grad = False
            
            # Replace head
            num_features = vit.heads.head.in_features
            vit.heads.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            
            self.backbone = vit
        except Exception as e:
            print(f"Vision Transformer not available: {e}. Falling back to ResNet50.")
            self.backbone = AdvancedFaceCNN(num_classes, use_pretrained).backbone
    
    def forward(self, x):
        return self.backbone(x)

class FaceDataset(Dataset):
    """Custom dataset for face recognition"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Get aggressive data augmentation transforms for small dataset"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Slightly larger for better quality
        transforms.RandomCrop(224),  # Random crop for more variation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.4), ratio=(0.3, 3.3))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),  # Center crop for validation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_dataset(data_path, save_extracted_faces=False, extracted_faces_dir='data/extracted_faces'):
    """Prepare dataset with face detection and optional face extraction"""
    images = []
    labels = []
    
    # Get class names directly from folder structure
    class_names = sorted([d for d in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, d))])
    
    if len(class_names) == 0:
        print(f"No class folders found in {data_path}")
        return images, labels, class_names
    
    print(f"Found classes: {class_names}")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create directory for extracted faces if saving is enabled
    if save_extracted_faces:
        os.makedirs(extracted_faces_dir, exist_ok=True)
        for class_name in class_names:
            os.makedirs(os.path.join(extracted_faces_dir, class_name), exist_ok=True)
        print(f"Extracted faces will be saved to: {extracted_faces_dir}")
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"Directory {class_path} not found")
            continue
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Found {len(image_files)} images for {class_name}")
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face - try multiple scales for better detection
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # Try different detection parameters for better face detection
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)  # Minimum face size
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
                    print(f"No face detected in {image_file} - skipping")
                    continue
                
                # Extract largest face (for consistency)
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Reduced minimum size - use better upscaling with pretrained model
                # Pretrained models handle smaller faces better due to learned features
                if w < 60 or h < 60:
                    print(f"Face too small in {image_file} (w={w}, h={h}) - skipping (minimum 60px required)")
                    continue
                
                # Add padding around face (25% of face size - improves recognition)
                padding = max(w, h) // 4
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                # Extract ONLY the face region with padding (not full image)
                face_img = image[y1:y2, x1:x2]
                face_h, face_w = face_img.shape[:2]
                
                # IMPROVED: Maintain aspect ratio and add padding to make square
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
                
                # Add padding to make it square (224x224) without distortion
                pad_h = (target_size - new_h) // 2
                pad_w = (target_size - new_w) // 2
                
                # Create square image with padding
                face_img_square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                face_img_square[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = face_img_resized
                face_img = face_img_square
                
                # Save extracted face if enabled (like tutorial approach)
                if save_extracted_faces:
                    # Create unique filename
                    base_name = os.path.splitext(image_file)[0]
                    face_filename = f"{base_name}_face_{w}x{h}_processed.jpg"
                    face_save_path = os.path.join(extracted_faces_dir, class_name, face_filename)
                    
                    # Convert RGB to BGR for saving (OpenCV uses BGR)
                    face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(face_save_path, face_img_bgr)
                    print(f"  → Saved extracted face: {face_save_path}")
                
                images.append(face_img)
                labels.append(class_idx)
                print(f"Processed {image_file} for {class_name}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Dataset prepared: {len(images)} images, {len(class_names)} classes")
    return images, labels, class_names

def train_model(model, train_loader, val_loader, num_epochs=50, class_weights=None, learning_rate_multiplier=1.0):
    """Train the PyTorch model"""
    # For very small datasets, use label smoothing to prevent overfitting
    # Use weighted CrossEntropyLoss - more stable for small datasets
    if class_weights is not None:
        # Add label smoothing for small datasets to prevent overconfidence
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print("Using weighted CrossEntropyLoss with label smoothing (0.1) to handle class imbalance and prevent overfitting")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("Using CrossEntropyLoss with label smoothing (0.1)")
    # Different learning rates for pretrained vs new layers
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'fc'):
        # Transfer learning: lower LR for pretrained, higher for new layers
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad]
        classifier_params = [p for n, p in model.named_parameters() if 'fc' in n]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 0.0001 * learning_rate_multiplier},  # Lower LR for pretrained
            {'params': classifier_params, 'lr': 0.001 * learning_rate_multiplier}   # Higher LR for new layers
        ], weight_decay=0.0001)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.0005 * learning_rate_multiplier, weight_decay=0.0001)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    # For very small datasets, increase patience and reduce early stopping sensitivity
    # Small datasets have high variance, so we need more patience
    patience = 30  # Early stopping patience (increased for small datasets)
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
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
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/pytorch_face_model.pth')
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
        
        scheduler.step()  # CosineAnnealingWarmRestarts doesn't need loss
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/pytorch_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training history plot saved to models/pytorch_training_history.png")

def evaluate_model(model, test_loader, class_names):
    """Evaluate the model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return accuracy, report

def main():
    """Main training function"""
    print("Advanced PyTorch Face Recognition Training")
    print("=" * 50)
    
    # Check data directory
    data_path = 'data/train'
    if not os.path.exists(data_path):
        print(f"Data directory {data_path} not found")
        return
    
    try:
        # Prepare dataset
        print("Preparing dataset...")
        # Set save_extracted_faces=False by default (set to True if you want to save extracted faces)
        images, labels, class_names = prepare_dataset(data_path, save_extracted_faces=False)
        
        if len(images) == 0:
            print("No images found!")
            return
        
        # For very small datasets, use a more conservative split
        # With only 18 images, we need more training data
        # Use 80% train, 10% val, 10% test (or even simpler: 90% train, 10% test)
        if len(images) < 30:
            # Very small dataset: use 85% train, 15% test (no separate validation)
            # We'll use cross-validation or just monitor training loss
            print("Small dataset detected - using 85/15 split (no separate validation set)")
            X_train, X_test, y_train, y_test = train_test_split(
                images, labels, test_size=0.15, random_state=42, stratify=labels
            )
            # Use training set for validation during training (will monitor training loss)
            X_val, y_val = X_train, y_train
        else:
            # Normal split for larger datasets
            X_train, X_test, y_train, y_test = train_test_split(
                images, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Calculate class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print(f"\nClass distribution in training set:")
        for idx, class_name in enumerate(class_names):
            count = class_counts.get(idx, 0)
            print(f"  {class_name}: {count} samples")
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Calculate class weights to handle imbalance
        total_samples = len(y_train)
        class_weights = []
        for idx in range(len(class_names)):
            class_count = class_counts.get(idx, 1)
            # Weight is inversely proportional to class frequency
            weight = total_samples / (len(class_names) * class_count)
            class_weights.append(weight)
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        print(f"\nClass weights (to balance training):")
        for idx, class_name in enumerate(class_names):
            print(f"  {class_name}: {class_weights[idx]:.3f}")
        
        # Get transforms
        train_transform, val_transform = get_transforms()
        
        # Create datasets
        train_dataset = FaceDataset(X_train, y_train, train_transform)
        val_dataset = FaceDataset(X_val, y_val, val_transform)
        test_dataset = FaceDataset(X_test, y_test, val_transform)
        
        # Create weighted sampler for balanced training
        sample_weights = []
        for label in y_train:
            sample_weights.append(class_weights[label])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders with weighted sampling (don't use shuffle with sampler)
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Model will be created in the training loop
        model = None
        
        # Train and test in loop until accuracy improves, using progressively better models
        print("Starting adaptive training loop (will use better models if needed)...")
        print("=" * 60)
        
        # Model progression: Start with ResNet50, move to better models if needed
        model_types = [
            ("ResNet50", AdvancedFaceCNN),
            ("EfficientNet-B3", EfficientNetFaceCNN),
            ("VisionTransformer", VisionTransformerFaceCNN)
        ]
        
        best_overall_acc = 0.0
        best_model_type = None
        best_model_state = None
        all_train_losses = []
        all_train_accuracies = []
        all_val_losses = []
        all_val_accuracies = []
        model_idx = 0
        no_improvement_count = 0
        max_no_improvement = 3  # Try 3 times with same model before moving to next
        
        while model_idx < len(model_types):
            model_name, model_class = model_types[model_idx]
            print(f"\n{'='*60}")
            print(f"TRAINING WITH MODEL: {model_name}")
            print(f"{'='*60}")
            
            attempt = 0
            model_best_acc = 0.0
            
            # Try this model multiple times with different settings
            while attempt < max_no_improvement:
                attempt += 1
                print(f"\n--- Attempt {attempt}/{max_no_improvement} with {model_name} ---")
                
                # Create new model instance
                model = model_class(num_classes=len(class_names), use_pretrained=True).to(device)
                
                # Adjust learning rate for each attempt
                lr_multiplier = 0.7 + 0.6 * ((attempt - 1) % 4) / 3  # 0.7, 0.9, 1.1, 1.3
                print(f"Learning rate multiplier: {lr_multiplier:.2f}x")
                
                # Train model
                try:
                    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
                        model, train_loader, val_loader, num_epochs=150, 
                        class_weights=class_weights_tensor, learning_rate_multiplier=lr_multiplier
                    )
                    
                    # Store training history
                    all_train_losses.extend(train_losses)
                    all_train_accuracies.extend(train_accuracies)
                    all_val_losses.extend(val_losses)
                    all_val_accuracies.extend(val_accuracies)
                    
                    # Load best model from this attempt
                    if os.path.exists('models/pytorch_face_model.pth'):
                        model.load_state_dict(torch.load('models/pytorch_face_model.pth', map_location=device))
                    
                    # Evaluate on validation set
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                    
                    current_val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
                    
                    print(f"\n{model_name} Attempt #{attempt} Results:")
                    print(f"  Validation Accuracy: {current_val_acc:.2f}%")
                    print(f"  Best with {model_name}: {max(model_best_acc, current_val_acc):.2f}%")
                    print(f"  Overall Best: {max(best_overall_acc, current_val_acc):.2f}%")
                    
                    # Check if this is better
                    if current_val_acc > model_best_acc:
                        model_best_acc = current_val_acc
                        print(f"  ✓ New best for {model_name}!")
                    
                    if current_val_acc > best_overall_acc:
                        improvement = current_val_acc - best_overall_acc
                        best_overall_acc = current_val_acc
                        best_model_type = model_name
                        best_model_state = model.state_dict().copy()
                        torch.save(model.state_dict(), 'models/pytorch_face_model_best.pth')
                        print(f"  🎉 NEW OVERALL BEST! (+{improvement:.2f}%)")
                        no_improvement_count = 0  # Reset counter
                    else:
                        no_improvement_count += 1
                        print(f"  No improvement (count: {no_improvement_count}/{max_no_improvement})")
                    
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    print(f"Moving to next model...")
                    break
                
                # If we got significant improvement, try a few more times with this model
                if current_val_acc > best_overall_acc - 5.0:  # Within 5% of best
                    continue
                
                # If no improvement after attempts, move to next model
                if no_improvement_count >= max_no_improvement:
                    print(f"\n⚠️  No improvement after {max_no_improvement} attempts with {model_name}")
                    print(f"Moving to next model...")
                    break
                
                import time
                time.sleep(0.5)
            
            # Move to next model
            model_idx += 1
            
            # If we got good accuracy, we can stop
            if best_overall_acc >= 60.0:
                print(f"\n🎉 SUCCESS! Reached {best_overall_acc:.2f}% accuracy with {best_model_type}")
                break
        
        # Load and use the best model
        if best_model_state is not None:
            print(f"\n{'='*60}")
            print(f"FINAL BEST MODEL: {best_model_type}")
            print(f"Best Validation Accuracy: {best_overall_acc:.2f}%")
            print(f"{'='*60}")
            
            # Create the correct model type
            if best_model_type == "EfficientNet-B3":
                model = EfficientNetFaceCNN(num_classes=len(class_names), use_pretrained=True).to(device)
            elif best_model_type == "VisionTransformer":
                model = VisionTransformerFaceCNN(num_classes=len(class_names), use_pretrained=True).to(device)
            else:
                model = AdvancedFaceCNN(num_classes=len(class_names), use_pretrained=True).to(device)
            
            model.load_state_dict(best_model_state)
            torch.save(model.state_dict(), 'models/pytorch_face_model.pth')
            print(f"✓ Best model saved to models/pytorch_face_model.pth")
        else:
            # Fallback: use last trained model
            print(f"\n⚠️  No best model found, using last trained model")
            if os.path.exists('models/pytorch_face_model.pth'):
                model = AdvancedFaceCNN(num_classes=len(class_names), use_pretrained=True).to(device)
                model.load_state_dict(torch.load('models/pytorch_face_model.pth', map_location=device))
            else:
                print("ERROR: No model available!")
                return
        
        # Final evaluation on test set
        print(f"\n{'='*60}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'='*60}")
        model.eval()
        test_accuracy, report = evaluate_model(model, test_loader, class_names)
        
        # Note: Class names are now read from folder structure, no need to save JSON
        
        # Save training history (from all attempts)
        history = {
            'train_losses': all_train_losses,
            'train_accuracies': all_train_accuracies,
            'val_losses': all_val_losses,
            'val_accuracies': all_val_accuracies,
            'test_accuracy': test_accuracy,
            'best_validation_accuracy': best_overall_acc,
            'best_model_type': best_model_type if best_model_type else 'ResNet50',
            'attempts_made': model_idx
        }
        
        with open('models/pytorch_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training history
        plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        
        print("Training completed successfully!")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        print("Model saved to models/pytorch_face_model.pth")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()



