# Module 2: Deep Learning for Plant Image Analysis

## Overview
This module covers deep learning techniques specifically for plant image analysis, including CNNs, transfer learning, segmentation, detection, and disease classification. You'll learn to build production-ready models using PyTorch and TensorFlow.

## 1. CNN Architectures for Plant Classification

### Basic CNN for Plant Classification

#### PyTorch Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PlantClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(PlantClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

#### TensorFlow Implementation
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_plant_classifier(num_classes=10):
    model = keras.Sequential([
        # Conv block 1
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        # Conv block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        # Conv block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        # FC layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

## 2. Transfer Learning for Plant Datasets

### Why Transfer Learning?
- Limited plant-specific datasets
- Pre-trained models on ImageNet provide good features
- Faster training and better performance
- Less data required

### PyTorch Transfer Learning
```python
import torchvision.models as models
from torchvision import transforms

def create_transfer_learning_model(num_classes, model_name='resnet50'):
    """Create a transfer learning model for plant classification"""
    
    # Load pre-trained model
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    elif model_name == 'vision_transformer':
        model = models.vit_b_16(pretrained=True)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
    
    return model

# Data augmentation for plant images
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### TensorFlow Transfer Learning
```python
def create_transfer_learning_model_tf(num_classes, base_model_name='ResNet50'):
    """Create transfer learning model with TensorFlow"""
    
    # Load base model
    if base_model_name == 'ResNet50':
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
])
```

### Fine-tuning Strategy
```python
def fine_tune_model(model, num_epochs_frozen=10, num_epochs_unfrozen=20):
    """Fine-tuning strategy: train head first, then unfreeze"""
    
    # Phase 1: Train only the head
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Train for num_epochs_frozen epochs
    # ... training loop ...
    
    # Phase 2: Unfreeze and fine-tune
    for param in model.base_model.parameters():
        param.requires_grad = True
    
    # Use lower learning rate for base model
    optimizer = torch.optim.Adam([
        {'params': model.base_model.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ])
    
    # Train for num_epochs_unfrozen epochs
    # ... training loop ...
```

## 3. Segmentation for Plant Parts

### U-Net for Plant Segmentation

#### PyTorch U-Net
```python
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._conv_block(n_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.dec4 = self._up_conv_block(1024, 512)
        self.dec3 = self._up_conv_block(512, 256)
        self.dec2 = self._up_conv_block(256, 128)
        self.dec1 = self._up_conv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        
        return self.final(dec1)
```

### Segmentation Loss Functions
```python
class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combine Dice and Cross-Entropy loss"""
    def __init__(self, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + (1 - self.dice_weight) * ce
```

## 4. Object Detection for Plant Counting

### YOLO for Plant Detection
```python
# Using YOLOv5 from Ultralytics
from ultralytics import YOLO

def train_plant_detector(data_yaml, epochs=100):
    """Train YOLO model for plant detection"""
    model = YOLO('yolov5s.pt')  # Start with pre-trained YOLOv5
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0
    )
    
    return model

# Custom YOLO implementation
class PlantDetector(nn.Module):
    """Custom YOLO-style detector for plants"""
    def __init__(self, num_classes=1):
        super(PlantDetector, self).__init__()
        # Backbone
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, (5 + num_classes) * 9, 1)  # 9 anchors, 5 box params + classes
        )
    
    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        return detections
```

### Faster R-CNN for Plant Detection
```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_plant_detector(num_classes=2):  # background + plant
    """Create Faster R-CNN for plant detection"""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
```

## 5. Disease Detection and Classification

### Multi-Class Disease Classification
```python
class DiseaseClassifier(nn.Module):
    """Classifier for plant disease detection"""
    def __init__(self, num_diseases=10):
        super(DiseaseClassifier, self).__init__()
        
        # Use EfficientNet as backbone
        from efficientnet_pytorch import EfficientNet
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        num_features = self.backbone._fc.in_features
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_diseases)
        )
        
        self.backbone._fc = nn.Identity()
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```

### Attention Mechanism for Disease Localization
```python
class AttentionDiseaseDetector(nn.Module):
    """Disease detector with attention mechanism"""
    def __init__(self, num_classes=10):
        super(AttentionDiseaseDetector, self).__init__()
        
        # Backbone
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Attention module
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        attention_map = self.attention(features)
        attended_features = features * attention_map
        output = self.classifier(attended_features)
        return output, attention_map
```

## 6. Multi-Task Learning for Phenotyping

### Multi-Task Model
```python
class MultiTaskPhenotypingModel(nn.Module):
    """Multi-task model for simultaneous trait prediction"""
    def __init__(self):
        super(MultiTaskPhenotypingModel, self).__init__()
        
        # Shared backbone
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Task-specific heads
        self.leaf_area_head = nn.Linear(2048, 1)
        self.height_head = nn.Linear(2048, 1)
        self.disease_head = nn.Linear(2048, 5)  # 5 disease classes
        self.count_head = nn.Linear(2048, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Task predictions
        leaf_area = self.leaf_area_head(features)
        height = self.height_head(features)
        disease = self.disease_head(features)
        count = self.count_head(features)
        
        return {
            'leaf_area': leaf_area,
            'height': height,
            'disease': disease,
            'count': count
        }

# Multi-task loss
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights or {'leaf_area': 1.0, 'height': 1.0, 
                                              'disease': 2.0, 'count': 1.0}
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        total_loss = 0
        
        # Regression tasks
        total_loss += self.task_weights['leaf_area'] * self.mse_loss(
            predictions['leaf_area'], targets['leaf_area']
        )
        total_loss += self.task_weights['height'] * self.mse_loss(
            predictions['height'], targets['height']
        )
        total_loss += self.task_weights['count'] * self.mse_loss(
            predictions['count'], targets['count']
        )
        
        # Classification task
        total_loss += self.task_weights['disease'] * self.ce_loss(
            predictions['disease'], targets['disease']
        )
        
        return total_loss
```

## 7. Training Pipeline

### Complete Training Script
```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
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
        val_loss = 0
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
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model
```

## 8. Model Evaluation

### Evaluation Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_model(model, test_loader, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    return all_preds, all_labels
```

## 9. Practical Exercises

### Exercise 1: Plant Classification
1. Load a plant dataset (e.g., PlantNet, Plant Village)
2. Create a transfer learning model
3. Train and evaluate
4. Deploy the model

### Exercise 2: Leaf Segmentation
1. Create a U-Net model
2. Train on leaf segmentation dataset
3. Evaluate using IoU and Dice coefficient
4. Visualize segmentation results

### Exercise 3: Disease Detection
1. Build a disease classifier
2. Implement attention mechanism
3. Visualize attention maps
4. Compare with baseline model

## Next Steps

Continue to [Module 3: Advanced Plant Phenotyping Techniques](03-advanced-phenotyping.md) for time-series analysis, 3D reconstruction, and production deployment.

