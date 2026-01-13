# Module 9: ML Model Development and Training

## Overview
This module covers ML model development best practices, training pipelines, hyperparameter tuning, experiment tracking, and automated training workflows.

## 1. Model Architecture Design

### Design Principles
- **Modularity**: Separate components (data loading, model, training, evaluation)
- **Configurability**: Use configuration files for hyperparameters
- **Reproducibility**: Set random seeds, version control
- **Scalability**: Design for distributed training
- **Maintainability**: Clean, documented code

### Project Structure
```
plant_phenotyping/
├── config/
│   ├── config.yaml
│   └── model_configs/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── plant_classifier.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── evaluation/
│       └── evaluator.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── tests/
├── requirements.txt
└── README.md
```

## 2. Training Pipelines

### PyTorch Training Pipeline
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

class Trainer:
    """Training pipeline for plant classification"""
    
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
    
    def _build_model(self):
        """Build model from config"""
        from src.models.plant_classifier import PlantClassifier
        return PlantClassifier(**self.config['model']).to(self.device)
    
    def _build_criterion(self):
        """Build loss function"""
        if self.config['training']['loss'] == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.config['training']['loss'] == 'focal':
            return FocalLoss()
    
    def _build_optimizer(self):
        """Build optimizer"""
        return Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        return ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    
    def train_epoch(self, train_loader: DataLoader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100 * correct / total
        }
    
    def validate(self, val_loader: DataLoader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100 * correct / total
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Complete training loop"""
        best_val_acc = 0
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
        
        return self.model
```

## 3. Hyperparameter Tuning

### Grid Search
```python
from itertools import product

def grid_search(config_space, train_fn):
    """Grid search for hyperparameters"""
    best_score = 0
    best_params = None
    
    # Generate all combinations
    keys = config_space.keys()
    values = config_space.values()
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        
        # Train with these parameters
        score = train_fn(params)
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score
```

### Random Search
```python
import random

def random_search(config_space, train_fn, n_trials=50):
    """Random search for hyperparameters"""
    best_score = 0
    best_params = None
    
    for _ in range(n_trials):
        # Sample random parameters
        params = {}
        for key, values in config_space.items():
            params[key] = random.choice(values)
        
        # Train with these parameters
        score = train_fn(params)
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score
```

### Optuna for Advanced Tuning
```python
import optuna

def objective(trial):
    """Objective function for Optuna"""
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    
    # Train model
    config = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'dropout': dropout
    }
    
    model = train_model(config)
    score = evaluate_model(model)
    
    return score

# Create study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get best parameters
best_params = study.best_params
best_score = study.best_value
```

## 4. Experiment Tracking

### MLflow Integration
```python
import mlflow
import mlflow.pytorch

def train_with_mlflow(config, train_loader, val_loader):
    """Train with MLflow tracking"""
    mlflow.set_experiment("plant_phenotyping")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        
        # Train model
        trainer = Trainer(config)
        model = trainer.train(train_loader, val_loader)
        
        # Evaluate
        metrics = trainer.validate(val_loader)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Log artifacts
        mlflow.log_artifact("config.yaml")
        
        return model, metrics
```

### Weights & Biases Integration
```python
import wandb

def train_with_wandb(config, train_loader, val_loader):
    """Train with W&B tracking"""
    wandb.init(
        project="plant-phenotyping",
        config=config
    )
    
    trainer = Trainer(config)
    
    for epoch in range(config['epochs']):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy']
        })
    
    # Log model
    wandb.log_model(model, "plant_classifier")
    
    wandb.finish()
```

## 5. Model Versioning

### Model Registry
```python
import json
from pathlib import Path
from datetime import datetime

class ModelRegistry:
    """Model versioning and registry"""
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
    
    def register_model(self, model_path: str, metadata: dict):
        """Register model with metadata"""
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_info = {
            'model_id': model_id,
            'model_path': model_path,
            'created_at': datetime.now().isoformat(),
            **metadata
        }
        
        # Save metadata
        metadata_path = self.registry_path / f"{model_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return model_id
    
    def get_model(self, model_id: str):
        """Get model metadata"""
        metadata_path = self.registry_path / f"{model_id}_metadata.json"
        with open(metadata_path) as f:
            return json.load(f)
    
    def list_models(self):
        """List all registered models"""
        models = []
        for metadata_file in self.registry_path.glob("*_metadata.json"):
            with open(metadata_file) as f:
                models.append(json.load(f))
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
```

## 6. Automated Model Training

### Training Script
```python
# scripts/train.py
import argparse
import yaml
from src.training.trainer import Trainer
from src.data.dataset import PlantDataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, default='./models')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load data
    train_dataset = PlantDataset(f"{args.data_path}/train")
    val_dataset = PlantDataset(f"{args.data_path}/val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Train
    trainer = Trainer(config)
    model = trainer.train(train_loader, val_loader)
    
    # Save model
    model_path = f"{args.output_path}/plant_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
```

## 7. Model Evaluation

### Comprehensive Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(self, test_loader, class_names):
        """Evaluate model on test set"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Classification report
        report = classification_report(
            all_labels, all_preds, target_names=class_names
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm, class_names)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        return metrics, all_probs
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
```

## 8. Best Practices

1. **Version Control**: Track code, configs, and data versions
2. **Reproducibility**: Set random seeds, document environment
3. **Experimentation**: Use experiment tracking tools
4. **Validation**: Use proper train/val/test splits
5. **Early Stopping**: Prevent overfitting
6. **Checkpointing**: Save models during training
7. **Monitoring**: Monitor training metrics and system resources
8. **Documentation**: Document model architecture and training process

## Next Steps

Proceed to [Module 10: Model Deployment and Serving](10-model-deployment-serving.md) to learn about deploying models to production.

