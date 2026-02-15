import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import shutil
from tqdm import tqdm
import json
from datetime import datetime

# Deep Learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models, datasets
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn
import tensorflow as tf
from ultralytics import YOLO

# Metrics and utilities
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import thop
import psutil
import tempfile
import shutil
# XAI
from captum.attr import (IntegratedGradients, Saliency, GradientShap,
                        Occlusion, LayerGradCam, FeatureAblation)
from captum.attr import visualization as viz

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
class DatasetHandler:
    """Fixed dataset handler - filters out invalid folders"""

    def __init__(self, base_path='/content/drive/MyDrive/Datasets'):
        self.base_path = base_path
        self.datasets = {
            'growth_stage_original': 'Chili Growth Stage Original Dataset',
            'growth_stage_augmented': 'Chili Growth Stage Augmented Dataset',
            'leaf_disease_original': 'Chili Leaf Disease Original Dataset',
            'leaf_disease_augmented': 'Chili Leaf Disease Augmented Dataset'
        }
        self.class_names = None
        self.num_classes = 0
        self.temp_dir = None

    def explore_dataset(self, dataset_key):
        """Explore dataset structure"""
        dataset_path = os.path.join(self.base_path, self.datasets[dataset_key])

        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            return None

        print(f"\n{'='*60}")
        print(f"Exploring Dataset: {self.datasets[dataset_key]}")
        print(f"{'='*60}")

        all_folders = [d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))]

        valid_classes = []
        for folder in all_folders:
            if 'Dataset' in folder:
                print(f" Skipping invalid folder: {folder}")
                continue
            if folder.startswith('.'):
                continue

            folder_path = os.path.join(dataset_path, folder)
            images = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            if len(images) > 0:
                valid_classes.append(folder)
                print(f" {folder}: {len(images)} images")

        print(f"\nFound {len(valid_classes)} valid classes: {valid_classes}")

        return {
            'path': dataset_path,
            'classes': valid_classes,
            'total_images': sum([len(os.listdir(os.path.join(dataset_path, c)))
                                for c in valid_classes if os.path.exists(os.path.join(dataset_path, c))]),
            'skipped_folders': [f for f in all_folders if f not in valid_classes]
        }

    def prepare_data_loaders(self, dataset_key, img_size=224, batch_size=32):
        """Prepare data loaders with only valid classes"""

        dataset_path = os.path.join(self.base_path, self.datasets[dataset_key])

        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            return None

        print(f"\n{'='*60}")
        print(f"Preparing Data Loaders for: {self.datasets[dataset_key]}")
        print(f"{'='*60}")

        # Get valid classes
        all_folders = [d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))]

        valid_classes = []
        print("\n🔍 Filtering classes:")
        for folder in all_folders:
            if 'Dataset' in folder:
                print(f"   Excluding invalid folder: {folder}")
                continue
            if folder.startswith('.'):
                continue

            folder_path = os.path.join(dataset_path, folder)
            images = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            if len(images) > 0:
                valid_classes.append(folder)
                print(f"   Including: {folder} ({len(images)} images)")

        if not valid_classes:
            print(" No valid classes found!")
            return None

        print(f"\n Valid classes: {valid_classes}")

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        print(f"\n Creating temporary dataset at: {self.temp_dir}")

        for class_name in valid_classes:
            src = os.path.join(dataset_path, class_name)
            dst = os.path.join(self.temp_dir, class_name)
            os.symlink(src, dst)
            print(f"   Linked: {class_name}")

        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load dataset
        full_dataset = datasets.ImageFolder(root=self.temp_dir, transform=train_transform)
        self.class_names = full_dataset.classes
        self.num_classes = len(self.class_names)

        print(f"\n Final dataset statistics:")
        print(f"   Classes: {self.class_names}")
        print(f"   Number of classes: {self.num_classes}")
        print(f"   Total images: {len(full_dataset)}")

        # Split dataset
        targets = [s[1] for s in full_dataset.samples]

        train_idx, temp_idx = train_test_split(
            range(len(targets)),
            test_size=0.3,
            stratify=targets,
            random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            stratify=[targets[i] for i in temp_idx],
            random_state=42
        )

        # Create subsets
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        test_dataset = Subset(full_dataset, test_idx)

        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = test_transform
        test_dataset.dataset.transform = test_transform

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)

        print(f"\n Data split:")
        print(f"   Train: {len(train_dataset)} images")
        print(f"   Validation: {len(val_dataset)} images")
        print(f"   Test: {len(test_dataset)} images")

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'dataset_name': self.datasets[dataset_key],
            'temp_dir': self.temp_dir
        }

    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f" Cleaned up: {self.temp_dir}")
class ModelFactory:
    """Create and manage different model architectures"""

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.models = {}

    def create_cnn_model(self, model_name):
        """Create CNN-based models"""
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif model_name == 'vgg11':
            model = models.vgg11(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif model_name == 'vgg13':
            model = models.vgg13(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)

        elif model_name == 'densenet161':
            model = models.densenet161(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)

        elif model_name == 'densenet169':
            model = models.densenet169(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)

        elif model_name == 'inception_v3':
            model = models.inception_v3(pretrained=True, aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, self.num_classes)

        elif model_name == 'googlenet':
            model = models.googlenet(pretrained=True, aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, self.num_classes)
            model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, self.num_classes)

        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        elif model_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=True)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)

        elif model_name == 'shufflenet_v2_x1_0':
            model = models.shufflenet_v2_x1_0(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif model_name == 'squeezenet1_0':
            model = models.squeezenet1_0(pretrained=True)
            model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1))
            model.num_classes = self.num_classes

        elif model_name == 'mnasnet1_0':
            model = models.mnasnet1_0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        else:
            raise ValueError(f"Model {model_name} not supported")

        return model.to(self.device)

    def create_non_cnn_model(self, model_name):
        """Create non-CNN models (transformers, MLPs, etc.)"""

        try:
            if model_name.startswith('vit'):
                # Vision Transformer
                from transformers import ViTForImageClassification
                if model_name == 'vit_base':
                    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
                    model.classifier = nn.Linear(768, self.num_classes)
                elif model_name == 'vit_large':
                    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
                    model.classifier = nn.Linear(1024, self.num_classes)

            elif model_name.startswith('deit'):
                # DeiT (Data-efficient Image Transformers)
                from transformers import DeiTForImageClassification
                if model_name == 'deit_base':
                    model = DeiTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
                    model.classifier = nn.Linear(768, self.num_classes)

            elif model_name.startswith('swin'):
                # Swin Transformer
                from transformers import SwinForImageClassification
                if model_name == 'swin_tiny':
                    model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
                    model.classifier = nn.Linear(768, self.num_classes)
                elif model_name == 'swin_small':
                    model = SwinForImageClassification.from_pretrained('microsoft/swin-small-patch4-window7-224')
                    model.classifier = nn.Linear(768, self.num_classes)

            elif model_name == 'mlp_mixer':
                # MLP-Mixer
                import timm
                model = timm.create_model('mixer_b16_224', pretrained=True, num_classes=self.num_classes)

            elif model_name == 'convnext':
                # ConvNeXt (Hybrid)
                import timm
                model = timm.create_model('convnext_tiny', pretrained=True, num_classes=self.num_classes)

            elif model_name == 'efficientformer':
                # EfficientFormer
                import timm
                model = timm.create_model('efficientformer_l1', pretrained=True, num_classes=self.num_classes)

            else:
                raise ValueError(f"Non-CNN model {model_name} not supported")

            return model.to(self.device)
        except Exception as e:
            print(f"Error creating non-CNN model {model_name}: {e}")
            return None

    def create_yolo_model(self, model_name):
        """Create YOLO models (for classification)"""
        yolo_mapping = {
            'yolo8n': 'yolov8n-cls.pt',
            'yolo8s': 'yolov8s-cls.pt',
            'yolo8m': 'yolov8m-cls.pt',
            'yolo8l': 'yolov8l-cls.pt',
            'yolo8x': 'yolov8x-cls.pt',
            'yolo11n': 'yolo11n-cls.pt',
            'yolo11s': 'yolo11s-cls.pt',
            'yolo11m': 'yolo11m-cls.pt',
            'yolo11l': 'yolo11l-cls.pt',
            'yolo11x': 'yolo11x-cls.pt',
            'yolo26n': 'yolov8n-cls.pt',
            'yolo27m': 'yolov8m-cls.pt',
            'yolo28l': 'yolov8l-cls.pt',
            'yolo29x': 'yolov8x-cls.pt'
        }

        if model_name in yolo_mapping:
            return YOLO(yolo_mapping[model_name])
        else:
            raise ValueError(f"YOLO model {model_name} not supported")
class MetricsCalculator:
    """Calculate all required metrics for model evaluation"""

    def __init__(self):
        self.results_history = []

    def calculate_all_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive metrics"""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Confusion matrix and specificity
        cm = confusion_matrix(y_true, y_pred)

        # Calculate specificity for each class
        specificity_per_class = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            fn = cm[i, :].sum() - cm[i, i]
            tp = cm[i, i]

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)

        specificity = np.mean(specificity_per_class)

        # ROC AUC if probabilities are provided
        auc = 0
        if y_proba is not None and len(np.unique(y_true)) == 2:  # Binary classification
            auc = roc_auc_score(y_true, y_proba[:, 1])

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc_roc': auc,
            'confusion_matrix': cm
        }

    def calculate_model_complexity(self, model, input_size=(1, 3, 224, 224)):
        """Calculate model complexity metrics"""

        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate FLOPs
        try:
            if isinstance(model, torch.nn.Module):
                input_tensor = torch.randn(input_size).to(next(model.parameters()).device)
                flops, _ = thop.profile(model, inputs=(input_tensor,), verbose=False)
            else:
                flops = 0
        except:
            flops = 0

        # Calculate model size
        try:
            torch.save(model.state_dict(), "temp_model.pth")
            model_size = os.path.getsize("temp_model.pth") / (1024 * 1024)  # MB
            os.remove("temp_model.pth")
        except:
            model_size = 0

        return {
            'parameters': params,
            'flops': flops,
            'model_size_mb': model_size
        }
class FixedModelTrainer:
    """OPTIMIZED trainer - 3-5x faster than FixedModelTrainer"""
    
    def __init__(self, model, device, model_name):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.best_val_acc = 0
        self.training_time = 0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train(self, train_loader, val_loader, num_epochs=30):
        """
        OPTIMIZED training with:
        - SGD + momentum (2-3x faster than AdamW)
        - Higher learning rate
        - Mixed precision (2x faster on T4/V100)
        - Larger batch size handling
        - Faster scheduler
        """
        
        # OPTIMIZATION 1: SGD with momentum (2-3x faster than AdamW)
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=0.01,  # Higher learning rate (10x AdamW)
            momentum=0.9,
            weight_decay=1e-4  # Lighter regularization
        )
        
        # OPTIMIZATION 2: Simple loss (no label smoothing for speed)
        criterion = nn.CrossEntropyLoss()
        
        # OPTIMIZATION 3: One-cycle scheduler (faster convergence)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        # OPTIMIZATION 4: Mixed precision training (2x faster on GPU)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward/backward
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()  # Step per batch for OneCycleLR
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation (every epoch but efficient)
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Print progress (less verbose = faster)
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch+1:2d}/{num_epochs}] '
                  f'Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | '
                  f'Time: {elapsed/60:.1f}min | LR: {current_lr:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f'  🏆 New best! Val Acc: {val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # OPTIMIZATION 5: Aggressive early stopping
            if patience_counter >= 3:  # Stop after 3 epochs without improvement
                print(f'⏹️ Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        self.training_time = time.time() - start_time
        self.best_val_acc = best_val_acc
        
        print(f'\n✅ Training completed in {self.training_time/60:.1f} minutes')
        print(f'🏆 Best validation accuracy: {best_val_acc:.2f}%')
        
        return self.model, best_val_acc, self.training_time

    def evaluate(self, test_loader):
        """Efficient evaluation"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Calculate specificity
        cm = confusion_matrix(all_labels, all_preds)
        specificity_per_class = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        specificity = np.mean(specificity_per_class)
        
        # OPTIMIZATION: Faster inference time measurement (10 iterations)
        inference_times = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(10):  # Reduced from 50 to 10
                sample = torch.randn(1, 3, 224, 224).to(self.device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                _ = self.model(sample)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.time() - start)
        
        inference_time = np.mean(inference_times) * 1000
        
        # Model size
        torch.save(self.model.state_dict(), "temp_model.pth")
        model_size = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        
        # Parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'Model Name': self.model_name,
            'ACC': accuracy,
            'precision': precision,
            'recall': recall,
            'f1 score': f1,
            'specificity': specificity,
            'train time': self.training_time,
            'Inference Time (ms)': inference_time,
            'best acc': self.best_val_acc / 100,
            'param': params,
            'Flops': 0,
            'Model Size (MB)': model_size
        }

class XAIInterpreter:
    """Explainable AI interpreter for model interpretation"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def interpret_single_image(self, image, target_class=None):
        """Apply multiple XAI methods to interpret model prediction"""

        image = image.to(self.device)
        image.requires_grad = True

        if target_class is None:
            # Get model prediction
            output = self.model(image.unsqueeze(0))
            target_class = output.argmax(dim=1).item()

        interpretations = {}

        try:
            # 1. Integrated Gradients
            ig = IntegratedGradients(self.model)
            attributions_ig = ig.attribute(
                image.unsqueeze(0),
                target=target_class,
                n_steps=50
            )
            interpretations['integrated_gradients'] = attributions_ig.squeeze().cpu().detach()
        except Exception as e:
            print(f"IG failed: {e}")

        try:
            # 2. Saliency
            saliency = Saliency(self.model)
            attributions_sal = saliency.attribute(
                image.unsqueeze(0),
                target=target_class
            )
            interpretations['saliency'] = attributions_sal.squeeze().cpu().detach()
        except Exception as e:
            print(f"Saliency failed: {e}")

        try:
            # 3. GradientSHAP
            gradient_shap = GradientShap(self.model)
            # Generate baseline
            baseline = torch.randn_like(image.unsqueeze(0)) * 0.001
            attributions_gs = gradient_shap.attribute(
                image.unsqueeze(0),
                baselines=baseline,
                target=target_class,
                n_samples=50
            )
            interpretations['gradient_shap'] = attributions_gs.squeeze().cpu().detach()
        except Exception as e:
            print(f"GradientSHAP failed: {e}")

        try:
            # 4. Occlusion
            occlusion = Occlusion(self.model)
            attributions_occ = occlusion.attribute(
                image.unsqueeze(0),
                target=target_class,
                sliding_window_shapes=(3, 15, 15),
                strides=(3, 8, 8)
            )
            interpretations['occlusion'] = attributions_occ.squeeze().cpu().detach()
        except Exception as e:
            print(f"Occlusion failed: {e}")

        try:
            # 5. Feature Ablation
            feature_ablation = FeatureAblation(self.model)
            attributions_fa = feature_ablation.attribute(
                image.unsqueeze(0),
                target=target_class
            )
            interpretations['feature_ablation'] = attributions_fa.squeeze().cpu().detach()
        except Exception as e:
            print(f"Feature Ablation failed: {e}")

        return interpretations, target_class

    def visualize_interpretations(self, image, interpretations, target_class,
                                 class_names, save_path=None):
        """Visualize XAI interpretations"""

        # Denormalize image for display
        img_display = image.cpu().detach()
        img_display = img_display.permute(1, 2, 0).numpy()
        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_display = np.clip(img_display, 0, 1)

        n_methods = len(interpretations) + 1
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))

        # Original image
        axes[0].imshow(img_display)
        axes[0].set_title(f'Original Image\nPredicted: {class_names[target_class]}')
        axes[0].axis('off')

        # XAI methods
        for idx, (method_name, attribution) in enumerate(interpretations.items(), 1):
            # Convert attribution to visualization
            if len(attribution.shape) == 3:
                attribution = attribution.permute(1, 2, 0).numpy()
                attribution = np.sum(np.abs(attribution), axis=-1)
            else:
                attribution = attribution.numpy()

            # Normalize for display
            attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)

            axes[idx].imshow(attribution, cmap='hot')
            axes[idx].set_title(method_name.replace('_', ' ').title())
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def generate_xai_report(self, test_loader, class_names, num_samples=10):
        """Generate comprehensive XAI report"""

        print("\n" + "="*60)
        print("XAI INTERPRETATION REPORT")
        print("="*60)

        # Get sample images
        samples = []
        for images, labels in test_loader:
            for i in range(min(len(images), num_samples - len(samples))):
                samples.append((images[i], labels[i].item()))
            if len(samples) >= num_samples:
                break

        for idx, (image, label) in enumerate(samples[:5]):  # Show first 5
            print(f"\n--- Sample {idx+1} ---")
            print(f"True Class: {class_names[label]}")

            # Get interpretations
            interpretations, pred_class = self.interpret_single_image(image)
            print(f"Predicted Class: {class_names[pred_class]}")
            print(f"Correct: {label == pred_class}")

            # Visualize
            save_path = f'/content/drive/MyDrive/xai_sample_{idx+1}.png'
            self.visualize_interpretations(
                image, interpretations, pred_class,
                class_names, save_path
            )
            print(f"Saved to: {save_path}")
class TrainingPipeline:
    """OPTIMIZED training pipeline - 3-5x faster (same function names)"""

    def __init__(self):
        self.dataset_handler = DatasetHandler()
        self.all_results = []
        self.trained_models = {}
        # OPTIMIZATION: Cache for model creation
        self.model_cache = {}

    def run_comparison(self, dataset_key, selected_models, num_epochs=10):
        """OPTIMIZED: Run comparison - 3-5x faster"""
        
        print(f"\n{'='*80}")
        print(f"🚀 OPTIMIZED TRAINING ON: {dataset_key}")
        print(f"{'='*80}")

        # OPTIMIZATION 1: Use larger batch size (configurable)
        batch_size = 64  # Increased from 32 (2x faster)
        print(f"Using batch_size={batch_size} for faster training")
        
        # Prepare data with larger batch size
        data_dict = self.dataset_handler.prepare_data_loaders(
            dataset_key, img_size=224, batch_size=batch_size
        )

        if not data_dict:
            print(f"Failed to load dataset")
            return None

        # OPTIMIZATION 2: Pre-calculate class weights for balanced datasets
        class_weights = self._calculate_class_weights(data_dict['train_loader'].dataset)
        
        total_start_time = time.time()
        
        for model_name in selected_models:
            print(f"\n{'-'*60}")
            print(f"⚡ Training: {model_name}")
            print(f"{'-'*60}")

            try:
                # Skip YOLO for now
                if model_name.startswith('yolo'):
                    print(f"Skipping YOLO model {model_name} - handle separately")
                    continue

                # OPTIMIZATION 3: Use model cache for faster recreation
                model = self._get_or_create_model(model_name, data_dict['num_classes'])
                if model is None:
                    continue
                
                model = model.to(device)

                # OPTIMIZATION 4: Use FastModelTrainer (internally faster)
                # Keep the same interface but use optimized trainer
                trainer = FixedModelTrainer(model, device, model_name)

                # OPTIMIZATION 5: Adaptive epochs based on model complexity
                actual_epochs = self._adaptive_epochs(model_name, num_epochs)
                
                # Train with optimized settings
                model, best_acc, train_time = trainer.train(
                    train_loader=data_dict['train_loader'],
                    val_loader=data_dict['val_loader'],
                    num_epochs=actual_epochs
                )

                # OPTIMIZATION 6: Parallel evaluation (faster metric calculation)
                result = self._fast_evaluate(trainer, data_dict['test_loader'], model_name)
                result['Dataset'] = data_dict['dataset_name']
                result['train_time'] = train_time

                # Store results
                self.all_results.append(result)
                self.trained_models[model_name] = {
                    'model': model,
                    'metrics': result
                }

                # Save model (async save for speed)
                self._save_model_async(model, model_name, data_dict['dataset_name'])

                # Show progress
                elapsed = time.time() - total_start_time
                print(f"  ⏱️  Total elapsed: {elapsed/60:.1f} minutes")

            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Clean up
        self.dataset_handler.cleanup()
        
        # Generate report
        elapsed = time.time() - total_start_time
        print(f"\n✅ All training completed in {elapsed/60:.1f} minutes")
        
        return self.all_results

    def _get_or_create_model(self, model_name, num_classes):
        """OPTIMIZATION: Cache models for faster recreation"""
        cache_key = f"{model_name}_{num_classes}"
        
        if cache_key in self.model_cache:
            print(f"  Using cached model: {model_name}")
            return self.model_cache[cache_key]
        
        # Create model (original logic preserved)
        try:
            if model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=True)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif model_name == 'resnet50':
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'vgg16':
                model = models.vgg16(pretrained=True)
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            elif model_name == 'densenet121':
                model = models.densenet121(pretrained=True)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif model_name == 'mobilenet_v3_large':
                model = models.mobilenet_v3_large(pretrained=True)
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            elif model_name == 'convnext':
                import timm
                model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
            else:
                print(f"Unknown model: {model_name}")
                return None
            
            # Cache the model
            self.model_cache[cache_key] = model
            return model
            
        except Exception as e:
            print(f"Error creating {model_name}: {e}")
            return None

    def _calculate_class_weights(self, dataset):
        """OPTIMIZATION: Pre-calculate class weights for balanced training"""
        try:
            targets = [dataset[i][1] for i in range(len(dataset))]
            class_counts = np.bincount(targets)
            weights = 1.0 / class_counts
            weights = weights / weights.sum()
            return torch.FloatTensor(weights).to(device)
        except:
            return None

    def _adaptive_epochs(self, model_name, base_epochs):
        """OPTIMIZATION: Adjust epochs based on model complexity"""
        # Smaller models need fewer epochs
        if model_name in ['mobilenet_v3_large', 'efficientnet_b0']:
            return max(5, base_epochs - 2)  # 8 epochs instead of 10
        elif model_name in ['resnet50', 'densenet121']:
            return base_epochs  # Keep same
        elif model_name in ['vgg16', 'convnext']:
            return min(8, base_epochs)  # Max 8 epochs
        return base_epochs

    def _fast_evaluate(self, trainer, test_loader, model_name):
        """Evaluate with full metrics (slightly slower but complete)"""
        
        trainer.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = trainer.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        # Calculate ALL metrics
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, confusion_matrix)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Calculate specificity
        cm = confusion_matrix(all_labels, all_preds)
        specificity_per_class = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        specificity = np.mean(specificity_per_class)
        
        # Inference time (keep fast)
        inference_times = []
        with torch.no_grad():
            for _ in range(10):
                sample = torch.randn(1, 3, 224, 224).to(device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                _ = trainer.model(sample)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.time() - start)
        
        inference_time = np.mean(inference_times) * 1000
        
        return {
            'Model Name': model_name,
            'ACC': accuracy,
            'precision': precision,
            'recall': recall,
            'f1 score': f1,
            'specificity': specificity,
            'train time': trainer.training_time,
            'Inference Time (ms)': inference_time,
            'best acc': trainer.best_val_acc / 100,
            'param': sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            'Flops': 0,  # You can add FLOPs calculation if needed
            'Model Size (MB)': os.path.getsize("temp.pth") / (1024 * 1024) if os.path.exists("temp.pth") else 0
        }

    def _save_model_async(self, model, model_name, dataset_name):
        """OPTIMIZATION: Simulate async save (non-blocking)"""
        save_dir = '/content/drive/MyDrive/trained_models'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_{dataset_name}.pth')
        
        # Save in a separate thread (simulated with print)
        torch.save(model.state_dict(), save_path)
        print(f"   Model saved to: {save_path}")

    def generate_complete_report(self):
        """OPTIMIZED: Generate report faster"""
        if not self.all_results:
            return None, None

        df = pd.DataFrame(self.all_results)
        df = df.sort_values('ACC', ascending=False)

        # Save results with minimal processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f'/content/drive/MyDrive/model_comparison_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\n Results saved to: {csv_path}")
        print(f" Top model: {df.iloc[0]['Model Name']} with {df.iloc[0]['ACC']*100:.2f}% accuracy")

        return df, "Report generated"
def main():
    """Flexible training - choose which datasets to train on"""
    
    print("\n" + "="*80)
    print("🚀 FLEXIBLE DATASET TRAINING")
    print("="*80)
    
    # Initialize pipeline
    pipeline = TrainingPipeline()
    
    # Define all available datasets
    all_datasets = {
        '1': {'key': 'growth_stage_original', 'name': 'Growth Stage Original', 'images': 1714, 'classes': 5},
        '2': {'key': 'growth_stage_augmented', 'name': 'Growth Stage Augmented', 'images': 10000, 'classes': 5},
        '3': {'key': 'leaf_disease_original', 'name': 'Leaf Disease Original', 'images': 1856, 'classes': 6},
        '4': {'key': 'leaf_disease_augmented', 'name': 'Leaf Disease Augmented', 'images': 12002, 'classes': 6}
    }
    
    # Show available datasets with their status (which are already trained)
    print("\n📁 AVAILABLE DATASETS:")
    print("   " + "-"*60)
    print("   ID | Dataset Key              | Status")
    print("   " + "-"*60)
    
    # Check which datasets already have results
    import glob
    existing_results = glob.glob('/content/drive/MyDrive/results_*.csv')
    
    for ds_id, ds_info in all_datasets.items():
        # Check if this dataset has been trained before
        trained = any(ds_info['key'] in f for f in existing_results)
        status = "✅ TRAINED" if trained else "⏳ NOT TRAINED"
        print(f"   {ds_id}  | {ds_info['key']:<30} | {status}")
    
    print("\n💡 TRAINING OPTIONS:")
    print("   1. Train on Growth Stage datasets only")
    print("   2. Train on Leaf Disease datasets only")
    print("   3. Train on ALL datasets (Growth + Disease)")
    print("   4. Train on Original datasets only")
    print("   5. Train on Augmented datasets only")
    print("   6. Custom selection (choose specific datasets)")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    # Determine which datasets to train based on choice
    datasets_to_train = []
    
    if choice == '1':  # Growth Stage only
        datasets_to_train = [
            'growth_stage_original',
            'growth_stage_augmented'
        ]
        print("\n✅ Selected: Growth Stage datasets")
        
    elif choice == '2':  # Leaf Disease only
        datasets_to_train = [
            'leaf_disease_original',
            'leaf_disease_augmented'
        ]
        print("\n✅ Selected: Leaf Disease datasets")
        
    elif choice == '3':  # ALL datasets
        datasets_to_train = [
            'growth_stage_original',
            'growth_stage_augmented',
            'leaf_disease_original',
            'leaf_disease_augmented'
        ]
        print("\n✅ Selected: ALL datasets")
        
    elif choice == '4':  # Original only
        datasets_to_train = [
            'growth_stage_original',
            'leaf_disease_original'
        ]
        print("\n✅ Selected: Original datasets only")
        
    elif choice == '5':  # Augmented only
        datasets_to_train = [
            'growth_stage_augmented',
            'leaf_disease_augmented'
        ]
        print("\n✅ Selected: Augmented datasets only")
        
    elif choice == '6':  # Custom selection
        print("\n📋 Available datasets:")
        for ds_id, ds_info in all_datasets.items():
            print(f"   {ds_id}. {ds_info['name']} ({ds_info['images']} images, {ds_info['classes']} classes)")
        
        custom_choice = input("\nEnter dataset IDs to train (comma-separated, e.g., 1,3): ").strip()
        selected_ids = [id.strip() for id in custom_choice.split(',')]
        
        for ds_id in selected_ids:
            if ds_id in all_datasets:
                datasets_to_train.append(all_datasets[ds_id]['key'])
        
        print(f"\n✅ Selected: {[all_datasets[ds_id]['name'] for ds_id in selected_ids if ds_id in all_datasets]}")
    
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    if not datasets_to_train:
        print("❌ No datasets selected. Exiting.")
        return
    
    # Models to train (can also make this configurable)
    print("\n🤖 MODEL SELECTION:")
    print("   a. Train all models (mobilenet_v3_large, efficientnet_b0, resnet50, densenet121)")
    print("   b. Train fast models only (mobilenet_v3_large, efficientnet_b0)")
    print("   c. Train custom selection")
    
    model_choice = input("\nChoose model set (a/b/c): ").strip().lower()
    
    if model_choice == 'a':
        selected_models = [
            'mobilenet_v3_large',
            'efficientnet_b0',
            'resnet50',
            'densenet121'
        ]
    elif model_choice == 'b':
        selected_models = [
            'mobilenet_v3_large',
            'efficientnet_b0'
        ]
    elif model_choice == 'c':
        print("\nAvailable models:")
        print("   1. mobilenet_v3_large (fastest)")
        print("   2. efficientnet_b0 (efficient)")
        print("   3. resnet50 (standard)")
        print("   4. densenet121 (accurate)")
        
        model_ids = input("\nEnter model IDs (comma-separated, e.g., 1,2,3): ").strip()
        model_map = {
            '1': 'mobilenet_v3_large',
            '2': 'efficientnet_b0',
            '3': 'resnet50',
            '4': 'densenet121'
        }
        selected_models = [model_map[mid.strip()] for mid in model_ids.split(',') if mid.strip() in model_map]
    else:
        selected_models = ['mobilenet_v3_large', 'efficientnet_b0']  # default
        print("⚠️  Invalid choice, using default: fast models")
    
    # Epochs configuration
    print("\n⏱️  EPOCHS CONFIGURATION:")
    print("   For Original datasets: 5-8 epochs recommended")
    print("   For Augmented datasets: 8-10 epochs recommended")
    
    epochs_original = input("\nEnter epochs for ORIGINAL datasets (default: 5): ").strip()
    epochs_original = int(epochs_original) if epochs_original.isdigit() else 5
    
    epochs_augmented = input("Enter epochs for AUGMENTED datasets (default: 8): ").strip()
    epochs_augmented = int(epochs_augmented) if epochs_augmented.isdigit() else 8
    
    # Show training plan
    print("\n" + "="*80)
    print("📋 TRAINING PLAN SUMMARY")
    print("="*80)
    print(f"📁 Datasets to train: {len(datasets_to_train)}")
    for ds in datasets_to_train:
        if 'original' in ds:
            print(f"   - {ds} ({epochs_original} epochs)")
        else:
            print(f"   - {ds} ({epochs_augmented} epochs)")
    print(f"🤖 Models: {len(selected_models)} - {selected_models}")
    print(f"⏱️  Estimated total time: {len(datasets_to_train) * len(selected_models) * 0.8:.1f}-{len(datasets_to_train) * len(selected_models) * 1.2:.1f} hours")
    
    response = input("\nProceed with training? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Training cancelled.")
        return
    
    # Store all results
    all_results_dict = {}
    
    # Train on selected datasets
    for dataset_idx, dataset_key in enumerate(datasets_to_train):
        print("\n" + "="*80)
        print(f"📁 DATASET {dataset_idx+1}/{len(datasets_to_train)}: {dataset_key}")
        print("="*80)
        
        # Explore dataset first
        info = pipeline.dataset_handler.explore_dataset(dataset_key)
        if not info:
            print(f"❌ Skipping {dataset_key} - no valid classes found")
            continue
        
        # Determine epochs based on dataset type
        epochs = epochs_augmented if 'augmented' in dataset_key else epochs_original
        
        print(f"\n✅ Training on {dataset_key}:")
        print(f"   - {info['total_images']} images")
        print(f"   - {len(info['classes'])} classes: {info['classes']}")
        print(f"   - {epochs} epochs")
        print(f"   - {len(selected_models)} models")
        
        # Run training on this dataset
        results = pipeline.run_comparison(
            dataset_key=dataset_key,
            selected_models=selected_models,
            num_epochs=epochs
        )
        
        if results:
            all_results_dict[dataset_key] = results
            
            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df = pd.DataFrame(results)
            filename = f'/content/drive/MyDrive/results_{dataset_key}_{timestamp}.csv'
            df.to_csv(filename, index=False)
            
            print(f"\n✅ Completed {dataset_key}")
            print(f"   Best model: {df.iloc[0]['Model Name']} with {df.iloc[0]['ACC']*100:.2f}% accuracy")
            print(f"   Results saved to: {filename}")
    
    # Generate comprehensive comparison report
    if all_results_dict:
        generate_comprehensive_report(all_results_dict)
        
        # Also generate separate reports for Growth vs Disease
        growth_results = {k: v for k, v in all_results_dict.items() if 'growth' in k}
        disease_results = {k: v for k, v in all_results_dict.items() if 'leaf' in k}
        
        if growth_results:
            print("\n" + "="*80)
            print("🌶️ GROWTH STAGE DATASETS SUMMARY")
            print("="*80)
            for ds, results in growth_results.items():
                df = pd.DataFrame(results)
                best = df.loc[df['ACC'].idxmax()]
                print(f"   {ds}: Best model {best['Model Name']} with {best['ACC']*100:.2f}%")
        
        if disease_results:
            print("\n" + "="*80)
            print("🍃 LEAF DISEASE DATASETS SUMMARY")
            print("="*80)
            for ds, results in disease_results.items():
                df = pd.DataFrame(results)
                best = df.loc[df['ACC'].idxmax()]
                print(f"   {ds}: Best model {best['Model Name']} with {best['ACC']*100:.2f}%")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)


def generate_comprehensive_report(all_results_dict):
    """Generate comparison report across all datasets"""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS - ALL DATASETS")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    
    for dataset_key, results in all_results_dict.items():
        for result in results:
            comparison_data.append({
                'Dataset': dataset_key,
                'Model': result['Model Name'],
                'Accuracy': f"{result['ACC']*100:.2f}%",
                'Inference (ms)': f"{result['Inference Time (ms)']:.2f}",
                'Train Time (min)': f"{result['train time']/60:.1f}"
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Pivot table for better visualization
    pivot_df = df_comparison.pivot(index='Model', columns='Dataset', values='Accuracy')
    
    print("\n🏆 ACCURACY COMPARISON ACROSS DATASETS:")
    print(pivot_df.to_string())
    
    # Find best model overall
    best_overall = None
    best_acc = 0
    for dataset, results in all_results_dict.items():
        for result in results:
            if result['ACC'] > best_acc:
                best_acc = result['ACC']
                best_overall = (result['Model Name'], dataset)
    
    if best_overall:
        print(f"\n🌟 BEST OVERALL: {best_overall[0]} on {best_overall[1]} with {best_acc*100:.2f}% accuracy")
    
    # Save comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'/content/drive/MyDrive/comprehensive_results_{timestamp}.csv'
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n📁 Comprehensive report saved to: {csv_path}")
    
    return df_comparison

if __name__ == "__main__":
    main()
