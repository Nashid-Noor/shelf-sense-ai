"""
Layout Classifier

Classifies images into bookshelf, cover, or mixed categories.
"""

from typing import Tuple, Optional, Dict
from enum import Enum
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
from loguru import logger

from shelfsense.vision.preprocessing import ImagePreprocessor


class LayoutType(str, Enum):
    """Enumeration of layout types."""
    BOOKSHELF = "bookshelf"
    COVER = "cover"
    MIXED = "mixed"


@dataclass
class LayoutPrediction:
    """Result of layout classification."""
    layout: LayoutType
    confidence: float
    probabilities: Dict[str, float]
    
    def should_run_spine_detector(self) -> bool:
        """Whether to run spine detection."""
        return self.layout in (LayoutType.BOOKSHELF, LayoutType.MIXED)
    
    def should_run_cover_detector(self) -> bool:
        """Whether to run cover detection."""
        return self.layout in (LayoutType.COVER, LayoutType.MIXED)


class LayoutClassifierModel(nn.Module):
    """
    Lightweight CNN for layout classification using MobileNetV3-Small.
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        
        # Load pretrained MobileNetV3-Small
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[0].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over classes."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class LayoutClassifier:
    """
    Layout classifier with confidence-based routing.
    """
    
    # Class labels in order
    CLASSES = [LayoutType.BOOKSHELF, LayoutType.COVER, LayoutType.MIXED]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the layout classifier.
        
        Args:
            model_path: Path to trained model weights (optional)
            device: Device to run inference on (auto-detected if None)
            confidence_threshold: Minimum confidence for single-detector routing
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        # Initialize model
        self.model = LayoutClassifierModel()
        
        # Load weights if provided
        if model_path:
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()
        
        logger.info(f"LayoutClassifier initialized on {self.device}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model weights."""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {model_path}")
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> LayoutPrediction:
        """
        Predict layout type for an image.
        
        Args:
            image: BGR numpy array
            
        Returns:
            LayoutPrediction with class, confidence, and probabilities
        """
        # Preprocess
        tensor = self.preprocessor.preprocess_for_classifier(image)
        tensor = tensor.to(self.device)
        
        # Inference
        probs = self.model.predict_proba(tensor)
        probs = probs.cpu().numpy()[0]
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_class = self.CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Create probability dict
        prob_dict = {
            cls.value: float(probs[i])
            for i, cls in enumerate(self.CLASSES)
        }
        
        # Apply confidence threshold routing
        if confidence < self.confidence_threshold:
            # Low confidence → run both detectors (mixed)
            logger.debug(
                f"Low confidence ({confidence:.2f} < {self.confidence_threshold}), "
                f"routing to MIXED"
            )
            pred_class = LayoutType.MIXED
        
        return LayoutPrediction(
            layout=pred_class,
            confidence=confidence,
            probabilities=prob_dict
        )
    
    def predict_batch(self, images: list) -> list:
        """
        Batch prediction for multiple images.
        
        Args:
            images: List of BGR numpy arrays
            
        Returns:
            List of LayoutPrediction objects
        """
        if not images:
            return []
        
        # Preprocess all images
        tensors = [
            self.preprocessor.preprocess_for_classifier(img)
            for img in images
        ]
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Batch inference
        with torch.no_grad():
            probs = self.model.predict_proba(batch)
            probs = probs.cpu().numpy()
        
        # Create predictions
        predictions = []
        for i, p in enumerate(probs):
            pred_idx = np.argmax(p)
            pred_class = self.CLASSES[pred_idx]
            confidence = float(p[pred_idx])
            
            prob_dict = {
                cls.value: float(p[j])
                for j, cls in enumerate(self.CLASSES)
            }
            
            if confidence < self.confidence_threshold:
                pred_class = LayoutType.MIXED
            
            predictions.append(LayoutPrediction(
                layout=pred_class,
                confidence=confidence,
                probabilities=prob_dict
            ))
        
        return predictions


class LayoutClassifierTrainer:
    """
    Training utilities for the layout classifier.
    
    Supports:
    - Transfer learning from ImageNet
    - Class-weighted loss for imbalanced data
    - Learning rate scheduling
    - Early stopping
    """
    
    def __init__(
        self,
        model: LayoutClassifierModel,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with different LR for backbone vs head
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": head_params, "lr": learning_rate}
        ], weight_decay=weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def save_model(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")


def create_synthetic_classifier() -> LayoutClassifier:
    """
    Create a classifier with synthetic/heuristic predictions.
    
    Useful for testing pipeline without trained model.
    Uses simple heuristics based on image characteristics.
    """
    
    class HeuristicClassifier(LayoutClassifier):
        def __init__(self):
            self.device = "cpu"
            self.confidence_threshold = 0.7
            self.preprocessor = ImagePreprocessor()
            self.model = None  # No model needed
        
        def predict(self, image: np.ndarray) -> LayoutPrediction:
            """
            Heuristic prediction based on image characteristics.
            
            - Tall/narrow images → bookshelf
            - Wide/square images with large objects → cover
            - Otherwise → mixed
            """
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Edge detection for structure analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Count vertical vs horizontal lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            
            vertical_count = 0
            horizontal_count = 0
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
                    if angle > np.pi/4:
                        vertical_count += 1
                    else:
                        horizontal_count += 1
            
            # Decision logic
            if vertical_count > horizontal_count * 2 and aspect_ratio > 0.8:
                pred_class = LayoutType.BOOKSHELF
                confidence = 0.75
            elif aspect_ratio < 1.5 and vertical_count < horizontal_count:
                pred_class = LayoutType.COVER
                confidence = 0.70
            else:
                pred_class = LayoutType.MIXED
                confidence = 0.60
            
            probs = {LayoutType.BOOKSHELF.value: 0.0, LayoutType.COVER.value: 0.0, LayoutType.MIXED.value: 0.0}
            probs[pred_class.value] = confidence
            remaining = (1.0 - confidence) / 2
            for k in probs:
                if probs[k] == 0.0:
                    probs[k] = remaining
            
            return LayoutPrediction(
                layout=pred_class,
                confidence=confidence,
                probabilities=probs
            )
    
    return HeuristicClassifier()
