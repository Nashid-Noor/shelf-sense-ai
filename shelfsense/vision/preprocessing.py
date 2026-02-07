"""
Image Preprocessing Utilities for ShelfSense AI

Handles all image preprocessing operations including:
- Resizing and normalization
- EXIF orientation correction
- Color space conversion
- Augmentation for training
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ExifTags
from dataclasses import dataclass
import torch
from torchvision import transforms
import albumentations as A
from loguru import logger


@dataclass
class PreprocessConfig:
    """Configuration for image preprocessing."""
    
    # Target sizes
    classifier_size: Tuple[int, int] = (224, 224)
    detector_size: Tuple[int, int] = (640, 640)
    roi_size: Tuple[int, int] = (224, 224)
    
    # Normalization (ImageNet stats)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Quality thresholds
    min_dimension: int = 100
    max_dimension: int = 4096
    min_contrast: float = 0.1


class ImagePreprocessor:
    """
    Comprehensive image preprocessing pipeline.
    
    Handles the transformation of raw uploaded images into
    formats suitable for each stage of the vision pipeline.
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        
        # PyTorch transforms for model input
        self._classifier_transform = transforms.Compose([
            transforms.Resize(self.config.classifier_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])
        
        self._roi_transform = transforms.Compose([
            transforms.Resize(self.config.roi_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])
    
    def load_image(self, image_source) -> np.ndarray:
        """
        Load image from various sources.
        
        Args:
            image_source: File path, bytes, PIL Image, or numpy array
            
        Returns:
            BGR numpy array (OpenCV format)
        """
        if isinstance(image_source, str):
            # File path
            image = cv2.imread(image_source)
            if image is None:
                raise ValueError(f"Could not load image from {image_source}")
            return image
        
        elif isinstance(image_source, bytes):
            # Raw bytes
            nparr = np.frombuffer(image_source, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image bytes")
            return image
        
        elif isinstance(image_source, Image.Image):
            # PIL Image
            return cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        
        elif isinstance(image_source, np.ndarray):
            # Already numpy array
            if len(image_source.shape) == 2:
                # Grayscale
                return cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
            return image_source
        
        else:
            raise TypeError(f"Unsupported image source type: {type(image_source)}")
    
    def fix_orientation(self, image: np.ndarray, exif_data: Optional[dict] = None) -> np.ndarray:
        """
        Fix image orientation based on EXIF data.
        
        Many phone photos are stored in a rotated format with EXIF
        orientation tags. This corrects that.
        """
        if exif_data is None:
            return image
        
        orientation_key = None
        for key in ExifTags.TAGS:
            if ExifTags.TAGS[key] == 'Orientation':
                orientation_key = key
                break
        
        if orientation_key is None or orientation_key not in exif_data:
            return image
        
        orientation = exif_data[orientation_key]
        
        # Apply rotation based on EXIF orientation
        if orientation == 3:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == 6:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return image
    
    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image quality for processing.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        height, width = image.shape[:2]
        
        # Check dimensions
        if min(height, width) < self.config.min_dimension:
            return False, f"Image too small: {width}x{height}, minimum is {self.config.min_dimension}px"
        
        if max(height, width) > self.config.max_dimension:
            return False, f"Image too large: {width}x{height}, maximum is {self.config.max_dimension}px"
        
        # Check contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std() / 255.0
        if contrast < self.config.min_contrast:
            return False, f"Image contrast too low: {contrast:.2f}, minimum is {self.config.min_contrast}"
        
        return True, ""
    
    def preprocess_for_classifier(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for layout classification model.
        
        Args:
            image: BGR numpy array
            
        Returns:
            Normalized tensor of shape (1, 3, 224, 224)
        """
        # Convert to RGB PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Apply transforms
        tensor = self._classifier_transform(pil_image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def preprocess_for_detector(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO detection.
        
        YOLO handles its own normalization, so we just resize
        while maintaining aspect ratio with padding.
        
        Args:
            image: BGR numpy array
            
        Returns:
            Resized BGR numpy array
        """
        target_size = self.config.detector_size[0]
        
        height, width = image.shape[:2]
        scale = target_size / max(height, width)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x = (target_size - new_width) // 2
        pad_y = (target_size - new_height) // 2
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
        
        return padded, scale, (pad_x, pad_y)
    
    def preprocess_roi(self, roi: np.ndarray) -> torch.Tensor:
        """
        Preprocess extracted ROI for embedding models.
        
        Args:
            roi: BGR numpy array of cropped book region
            
        Returns:
            Normalized tensor
        """
        # Convert to RGB PIL Image
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_roi = Image.fromarray(rgb_roi)
        
        return self._roi_transform(pil_roi)
    
    def enhance_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        Enhance ROI image for better OCR results.
        
        Applies contrast enhancement and denoising.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Convert back to 3-channel for consistency
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


class AugmentationPipeline:
    """
    Data augmentation for training.
    
    Separate augmentation pipelines for spine and cover detection
    to handle their different characteristics.
    """
    
    @staticmethod
    def get_spine_augmentations(p: float = 0.5) -> A.Compose:
        """
        Augmentations optimized for bookshelf/spine images.
        
        Focus on lighting variation and perspective since bookshelves
        are often photographed in challenging conditions.
        """
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=p
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=p * 0.6
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=p * 0.6),
            A.MotionBlur(blur_limit=5, p=p * 0.4),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                p=p * 0.6
            ),
            A.Perspective(scale=(0.02, 0.05), p=p * 0.6),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=3,
                p=p * 0.5
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    @staticmethod
    def get_cover_augmentations(p: float = 0.5) -> A.Compose:
        """
        Augmentations optimized for front cover images.
        
        More aggressive rotation and perspective since covers
        can be photographed at various angles.
        """
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=p
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=40,
                val_shift_limit=40,
                p=p * 0.6
            ),
            A.Rotate(limit=15, p=p * 0.8),
            A.Perspective(scale=(0.05, 0.15), p=p * 0.8),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0.5,
                p=p * 0.2
            ),
            A.CoarseDropout(
                max_holes=3,
                max_height=50,
                max_width=50,
                p=p * 0.4
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=p * 0.4),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    @staticmethod
    def get_ocr_augmentations(p: float = 0.3) -> A.Compose:
        """
        Light augmentations for OCR robustness.
        
        Careful not to destroy text readability.
        """
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=p
            ),
            A.GaussNoise(var_limit=(5.0, 20.0), p=p * 0.5),
            A.Blur(blur_limit=3, p=p * 0.3),
        ])
