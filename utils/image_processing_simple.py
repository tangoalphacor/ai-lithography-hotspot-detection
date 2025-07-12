"""
Simplified Image Processing Utilities
====================================

Basic image processing utilities that work with standard packages.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Union, Tuple, List, Optional
import io
import base64


class ImageProcessor:
    """Basic image processing utilities."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    def load_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Image.Image:
        """Load image from various input types."""
        if isinstance(image_input, str):
            return Image.open(image_input)
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return Image.fromarray(image_input, mode='L')
            elif len(image_input.shape) == 3:
                return Image.fromarray(image_input, mode='RGB')
            else:
                raise ValueError(f"Unsupported array shape: {image_input.shape}")
        elif isinstance(image_input, Image.Image):
            return image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int], 
                    method: str = 'lanczos') -> Image.Image:
        """Resize image with specified method."""
        resample_methods = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        
        method_enum = resample_methods.get(method.lower(), Image.LANCZOS)
        return image.resize(size, method_enum)
    
    def enhance_contrast(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        """Enhance image contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def enhance_brightness(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Enhance image brightness."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """Convert image to grayscale."""
        return image.convert('L')
    
    def convert_to_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB."""
        return image.convert('RGB')
    
    def calculate_image_stats(self, image: Image.Image) -> dict:
        """Calculate basic image statistics."""
        img_array = np.array(image)
        
        stats = {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
            'min': float(np.min(img_array)),
            'max': float(np.max(img_array))
        }
        
        if len(img_array.shape) == 3:
            stats['channels'] = img_array.shape[2]
        else:
            stats['channels'] = 1
        
        return stats
    
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
