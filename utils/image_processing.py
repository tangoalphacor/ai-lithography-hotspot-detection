"""
Image Processing Utilities
==========================

Utilities for image preprocessing, augmentation, and manipulation
for lithography hotspot detection.

Author: AI Assistant
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Union, Tuple, List, Optional
import io
import base64


class ImageProcessor:
    """Comprehensive image processing utilities."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    def load_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Image.Image:
        """
        Load image from various input types.
        
        Args:
            image_input: Image path, bytes, numpy array, or PIL Image
            
        Returns:
            PIL Image object
        """
        if isinstance(image_input, str):
            # Load from file path
            return Image.open(image_input)
        elif isinstance(image_input, bytes):
            # Load from bytes
            return Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
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
        """
        Resize image with specified method.
        
        Args:
            image: Input PIL Image
            size: Target size (width, height)
            method: Resampling method
            
        Returns:
            Resized PIL Image
        """
        resample_methods = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        
        method_enum = resample_methods.get(method.lower(), Image.LANCZOS)
        return image.resize(size, method_enum)
    
    def normalize_image(self, image: Union[Image.Image, np.ndarray], 
                       target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        Normalize image to specified range.
        
        Args:
            image: Input image
            target_range: Target value range
            
        Returns:
            Normalized image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to float
        image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Scale to target range
        min_val, max_val = target_range
        image = image * (max_val - min_val) + min_val
        
        return image
    
    def enhance_contrast(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        """
        Enhance image contrast.
        
        Args:
            image: Input PIL Image
            factor: Contrast enhancement factor
            
        Returns:
            Contrast-enhanced image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def enhance_brightness(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """
        Enhance image brightness.
        
        Args:
            image: Input PIL Image
            factor: Brightness enhancement factor
            
        Returns:
            Brightness-enhanced image
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def enhance_sharpness(self, image: Image.Image, factor: float = 1.3) -> Image.Image:
        """
        Enhance image sharpness.
        
        Args:
            image: Input PIL Image
            factor: Sharpness enhancement factor
            
        Returns:
            Sharpness-enhanced image
        """
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def apply_gaussian_blur(self, image: Image.Image, radius: float = 1.0) -> Image.Image:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input PIL Image
            radius: Blur radius
            
        Returns:
            Blurred image
        """
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def apply_unsharp_mask(self, image: Image.Image, radius: float = 2.0, 
                          percent: int = 150, threshold: int = 3) -> Image.Image:
        """
        Apply unsharp mask filter for edge enhancement.
        
        Args:
            image: Input PIL Image
            radius: Mask radius
            percent: Enhancement percentage
            threshold: Threshold for enhancement
            
        Returns:
            Enhanced image
        """
        return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    
    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """
        Convert image to grayscale.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Grayscale image
        """
        return image.convert('L')
    
    def convert_to_rgb(self, image: Image.Image) -> Image.Image:
        """
        Convert image to RGB.
        
        Args:
            image: Input PIL Image
            
        Returns:
            RGB image
        """
        return image.convert('RGB')
    
    def apply_histogram_equalization(self, image: Image.Image) -> Image.Image:
        """
        Apply histogram equalization to enhance contrast.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Equalized image
        """
        # Convert to numpy for processing
        img_array = np.array(image)
        
        if len(img_array.shape) == 2:
            # Grayscale image
            equalized = cv2.equalizeHist(img_array)
        else:
            # Color image - equalize each channel
            img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        return Image.fromarray(equalized)
    
    def apply_clahe(self, image: Image.Image, clip_limit: float = 2.0, 
                   tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input PIL Image
            clip_limit: Clipping limit for contrast enhancement
            tile_grid_size: Size of the neighborhood area
            
        Returns:
            CLAHE-enhanced image
        """
        img_array = np.array(image)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(img_array.shape) == 2:
            # Grayscale image
            enhanced = clahe.apply(img_array)
        else:
            # Color image - apply to luminance channel
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
            enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
    
    def detect_edges(self, image: Image.Image, method: str = 'canny', 
                    low_threshold: int = 50, high_threshold: int = 150) -> Image.Image:
        """
        Detect edges in image.
        
        Args:
            image: Input PIL Image
            method: Edge detection method ('canny', 'sobel', 'laplacian')
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny
            
        Returns:
            Edge-detected image
        """
        # Convert to grayscale numpy array
        gray = np.array(self.convert_to_grayscale(image))
        
        if method.lower() == 'canny':
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        elif method.lower() == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        elif method.lower() == 'laplacian':
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        return Image.fromarray(edges)
    
    def rotate_image(self, image: Image.Image, angle: float, 
                    expand: bool = True, fillcolor: int = 0) -> Image.Image:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input PIL Image
            angle: Rotation angle in degrees
            expand: Whether to expand image to fit rotated content
            fillcolor: Fill color for empty areas
            
        Returns:
            Rotated image
        """
        return image.rotate(angle, expand=expand, fillcolor=fillcolor)
    
    def flip_image(self, image: Image.Image, direction: str = 'horizontal') -> Image.Image:
        """
        Flip image horizontally or vertically.
        
        Args:
            image: Input PIL Image
            direction: Flip direction ('horizontal' or 'vertical')
            
        Returns:
            Flipped image
        """
        if direction.lower() == 'horizontal':
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction.lower() == 'vertical':
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError(f"Unknown flip direction: {direction}")
    
    def crop_image(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Crop image to specified bounding box.
        
        Args:
            image: Input PIL Image
            bbox: Bounding box (left, top, right, bottom)
            
        Returns:
            Cropped image
        """
        return image.crop(bbox)
    
    def pad_image(self, image: Image.Image, padding: Union[int, Tuple[int, ...]], 
                 fill_color: Union[int, Tuple[int, ...]] = 0) -> Image.Image:
        """
        Add padding to image.
        
        Args:
            image: Input PIL Image
            padding: Padding size(s)
            fill_color: Fill color for padding
            
        Returns:
            Padded image
        """
        from PIL import ImageOps
        return ImageOps.expand(image, padding, fill_color)
    
    def calculate_image_stats(self, image: Image.Image) -> dict:
        """
        Calculate image statistics.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Dictionary with image statistics
        """
        img_array = np.array(image)
        
        stats = {
            'width': image.width,
            'height': image.height,
            'channels': len(img_array.shape),
            'mode': image.mode,
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
            'min': float(np.min(img_array)),
            'max': float(np.max(img_array)),
            'median': float(np.median(img_array))
        }
        
        if len(img_array.shape) == 3:
            stats['channels'] = img_array.shape[2]
            stats['channel_means'] = [float(np.mean(img_array[:, :, i])) for i in range(img_array.shape[2])]
        
        return stats
    
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: Input PIL Image
            format: Output format
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def base64_to_image(self, base64_str: str) -> Image.Image:
        """
        Convert base64 string to PIL Image.
        
        Args:
            base64_str: Base64 encoded image string
            
        Returns:
            PIL Image object
        """
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))
    
    def batch_process_images(self, images: List[Image.Image], 
                           operations: List[dict]) -> List[Image.Image]:
        """
        Apply batch processing operations to multiple images.
        
        Args:
            images: List of input images
            operations: List of operation dictionaries
            
        Returns:
            List of processed images
        """
        processed_images = []
        
        for image in images:
            processed_image = image.copy()
            
            for operation in operations:
                op_type = operation.get('type')
                params = operation.get('params', {})
                
                if op_type == 'resize':
                    processed_image = self.resize_image(processed_image, **params)
                elif op_type == 'enhance_contrast':
                    processed_image = self.enhance_contrast(processed_image, **params)
                elif op_type == 'enhance_brightness':
                    processed_image = self.enhance_brightness(processed_image, **params)
                elif op_type == 'gaussian_blur':
                    processed_image = self.apply_gaussian_blur(processed_image, **params)
                elif op_type == 'rotate':
                    processed_image = self.rotate_image(processed_image, **params)
                elif op_type == 'flip':
                    processed_image = self.flip_image(processed_image, **params)
                # Add more operations as needed
            
            processed_images.append(processed_image)
        
        return processed_images
    
    def create_image_grid(self, images: List[Image.Image], 
                         grid_size: Optional[Tuple[int, int]] = None,
                         spacing: int = 5, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Create a grid layout from multiple images.
        
        Args:
            images: List of input images
            grid_size: Grid dimensions (rows, cols). If None, auto-calculate
            spacing: Spacing between images
            background_color: Background color
            
        Returns:
            Grid image
        """
        if not images:
            raise ValueError("No images provided")
        
        # Calculate grid size if not provided
        if grid_size is None:
            num_images = len(images)
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        
        # Get maximum dimensions
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # Calculate grid dimensions
        grid_width = cols * max_width + (cols - 1) * spacing
        grid_height = rows * max_height + (rows - 1) * spacing
        
        # Create grid image
        grid_image = Image.new('RGB', (grid_width, grid_height), background_color)
        
        # Place images in grid
        for i, image in enumerate(images):
            if i >= rows * cols:
                break
            
            row = i // cols
            col = i % cols
            
            x = col * (max_width + spacing)
            y = row * (max_height + spacing)
            
            # Resize image to fit grid cell if needed
            if image.size != (max_width, max_height):
                image = self.resize_image(image, (max_width, max_height))
            
            grid_image.paste(image, (x, y))
        
        return grid_image
