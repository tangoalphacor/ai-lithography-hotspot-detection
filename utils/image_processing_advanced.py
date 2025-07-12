"""
Advanced Image Processing Utilities
==================================

Comprehensive image processing utilities with advanced algorithms
for lithography hotspot detection applications.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
import time
from pathlib import Path
import io
import base64
from dataclasses import dataclass
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

@dataclass
class ProcessingMetrics:
    """Metrics for image processing operations"""
    processing_time: float
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    file_size_reduction: float
    quality_score: float
    operations_applied: List[str]

class AdvancedImageProcessor:
    """Advanced image processing with state-of-the-art algorithms"""
    
    def __init__(self, enable_gpu_acceleration: bool = False):
        self.enable_gpu = enable_gpu_acceleration
        self.processing_history = []
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
        
        # Initialize advanced algorithms
        self.setup_advanced_algorithms()
    
    def setup_advanced_algorithms(self):
        """Initialize advanced processing algorithms"""
        # Morphological kernels
        self.morphology_kernels = {
            'ellipse_5': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            'rect_3': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            'cross_7': cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
        }
        
        # Gabor filter bank
        self.gabor_filters = self._create_gabor_filter_bank()
        
        # Adaptive parameters
        self.adaptive_params = {
            'noise_variance_threshold': 100,
            'edge_density_threshold': 0.15,
            'contrast_enhancement_factor': 1.2
        }
    
    def _create_gabor_filter_bank(self) -> List[np.ndarray]:
        """Create a bank of Gabor filters for texture analysis"""
        filters = []
        for theta in range(0, 180, 30):  # 6 orientations
            for frequency in [0.1, 0.3, 0.5]:  # 3 frequencies
                kernel = cv2.getGaborKernel((21, 21), 3, np.radians(theta), 
                                          2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                filters.append(kernel)
        return filters
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray], 
                        target_size: Optional[Tuple[int, int]] = None,
                        enhance_quality: bool = True,
                        normalize: bool = True) -> Dict[str, Any]:
        """
        Advanced image preprocessing with quality enhancement
        
        Args:
            image: Input image (PIL Image or numpy array)
            target_size: Target size for resizing
            enhance_quality: Apply quality enhancement
            normalize: Apply normalization
            
        Returns:
            Dictionary containing processed image and metadata
        """
        start_time = time.time()
        operations_applied = []
        
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        original_size = image.size
        original_mode = image.mode
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            operations_applied.append('color_conversion')
        
        # Quality enhancement
        if enhance_quality:
            image = self._enhance_image_quality(image)
            operations_applied.append('quality_enhancement')
        
        # Resize if target size specified
        if target_size:
            image = self._smart_resize(image, target_size)
            operations_applied.append('smart_resize')
        
        # Convert to numpy for further processing
        img_array = np.array(image)
        
        # Noise reduction
        img_array = self._advanced_noise_reduction(img_array)
        operations_applied.append('noise_reduction')
        
        # Normalization
        if normalize:
            img_array = self._advanced_normalization(img_array)
            operations_applied.append('normalization')
        
        # Convert back to PIL
        processed_image = Image.fromarray(img_array.astype(np.uint8))
        
        # Calculate metrics
        processing_time = time.time() - start_time
        quality_score = self._calculate_image_quality(img_array)
        
        metrics = ProcessingMetrics(
            processing_time=processing_time,
            original_size=original_size,
            processed_size=processed_image.size,
            file_size_reduction=0.0,  # Would need actual file sizes
            quality_score=quality_score,
            operations_applied=operations_applied
        )
        
        return {
            'processed_image': processed_image,
            'original_image': image,
            'metrics': metrics,
            'array': img_array,
            'success': True
        }
    
    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Apply advanced quality enhancement techniques"""
        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Color enhancement
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        
        # Advanced denoising using PIL filters
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
    
    def _smart_resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Smart resizing with aspect ratio preservation and quality optimization"""
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calculate aspect ratios
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        
        # Determine best resize strategy
        if abs(original_ratio - target_ratio) < 0.1:
            # Similar aspect ratios - direct resize
            return image.resize(target_size, Image.Resampling.LANCZOS)
        else:
            # Different aspect ratios - resize and pad
            if original_ratio > target_ratio:
                # Image is wider - fit to width
                new_width = target_width
                new_height = int(target_width / original_ratio)
            else:
                # Image is taller - fit to height
                new_height = target_height
                new_width = int(target_height * original_ratio)
            
            # Resize with high quality
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            final_image = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            final_image.paste(resized, (paste_x, paste_y))
            
            return final_image
    
    def _advanced_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Advanced noise reduction using multiple algorithms"""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Estimate noise level
        noise_level = self._estimate_noise_level(img_float)
        
        if noise_level > 0.05:  # High noise
            # Apply Non-local Means Denoising
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        else:
            # Apply Gaussian blur for low noise
            denoised = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        return denoised
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range
        return min(laplacian_var / 10000, 1.0)
    
    def _advanced_normalization(self, image: np.ndarray) -> np.ndarray:
        """Advanced normalization with adaptive histogram equalization"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Apply CLAHE directly to grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(image)
        
        return normalized
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """Calculate image quality score based on multiple metrics"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Sharpness (Tenengrad variance)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sharpness = np.var(gradient_magnitude)
        
        # Contrast (RMS contrast)
        contrast = np.sqrt(np.mean((gray - np.mean(gray))**2))
        
        # Brightness distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        brightness_score = 1.0 - np.abs(0.5 - np.mean(gray) / 255.0)
        
        # Combine metrics (normalize and weight)
        sharpness_norm = min(sharpness / 10000, 1.0)
        contrast_norm = min(contrast / 128, 1.0)
        
        quality_score = (0.4 * sharpness_norm + 0.4 * contrast_norm + 0.2 * brightness_score)
        
        return float(quality_score)
    
    def extract_advanced_features(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Extract comprehensive image features for analysis"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        features = {}
        
        # Convert to grayscale for some operations
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Statistical features
        features['statistical'] = self._extract_statistical_features(gray)
        
        # Texture features using Gabor filters
        features['texture'] = self._extract_gabor_features(gray)
        
        # Edge features
        features['edge'] = self._extract_edge_features(gray)
        
        # Frequency domain features
        features['frequency'] = self._extract_frequency_features(gray)
        
        # Morphological features
        features['morphological'] = self._extract_morphological_features(gray)
        
        return features
    
    def _extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from image"""
        return {
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'variance': float(np.var(image)),
            'skewness': float(self._calculate_skewness(image)),
            'kurtosis': float(self._calculate_kurtosis(image)),
            'entropy': float(self._calculate_entropy(image)),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'median': float(np.median(image)),
            'percentile_25': float(np.percentile(image, 25)),
            'percentile_75': float(np.percentile(image, 75))
        }
    
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of image intensity distribution"""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0.0
        return np.mean(((image - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, image: np.ndarray) -> float:
        """Calculate kurtosis of image intensity distribution"""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0.0
        return np.mean(((image - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of image"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
    
    def _extract_gabor_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture features using Gabor filter bank"""
        features = {}
        
        for i, kernel in enumerate(self.gabor_filters):
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            features[f'gabor_{i}_mean'] = float(np.mean(filtered))
            features[f'gabor_{i}_std'] = float(np.std(filtered))
            features[f'gabor_{i}_energy'] = float(np.sum(filtered ** 2))
        
        return features
    
    def _extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract edge-related features"""
        # Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Sobel gradients
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return {
            'edge_density': float(edge_density),
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude)),
            'gradient_max': float(np.max(gradient_magnitude))
        }
    
    def _extract_frequency_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features using FFT"""
        # Apply FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Extract features from frequency domain
        return {
            'freq_mean': float(np.mean(magnitude_spectrum)),
            'freq_std': float(np.std(magnitude_spectrum)),
            'freq_energy': float(np.sum(magnitude_spectrum ** 2)),
            'freq_entropy': float(self._calculate_entropy(magnitude_spectrum.astype(np.uint8)))
        }
    
    def _extract_morphological_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract morphological features"""
        features = {}
        
        # Apply different morphological operations
        for name, kernel in self.morphology_kernels.items():
            # Opening
            opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            features[f'opening_{name}_mean'] = float(np.mean(opening))
            
            # Closing
            closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            features[f'closing_{name}_mean'] = float(np.mean(closing))
            
            # Gradient
            gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
            features[f'gradient_{name}_mean'] = float(np.mean(gradient))
        
        return features
    
    def apply_advanced_filters(self, image: Union[Image.Image, np.ndarray], 
                             filter_type: str, **kwargs) -> Dict[str, Any]:
        """Apply advanced image filters"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        start_time = time.time()
        
        if filter_type == "bilateral":
            filtered = cv2.bilateralFilter(img_array, 9, 75, 75)
        elif filter_type == "guided":
            filtered = self._guided_filter(img_array, **kwargs)
        elif filter_type == "anisotropic":
            filtered = self._anisotropic_diffusion(img_array, **kwargs)
        elif filter_type == "wavelet_denoise":
            filtered = self._wavelet_denoising(img_array, **kwargs)
        else:
            filtered = img_array
        
        processing_time = time.time() - start_time
        
        return {
            'filtered_image': Image.fromarray(filtered.astype(np.uint8)),
            'original_image': image,
            'filter_type': filter_type,
            'processing_time': processing_time,
            'success': True
        }
    
    def _guided_filter(self, image: np.ndarray, radius: int = 8, epsilon: float = 0.01) -> np.ndarray:
        """Implement guided filter for edge-preserving smoothing"""
        if len(image.shape) == 3:
            guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            guide = image.copy()
        
        # Convert to float
        I = guide.astype(np.float64) / 255.0
        p = image.astype(np.float64) / 255.0
        
        # Compute local statistics
        mean_I = cv2.blur(I, (radius, radius))
        mean_p = cv2.blur(p, (radius, radius)) if len(p.shape) == 2 else np.stack([cv2.blur(p[:,:,i], (radius, radius)) for i in range(3)], axis=2)
        corr_Ip = cv2.blur(I * p, (radius, radius)) if len(p.shape) == 2 else np.stack([cv2.blur(I * p[:,:,i], (radius, radius)) for i in range(3)], axis=2)
        var_I = cv2.blur(I * I, (radius, radius)) - mean_I * mean_I
        
        # Compute a and b
        cov_Ip = corr_Ip - mean_I[:,:,None] * mean_p if len(p.shape) == 3 else corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I[:,:,None] + epsilon) if len(p.shape) == 3 else cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I[:,:,None] if len(p.shape) == 3 else mean_p - a * mean_I
        
        # Compute mean of a and b
        mean_a = cv2.blur(a, (radius, radius)) if len(a.shape) == 2 else np.stack([cv2.blur(a[:,:,i], (radius, radius)) for i in range(3)], axis=2)
        mean_b = cv2.blur(b, (radius, radius)) if len(b.shape) == 2 else np.stack([cv2.blur(b[:,:,i], (radius, radius)) for i in range(3)], axis=2)
        
        # Compute output
        q = mean_a * I[:,:,None] + mean_b if len(p.shape) == 3 else mean_a * I + mean_b
        
        return (q * 255).astype(np.uint8)
    
    def _anisotropic_diffusion(self, image: np.ndarray, num_iter: int = 10, 
                              delta_t: float = 0.25, kappa: float = 30) -> np.ndarray:
        """Implement anisotropic diffusion for edge-preserving smoothing"""
        if len(image.shape) == 3:
            # Process each channel separately
            result = np.zeros_like(image)
            for i in range(3):
                result[:,:,i] = self._anisotropic_diffusion_single_channel(
                    image[:,:,i], num_iter, delta_t, kappa
                )
            return result
        else:
            return self._anisotropic_diffusion_single_channel(image, num_iter, delta_t, kappa)
    
    def _anisotropic_diffusion_single_channel(self, image: np.ndarray, num_iter: int, 
                                            delta_t: float, kappa: float) -> np.ndarray:
        """Anisotropic diffusion for single channel"""
        img = image.astype(np.float64)
        
        for _ in range(num_iter):
            # Compute gradients
            nabla_n = np.roll(img, -1, axis=0) - img  # North gradient
            nabla_s = np.roll(img, 1, axis=0) - img   # South gradient
            nabla_e = np.roll(img, -1, axis=1) - img  # East gradient
            nabla_w = np.roll(img, 1, axis=1) - img   # West gradient
            
            # Compute conduction coefficients
            c_n = np.exp(-(nabla_n/kappa)**2)
            c_s = np.exp(-(nabla_s/kappa)**2)
            c_e = np.exp(-(nabla_e/kappa)**2)
            c_w = np.exp(-(nabla_w/kappa)**2)
            
            # Update image
            img += delta_t * (c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w)
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def _wavelet_denoising(self, image: np.ndarray, wavelet: str = 'db4', 
                          sigma: Optional[float] = None) -> np.ndarray:
        """Wavelet-based denoising (simplified implementation)"""
        # This is a simplified version - real wavelet denoising would require PyWavelets
        # For now, use a combination of Gaussian blur and unsharp masking
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Unsharp masking
        unsharp_strength = 0.5
        denoised = cv2.addWeighted(image, 1 + unsharp_strength, blurred, -unsharp_strength, 0)
        
        return np.clip(denoised, 0, 255).astype(np.uint8)
    
    def create_image_pyramid(self, image: Union[Image.Image, np.ndarray], 
                           levels: int = 4) -> List[np.ndarray]:
        """Create Gaussian pyramid for multi-scale analysis"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        pyramid = [img_array]
        current = img_array.copy()
        
        for i in range(levels - 1):
            # Apply Gaussian blur and downsample
            blurred = cv2.GaussianBlur(current, (5, 5), 1.0)
            downsampled = cv2.resize(blurred, 
                                   (current.shape[1] // 2, current.shape[0] // 2),
                                   interpolation=cv2.INTER_AREA)
            pyramid.append(downsampled)
            current = downsampled
        
        return pyramid
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing operations"""
        return {
            'total_operations': len(self.processing_history),
            'recent_operations': self.processing_history[-10:] if self.processing_history else [],
            'supported_formats': self.supported_formats,
            'gpu_acceleration': self.enable_gpu,
            'available_filters': [
                'bilateral', 'guided', 'anisotropic', 'wavelet_denoise'
            ],
            'feature_types': [
                'statistical', 'texture', 'edge', 'frequency', 'morphological'
            ]
        }

# Factory function
def get_image_processor(enable_gpu: bool = False) -> AdvancedImageProcessor:
    """Factory function to create advanced image processor"""
    return AdvancedImageProcessor(enable_gpu_acceleration=enable_gpu)
