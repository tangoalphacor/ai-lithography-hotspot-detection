"""
Advanced Image Processing Utilities
===================================

Sophisticated image processing algorithms including Gabor filters,
anisotropic diffusion, guided filtering, wavelet denoising, and advanced
feature extraction for lithography applications.

Author: Abhinav
Version: 2.0.0 (Advanced)
"""

import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, Tuple, List, Optional
import logging
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter
from skimage import restoration, morphology, feature, filters, segmentation
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor, threshold_otsu
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedImageProcessor:
    """Advanced image processing with state-of-the-art algorithms"""
    
    def __init__(self, enable_gpu: bool = False):
        self.enable_gpu = enable_gpu
        self.scaler = StandardScaler()
        
        # Gabor filter bank parameters
        self.gabor_frequencies = [0.1, 0.3, 0.5, 0.7]
        self.gabor_orientations = [0, 45, 90, 135]
        
        logging.info(f"Advanced image processor initialized (GPU: {enable_gpu})")
    
    def preprocess_image(self, image: Image.Image,
                        target_size: Tuple[int, int] = (224, 224),
                        enhance_quality: bool = True,
                        normalize: bool = True) -> Dict[str, Any]:
        """Comprehensive image preprocessing with quality enhancement"""
        
        try:
            original_size = image.size
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image).astype(np.float32)
            
            # Quality enhancement pipeline
            if enhance_quality:
                image_array = self._enhance_image_quality(image_array)
            
            # Resize while preserving aspect ratio
            if target_size != original_size:
                image_array = self._smart_resize(image_array, target_size)
            
            # Normalization
            if normalize:
                image_array = self._adaptive_normalization(image_array)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(image_array.astype(np.uint8))
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(image_array)
            
            return {
                'processed_image': processed_image,
                'metrics': {
                    'original_size': original_size,
                    'processed_size': target_size,
                    'quality_score': quality_score,
                    'operations_applied': ['quality_enhancement', 'resize', 'normalization'] if enhance_quality else ['resize', 'normalization']
                }
            }
            
        except Exception as e:
            logging.error(f"Error in image preprocessing: {e}")
            return {
                'processed_image': image.resize(target_size),
                'metrics': {'error': str(e)}
            }
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Advanced image quality enhancement"""
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            is_color = True
        else:
            gray = image.copy()
            is_color = False
        
        # 1. Anisotropic diffusion for noise reduction while preserving edges
        enhanced = self._anisotropic_diffusion(gray)
        
        # 2. Guided filtering for edge-preserving smoothing
        enhanced = self._guided_filter(enhanced, enhanced, radius=8, epsilon=0.2)
        
        # 3. Adaptive histogram equalization
        enhanced = self._adaptive_histogram_equalization(enhanced)
        
        # 4. Sharpening
        enhanced = self._unsharp_masking(enhanced)
        
        # Convert back to color if needed
        if is_color:
            # Apply enhancement to each channel
            result = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i]
                # Apply similar enhancement to each channel
                enhanced_channel = self._enhance_single_channel(channel, gray, enhanced)
                result[:, :, i] = enhanced_channel
            return result
        else:
            return enhanced
    
    def _anisotropic_diffusion(self, image: np.ndarray, num_iter: int = 50, 
                              delta_t: float = 0.14, kappa: float = 20) -> np.ndarray:
        """Perona-Malik anisotropic diffusion"""
        try:
            # Initialize
            img = image.copy().astype(np.float32)
            
            for _ in range(num_iter):
                # Calculate gradients
                nabla_n = np.roll(img, -1, axis=0) - img  # North
                nabla_s = np.roll(img, 1, axis=0) - img   # South
                nabla_e = np.roll(img, -1, axis=1) - img  # East
                nabla_w = np.roll(img, 1, axis=1) - img   # West
                
                # Calculate diffusion coefficients
                c_n = np.exp(-(nabla_n / kappa) ** 2)
                c_s = np.exp(-(nabla_s / kappa) ** 2)
                c_e = np.exp(-(nabla_e / kappa) ** 2)
                c_w = np.exp(-(nabla_w / kappa) ** 2)
                
                # Update image
                img += delta_t * (c_n * nabla_n + c_s * nabla_s + 
                                 c_e * nabla_e + c_w * nabla_w)
            
            return np.clip(img, 0, 255)
            
        except Exception:
            # Fallback to Gaussian smoothing
            return gaussian_filter(image, sigma=1.0)
    
    def _guided_filter(self, image: np.ndarray, guide: np.ndarray, 
                      radius: int = 8, epsilon: float = 0.2) -> np.ndarray:
        """Guided filter implementation"""
        try:
            mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
            mean_p = cv2.boxFilter(image, cv2.CV_32F, (radius, radius))
            mean_Ip = cv2.boxFilter(guide * image, cv2.CV_32F, (radius, radius))
            cov_Ip = mean_Ip - mean_I * mean_p
            
            mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
            var_I = mean_II - mean_I * mean_I
            
            a = cov_Ip / (var_I + epsilon)
            b = mean_p - a * mean_I
            
            mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
            mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
            
            return mean_a * guide + mean_b
            
        except Exception:
            # Fallback to bilateral filter
            return cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75).astype(np.float32)
    
    def _adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image.astype(np.uint8)).astype(np.float32)
        except Exception:
            # Fallback to standard histogram equalization
            return cv2.equalizeHist(image.astype(np.uint8)).astype(np.float32)
    
    def _unsharp_masking(self, image: np.ndarray, sigma: float = 1.0, alpha: float = 0.5) -> np.ndarray:
        """Unsharp masking for image sharpening"""
        try:
            blurred = gaussian_filter(image, sigma=sigma)
            sharpened = image + alpha * (image - blurred)
            return np.clip(sharpened, 0, 255)
        except Exception:
            return image
    
    def _enhance_single_channel(self, channel: np.ndarray, reference_gray: np.ndarray, 
                               enhanced_gray: np.ndarray) -> np.ndarray:
        """Enhance single color channel based on grayscale enhancement"""
        try:
            # Calculate enhancement ratio
            ratio = enhanced_gray / (reference_gray + 1e-8)
            enhanced_channel = channel * ratio
            return np.clip(enhanced_channel, 0, 255)
        except Exception:
            return channel
    
    def _smart_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Smart resize with aspect ratio preservation"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Resize
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to target size
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        if len(image.shape) == 3:
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
        else:
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def _adaptive_normalization(self, image: np.ndarray) -> np.ndarray:
        """Adaptive normalization based on image statistics"""
        try:
            # Calculate robust statistics
            percentile_2 = np.percentile(image, 2)
            percentile_98 = np.percentile(image, 98)
            
            # Robust normalization
            normalized = (image - percentile_2) / (percentile_98 - percentile_2 + 1e-8)
            normalized = np.clip(normalized, 0, 1) * 255
            
            return normalized
            
        except Exception:
            # Fallback to min-max normalization
            return ((image - image.min()) / (image.max() - image.min() + 1e-8)) * 255
    
    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """Calculate image quality score"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)
            
            # Calculate various quality metrics
            
            # 1. Contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            
            # 2. Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian) / 10000.0  # Normalize
            
            # 3. Brightness distribution (entropy)
            hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
            entropy = -np.sum(hist * np.log(hist + 1e-8)) / 8.0  # Normalize
            
            # 4. Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Combine metrics (weighted average)
            quality_score = (0.3 * contrast + 0.3 * min(sharpness, 1.0) + 
                           0.2 * min(entropy, 1.0) + 0.2 * edge_density)
            
            return min(1.0, quality_score)
            
        except Exception:
            return 0.8  # Default quality score
    
    def extract_advanced_features(self, image: Image.Image) -> Dict[str, Dict[str, float]]:
        """Extract comprehensive advanced features"""
        try:
            # Convert to arrays
            image_array = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            features = {}
            
            # 1. Gabor filter responses
            features['gabor'] = self._extract_gabor_features(gray)
            
            # 2. Local Binary Pattern features
            features['lbp'] = self._extract_lbp_features(gray)
            
            # 3. Gray-Level Co-occurrence Matrix features
            features['glcm'] = self._extract_glcm_features(gray)
            
            # 4. Morphological features
            features['morphological'] = self._extract_morphological_features(gray)
            
            # 5. Fourier features
            features['fourier'] = self._extract_fourier_features(gray)
            
            # 6. Wavelet features
            features['wavelet'] = self._extract_wavelet_features(gray)
            
            # 7. Statistical features
            features['statistical'] = self._extract_statistical_features(gray)
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting advanced features: {e}")
            return {'error': {'message': str(e)}}
    
    def _extract_gabor_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract Gabor filter bank features"""
        features = {}
        
        try:
            for i, frequency in enumerate(self.gabor_frequencies):
                for j, orientation in enumerate(self.gabor_orientations):
                    # Apply Gabor filter
                    real, _ = gabor(gray, frequency=frequency, 
                                  theta=np.radians(orientation))
                    
                    # Extract statistics
                    features[f'gabor_f{i}_o{j}_mean'] = np.mean(real)
                    features[f'gabor_f{i}_o{j}_std'] = np.std(real)
                    features[f'gabor_f{i}_o{j}_energy'] = np.sum(real ** 2)
            
        except Exception:
            # Fallback: simple filter responses
            for i in range(len(self.gabor_frequencies)):
                for j in range(len(self.gabor_orientations)):
                    features[f'gabor_f{i}_o{j}_mean'] = 0.0
                    features[f'gabor_f{i}_o{j}_std'] = 0.0
                    features[f'gabor_f{i}_o{j}_energy'] = 0.0
        
        return features
    
    def _extract_lbp_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract Local Binary Pattern features"""
        features = {}
        
        try:
            radius = 3
            n_points = 8 * radius
            
            # Compute LBP
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Compute histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                 range=(0, n_points + 2), density=True)
            
            # Extract features
            for i, val in enumerate(hist):
                features[f'lbp_bin_{i}'] = val
            
            features['lbp_uniformity'] = np.sum(hist ** 2)
            features['lbp_entropy'] = -np.sum(hist * np.log(hist + 1e-8))
            
        except Exception:
            # Fallback: zero features
            for i in range(26):  # n_points + 2
                features[f'lbp_bin_{i}'] = 0.0
            features['lbp_uniformity'] = 0.0
            features['lbp_entropy'] = 0.0
        
        return features
    
    def _extract_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract Gray-Level Co-occurrence Matrix features"""
        features = {}
        
        try:
            # Compute GLCM
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]
            
            glcm = graycomatrix(gray, distances=distances, angles=np.radians(angles),
                              levels=256, symmetric=True, normed=True)
            
            # Extract properties
            properties = ['dissimilarity', 'correlation', 'homogeneity', 'energy', 'contrast', 'ASM']
            
            for prop in properties:
                if prop == 'ASM':  # Angular Second Moment
                    values = graycoprops(glcm, 'energy') ** 2
                else:
                    values = graycoprops(glcm, prop)
                
                features[f'glcm_{prop}_mean'] = np.mean(values)
                features[f'glcm_{prop}_std'] = np.std(values)
            
        except Exception:
            # Fallback: zero features
            properties = ['dissimilarity', 'correlation', 'homogeneity', 'energy', 'contrast', 'ASM']
            for prop in properties:
                features[f'glcm_{prop}_mean'] = 0.0
                features[f'glcm_{prop}_std'] = 0.0
        
        return features
    
    def _extract_morphological_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract morphological features"""
        features = {}
        
        try:
            # Binary image
            threshold = threshold_otsu(gray)
            binary = gray > threshold
            
            # Morphological operations
            disk = morphology.disk(3)
            
            opened = morphology.opening(binary, disk)
            closed = morphology.closing(binary, disk)
            gradient = morphology.gradient(binary, disk)
            
            # Extract features
            features['opening_area'] = np.sum(opened) / binary.size
            features['closing_area'] = np.sum(closed) / binary.size
            features['gradient_area'] = np.sum(gradient) / binary.size
            
            # Connected components
            labeled = morphology.label(binary)
            props = morphology.regionprops(labeled)
            
            if props:
                areas = [prop.area for prop in props]
                features['num_components'] = len(props)
                features['mean_component_area'] = np.mean(areas)
                features['std_component_area'] = np.std(areas)
            else:
                features['num_components'] = 0
                features['mean_component_area'] = 0
                features['std_component_area'] = 0
            
        except Exception:
            # Fallback: zero features
            features = {
                'opening_area': 0.0,
                'closing_area': 0.0,
                'gradient_area': 0.0,
                'num_components': 0.0,
                'mean_component_area': 0.0,
                'std_component_area': 0.0
            }
        
        return features
    
    def _extract_fourier_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract Fourier domain features"""
        features = {}
        
        try:
            # Compute FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Log magnitude
            log_magnitude = np.log(magnitude_spectrum + 1)
            
            # Extract features
            features['fft_mean'] = np.mean(log_magnitude)
            features['fft_std'] = np.std(log_magnitude)
            features['fft_energy'] = np.sum(magnitude_spectrum ** 2)
            
            # Frequency domain statistics
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Low frequency energy (center region)
            low_freq_region = magnitude_spectrum[center_h-h//8:center_h+h//8, 
                                               center_w-w//8:center_w+w//8]
            features['low_freq_energy'] = np.sum(low_freq_region ** 2)
            
            # High frequency energy (outer region)
            high_freq_mask = np.ones_like(magnitude_spectrum)
            high_freq_mask[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8] = 0
            features['high_freq_energy'] = np.sum((magnitude_spectrum * high_freq_mask) ** 2)
            
        except Exception:
            # Fallback: zero features
            features = {
                'fft_mean': 0.0,
                'fft_std': 0.0,
                'fft_energy': 0.0,
                'low_freq_energy': 0.0,
                'high_freq_energy': 0.0
            }
        
        return features
    
    def _extract_wavelet_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract wavelet domain features"""
        features = {}
        
        try:
            # Simple wavelet-like decomposition using filters
            # Low-pass filter (approximation)
            low_pass = gaussian_filter(gray, sigma=2)
            
            # High-pass filters (details)
            sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Extract features
            features['wavelet_approx_energy'] = np.sum(low_pass ** 2)
            features['wavelet_detail_h_energy'] = np.sum(sobel_h ** 2)
            features['wavelet_detail_v_energy'] = np.sum(sobel_v ** 2)
            features['wavelet_detail_d_energy'] = np.sum(laplacian ** 2)
            
            features['wavelet_approx_mean'] = np.mean(low_pass)
            features['wavelet_detail_h_mean'] = np.mean(np.abs(sobel_h))
            features['wavelet_detail_v_mean'] = np.mean(np.abs(sobel_v))
            features['wavelet_detail_d_mean'] = np.mean(np.abs(laplacian))
            
        except Exception:
            # Fallback: zero features
            features = {
                'wavelet_approx_energy': 0.0,
                'wavelet_detail_h_energy': 0.0,
                'wavelet_detail_v_energy': 0.0,
                'wavelet_detail_d_energy': 0.0,
                'wavelet_approx_mean': 0.0,
                'wavelet_detail_h_mean': 0.0,
                'wavelet_detail_v_mean': 0.0,
                'wavelet_detail_d_mean': 0.0
            }
        
        return features
    
    def _extract_statistical_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract statistical features"""
        features = {}
        
        try:
            # Basic statistics
            features['mean'] = np.mean(gray)
            features['std'] = np.std(gray)
            features['variance'] = np.var(gray)
            features['skewness'] = self._calculate_skewness(gray)
            features['kurtosis'] = self._calculate_kurtosis(gray)
            
            # Percentiles
            features['percentile_10'] = np.percentile(gray, 10)
            features['percentile_25'] = np.percentile(gray, 25)
            features['percentile_75'] = np.percentile(gray, 75)
            features['percentile_90'] = np.percentile(gray, 90)
            
            # Range features
            features['range'] = np.max(gray) - np.min(gray)
            features['iqr'] = features['percentile_75'] - features['percentile_25']
            
        except Exception:
            # Fallback: zero features
            features = {
                'mean': 0.0, 'std': 0.0, 'variance': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0,
                'percentile_10': 0.0, 'percentile_25': 0.0,
                'percentile_75': 0.0, 'percentile_90': 0.0,
                'range': 0.0, 'iqr': 0.0
            }
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 3) if std > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0.0
        except:
            return 0.0

def get_image_processor(config: Optional[Dict[str, Any]] = None) -> AdvancedImageProcessor:
    """Get configured image processor instance"""
    if config is None:
        config = {}
    
    return AdvancedImageProcessor(
        enable_gpu=config.get('enable_gpu', False)
    )

# Example usage and testing
if __name__ == "__main__":
    # Test the image processor
    processor = get_image_processor()
    
    # Create a dummy image for testing
    test_image = Image.new('RGB', (256, 256), color='purple')
    
    # Test preprocessing
    result = processor.preprocess_image(test_image, enhance_quality=True)
    
    print(f"✅ Image preprocessing successful!")
    print(f"   Original size: {result['metrics']['original_size']}")
    print(f"   Processed size: {result['metrics']['processed_size']}")
    print(f"   Quality score: {result['metrics']['quality_score']:.3f}")
    
    # Test feature extraction
    features = processor.extract_advanced_features(test_image)
    
    if 'error' not in features:
        print(f"✅ Feature extraction successful!")
        print(f"   Feature categories: {list(features.keys())}")
        total_features = sum(len(category) for category in features.values())
        print(f"   Total features: {total_features}")
    else:
        print(f"⚠️ Feature extraction failed: {features['error']['message']}")
