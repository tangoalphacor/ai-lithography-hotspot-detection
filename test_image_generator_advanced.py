"""
Advanced Test Image Generator for AI Lithography Hotspot Detection
=================================================================

This module provides comprehensive test image generation capabilities using multiple sources:
1. Procedural generation (geometric patterns, noise, synthetic hotspots)
2. API-based generation (Picsum, Lorem Picsum for base images)
3. Synthetic lithography pattern generation
4. Real-world style SEM image simulation

Features:
- Multiple image sizes and formats
- Lithography-specific patterns
- Hotspot simulation
- Batch generation
- Quality control metrics
"""

import streamlit as st
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
import zipfile
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import time
import random
from datetime import datetime

class TestImageGenerator:
    """Advanced test image generator with multiple sources and patterns"""
    
    def __init__(self):
        self.api_sources = {
            'picsum': 'https://picsum.photos',
            'placeholder': 'https://via.placeholder.com',
            'lorem_pixel': 'http://lorempixel.com',
            'unsplash_source': 'https://source.unsplash.com'
        }
        
        self.pattern_types = [
            'geometric_grid',
            'concentric_circles',
            'lithography_lines',
            'hotspot_simulation',
            'noise_patterns',
            'gradient_patterns',
            'checkerboard',
            'spiral_patterns',
            'random_shapes',
            'sem_style'
        ]
        
        self.standard_sizes = {
            'Small (256x256)': (256, 256),
            'Medium (512x512)': (512, 512),
            'Large (1024x1024)': (1024, 1024),
            'Wide (800x400)': (800, 400),
            'Tall (400x800)': (400, 800),
            'Ultra HD (2048x2048)': (2048, 2048),
            'Custom': None
        }

    def generate_procedural_image(self, width: int, height: int, pattern_type: str, 
                                 complexity: float = 0.5, add_hotspots: bool = True) -> Image.Image:
        """Generate procedural test images with lithography-specific patterns"""
        
        # Create base image
        img = Image.new('RGB', (width, height), 'black')
        draw = ImageDraw.Draw(img)
        
        if pattern_type == 'geometric_grid':
            return self._create_geometric_grid(img, draw, complexity)
        elif pattern_type == 'concentric_circles':
            return self._create_concentric_circles(img, draw, complexity)
        elif pattern_type == 'lithography_lines':
            return self._create_lithography_lines(img, draw, complexity, add_hotspots)
        elif pattern_type == 'hotspot_simulation':
            return self._create_hotspot_simulation(img, draw, complexity)
        elif pattern_type == 'noise_patterns':
            return self._create_noise_patterns(width, height, complexity)
        elif pattern_type == 'gradient_patterns':
            return self._create_gradient_patterns(width, height, complexity)
        elif pattern_type == 'checkerboard':
            return self._create_checkerboard(img, draw, complexity)
        elif pattern_type == 'spiral_patterns':
            return self._create_spiral_patterns(img, draw, complexity)
        elif pattern_type == 'random_shapes':
            return self._create_random_shapes(img, draw, complexity, add_hotspots)
        elif pattern_type == 'sem_style':
            return self._create_sem_style(width, height, complexity, add_hotspots)
        else:
            return self._create_default_pattern(img, draw)

    def _create_geometric_grid(self, img: Image.Image, draw: ImageDraw.Draw, complexity: float) -> Image.Image:
        """Create geometric grid patterns similar to lithography layouts"""
        width, height = img.size
        grid_size = int(20 + complexity * 50)
        
        # Draw grid lines
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill='white', width=2)
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill='white', width=2)
            
        # Add some filled rectangles
        for _ in range(int(10 * complexity)):
            x1 = random.randint(0, width - grid_size)
            y1 = random.randint(0, height - grid_size)
            x2 = x1 + random.randint(grid_size//2, grid_size*2)
            y2 = y1 + random.randint(grid_size//2, grid_size*2)
            draw.rectangle([x1, y1, x2, y2], fill='gray', outline='white')
            
        return img

    def _create_concentric_circles(self, img: Image.Image, draw: ImageDraw.Draw, complexity: float) -> Image.Image:
        """Create concentric circle patterns"""
        width, height = img.size
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2
        
        num_circles = int(5 + complexity * 15)
        for i in range(num_circles):
            radius = (i + 1) * max_radius // num_circles
            color = int(255 * (1 - i / num_circles))
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], 
                        outline=(color, color, color), width=2)
            
        return img

    def _create_lithography_lines(self, img: Image.Image, draw: ImageDraw.Draw, 
                                 complexity: float, add_hotspots: bool) -> Image.Image:
        """Create lithography-style line patterns with potential hotspots"""
        width, height = img.size
        line_spacing = int(10 + complexity * 20)
        
        # Horizontal lines
        for y in range(0, height, line_spacing):
            draw.line([(0, y), (width, y)], fill='white', width=2)
            
        # Vertical lines
        for x in range(0, width, line_spacing):
            draw.line([(x, 0), (x, height)], fill='white', width=2)
            
        # Add some diagonal lines for complexity
        if complexity > 0.5:
            for _ in range(int(5 * complexity)):
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = random.randint(0, width), random.randint(0, height)
                draw.line([(x1, y1), (x2, y2)], fill='gray', width=1)
        
        # Add hotspots (problematic areas)
        if add_hotspots:
            for _ in range(int(3 + complexity * 7)):
                x = random.randint(50, width - 50)
                y = random.randint(50, height - 50)
                size = random.randint(10, 30)
                # Create irregular hotspot shape
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill='red', outline='yellow', width=2)
                
        return img

    def _create_hotspot_simulation(self, img: Image.Image, draw: ImageDraw.Draw, complexity: float) -> Image.Image:
        """Create images specifically designed to test hotspot detection"""
        width, height = img.size
        
        # Create base lithography pattern
        base_pattern = self._create_lithography_lines(img, draw, 0.3, False)
        
        # Add various types of hotspots
        hotspot_types = ['bridge', 'pinch', 'corner', 'line_end', 'via_contact']
        num_hotspots = int(2 + complexity * 8)
        
        for _ in range(num_hotspots):
            hotspot_type = random.choice(hotspot_types)
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            
            if hotspot_type == 'bridge':
                # Two close parallel lines
                draw.rectangle([x-20, y-2, x+20, y+2], fill='white')
                draw.rectangle([x-20, y+8, x+20, y+12], fill='white')
                # Problematic bridge
                draw.rectangle([x-5, y+3, x+5, y+7], fill='red')
                
            elif hotspot_type == 'pinch':
                # Narrowing line
                for i in range(40):
                    width_var = max(1, 10 - abs(i - 20) // 3)
                    draw.rectangle([x+i-20, y-width_var, x+i-20+1, y+width_var], fill='white')
                # Mark the pinch point
                draw.circle([x, y], 5, fill='red')
                
            elif hotspot_type == 'corner':
                # Sharp corner
                draw.rectangle([x-15, y-2, x+2, y+2], fill='white')
                draw.rectangle([x-2, y-15, x+2, y+2], fill='white')
                # Mark potential rounding issue
                draw.circle([x, y], 3, fill='red')
                
        return base_pattern

    def _create_noise_patterns(self, width: int, height: int, complexity: float) -> Image.Image:
        """Create noise-based patterns for robustness testing"""
        # Generate noise array
        noise = np.random.random((height, width, 3))
        
        # Apply different noise types based on complexity
        if complexity < 0.3:
            # Gaussian noise
            noise = np.random.normal(0.5, 0.2, (height, width, 3))
        elif complexity < 0.6:
            # Salt and pepper noise
            noise = np.random.choice([0, 1], size=(height, width, 3), p=[0.7, 0.3])
        else:
            # Perlin-like noise
            for i in range(height):
                for j in range(width):
                    noise[i, j] = np.sin(i * 0.1) * np.cos(j * 0.1) * 0.5 + 0.5
                    
        # Convert to image
        noise = np.clip(noise * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(noise)

    def _create_gradient_patterns(self, width: int, height: int, complexity: float) -> Image.Image:
        """Create gradient patterns for illumination testing"""
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        if complexity < 0.5:
            # Linear gradient
            for i in range(width):
                intensity = int(255 * i / width)
                img_array[:, i] = [intensity, intensity, intensity]
        else:
            # Radial gradient
            center_x, center_y = width // 2, height // 2
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            for i in range(height):
                for j in range(width):
                    dist = np.sqrt((j - center_x)**2 + (i - center_y)**2)
                    intensity = int(255 * (1 - dist / max_dist))
                    img_array[i, j] = [intensity, intensity, intensity]
                    
        return Image.fromarray(img_array)

    def _create_checkerboard(self, img: Image.Image, draw: ImageDraw.Draw, complexity: float) -> Image.Image:
        """Create checkerboard patterns"""
        width, height = img.size
        square_size = int(20 + complexity * 40)
        
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    draw.rectangle([j, i, j + square_size, i + square_size], fill='white')
                    
        return img

    def _create_spiral_patterns(self, img: Image.Image, draw: ImageDraw.Draw, complexity: float) -> Image.Image:
        """Create spiral patterns"""
        width, height = img.size
        center_x, center_y = width // 2, height // 2
        
        # Draw spiral
        angle = 0
        radius = 0
        points = []
        
        while radius < min(width, height) // 2:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((int(x), int(y)))
            
            angle += 0.2
            radius += complexity * 2
            
        if len(points) > 1:
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill='white', width=2)
                
        return img

    def _create_random_shapes(self, img: Image.Image, draw: ImageDraw.Draw, 
                             complexity: float, add_hotspots: bool) -> Image.Image:
        """Create random geometric shapes"""
        width, height = img.size
        num_shapes = int(5 + complexity * 20)
        
        for _ in range(num_shapes):
            shape_type = random.choice(['rectangle', 'ellipse', 'line', 'polygon'])
            color = tuple(random.randint(100, 255) for _ in range(3))
            
            if shape_type == 'rectangle':
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = random.randint(x1, width), random.randint(y1, height)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
            elif shape_type == 'ellipse':
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = random.randint(x1, width), random.randint(y1, height)
                draw.ellipse([x1, y1, x2, y2], outline=color, width=2)
                
            elif shape_type == 'line':
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = random.randint(0, width), random.randint(0, height)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
                
        if add_hotspots:
            # Add some problematic overlapping areas
            for _ in range(int(2 + complexity * 5)):
                x = random.randint(20, width - 20)
                y = random.randint(20, height - 20)
                draw.ellipse([x-10, y-10, x+10, y+10], fill='red', outline='yellow')
                
        return img

    def _create_sem_style(self, width: int, height: int, complexity: float, add_hotspots: bool) -> Image.Image:
        """Create SEM-style images with realistic lithography patterns"""
        img_array = np.random.normal(128, 30, (height, width)).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L').convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Add realistic lithography structures
        num_structures = int(10 + complexity * 30)
        
        for _ in range(num_structures):
            structure_type = random.choice(['via', 'line', 'contact', 'metal_layer'])
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            
            if structure_type == 'via':
                size = random.randint(5, 15)
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill=(200, 200, 200), outline=(255, 255, 255))
                           
            elif structure_type == 'line':
                length = random.randint(30, 100)
                width_line = random.randint(2, 8)
                angle = random.uniform(0, 2 * np.pi)
                
                x2 = x + length * np.cos(angle)
                y2 = y + length * np.sin(angle)
                draw.line([(x, y), (int(x2), int(y2))], 
                         fill=(180, 180, 180), width=width_line)
                         
        # Add noise and blur for realistic SEM appearance
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        if add_hotspots:
            draw = ImageDraw.Draw(img)
            for _ in range(int(1 + complexity * 3)):
                x = random.randint(30, width - 30)
                y = random.randint(30, height - 30)
                # Create defect-like hotspot
                draw.ellipse([x-8, y-8, x+8, y+8], 
                           fill=(255, 100, 100), outline=(255, 0, 0))
                           
        return img

    def _create_default_pattern(self, img: Image.Image, draw: ImageDraw.Draw) -> Image.Image:
        """Create default pattern if type not recognized"""
        width, height = img.size
        
        # Simple grid pattern
        for i in range(0, width, 50):
            draw.line([(i, 0), (i, height)], fill='white')
        for j in range(0, height, 50):
            draw.line([(0, j), (width, j)], fill='white')
            
        return img

    def fetch_api_image(self, width: int, height: int, source: str = 'picsum') -> Optional[Image.Image]:
        """Fetch image from API source and process for lithography testing"""
        try:
            if source == 'picsum':
                url = f"{self.api_sources['picsum']}/{width}/{height}?random={int(time.time())}"
            elif source == 'placeholder':
                url = f"{self.api_sources['placeholder']}/{width}x{height}/CCCCCC/FFFFFF?text=Test+Image"
            elif source == 'unsplash_source':
                url = f"{self.api_sources['unsplash_source']}/{width}x{height}/?sig={int(time.time())}"
            else:
                return None
                
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                # Convert to grayscale for lithography testing
                img = img.convert('L').convert('RGB')
                return img
            else:
                return None
                
        except Exception as e:
            st.error(f"Failed to fetch image from {source}: {str(e)}")
            return None

    def generate_batch(self, sizes: List[Tuple[int, int]], patterns: List[str], 
                      complexity: float = 0.5, include_api: bool = True) -> Dict[str, List[Image.Image]]:
        """Generate batch of test images"""
        results = {}
        
        for size in sizes:
            width, height = size
            size_key = f"{width}x{height}"
            results[size_key] = []
            
            # Generate procedural images
            for pattern in patterns:
                try:
                    img = self.generate_procedural_image(width, height, pattern, complexity, True)
                    results[size_key].append((img, f"{pattern}_{size_key}"))
                except Exception as e:
                    st.warning(f"Failed to generate {pattern} pattern: {str(e)}")
                    
            # Add API-based images if requested
            if include_api:
                for source in ['picsum', 'placeholder']:
                    try:
                        api_img = self.fetch_api_image(width, height, source)
                        if api_img:
                            results[size_key].append((api_img, f"api_{source}_{size_key}"))
                    except Exception as e:
                        st.warning(f"Failed to fetch from {source}: {str(e)}")
                        
        return results

    def create_zip_package(self, images_dict: Dict[str, List[Tuple[Image.Image, str]]]) -> bytes:
        """Create ZIP package of generated images"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add metadata
            metadata = {
                "generated_at": datetime.now().isoformat(),
                "generator_version": "2.0.0",
                "total_images": sum(len(imgs) for imgs in images_dict.values()),
                "sizes": list(images_dict.keys()),
                "description": "Test images for AI Lithography Hotspot Detection"
            }
            
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            # Add images
            for size_key, image_list in images_dict.items():
                for i, (img, name) in enumerate(image_list):
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    filename = f"{size_key}/{name}.png"
                    zip_file.writestr(filename, img_buffer.getvalue())
                    
            # Add README
            readme_content = """# Test Images for AI Lithography Hotspot Detection

## Contents
This package contains test images generated for evaluating AI-based lithography hotspot detection systems.

## Image Types
- **Geometric Patterns**: Grid patterns, concentric circles
- **Lithography Patterns**: Line patterns with simulated hotspots
- **Noise Patterns**: Various noise types for robustness testing
- **SEM-style**: Simulated SEM images with realistic defects
- **API Images**: External images processed for testing

## Usage
Upload these images to the AI Lithography Hotspot Detection application to test:
- Model accuracy and robustness
- Visualization capabilities
- Processing speed
- User interface responsiveness

## Hotspot Locations
Images with 'hotspot' in the name contain simulated defects marked in red/yellow colors.
"""
            zip_file.writestr("README.md", readme_content)
            
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

def create_test_image_generator_ui():
    """Create Streamlit UI for test image generation"""
    st.header("🎨 Advanced Test Image Generator")
    st.markdown("""
    Generate high-quality test images for lithography hotspot detection with multiple patterns,
    sizes, and sources including API integration and procedural generation.
    """)
    
    generator = TestImageGenerator()
    
    # Configuration section
    with st.expander("⚙️ Generation Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📐 Image Sizes")
            selected_sizes = st.multiselect(
                "Select image sizes:",
                options=list(generator.standard_sizes.keys()),
                default=['Medium (512x512)', 'Large (1024x1024)']
            )
            
            # Custom size option
            if 'Custom' in selected_sizes:
                custom_width = st.number_input("Custom Width:", min_value=64, max_value=4096, value=512)
                custom_height = st.number_input("Custom Height:", min_value=64, max_value=4096, value=512)
        
        with col2:
            st.subheader("🎭 Pattern Types")
            selected_patterns = st.multiselect(
                "Select pattern types:",
                options=generator.pattern_types,
                default=['lithography_lines', 'hotspot_simulation', 'sem_style']
            )
            
        with col3:
            st.subheader("🔧 Advanced Options")
            complexity = st.slider("Pattern Complexity:", 0.0, 1.0, 0.5, 0.1)
            include_api = st.checkbox("Include API images", value=True)
            add_hotspots = st.checkbox("Add simulated hotspots", value=True)
    
    # Generation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Generate Preview", type="primary", use_container_width=True):
            if selected_sizes and selected_patterns:
                # Generate single preview image
                size_name = selected_sizes[0]
                pattern = selected_patterns[0]
                
                if size_name == 'Custom':
                    size = (custom_width, custom_height)
                else:
                    size = generator.standard_sizes[size_name]
                
                with st.spinner(f"Generating preview: {pattern} ({size[0]}x{size[1]})..."):
                    try:
                        preview_img = generator.generate_procedural_image(
                            size[0], size[1], pattern, complexity, add_hotspots
                        )
                        
                        st.success("✅ Preview generated!")
                        st.image(preview_img, caption=f"{pattern} - {size[0]}x{size[1]}", use_column_width=True)
                        
                        # Download option for preview
                        img_buffer = io.BytesIO()
                        preview_img.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Preview",
                            data=img_buffer.getvalue(),
                            file_name=f"preview_{pattern}_{size[0]}x{size[1]}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Failed to generate preview: {str(e)}")
            else:
                st.warning("Please select at least one size and one pattern type.")
    
    with col2:
        if st.button("📦 Generate Batch", use_container_width=True):
            if selected_sizes and selected_patterns:
                # Prepare sizes list
                sizes_list = []
                for size_name in selected_sizes:
                    if size_name == 'Custom':
                        sizes_list.append((custom_width, custom_height))
                    else:
                        sizes_list.append(generator.standard_sizes[size_name])
                
                total_images = len(sizes_list) * len(selected_patterns)
                if include_api:
                    total_images += len(sizes_list) * 2  # picsum + placeholder
                
                with st.spinner(f"Generating {total_images} test images..."):
                    try:
                        # Generate batch
                        progress_bar = st.progress(0)
                        results = {}
                        
                        for i, size in enumerate(sizes_list):
                            width, height = size
                            size_key = f"{width}x{height}"
                            results[size_key] = []
                            
                            # Generate procedural images
                            for j, pattern in enumerate(selected_patterns):
                                img = generator.generate_procedural_image(
                                    width, height, pattern, complexity, add_hotspots
                                )
                                results[size_key].append((img, f"{pattern}_{size_key}"))
                                
                                progress = (i * len(selected_patterns) + j + 1) / total_images
                                progress_bar.progress(progress)
                            
                            # Add API images
                            if include_api:
                                for source in ['picsum', 'placeholder']:
                                    api_img = generator.fetch_api_image(width, height, source)
                                    if api_img:
                                        results[size_key].append((api_img, f"api_{source}_{size_key}"))
                        
                        # Create ZIP package
                        zip_data = generator.create_zip_package(results)
                        
                        st.success(f"✅ Generated {total_images} test images successfully!")
                        
                        # Display summary
                        st.subheader("📊 Generation Summary")
                        for size_key, image_list in results.items():
                            st.write(f"**{size_key}**: {len(image_list)} images")
                        
                        # Download button
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="📥 Download ZIP Package",
                            data=zip_data,
                            file_name=f"test_images_{timestamp}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Batch generation failed: {str(e)}")
                        
            else:
                st.warning("Please select at least one size and one pattern type.")
    
    with col3:
        if st.button("🌐 Fetch API Sample", use_container_width=True):
            if selected_sizes:
                size_name = selected_sizes[0]
                if size_name == 'Custom':
                    size = (custom_width, custom_height)
                else:
                    size = generator.standard_sizes[size_name]
                
                with st.spinner("Fetching sample from Picsum API..."):
                    try:
                        api_img = generator.fetch_api_image(size[0], size[1], 'picsum')
                        if api_img:
                            st.success("✅ API sample fetched!")
                            st.image(api_img, caption=f"Picsum API - {size[0]}x{size[1]}", use_column_width=True)
                            
                            # Download option
                            img_buffer = io.BytesIO()
                            api_img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download API Sample",
                                data=img_buffer.getvalue(),
                                file_name=f"api_sample_{size[0]}x{size[1]}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to fetch API sample")
                    except Exception as e:
                        st.error(f"API fetch failed: {str(e)}")
            else:
                st.warning("Please select at least one size.")

    # Information section
    with st.expander("ℹ️ About Test Image Generation"):
        st.markdown("""
        ### 🎯 Purpose
        This tool generates comprehensive test images for evaluating AI lithography hotspot detection systems.
        
        ### 📋 Pattern Types
        - **Geometric Grid**: Regular grid patterns for basic testing
        - **Concentric Circles**: Radial patterns for rotation invariance
        - **Lithography Lines**: Realistic line patterns with simulated hotspots
        - **Hotspot Simulation**: Specific defect patterns (bridges, pinches, corners)
        - **Noise Patterns**: Various noise types for robustness testing
        - **SEM Style**: Simulated SEM images with realistic appearance
        
        ### 🌐 API Sources
        - **Picsum**: Random images for diverse testing scenarios
        - **Placeholder**: Solid color backgrounds for controlled testing
        
        ### 💡 Usage Tips
        - Use lower complexity for basic testing, higher for stress testing
        - Include API images for testing with real-world textures
        - Generate multiple sizes to test scalability
        - Enable hotspots to test detection accuracy
        """)

if __name__ == "__main__":
    create_test_image_generator_ui()
