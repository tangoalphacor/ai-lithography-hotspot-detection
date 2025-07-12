"""
Basic Test Image Generator for Fallback Support
=============================================

Simple test image generator that works without advanced dependencies.
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import random
from datetime import datetime

def create_basic_test_image_generator():
    """Create basic test image generator UI"""
    st.header("🎨 Basic Test Image Generator")
    st.markdown("Generate simple test images for lithography hotspot detection testing.")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Image Settings")
        width = st.number_input("Width:", min_value=64, max_value=2048, value=512)
        height = st.number_input("Height:", min_value=64, max_value=2048, value=512)
        
        pattern_type = st.selectbox(
            "Pattern Type:",
            ["grid", "circles", "lines", "checkerboard", "noise"]
        )
    
    with col2:
        st.subheader("🎭 Pattern Settings")
        complexity = st.slider("Complexity:", 0.1, 1.0, 0.5, 0.1)
        add_hotspots = st.checkbox("Add simulated hotspots", value=True)
        
        if add_hotspots:
            num_hotspots = st.slider("Number of hotspots:", 1, 10, 3)
    
    # Generate button
    if st.button("🚀 Generate Test Image", type="primary"):
        with st.spinner("Generating test image..."):
            # Create base image
            img = Image.new('RGB', (width, height), 'black')
            draw = ImageDraw.Draw(img)
            
            # Generate pattern
            if pattern_type == "grid":
                grid_size = int(20 + complexity * 50)
                for x in range(0, width, grid_size):
                    draw.line([(x, 0), (x, height)], fill='white', width=2)
                for y in range(0, height, grid_size):
                    draw.line([(0, y), (width, y)], fill='white', width=2)
                    
            elif pattern_type == "circles":
                center_x, center_y = width // 2, height // 2
                max_radius = min(width, height) // 2
                num_circles = int(5 + complexity * 10)
                
                for i in range(num_circles):
                    radius = (i + 1) * max_radius // num_circles
                    color = int(255 * (1 - i / num_circles))
                    draw.ellipse([center_x - radius, center_y - radius, 
                                center_x + radius, center_y + radius], 
                               outline=(color, color, color), width=2)
                               
            elif pattern_type == "lines":
                line_spacing = int(10 + complexity * 30)
                for y in range(0, height, line_spacing):
                    draw.line([(0, y), (width, y)], fill='white', width=2)
                for x in range(0, width, line_spacing):
                    draw.line([(x, 0), (x, height)], fill='white', width=2)
                    
            elif pattern_type == "checkerboard":
                square_size = int(20 + complexity * 40)
                for i in range(0, height, square_size):
                    for j in range(0, width, square_size):
                        if (i // square_size + j // square_size) % 2 == 0:
                            draw.rectangle([j, i, j + square_size, i + square_size], fill='white')
                            
            elif pattern_type == "noise":
                # Generate noise array
                noise = np.random.random((height, width, 3))
                noise = np.clip(noise * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(noise)
                draw = ImageDraw.Draw(img)
            
            # Add hotspots if requested
            if add_hotspots:
                for _ in range(num_hotspots):
                    x = random.randint(20, width - 20)
                    y = random.randint(20, height - 20)
                    size = random.randint(5, 15)
                    draw.ellipse([x-size, y-size, x+size, y+size], 
                               fill='red', outline='yellow', width=2)
            
            st.success("✅ Test image generated!")
            st.image(img, caption=f"{pattern_type.title()} Pattern ({width}x{height})", use_column_width=True)
            
            # Download option
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Test Image",
                data=img_buffer.getvalue(),
                file_name=f"test_image_{pattern_type}_{timestamp}.png",
                mime="image/png"
            )
    
    # Information
    with st.expander("ℹ️ About Basic Test Images"):
        st.markdown("""
        ### 🎯 Purpose
        Generate simple test images for basic functionality testing.
        
        ### 📋 Available Patterns
        - **Grid**: Regular grid patterns
        - **Circles**: Concentric circle patterns  
        - **Lines**: Horizontal and vertical lines
        - **Checkerboard**: Alternating squares
        - **Noise**: Random noise patterns
        
        ### 💡 Usage
        - Use these images to test basic upload and processing functionality
        - Simulated hotspots (red circles) test detection capabilities
        - Adjust complexity to vary pattern density
        """)

if __name__ == "__main__":
    create_basic_test_image_generator()
