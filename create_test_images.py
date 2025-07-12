"""
Create sample test images for the Streamlit app testing
"""
import numpy as np
from PIL import Image
import os

def create_test_images():
    """Create various test images for comprehensive testing"""
    
    # Create test_images directory
    os.makedirs('test_images', exist_ok=True)
    
    # Test Image 1: Small gradient image
    print("Creating small_test.jpg...")
    img_array = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            img_array[i, j] = [
                int(255 * i / 256),  # Red gradient
                int(255 * j / 256),  # Green gradient
                128  # Blue constant
            ]
    img = Image.fromarray(img_array)
    img.save('test_images/small_test.jpg')
    
    # Test Image 2: Medium checkerboard
    print("Creating medium_test.png...")
    img_array = np.zeros((512, 512, 3), dtype=np.uint8)
    square_size = 32
    for i in range(512):
        for j in range(512):
            if ((i // square_size) + (j // square_size)) % 2:
                img_array[i, j] = [255, 255, 255]  # White
            else:
                img_array[i, j] = [50, 50, 50]  # Dark gray
    img = Image.fromarray(img_array)
    img.save('test_images/medium_test.png')
    
    # Test Image 3: Large circles pattern
    print("Creating large_test.jpg...")
    img_array = np.zeros((1024, 1024, 3), dtype=np.uint8)
    center_x, center_y = 512, 512
    for i in range(1024):
        for j in range(1024):
            distance = np.sqrt((j - center_x)**2 + (i - center_y)**2)
            intensity = int(255 * (np.sin(distance / 20) + 1) / 2)
            img_array[i, j] = [intensity, 100, 200]
    img = Image.fromarray(img_array)
    img.save('test_images/large_test.jpg')
    
    # Test Image 4: Wide aspect ratio
    print("Creating wide_test.png...")
    img_array = np.random.randint(0, 256, (400, 800, 3), dtype=np.uint8)
    # Add some structure
    for i in range(0, 400, 50):
        img_array[i:i+10, :] = [255, 255, 0]  # Yellow lines
    img = Image.fromarray(img_array)
    img.save('test_images/wide_test.png')
    
    # Test Image 5: Tall aspect ratio
    print("Creating tall_test.jpg...")
    img_array = np.random.randint(0, 256, (600, 300, 3), dtype=np.uint8)
    # Add some structure
    for j in range(0, 300, 30):
        img_array[:, j:j+5] = [0, 255, 255]  # Cyan lines
    img = Image.fromarray(img_array)
    img.save('test_images/tall_test.jpg')
    
    print("\n✅ All test images created successfully!")
    print("📁 Location: test_images/ folder")
    print("\n📋 Test Images Created:")
    print("  • small_test.jpg (256x256) - Gradient pattern")
    print("  • medium_test.png (512x512) - Checkerboard pattern")
    print("  • large_test.jpg (1024x1024) - Circles pattern")
    print("  • wide_test.png (400x800) - Wide aspect ratio")
    print("  • tall_test.jpg (600x300) - Tall aspect ratio")
    
    return True

if __name__ == "__main__":
    create_test_images()
