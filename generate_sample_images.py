"""
Sample Image Generator for Banknote Authentication Demo
======================================================

This script creates synthetic banknote images with different characteristics
to demonstrate the GUI application's functionality.
"""

import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random


class BanknoteImageGenerator:
    """Generate synthetic banknote images for testing"""
    
    def __init__(self, output_dir='sample_images'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Standard banknote dimensions (scaled down)
        self.width = 400
        self.height = 200
    
    def generate_genuine_banknote(self, filename):
        """Generate a synthetic genuine banknote image"""
        # Create base image with consistent patterns
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        
        # Add consistent geometric patterns
        self.add_geometric_patterns(image, regularity=0.9)
        self.add_text_elements(image, clarity=0.9)
        self.add_security_features(image, quality=0.9)
        
        # Add subtle noise (genuine banknotes have minimal noise)
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        cv2.imwrite(str(self.output_dir / filename), image)
        print(f"Generated genuine banknote: {filename}")
    
    def generate_forged_banknote(self, filename):
        """Generate a synthetic forged banknote image"""
        # Create base image with more variations
        base_color = random.randint(220, 250)
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * base_color
        
        # Add irregular patterns (typical of forgeries)
        self.add_geometric_patterns(image, regularity=0.6)
        self.add_text_elements(image, clarity=0.7)
        self.add_security_features(image, quality=0.5)
        
        # Add more noise (forgeries often have printing artifacts)
        noise = np.random.normal(0, 15, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add printing artifacts
        self.add_printing_artifacts(image)
        
        # Save image
        cv2.imwrite(str(self.output_dir / filename), image)
        print(f"Generated forged banknote: {filename}")
    
    def add_geometric_patterns(self, image, regularity=0.8):
        """Add geometric patterns to the banknote"""
        h, w = image.shape[:2]
        
        # Add grid pattern
        grid_spacing = int(20 * regularity + 10 * (1 - regularity) * random.random())
        
        for i in range(0, h, grid_spacing):
            cv2.line(image, (0, i), (w, i), (200, 200, 200), 1)
        
        for j in range(0, w, grid_spacing):
            cv2.line(image, (j, 0), (j, h), (200, 200, 200), 1)
        
        # Add circles
        num_circles = int(5 * regularity + 3 * (1 - regularity))
        for _ in range(num_circles):
            center_x = random.randint(50, w - 50)
            center_y = random.randint(30, h - 30)
            radius = random.randint(10, 25)
            color = tuple([random.randint(180, 220) for _ in range(3)])
            cv2.circle(image, (center_x, center_y), radius, color, 2)
    
    def add_text_elements(self, image, clarity=0.8):
        """Add text elements to simulate banknote text"""
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to use a default font, fallback to default if not available
        try:
            font_size = int(24 * clarity)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Add sample text
        texts = ["SAMPLE", "BANK NOTE", "100", "SECURE"]
        positions = [(50, 30), (250, 30), (50, 120), (250, 120)]
        
        for text, pos in zip(texts, positions):
            # Add some jitter for forgeries
            if clarity < 0.8:
                pos = (pos[0] + random.randint(-3, 3), pos[1] + random.randint(-2, 2))
            
            color = tuple([random.randint(50, 100) for _ in range(3)])
            draw.text(pos, text, fill=color, font=font)
        
        # Convert back to OpenCV format
        image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def add_security_features(self, image, quality=0.8):
        """Add security features like watermarks and microprint"""
        h, w = image.shape[:2]
        
        # Add watermark-like overlay
        overlay = np.zeros_like(image)
        
        # Create diamond pattern
        diamond_size = int(30 * quality)
        for y in range(0, h, diamond_size * 2):
            for x in range(0, w, diamond_size * 2):
                pts = np.array([
                    [x + diamond_size, y],
                    [x + diamond_size * 2, y + diamond_size],
                    [x + diamond_size, y + diamond_size * 2],
                    [x, y + diamond_size]
                ], np.int32)
                cv2.fillPoly(overlay, [pts], (30, 30, 30))
        
        # Blend with original
        alpha = 0.1 * quality
        image[:] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    def add_printing_artifacts(self, image):
        """Add printing artifacts typical of forgeries"""
        h, w = image.shape[:2]
        
        # Add random streaks
        for _ in range(random.randint(2, 8)):
            start_y = random.randint(0, h)
            end_y = random.randint(0, h)
            x = random.randint(0, w)
            color = tuple([random.randint(100, 150) for _ in range(3)])
            cv2.line(image, (x, start_y), (x, end_y), color, random.randint(1, 3))
        
        # Add irregular dots
        for _ in range(random.randint(10, 30)):
            x = random.randint(0, w)
            y = random.randint(0, h)
            color = tuple([random.randint(0, 100) for _ in range(3)])
            cv2.circle(image, (x, y), random.randint(1, 3), color, -1)
    
    def generate_sample_set(self, num_genuine=5, num_forged=5):
        """Generate a complete set of sample images"""
        print(f"Generating sample banknote images in: {self.output_dir}")
        
        # Generate genuine banknotes
        for i in range(num_genuine):
            filename = f"genuine_banknote_{i+1:02d}.jpg"
            self.generate_genuine_banknote(filename)
        
        # Generate forged banknotes
        for i in range(num_forged):
            filename = f"forged_banknote_{i+1:02d}.jpg"
            self.generate_forged_banknote(filename)
        
        print(f"\nGenerated {num_genuine + num_forged} sample images!")
        print(f"Images saved to: {self.output_dir.absolute()}")


def main():
    """Generate sample images for the GUI demo"""
    generator = BanknoteImageGenerator()
    generator.generate_sample_set(num_genuine=6, num_forged=6)
    
    # Create a readme file for the samples
    readme_content = """
# Sample Banknote Images

This directory contains synthetic banknote images for testing the GUI application.

## Files:
- `genuine_banknote_*.jpg` - Synthetic genuine banknotes with consistent patterns
- `forged_banknote_*.jpg` - Synthetic forged banknotes with irregular patterns

## Usage:
1. Run the GUI application: `python gui_banknote_classifier.py`
2. Click "Use Sample Images" to automatically load the first sample
3. Or use "Choose Image File" to select specific sample images

## Note:
These are synthetic images created for demonstration purposes only.
They simulate the visual characteristics that might distinguish genuine from forged banknotes.
    """.strip()
    
    with open(generator.output_dir / "README.md", "w") as f:
        f.write(readme_content)


if __name__ == "__main__":
    main()
