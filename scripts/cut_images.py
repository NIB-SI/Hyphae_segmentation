#!/usr/bin/env python3
"""
Circle cutting for well plate images.
Extracts circular region (2880px diameter) from center.
"""

import os
from PIL import Image, ImageDraw

def cut_single_image(input_path, output_path, diameter=2880):
    """Cut circular region from image center."""
    with Image.open(input_path) as img:
        img = img.convert("RGBA")
        
        # Create circular mask
        output = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
        mask = Image.new("L", (diameter, diameter), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([0, 0, diameter, diameter], fill=255)
        
        # Crop from center
        center_x, center_y = img.width // 2, img.height // 2
        left = center_x - diameter // 2
        top = center_y - diameter // 2
        cropped = img.crop((left, top, left + diameter, top + diameter))
        
        # Apply mask and save
        output.paste(cropped, (0, 0), mask)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output.save(output_path, "PNG")


def cut_folder(input_folder, output_folder):
    """Cut all images in folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    count = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            
            if not os.path.exists(output_path):
                cut_single_image(input_path, output_path)
                count += 1
    
    return count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cut circular regions from well plate images")
    parser.add_argument('input_folder', type=str, help='Input folder with images')
    parser.add_argument('output_folder', type=str, help='Output folder for cut images')
    
    args = parser.parse_args()
    
    count = cut_folder(args.input_folder, args.output_folder)
    print(f"Cut {count} images")
