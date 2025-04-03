import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wall_detection.detector import detect_walls, draw_walls
from src.wall_detection.image_utils import load_image, save_image, convert_to_rgb

def process_image(image_path, min_contour_area=100, blur_kernel_size=5, canny_threshold1=50, canny_threshold2=150,
                  wall_color=None, color_threshold=20):
    """Process an image to detect walls and return the result."""
    # Load the image
    image = load_image(image_path)
    
    # Detect walls with adjustable parameters
    contours = detect_walls(
        image, 
        min_contour_area=min_contour_area,
        max_contour_area=None,  # Always use None
        blur_kernel_size=blur_kernel_size,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
        wall_color=wall_color,
        color_threshold=color_threshold
    )
    
    # Draw walls on the image
    image_with_walls = draw_walls(image, contours)
    
    # Convert to RGB for display
    image_with_walls_rgb = convert_to_rgb(image_with_walls)
    
    return image_with_walls_rgb

def save_detected_walls(image, output_path):
    """Save the processed image with detected walls to the specified output path."""
    # Convert back to BGR for saving
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    save_image(image_bgr, output_path)
    print(f"Processed image saved to {output_path}")

def display_image(image):
    """Display the image using matplotlib."""
    plt.figure(figsize=(10, 13))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def main(input_image_path, output_image_path, min_contour_area=100, blur_kernel_size=5, canny_threshold1=50, canny_threshold2=150, 
         wall_color=None, color_threshold=20, show_result=True):
    """Main function to detect walls in an image and save/show the result."""
    # Process the image
    result_image = process_image(input_image_path, min_contour_area, blur_kernel_size, canny_threshold1, canny_threshold2,
                                 wall_color, color_threshold)
    
    # Save the result
    save_detected_walls(result_image, output_image_path)
    
    # Display the result if requested
    if show_result:
        display_image(result_image)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Detect walls in images.')
    parser.add_argument('--input', '-i', default="data/input/CavernPitPublic-785x1024.jpg", help='Path to input image')
    parser.add_argument('--output', '-o', default="data/output/detected_walls.png", help='Path to save output image')
    parser.add_argument('--min-area', '-a', type=int, default=100, help='Minimum contour area to filter small artifacts')
    parser.add_argument('--blur', '-b', type=int, default=5, help='Gaussian blur kernel size (odd number)')
    parser.add_argument('--canny1', '-c1', type=int, default=50, help='Canny edge detection lower threshold')
    parser.add_argument('--canny2', '-c2', type=int, default=150, help='Canny edge detection upper threshold')
    parser.add_argument('--wall-color', type=str, help='Wall color in hex format (e.g., #000000 for black)')
    parser.add_argument('--color-threshold', type=int, default=20, help='Color threshold (0-100)')
    args = parser.parse_args()
    
    # Parse wall color if provided
    wall_color = None
    if args.wall_color:
        hex_color = args.wall_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        wall_color = (b, g, r)  # OpenCV uses BGR
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    main(args.input, args.output, args.min_area, args.blur, args.canny1, args.canny2,
         wall_color=wall_color, color_threshold=args.color_threshold)