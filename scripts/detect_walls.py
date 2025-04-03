import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wall_detection.detector import detect_walls, draw_walls
from src.wall_detection.image_utils import load_image, save_image, convert_to_rgb

def process_image(image_path):
    """Process an image to detect walls and return the result."""
    # Load the image
    image = load_image(image_path)
    
    # Detect walls
    contours = detect_walls(image)
    
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

def main(input_image_path, output_image_path, show_result=True):
    """Main function to detect walls in an image and save/show the result."""
    # Process the image
    result_image = process_image(input_image_path)
    
    # Save the result
    save_detected_walls(result_image, output_image_path)
    
    # Display the result if requested
    if show_result:
        display_image(result_image)

if __name__ == "__main__":
    # Default paths
    input_image = "data/input/CavernPitPublic-785x1024.jpg"
    output_image = "data/output/detected_walls.png"
    
    # Check if paths are provided as arguments
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    if len(sys.argv) > 2:
        output_image = sys.argv[2]
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    
    main(input_image, output_image)