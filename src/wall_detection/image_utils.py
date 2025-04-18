import cv2
import numpy as np

def load_image(image_path):
    """Load an image from the specified path."""
    # Use IMREAD_UNCHANGED to properly handle WebP files and transparency
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Handle WebP/PNG images that might have alpha channel (4 channels)
    if len(image.shape) > 2 and image.shape[2] == 4:
        # Convert RGBA to BGR by removing alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
    return image

def save_image(image, save_path):
    """Save an image to the specified path."""
    # Get file extension (including the period)
    file_extension = save_path[save_path.rfind('.'):].lower()
    
    # Use appropriate parameters based on file extension
    if file_extension == '.webp':
        cv2.imwrite(save_path, image, [cv2.IMWRITE_WEBP_QUALITY, 95])
    else:
        cv2.imwrite(save_path, image)

def preprocess_image(image):
    """Preprocess an image for wall detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred

def detect_edges(image, low_threshold=50, high_threshold=150):
    """Detect edges in an image using Canny edge detection."""
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

def convert_to_rgb(image):
    """Convert BGR image to RGB for display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
