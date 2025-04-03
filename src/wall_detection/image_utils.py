import cv2
import numpy as np

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def save_image(image, save_path):
    """Save an image to the specified path."""
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
