import cv2
import numpy as np
import os
from urllib.parse import urlparse
import requests
from io import BytesIO
import traceback

def load_image(image_path):
    """
    Load an image from the specified path with enhanced WebP support.
    WebP images are converted to PNG format in memory for consistent color handling.
    
    Parameters:
    - image_path: Path to the image file or URL
    
    Returns:
    - Image as a NumPy array
    """
    try:
        # Check if path is a URL
        is_url = False
        if isinstance(image_path, str) and (image_path.startswith('http://') or image_path.startswith('https://')):
            is_url = True
            
        # Get file extension
        if is_url:
            parsed_url = urlparse(image_path)
            file_extension = os.path.splitext(parsed_url.path)[1].lower()
        else:
            file_extension = os.path.splitext(image_path)[1].lower()
        
        # Handle WebP specifically for better quality and compatibility
        is_webp = file_extension == '.webp'
        
        # Load image
        image = None
        if is_url:
            # Handle URL loading
            response = requests.get(image_path)
            img_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        else:
            # Try standard loading first for all formats
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            # If standard loading failed for WebP, try alternate approach
            if image is None and is_webp:
                print(f"Standard loading failed for WebP file: {image_path}, trying alternate method...")
                try:
                    # Read raw bytes first
                    with open(image_path, 'rb') as f:
                        img_bytes = f.read()
                    
                    # Decode WebP with maximum quality
                    img_array = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                    
                    if image is None:
                        # If still failed, try loading with standard IMREAD_COLOR flag
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                except Exception as e:
                    print(f"Failed to load WebP with alternate method: {e}")
                    traceback.print_exc()
                    # Fall back to original method as a last resort
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            raise FileNotFoundError(f"Image not found or couldn't be loaded from {image_path}")
        
        # Get original dimensions for debugging
        original_shape = image.shape
        
        # Process alpha channel consistently for ALL image formats
        if len(image.shape) > 2 and image.shape[2] == 4:
            # Extract the alpha channel
            alpha_channel = image[:, :, 3]
            
            # Check if alpha channel has any transparency (non-255 values)
            has_transparency = np.any(alpha_channel < 255)
            
            if has_transparency:
                # Create a white background
                white_background = np.ones_like(image[:, :, :3], dtype=np.uint8) * 255
                
                # Extract alpha channel and normalize to 0-1
                alpha = alpha_channel / 255.0
                
                # Expand dimensions for broadcasting
                alpha = np.expand_dims(alpha, axis=2)
                
                # Blend image with white background based on alpha
                image = (image[:, :, :3] * alpha + white_background * (1 - alpha)).astype(np.uint8)
            else:
                # If alpha is all 255 (fully opaque), just drop the channel
                image = image[:, :, :3]
                
        # CRITICAL FIX: For WebP files, ensure identical color processing as PNG
        if is_webp:
            print(f"Converting WebP to PNG format in memory for consistent color handling...")
            print(f"Original WebP shape: {original_shape}, Current shape: {image.shape}")
            
            # Always convert through PNG format in memory to ensure consistent color handling
            # Use compression=0 for lossless conversion
            _, png_data = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            image = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)
            
            print(f"After PNG conversion shape: {image.shape}")
        
        return image
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        traceback.print_exc()
        raise

def save_image(image, save_path):
    """Save an image to the specified path with format-appropriate settings."""
    # Get file extension (including the period)
    file_extension = os.path.splitext(save_path)[1].lower()
    
    # Use appropriate parameters based on file extension
    if file_extension == '.webp':
        # Use high quality WebP encoding (100 is max quality)
        params = [cv2.IMWRITE_WEBP_QUALITY, 98]
        cv2.imwrite(save_path, image, params)
    elif file_extension == '.png':
        # Use best compression for PNG
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        cv2.imwrite(save_path, image, params)
    elif file_extension == '.jpg' or file_extension == '.jpeg':
        # Use high quality JPEG encoding
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        cv2.imwrite(save_path, image, params)
    else:
        # Default for other formats
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
    """
    Convert BGR image to RGB for display with enhanced error handling.
    Ensures consistent color handling between PNG and WebP formats.
    """
    try:
        if image is None:
            print("Warning: Attempted to convert None image to RGB")
            # Return a small red image as an error indicator
            return np.ones((100, 100, 3), dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8)
            
        if image.ndim == 2:  # Handle grayscale images
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # Handle BGRA images (like some WebP)
            # Create a white background
            white_background = np.ones_like(image[:, :, :3], dtype=np.uint8) * 255
            
            # Extract alpha channel and normalize to 0-1
            alpha = image[:, :, 3] / 255.0
            
            # Expand dimensions for broadcasting
            alpha = np.expand_dims(alpha, axis=2)
            
            # Blend image with white background based on alpha - ensures consistent color values
            blended = (image[:, :, :3] * alpha + white_background * (1 - alpha)).astype(np.uint8)
            
            # Convert from BGR to RGB
            return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        else:  # Regular BGR image
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error converting image to RGB: {e}")
        traceback.print_exc()
        # Return a red image to indicate an error
        return np.ones((100, 100, 3), dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8)
