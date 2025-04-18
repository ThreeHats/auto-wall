def load_image(image_path):
    """Load an image from the specified file path."""
    import cv2
    # Use IMREAD_UNCHANGED to properly handle WebP files and transparency
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    # Handle WebP/PNG images that might have alpha channel (4 channels)
    if len(image.shape) > 2 and image.shape[2] == 4:
        # Convert RGBA to BGR by removing alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
    return image

def convert_to_gray(image):
    """Convert an image to grayscale."""
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def save_image(image, output_path):
    """Save an image to the specified file path."""
    import cv2
    # Get file extension (including the period)
    file_extension = output_path[output_path.rfind('.'):].lower() if '.' in output_path else ''
    
    # Use appropriate parameters based on file extension
    if file_extension == '.webp':
        cv2.imwrite(output_path, image, [cv2.IMWRITE_WEBP_QUALITY, 95])
    else:
        cv2.imwrite(output_path, image)

def resize_image(image, width, height):
    """Resize an image to the specified width and height."""
    import cv2
    return cv2.resize(image, (width, height))