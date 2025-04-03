def load_image(image_path):
    """Load an image from the specified file path."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def convert_to_gray(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def save_image(image, output_path):
    """Save an image to the specified file path."""
    cv2.imwrite(output_path, image)

def resize_image(image, width, height):
    """Resize an image to the specified width and height."""
    return cv2.resize(image, (width, height))