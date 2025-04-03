import cv2
import numpy as np

def detect_walls(image, min_contour_area=100, max_contour_area=None, blur_kernel_size=5, canny_threshold1=50, canny_threshold2=150):
    """
    Detect walls in an image with adjustable parameters.
    
    Parameters:
    - image: Input image
    - min_contour_area: Minimum area of contours to keep (filters small artifacts)
    - max_contour_area: Maximum area of contours to keep (None means no upper limit)
    - blur_kernel_size: Kernel size for Gaussian blur (use 1 for no blur)
    - canny_threshold1: Lower threshold for Canny edge detection
    - canny_threshold2: Upper threshold for Canny edge detection
    
    Returns:
    - List of contours representing walls
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise if blur_kernel_size > 1
    if blur_kernel_size > 1:
        blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    else:
        blurred = gray  # No blur if kernel size is 1

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area and (max_contour_area is None or area <= max_contour_area):
            filtered_contours.append(contour)

    return filtered_contours

def draw_walls(image, contours, color=(0, 255, 0), thickness=2):
    """
    Draw detected wall contours on an image.
    
    Parameters:
    - image: Input image
    - contours: List of contours to draw
    - color: RGB color tuple for drawing
    - thickness: Line thickness
    
    Returns:
    - Image with contours drawn
    """
    image_with_walls = image.copy()
    cv2.drawContours(image_with_walls, contours, -1, color, thickness)
    return image_with_walls

def merge_contours(image, contours, dilation_iterations=2, min_merge_distance=3.0):
    """
    Merge nearby or overlapping contours by dilating and re-detecting contours.
    
    Parameters:
    - image: Input image (used for dimensions)
    - contours: List of contours to merge
    - dilation_iterations: Number of dilation iterations to perform
    - min_merge_distance: Minimum distance (in pixels) between contours to be merged
                          Controls the kernel size for dilation (can be float)
    
    Returns:
    - List of merged contours
    """
    # Create an empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw the contours onto the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Adjust kernel size based on min_merge_distance
    # Round to nearest odd integer to ensure valid kernel size
    kernel_size = max(3, int(round(min_merge_distance)))
    if kernel_size % 2 == 0:  # Ensure kernel size is odd
        kernel_size += 1
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate the mask to merge nearby contours
    dilated_mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    # Find contours again from the dilated mask
    merged_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return merged_contours