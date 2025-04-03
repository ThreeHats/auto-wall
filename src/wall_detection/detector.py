import cv2
import numpy as np

def detect_walls(image, min_contour_area=100, max_contour_area=None, blur_kernel_size=5, 
                canny_threshold1=50, canny_threshold2=150, edge_margin=0):
    """
    Detect walls in an image with adjustable parameters.
    
    Parameters:
    - image: Input image
    - min_contour_area: Minimum area of contours to keep (filters small artifacts)
    - max_contour_area: Maximum area of contours to keep (None means no upper limit)
    - blur_kernel_size: Kernel size for Gaussian blur (use 1 for no blur)
    - canny_threshold1: Lower threshold for Canny edge detection
    - canny_threshold2: Upper threshold for Canny edge detection
    - edge_margin: Number of pixels from the edge to place the cutting border (0 means no cutting)
    
    Returns:
    - List of contours representing walls
    """
    # Convert to grayscale
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
    
    # Initial filtering by size (if not dealing with edge margin)
    if edge_margin <= 0:
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_contour_area and (max_contour_area is None or area <= max_contour_area):
                filtered_contours.append(contour)
        return filtered_contours
    
    # Edge margin is active, so we need to handle edge-touching contours specially
    height, width = image.shape[:2]
    
    # Create masks for edge detection
    center_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.rectangle(
        center_mask,
        (edge_margin, edge_margin),
        (width - edge_margin, height - edge_margin),
        255, -1
    )
    edge_mask = np.zeros_like(center_mask)
    cv2.rectangle(
        edge_mask,
        (0, 0),
        (width - 1, height - 1),
        255, 1
    )
    
    # Process contours
    filtered_contours = []
    
    for contour in contours:
        original_area = cv2.contourArea(contour)
        
        # Skip if clearly too small
        if original_area < min_contour_area / 2:  # Use a lower threshold for initial filtering
            continue
        
        # Skip if too large
        if max_contour_area is not None and original_area > max_contour_area:
            continue
        
        # Create a mask for this contour
        contour_mask = np.zeros_like(center_mask)
        cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)
        
        # Check if contour touches the edge
        edge_intersection = cv2.bitwise_and(contour_mask, edge_mask)
        touches_edge = cv2.countNonZero(edge_intersection) > 0
        
        if touches_edge:
            # Count the number of distinct intersections with the edge
            edge_points = np.argwhere(edge_intersection > 0)
            if len(edge_points) > 0:
                # Cluster edge points to find distinct intersections
                # We'll use a simple distance threshold approach
                distance_threshold = max(width, height) // 20  # Adjust as needed
                distinct_intersections = []
                
                for point in edge_points:
                    y, x = point
                    # Check if this point is close to any existing intersection
                    is_new_intersection = True
                    for i, existing in enumerate(distinct_intersections):
                        if np.sqrt((y - existing[0])**2 + (x - existing[1])**2) < distance_threshold:
                            is_new_intersection = False
                            break
                    
                    if is_new_intersection:
                        distinct_intersections.append((y, x))
                
                # If contour touches the edge in multiple places, use its full area for min_area check
                if len(distinct_intersections) >= 2:
                    if original_area >= min_contour_area:
                        # This contour passes the min_area check
                        # Now extract only the portion inside the center region
                        center_portion = cv2.bitwise_and(contour_mask, center_mask)
                        center_contours, _ = cv2.findContours(center_portion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Add all center portions to filtered contours
                        for center_contour in center_contours:
                            if cv2.contourArea(center_contour) > 0:
                                filtered_contours.append(center_contour)
                    
                    continue  # Skip the normal processing
        
        # For contours that don't touch the edge in multiple places or are entirely inside
        center_portion = cv2.bitwise_and(contour_mask, center_mask)
        center_contours, _ = cv2.findContours(center_portion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for center_contour in center_contours:
            area = cv2.contourArea(center_contour)
            if area >= min_contour_area and (max_contour_area is None or area <= max_contour_area):
                filtered_contours.append(center_contour)
    
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

    # Handle the float value for min_merge_distance
    # Calculate base kernel size and iterations to account for fractional distances
    base_kernel_size = max(3, int(min_merge_distance))
    if base_kernel_size % 2 == 0:
        base_kernel_size += 1  # Ensure kernel size is odd
    
    # Calculate iterations based on the fractional part
    fractional_part = min_merge_distance - int(min_merge_distance)
    effective_iterations = dilation_iterations
    
    if fractional_part > 0:
        # Scale iterations by the fractional part (higher fraction = more iterations)
        effective_iterations = int(dilation_iterations + round(fractional_part * 2))
    
    kernel = np.ones((base_kernel_size, base_kernel_size), np.uint8)
    
    # Dilate the mask to merge nearby contours
    dilated_mask = cv2.dilate(mask, kernel, iterations=effective_iterations)

    # Find contours again from the dilated mask
    merged_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return merged_contours