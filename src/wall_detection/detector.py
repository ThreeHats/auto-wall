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
    
    # Process contours - Generate filled mask
    contour_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # Find touching contours - dilate slightly and run connectedComponents
    kernel = np.ones((3, 3), np.uint8)
    working_mask = cv2.dilate(contour_mask, kernel, iterations=1)
    
    # Find connected components (treats touching contours as one)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(working_mask)
    
    # Process each connected component
    result_contours = []
    
    # No edge margin - simpler processing
    if edge_margin <= 0:
        for i in range(1, num_labels):  # Skip background (label 0)
            # Get the area of this component
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Apply area filtering
            if area >= min_contour_area and (max_contour_area is None or area <= max_contour_area):
                # Extract the contours for this component
                component_mask = np.zeros_like(labels, dtype=np.uint8)
                component_mask[labels == i] = 255
                
                # Find contours of this component
                component_contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Add all contours from this component to the result
                result_contours.extend(component_contours)
    
    # Handle edge margin case
    else:
        # Create center mask
        height, width = gray.shape
        center_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(
            center_mask,
            (edge_margin, edge_margin),
            (width - edge_margin, height - edge_margin),
            255, -1
        )
        
        # Create edge mask for detecting edge intersections
        edge_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(edge_mask, (0, 0), (width-1, height-1), 255, 1)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            # Create a mask for this component
            component_mask = np.zeros_like(labels, dtype=np.uint8)
            component_mask[labels == i] = 255
            
            # Get area of this component
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Check if this component touches the edge
            edge_intersection = cv2.bitwise_and(component_mask, edge_mask)
            touches_edge = cv2.countNonZero(edge_intersection) > 0
            
            if touches_edge:
                # Find edge intersection points
                edge_points = np.argwhere(edge_intersection > 0)
                
                if len(edge_points) > 0:
                    # Cluster edge points to find distinct intersections
                    distance_threshold = max(width, height) // 20
                    distinct_intersections = []
                    
                    for point in edge_points:
                        y, x = point
                        is_new_intersection = True
                        
                        for existing in distinct_intersections:
                            if np.sqrt((y - existing[0])**2 + (x - existing[1])**2) < distance_threshold:
                                is_new_intersection = False
                                break
                        
                        if is_new_intersection:
                            distinct_intersections.append((y, x))
                    
                    # If component touches edge in multiple places, use full area for filtering
                    if len(distinct_intersections) >= 2 and area >= min_contour_area:
                        # Get the portion inside the center region
                        center_portion = cv2.bitwise_and(component_mask, center_mask)
                        if cv2.countNonZero(center_portion) > 0:
                            component_contours, _ = cv2.findContours(center_portion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            result_contours.extend(component_contours)
                        continue
            
            # Process the portion in the center region
            center_portion = cv2.bitwise_and(component_mask, center_mask)
            center_area = cv2.countNonZero(center_portion)
            
            # Apply area filtering
            if center_area >= min_contour_area and (max_contour_area is None or center_area <= max_contour_area):
                center_contours, _ = cv2.findContours(center_portion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                result_contours.extend(center_contours)
    
    return result_contours

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