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
        # Create center mask and edge boundary
        height, width = gray.shape
        center_mask = np.zeros_like(gray, dtype=np.uint8)
        edge_boundary = np.zeros_like(gray, dtype=np.uint8)
        
        # Draw the center region and boundary
        cv2.rectangle(
            center_mask,
            (edge_margin, edge_margin),
            (width - edge_margin, height - edge_margin),
            255, -1  # Filled rectangle
        )
        
        cv2.rectangle(
            edge_boundary,
            (edge_margin, edge_margin),
            (width - edge_margin, height - edge_margin),
            255, 2   # Thick boundary only
        )
        
        # Create edge mask for detecting edge intersections
        edge_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(edge_mask, (0, 0), (width-1, height-1), 255, 1)
        
        # Process each connected component separately
        for i in range(1, num_labels):
            # Create a mask for this component
            component_mask = np.zeros_like(labels, dtype=np.uint8)
            component_mask[labels == i] = 255
            
            # Get area of this component
            component_area = stats[i, cv2.CC_STAT_AREA]
            
            # Check if this component touches the boundary
            boundary_intersection = cv2.bitwise_and(component_mask, edge_boundary)
            touches_boundary = cv2.countNonZero(boundary_intersection) > 0
            
            # Check if component touches image edge
            edge_intersection = cv2.bitwise_and(component_mask, edge_mask) 
            touches_edge = cv2.countNonZero(edge_intersection) > 0
            
            # Special handling for components that cross the boundary
            if touches_boundary:
                # 1. Get the portion of the component inside the center region
                center_portion = cv2.bitwise_and(component_mask, center_mask)
                
                # 2. Check if the component leaves and re-enters the center region
                # Find contours in center portion
                center_contours, _ = cv2.findContours(center_portion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # If multiple separate contours in center, they might be part of a larger wall that crosses the boundary
                if len(center_contours) > 1 and component_area >= min_contour_area:
                    # Create a combined mask to close the contours along the boundary
                    combined_mask = center_portion.copy()
                    
                    # Draw lines along the intersection with the boundary to close contours
                    boundary_points = np.argwhere(boundary_intersection > 0)
                    
                    if len(boundary_points) >= 2:
                        # Find clusters of boundary points
                        distance_threshold = max(width, height) // 20
                        point_clusters = []
                        
                        for point in boundary_points:
                            y, x = point
                            added = False
                            
                            # Try to add point to an existing cluster
                            for cluster in point_clusters:
                                for cluster_point in cluster:
                                    if np.sqrt((y - cluster_point[0])**2 + (x - cluster_point[1])**2) < distance_threshold:
                                        cluster.append((y, x))
                                        added = True
                                        break
                                if added:
                                    break
                            
                            # If not added to any cluster, create a new one
                            if not added:
                                point_clusters.append([(y, x)])
                        
                        # Connect points across clusters to close the contour
                        if len(point_clusters) >= 2:
                            # Get centroids of clusters
                            centroids = []
                            for cluster in point_clusters:
                                cy = sum(p[0] for p in cluster) // len(cluster)
                                cx = sum(p[1] for p in cluster) // len(cluster)
                                centroids.append((cy, cx))
                            
                            # Draw lines between centroids to close contours
                            for i in range(len(centroids)):
                                pt1 = (centroids[i][1], centroids[i][0])  # Convert y,x to x,y for drawing
                                pt2 = (centroids[(i+1) % len(centroids)][1], centroids[(i+1) % len(centroids)][0])
                                cv2.line(combined_mask, pt1, pt2, 255, 2)
                    
                    # Extract closed contours
                    closed_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Add closed contours to result if they meet the area criteria
                    for contour in closed_contours:
                        area = cv2.contourArea(contour)
                        if area >= min_contour_area and (max_contour_area is None or area <= max_contour_area):
                            result_contours.append(contour)
                    
                    continue  # Skip standard processing below
                
                # If this component has significant area and touches edges in multiple places
                elif touches_edge:
                    edge_points = np.argwhere(edge_intersection > 0)
                    
                    if len(edge_points) > 10:  # Use a threshold to avoid spurious detections
                        # Use component's full area for min_area check
                        if component_area >= min_contour_area:
                            # Close the contour by adding the boundary edges where it intersects
                            closed_mask = center_portion.copy()
                            boundary_points = np.argwhere(boundary_intersection > 0)
                            
                            for point in boundary_points:
                                y, x = point
                                cv2.circle(closed_mask, (x, y), 2, 255, -1)
                            
                            # Dilate to connect nearby points
                            closed_mask = cv2.dilate(closed_mask, kernel, iterations=1)
                            
                            # Extract contours from closed mask
                            closed_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for contour in closed_contours:
                                area = cv2.contourArea(contour)
                                if area >= min_contour_area and (max_contour_area is None or area <= max_contour_area):
                                    result_contours.append(contour)
                            
                            continue  # Skip standard processing
            
            # Standard processing for components fully inside the center region
            # or those that don't need special handling
            center_portion = cv2.bitwise_and(component_mask, center_mask)
            center_area = cv2.countNonZero(center_portion)
            
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