import cv2
import numpy as np

def detect_walls(image, min_contour_area=100, max_contour_area=None, blur_kernel_size=5, 
                canny_threshold1=50, canny_threshold2=150, edge_margin=0,
                wall_colors=None, color_threshold=20):
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
    - wall_colors: Can be one of:
                  - None to disable color detection
                  - A single BGR color tuple
                  - A list of BGR color tuples
                  - A list of (BGR color tuple, threshold) pairs for per-color thresholds
    - color_threshold: Default threshold for colors without specific threshold (0-100)
    
    Returns:
    - List of contours representing walls
    """
    # Create a copy of the image for processing
    working_image = image.copy()
    
    # If wall colors are provided, use direct color-based contour detection
    if wall_colors is not None:
        if not isinstance(wall_colors, list):
            # Convert single color to list with default threshold
            wall_colors = [(wall_colors, color_threshold)]
        elif len(wall_colors) > 0 and not isinstance(wall_colors[0], tuple):
            # Convert list of colors to list of (color, default_threshold) tuples
            wall_colors = [(color, color_threshold) for color in wall_colors]
        elif len(wall_colors) > 0 and isinstance(wall_colors[0], tuple) and len(wall_colors[0]) == 3:
            # It's a list of color tuples, convert to (color, threshold) format
            wall_colors = [(color, color_threshold) for color in wall_colors]
        
        # Create a mask that combines all specified colors with their thresholds
        color_mask = create_multi_color_mask(image, wall_colors)
        
        # Find contours directly on the color mask
        # Changed from RETR_EXTERNAL to RETR_CCOMP to detect holes/interior walls
        color_contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours based on hierarchy to include both outer and inner contours
        result_contours = process_contours_with_hierarchy(color_contours, hierarchy, min_contour_area, max_contour_area)
        
        print(f"Color-based detection found {len(color_contours)} contours, {len(result_contours)} after filtering by area")
        
        # Return the filtered contours
        return result_contours
    
    # If no wall colors provided, continue with standard edge detection approach
    # Convert to grayscale
    gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise if blur_kernel_size > 1
    if blur_kernel_size > 1:
        blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    else:
        blurred = gray  # No blur if kernel size is 1

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    
    # Find contours - changed to retrieve hierarchical contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
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
                
                # Find all contours in this component (including holes)
                component_contours, component_hierarchy = cv2.findContours(
                    component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Process and add contours from this component
                component_result = process_contours_with_hierarchy(
                    component_contours, component_hierarchy, min_contour_area, max_contour_area
                )
                result_contours.extend(component_result)
    
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

def process_contours_with_hierarchy(contours, hierarchy, min_contour_area, max_contour_area):
    """
    Process contours based on hierarchy to include both exterior and interior contours.
    
    Parameters:
    - contours: List of detected contours
    - hierarchy: Contour hierarchy from cv2.findContours
    - min_contour_area: Minimum area for contour filtering
    - max_contour_area: Maximum area for contour filtering (None for no limit)
    
    Returns:
    - List of filtered contours
    """
    if len(contours) == 0:
        return []
        
    result_contours = []
    
    # Hierarchy format: [Next, Previous, First_Child, Parent]
    # If hierarchy is None (can happen with RETR_LIST), treat all contours as top level
    if hierarchy is None:
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area >= min_contour_area and (max_contour_area is None or area <= max_contour_area):
                result_contours.append(contour)
        return result_contours
                
    # Process contours by hierarchy
    hierarchy = hierarchy[0]  # OpenCV returns hierarchy as a single-item list
    
    # First pass: include all contours that meet area requirements
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= min_contour_area and (max_contour_area is None or area <= max_contour_area):
            result_contours.append(contour)
            
    return result_contours

def create_multi_color_mask(image, color_threshold_pairs):
    """
    Create a binary mask for multiple colors with individual thresholds.
    
    Parameters:
    - image: Input BGR image
    - color_threshold_pairs: List of ((B,G,R), threshold) tuples
    
    Returns:
    - Binary mask with matching pixels as white (255)
    """
    if not color_threshold_pairs:
        return np.zeros(image.shape[:2], dtype=np.uint8)
        
    # Start with an empty mask
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Create a mask for each color and threshold pair, combining them with OR operation
    for color, threshold in color_threshold_pairs:
        color_mask = create_color_mask(image, color, threshold)
        combined_mask = cv2.bitwise_or(combined_mask, color_mask)
    
    return combined_mask

def create_color_mask(image, target_color, threshold):
    """
    Create a binary mask where pixels similar to target_color are white (255),
    and all other pixels are black (0).
    
    Parameters:
    - image: Input BGR image
    - target_color: (B,G,R) tuple of the target color
    - threshold: How close a color needs to be to target_color (0-100)
                Higher values are more lenient, lower values more strict
                A value of 0 means exact color match only
    
    Returns:
    - Binary mask with matching pixels as white (255)
    """
    # Special case for threshold=0: exact color match only
    if threshold == 0:
        # Create an empty mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Extract channels
        b, g, r = cv2.split(image)
        
        # Set pixels that exactly match the target color
        exact_match = (b == target_color[0]) & (g == target_color[1]) & (r == target_color[2])
        mask[exact_match] = 255
        
        return mask
    
    # Regular case for threshold > 0
    # Convert threshold from percentage (0-100) to actual distance in color space
    # Maximum possible distance in RGB space is sqrt(255²+255²+255²) ≈ 441.7
    max_distance = 255.0 * np.sqrt(3)
    distance_threshold = (threshold / 100.0) * (max_distance / 2)
    
    # Extract channels
    b, g, r = cv2.split(image)
    
    # Calculate absolute difference from target color for each channel
    b_diff = np.abs(b.astype(np.float32) - target_color[0])
    g_diff = np.abs(g.astype(np.float32) - target_color[1])
    r_diff = np.abs(r.astype(np.float32) - target_color[2])
    
    # Calculate Euclidean distance in color space
    color_distance = np.sqrt(b_diff*b_diff + g_diff*g_diff + r_diff*r_diff)
    
    # Create binary mask where pixels closer than threshold are white (255)
    mask = np.zeros_like(b, dtype=np.uint8)
    mask[color_distance <= distance_threshold] = 255
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    
    # First close to connect nearby areas (fills small holes)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Then open to remove small noise (removes small isolated areas)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask

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
                          If 0, only overlapping contours will be merged
    
    Returns:
    - List of merged contours
    """
    # Create an empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw the contours onto the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Special case for min_merge_distance = 0: only merge overlapping contours
    if min_merge_distance == 0:
        # Find contours without any dilation
        merged_contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # Process inner and outer contours
        return process_contours_with_hierarchy(merged_contours, hierarchy, 0, None)

    # Handle the float value for min_merge_distance for non-zero values
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

    # Find contours again from the dilated mask (using hierarchical retrieval mode)
    merged_contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Process inner and outer contours
    return process_contours_with_hierarchy(merged_contours, hierarchy, 0, None)

def split_edge_contours(image, contours):
    """
    Split contours that touch the image edges by cutting and closing them.
    
    Parameters:
    - image: Input image (used for dimensions)
    - contours: List of contours to process
    
    Returns:
    - List of contours with edge-touching contours split
    """
    height, width = image.shape[:2]
    result_contours = []
    
    # Define edge boundaries
    edge_border = 1  # How many pixels from the actual edge to consider "edge"
    
    # Create a mask of the edge region
    edge_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(edge_mask, (0, 0), (width-1, height-1), 255, edge_border)
    
    # Track statistics
    original_count = len(contours)
    edge_touching_count = 0
    new_contours_count = 0
    unchanged_count = 0
    
    # Process each contour
    for contour in contours:
        # Create mask for this contour
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, 1)  # Draw just the contour lines
        
        # Check if contour touches edge
        edge_intersection = cv2.bitwise_and(contour_mask, edge_mask)
        if cv2.countNonZero(edge_intersection) > 0:
            edge_touching_count += 1
            
            # Find points where contour intersects the edge
            intersect_points = []
            contour_array = contour.reshape(-1, 2)
            
            # Check each point in the contour
            for i, point in enumerate(contour_array):
                x, y = point
                
                # Check if point is on edge
                if (x < edge_border or x >= width - edge_border or 
                    y < edge_border or y >= height - edge_border):
                    
                    # Find the adjacent points
                    prev_idx = (i - 1) % len(contour_array)
                    next_idx = (i + 1) % len(contour_array)
                    
                    # Only consider points where one adjacent point is inside
                    prev_inside = (edge_border <= contour_array[prev_idx][0] < width - edge_border and 
                                   edge_border <= contour_array[prev_idx][1] < height - edge_border)
                    
                    next_inside = (edge_border <= contour_array[next_idx][0] < width - edge_border and 
                                   edge_border <= contour_array[next_idx][1] < height - edge_border)
                    
                    if prev_inside or next_inside:
                        intersect_points.append((i, x, y))
            
            # If we found intersection points, create new contours
            if len(intersect_points) >= 2:  # Need at least 2 points to form a new contour
                # Find segments of the contour that don't touch the edge
                segments = []
                current_segment = []
                in_segment = False
                
                # Go through the entire contour
                total_points = len(contour_array)
                intersection_indices = [p[0] for p in intersect_points]
                
                for i in range(total_points + len(intersection_indices)):
                    idx = i % total_points
                    
                    # Check if this is an intersection point
                    is_intersection = idx in intersection_indices
                    
                    # Check if point is on interior (away from edge)
                    x, y = contour_array[idx]
                    is_interior = (edge_border <= x < width - edge_border and 
                                  edge_border <= y < height - edge_border)
                    
                    if is_interior and not in_segment:
                        # Start a new segment
                        in_segment = True
                        current_segment = [contour_array[idx].tolist()]
                    elif is_interior:
                        # Continue the segment
                        current_segment.append(contour_array[idx].tolist())
                    elif in_segment:
                        # End the segment
                        in_segment = False
                        if len(current_segment) >= 3:  # Need at least 3 points for a valid contour
                            segments.append(np.array(current_segment, dtype=np.int32))
                            current_segment = []
                
                # Add any valid segments as new contours
                new_segments_count = 0
                for segment in segments:
                    if len(segment) >= 3:  # Make sure it's a valid contour
                        # Convert segment to contour format
                        new_contour = segment.reshape(-1, 1, 2)
                        
                        # Extend the segment slightly to ensure it's closed away from the edges
                        # This simply duplicates the end points to create a loop
                        if len(segment) >= 2:
                            # Draw a line between the end points of the segment
                            segment_mask = np.zeros((height, width), dtype=np.uint8)
                            for i in range(len(segment) - 1):
                                cv2.line(segment_mask, tuple(segment[i]), tuple(segment[i+1]), 255, 2)
                            
                            # Find the contours in the mask - this will give us a closed contour
                            closed_contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for closed_contour in closed_contours:
                                if len(closed_contour) >= 4:  # Ensure it's valid
                                    result_contours.append(closed_contour)
                                    new_segments_count += 1
                
                # If we created new contours, we're done with this contour
                if new_segments_count > 0:
                    new_contours_count += new_segments_count
                    continue
                
            # If we couldn't create new contours, try the mask-based approach as fallback
            interior_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(interior_mask, [contour], 0, 255, cv2.FILLED)
            
            # Remove the edge regions
            cv2.rectangle(interior_mask, (0, 0), (width-1, height-1), 0, edge_border)
            
            # Find contours in the interior mask - use hierarchical retrieval
            interior_contours, hierarchy = cv2.findContours(interior_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process both outer and inner contours
            processed_interior = process_contours_with_hierarchy(interior_contours, hierarchy, 0, None)
            
            # Add any valid interior contours
            interior_added = 0
            for interior_contour in processed_interior:
                if len(interior_contour) >= 4:  # Minimum meaningful contour
                    result_contours.append(interior_contour)
                    interior_added += 1
                    
            if interior_added > 0:
                new_contours_count += interior_added
                continue
            
            # If all else fails, keep the original contour
            result_contours.append(contour)
            unchanged_count += 1
        else:
            # Contour doesn't touch edge, keep as-is
            result_contours.append(contour)
            unchanged_count += 1
    
    # Print detailed statistics
    print(f"Split edge contours: {original_count} original, {edge_touching_count} edge-touching, "
          f"{new_contours_count} new contours created, {unchanged_count} kept unchanged, {len(result_contours)} total")
    
    return result_contours

def scale_contours(contours, scale_factor):
    """
    Scale contours by the given factor.
    
    Parameters:
    - contours: List of contours to scale
    - scale_factor: Factor to scale by (>1 enlarges, <1 shrinks)
    
    Returns:
    - List of scaled contours
    """
    if scale_factor == 1.0:
        return contours.copy()
        
    scaled_contours = []
    for contour in contours:
        # Create a scaled copy
        scaled_contour = contour.copy().astype(np.float32)
        scaled_contour *= scale_factor
        scaled_contours.append(scaled_contour.astype(np.int32))
        
    return scaled_contours