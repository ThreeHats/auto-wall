import cv2
import numpy as np
import json
import uuid
from collections import defaultdict, deque
from functools import lru_cache

from src.wall_detection.detector import process_contours_with_hierarchy

# Cache for brush patterns to avoid recreating them
_brush_pattern_cache = {}

# Brush pattern generator with caching
def get_brush_pattern(brush_size):
    """
    Get a circular brush pattern of the specified size.
    Uses caching to avoid recreating patterns for the same size.
    
    Parameters:
    - brush_size: Radius of the brush
    
    Returns:
    - Binary mask with the brush pattern (255 for brush area, 0 elsewhere)
    """
    # Check if we already have this pattern cached
    if brush_size in _brush_pattern_cache:
        return _brush_pattern_cache[brush_size]
        
    # Create a new pattern if not in cache
    pattern_size = brush_size * 2 + 1
    pattern = np.zeros((pattern_size, pattern_size), dtype=np.uint8)
    cv2.circle(pattern, (brush_size, brush_size), brush_size, 255, -1)
    
    # Store in cache (limit cache size)
    if len(_brush_pattern_cache) > 20:  # Limit cache size
        # Remove a random key if cache is too large
        _brush_pattern_cache.pop(next(iter(_brush_pattern_cache)))
    _brush_pattern_cache[brush_size] = pattern
    
    return pattern

# color or app?
def create_mask_from_contours(image_shape, contours, color=(0, 255, 0, 255)):
    """
    Create a transparent mask from contours.
    
    Parameters:
    - image_shape: Shape of the original image (height, width)
    - contours: List of contours to convert to mask
    - color: BGRA color tuple for the mask (default: green)
    
    Returns:
    - BGRA mask with contours filled
    """
    # Create a transparent mask (4-channel BGRA)
    height, width = image_shape[:2]
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Draw the contours on the mask
    temp_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(temp_mask, contours, -1, 255, thickness=2)
    
    # Set the color and alpha for the mask where contours exist
    mask[temp_mask == 255] = color
    
    return mask

# color or app?
def blend_image_with_mask(image, mask, region=None):
    """
    Blend an image with a transparent mask (optimized version).
    
    Parameters:
    - image: Original BGR image
    - mask: BGRA mask
    - region: Optional tuple (x, y, width, height) specifying region to blend
             If provided, only this region will be processed
    
    Returns:
    - BGRA image with mask blended if region is None
    - BGRA image of just the blended region if region is provided
    """
    if region is None:
        # Process the entire image (original behavior)
        # Check if dimensions match
        if image.shape[:2] != mask.shape[:2]:
            print(f"Warning: Image dimensions {image.shape[:2]} don't match mask dimensions {mask.shape[:2]}")
            # Create a properly sized mask instead of failing
            height, width = image.shape[:2]
            new_mask = np.zeros((height, width, 4), dtype=np.uint8)
            # Use the original mask data where possible (for the smaller dimension)
            h_limit = min(height, mask.shape[0])
            w_limit = min(width, mask.shape[1])
            new_mask[:h_limit, :w_limit] = mask[:h_limit, :w_limit]
            mask = new_mask
        
        # Convert the image to BGRA if it's BGR
        if image.shape[2] == 3:
            bgra_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            bgra_image = image.copy()
        
        # Only process pixels where mask has non-zero alpha
        alpha_mask = mask[:, :, 3] > 0
        
        # Direct copy of mask pixels to result where alpha > 0
        # This is much faster than per-pixel alpha blending
        if np.any(alpha_mask):
            bgra_image[alpha_mask] = mask[alpha_mask]
        
        return bgra_image
    else:
        # Process only a specific region
        x, y, w, h = region
        # Extract the region from both image and mask
        x_max = min(x + w, image.shape[1])
        y_max = min(y + h, image.shape[0])
        w = x_max - x
        h = y_max - y
        
        if w <= 0 or h <= 0:
            return None  # Invalid region
        
        image_region = image[y:y_max, x:x_max].copy()
        mask_region = mask[y:y_max, x:x_max]
        
        # Ensure image_region is BGRA
        if image_region.shape[2] == 3:
            image_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2BGRA)
        
        # Only process pixels where mask has non-zero alpha in this region
        alpha_mask = mask_region[:, :, 3] > 0
        
        # Direct copy of mask pixels to result where alpha > 0
        if np.any(alpha_mask):
            image_region[alpha_mask] = mask_region[alpha_mask]
        
        return image_region

# color
def draw_on_mask(mask, x, y, brush_size, color=(0, 255, 0, 255), erase=False):
    """
    Draw on a mask at the specified coordinates (high performance version).
    
    Parameters:
    - mask: BGRA mask to draw on
    - x, y: Coordinates to draw at
    - brush_size: Size of the brush/eraser
    - color: BGRA color to draw with (ignored if erase=True)
    - erase: Whether to erase (True) or draw (False)
    
    Returns:
    - Tuple containing (updated mask, affected region tuple (x_min, y_min, width, height))
    """
    # Calculate bounds for the affected region (with bounds checking)
    height, width = mask.shape[:2]
    x_min = max(0, x - brush_size)
    y_min = max(0, y - brush_size)
    x_max = min(width, x + brush_size + 1)
    y_max = min(height, y + brush_size + 1)
    
    # If the brush is completely outside the image, return early
    if x_min >= width or y_min >= height or x_max <= 0 or y_max <= 0:
        return mask, None
    
    # For small and medium brush sizes, use a cached brush pattern
    if brush_size <= 100:  # Expanded range for cached patterns
        # Get the brush pattern from cache (or create if not cached)
        brush_pattern = get_brush_pattern(brush_size)
        
        # Extract the subset of the pattern we need for this position
        pattern_y_min = max(0, y_min - (y - brush_size))
        pattern_y_max = brush_pattern.shape[0] - max(0, (y + brush_size + 1) - y_max)
        pattern_x_min = max(0, x_min - (x - brush_size))
        pattern_x_max = brush_pattern.shape[1] - max(0, (x + brush_size + 1) - x_max)
        
        # Get the brush pattern for the visible area
        circle_mask = brush_pattern[pattern_y_min:pattern_y_max, pattern_x_min:pattern_x_max]
        
        # Create mask indices once
        circle_indices = circle_mask == 255
        
        # Apply the brush to the mask efficiently
        if erase:
            # Use direct indexing for better performance
            mask_region = mask[y_min:y_max, x_min:x_max]
            mask_region[circle_indices, 3] = 0
        else:
            # For better performance with array operations
            mask_view = mask[y_min:y_max, x_min:x_max]
            
            # Pre-create the color array once for all channels
            color_array = np.array(color, dtype=np.uint8)
            
            # Use direct assignment for better performance
            for i in range(4):  # Process all 4 channels (BGRA)
                channel_view = mask_view[:, :, i]
                channel_view[circle_indices] = color_array[i]
    else:
        # Fallback to mesh grid for very large brushes
        y_indices, x_indices = np.mgrid[y_min:y_max, x_min:x_max]
        # Calculate squared distances (avoid sqrt for better performance)
        distances_squared = (x_indices - x)**2 + (y_indices - y)**2
        # Create mask where distance <= brush_size
        circle_mask = distances_squared <= brush_size**2
        
        # Apply the brush to the mask efficiently
        mask_view = mask[y_min:y_max, x_min:x_max]
        
        if erase:
            # Use direct indexing for alpha channel
            mask_view[circle_mask, 3] = 0
        else:
        # Create a color view for each channel
            for i in range(4):
                mask_view[circle_mask, i] = color[i]
    
    # Return the updated mask and the affected region bounds
    affected_region = (x_min, y_min, x_max - x_min, y_max - y_min)
    return mask, affected_region

# export
def contours_to_foundry_walls(contours, image_shape, simplify_tolerance=0.0, max_wall_length=50, max_walls=5000, merge_distance=1.0, angle_tolerance=0.5, max_gap=5.0, grid_size=0, allow_half_grid=True, grid_offset_x=0.0, grid_offset_y=0.0):
    """
    Convert OpenCV contours to Foundry VTT wall data format with intelligent segmentation.
    
    Parameters:
    - contours: List of contours to convert
    - image_shape: Original image shape (height, width) for proper scaling
    - simplify_tolerance: Tolerance for Douglas-Peucker simplification algorithm
                         Set to 0 to disable simplification (preserves curves)
                         Low values (0.01-0.2) create more detailed walls
    - max_wall_length: Maximum length for a single wall segment
    - max_walls: Maximum number of walls to generate
    - grid_size: Size of the grid in pixels (0 to disable grid snapping)
    - allow_half_grid: Whether to allow snapping to half-grid positions
    - grid_offset_x: Horizontal grid offset in pixels
    - grid_offset_y: Vertical grid offset in pixels
    
    Returns:
    - List of walls in Foundry VTT format
    """
    height, width = image_shape[:2]
    foundry_walls = []
    wall_count = 0
    current_tolerance = simplify_tolerance
    
    # Use extremely minimal tolerance if no simplification is wanted
    # This helps eliminate microscopic gaps without visibly affecting shape
    minimal_tolerance = simplify_tolerance
    
    # Process inner and outer contours equally
    # Don't sort by area as that prioritizes outer walls and might skip inner walls
    # if we hit the max_walls limit
    # Instead, handle inner and outer contours based on their order in the list
    
    for contour in contours:
        # Skip if we've reached the maximum number of walls
        if wall_count >= max_walls:
            break
        
        # Always apply a minimal simplification to ensure connectivity
        # This will remove duplicate points and microscopic variations but preserve all visible details
        contour_length = cv2.arcLength(contour, True)
        epsilon = minimal_tolerance * contour_length
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        
        # Skip very small or invalid contours
        if len(points) < 3:
            continue
        
        # Create walls along contour as a continuous path
        current_segments = []
        
        # Process each contour as a single continuous path
        for i in range(len(points)):
            start_point = points[i]
            end_point = points[(i+1) % len(points)]  # Use modulo to loop back to first point
            
            # Calculate the distance between points
            distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            
            # Skip very short segments (less than 3 pixels)
            if distance < 3:
                continue
                
            # If segment is longer than max_wall_length, break it into smaller pieces
            if distance > max_wall_length:
                # Determine number of segments needed
                num_segments = int(np.ceil(distance / max_wall_length))
                
                # Create each segment
                for j in range(num_segments):
                    # Check wall count limit
                    if wall_count >= max_walls:
                        break
                    
                    # Calculate interpolation factors
                    t_start = j / num_segments
                    t_end = (j + 1) / num_segments
                    
                    # Calculate segment points
                    start_x = start_point[0] + t_start * (end_point[0] - start_point[0])
                    start_y = start_point[1] + t_start * (end_point[1] - start_point[1])
                    end_x = start_point[0] + t_end * (end_point[0] - start_point[0])
                    end_y = start_point[1] + t_end * (end_point[1] - start_point[1])
                    
                    # Create wall segment
                    wall_id = generate_foundry_id()
                    wall = {
                        "light": 20,
                        "sight": 20,
                        "sound": 20,
                        "move": 20,
                        "c": [
                            float(start_x),
                            float(start_y),
                            float(end_x),
                            float(end_y)
                        ],
                        "_id": wall_id,
                        "dir": 0,
                        "door": 0,
                        "ds": 0,
                        "threshold": {
                            "light": None,
                            "sight": None,
                            "sound": None,
                            "attenuation": False
                        },
                        "flags": {}
                    }
                    
                    current_segments.append(wall)
                    wall_count += 1
            else:
                # Create a single wall segment for shorter distances
                wall_id = generate_foundry_id()
                wall = {
                    "light": 20,
                    "sight": 20,
                    "sound": 20,
                    "move": 20,
                    "c": [
                        float(start_point[0]),
                        float(start_point[1]),
                        float(end_point[0]),
                        float(end_point[1])
                    ],
                    "_id": wall_id,
                    "dir": 0,
                    "door": 0,
                    "ds": 0,
                    "threshold": {
                        "light": None,
                        "sight": None,
                        "sound": None,
                        "attenuation": False
                    },
                    "flags": {}
                }
                
                current_segments.append(wall)
                wall_count += 1
        
        # Add all segments from this contour
        foundry_walls.extend(current_segments)
        
        # If approaching the wall limit and simplification is enabled, increase tolerance
        if wall_count > 0.8 * max_walls and simplify_tolerance > 0:
            current_tolerance *= 1.5
    
    print(f"Generated {wall_count} wall segments for Foundry VTT")
    
    # Apply grid snapping if requested
    if grid_size > 0:
        foundry_walls = snap_walls_to_grid(foundry_walls, grid_size, allow_half_grid, grid_offset_x, grid_offset_y)
        print(f"Snapped walls to grid (size={grid_size}, half-grid={allow_half_grid}, offset=({grid_offset_x}, {grid_offset_y}))")
    
    # Perform connectivity check - merge segments with endpoints very close to each other
    connected_walls = ensure_wall_connectivity(foundry_walls, merge_distance=merge_distance, angle_tolerance=angle_tolerance, max_gap=max_gap)
    
    return connected_walls

# thinning
def thin_contour(input_data, target_width=5, max_iterations=3):
    """
    Thins a contour to approximately target_width pixels using controlled erosion.
    
    Parameters:
    - input_data: Either an OpenCV contour or a binary mask
    - target_width: Target width in pixels (default: 3)
    - max_iterations: Maximum number of erosion iterations to prevent infinite loops
    
    Returns:
    - Thinned contour
    """
    # Check if input is a binary mask (2D array with uint8 dtype)
    if isinstance(input_data, np.ndarray) and len(input_data.shape) == 2 and input_data.dtype == np.uint8:
        # Extract contours from the mask - use hierarchical retrieval
        contours, hierarchy = cv2.findContours(input_data, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found in mask")
            return None
            
        # Process contours to include inner ones
        processed_contours = process_contours_with_hierarchy(contours, hierarchy, 0, None)
        if not processed_contours:
            return None
            
        # Use the largest contour
        contour = max(processed_contours, key=cv2.contourArea)
    else:
        contour = input_data
    
    # Find bounding rectangle to create an appropriate sized mask
    x, y, w, h = cv2.boundingRect(contour)
    # Add padding to avoid boundary issues
    padding = 10
    mask = np.zeros((h + 2*padding, w + 2*padding), dtype=np.uint8)
    
    # Shift contour to fit within the padded mask
    shifted_contour = contour.copy()
    shifted_contour[:, :, 0] = contour[:, :, 0] - x + padding
    shifted_contour[:, :, 1] = contour[:, :, 1] - y + padding
    
    # Draw the contour on the mask
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
    
    # Create a copy for the iterative thinning
    img1 = mask.copy()
    
    # Structuring Element - use a smaller kernel for more controlled erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Keep track of width during erosion
    thin = img1.copy()
    iterations = 0
    
    while iterations < max_iterations:
        # Calculate current contours and estimate width
        temp_contours, _ = cv2.findContours(thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours left, break
        if not temp_contours:
            break
            
        # Find largest contour
        largest = max(temp_contours, key=cv2.contourArea)
        
        # Calculate width
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        est_width = 2 * area / perimeter if perimeter > 0 else 0
        
        # If width is at or below target, stop erosion
        if est_width <= target_width:
            break
        
        # Erode once more
        thin = cv2.erode(thin, kernel)
        iterations += 1
    
    print(f"Contour thinning stopped after {iterations} iterations")
    
    # Find contours of the thinned mask
    thinned_contours, _ = cv2.findContours(thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not thinned_contours:
        print("No thinned contours found, returning original contour")
        return contour
        
    # Find the largest thinned contour (in case multiple were created)
    largest_contour = max(thinned_contours, key=cv2.contourArea)
    
    # Shift the contour back to original coordinates
    largest_contour[:, :, 0] = largest_contour[:, :, 0] + x - padding
    largest_contour[:, :, 1] = largest_contour[:, :, 1] + y - padding
    
    print(f"Thinned contour found with {len(largest_contour)} points")
    return largest_contour

# export
def ensure_wall_connectivity(walls, merge_distance=1.0, angle_tolerance=0.5, max_gap=5.0):
    """
    Ensure walls are properly connected by merging endpoints that are very close to each other.
    Also removes duplicate walls after points have been merged.
    
    Parameters:
    - walls: List of wall segments
    - merge_distance: Maximum distance between endpoints to be considered for merging
    - angle_tolerance: Maximum angle difference to consider walls collinear (degrees)
    - max_gap: Maximum gap between walls to be considered for merging
    
    Returns:
    - List of connected walls with duplicates removed
    """
    if len(walls) <= 1:
        return walls
    
    # Extract all points from walls
    all_points = []
    for wall in walls:
        all_points.append((wall["c"][0], wall["c"][1]))  # Start point
        all_points.append((wall["c"][2], wall["c"][3]))  # End point
    
    # Find clusters of nearby points
    merged_points = merge_nearby_points(all_points, merge_distance)
    
    # Create new walls with merged points
    new_walls = []
    wall_hash_set = set()  # To track unique walls
    
    for wall in walls:
        # Get original points
        start_x, start_y = wall["c"][0], wall["c"][1]
        end_x, end_y = wall["c"][2], wall["c"][3]
        
        # Find their merged coordinates
        new_start = merged_points.get((start_x, start_y), (start_x, start_y))
        new_end = merged_points.get((end_x, end_y), (end_x, end_y))
        
        # Skip walls that became too short after merging
        new_dist = ((new_end[0] - new_start[0])**2 + (new_end[1] - new_start[1])**2)**0.5
        if new_dist < 2:  # Minimum meaningful wall length
            continue
        
        # Create a unique key for this wall (sort points to treat A->B same as B->A)
        wall_key = tuple(sorted([new_start, new_end]))
        
        # Only add if this wall doesn't exist yet
        if wall_key not in wall_hash_set:
            # Create a new wall with merged points
            new_wall = wall.copy()
            new_wall["c"] = [float(new_start[0]), float(new_start[1]), 
                             float(new_end[0]), float(new_end[1])]
            new_walls.append(new_wall)
            wall_hash_set.add(wall_key)
    
    print(f"Wall connectivity: {len(walls)} original walls reduced to {len(new_walls)} walls")
    
    # Further optimize by merging collinear walls (straight lines)
    optimized_walls = merge_collinear_walls(new_walls, angle_tolerance=angle_tolerance, max_gap=max_gap)
    
    print(f"Collinear merging: {len(new_walls)} connected walls reduced to {len(optimized_walls)} walls")
    
    return optimized_walls

# export
def merge_collinear_walls(walls, angle_tolerance=0.5, max_gap=5.0):
    """
    Merge walls that form straight lines (collinear walls) into single walls.
    
    Parameters:
    - walls: List of wall segments
    - angle_tolerance: Maximum angle difference in degrees to consider walls collinear
                       Set to 0 to only merge perfectly aligned walls
    - max_gap: Maximum gap between walls to be considered for merging
               Set to 0 to disable gap merging
    
    Returns:
    - List of walls with collinear segments merged
    """
    # If both parameters are 0, no merging will occur, so return early
    if angle_tolerance <= 0 and max_gap <= 0:
        return walls
        
    if len(walls) <= 1:
        return walls
    
    # Function to calculate a wall's angle (0-180 degrees)
    def calculate_angle(wall):
        x1, y1, x2, y2 = wall["c"]
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx)) % 180
        return round(angle, 2)  # Round to 2 decimal places for grouping
    
    # Function to check if two walls can be merged (close enough and collinear)
    def can_merge(wall1, wall2):
        # Get endpoints
        x1a, y1a, x2a, y2a = wall1["c"]
        x1b, y1b, x2b, y2b = wall2["c"]
        
        # Calculate distances between endpoints
        dist1 = np.sqrt((x2a - x1b)**2 + (y2a - y1b)**2)  # End of wall1 to start of wall2
        dist2 = np.sqrt((x2b - x1a)**2 + (y2b - y1a)**2)  # End of wall2 to start of wall1
        dist3 = np.sqrt((x1a - x1b)**2 + (y1a - y1b)**2)  # Start of wall1 to start of wall2
        dist4 = np.sqrt((x2a - x2b)**2 + (y2a - y2b)**2)  # End of wall1 to end of wall2
        
        # If any pair of endpoints is close enough, consider them connectible
        min_dist = min(dist1, dist2, dist3, dist4)
        if max_gap <= 0 or min_dist > max_gap:
            return False
        
        # Check if the walls are collinear (angle difference within tolerance)
        angle1 = calculate_angle(wall1)
        angle2 = calculate_angle(wall2)
        angle_diff = abs(angle1 - angle2)
        
        if angle_tolerance <= 0:
            # Require exact match if tolerance is 0
            return angle_diff == 0 or angle_diff == 180
        else:
            return angle_diff <= angle_tolerance or abs(angle_diff - 180) <= angle_tolerance
    
    # Function to merge two walls into one
    def merge_walls(wall1, wall2):
        # Get endpoints of both walls
        x1a, y1a, x2a, y2a = wall1["c"]
        x1b, y1b, x2b, y2b = wall2["c"]
        
        # Determine which endpoints to use based on which are farthest apart
        points = [(x1a, y1a), (x2a, y2a), (x1b, y1b), (x2b, y2b)]
        
        # Find the two points that are farthest apart
        max_dist = 0
        best_pair = (0, 0)
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = (points[j][0] - points[i][0])**2 + (points[j][1] - points[i][1])**2
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (i, j)
        
        # Create a new wall using the farthest endpoints
        merged_wall = wall1.copy()
        merged_wall["c"] = [
            float(points[best_pair[0]][0]),
            float(points[best_pair[0]][1]),
            float(points[best_pair[1]][0]),
            float(points[best_pair[1]][1])
        ]
        
        return merged_wall
    
    # Group walls by their angle
    angle_groups = defaultdict(list)
    for wall in walls:
        angle = calculate_angle(wall)
        angle_groups[angle].append(wall)
    
    result_walls = []
    
    # Process each group of walls with similar angles
    for angle, group_walls in angle_groups.items():
        if len(group_walls) <= 1:
            # If only one wall in group, add it directly
            result_walls.extend(group_walls)
            continue
        
        # For each group, create a list of walls that need processing
        remaining_walls = deque(group_walls)
        
        while remaining_walls:
            current_wall = remaining_walls.popleft()
            merged = False
            
            # Try to merge with each other wall in the group
            for i in range(len(remaining_walls)):
                other_wall = remaining_walls[0]  # Always check the first wall
                remaining_walls.popleft()  # Remove it for checking
                
                if can_merge(current_wall, other_wall):
                    # Merge walls and put back in queue for further merging
                    current_wall = merge_walls(current_wall, other_wall)
                    merged = True
                else:
                    # Can't merge, put back at the end of the queue
                    remaining_walls.append(other_wall)
            
            # If we didn't merge with any wall, this wall is done
            if not merged:
                result_walls.append(current_wall)
            else:
                # Try to merge with more walls
                remaining_walls.append(current_wall)
    
    return result_walls

# export
def merge_nearby_points(points, merge_distance):
    """
    Merge points that are within merge_distance distance of each other.
    
    Parameters:
    - points: List of (x, y) tuples
    - merge_distance: Distance threshold for merging
    
    Returns:
    - Dictionary mapping original points to their merged positions
    """
    # If merge_distance is 0, don't merge any points
    if merge_distance <= 0:
        return {}  # Empty dictionary means no points are merged
        
    # Convert to numpy array for faster calculations
    points = [tuple(map(float, p)) for p in points]
    points_array = np.array(points)
    
    # Dictionary to store the mapping from original points to merged points
    point_map = {}
    
    # Process points sequentially
    for i, point in enumerate(points):
        # Skip if this point is already processed
        if point in point_map:
            continue
        
        # Find all points within threshold distance
        if len(points_array) > 0:
            distances = np.sqrt(np.sum((points_array - np.array(point))**2, axis=1))
            nearby_indices = np.where(distances <= merge_distance)[0]
            
            if len(nearby_indices) > 0:
                # Calculate the average position for the cluster
                cluster_points = points_array[nearby_indices]
                avg_point = tuple(np.mean(cluster_points, axis=0))
                
                # Map all points in this cluster to the average position
                for idx in nearby_indices:
                    original_point = tuple(points_array[idx])
                    point_map[original_point] = avg_point
                
                # Remove processed points to speed up next iterations
                mask = np.ones(len(points_array), dtype=bool)
                mask[nearby_indices] = False
                points_array = points_array[mask]
    
    return point_map

# export
def generate_foundry_id():
    """Generate a unique ID for a Foundry VTT wall."""
    # Generate a random UUID in the correct format for Foundry
    return ''.join(uuid.uuid4().hex.upper()[0:16])

# export
def export_mask_to_foundry_json(mask_or_contours, image_shape, filename, 
                               simplify_tolerance=0.0, max_wall_length=50, 
                               max_walls=5000, merge_distance=1.0, angle_tolerance=0.5, max_gap=5.0,
                               grid_size=0, allow_half_grid=True):
    """
    Export a mask or contours to a Foundry VTT compatible JSON file.
    
    Parameters:
    - mask_or_contours: Either a binary mask or list of contours
    - image_shape: Original image shape (height, width)
    - filename: Path to save the JSON file
    - simplify_tolerance: Tolerance for contour simplification (default 0 = no simplification)
    - max_wall_length: Maximum length for a single wall segment
    - max_walls: Maximum number of walls to generate
    - merge_distance: Distance threshold for merging nearby wall endpoints (pixels)
    - grid_size: Size of the grid in pixels (0 to disable grid snapping)
    - allow_half_grid: Whether to allow snapping to half-grid positions
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Determine if input is a mask or contours
        if isinstance(mask_or_contours, np.ndarray):
            # Ensure the mask is properly sized for the target dimensions
            mask = mask_or_contours
            mask_h, mask_w = mask.shape[:2]
            target_h, target_w = image_shape[:2]
            
            # Resize mask if dimensions don't match
            if mask_h != target_h or mask_w != target_w:
                print(f"Resizing mask from {mask_w}x{mask_h} to {target_w}x{target_h}")
                mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
            # Extract contours from the mask - use hierarchical retrieval
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Process contours to include inner ones
            contours = process_contours_with_hierarchy(contours, hierarchy, 0, None)
        else:
            # Assume it's already a list of contours
            contours = mask_or_contours
            
            # Verify contours match image dimensions
            all_points = []
            for contour in contours:
                contour_points = contour.reshape(-1, 2)
                for point in contour_points:
                    x, y = point
                    # Keep track of max values to detect scaling issues
                    all_points.append((x, y))
            
            if all_points:
                max_x = max([p[0] for p in all_points])
                max_y = max([p[1] for p in all_points])
                target_h, target_w = image_shape[:2]
                
                # Print warning if contours seem to exceed image dimensions
                if max_x > target_w * 1.1 or max_y > target_h * 1.1:
                    print(f"WARNING: Contours exceed image dimensions. Max point: ({max_x}, {max_y}), Image: {target_w}x{target_h}")
                    print("Coordinates may need to be scaled.")
        
        # Convert to Foundry walls format
        foundry_walls = contours_to_foundry_walls(
            contours, 
            image_shape, 
            simplify_tolerance=simplify_tolerance,
            max_wall_length=max_wall_length,
            max_walls=max_walls,
            merge_distance=merge_distance,
            angle_tolerance=angle_tolerance,
            max_gap=max_gap,
            grid_size=grid_size,
            allow_half_grid=allow_half_grid
        )
        
        # We don't need to run ensure_wall_connectivity again, as it's already done in contours_to_foundry_walls
        # Just use the walls directly
        
        # Write the list of walls directly to the JSON file
        with open(filename, 'w') as f:
            json.dump(foundry_walls, f, indent=2)
        
        print(f"Exported {len(foundry_walls)} walls to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting to Foundry VTT format: {e}")
        import traceback
        traceback.print_exc()
        return False

# export
def snap_walls_to_grid(walls, grid_size, allow_half_grid=True, grid_offset_x=0.0, grid_offset_y=0.0):
    """
    Snap wall endpoints to a grid with optional offset.
    
    Parameters:
    - walls: List of Foundry VTT wall objects
    - grid_size: Grid size in pixels
    - allow_half_grid: If True, points can snap to half grid positions
    - grid_offset_x: Horizontal grid offset in pixels
    - grid_offset_y: Vertical grid offset in pixels
    
    Returns:
    - List of walls with endpoints snapped to grid
    """
    if grid_size <= 0:
        return walls  # No snapping if grid size is 0 or negative
    
    snapped_walls = []
    
    for wall in walls:
        # Extract the wall coordinates
        start_x, start_y, end_x, end_y = wall["c"]
        
        # Apply offset to coordinates before snapping
        offset_start_x = start_x - grid_offset_x
        offset_start_y = start_y - grid_offset_y
        offset_end_x = end_x - grid_offset_x
        offset_end_y = end_y - grid_offset_y
        
        # Calculate the nearest grid position
        if allow_half_grid:
            # Snap to nearest half-grid position
            snapped_start_x = round(offset_start_x / (grid_size / 2)) * (grid_size / 2) + grid_offset_x
            snapped_start_y = round(offset_start_y / (grid_size / 2)) * (grid_size / 2) + grid_offset_y
            snapped_end_x = round(offset_end_x / (grid_size / 2)) * (grid_size / 2) + grid_offset_x
            snapped_end_y = round(offset_end_y / (grid_size / 2)) * (grid_size / 2) + grid_offset_y
        else:
            # Snap to nearest full-grid position
            snapped_start_x = round(offset_start_x / grid_size) * grid_size + grid_offset_x
            snapped_start_y = round(offset_start_y / grid_size) * grid_size + grid_offset_y
            snapped_end_x = round(offset_end_x / grid_size) * grid_size + grid_offset_x
            snapped_end_y = round(offset_end_y / grid_size) * grid_size + grid_offset_y
        
        # Create a new wall with snapped coordinates
        snapped_wall = wall.copy()
        snapped_wall["c"] = [
            float(snapped_start_x),
            float(snapped_start_y),
            float(snapped_end_x),
            float(snapped_end_y)
        ]
        
        # Skip walls that became zero-length after snapping
        if snapped_start_x == snapped_end_x and snapped_start_y == snapped_end_y:
            continue
        
        snapped_walls.append(snapped_wall)
    
    return snapped_walls

# export
def contours_to_uvtt_walls(contours, image_shape, original_image=None, simplify_tolerance=0.0, max_wall_length=50, max_walls=5000, merge_distance=1.0, angle_tolerance=0.5, max_gap=5.0, grid_size=0, allow_half_grid=True, grid_offset_x=0.0, grid_offset_y=0.0, lights=None, overlay_grid_size=None):
    """
    Convert OpenCV contours to Universal VTT format with intelligent segmentation.
    
    Parameters:
    - contours: List of contours to convert
    - image_shape: Original image shape (height, width) for proper scaling
    - original_image: The original image data (numpy array) to include as base64
    - simplify_tolerance: Tolerance for Douglas-Peucker simplification algorithm
    - max_wall_length: Maximum length for a single wall segment
    - max_walls: Maximum number of walls to generate
    - grid_size: Size of the grid in pixels (0 to disable grid snapping)
    - allow_half_grid: Whether to allow snapping to half-grid positions
    - lights: List of light dictionaries to include in the UVTT export
    - overlay_grid_size: Grid overlay size to use as pixels_per_grid in UVTT file
    
    Returns:
    - Dictionary in Universal VTT format with walls in 'line_of_sight' array
    """
    height, width = image_shape[:2]
    
    # TODO: implement UVTT-specific wall generation logic instead of reusing Foundry logic
    # First convert to foundry format to reuse existing logic
    foundry_walls = contours_to_foundry_walls(
        contours, image_shape, simplify_tolerance, max_wall_length, 
        max_walls, merge_distance, angle_tolerance, max_gap, grid_size, allow_half_grid, 
        grid_offset_x, grid_offset_y
    )
    
    # Convert foundry walls to UVTT line_of_sight format
    line_of_sight = []
    line_of_sight_preview_pixels = []  # Store pixel coordinates for preview
    # Always use overlay_grid_size as pixels_per_grid if provided, otherwise fall back to grid_size or default
    pixels_per_grid_unit = overlay_grid_size if overlay_grid_size and overlay_grid_size > 0 else (grid_size if grid_size > 0 else 70)
    
    for wall in foundry_walls:
        # Foundry wall format: {"c": [start_x, start_y, end_x, end_y]}
        # UVTT format: [{"x": x, "y": y}, {"x": x, "y": y}, ...]
        # Convert pixel coordinates to grid coordinates by dividing by pixels_per_grid
        start_x, start_y, end_x, end_y = wall["c"]
        
        # Grid coordinates for UVTT file
        wall_points = [
            {"x": float(start_x / pixels_per_grid_unit), "y": float(start_y / pixels_per_grid_unit)},
            {"x": float(end_x / pixels_per_grid_unit), "y": float(end_y / pixels_per_grid_unit)}
        ]
        line_of_sight.append(wall_points)
        
        # Pixel coordinates for preview
        wall_pixels = [
            {"x": float(start_x), "y": float(start_y)},
            {"x": float(end_x), "y": float(end_y)}
        ]
        line_of_sight_preview_pixels.append(wall_pixels)
    
    # Get image as base64 for UVTT format
    image_base64 = ""
    if original_image is not None:
        try:
            import base64
            import cv2
            
            # Convert image to PNG format for base64 encoding
            # OpenCV uses BGR format, but for PNG encoding we should use the original format
            # Don't convert unless we know it's needed
            image_to_encode = original_image
            
            # Encode as PNG (OpenCV handles BGR format correctly for PNG)
            success, buffer = cv2.imencode('.png', image_to_encode)
            if success:
                # Convert to base64
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                print(f"Image encoded as base64 (size: {len(image_base64)} characters)")
            else:
                print("Failed to encode image as PNG")
        except Exception as e:
            print(f"Error encoding image as base64: {e}")
            pass
    
    # Create UVTT format structure
    # Use overlay_grid_size for pixels_per_grid and map_size calculations
    uvtt_data = {
        "format": 0.3,
        "resolution": {
            "map_origin": {
                "x": float(grid_offset_x),
                "y": float(grid_offset_y)
            },
            "map_size": {
                "x": float(width / pixels_per_grid_unit),
                "y": float(height / pixels_per_grid_unit)
            },
            "pixels_per_grid": int(pixels_per_grid_unit)
        },
        "line_of_sight": line_of_sight,
        "objects_line_of_sight": [],
        "portals": [],
        "environment": {
            "ambient_light": "77a8c8a2"
        },
        "lights": lights if lights else [],
        "_preview_pixels": line_of_sight_preview_pixels  # Internal use for preview display
    }
    
    # Add image as base64 if available
    if image_base64:
        uvtt_data["image"] = image_base64
    
    print(f"Generated {len(line_of_sight)} wall segments for Universal VTT")
    
    return uvtt_data
