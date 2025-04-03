import cv2
import numpy as np
import json
import uuid
from collections import defaultdict

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

def blend_image_with_mask(image, mask):
    """
    Blend an image with a transparent mask (optimized version).
    
    Parameters:
    - image: Original BGR image
    - mask: BGRA mask
    
    Returns:
    - BGRA image with mask blended
    """
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

def draw_on_mask(mask, x, y, brush_size, color=(0, 255, 0, 255), erase=False):
    """
    Draw on a mask at the specified coordinates (optimized version).
    
    Parameters:
    - mask: BGRA mask to draw on
    - x, y: Coordinates to draw at
    - brush_size: Size of the brush/eraser
    - color: BGRA color to draw with (ignored if erase=True)
    - erase: Whether to erase (True) or draw (False)
    
    Returns:
    - Updated mask
    """
    # Calculate bounds for the affected region (with bounds checking)
    height, width = mask.shape[:2]
    x_min = max(0, x - brush_size)
    y_min = max(0, y - brush_size)
    x_max = min(width, x + brush_size + 1)
    y_max = min(height, y + brush_size + 1)
    
    # If the brush is completely outside the image, return early
    if x_min >= width or y_min >= height or x_max <= 0 or y_max <= 0:
        return mask
    
    # Calculate distance from center for each pixel in the affected region
    y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
    distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
    
    # Create a mask of pixels within brush radius
    circle_mask = distances <= brush_size
    
    # Apply the brush to the mask efficiently
    if erase:
        # Set alpha channel to 0 where brush was applied
        mask[y_min:y_max, x_min:x_max, 3][circle_mask] = 0
    else:
        # Set color and alpha where brush was applied (vectorized operation)
        for c in range(4):  # For all BGRA channels
            mask[y_min:y_max, x_min:x_max, c][circle_mask] = color[c]
    
    return mask

def contours_to_foundry_walls(contours, image_shape, simplify_tolerance=0.0, max_wall_length=50, max_walls=5000):
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
    
    Returns:
    - List of walls in Foundry VTT format
    """
    height, width = image_shape[:2]
    foundry_walls = []
    wall_count = 0
    current_tolerance = simplify_tolerance
    
    # Use extremely minimal tolerance if no simplification is wanted
    # This helps eliminate microscopic gaps without visibly affecting shape
    if simplify_tolerance <= 0:
        minimal_tolerance = 0.0005  # Extremely small tolerance
    else:
        minimal_tolerance = simplify_tolerance
    
    # Sort contours by area (largest first) to prioritize main walls
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
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
    
    # Perform connectivity check - merge segments with endpoints very close to each other
    connected_walls = ensure_wall_connectivity(foundry_walls)
    
    return connected_walls

def ensure_wall_connectivity(walls, proximity_threshold=1.0):
    """
    Ensure walls are properly connected by merging endpoints that are very close to each other.
    Also removes duplicate walls after points have been merged.
    
    Parameters:
    - walls: List of wall segments
    - proximity_threshold: Maximum distance between endpoints to be considered for merging
    
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
    merged_points = merge_nearby_points(all_points, proximity_threshold)
    
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
    
    print(f"Wall optimization: {len(walls)} original walls reduced to {len(new_walls)} walls")
    return new_walls

def merge_nearby_points(points, proximity_threshold):
    """
    Merge points that are within proximity_threshold distance of each other.
    
    Parameters:
    - points: List of (x, y) tuples
    - proximity_threshold: Distance threshold for merging
    
    Returns:
    - Dictionary mapping original points to their merged positions
    """
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
            nearby_indices = np.where(distances <= proximity_threshold)[0]
            
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

def generate_foundry_id():
    """Generate a unique ID for a Foundry VTT wall."""
    # Generate a random UUID in the correct format for Foundry
    return ''.join(uuid.uuid4().hex.upper()[0:16])

def export_mask_to_foundry_json(mask_or_contours, image_shape, filename, 
                               simplify_tolerance=0.0, max_wall_length=50, 
                               max_walls=5000, merge_distance=1.0):
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
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Determine if input is a mask or contours
        if isinstance(mask_or_contours, np.ndarray):
            # Extract contours from the mask
            contours, _ = cv2.findContours(
                mask_or_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            # Assume it's already a list of contours
            contours = mask_or_contours
        
        # Convert to Foundry walls format
        foundry_walls = contours_to_foundry_walls(
            contours, 
            image_shape, 
            simplify_tolerance=simplify_tolerance,
            max_wall_length=max_wall_length,
            max_walls=max_walls
        )
        
        # Optimize walls by merging nearby points and removing duplicates
        foundry_walls = ensure_wall_connectivity(foundry_walls, proximity_threshold=merge_distance)
        
        # Write the list of walls directly to the JSON file
        with open(filename, 'w') as f:
            json.dump(foundry_walls, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error exporting to Foundry VTT format: {e}")
        return False
