import cv2
import numpy as np
import json
import uuid
from collections import defaultdict, deque

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

def contours_to_foundry_walls(contours, image_shape, simplify_tolerance=0.0, max_wall_length=50, max_walls=5000, angle_tolerance=0.5, max_gap=5.0):
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
    
    print(f"Wall connectivity: {len(walls)} original walls reduced to {len(new_walls)} walls")
    
    # Further optimize by merging collinear walls (straight lines)
    optimized_walls = merge_collinear_walls(new_walls, angle_tolerance=0.5, max_gap=5.0)
    
    print(f"Collinear merging: {len(new_walls)} connected walls reduced to {len(optimized_walls)} walls")
    
    return optimized_walls

def merge_collinear_walls(walls, angle_tolerance=0.5, max_gap=5.0):
  """
  Merge walls that form straight lines (collinear walls) into single walls.
  
  Parameters:
  - walls: List of wall segments
  - angle_tolerance: Maximum angle difference in degrees to consider walls collinear
  - max_gap: Maximum gap between walls to be considered for merging
  
  Returns:
  - List of walls with collinear segments merged
  """
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
    if min_dist > max_gap:
      return False
    
    # Check if the walls are collinear (angle difference within tolerance)
    angle1 = calculate_angle(wall1)
    angle2 = calculate_angle(wall2)
    angle_diff = abs(angle1 - angle2)
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
                               max_walls=5000, merge_distance=1.0, angle_tolerance=0.5, max_gap=5.0):
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
            # Ensure the mask is properly sized for the target dimensions
            mask = mask_or_contours
            mask_h, mask_w = mask.shape[:2]
            target_h, target_w = image_shape[:2]
            
            # Resize mask if dimensions don't match
            if mask_h != target_h or mask_w != target_w:
                print(f"Resizing mask from {mask_w}x{mask_h} to {target_w}x{target_h}")
                mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
            # Extract contours from the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
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
            angle_tolerance=angle_tolerance,
            max_gap=max_gap
        )
        
        # Optimize walls by merging nearby points, removing duplicates,
        # and merging collinear walls (straight lines)
        foundry_walls = ensure_wall_connectivity(foundry_walls, proximity_threshold=merge_distance)
        
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
