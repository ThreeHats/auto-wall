import cv2
import numpy as np
import json

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

def contours_to_foundry_walls(contours, image_shape, simplify_tolerance=0.1, max_wall_length=50, max_walls=5000):
    """
    Convert OpenCV contours to Foundry VTT wall data format with intelligent segmentation.
    
    Parameters:
    - contours: List of contours to convert
    - image_shape: Original image shape (height, width) for proper scaling
    - simplify_tolerance: Tolerance for Douglas-Peucker simplification algorithm
                         Lower values create more detailed walls (0.01-1.0 recommended)
    - max_wall_length: Maximum length for a single wall segment
    - max_walls: Maximum number of walls to generate
    
    Returns:
    - Dictionary with walls data in Foundry VTT format
    """
    height, width = image_shape[:2]
    foundry_data = {"walls": []}
    wall_count = 0
    current_tolerance = simplify_tolerance
    
    # Sort contours by area (largest first) to prioritize main walls
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        # Skip if we've reached the maximum number of walls
        if wall_count >= max_walls:
            break
        
        # Apply simplification based on contour length
        contour_length = cv2.arcLength(contour, True)
        epsilon = current_tolerance * contour_length
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert the simplified contour to points
        points = approx.reshape(-1, 2)
        
        # Skip very small or invalid contours
        if len(points) < 3:
            continue
        
        # Process each pair of adjacent points
        segments_from_contour = []
        
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            # Calculate distance between points
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Skip very short segments
            if distance < 3:
                continue
            
            # If distance is greater than max_wall_length, break into multiple segments
            if distance > max_wall_length:
                num_segments = int(np.ceil(distance / max_wall_length))
                
                # Create intermediate points to break long segments
                for j in range(num_segments):
                    if wall_count >= max_walls:
                        break
                        
                    # Calculate segment start and end points
                    t_start = j / num_segments
                    t_end = (j + 1) / num_segments
                    
                    # Linear interpolation between p1 and p2
                    start_x = p1[0] + t_start * (p2[0] - p1[0])
                    start_y = p1[1] + t_start * (p2[1] - p1[1])
                    end_x = p1[0] + t_end * (p2[0] - p1[0])
                    end_y = p1[1] + t_end * (p2[1] - p1[1])
                    
                    # Create wall segment
                    wall = {
                        "c": [float(start_x), float(start_y), float(end_x), float(end_y)],
                        "move": 1,  # Movement restriction
                        "sense": 1,  # Light restriction
                        "dir": 0,    # Bidirectional wall
                        "door": 0,   # Not a door
                        "ds": 0,     # Door state (closed)
                        "flags": {}  # No special flags
                    }
                    
                    segments_from_contour.append(wall)
                    wall_count += 1
            else:
                # Create a single wall segment
                wall = {
                    "c": [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])],
                    "move": 1,
                    "sense": 1,
                    "dir": 0,
                    "door": 0,
                    "ds": 0,
                    "flags": {}
                }
                
                segments_from_contour.append(wall)
                wall_count += 1
        
        # Add all segments from this contour
        foundry_data["walls"].extend(segments_from_contour)
        
        # If approaching the wall limit, increase simplification to reduce wall count
        if wall_count > 0.8 * max_walls and current_tolerance < 1.0:
            current_tolerance *= 1.5
    
    print(f"Generated {wall_count} wall segments for Foundry VTT")
    return foundry_data

def export_mask_to_foundry_json(mask_or_contours, image_shape, filename, 
                               simplify_tolerance=0.1, max_wall_length=50, 
                               max_walls=5000):
    """
    Export a mask or contours to a Foundry VTT compatible JSON file.
    
    Parameters:
    - mask_or_contours: Either a binary mask or list of contours
    - image_shape: Original image shape (height, width)
    - filename: Path to save the JSON file
    - simplify_tolerance: Tolerance for contour simplification (lower = more detail)
    - max_wall_length: Maximum length for a single wall segment
    - max_walls: Maximum number of walls to generate
    
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
        foundry_data = contours_to_foundry_walls(
            contours, 
            image_shape, 
            simplify_tolerance=simplify_tolerance,
            max_wall_length=max_wall_length,
            max_walls=max_walls
        )
        
        # Write to JSON file
        with open(filename, 'w') as f:
            json.dump(foundry_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error exporting to Foundry VTT format: {e}")
        return False
