"""
Light detection module for Auto-Wall.
Detects bright spots in images and creates light points for UVTT export.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def detect_lights(image: np.ndarray, threshold: float = 0.8, min_area: int = 5, max_area: int = 500, 
                 light_colors: List[Tuple[Tuple[int, int, int], float]] = None, 
                 merge_distance: float = 20.0) -> List[Dict]:
    """
    Detect bright spots or specific colors in an image and create light points.
    
    Args:
        image: Input image (BGR or RGB)
        threshold: Brightness threshold (0.0 to 1.0, where 1.0 is pure white) - used when light_colors is None
        min_area: Minimum area in pixels for a light spot
        max_area: Maximum area in pixels for a light spot
        light_colors: Optional list of (BGR_color_tuple, threshold) pairs for color-based detection
        merge_distance: Distance in pixels to merge nearby lights (0 to disable merging)
        
    Returns:
        List of light dictionaries with position, color, range, intensity, and shadows
    """
    if image is None or len(image.shape) == 0:
        return []
    
    # If specific light colors are provided, use color-based detection
    if light_colors and len(light_colors) > 0:
        lights = detect_lights_by_color(image, light_colors, min_area, max_area)
    else:
        # Otherwise, use brightness-based detection
        lights = detect_lights_by_brightness(image, threshold, min_area, max_area)
    
    # Merge nearby lights if enabled
    if merge_distance > 0 and len(lights) > 1:
        lights = merge_lights(lights, merge_distance)
    
    return lights


def detect_lights_by_brightness(image: np.ndarray, threshold: float, min_area: int, max_area: int) -> List[Dict]:
    """Detect lights based on brightness threshold."""
    
    # Convert to grayscale for brightness detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize to 0-1 range
    gray_normalized = gray.astype(np.float32) / 255.0
    
    # Create binary mask for bright areas
    bright_mask = (gray_normalized >= threshold).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of bright areas
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return process_light_contours(image, contours, min_area, max_area, gray_normalized)


def detect_lights_by_color(image: np.ndarray, light_colors: List[Tuple[Tuple[int, int, int], float]], 
                          min_area: int, max_area: int) -> List[Dict]:
    """Detect lights based on specific colors and thresholds."""
    if len(image.shape) != 3:
        return []  # Color detection requires color image
    
    # Create combined mask for all light colors
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for (target_color_bgr, color_threshold) in light_colors:
        # Convert BGR to numpy array for processing
        target_color = np.array(target_color_bgr, dtype=np.uint8)
        
        # Calculate color distance using Euclidean distance in BGR space
        color_diff = np.sqrt(np.sum((image.astype(np.float32) - target_color.astype(np.float32))**2, axis=2))
        
        # Create mask for pixels within threshold
        color_mask = (color_diff <= color_threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Add to combined mask
        combined_mask = cv2.bitwise_or(combined_mask, color_mask)
    
    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # For color-based detection, create a normalized brightness map based on the original image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_normalized = gray.astype(np.float32) / 255.0
    
    return process_light_contours(image, contours, min_area, max_area, gray_normalized)


def process_light_contours(image: np.ndarray, contours, min_area: int, max_area: int, 
                          gray_normalized: np.ndarray) -> List[Dict]:
    """Process contours to create light dictionaries."""
    lights = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
            
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
            continue
        
        # Extract color from the original image at the center point
        if len(image.shape) == 3:
            # Get BGR values and convert to hex
            b, g, r = image[center_y, center_x]
            color_hex = f"ff{r:02x}{g:02x}{b:02x}"
        else:
            # Grayscale - convert to white with the grayscale value
            gray_val = image[center_y, center_x]
            color_hex = f"ff{gray_val:02x}{gray_val:02x}{gray_val:02x}"
        
        # Calculate light properties based on area and brightness
        # Larger areas get bigger range, brighter areas get higher intensity
        brightness = gray_normalized[center_y, center_x]
        base_range = np.sqrt(area / np.pi) / 50.0  # Convert pixel radius to grid units (rough estimate)
        light_range = max(1.0, min(5.0, base_range))  # Clamp between 1 and 5 grid units
        
        intensity = min(1.0, brightness * 1.2)  # Scale brightness to intensity
        
        light = {
            "position": {
                "x": float(center_x),
                "y": float(center_y)
            },
            "range": float(light_range),
            "intensity": float(intensity),
            "color": color_hex,
            "shadows": True,
            "_area": area,  # Store for debugging/adjustment
            "_brightness": float(brightness)  # Store for debugging/adjustment
        }
        
        lights.append(light)
    
    return lights


def scale_lights_to_grid(lights: List[Dict], image_shape: Tuple[int, int], grid_size: float = 70.0, scale_factor: float = 1.0) -> List[Dict]:
    """
    Scale light positions from pixel coordinates to grid coordinates.
    
    Args:
        lights: List of light dictionaries
        image_shape: (height, width) of the image
        grid_size: Size of one grid square in pixels
        scale_factor: Scale factor from working image to original image (1/app.scale_factor)
        
    Returns:
        List of lights with scaled positions
    """
    if not lights:
        return []
    
    scaled_lights = []
    height, width = image_shape[:2]
    
    for light in lights:
        scaled_light = light.copy()
        
        # Convert pixel coordinates to grid coordinates
        pixel_x = light["position"]["x"]
        pixel_y = light["position"]["y"]
        
        # Store original working image pixel coordinates
        scaled_light["_working_pixel_x"] = float(pixel_x)
        scaled_light["_working_pixel_y"] = float(pixel_y)
        
        # Scale up to original image coordinates if needed
        if scale_factor != 1.0:
            original_pixel_x = pixel_x / scale_factor
            original_pixel_y = pixel_y / scale_factor
        else:
            original_pixel_x = pixel_x
            original_pixel_y = pixel_y
        
        # Store original image pixel coordinates for accurate drawing
        scaled_light["_original_pixel_x"] = float(original_pixel_x)
        scaled_light["_original_pixel_y"] = float(original_pixel_y)
        
        # Convert original image pixels to grid coordinates
        grid_x = original_pixel_x / grid_size
        grid_y = original_pixel_y / grid_size
        
        scaled_light["position"] = {
            "x": float(grid_x),
            "y": float(grid_y)
        }
        
        scaled_lights.append(scaled_light)
    
    return scaled_lights


def merge_lights(lights: List[Dict], merge_distance: float = 20.0) -> List[Dict]:
    """
    Merge lights that are close to each other to reduce duplicate detections.
    
    Args:
        lights: List of light dictionaries
        merge_distance: Maximum distance in pixels to merge lights
        
    Returns:
        List of merged light dictionaries
    """
    if not lights or merge_distance <= 0:
        return lights
    
    merged_lights = []
    used_indices = set()
    
    for i, light1 in enumerate(lights):
        if i in used_indices:
            continue
            
        # Start a new group with this light
        group = [light1]
        used_indices.add(i)
        
        # Find nearby lights to merge
        for j, light2 in enumerate(lights):
            if j <= i or j in used_indices:
                continue
                
            # Calculate distance between lights
            x1, y1 = light1["position"]["x"], light1["position"]["y"]
            x2, y2 = light2["position"]["x"], light2["position"]["y"]
            
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            if distance <= merge_distance:
                group.append(light2)
                used_indices.add(j)
        
        # Merge the group into a single light
        if len(group) == 1:
            merged_lights.append(group[0])
        else:
            merged_light = merge_light_group(group)
            merged_lights.append(merged_light)
    
    return merged_lights


def merge_light_group(lights: List[Dict]) -> Dict:
    """
    Merge a group of lights into a single light by averaging properties.
    
    Args:
        lights: List of light dictionaries to merge
        
    Returns:
        Single merged light dictionary
    """
    if len(lights) == 1:
        return lights[0]
    
    # Calculate weighted average position based on intensity
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    total_intensity = 0
    max_range = 0
    total_area = 0
    
    # Extract color from the brightest light
    brightest_light = max(lights, key=lambda l: l.get("intensity", 0))
    color = brightest_light.get("color", "ffffffff")
    
    for light in lights:
        intensity = light.get("intensity", 1.0)
        area = light.get("_area", 10)
        
        # Weight by intensity and area
        weight = intensity * area
        total_weight += weight
        
        weighted_x += light["position"]["x"] * weight
        weighted_y += light["position"]["y"] * weight
        
        total_intensity += intensity
        max_range = max(max_range, light.get("range", 2.0))
        total_area += area
    
    # Calculate averaged properties
    if total_weight > 0:
        avg_x = weighted_x / total_weight
        avg_y = weighted_y / total_weight
    else:
        # Fallback to simple average
        avg_x = sum(l["position"]["x"] for l in lights) / len(lights)
        avg_y = sum(l["position"]["y"] for l in lights) / len(lights)
    
    avg_intensity = min(1.0, total_intensity / len(lights))  # Average intensity, capped at 1.0
    
    # Create merged light
    merged_light = {
        "position": {
            "x": float(avg_x),
            "y": float(avg_y)
        },
        "range": float(max_range),  # Use the largest range
        "intensity": float(avg_intensity),
        "color": color,
        "shadows": True,
        "_area": total_area,  # Sum of areas
        "_brightness": float(avg_intensity),  # Use intensity as brightness
        "_merged_count": len(lights)  # Track how many lights were merged
    }
    
    return merged_light


def draw_lights_on_image(image: np.ndarray, lights: List[Dict], grid_size: float = 70.0, 
                        show_range: bool = True, alpha: float = 0.3) -> np.ndarray:
    """
    Draw light effects on an image for preview.
    
    Args:
        image: Input image to draw on
        lights: List of light dictionaries
        grid_size: Size of one grid square in pixels
        show_range: Whether to show light range circles
        alpha: Transparency for light effects
        
    Returns:
        Image with light effects drawn
    """
    if not lights or image is None:
        return image
    
    preview_image = image.copy()
    overlay = image.copy()
    
    for light in lights:
        # Get position in pixels
        if "_original_pixel_x" in light and "_original_pixel_y" in light:
            # Use original pixel coordinates for accurate drawing
            pixel_x = int(light["_original_pixel_x"])
            pixel_y = int(light["_original_pixel_y"])
        elif "position" in light:
            # Fall back to grid coordinates if original pixels aren't available
            pos_x = light["position"]["x"]
            pos_y = light["position"]["y"]
            
            # Convert from grid coordinates to pixels
            pixel_x = int(pos_x * grid_size)
            pixel_y = int(pos_y * grid_size)
        else:
            continue
        
        # Parse color
        color_hex = light.get("color", "ffffffff")
        if color_hex.startswith("ff"):
            color_hex = color_hex[2:]  # Remove alpha prefix
        
        try:
            # Convert hex to BGR
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            color_bgr = (b, g, r)
        except:
            color_bgr = (255, 255, 200)  # Default warm white
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
            continue
            
        # Draw light center point
        cv2.circle(preview_image, (pixel_x, pixel_y), 3, color_bgr, -1)
        cv2.circle(preview_image, (pixel_x, pixel_y), 4, (255, 255, 255), 1)
        
        # Draw range circle if requested
        if show_range:
            light_range = light.get("range", 2.0)
            # If we're using original pixel coords, adjust range calculation
            if "_original_pixel_x" in light and "_original_pixel_y" in light:
                # The range is in grid units, so we still need to convert to pixels
                range_radius = int(light_range * grid_size)
            else:
                # Standard calculation
                range_radius = int(light_range * grid_size)
            cv2.circle(overlay, (pixel_x, pixel_y), range_radius, color_bgr, 2)
            
        # Draw light effect (soft glow)
        intensity = light.get("intensity", 1.0)
        # Similar adjustment for effect radius
        if "_original_pixel_x" in light and "_original_pixel_y" in light:
            effect_radius = int(light.get("range", 2.0) * grid_size * 0.7)
        else:
            effect_radius = int(light.get("range", 2.0) * grid_size * 0.7)
        
        if effect_radius > 0:
            # Create a soft gradient effect
            for i in range(3, effect_radius, max(1, effect_radius // 8)):
                fade_alpha = alpha * intensity * (1.0 - (i / effect_radius)) * 0.5
                if fade_alpha > 0.01:
                    # Draw the circle with bounds checking
                    cv2.circle(overlay, (pixel_x, pixel_y), i, color_bgr, 1)
    
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, preview_image, 1 - alpha, 0, preview_image)
    
    return preview_image


def find_light_under_cursor(lights: List[Dict], x: int, y: int, grid_size: float = 70.0, 
                           max_distance: int = 15) -> int:
    """
    Find a light point under the cursor position.
    
    Args:
        lights: List of light dictionaries
        x, y: Cursor position in pixels
        grid_size: Size of one grid square in pixels
        max_distance: Maximum distance in pixels to consider a hit
        
    Returns:
        Index of the light under cursor, or -1 if none found
    """
    if not lights:
        return -1
    
    closest_index = -1
    closest_distance = float('inf')
    
    for i, light in enumerate(lights):
        # Use original pixel coordinates if available
        if "_original_pixel_x" in light and "_original_pixel_y" in light:
            light_x = int(light["_original_pixel_x"])
            light_y = int(light["_original_pixel_y"])
        elif "position" in light:
            # Fall back to grid position
            pos_x = light["position"]["x"]
            pos_y = light["position"]["y"]
            
            light_x = int(pos_x * grid_size)
            light_y = int(pos_y * grid_size)
        else:
            continue
        
        # Calculate distance
        distance = np.sqrt((x - light_x) ** 2 + (y - light_y) ** 2)
        
        if distance <= max_distance and distance < closest_distance:
            closest_distance = distance
            closest_index = i
    
    return closest_index


def move_light(lights: List[Dict], light_index: int, new_x: int, new_y: int, 
               grid_size: float = 70.0, use_grid_coords: bool = True) -> bool:
    """
    Move a light to a new position.
    
    Args:
        lights: List of light dictionaries
        light_index: Index of light to move
        new_x, new_y: New position (in pixels)
        grid_size: Size of one grid square in pixels
        use_grid_coords: Whether to store coordinates as grid units
        
    Returns:
        True if light was moved successfully
    """
    if light_index < 0 or light_index >= len(lights):
        return False
    
    # Always store original pixel coordinates
    lights[light_index]["_original_pixel_x"] = float(new_x)
    lights[light_index]["_original_pixel_y"] = float(new_y)
    
    if use_grid_coords:
        # Convert pixels to grid coordinates
        grid_x = new_x / grid_size
        grid_y = new_y / grid_size
        lights[light_index]["position"] = {
            "x": float(grid_x),
            "y": float(grid_y)
        }
    else:
        # Store as pixel coordinates
        lights[light_index]["position"] = {
            "x": float(new_x),
            "y": float(new_y)
        }
    
    return True
