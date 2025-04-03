import cv2
import numpy as np

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
    Blend an image with a transparent mask.
    
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
    
    # Extract alpha channel from mask
    alpha_mask = mask[:, :, 3] / 255.0
    
    # Blend each channel
    for c in range(3):  # Only blend BGR channels
        bgra_image[:, :, c] = bgra_image[:, :, c] * (1 - alpha_mask) + mask[:, :, c] * alpha_mask
    
    return bgra_image

def draw_on_mask(mask, x, y, brush_size, color=(0, 255, 0, 255), erase=False):
    """
    Draw on a mask at the specified coordinates.
    
    Parameters:
    - mask: BGRA mask to draw on
    - x, y: Coordinates to draw at
    - brush_size: Size of the brush/eraser
    - color: BGRA color to draw with (ignored if erase=True)
    - erase: Whether to erase (True) or draw (False)
    
    Returns:
    - Updated mask
    """
    # Create a temporary mask for the brush
    height, width = mask.shape[:2]
    brush_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw a filled circle on the temporary mask
    cv2.circle(brush_mask, (x, y), brush_size, 255, -1)
    
    # Apply the brush to the mask
    if erase:
        # Set alpha channel to 0 where brush was applied
        mask[brush_mask == 255, 3] = 0
    else:
        # Set color and alpha where brush was applied
        mask[brush_mask == 255] = color
    
    return mask
