import os
import cv2
import json
import numpy as np
import pygame

def ensure_directory_exists(directory):
    """Ensure that the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(image, filename):
    """Save an image to the specified filename."""
    directory = os.path.dirname(filename)
    ensure_directory_exists(directory)
    cv2.imwrite(filename, image)

def load_image(filename):
    """Load an image from the specified filename."""
    if os.path.exists(filename):
        image = cv2.imread(filename)
        if image is None:
            raise ValueError(f"Failed to load image: {filename}")
        return image
    else:
        raise FileNotFoundError(f"The file {filename} does not exist.")

def save_wall_data(walls, filename):
    """Save wall data to a JSON file."""
    directory = os.path.dirname(filename)
    ensure_directory_exists(directory)
    
    with open(filename, 'w') as f:
        json.dump(walls, f)

def load_wall_data(filename):
    """Load wall data from a JSON file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
        
    with open(filename, 'r') as f:
        return json.load(f)

def pygame_surface_to_cv2(surface):
    """Convert a Pygame surface to an OpenCV image."""
    # Get the size of the surface
    width, height = surface.get_size()
    
    # Create a numpy array from the surface
    surf_array = pygame.surfarray.array3d(surface)
    
    # Convert from (width, height, 3) to (height, width, 3)
    surf_array = surf_array.transpose([1, 0, 2])
    
    # Convert from RGB to BGR
    image = cv2.cvtColor(surf_array, cv2.COLOR_RGB2BGR)
    
    return image

def cv2_to_pygame_surface(image):
    """Convert an OpenCV image to a Pygame surface."""
    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a Pygame surface from the numpy array
    surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
    
    return surface
