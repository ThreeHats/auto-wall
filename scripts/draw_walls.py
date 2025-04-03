import os
import cv2
import sys
import pygame

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.app import WallDetectionApp

def save_image(image, filename):
    """Save an image to the specified filename."""
    cv2.imwrite(filename, image)

def load_image(filename):
    """Load an image from the specified filename."""
    if os.path.exists(filename):
        return cv2.imread(filename)
    else:
        raise FileNotFoundError(f"The file {filename} does not exist.")

def save_map_as_png(surface, filename):
    """Save the current map surface as a PNG file."""
    cv2.imwrite(filename, surface)

def load_map_from_png(filename):
    """Load a map from a PNG file."""
    if os.path.exists(filename):
        return cv2.imread(filename)
    else:
        raise FileNotFoundError(f"The file {filename} does not exist.")

if __name__ == "__main__":
    # Launch the full application
    app = WallDetectionApp()
    app.run()