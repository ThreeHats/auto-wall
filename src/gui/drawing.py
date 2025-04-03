import pygame
import numpy as np
import cv2

class DrawingCanvas:
    def __init__(self, width=800, height=600, background_color=(255, 255, 255)):
        """Initialize a drawing canvas for wall drawing."""
        self.width = width
        self.height = height
        self.background_color = background_color
        self.surface = pygame.Surface((width, height))
        self.surface.fill(background_color)
        self.drawing = False
        self.last_pos = None
        self.line_color = (0, 0, 0)  # Default: black
        self.line_thickness = 2
    
    def start_drawing(self, position):
        """Start a drawing operation."""
        self.drawing = True
        self.last_pos = position
    
    def stop_drawing(self):
        """Stop the current drawing operation."""
        self.drawing = False
        self.last_pos = None
    
    def draw_line(self, current_pos):
        """Draw a line from last position to current position."""
        if self.drawing and self.last_pos:
            pygame.draw.line(
                self.surface, 
                self.line_color,
                self.last_pos,
                current_pos,
                self.line_thickness
            )
            self.last_pos = current_pos
    
    def clear(self):
        """Clear the drawing surface."""
        self.surface.fill(self.background_color)
    
    def get_surface(self):
        """Get the current drawing surface."""
        return self.surface
    
    def set_background_image(self, image):
        """Set a background image for the canvas."""
        # Convert OpenCV image to Pygame surface
        image = cv2.resize(image, (self.width, self.height))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
        self.surface.blit(image_surface, (0, 0))
    
    def get_image(self):
        """Get the current drawing as an OpenCV image."""
        pygame_surface_array = pygame.surfarray.array3d(self.surface)
        # Convert from RGB to BGR (OpenCV format)
        image = cv2.cvtColor(pygame_surface_array.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
        return image
