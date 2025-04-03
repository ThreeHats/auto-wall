import pygame
import cv2
import numpy as np
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.gui.drawing import DrawingCanvas
from src.wall_detection.detector import detect_walls, draw_walls
from src.wall_detection.image_utils import load_image, save_image, preprocess_image, convert_to_rgb
from src.utils.file_handlers import save_wall_data, load_wall_data

class WallDetectionApp:
    def __init__(self):
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Auto-Wall: Battle Map Wall Detection')
        
        self.canvas = DrawingCanvas(self.width, self.height)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Application state
        self.current_image_path = None
        self.current_image = None
        self.detected_contours = None
        self.draw_mode = False  # True for manual drawing, False for detection
        
        # UI elements
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Wall detection parameters
        self.min_contour_area = 100
        self.max_contour_area = None  # No upper limit by default
        self.blur_kernel_size = 5
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
    
    def load_image(self, image_path):
        """Load an image for processing."""
        try:
            self.current_image = load_image(image_path)
            self.current_image_path = image_path
            self.canvas.set_background_image(self.current_image)
            self.detected_contours = None
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def detect_walls_in_image(self):
        """Detect walls in the current image."""
        if self.current_image is not None:
            try:
                contours = detect_walls(
                    self.current_image, 
                    min_contour_area=self.min_contour_area,
                    max_contour_area=self.max_contour_area,
                    blur_kernel_size=self.blur_kernel_size,
                    canny_threshold1=self.canny_threshold1,
                    canny_threshold2=self.canny_threshold2
                )
                self.detected_contours = contours
                image_with_walls = draw_walls(self.current_image, contours)
                self.canvas.set_background_image(image_with_walls)
            except Exception as e:
                print(f"Error detecting walls: {e}")
    
    def save_result(self, filename):
        """Save the current canvas as an image."""
        try:
            image = self.canvas.get_image()
            save_image(image, filename)
            print(f"Saved result to {filename}")
        except Exception as e:
            print(f"Error saving image: {e}")
    
    def handle_events(self):
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                # Check for Ctrl + S for saving
                if event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.save_result('data/output/result.png')
                
                elif event.key == pygame.K_o:  # Open image
                    # This would normally open a file dialog
                    self.load_image('data/input/CavernPitPublic-785x1024.jpg')
                elif event.key == pygame.K_d:  # Detect walls
                    self.detect_walls_in_image()
                elif event.key == pygame.K_m:  # Toggle drawing mode
                    self.draw_mode = not self.draw_mode
                elif event.key == pygame.K_c:  # Clear canvas
                    if self.current_image is not None:
                        self.canvas.set_background_image(self.current_image)
                    else:
                        self.canvas.clear()
                
                # Parameter adjustment keys
                elif event.key == pygame.K_UP:
                    self.min_contour_area += 50
                    print(f"Min contour area: {self.min_contour_area}")
                elif event.key == pygame.K_DOWN and self.min_contour_area > 50:
                    self.min_contour_area -= 50
                    print(f"Min contour area: {self.min_contour_area}")
                elif event.key == pygame.K_RIGHT:
                    self.canny_threshold1 += 10
                    print(f"Canny Threshold1: {self.canny_threshold1}")
                elif event.key == pygame.K_LEFT and self.canny_threshold1 > 10:
                    self.canny_threshold1 -= 10
                    print(f"Canny Threshold1: {self.canny_threshold1}")
                elif event.key == pygame.K_w:
                    self.canny_threshold2 += 10
                    print(f"Canny Threshold2: {self.canny_threshold2}")
                elif event.key == pygame.K_s and self.canny_threshold2 > self.canny_threshold1 + 10:
                    self.canny_threshold2 -= 10
                    print(f"Canny Threshold2: {self.canny_threshold2}")
                elif event.key == pygame.K_b:
                    self.blur_kernel_size += 2
                    if self.blur_kernel_size % 2 == 0:
                        self.blur_kernel_size += 1  # Ensure kernel size is odd
                    print(f"Blur Kernel Size: {self.blur_kernel_size}")
                elif event.key == pygame.K_v and self.blur_kernel_size > 3:
                    self.blur_kernel_size -= 2
                    if self.blur_kernel_size % 2 == 0:
                        self.blur_kernel_size -= 1  # Ensure kernel size is odd
                    print(f"Blur Kernel Size: {self.blur_kernel_size}")
            
            elif self.draw_mode:
                # Handle drawing events
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.canvas.start_drawing(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.canvas.stop_drawing()
                elif event.type == pygame.MOUSEMOTION:
                    self.canvas.draw_line(event.pos)
    
    def update(self):
        """Update the application state."""
        pass  # No continuous updates needed for now
    
    def render(self):
        """Render the current state to the screen."""
        self.screen.fill((230, 230, 230))
        
        # Draw the canvas
        self.screen.blit(pygame.surfarray.make_surface(
            pygame.surfarray.array3d(self.canvas.get_surface())), (0, 0))
        
        # Draw UI elements
        mode_text = self.font.render(
            f"Mode: {'Drawing' if self.draw_mode else 'Detection'}", 
            True, (0, 0, 0))
        self.screen.blit(mode_text, (10, 10))
        
        # Show wall detection parameters
        params_text1 = self.font.render(
            f"Min Area: {self.min_contour_area} | Max Area: {self.max_contour_area or 'No limit'}", 
            True, (0, 0, 0))
        self.screen.blit(params_text1, (10, 35))
        
        params_text2 = self.font.render(
            f"Blur: {self.blur_kernel_size} | Canny1: {self.canny_threshold1} | Canny2: {self.canny_threshold2}", 
            True, (0, 0, 0))
        self.screen.blit(params_text2, (10, 60))
        
        help_text = self.font.render(
            "O: Open | D: Detect | Ctrl+S: Save | ↑↓: Min Area | ←→: Canny1 | W/S: Canny2 | B/V: Blur", 
            True, (0, 0, 0))
        self.screen.blit(help_text, (10, self.height - 30))
        
        pygame.display.flip()
    
    def run(self):
        """Run the main application loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()

# Keep original functions for backward compatibility
def load_image(image_path):
    """Load an image from the specified path."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def save_image(image, save_path):
    """Save an image to the specified path."""
    cv2.imwrite(save_path, image)

if __name__ == "__main__":
    app = WallDetectionApp()
    app.run()