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

class Slider:
    def __init__(self, x, y, width, min_val, max_val, initial_val, label, font):
        self.rect = pygame.Rect(x, y, width, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.font = font
        self.dragging = False

    def draw(self, screen):
        pygame.draw.rect(screen, (200, 200, 200), self.rect)
        pygame.draw.rect(screen, (100, 100, 100), (self.rect.x, self.rect.y + 8, self.rect.width, 4))
        handle_x = self.rect.x + int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        pygame.draw.circle(screen, (0, 0, 0), (handle_x, self.rect.y + 10), 8)
        label_surface = self.font.render(f"{self.label}: {self.value}", True, (0, 0, 0))
        screen.blit(label_surface, (self.rect.x, self.rect.y - 20))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x
            rel_x = max(0, min(self.rect.width, rel_x))
            self.value = int(self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val))

class WallDetectionApp:
    def __init__(self):
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Auto-Wall: Battle Map Wall Detection')
        
        self.canvas = DrawingCanvas(self.width, self.height - 150)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Application state
        self.current_image_path = None
        self.current_image = None
        self.detected_contours = None
        
        # UI elements
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Wall detection parameters
        self.min_contour_area = 100
        self.max_contour_area = 10000
        self.blur_kernel_size = 5
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # Sliders
        self.sliders = [
            Slider(10, self.height - 140, 200, 0, 1000, self.min_contour_area, "Min Area", self.font),
            Slider(10, self.height - 110, 200, 0, 10000, self.max_contour_area, "Max Area", self.font),
            Slider(10, self.height - 80, 200, 0, 21, self.blur_kernel_size, "Blur", self.font),
            Slider(10, self.height - 50, 200, 0, 255, self.canny_threshold1, "Canny1", self.font),
            Slider(10, self.height - 20, 200, 0, 255, self.canny_threshold2, "Canny2", self.font),
        ]
    
    def load_image(self, image_path):
        """Load an image for processing."""
        try:
            self.current_image = load_image(image_path)
            self.current_image_path = image_path
            self.canvas.set_background_image(self.current_image)
            self.detect_walls_in_image()
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def detect_walls_in_image(self):
        """Detect walls in the current image."""
        if self.current_image is not None:
            try:
                self.min_contour_area = self.sliders[0].value
                self.max_contour_area = self.sliders[1].value
                self.blur_kernel_size = self.sliders[2].value
                if self.blur_kernel_size % 2 == 0:
                    self.blur_kernel_size += 1  # Ensure kernel size is odd
                self.canny_threshold1 = self.sliders[3].value
                self.canny_threshold2 = self.sliders[4].value
                
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
                if event.key == pygame.K_o:  # Open image
                    self.load_image('data/input/CavernPitPublic-785x1024.jpg')
                elif event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_CTRL):  # Save result
                    self.save_result('data/output/result.png')
            
            for slider in self.sliders:
                slider.handle_event(event)
    
    def update(self):
        """Update the application state."""
        self.detect_walls_in_image()
    
    def render(self):
        """Render the current state to the screen."""
        self.screen.fill((230, 230, 230))
        
        # Draw the canvas
        self.screen.blit(pygame.surfarray.make_surface(
            pygame.surfarray.array3d(self.canvas.get_surface())), (0, 0))
        
        # Draw sliders
        for slider in self.sliders:
            slider.draw(self.screen)
        
        pygame.display.flip()
    
    def run(self):
        """Run the main application loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    app = WallDetectionApp()
    app.run()