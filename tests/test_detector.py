import unittest
import cv2
import numpy as np
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wall_detection.detector import detect_walls, draw_walls

class TestDetector(unittest.TestCase):
    def setUp(self):
        # Create a simple test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a simple rectangle
        cv2.rectangle(self.test_image, (20, 20), (80, 80), (255, 255, 255), 2)
    
    def test_detect_walls(self):
        contours = detect_walls(self.test_image)
        self.assertIsNotNone(contours)
        self.assertTrue(len(contours) > 0)
    
    def test_draw_walls(self):
        contours = detect_walls(self.test_image)
        result = draw_walls(self.test_image, contours)
        self.assertEqual(result.shape, self.test_image.shape)

if __name__ == "__main__":
    unittest.main()
