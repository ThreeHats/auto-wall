import unittest
import cv2
import numpy as np
import os
import sys
import tempfile

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wall_detection.image_utils import preprocess_image, detect_edges, convert_to_rgb

class TestImageUtils(unittest.TestCase):
    def setUp(self):
        # Create a simple test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a simple rectangle
        cv2.rectangle(self.test_image, (20, 20), (80, 80), (255, 255, 255), 2)
    
    def test_preprocess_image(self):
        processed = preprocess_image(self.test_image)
        self.assertEqual(len(processed.shape), 2)  # Should be grayscale (2D)
    
    def test_detect_edges(self):
        processed = preprocess_image(self.test_image)
        edges = detect_edges(processed)
        self.assertEqual(len(edges.shape), 2)  # Should be a 2D binary image
    
    def test_convert_to_rgb(self):
        rgb = convert_to_rgb(self.test_image)
        self.assertEqual(len(rgb.shape), 3)  # Should still be 3D
        self.assertEqual(rgb.shape[2], 3)  # Should have 3 channels

if __name__ == "__main__":
    unittest.main()
