import cv2
import numpy as np

def detect_walls(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection with adjusted thresholds
    edges_tuned = cv2.Canny(blurred, 50, 150)

    # Find contours with the adjusted edges
    contours_tuned, _ = cv2.findContours(edges_tuned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours_tuned

def draw_walls(image, contours):
    image_with_tuned_walls = image.copy()
    cv2.drawContours(image_with_tuned_walls, contours, -1, (0, 255, 0), 2)
    return image_with_tuned_walls