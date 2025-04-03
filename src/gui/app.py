import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QWidget, QFileDialog, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from src.wall_detection.detector import detect_walls, draw_walls, merge_contours
from src.wall_detection.image_utils import load_image, save_image, convert_to_rgb


class WallDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto-Wall: Battle Map Wall Detection")
        self.setGeometry(100, 100, 1000, 800)

        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Controls
        self.controls_layout = QVBoxLayout()
        self.layout.addLayout(self.controls_layout)

        # Sliders
        self.sliders = {}
        self.add_slider("Min Area", 0, 10000, 100)
        self.add_slider("Max Area", 0, 100000, 10000)
        self.add_slider("Blur", 1, 21, 5, step=2)
        self.add_slider("Canny1", 0, 255, 50)
        self.add_slider("Canny2", 0, 255, 150)
        self.add_slider("Edge Margin", 0, 50, 5)  # New slider for edge margin
        
        # Use a scaling factor of 10 for float values (0.1 to 10.0 with 0.1 precision)
        self.add_slider("Min Merge Distance", 1, 100, 30, scale_factor=0.1)  # Default 3.0

        # Buttons
        self.buttons_layout = QHBoxLayout()
        self.controls_layout.addLayout(self.buttons_layout)

        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        self.buttons_layout.addWidget(self.open_button)

        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.buttons_layout.addWidget(self.save_button)

        # Checkboxes for merge options
        self.merge_options_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.merge_options_layout)

        self.merge_before_min_area = QCheckBox("Merge Before Min Area")
        self.merge_before_min_area.setChecked(False)
        self.merge_options_layout.addWidget(self.merge_before_min_area)

        self.merge_after_min_area = QCheckBox("Merge After Min Area")
        self.merge_after_min_area.setChecked(False)
        self.merge_options_layout.addWidget(self.merge_after_min_area)

        # State
        self.current_image = None
        self.processed_image = None

    def add_slider(self, label, min_val, max_val, initial_val, step=1, scale_factor=None):
        """Add a slider with a label."""
        slider_layout = QHBoxLayout()
        
        # Store scale factor if provided
        if scale_factor:
            display_value = initial_val * scale_factor
            display_text = f"{label}: {display_value:.1f}"
        else:
            display_value = initial_val
            display_text = f"{label}: {display_value}"
            
        slider_label = QLabel(display_text)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(initial_val)
        slider.setSingleStep(step)
        
        # Connect with scale factor handling
        slider.valueChanged.connect(
            lambda value, lbl=slider_label, lbl_text=label, sf=scale_factor: 
            self.update_slider(lbl, lbl_text, value, sf)
        )
        slider.valueChanged.connect(self.update_image)

        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(slider)
        self.controls_layout.addLayout(slider_layout)

        self.sliders[label] = slider

    def update_slider(self, label, label_text, value, scale_factor=None):
        """Update the slider label."""
        if scale_factor:
            scaled_value = value * scale_factor
            label.setText(f"{label_text}: {scaled_value:.1f}")
        else:
            label.setText(f"{label_text}: {value}")

    def open_image(self):
        """Open an image file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.current_image = load_image(file_path)
            self.update_image()

    def save_image(self):
        """Save the processed image."""
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                save_image(self.processed_image, file_path)

    def update_image(self):
        """Update the displayed image based on the current settings."""
        if self.current_image is None:
            return

        # Get slider values
        min_area = self.sliders["Min Area"].value()
        max_area = self.sliders["Max Area"].value()
        blur = self.sliders["Blur"].value()
        
        # Handle special case for blur=1 (no blur) and ensure odd values
        if blur > 1 and blur % 2 == 0:
            blur += 1  # Ensure kernel size is odd when > 1
        
        canny1 = self.sliders["Canny1"].value()
        canny2 = self.sliders["Canny2"].value()
        edge_margin = self.sliders["Edge Margin"].value()
        
        # Get min_merge_distance as a float value
        min_merge_distance = self.sliders["Min Merge Distance"].value() * 0.1
        
        # Debug output of parameters
        print(f"Parameters: min_area={min_area}, blur={blur}, canny1={canny1}, canny2={canny2}, edge_margin={edge_margin}")

        # Process the image directly with detect_walls
        contours = detect_walls(
            self.current_image,
            min_contour_area=min_area,
            max_contour_area=max_area,
            blur_kernel_size=blur,
            canny_threshold1=canny1,
            canny_threshold2=canny2,
            edge_margin=edge_margin
        )
        
        print(f"Detected {len(contours)} contours before merging")

        # Merge before Min Area if specified
        if self.merge_before_min_area.isChecked():
            contours = merge_contours(
                self.current_image, 
                contours, 
                min_merge_distance=min_merge_distance
            )
            print(f"After merge before min area: {len(contours)} contours")

        # Filter contours by area
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        print(f"After min area filter: {len(contours)} contours")

        # Merge after Min Area if specified
        if self.merge_after_min_area.isChecked():
            contours = merge_contours(
                self.current_image, 
                contours, 
                min_merge_distance=min_merge_distance
            )
            print(f"After merge after min area: {len(contours)} contours")

        # Ensure contours are not empty
        if not contours:
            print("No contours found after processing.")
            self.processed_image = self.current_image.copy()
        else:
            # Draw merged contours
            self.processed_image = draw_walls(self.current_image, contours)

        # Convert to QPixmap and display
        rgb_image = convert_to_rgb(self.processed_image)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WallDetectionApp()
    window.show()
    sys.exit(app.exec())