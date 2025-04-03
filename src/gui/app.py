import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QWidget, QFileDialog
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
        self.add_slider("Min Area", 0, 1000, 100)
        self.add_slider("Max Area", 0, 10000, 10000)
        self.add_slider("Blur", 3, 21, 5, step=2)
        self.add_slider("Canny1", 0, 255, 50)
        self.add_slider("Canny2", 0, 255, 150)

        # Buttons
        self.buttons_layout = QHBoxLayout()
        self.controls_layout.addLayout(self.buttons_layout)

        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        self.buttons_layout.addWidget(self.open_button)

        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.buttons_layout.addWidget(self.save_button)

        # State
        self.current_image = None
        self.processed_image = None

    def add_slider(self, label, min_val, max_val, initial_val, step=1):
        """Add a slider with a label."""
        slider_layout = QHBoxLayout()
        slider_label = QLabel(f"{label}: {initial_val}")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(initial_val)
        slider.setSingleStep(step)
        slider.valueChanged.connect(lambda value, lbl=slider_label, lbl_text=label: self.update_slider(lbl, lbl_text, value))
        slider.valueChanged.connect(self.update_image)

        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(slider)
        self.controls_layout.addLayout(slider_layout)

        self.sliders[label] = slider

    def update_slider(self, label, label_text, value):
        """Update the slider label."""
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
        if blur % 2 == 0:
            blur += 1  # Ensure blur kernel size is odd
        canny1 = self.sliders["Canny1"].value()
        canny2 = self.sliders["Canny2"].value()

        # Process the image
        contours = detect_walls(
            self.current_image,
            min_contour_area=min_area,
            max_contour_area=max_area,
            blur_kernel_size=blur,
            canny_threshold1=canny1,
            canny_threshold2=canny2,
        )

        # Merge nearby contours
        merged_contours = merge_contours(self.current_image, contours)

        # Draw merged contours
        self.processed_image = draw_walls(self.current_image, merged_contours)

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