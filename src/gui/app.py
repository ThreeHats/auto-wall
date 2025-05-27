import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QWidget, 
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup, QColorDialog, QListWidget, QListWidgetItem,
    QScrollArea, QSizePolicy, QDialog, QDialogButtonBox, QFrame, QSpinBox, QInputDialog, QDoubleSpinBox,
    QMessageBox, QGridLayout, QComboBox, QMenu, QLineEdit
)
from PyQt6.QtCore import Qt, QPoint, QRect, QBuffer, QUrl
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QMouseEvent, QCursor, QClipboard, QGuiApplication, QKeySequence, QShortcut, QDesktopServices, QAction
import cv2
import numpy as np
import math
import json
import requests
import io
import urllib.parse
from sklearn.cluster import KMeans
from collections import deque
import copy

from src.wall_detection.detector import detect_walls, draw_walls, merge_contours, split_edge_contours, remove_hatching_lines
from src.wall_detection.image_utils import load_image, save_image, convert_to_rgb
from src.wall_detection.mask_editor import create_mask_from_contours, blend_image_with_mask, draw_on_mask, export_mask_to_foundry_json, contours_to_foundry_walls, thin_contour
from src.utils.update_checker import check_for_updates
from src.gui.drawing_tools import DrawingTools

# Define preset file paths
DETECTION_PRESETS_FILE = "detection_presets.json"
EXPORT_PRESETS_FILE = "export_presets.json" # Add this line

class InteractiveImageLabel(QLabel):
    """Custom QLabel that handles mouse events for contour/line deletion and mask editing."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        # Set mouse tracking to enable hover effects
        self.setMouseTracking(True)
        # Enable mouse interaction
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # For drag selection
        self.selection_start = None
        self.selection_current = None
        # For drawing
        self.last_point = None
        
    def mousePressEvent(self, event):
        """Handle mouse click events."""
        if self.parent_app:
            pos = event.position()
            x, y = int(pos.x()), int(pos.y())
            self.selection_start = QPoint(x, y)
            self.selection_current = self.selection_start
            
            # Handle left button clicks
            if event.button() == Qt.MouseButton.LeftButton:
                if self.parent_app.deletion_mode_enabled or self.parent_app.color_selection_mode_enabled:
                    self.parent_app.start_selection(x, y)
                elif self.parent_app.edit_mask_mode_enabled:
                    # Start drawing on mask
                    self.last_point = QPoint(x, y)
                    self.parent_app.drawing_tools.start_drawing(x, y)
                elif self.parent_app.thin_mode_enabled:
                    self.parent_app.start_selection(x, y)
                
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for hover highlighting, drag selection, and drawing."""
        if self.parent_app:
            pos = event.position()
            x, y = int(pos.x()), int(pos.y())
            
            # If dragging with left button
            if self.selection_start and event.buttons() & Qt.MouseButton.LeftButton:
                self.selection_current = QPoint(x, y)
                if self.parent_app.deletion_mode_enabled or self.parent_app.color_selection_mode_enabled:
                    self.parent_app.update_selection(x, y)
                elif self.parent_app.edit_mask_mode_enabled and self.last_point:
                    # Continue drawing on mask
                    current_point = QPoint(x, y)
                    self.parent_app.drawing_tools.continue_drawing(self.last_point.x(), self.last_point.y(), x, y)
                    self.last_point = current_point
                elif self.parent_app.thin_mode_enabled:
                    self.parent_app.update_selection(x, y)
            # Just hovering - this always runs for any mouse movement
            else:
                if self.parent_app.deletion_mode_enabled or self.parent_app.thin_mode_enabled:
                    self.parent_app.handle_hover(pos.x(), pos.y())
                elif self.parent_app.edit_mask_mode_enabled:
                    # Always update brush preview when hovering in edit mask mode
                    self.parent_app.drawing_tools.update_brush_preview(x, y)
                
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for completing drag selection or drawing."""
        if self.parent_app:
            if self.selection_start and event.button() == Qt.MouseButton.LeftButton:
                pos = event.position()
                x, y = int(pos.x()), int(pos.y())
                self.selection_current = QPoint(x, y)
                
                if self.parent_app.deletion_mode_enabled or self.parent_app.color_selection_mode_enabled:
                    self.parent_app.end_selection(x, y)
                elif self.parent_app.edit_mask_mode_enabled:
                    # End drawing on mask
                    self.parent_app.drawing_tools.end_drawing()
                    self.last_point = None
                elif self.parent_app.thin_mode_enabled:
                    self.parent_app.end_selection(x, y)
                
                # Clear selection points
                self.selection_start = None
                self.selection_current = None
                
        super().mouseReleaseEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leaving the widget."""
        if self.parent_app:
            if self.parent_app.deletion_mode_enabled or self.parent_app.thin_mode_enabled:
                self.parent_app.clear_hover()
            elif self.parent_app.edit_mask_mode_enabled:
                # Clear brush preview when mouse leaves the widget
                self.parent_app.drawing_tools.clear_brush_preview()
        super().leaveEvent(event)


class WallDetectionApp(QMainWindow):
    def __init__(self, version="0.9.0", github_repo="ThreeHats/auto-wall"):
        super().__init__()
        self.app_version = version
        self.github_repo = github_repo
        self.update_available = False
        self.update_url = ""

        self.drawing_tools = DrawingTools(self)
        
        self.setWindowTitle(f"Auto-Wall: Battle Map Wall Detection v{self.app_version}")
        
        # Get the screen size and set the window to maximize
        screen = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        
        # Apply dark theme stylesheet
        self.apply_stylesheet()
        
        self.showMaximized()

        # Main layout - use central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Use horizontal layout for main container (controls on left, image on right)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Left panel for controls (using scroll area for many controls)
        self.controls_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_panel)
        
        # Detection mode selection at the top
        self.detection_mode_title = QLabel("Detection Mode:")
        self.detection_mode_title.setStyleSheet("font-weight: bold;")
        self.controls_layout.addWidget(self.detection_mode_title)
        
        self.detection_mode_layout = QHBoxLayout()
        self.controls_layout.addLayout(self.detection_mode_layout)
        
        self.detection_mode_group = QButtonGroup()
        self.edge_detection_radio = QRadioButton("Edge Detection")
        self.color_detection_radio = QRadioButton("Color Detection")
        self.edge_detection_radio.setChecked(True)  # Default to edge detection
        self.detection_mode_group.addButton(self.edge_detection_radio)
        self.detection_mode_group.addButton(self.color_detection_radio)
        
        self.detection_mode_layout.addWidget(self.edge_detection_radio)
        self.detection_mode_layout.addWidget(self.color_detection_radio)
        
        # Connect detection mode radio buttons
        self.edge_detection_radio.toggled.connect(self.toggle_detection_mode_radio)
        self.color_detection_radio.toggled.connect(self.toggle_detection_mode_radio)
        
        # Add a separator after detection mode
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.controls_layout.addWidget(separator)

        # Mode selection (Detection/Deletion/Color Selection/Edit Mask)
        self.mode_layout = QHBoxLayout()
        self.controls_layout.addLayout(self.mode_layout)
        
        self.mode_label = QLabel("Tool:")
        self.mode_layout.addWidget(self.mode_label)
        
        self.deletion_mode_radio = QRadioButton("Deletion")
        self.thin_mode_radio = QRadioButton("Thin")
        self.deletion_mode_radio.setChecked(True)
        self.color_selection_mode_radio = QRadioButton("Color Pick")
        self.color_selection_mode_radio.setVisible(False)  # Hide this radio button initially
        self.edit_mask_mode_radio = QRadioButton("Edit Mask")
        self.edit_mask_mode_radio.setVisible(False)  # Hide this radio button initially

        self.mode_layout.addWidget(self.deletion_mode_radio)
        self.mode_layout.addWidget(self.thin_mode_radio)
        self.mode_layout.addWidget(self.color_selection_mode_radio)
        self.mode_layout.addWidget(self.edit_mask_mode_radio)
        
        # Connect mode radio buttons
        self.deletion_mode_radio.toggled.connect(self.toggle_mode)
        self.thin_mode_radio.toggled.connect(self.toggle_mode)
        self.color_selection_mode_radio.toggled.connect(self.toggle_mode)
        self.edit_mask_mode_radio.toggled.connect(self.toggle_mode)

        # Add a separator after Tool selection
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.controls_layout.addWidget(separator)

        # Add mask editing options
        self.mask_edit_options = QWidget()
        self.mask_edit_layout = QVBoxLayout(self.mask_edit_options)
        self.mask_edit_layout.setContentsMargins(0, 0, 0, 0)
        
        # Brush size control
        self.brush_size_layout = QHBoxLayout()
        self.brush_size_label = QLabel("Brush Size:")
        self.brush_size_layout.addWidget(self.brush_size_label)
        
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.valueChanged.connect(self.drawing_tools.update_brush_size)
        self.brush_size_layout.addWidget(self.brush_size_slider)
        
        self.brush_size_value = QLabel("10")
        self.brush_size_layout.addWidget(self.brush_size_value)
        self.mask_edit_layout.addLayout(self.brush_size_layout)
        
        # Draw/Erase mode
        self.draw_mode_layout = QHBoxLayout()
        self.draw_radio = QRadioButton("Draw")
        self.erase_radio = QRadioButton("Erase")
        self.draw_radio.setChecked(True)
        
        self.draw_mode_group = QButtonGroup()
        self.draw_mode_group.addButton(self.draw_radio)
        self.draw_mode_group.addButton(self.erase_radio)
        
        self.draw_mode_layout.addWidget(self.draw_radio)
        self.draw_mode_layout.addWidget(self.erase_radio)
        self.mask_edit_layout.addLayout(self.draw_mode_layout)
        
        # Add drawing tools section
        self.drawing_tools_label = QLabel("Drawing Tools:")
        self.drawing_tools_label.setStyleSheet("font-weight: bold;")
        self.mask_edit_layout.addWidget(self.drawing_tools_label)
        
        # Drawing tool selection
        self.draw_tool_layout = QGridLayout()
        
        # Create radio buttons for each drawing tool
        self.brush_tool_radio = QRadioButton("Brush")
        self.line_tool_radio = QRadioButton("Line")
        self.rectangle_tool_radio = QRadioButton("Rectangle")
        self.circle_tool_radio = QRadioButton("Circle")
        self.ellipse_tool_radio = QRadioButton("Ellipse")
        self.fill_tool_radio = QRadioButton("Fill")
        
        # Set brush as default
        self.brush_tool_radio.setChecked(True)
        
        # Add tools to a button group
        self.draw_tool_group = QButtonGroup()
        self.draw_tool_group.addButton(self.brush_tool_radio)
        self.draw_tool_group.addButton(self.line_tool_radio)
        self.draw_tool_group.addButton(self.rectangle_tool_radio)
        self.draw_tool_group.addButton(self.circle_tool_radio)
        self.draw_tool_group.addButton(self.ellipse_tool_radio)
        self.draw_tool_group.addButton(self.fill_tool_radio)
        
        # Add tools to the layout in a grid (2x3)
        self.draw_tool_layout.addWidget(self.brush_tool_radio, 0, 0)
        self.draw_tool_layout.addWidget(self.line_tool_radio, 0, 1)
        self.draw_tool_layout.addWidget(self.rectangle_tool_radio, 0, 2)
        self.draw_tool_layout.addWidget(self.circle_tool_radio, 1, 0)
        self.draw_tool_layout.addWidget(self.ellipse_tool_radio, 1, 1)
        self.draw_tool_layout.addWidget(self.fill_tool_radio, 1, 2)
        
        self.mask_edit_layout.addLayout(self.draw_tool_layout)

        # Add a separator at the bottom of mask editing options
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.mask_edit_layout.addWidget(separator)

        self.controls_layout.addWidget(self.mask_edit_options)
        self.mask_edit_options.setVisible(False)

        # Add thinning tool options
        self.thin_options = QWidget()
        self.thin_layout = QVBoxLayout(self.thin_options)
        self.thin_layout.setContentsMargins(0, 0, 0, 0)
        
        # Target width control
        self.target_width_layout = QHBoxLayout()
        self.target_width_label = QLabel("Target Width:")
        self.target_width_layout.addWidget(self.target_width_label)
        
        self.target_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.target_width_slider.setMinimum(1)
        self.target_width_slider.setMaximum(10)
        self.target_width_slider.setValue(5)
        self.target_width_slider.valueChanged.connect(self.update_target_width)
        self.target_width_layout.addWidget(self.target_width_slider)
        
        self.target_width_value = QLabel("5")
        self.target_width_layout.addWidget(self.target_width_value)
        self.thin_layout.addLayout(self.target_width_layout)
        
        # Max iterations control
        self.max_iterations_layout = QHBoxLayout()
        self.max_iterations_label = QLabel("Max Iterations:")
        self.max_iterations_layout.addWidget(self.max_iterations_label)
        
        self.max_iterations_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_iterations_slider.setMinimum(1)
        self.max_iterations_slider.setMaximum(20)
        self.max_iterations_slider.setValue(3)
        self.max_iterations_slider.valueChanged.connect(self.update_max_iterations)
        self.max_iterations_layout.addWidget(self.max_iterations_slider)
        
        self.max_iterations_value = QLabel("3")
        self.max_iterations_layout.addWidget(self.max_iterations_value)
        self.thin_layout.addLayout(self.max_iterations_layout)

        # Add a separator at the bottom of thinning options
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.thin_layout.addWidget(separator)
        
        # Add thinning options to main controls
        self.controls_layout.addWidget(self.thin_options)
        self.thin_options.setVisible(False)
        
        # Store thinning parameters
        self.target_width = 5
        self.max_iterations = 3

        # Add color selection options
        self.color_selection_options = QWidget()
        self.color_selection_layout = QVBoxLayout(self.color_selection_options)
        self.color_selection_layout.setContentsMargins(0, 0, 0, 0)
        
        self.color_count_layout = QHBoxLayout()
        self.color_selection_layout.addLayout(self.color_count_layout)

        self.color_count_label = QLabel("Colors:")
        self.color_count_layout.addWidget(self.color_count_label)
        
        self.color_count_spinner = QSpinBox()
        self.color_count_spinner.setMinimum(1)
        self.color_count_spinner.setMaximum(10)
        self.color_count_spinner.setValue(3)
        self.color_count_spinner.setToolTip("Number of colors to extract")
        self.color_count_layout.addWidget(self.color_count_spinner)

        # Add a separator at the bottom of color selection options
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.color_selection_layout.addWidget(separator)
        
        self.controls_layout.addWidget(self.color_selection_options)
        self.color_selection_options.setVisible(False)
        
        # Wrap controls in a scroll area to handle many options
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.controls_panel)
        self.scroll_area.setMinimumWidth(350)  # Set minimum width for controls panel
        self.scroll_area.setMaximumWidth(400)  # Set maximum width for controls panel
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Prevent horizontal scrolling
        self.main_layout.addWidget(self.scroll_area)
        
        # Right panel for image display
        self.image_panel = QWidget()
        self.image_layout = QVBoxLayout(self.image_panel)
        self.main_layout.addWidget(self.image_panel, 1)  # Give image panel more space (stretch factor 1)
        
        # Image display (using custom interactive label)
        self.image_label = InteractiveImageLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_layout.addWidget(self.image_label)

        # Sliders
        self.sliders = {}
        
        # Add Min Area slider with mode selection
        self.min_area_mode_layout = QHBoxLayout()
        self.min_area_mode_label = QLabel("Min Area Mode:")
        self.min_area_mode_layout.addWidget(self.min_area_mode_label)
        
        self.min_area_mode_group = QButtonGroup()
        self.min_area_percentage_radio = QRadioButton("Percentage")
        self.min_area_pixels_radio = QRadioButton("Pixels")
        self.min_area_percentage_radio.setChecked(True)  # Default to percentage mode
        self.min_area_mode_group.addButton(self.min_area_percentage_radio)
        self.min_area_mode_group.addButton(self.min_area_pixels_radio)
        self.min_area_mode_layout.addWidget(self.min_area_percentage_radio)
        self.min_area_mode_layout.addWidget(self.min_area_pixels_radio)
        
        # Connect mode radio buttons
        self.min_area_percentage_radio.toggled.connect(self.toggle_min_area_mode)
        self.min_area_pixels_radio.toggled.connect(self.toggle_min_area_mode)
        
        self.controls_layout.addLayout(self.min_area_mode_layout)
        
        # Min Area is now a percentage (0.0001% to 1% of image area) or pixels (1 to 1000)
        self.add_slider("Min Area", 1, 25000, 100, scale_factor=0.001)  # Default 0.1%
        self.add_slider("Smoothing", 1, 21, 5, step=2)  # Changed from "Blur"
        self.add_slider("Edge Sensitivity", 0, 255, 255)  # Changed from "Canny1"
        self.add_slider("Edge Threshold", 0, 255, 106)  # Changed from "Canny2"
        self.add_slider("Edge Margin", 0, 50, 0)
        
        # Checkboxes for merge options
        self.merge_options_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.merge_options_layout)

        self.merge_contours = QCheckBox("Merge Contours")
        self.merge_contours.setChecked(False)
        self.merge_options_layout.addWidget(self.merge_contours)

        # Use a scaling factor of 10 for float values (0 to 10.0 with 0.1 precision)
        self.add_slider("Min Merge Distance", 0, 100, 5, scale_factor=0.1)  # Default 0.5
        
        # Deletion mode is initially disabled
        self.deletion_mode_enabled = True
        self.color_selection_mode_enabled = False
        self.edit_mask_mode_enabled = False
        self.thin_mode_enabled = False  # Add new state variable for thin mode
        
        # Create layout for hatching controls
        self.hatching_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.hatching_layout)
        
        # Checkbox to enable/disable hatching removal
        self.remove_hatching_checkbox = QCheckBox("Enable Hatching Removal")
        self.remove_hatching_checkbox.setChecked(False)
        self.remove_hatching_checkbox.toggled.connect(self.toggle_hatching_removal)
        self.hatching_layout.addWidget(self.remove_hatching_checkbox)
        
        # Create container for hatching options
        self.hatching_options = QWidget()
        self.hatching_options_layout = QVBoxLayout(self.hatching_options)
        self.hatching_options_layout.setContentsMargins(0, 0, 0, 0)
        self.hatching_layout.addWidget(self.hatching_options)
        
        # Hatching color selection
        self.hatching_color_layout = QHBoxLayout()
        self.hatching_options_layout.addLayout(self.hatching_color_layout)
        
        self.hatching_color_label = QLabel("Hatching Color:")
        self.hatching_color_layout.addWidget(self.hatching_color_label)
        
        self.hatching_color_button = QPushButton()
        self.hatching_color_button.setFixedSize(30, 20)
        self.hatching_color_button.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.hatching_color_button.clicked.connect(self.select_hatching_color)
        self.hatching_color_layout.addWidget(self.hatching_color_button)
        self.hatching_color = QColor(0, 0, 0)  # Default to black
        
        # Hatching color threshold slider
        self.hatching_threshold_layout = QHBoxLayout()
        self.hatching_options_layout.addLayout(self.hatching_threshold_layout)
        
        self.hatching_threshold_label = QLabel("Color Threshold:")
        self.hatching_threshold_layout.addWidget(self.hatching_threshold_label)
        
        self.hatching_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.hatching_threshold_slider.setMinimum(0)
        self.hatching_threshold_slider.setMaximum(300)
        self.hatching_threshold_slider.setValue(100)  # Default value 10.0
        self.hatching_threshold_slider.valueChanged.connect(self.update_hatching_threshold)
        self.hatching_threshold_layout.addWidget(self.hatching_threshold_slider)
        
        self.hatching_threshold_value = QLabel("10.0")
        self.hatching_threshold_layout.addWidget(self.hatching_threshold_value)
        self.hatching_threshold = 10.0  # Store the actual value
        
        # Maximum hatching width slider
        self.hatching_width_layout = QHBoxLayout()
        self.hatching_options_layout.addLayout(self.hatching_width_layout)
        
        self.hatching_width_label = QLabel("Max Width:")
        self.hatching_width_layout.addWidget(self.hatching_width_label)
        
        self.hatching_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.hatching_width_slider.setMinimum(1)
        self.hatching_width_slider.setMaximum(20)
        self.hatching_width_slider.setValue(3)
        self.hatching_width_slider.valueChanged.connect(self.update_hatching_width)
        self.hatching_width_layout.addWidget(self.hatching_width_slider)
        
        self.hatching_width_value = QLabel("3")
        self.hatching_width_layout.addWidget(self.hatching_width_value)
        self.hatching_width = 3  # Store the actual value
        
        # Initially hide the hatching options until enabled
        self.hatching_options.setVisible(False)
        
        # Add color detection section
        self.color_section = QWidget()  # Container widget for the entire section
        self.color_section_layout = QVBoxLayout(self.color_section)
        
        self.color_section_title = QLabel("Color Detection:")
        self.color_section_title.setStyleSheet("font-weight: bold;")
        self.color_section_layout.addWidget(self.color_section_title)
        
        # Multiple wall color selection
        self.wall_colors_layout = QVBoxLayout()
        self.color_section_layout.addLayout(self.wall_colors_layout)
        
        self.wall_colors_label = QLabel("Wall Colors:")
        self.wall_colors_layout.addWidget(self.wall_colors_label)
        
        # List widget to display selected colors
        self.wall_colors_list = QListWidget()
        self.wall_colors_list.setMaximumHeight(100)
        self.wall_colors_list.itemClicked.connect(self.select_color)
        self.wall_colors_list.itemDoubleClicked.connect(self.edit_wall_color)
        self.wall_colors_layout.addWidget(self.wall_colors_list)
        
        # Buttons for color management
        self.color_buttons_layout = QHBoxLayout()
        self.wall_colors_layout.addLayout(self.color_buttons_layout)
        
        self.add_color_button = QPushButton("Add Color")
        self.add_color_button.clicked.connect(self.add_wall_color)
        self.color_buttons_layout.addWidget(self.add_color_button)
        
        self.remove_color_button = QPushButton("Remove Color")
        self.remove_color_button.clicked.connect(self.remove_wall_color)
        self.color_buttons_layout.addWidget(self.remove_color_button)
        
        # Replace the edit threshold button with direct threshold controls
        self.threshold_container = QWidget()
        self.threshold_layout = QVBoxLayout(self.threshold_container)
        self.threshold_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add a header for the threshold section
        self.threshold_header = QLabel("Selected Color Threshold:")
        self.threshold_header.setStyleSheet("font-weight: bold;")
        self.threshold_layout.addWidget(self.threshold_header)
        
        # Create the threshold slider
        self.current_threshold_layout = QHBoxLayout()
        self.threshold_layout.addLayout(self.current_threshold_layout)
        
        self.threshold_label = QLabel("Threshold: 10.0")
        self.current_threshold_layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(300)
        self.threshold_slider.setValue(100)  # Default value 10.0
        self.threshold_slider.valueChanged.connect(self.update_selected_threshold)
        self.current_threshold_layout.addWidget(self.threshold_slider)
        
        # Add a separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.threshold_layout.addWidget(separator)
        
        self.wall_colors_layout.addWidget(self.threshold_container)
        
        # Initially hide the threshold controls until a color is selected
        self.threshold_container.setVisible(False)
        
        # Add the color section to the main controls layout
        self.controls_layout.addWidget(self.color_section)
        
        # Initially hide the entire color section
        self.color_section.setVisible(False)
        
        # Store the currently selected color item
        self.selected_color_item = None

        # Add a checkbox for high-resolution processing
        self.high_res_checkbox = QCheckBox("Process at Full Resolution")
        self.high_res_checkbox.setChecked(False)
        self.high_res_checkbox.setToolTip("Process at full resolution (slower but more accurate)")
        self.high_res_checkbox.stateChanged.connect(self.reload_working_image)
        self.controls_layout.addWidget(self.high_res_checkbox)
        
        # Group edge detection settings
        self.edge_detection_widgets = []
        self.edge_detection_widgets.append(self.sliders["Edge Sensitivity"])
        self.edge_detection_widgets.append(self.sliders["Edge Threshold"])
        
        # State for color detection - now a list of colors
        self.wall_colors = []  # List to store QColor objects
        
        # Add a default black color with default threshold
        self.add_wall_color_to_list(QColor(0, 0, 0), 10.0)
        
        # State
        self.original_image = None  # Original full-size image
        self.current_image = None   # Working image (possibly scaled down)
        self.max_working_dimension = 1500  # Maximum dimension for processing
        self.scale_factor = 1.0     # Scale factor between original and working image
        self.processed_image = None
        self.current_contours = []
        self.display_scale_factor = 1.0
        self.display_offset = (0, 0)
        
        # Additional state for hover highlighting
        self.highlighted_contour_index = -1  # -1 means no contour is highlighted
        self.original_processed_image = None  # Store original image without highlight

        # Additional state for drag selection
        self.selecting = False
        self.selection_start_img = None
        self.selection_current_img = None
        self.selected_contour_indices = []

        # Additional state for color selection
        self.selecting_colors = False
        self.color_selection_start = None
        self.color_selection_current = None

        # Additional state for mask editing


        # Add these new variables to track the brush preview state
        self.brush_preview_active = False
        self.last_preview_image = None
        self.foundry_preview_active = False 

        # --- Presets UI ---
        # Main vertical layout for presets
        presets_main_layout = QVBoxLayout()
        presets_main_layout.setSpacing(2) # Reduce spacing between the two lines

        # First line: Label and ComboBox
        presets_line1_layout = QHBoxLayout()
        presets_line1_layout.addWidget(QLabel("Detection Presets:")) # Stretch 0

        self.detection_preset_combo = QComboBox()
        self.detection_preset_combo.setToolTip("Load a detection settings preset")
        self.detection_preset_combo.currentIndexChanged.connect(self.load_detection_preset_selected)
        presets_line1_layout.addWidget(self.detection_preset_combo, 1) # Stretch 1 to take available space

        # Second line: Buttons
        presets_line2_layout = QHBoxLayout()
        presets_line2_layout.addStretch(1) # Add stretch to push buttons to the right

        save_preset_button = QPushButton("Save Preset")
        save_preset_button.setObjectName("save_preset_button")
        save_preset_button.setToolTip("Save current detection settings as a new preset")
        save_preset_button.clicked.connect(self.save_detection_preset)
        presets_line2_layout.addWidget(save_preset_button) # Stretch 0

        manage_presets_button = QPushButton("Manage")
        manage_presets_button.setObjectName("manage_presets_button")
        manage_presets_button.setToolTip("Manage saved detection presets")
        manage_presets_button.clicked.connect(self.manage_detection_presets)
        presets_line2_layout.addWidget(manage_presets_button) # Stretch 0

        # Add the two lines to the main presets layout
        presets_main_layout.addLayout(presets_line1_layout)
        presets_main_layout.addLayout(presets_line2_layout)

        # Add the main presets layout to the controls layout
        self.controls_layout.addLayout(presets_main_layout)
        # --- Presets UI End ---


        # Add a divider with auto margin on top
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        
        # Add a stretch item above the separator to push it to the bottom
        self.controls_layout.addStretch(1) # Stretch is now AFTER presets
        
        self.controls_layout.addWidget(separator)
        
        # Buttons
        self.buttons_layout = QHBoxLayout()
        self.controls_layout.addLayout(self.buttons_layout)

        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        self.buttons_layout.addWidget(self.open_button)

        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.buttons_layout.addWidget(self.save_button)

        # Add URL image loading button next to Open Image button
        self.url_button = QPushButton("Load from URL")
        self.url_button.clicked.connect(self.load_image_from_url)
        self.url_button.setToolTip("Load image from URL in clipboard")
        self.buttons_layout.addWidget(self.url_button)
        
        # Create a layout for the wall action buttons
        self.wall_actions_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.wall_actions_layout)

        # Move the Bake button to the main sidebar
        self.bake_button = QPushButton("Bake Contours to Mask")
        self.bake_button.clicked.connect(self.bake_contours_to_mask)
        self.wall_actions_layout.addWidget(self.bake_button)
        
        # Add Undo button
        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setEnabled(False)  # Initially disabled until actions are performed
        self.wall_actions_layout.addWidget(self.undo_button)
        
        # Move the Export to Foundry button
        self.export_foundry_button = QPushButton("Export to Foundry VTT")
        self.export_foundry_button.clicked.connect(self.export_to_foundry_vtt)
        self.export_foundry_button.setToolTip("Export walls as JSON for Foundry VTT")
        self.export_foundry_button.setEnabled(False)  # Initially disabled
        self.wall_actions_layout.addWidget(self.export_foundry_button)
        
        # Move the Save Foundry Walls and Cancel Preview buttons
        self.save_foundry_button = QPushButton("Save Foundry Walls")
        self.save_foundry_button.clicked.connect(self.save_foundry_preview)
        self.save_foundry_button.setToolTip("Save the previewed walls to Foundry VTT JSON file")
        self.save_foundry_button.setEnabled(False)
        self.wall_actions_layout.addWidget(self.save_foundry_button)
        
        self.cancel_foundry_button = QPushButton("Cancel Preview")
        self.cancel_foundry_button.clicked.connect(self.cancel_foundry_preview)
        self.cancel_foundry_button.setToolTip("Return to normal view")
        self.cancel_foundry_button.setEnabled(False)
        self.wall_actions_layout.addWidget(self.cancel_foundry_button)
        
        # Add a copy to clipboard button next to Save Foundry Walls
        self.copy_foundry_button = QPushButton("Copy to Clipboard")
        self.copy_foundry_button.clicked.connect(self.copy_foundry_to_clipboard)
        self.copy_foundry_button.setToolTip("Copy the walls JSON to clipboard for Foundry VTT")
        self.copy_foundry_button.setEnabled(False)
        self.wall_actions_layout.addWidget(self.copy_foundry_button)
        
        # Connect drawing tool radio buttons to handler
        self.brush_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.line_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.rectangle_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.circle_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.ellipse_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.fill_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)

        # Add history tracking for undo feature
        self.history = deque(maxlen=5)  # Store up to 5 previous states
        
        # Add keyboard shortcut for undo
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo)

        # Create the update notification widget (initially hidden)
        self.update_notification = QWidget(self)
        self.update_notification.setObjectName("updateNotification")
        self.update_notification.setStyleSheet("""
            #updateNotification {
                background-color: #0c84e4;
                border-radius: 4px;
                padding: 4px;
            }
            QLabel {
                color: white;
                font-weight: bold;
            }
        """)
        
        update_layout = QHBoxLayout(self.update_notification)
        update_layout.setContentsMargins(8, 4, 8, 4)
        
        # Add an icon for the update notification
        update_icon_label = QLabel()
        update_icon = QPixmap(24, 24)
        update_icon.fill(Qt.GlobalColor.transparent)
        painter = QPainter(update_icon)
        painter.setPen(Qt.GlobalColor.white)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(8, 4, 8, 16)
        painter.drawPolygon([QPoint(5, 10), QPoint(12, 2), QPoint(19, 10)])
        painter.end()
        update_icon_label.setPixmap(update_icon)
        update_layout.addWidget(update_icon_label)
        
        # Add text for the update notification
        self.update_text = QLabel("Update Available! Click to download")
        self.update_text.setCursor(Qt.CursorShape.PointingHandCursor)
        update_layout.addWidget(self.update_text)
        
        # Position the notification in the top right corner
        self.update_notification.setGeometry(
            self.width() - 250, 10, 240, 40
        )
        self.update_notification.hide()  # Initially hidden
        
        # Connect the click event to open the download page
        self.update_notification.mousePressEvent = self.open_update_url

        # Preset management
        self.detection_presets = {}
        self.export_presets = {} # Will be used later
        self.load_presets_from_file() # Load presets on startup

        # Populate preset combo box AFTER loading presets
        self.update_detection_preset_combo()

        # Initialize last export settings storage
        self.last_export_settings = None # Will be used later

        # Apply stylesheet at the end
        self.apply_stylesheet()

        # Check for updates
        self.check_for_updates()

    # --- Preset Management ---

    def get_default_detection_presets(self):
        """Returns a dictionary of default detection presets."""
        return {
            "Default": {
                "sliders": {"Smoothing": 3, "Edge Sensitivity": 50, "Edge Threshold": 150, "Edge Margin": 2, "Min Merge Distance": 10, "Min Area": 124},
                "checkboxes": {"High Resolution": False, "Merge Contours": True, "Remove Hatching": False},
                "radios": {"Edge Detection": True, "Color Detection": False, "Min Area Percentage": True, "Min Area Pixels": False},
                "colors": [], # Default: no specific colors
                "hatching": {"color": [0, 0, 0], "threshold": 10.0, "width": 3} # Default hatching settings
            },
            "Fine Detail": {
                "sliders": {
                    "Min Area": 4450,
                    "Smoothing": 3,
                    "Edge Sensitivity": 255,
                    "Edge Threshold": 140,
                    "Edge Margin": 1,
                    "Min Merge Distance": 5
                },
                "checkboxes": {
                    "High Resolution": False,
                    "Merge Contours": False,
                    "Remove Hatching": False
                },
                "radios": {
                    "Edge Detection": True,
                    "Color Detection": False,
                    "Min Area Percentage": False,
                    "Min Area Pixels": True
                },
                "colors": [],
                "hatching": {
                    "color": [
                        0,
                        0,
                        0
                    ],
                    "threshold": 10.0,
                    "width": 3
                }
            },
             "Color Focus (Solid Black)": {
                "sliders": {"Smoothing": 3, "Edge Sensitivity": 50, "Edge Threshold": 150, "Edge Margin": 2, "Min Merge Distance": 10, "Min Area": 500},
                "checkboxes": {"High Resolution": False, "Merge Contours": True, "Remove Hatching": False},
                "radios": {"Edge Detection": False, "Color Detection": True, "Min Area Percentage": True, "Min Area Pixels": False},
                "colors": [{"color": [0, 0, 0], "threshold": 0.0}], # Black color with threshold 0
                "hatching": {"color": [0, 0, 0], "threshold": 10.0, "width": 3}
            },
            "B/W with Hatching": {
                "sliders": {
                    "Min Area": 464,
                    "Smoothing": 5,
                    "Edge Sensitivity": 255,
                    "Edge Threshold": 106,
                    "Edge Margin": 0,
                    "Min Merge Distance": 5
                },
                "checkboxes": {
                    "High Resolution": False,
                    "Merge Contours": False,
                    "Remove Hatching": True
                },
                "radios": {
                    "Edge Detection": False,
                    "Color Detection": True,
                    "Min Area Percentage": False,
                    "Min Area Pixels": True
                },
                "colors": [
                    {
                        "color": [
                            0,
                            0,
                            0
                        ],
                        "threshold": 0.0
                    }
                ],
                "hatching": {
                    "color": [
                        0,
                        0,
                        0
                    ],
                    "threshold": 0.0,
                    "width": 1
                }
            }
        }

    def get_default_export_presets(self):
        return {
            "Default": {
                "simplify_tolerance": 0.0005,
                "max_wall_length": 50,
                "max_walls": 5000,
                "merge_distance": 25.0,
                "angle_tolerance": 1.0,
                "max_gap": 10.0,
                "grid_size": 0,
                "allow_half_grid": False
            },
            "Maze Example": {
                "simplify_tolerance": 0.0,
                "max_wall_length": 50,
                "max_walls": 20000,
                "merge_distance": 25.0,
                "angle_tolerance": 1.0,
                "max_gap": 10.0,
                "grid_size": 72,
                "allow_half_grid": False
            },
            "Large Optimized": {
                "simplify_tolerance": 0.0005,
                "max_wall_length": 200,
                "max_walls": 20000,
                "merge_distance": 100.0,
                "angle_tolerance": 1.0,
                "max_gap": 10.0,
                "grid_size": 0,
                "allow_half_grid": False
            },
            "Large Extra Optimized": {
                "simplify_tolerance": 0.0005,
                "max_wall_length": 200,
                "max_walls": 20000,
                "merge_distance": 200.0,
                "angle_tolerance": 1.0,
                "max_gap": 10.0,
                "grid_size": 0,
                "allow_half_grid": False
            }
        }

    def load_presets_from_file(self):
        """Load detection and export presets from JSON files."""
        # Load Detection Presets
        self.detection_presets = self.get_default_detection_presets() # Start with defaults
        if os.path.exists(DETECTION_PRESETS_FILE):
            try:
                with open(DETECTION_PRESETS_FILE, 'r') as f:
                    user_presets = json.load(f)
                    # Merge user presets, potentially overwriting defaults if names clash
                    self.detection_presets.update(user_presets)
                print(f"Loaded detection presets from {DETECTION_PRESETS_FILE}")
            except json.JSONDecodeError:
                print(f"Error: Could not decode {DETECTION_PRESETS_FILE}. Using defaults.")
            except Exception as e:
                print(f"Error loading detection presets: {e}. Using defaults.")
        else:
            print(f"{DETECTION_PRESETS_FILE} not found. Using default detection presets.")

        # Load Export Presets (Placeholder for later)
        # self.export_presets = self.get_default_export_presets()
        # if os.path.exists(EXPORT_PRESETS_FILE):
        #     try:
        #         with open(EXPORT_PRESETS_FILE, 'r') as f:
        #             user_presets = json.load(f)
        #             self.export_presets.update(user_presets)
        #         print(f"Loaded export presets from {EXPORT_PRESETS_FILE}")
        #     except Exception as e:
        #         print(f"Error loading export presets: {e}")
        # else:
        #     print(f"{EXPORT_PRESETS_FILE} not found. Using default export presets.")

        # Load Export Presets
        self.export_presets = self.get_default_export_presets()
        if os.path.exists(EXPORT_PRESETS_FILE):
            try:
                with open(EXPORT_PRESETS_FILE, 'r') as f:
                    user_presets = json.load(f)
                    self.export_presets.update(user_presets)
            except Exception as e:
                print(f"Error loading export presets: {e}")

    def save_presets_to_file(self):
        """Save user-defined detection and export presets to JSON files."""
        # Save Detection Presets (only non-default ones)
        default_preset_names = self.get_default_detection_presets().keys()
        user_detection_presets = {
            name: preset for name, preset in self.detection_presets.items()
            if name not in default_preset_names
        }
        try:
            with open(DETECTION_PRESETS_FILE, 'w') as f:
                json.dump(user_detection_presets, f, indent=4)
            print(f"Saved user detection presets to {DETECTION_PRESETS_FILE}")
        except Exception as e:
            print(f"Error saving detection presets: {e}")

        # Save Export Presets (Placeholder for later)
        # default_export_names = self.get_default_export_presets().keys()
        # user_export_presets = { ... }
        # try:
        #     with open(EXPORT_PRESETS_FILE, 'w') as f:
        #         json.dump(user_export_presets, f, indent=4)
        #     print(f"Saved user export presets to {EXPORT_PRESETS_FILE}")
        # except Exception as e:
        #     print(f"Error saving export presets: {e}")

        # Save Export Presets
        default_export_names = self.get_default_export_presets().keys()
        user_export_presets = {
            name: preset for name, preset in self.export_presets.items()
            if name not in default_export_names
        }
        try:
            with open(EXPORT_PRESETS_FILE, 'w') as f:
                json.dump(user_export_presets, f, indent=4)
        except Exception as e:
            print(f"Error saving export presets: {e}")

    def update_detection_preset_combo(self):
        """Update the detection preset combo box with current preset names."""
        self.detection_preset_combo.blockSignals(True) # Prevent triggering load while updating
        current_selection = self.detection_preset_combo.currentText()
        self.detection_preset_combo.clear()
        # Add a placeholder item first
        self.detection_preset_combo.addItem("-- Select Preset --")
        # Add sorted preset names
        sorted_names = sorted(self.detection_presets.keys())
        for name in sorted_names:
            self.detection_preset_combo.addItem(name)

        # Try to restore previous selection
        index = self.detection_preset_combo.findText(current_selection)
        if index != -1:
            self.detection_preset_combo.setCurrentIndex(index)
        else:
            self.detection_preset_combo.setCurrentIndex(0) # Select placeholder

        self.detection_preset_combo.blockSignals(False)

    def update_export_preset_combo(self):
        self.export_preset_combo.blockSignals(True)
        self.export_preset_combo.clear()
        self.export_preset_combo.addItem("-- Select Preset --")
        for name in sorted(self.export_presets.keys()):
            self.export_preset_combo.addItem(name)
        self.export_preset_combo.blockSignals(False)

    def get_current_detection_settings(self):
        """Gather current detection settings into a dictionary."""
        settings = {
            "sliders": {},
            "checkboxes": {},
            "radios": {},
            "colors": [],
            "hatching": {}
        }

        # Sliders
        for name, info in self.sliders.items():
            if 'slider' in info:
                settings["sliders"][name] = info['slider'].value()

        # Checkboxes
        settings["checkboxes"]["High Resolution"] = self.high_res_checkbox.isChecked()
        settings["checkboxes"]["Merge Contours"] = self.merge_contours.isChecked()
        settings["checkboxes"]["Remove Hatching"] = self.remove_hatching_checkbox.isChecked()

        # Radio Buttons
        settings["radios"]["Edge Detection"] = self.edge_detection_radio.isChecked()
        settings["radios"]["Color Detection"] = self.color_detection_radio.isChecked()
        settings["radios"]["Min Area Percentage"] = self.min_area_percentage_radio.isChecked()
        settings["radios"]["Min Area Pixels"] = self.min_area_pixels_radio.isChecked()

        # Colors
        for i in range(self.wall_colors_list.count()):
            item = self.wall_colors_list.item(i)
            color_data = item.data(Qt.ItemDataRole.UserRole)
            qcolor = color_data["color"]
            settings["colors"].append({
                "color": [qcolor.red(), qcolor.green(), qcolor.blue()],
                "threshold": color_data["threshold"]
            })

        # Hatching Settings
        settings["hatching"]["color"] = [self.hatching_color.red(), self.hatching_color.green(), self.hatching_color.blue()]
        settings["hatching"]["threshold"] = self.hatching_threshold
        settings["hatching"]["width"] = self.hatching_width

        return settings

    def get_current_export_settings(self):
        """Gather current export settings into a dictionary."""
        settings = {
            "simplify_tolerance": self.simplify_tolerance_input.value(),
            "max_wall_length": self.max_wall_length_input.value(),
            "max_walls": self.max_walls_input.value(),
            "merge_distance": self.merge_distance_input.value(),
            "angle_tolerance": self.angle_tolerance_input.value(),
            "max_gap": self.max_gap_input.value(),
            "grid_size": self.grid_size_input.value(),
            "allow_half_grid": self.allow_half_grid.isChecked()
        }
        return settings

    def apply_detection_settings(self, settings):
        """Apply settings from a dictionary to the UI and internal state."""
        if not settings:
            print("Warning: Attempted to apply empty settings.")
            return

        # Block signals to prevent unwanted updates during application
        for info in self.sliders.values():
            if 'slider' in info: 
                info['slider'].blockSignals(True)
        self.high_res_checkbox.blockSignals(True)
        self.merge_contours.blockSignals(True)
        self.remove_hatching_checkbox.blockSignals(True)
        self.edge_detection_radio.blockSignals(True)
        self.color_detection_radio.blockSignals(True)
        self.min_area_percentage_radio.blockSignals(True)
        self.min_area_pixels_radio.blockSignals(True)
        self.wall_colors_list.blockSignals(True)
        self.hatching_color_button.blockSignals(True)
        self.hatching_threshold_slider.blockSignals(True)
        self.hatching_width_slider.blockSignals(True)

        try:
            # Apply Radio Buttons FIRST (handle mutually exclusive groups)
            # This ensures min_area mode is set before we apply the slider values
            if "radios" in settings:
                # Set detection mode radio buttons
                if "Edge Detection" in settings["radios"]:
                    self.edge_detection_radio.setChecked(settings["radios"]["Edge Detection"])
                if "Color Detection" in settings["radios"]:
                    self.color_detection_radio.setChecked(settings["radios"]["Color Detection"])
                
                # Set min area mode radio buttons
                if "Min Area Percentage" in settings["radios"] and settings["radios"]["Min Area Percentage"]:
                    self.min_area_percentage_radio.setChecked(True)
                    self.using_pixels_mode = False
                elif "Min Area Pixels" in settings["radios"] and settings["radios"]["Min Area Pixels"]:
                    self.min_area_pixels_radio.setChecked(True)
                    self.using_pixels_mode = True

            # Apply Sliders
            if "sliders" in settings:
                for name, value in settings["sliders"].items():
                    if name in self.sliders:
                        # Update slider with the preset value
                        self.sliders[name]['slider'].setValue(value)
                        
                        # Update label with proper format based on mode
                        if name == "Min Area":
                            label = self.sliders[name]['label']
                            if hasattr(self, 'using_pixels_mode') and self.using_pixels_mode:
                                # In pixel mode, show raw value
                                label.setText(f"Min Area: {value} px")
                            elif 'scale' in self.sliders[name]:
                                # In percentage mode, apply scale factor
                                scale = self.sliders[name]['scale']
                                label.setText(f"Min Area: {value * scale:.1f}")

            # Apply Checkboxes
            if "checkboxes" in settings:
                if "High Resolution" in settings["checkboxes"]:
                    self.high_res_checkbox.setChecked(settings["checkboxes"]["High Resolution"])
                if "Merge Contours" in settings["checkboxes"]:
                    self.merge_contours.setChecked(settings["checkboxes"]["Merge Contours"])
                if "Remove Hatching" in settings["checkboxes"]:
                    self.remove_hatching_checkbox.setChecked(settings["checkboxes"]["Remove Hatching"])

            # Apply Colors
            if "colors" in settings:
                # Clear existing colors
                self.wall_colors_list.clear()
                
                # Add all colors from the preset
                for color_data in settings["colors"]:
                    rgb = color_data["color"]
                    qcolor = QColor(rgb[0], rgb[1], rgb[2])
                    threshold = color_data["threshold"]
                    self.add_wall_color_to_list(qcolor, threshold)

            # Apply Hatching Settings
            if "hatching" in settings:
                if "color" in settings["hatching"]:
                    rgb = settings["hatching"]["color"]
                    self.hatching_color = QColor(rgb[0], rgb[1], rgb[2])
                    self.hatching_color_button.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});")
                
                if "threshold" in settings["hatching"]:
                    self.hatching_threshold = settings["hatching"]["threshold"]
                    self.hatching_threshold_slider.setValue(int(self.hatching_threshold * 10))
                    self.hatching_threshold_value.setText(f"{self.hatching_threshold:.1f}")
                
                if "width" in settings["hatching"]:
                    self.hatching_width = settings["hatching"]["width"]
                    self.hatching_width_slider.setValue(self.hatching_width)
                    self.hatching_width_value.setText(str(self.hatching_width))

        finally:
            # Unblock signals
            for info in self.sliders.values():
                if 'slider' in info:
                    info['slider'].blockSignals(False)
            self.high_res_checkbox.blockSignals(False)
            self.merge_contours.blockSignals(False)
            self.remove_hatching_checkbox.blockSignals(False)
            self.edge_detection_radio.blockSignals(False)
            self.color_detection_radio.blockSignals(False)
            self.min_area_percentage_radio.blockSignals(False)
            self.min_area_pixels_radio.blockSignals(False)
            self.wall_colors_list.blockSignals(False)
            self.hatching_color_button.blockSignals(False)
            self.hatching_threshold_slider.blockSignals(False)
            self.hatching_width_slider.blockSignals(False)

        # Now that all settings are applied, explicitly call toggle_detection_mode_radio
        # to ensure the UI reflects the detection mode correctly
        if "radios" in settings:
            self.toggle_detection_mode_radio(self.color_detection_radio.isChecked())

        # Trigger image update after applying settings
        if self.current_image is not None:
            self.update_image()
        self.setStatusTip("Applied detection preset.")

    def save_detection_preset(self):
        """Save the current detection settings as a new preset."""
        preset_name, ok = QInputDialog.getText(self, "Save Detection Preset", "Enter preset name:")
        if ok and preset_name:
            # Check if overwriting a default preset
            if preset_name in self.get_default_detection_presets():
                 reply = QMessageBox.question(self, "Overwrite Default Preset?",
                                             f"'{preset_name}' is a default preset. Overwrite it?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                 if reply == QMessageBox.StandardButton.No:
                     return # User cancelled overwrite

            # Check if overwriting an existing user preset
            elif preset_name in self.detection_presets:
                 reply = QMessageBox.question(self, "Overwrite Preset?",
                                             f"Preset '{preset_name}' already exists. Overwrite?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                 if reply == QMessageBox.StandardButton.No:
                     return # User cancelled overwrite

            current_settings = self.get_current_detection_settings()
            self.detection_presets[preset_name] = current_settings
            self.update_detection_preset_combo()
            # Select the newly saved preset in the combo box
            index = self.detection_preset_combo.findText(preset_name)
            if index != -1:
                self.detection_preset_combo.setCurrentIndex(index)
            self.save_presets_to_file() # Persist changes
            self.setStatusTip(f"Saved detection preset '{preset_name}'.")
        elif ok and not preset_name:
             QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")


    def load_detection_preset_selected(self, index):
        """Load the detection preset selected in the combo box."""
        preset_name = self.detection_preset_combo.itemText(index)
        if index > 0 and preset_name in self.detection_presets: # Index 0 is placeholder
            print(f"Loading detection preset: {preset_name}")
            self.apply_detection_settings(self.detection_presets[preset_name])
            self.setStatusTip(f"Loaded detection preset '{preset_name}'.")
        elif index == 0:
             self.setStatusTip("Select a detection preset to load.")


    def manage_detection_presets(self):
        """Show a dialog or menu to manage (delete) user presets."""
        default_preset_names = set(self.get_default_detection_presets().keys())
        user_preset_names = sorted([name for name in self.detection_presets.keys() if name not in default_preset_names])

        if not user_preset_names:
            QMessageBox.information(self, "Manage Presets", "No user-defined presets to manage.")
            return

        preset_to_delete, ok = QInputDialog.getItem(self, "Delete User Preset",
                                                    "Select preset to delete:", user_preset_names, 0, False)

        if ok and preset_to_delete:
            reply = QMessageBox.question(self, "Confirm Deletion",
                                         f"Are you sure you want to delete the preset '{preset_to_delete}'?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if preset_to_delete in self.detection_presets:
                    del self.detection_presets[preset_to_delete]
                    self.update_detection_preset_combo()
                    self.save_presets_to_file() # Persist deletion
                    self.setStatusTip(f"Deleted preset '{preset_to_delete}'.")
                    print(f"Deleted preset: {preset_to_delete}")
                else:
                     QMessageBox.warning(self, "Error", f"Preset '{preset_to_delete}' not found.")


    def save_export_preset(self):
        if self.current_export_settings is None:
            QMessageBox.warning(self, "Save Export Preset", "No export settings available to save.")
            return

        preset_name, ok = QInputDialog.getText(self, "Save Export Preset", "Enter preset name:")
        if ok and preset_name:
            self.export_presets[preset_name] = copy.deepcopy(self.current_export_settings)
            self.save_presets_to_file()
            self.setStatusTip(f"Saved export preset '{preset_name}'")

    def load_export_preset_selected(self, index):
        preset_name = self.export_preset_combo.itemText(index)
        if index > 0 and preset_name in self.export_presets:
            self.current_export_settings = copy.deepcopy(self.export_presets[preset_name])

    def manage_export_presets(self):
        user_preset_names = [name for name in self.export_presets.keys() if name not in self.get_default_export_presets()]
        if not user_preset_names:
            QMessageBox.information(self, "Manage Export Presets", "No user-defined export presets to manage.")
            return

        preset_to_delete, ok = QInputDialog.getItem(self, "Delete Export Preset", "Select preset to delete:", user_preset_names, 0, False)
        if ok and preset_to_delete:
            del self.export_presets[preset_to_delete]
            self.save_presets_to_file()
            self.setStatusTip(f"Deleted export preset '{preset_to_delete}'")

    def save_export_preset_dialog(self):
        """Open a dialog to save current export settings as a preset."""
        if self.current_export_settings is None:
            # Create default export settings if none exist
            self.current_export_settings = {
                "simplify_tolerance": 0.0005,
                "max_wall_length": 50,
                "max_walls": 5000,
                "merge_distance": 25.0,
                "angle_tolerance": 1.0,
                "max_gap": 10.0,
                "grid_size": 0,
                "allow_half_grid": False
            }

        preset_name, ok = QInputDialog.getText(self, "Save Export Preset", "Enter preset name:")
        if ok and preset_name:
            # Check if overwriting a default preset
            if preset_name in self.get_default_export_presets():
                reply = QMessageBox.question(self, "Overwrite Default Preset?",
                                            f"'{preset_name}' is a default preset. Overwrite it?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return  # User cancelled overwrite

            # Check if overwriting an existing user preset
            elif preset_name in self.export_presets:
                reply = QMessageBox.question(self, "Overwrite Preset?",
                                            f"Preset '{preset_name}' already exists. Overwrite?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return  # User cancelled overwrite

            # Save the preset
            self.export_presets[preset_name] = copy.deepcopy(self.current_export_settings)
            self.update_export_preset_combo()
            self.save_presets_to_file()
            self.setStatusTip(f"Saved export preset '{preset_name}'")
        elif ok and not preset_name:
            QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")

    # ... existing methods ...

    # Modify update_image to use self.using_pixels_mode correctly
    def update_image(self):
        """Update the displayed image based on the current settings."""
        if self.current_image is None:
            return

        # Get slider values
        blur = self.sliders["Smoothing"]['slider'].value()
        
        # Handle special case for blur=1 (no blur) and ensure odd values
        if blur > 1 and blur % 2 == 0:
            blur += 1
        
        canny1 = self.sliders["Edge Sensitivity"]['slider'].value()
        canny2 = self.sliders["Edge Threshold"]['slider'].value()
        edge_margin = self.sliders["Edge Margin"]['slider'].value()
        
        # Get min_merge_distance as a float value
        min_merge_distance = self.sliders["Min Merge Distance"]['slider'].value() * 0.1
        
        # Calculate min area based on mode (percentage or pixels)
        min_area_value = self.sliders["Min Area"]['slider'].value()
        image_area = self.current_image.shape[0] * self.current_image.shape[1]
        
        if hasattr(self, 'using_pixels_mode') and self.using_pixels_mode:
            # Min area is in pixels (1 to 1000)
            min_area = min_area_value
            working_min_area = min_area
            # For display/logging purposes
            min_area_percentage = (min_area / image_area) * 100.0
        else:
            # Min area is a percentage (0.0001% to 1%)
            min_area_percentage = min_area_value * 0.001
            min_area = int(image_area * min_area_percentage / 100.0)
            working_min_area = min_area
        
        # If we're working with a scaled image, the min area needs to be scaled too
        if self.scale_factor != 1.0 and not (hasattr(self, 'using_pixels_mode') and self.using_pixels_mode):
            # Only scale min_area if we're using percentages
            working_min_area = int(min_area * self.scale_factor * self.scale_factor)
        elif self.scale_factor != 1.0:
            # If using pixels mode, scale the pixels to the working image size
            working_min_area = int(min_area * self.scale_factor * self.scale_factor)
        
        # Working image that we'll pass to the detection function
        processed_image = self.current_image.copy()
        
        # Apply hatching removal if enabled
        if self.remove_hatching_checkbox.isChecked():
            # Convert QColor to BGR tuple for OpenCV
            hatching_color_bgr = (
                self.hatching_color.blue(),
                self.hatching_color.green(),
                self.hatching_color.red()
            )
            print(f"Removing hatching lines: Color={hatching_color_bgr}, Threshold={self.hatching_threshold:.1f}, Width={self.hatching_width}")
            
            # Apply the hatching removal
            from src.wall_detection.detector import remove_hatching_lines
            processed_image = remove_hatching_lines(
                processed_image, 
                hatching_color_bgr, 
                self.hatching_threshold, 
                self.hatching_width
            )
        
        # Set up color detection parameters with per-color thresholds
        wall_colors_with_thresholds = None
        default_threshold = 0
        
        if self.color_detection_radio.isChecked() and self.wall_colors_list.count() > 0:
            # Extract all colors and thresholds from the list widget
            wall_colors_with_thresholds = []
            for i in range(self.wall_colors_list.count()):
                item = self.wall_colors_list.item(i)
                color_data = item.data(Qt.ItemDataRole.UserRole)
                color = color_data["color"]
                threshold = color_data["threshold"]
                
                # Convert Qt QColor to OpenCV BGR color and pair with threshold
                bgr_color = (
                    color.blue(),
                    color.green(),
                    color.red()
                )
                wall_colors_with_thresholds.append((bgr_color, threshold))
            
            print(f"Using {len(wall_colors_with_thresholds)} colors for detection with individual thresholds")
        
        # Debug output of parameters
        if hasattr(self, 'using_pixels_mode') and self.using_pixels_mode:
            print(f"Parameters: min_area={min_area} pixels (working: {working_min_area}), "
                  f"blur={blur}, canny1={canny1}, canny2={canny2}, edge_margin={edge_margin}")
        else:
            print(f"Parameters: min_area={min_area} (working: {working_min_area}, {min_area_percentage:.4f}% of image), "
                  f"blur={blur}, canny1={canny1}, canny2={canny2}, edge_margin={edge_margin}")

        # Process the image directly with detect_walls
        contours = detect_walls(
            processed_image,
            min_contour_area=working_min_area,
            max_contour_area=None,
            blur_kernel_size=blur,
            canny_threshold1=canny1,
            canny_threshold2=canny2,
            edge_margin=edge_margin,
            wall_colors=wall_colors_with_thresholds,
            color_threshold=default_threshold
        )
        
        print(f"Detected {len(contours)} contours before merging")

        # Merge before Min Area if specified
        if self.merge_contours.isChecked():
            contours = merge_contours(
                processed_image, 
                contours, 
                min_merge_distance=min_merge_distance
            )
            print(f"After merge before min area: {len(contours)} contours")
        
        # Filter contours by area BEFORE splitting edges
        contours = [c for c in contours if cv2.contourArea(c) >= working_min_area]
        print(f"After min area filter: {len(contours)} contours")

        # Split contours that touch image edges AFTER area filtering, but only if not in color detection mode
        if not self.color_detection_radio.isChecked():
            split_contours = split_edge_contours(processed_image, contours)

            # Use a much lower threshold for split contours to keep them all
            # Use absolute minimum value instead of relative to min_area
            min_split_area = 5.0 * (self.scale_factor * self.scale_factor)  # Scale with image
            filtered_contours = []
            
            # Keep track of how many contours were kept vs filtered
            kept_count = 0
            filtered_count = 0
            
            for contour in split_contours:
                area = cv2.contourArea(contour)
                if area >= min_split_area:
                    filtered_contours.append(contour)
                    kept_count += 1
                else:
                    filtered_count += 1
            
            contours = filtered_contours
            print(f"After edge splitting: kept {kept_count}, filtered {filtered_count} tiny fragments")

        # Save the current contours for interactive editing
        self.current_contours = contours

        # Ensure contours are not empty
        if not contours:
            print("No contours found after processing.")
            self.processed_image = processed_image
        else:
            # Draw merged contours
            self.processed_image = draw_walls(processed_image, contours)

        # Save the original image for highlighting
        if self.processed_image is not None:
            self.original_processed_image = self.processed_image.copy()
            
        # Clear any existing selection when re-detecting
        self.clear_selection()
        
        # Reset highlighted contour when re-detecting
        self.highlighted_contour_index = -1

        # Convert to QPixmap and display
        self.display_image(self.processed_image)

    # Modify toggle_min_area_mode to update self.using_pixels_mode reliably
    def toggle_min_area_mode(self):
        """Toggle between percentage and pixel mode for Min Area."""
        min_area_slider = self.sliders["Min Area"]['slider']
        min_area_label = self.sliders["Min Area"]['label']
        label_text = "Min Area"

        if self.min_area_percentage_radio.isChecked():
            # Switch to percentage mode (0.001% to 25%)
            # Slider range 1 to 25000 represents this
            min_area_slider.setMinimum(1)
            min_area_slider.setMaximum(25000) # Represents 25% with scale 0.001

            # If coming from pixels mode, try to convert value
            if hasattr(self, 'using_pixels_mode') and self.using_pixels_mode and self.current_image is not None:
                current_pixel_value = min_area_slider.value()
                image_area = self.current_image.shape[0] * self.current_image.shape[1]
                if image_area > 0:
                    percentage = (current_pixel_value / image_area) * 100.0
                    slider_value = max(1, min(25000, int(percentage / 0.001))) # Convert back to slider scale
                    min_area_slider.setValue(slider_value)

            self.using_pixels_mode = False
            self.update_slider(min_area_label, label_text, min_area_slider.value(), 0.001)
            print("Switched Min Area mode to Percentage")

        else: # Pixels mode is checked
            # Switch to pixels mode (1 to 1000 pixels)
            min_area_slider.setMinimum(1)
            min_area_slider.setMaximum(1000)

            # If coming from percentage mode, try to convert value
            if (not hasattr(self, 'using_pixels_mode') or not self.using_pixels_mode) and self.current_image is not None:
                 current_slider_value = min_area_slider.value()
                 percentage = current_slider_value * 0.001
                 image_area = self.current_image.shape[0] * self.current_image.shape[1]
                 if image_area > 0:
                     pixel_value = max(1, min(1000, int((percentage / 100.0) * image_area)))
                     min_area_slider.setValue(pixel_value)
                 else:
                      # If no image, set to a reasonable default pixel value if converting
                      min_area_slider.setValue(min(1000, max(1, 50))) # e.g., 50 pixels
            elif self.current_image is None:
                 # If no image and already in pixels mode (or first time), ensure value is in range
                 min_area_slider.setValue(min(1000, max(1, min_area_slider.value())))


            self.using_pixels_mode = True
            self.update_image()

    # app
    def reload_working_image(self):
        """Reload the working image when resolution setting changes."""
        if self.original_image is None:
            return
        
        # Recreate the working image with the current checkbox state
        self.current_image, self.scale_factor = self.create_working_image(self.original_image)
        print(f"Resolution changed: Working size {self.current_image.shape}, Scale factor {self.scale_factor}")
        
        # Update the image with new resolution
        self.update_image()

    # app
    def add_slider(self, label, min_val, max_val, initial_val, step=1, scale_factor=None):
        """Add a slider with a label."""
        # Create a container widget to hold the slider and label
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        
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
        
        # Add the container to the controls layout
        self.controls_layout.addWidget(slider_container)

        # Store both the slider, its container, AND the label in the dictionary
        self.sliders[label] = {'slider': slider, 'container': slider_container, 'label': slider_label}
        
        # Store scale factor if provided (moved after self.sliders assignment)
        if scale_factor:
            self.sliders[label]['scale'] = scale_factor

    # app
    def update_slider(self, label, label_text, value, scale_factor=None):
        """Update the slider label."""
        # Special case for Min Area slider in pixel mode
        if label_text == "Min Area" and hasattr(self, 'using_pixels_mode') and self.using_pixels_mode:
            label.setText(f"{label_text}: {value} px")
        elif scale_factor:
            display_value = value * scale_factor
            label.setText(f"{label_text}: {display_value:.3f}%")
        else:
            label.setText(f"{label_text}: {value}")

    # app
    def toggle_mode(self):
        """Toggle between detection, deletion, color selection, edit mask, and thinning modes."""
        # Check if we need to save state before mode changes
        if self.processed_image is not None:
            previous_mode = None
            if hasattr(self, 'edit_mask_mode_enabled') and self.edit_mask_mode_enabled:
                previous_mode = 'mask'
            
            # Update mode flags based on radio button states
            self.color_selection_mode_enabled = self.color_selection_mode_radio.isChecked()
            self.deletion_mode_enabled = self.deletion_mode_radio.isChecked()
            self.edit_mask_mode_enabled = self.edit_mask_mode_radio.isChecked()
            self.thin_mode_enabled = self.thin_mode_radio.isChecked()
            
            # If switching to/from mask mode, save state
            current_mode = 'mask' if self.edit_mask_mode_enabled else 'contour'
            if previous_mode != current_mode:
                self.save_state()
        else:
            # Just update the mode flags as before
            self.color_selection_mode_enabled = self.color_selection_mode_radio.isChecked()
            self.deletion_mode_enabled = self.deletion_mode_radio.isChecked()
            self.edit_mask_mode_enabled = self.edit_mask_mode_radio.isChecked()
            self.thin_mode_enabled = self.thin_mode_radio.isChecked()
        
        # Show/hide color selection options
        self.color_selection_options.setVisible(self.color_selection_mode_enabled)
        
        # Show/hide mask edit options
        self.mask_edit_options.setVisible(self.edit_mask_mode_enabled)
        
        # Show/hide thinning options
        self.thin_options.setVisible(self.thin_mode_enabled)
        
        if self.deletion_mode_enabled:
            self.setStatusTip("Deletion Mode: Click inside contours or on lines to delete them")
            # Store original image for highlighting
            if self.processed_image is not None:
                self.original_processed_image = self.processed_image.copy()
        elif self.color_selection_mode_enabled:
            self.setStatusTip("Color Selection Mode: Drag to select an area for color extraction")
            # Store original image for selection rectangle
            if self.processed_image is not None:
                self.original_processed_image = self.processed_image.copy()
        elif self.edit_mask_mode_enabled:
            self.setStatusTip("Edit Mask Mode: Draw or erase on the mask layer")
            # Make sure we have a mask to edit
            if self.mask_layer is None and self.current_image is not None:
                # Create an empty mask if none exists
                self.create_empty_mask()
            if self.processed_image is not None:
                self.original_processed_image = self.processed_image.copy()
                # Display the mask with the image
                self.update_display_with_mask()
                
            # Reset brush preview state when entering edit mode
            self.brush_preview_active = False
            
            # Get current cursor position to start showing brush immediately
            if self.current_image is not None:
                cursor_pos = self.image_label.mapFromGlobal(QCursor.pos())
                if self.image_label.rect().contains(cursor_pos):
                    self.drawing_tools.update_brush_preview(cursor_pos.x(), cursor_pos.y())
        elif self.thin_mode_enabled:
            self.setStatusTip("Thinning Mode: Click on contours to thin them")
            # Store original image for highlighting
            if self.processed_image is not None:
                self.original_processed_image = self.processed_image.copy()
        else:
            self.setStatusTip("")
            # Clear any highlighting
            self.clear_hover()
            # Display normal image without mask
            if self.processed_image is not None:
                self.display_image(self.processed_image)
        
        # Clear any selection when switching modes
        self.clear_selection()
        
        # Make sure to initialize the drawing position attribute
        if hasattr(self, 'last_drawing_position'):
            self.last_drawing_position = None
        else:
            setattr(self, 'last_drawing_position', None)

    # drawing


    # drawing


    # drawing

    # app
    def create_empty_mask(self):
        """Create an empty transparent mask layer."""
        if self.current_image is None:
            return
            
        height, width = self.current_image.shape[:2]
        # Create a transparent mask (4th channel is alpha, all 0 = fully transparent)
        self.mask_layer = np.zeros((height, width, 4), dtype=np.uint8)

    # app
    def bake_contours_to_mask(self):
        """Bake the current contours to the mask layer."""
        if self.current_image is None or not self.current_contours:
            return
        
        # Save state before modifying
        self.save_state()
            
        # Create the mask from contours
        self.mask_layer = create_mask_from_contours(
            self.current_image.shape, 
            self.current_contours,
            color=(0, 255, 0, 255)  # Green
        )
        
        # Switch to mask editing mode
        self.edit_mask_mode_radio.setVisible(True)
        self.edit_mask_mode_radio.setChecked(True)
        
        # Enable the Export to Foundry VTT button
        self.export_foundry_button.setEnabled(True)
        
        # Update display
        self.update_display_with_mask()

    # app
    def update_display_with_mask(self):
        """Update the display to show the image with the mask overlay."""
        if self.current_image is None or self.mask_layer is None:
            return
        
        # Blend the image with the mask
        display_image = blend_image_with_mask(self.current_image, self.mask_layer)
        
        # Display the blended image
        self.display_image(display_image)
        
        # Store this as the baseline image for brush preview
        self.last_preview_image = display_image.copy()
        
        # Important: Also update the processed_image
        self.processed_image = display_image.copy()

    # app
    def toggle_detection_mode_radio(self, checked): # 'checked' parameter is from the signal, might not reflect the final state if called manually
        """Toggle controls visibility based on the currently selected detection mode radio button."""
        # Always check the current state of the radio buttons, ignore the 'checked' parameter from the signal

        if self.color_detection_radio.isChecked():
            # Color Detection Mode is active
            self.edge_detection_radio.setChecked(False) # Ensure consistency

            # Hide edge detection controls and their labels
            sliders_to_hide = ["Smoothing", "Edge Sensitivity", "Edge Threshold", "Edge Margin"]
            for slider_name, slider_info in self.sliders.items():
                # Hide the entire container which includes both slider and label
                if 'container' in slider_info and slider_name in sliders_to_hide:
                    slider_info['container'].setVisible(False)

            # Show color detection controls
            self.color_section.setVisible(True)
            self.color_selection_mode_radio.setVisible(True) # Show color pick tool

            # Update labels to reflect active/inactive state (optional)
            # self.color_section_title.setText("Color Detection:")
            # self.color_section_title.setStyleSheet("font-weight: bold;")
        else: # Edge Detection Mode is active (self.edge_detection_radio should be checked)
            self.color_detection_radio.setChecked(False) # Ensure consistency
            if not self.edge_detection_radio.isChecked(): # Double check and force if needed
                 self.edge_detection_radio.setChecked(True)

            # Show edge detection controls and their labels
            sliders_to_show = ["Smoothing", "Edge Sensitivity", "Edge Threshold", "Edge Margin"]
            for slider_name, slider_info in self.sliders.items():
                # Show the entire container which includes both slider and label
                if 'container' in slider_info and slider_name in sliders_to_show:
                    slider_info['container'].setVisible(True)

            # Hide color detection controls
            self.color_section.setVisible(False)
            self.color_selection_mode_radio.setVisible(False) # Hide color pick tool


        # Update the detection if an image is loaded
        if self.current_image is not None:
            self.update_image()


    # app
    def clear_selection(self):
        """Clear the current selection."""
        # Also check if we're in Foundry preview mode
        if self.foundry_preview_active and self.foundry_walls_preview:
            # If in preview mode, redraw the preview instead
            self.display_foundry_preview()
            return
        
        # Original code for normal mode
        self.selecting = False
        self.selection_start_img = None
        self.selection_current_img = None
        self.selected_contour_indices = []
        
        self.selecting_colors = False
        self.color_selection_start = None
        self.color_selection_current = None
        
        # Redraw without selection rectangle
        if self.processed_image is not None and self.original_processed_image is not None:
            self.processed_image = self.original_processed_image.copy()
            self.display_image(self.processed_image)

    # app
    def start_selection(self, x, y):
        """Start a selection rectangle at the given coordinates."""
        # Convert to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        
        if img_x is None or img_y is None:
            return
            
        if self.deletion_mode_enabled:
            # Check if click is on a contour edge
            min_distance = float('inf')
            found_contour_index = -1
            
            for i, contour in enumerate(self.current_contours):
                contour_points = contour.reshape(-1, 2)
                
                for j in range(len(contour_points)):
                    p1 = contour_points[j]
                    p2 = contour_points[(j + 1) % len(contour_points)]
                    distance = self.point_to_line_distance(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                    
                    # If point is close enough to a line segment
                    if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                        min_distance = distance
                        found_contour_index = i
            
            # If click is on a contour edge, handle as single click
            if found_contour_index != -1:
                self.handle_deletion_click(x, y)
                return
                
            # Otherwise, start a selection
            self.selecting = True
            self.selection_start_img = (img_x, img_y)
            self.selection_current_img = (img_x, img_y)
            self.selected_contour_indices = []
            
        elif self.color_selection_mode_enabled:
            # Start color selection rectangle
            self.selecting_colors = True
            self.color_selection_start = (img_x, img_y)
            self.color_selection_current = (img_x, img_y)
        elif self.thin_mode_enabled:
            # Check if click is on a contour edge
            min_distance = float('inf')
            found_contour_index = -1
            
            for i, contour in enumerate(self.current_contours):
                contour_points = contour.reshape(-1, 2)
                
                for j in range(len(contour_points)):
                    p1 = contour_points[j]
                    p2 = contour_points[(j + 1) % len(contour_points)]
                    distance = self.point_to_line_distance(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                    
                    # If point is close enough to a line segment
                    if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                        min_distance = distance
                        found_contour_index = i
            
            # If click is on a contour edge, handle as single click
            if found_contour_index != -1:
                self.handle_thinning_click(x, y)
                return
                
            # Otherwise, start a selection for thinning multiple contours
            self.selecting = True
            self.selection_start_img = (img_x, img_y)
            self.selection_current_img = (img_x, img_y)
            self.selected_contour_indices = []

    # app
    def update_selection(self, x, y):
        """Update the current selection rectangle to the given coordinates."""
        # Convert to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        
        if img_x is None or img_y is None:
            return
            
        if self.deletion_mode_enabled and self.selecting:
            self.selection_current_img = (img_x, img_y)
            self.update_selection_display()
            
        elif self.color_selection_mode_enabled and self.selecting_colors:
            self.color_selection_current = (img_x, img_y)
            self.update_color_selection_display()
        elif self.thin_mode_enabled and self.selecting:
            self.selection_current_img = (img_x, img_y)
            self.update_selection_display()

    # app
    def update_selection_display(self):
        """Update the display with the selection rectangle and highlighted contours."""
        if not self.selecting or self.original_processed_image is None:
            return
            
        # Start with the original image
        self.processed_image = self.original_processed_image.copy()
        
        # Calculate selection rectangle
        x1 = min(self.selection_start_img[0], self.selection_current_img[0])
        y1 = min(self.selection_start_img[1], self.selection_current_img[1])
        x2 = max(self.selection_start_img[0], self.selection_current_img[0])
        y2 = max(self.selection_start_img[1], self.selection_current_img[1])
        
        # Draw semi-transparent selection rectangle
        overlay = self.processed_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 100, 200), 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 100, 200), -1)
        cv2.addWeighted(overlay, 0.3, self.processed_image, 0.7, 0, self.processed_image)
        
        # Find and highlight contours within the selection - only using edge detection
        self.selected_contour_indices = []
        
        for i, contour in enumerate(self.current_contours):
            contour_points = contour.reshape(-1, 2)
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                
                # Check if any part of this line segment is in the selection rectangle
                # First check if either endpoint is in the rectangle
                if ((x1 <= p1[0] <= x2 and y1 <= p1[1] <= y2) or 
                    (x1 <= p2[0] <= x2 and y1 <= p2[1] <= y2)):
                    self.selected_contour_indices.append(i)
                    # Highlight with different colors based on mode
                    highlight_color = (0, 0, 255) if self.deletion_mode_enabled else (255, 0, 255)  # Red for delete, Magenta for thin
                    cv2.drawContours(self.processed_image, [contour], 0, highlight_color, 2)
                    break
                
                # If neither endpoint is in the rectangle, check if the line intersects the rectangle
                # by checking against all four edges of the rectangle
                rect_edges = [
                    ((x1, y1), (x2, y1)),  # Top edge
                    ((x2, y1), (x2, y2)),  # Right edge
                    ((x2, y2), (x1, y2)),  # Bottom edge
                    ((x1, y2), (x1, y1))   # Left edge
                ]
                
                for rect_p1, rect_p2 in rect_edges:
                    if self.line_segments_intersect(p1[0], p1[1], p2[0], p2[1], 
                                                  rect_p1[0], rect_p1[1], rect_p2[0], rect_p2[1]):
                        self.selected_contour_indices.append(i)
                        # Highlight with different colors based on mode
                        highlight_color = (0, 0, 255) if self.deletion_mode_enabled else (255, 0, 255)
                        cv2.drawContours(self.processed_image, [contour], 0, highlight_color, 2)
                        break
                
                # If we've already added this contour, no need to check more line segments
                if i in self.selected_contour_indices:
                    break
                    
        # Display the updated image
        self.display_image(self.processed_image)

    # color
    def update_color_selection_display(self):
        """Update the display with the color selection rectangle."""
        if not self.selecting_colors or self.original_processed_image is None:
            return
            
        # Start with the original image
        self.processed_image = self.original_processed_image.copy()
        
        # Calculate selection rectangle
        x1 = min(self.color_selection_start[0], self.color_selection_current[0])
        y1 = min(self.color_selection_start[1], self.color_selection_current[1])
        x2 = max(self.color_selection_start[0], self.color_selection_current[0])
        y2 = max(self.color_selection_start[1], self.color_selection_current[1])
        
        # Draw semi-transparent selection rectangle
        overlay = self.processed_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), -1)
        cv2.addWeighted(overlay, 0.3, self.processed_image, 0.7, 0, self.processed_image)
                    
        # Display the updated image
        self.display_image(self.processed_image)

    # app
    def end_selection(self, x, y):
        """Complete the selection and process it according to the current mode."""
        # Convert to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        
        if img_x is None or img_y is None:
            self.clear_selection()
            return
            
        if self.deletion_mode_enabled and self.selecting:
            self.selection_current_img = (img_x, img_y)
            
            # Calculate selection rectangle
            x1 = min(self.selection_start_img[0], self.selection_current_img[0])
            y1 = min(self.selection_start_img[1], self.selection_current_img[1])
            x2 = max(self.selection_start_img[0], self.selection_current_img[0])
            y2 = max(self.selection_start_img[1], self.selection_current_img[1])
            
            # Find contours within the selection
            self.selected_contour_indices = []
            
            for i, contour in enumerate(self.current_contours):
                # Check if contour is at least partially within selection rectangle
                for point in contour:
                    px, py = point[0]
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        self.selected_contour_indices.append(i)
                        break
            
            # If we have selected contours, delete them immediately
            if self.selected_contour_indices:
                self.delete_selected_contours()
            else:
                # If no contours were selected, just clear the selection
                self.clear_selection()
                
        elif self.color_selection_mode_enabled and self.selecting_colors:
            self.color_selection_current = (img_x, img_y)
            
            # Calculate selection rectangle
            x1 = min(self.color_selection_start[0], self.color_selection_current[0])
            y1 = min(self.color_selection_start[1], self.color_selection_current[1])
            x2 = max(self.color_selection_start[0], self.color_selection_current[0])
            y2 = max(self.color_selection_start[1], self.color_selection_current[1])
            
            # Make sure we have a valid selection area
            if x1 < x2 and y1 < y2 and x2 - x1 > 5 and y2 - y1 > 5:
                # Extract colors from the selected area
                self.extract_colors_from_selection(x1, y1, x2, y2)
            else:
                print("Selection area too small")
            
            # Clear the selection
            self.clear_selection()
        elif self.thin_mode_enabled and self.selecting:
            self.selection_current_img = (img_x, img_y)
            
            # Calculate selection rectangle
            x1 = min(self.selection_start_img[0], self.selection_current_img[0])
            y1 = min(self.selection_start_img[1], self.selection_current_img[1])
            x2 = max(self.selection_start_img[0], self.selection_current_img[0])
            y2 = max(self.selection_start_img[1], self.selection_current_img[1])
            
            # Find contours within the selection
            self.selected_contour_indices = []
            
            for i, contour in enumerate(self.current_contours):
                # Check if contour is at least partially within selection rectangle
                for point in contour:
                    px, py = point[0]
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        self.selected_contour_indices.append(i)
                        break
            
            # If we have selected contours, thin them
            if self.selected_contour_indices:
                self.thin_selected_contours()
            else:
                # If no contours were selected, just clear the selection
                self.clear_selection()

    # color
    def extract_colors_from_selection(self, x1, y1, x2, y2):
        """Extract dominant colors from the selected region."""
        if self.current_image is None:
            return
            
        # Extract the selected region from the image
        region = self.current_image[y1:y2, x1:x2]
        
        if region.size == 0:
            print("Selected region is empty")
            return
            
        # Reshape the region for clustering
        pixels = region.reshape(-1, 3)
        
        # Get the number of colors to extract
        num_colors = self.color_count_spinner.value()
        
        # Use K-means clustering to find the dominant colors
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors (cluster centers)
        colors = kmeans.cluster_centers_
        
        # Add each color to the color list
        for color in colors:
            bgr_color = color.astype(int)
            qt_color = QColor(bgr_color[2], bgr_color[1], bgr_color[0])  # Convert BGR to RGB
            
            # Add the color with a threshold of 0 (exact match) initially
            item = self.add_wall_color_to_list(qt_color, 0)
            
            # Select the new color
            self.wall_colors_list.setCurrentItem(item)
            self.select_color(item)
        
        print(f"Extracted {num_colors} colors from selected region")
        
        # Update the image with the new colors
        self.update_image()
    
    # delete
    def delete_selected_contours(self):
        """Delete the selected contours from the current image."""
        if not self.selected_contour_indices:
            return
        
        # Save state before modifying
        self.save_state()
        
        # Delete selected contours
        for index in sorted(self.selected_contour_indices, reverse=True):
            if 0 <= index < len(self.current_contours):
                print(f"Deleting contour {index} (area: {cv2.contourArea(self.current_contours[index])})")
                self.current_contours.pop(index)
        
        # Clear selection and update display
        self.clear_selection()
        self.update_display_from_contours()

    # app
    def handle_hover(self, x, y):
        """Handle mouse hover events for highlighting contours."""
        if not self.current_contours or self.current_image is None:
            return
            
        # Convert display coordinates to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        
        # Check if coordinates are valid
        if img_x is None or img_y is None:
            self.clear_hover()
            return
            
        # Find the contour under the cursor - only check edges
        found_index = -1
        min_distance = float('inf')
        
        # Check if cursor is on a contour edge
        for i, contour in enumerate(self.current_contours):
            contour_points = contour.reshape(-1, 2)
            
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                distance = self.point_to_line_distance(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                
                # If point is close enough to a line segment and closer than any previous match
                if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                    min_distance = distance
                    found_index = i
        
        # Update highlight if needed
        if found_index != self.highlighted_contour_index:
            self.highlighted_contour_index = found_index
            self.update_highlight()

    # app
    def clear_hover(self):
        """Clear any contour highlighting."""
        if self.highlighted_contour_index != -1:
            self.highlighted_contour_index = -1
            self.update_highlight()

    # app
    def update_highlight(self):
        """Update the display with highlighted contour."""
        if self.original_processed_image is None:
            return
            
        # Start with the original image (without highlights)
        self.processed_image = self.original_processed_image.copy()
        
        # If a contour is highlighted, draw it with a different color/thickness
        if self.highlighted_contour_index != -1 and self.highlighted_contour_index < len(self.current_contours):
            # Use different colors based on the current mode
            if self.deletion_mode_enabled:
                highlight_color = (0, 0, 255)  # Red for delete
            elif self.thin_mode_enabled:
                highlight_color = (255, 0, 255)  # Magenta for thin
            else:
                highlight_color = (0, 0, 255)  # Default: red
                
            highlight_thickness = 3
            cv2.drawContours(
                self.processed_image, 
                [self.current_contours[self.highlighted_contour_index]], 
                0, highlight_color, highlight_thickness
            )
            
        # Update the display
        self.display_image(self.processed_image)

    # util
    def convert_to_image_coordinates(self, display_x, display_y):
        """Convert display coordinates to image coordinates."""
        if self.current_image is None:
            return None, None
            
        img_height, img_width = self.current_image.shape[:2]
        display_width = self.image_label.width()
        display_height = self.image_label.height()
        
        # Calculate scaling and offset (centered in the label)
        scale_x = display_width / img_width
        scale_y = display_height / img_height
        scale = min(scale_x, scale_y)
        
        offset_x = (display_width - img_width * scale) / 2
        offset_y = (display_height - img_height * scale) / 2
        
        # Convert to image coordinates
        img_x = int((display_x - offset_x) / scale)
        img_y = int((display_y - offset_y) / scale)
        
        # Check if click is out of bounds
        if img_x < 0 or img_x >= img_width or img_y < 0 or img_y >= img_height:
            return None, None
            
        return img_x, img_y

    # delete
    def handle_deletion_click(self, x, y):
        """Handle clicks for deletion mode."""
        if not self.current_contours or self.current_image is None:
            return
            
        # Convert display coordinates to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        
        # Check if coordinates are valid
        if img_x is None or img_y is None:
            return
            
        # Clear any existing selection when handling a single click
        self.clear_selection()
        
        # Save state before deleting
        self.save_state()
        
        # Use the highlighted contour if available
        if self.highlighted_contour_index != -1:
            print(f"Deleting highlighted contour {self.highlighted_contour_index}")
            self.current_contours.pop(self.highlighted_contour_index)
            self.highlighted_contour_index = -1  # Reset highlight
            self.update_display_from_contours()
            return
        
        # Find contours where the click is on or near an edge
        min_distance = float('inf')
        closest_contour_index = -1
        
        # Check if click is on or near a contour edge
        for i, contour in enumerate(self.current_contours):
            contour_points = contour.reshape(-1, 2)
            
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                distance = self.point_to_line_distance(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                
                # If point is close enough to a line segment
                if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                    min_distance = distance
                    closest_contour_index = i
        
        # If click is on or near an edge, delete that contour
        if closest_contour_index != -1:
            print(f"Deleting contour {closest_contour_index} (edge clicked)")
            self.current_contours.pop(closest_contour_index)
            self.update_display_from_contours()
            return

    # thinning
    def thin_selected_contour(self, contour):
        """Thin a single contour using morphological thinning."""
        # Create a mask for the contour
        mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply the thinning operation using the imported function
        # Pass the current target width and max iterations settings
        thinned_contour = thin_contour(mask, target_width=self.target_width, max_iterations=self.max_iterations)
        
        # No need to extract contours, thin_contour() already returns a contour object
        if thinned_contour is not None:
            return thinned_contour
        else:
            # If thinning failed, return the original contour
            return contour

    # thinning
    def thin_selected_contours(self):
        """Thin the selected contours."""
        if not self.selected_contour_indices:
            return
        
        # Save state before modifying
        self.save_state()
        
        # Thin each selected contour
        for idx in sorted(self.selected_contour_indices):
            if 0 <= idx < len(self.current_contours):
                # Get the contour
                contour = self.current_contours[idx]
                # Apply thinning
                thinned_contour = self.thin_selected_contour(contour)
                # Replace the original with the thinned version
                self.current_contours[idx] = thinned_contour
        
        # Clear selection and update display
        self.clear_selection()
        self.update_display_from_contours()

    # thinning
    def handle_thinning_click(self, x, y):
        """Handle clicks for thinning mode."""
        if not self.current_contours or self.current_image is None:
            return
            
        # Convert display coordinates to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        
        # Check if coordinates are valid
        if img_x is None or img_y is None:
            return
            
        # Clear any existing selection when handling a single click
        self.clear_selection()
        
        # Save state before modifying
        self.save_state()
        
        # Use the highlighted contour if available
        if self.highlighted_contour_index != -1:
            print(f"Thinning highlighted contour {self.highlighted_contour_index}")
            contour = self.current_contours[self.highlighted_contour_index]
            thinned_contour = self.thin_selected_contour(contour)
            self.current_contours[self.highlighted_contour_index] = thinned_contour
            self.highlighted_contour_index = -1  # Reset highlight
            self.update_display_from_contours()
            return
            
        # Find contours where the click is on or near an edge
        min_distance = float('inf')
        closest_contour_index = -1
        
        # Check if click is on or near a contour edge
        for i, contour in enumerate(self.current_contours):
            contour_points = contour.reshape(-1, 2)
            
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                distance = self.point_to_line_distance(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                
                # If point is close enough to a line segment
                if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                    min_distance = distance
                    closest_contour_index = i
        
        # If click is on or near an edge, thin that contour
        if closest_contour_index != -1:
            print(f"Thinning contour {closest_contour_index} (edge clicked)")
            contour = self.current_contours[closest_contour_index]
            thinned_contour = self.thin_selected_contour(contour)
            self.current_contours[closest_contour_index] = thinned_contour
            self.update_display_from_contours()
            return

    # thinning
    def thin_contour(self, contour):
        """Thin a single contour using morphological thinning."""
        # Create a mask for the contour
        mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

    # util
    def point_to_line_distance(self, x, y, x1, y1, x2, y2):
        """Calculate the distance from point (x,y) to line segment (x1,y1)-(x2,y2)."""
        # Line segment length squared
        l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        if l2 == 0:  # Line segment is a point
            return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        
        # Calculate projection of point onto line
        t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2
        
        # If projection is outside segment, calculate distance to endpoints
        if t < 0:
            return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        elif t > 1:
            return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
        
        # Calculate distance to line
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        return math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)

    # util
    def line_segments_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Check if two line segments (x1,y1)-(x2,y2) and (x3,y3)-(x4,y4) intersect."""
        # Calculate the direction vectors
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        # Calculate the determinant
        d = dx1 * dy2 - dy1 * dx2
        
        # If determinant is zero, lines are parallel and don't intersect
        if d == 0:
            return False
            
        # Calculate the parameters for the intersection point
        t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / d
        t2 = ((x1 - x3) * dy1 - (y1 - y3) * dx1) / (-d)
        
        # Check if the intersection point lies on both line segments
        return 0 <= t1 <= 1 and 0 <= t2 <= 1

    # app
    def update_display_from_contours(self):
        """Update the display with the current contours."""
        if self.current_image is not None and self.current_contours:
            self.processed_image = draw_walls(self.current_image, self.current_contours)
            self.original_processed_image = self.processed_image.copy()
            self.display_image(self.processed_image)
        elif self.current_image is not None:
            self.processed_image = self.current_image.copy()
            self.original_processed_image = self.processed_image.copy()
            self.display_image(self.processed_image)

    # app
    def display_image(self, image):
        """Display an image on the image label."""
        # Only proceed if the image label exists and has a valid size
        if not hasattr(self, 'image_label') or self.image_label.width() <= 0 or self.image_label.height() <= 0:
            return
            
        rgb_image = convert_to_rgb(image)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_image = QImage(rgb_image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    # app
    def open_image(self):
        """Open an image file and prepare scaled versions for processing."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_path:
            # Get file extension
            file_extension = os.path.splitext(file_path)[1].lower()
            is_webp = file_extension == '.webp'
            
            # Load the original full-resolution image
            print(f"Loading image: {file_path}")
            self.original_image = load_image(file_path)
            
            # Verify image loaded correctly
            if self.original_image is None:
                QMessageBox.critical(self, "Error", f"Failed to load image: {file_path}")
                return
                
            # Log the image dimensions and type info for debugging
            h, w = self.original_image.shape[:2]
            channels = self.original_image.shape[2] if len(self.original_image.shape) > 2 else 1
            print(f"Image loaded: {w}x{h}, {channels} channels, {self.original_image.dtype}")
            
            # For WebP files, log whether conversion was applied
            if is_webp:
                print(f"WebP image detected: {file_path}")
            
            # Clear history when loading a new image
            self.history.clear()
            self.undo_button.setEnabled(False)
            
            # Create a scaled down version for processing if needed
            self.current_image, self.scale_factor = self.create_working_image(self.original_image)
            
            print(f"Image prepared: Original size {self.original_image.shape}, Working size {self.current_image.shape}, Scale factor {self.scale_factor}")
            
            # Reset the mask layer when loading a new image to prevent dimension mismatch
            self.mask_layer = None
            self.foundry_walls_preview = None

            self.set_controls_enabled(True)
            
            # Reset button states when loading a new image
            self.export_foundry_button.setEnabled(False)
            
            # Reset the current overlays and detected contours
            self.current_contours = None
            self.edges_overlay = None
            
            # Update the image display
            self.update_image()

    # app
    def load_image_from_url(self):
        """Load an image from a URL in the clipboard."""
        # Get clipboard content
        clipboard = QApplication.clipboard()
        clipboard_text = clipboard.text().strip()
        
        # Check if it's a valid URL
        if not clipboard_text:
            QMessageBox.warning(self, "Invalid URL", "Clipboard is empty")
            return
            
        try:
            # Check if it's a valid URL
            parsed_url = urllib.parse.urlparse(clipboard_text)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                QMessageBox.warning(self, "Invalid URL", f"The clipboard does not contain a valid URL:\n{clipboard_text}")
                return
            
            # Download image from URL
            self.setStatusTip(f"Downloading image from {clipboard_text}...")
            response = requests.get(clipboard_text, stream=True, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            # Check if content type is an image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                QMessageBox.warning(self, "Invalid Content", f"The URL does not point to an image (Content-Type: {content_type})")
                return
                
            # Convert response content to an image
            image_data = io.BytesIO(response.content)
            image_array = np.frombuffer(image_data.read(), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if img is None:
                QMessageBox.warning(self, "Loading Error", "Could not decode image from URL")
                return
            
            # Load the image into the application
            self.original_image = img
            self.current_image, self.scale_factor = self.create_working_image(self.original_image)
            
            print(f"Image loaded from URL: Original size {self.original_image.shape}, Working size {self.current_image.shape}, Scale factor {self.scale_factor}")
            
            # Reset the mask layer when loading a new image to prevent dimension mismatch
            self.mask_layer = None
            self.foundry_walls_preview = None

            self.set_controls_enabled(True)
            
            # Reset button states when loading a new image
            self.export_foundry_button.setEnabled(False)
            self.save_foundry_button.setEnabled(False)
            self.cancel_foundry_button.setEnabled(False)
            self.copy_foundry_button.setEnabled(False)
            
            # Update the display
            self.setStatusTip(f"Image loaded from URL. Size: {img.shape[1]}x{img.shape[0]}")
            self.update_image()
            
        except requests.exceptions.RequestException as e:
            QMessageBox.warning(self, "Download Error", f"Failed to download the image:\n{str(e)}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image from URL:\n{str(e)}")

    # app
    def create_working_image(self, image):
        """Create a working copy of the image, scaling it down if it's too large."""
        # Check if we should use full resolution
        if self.high_res_checkbox.isChecked():
            return image.copy(), 1.0
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate scale factor if image is larger than the maximum working dimension
        max_dim = max(width, height)
        if max_dim <= self.max_working_dimension:
            # Image is already small enough - use as is
            return image.copy(), 1.0
        
        # Calculate scale factor and new dimensions
        scale_factor = self.max_working_dimension / max_dim
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize the image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized, scale_factor

    # app
    def scale_contours_to_original(self, contours, scale_factor):
        """Scale contours back to the original image size."""
        if scale_factor == 1.0:
            # No scaling needed
            return contours
            
        scaled_contours = []
        for contour in contours:
            # Create a scaled copy of the contour
            scaled_contour = contour.copy().astype(np.float32)
            scaled_contour /= scale_factor  # Scale coordinates
            scaled_contours.append(scaled_contour.astype(np.int32))
        
        return scaled_contours
    
    # app
    def scale_contours_to_working(self, contours, scale_factor):
        """Scale contours to the working image size."""
        if scale_factor == 1.0:
            # No scaling needed
            return contours
            
        scaled_contours = []
        for contour in contours:
            # Create a scaled copy of the contour
            scaled_contour = contour.copy().astype(np.float32)
            scaled_contour *= scale_factor  # Scale coordinates
            scaled_contours.append(scaled_contour.astype(np.int32))
        
        return scaled_contours

    # app
    def save_image(self):
        """Save the processed image at full resolution."""
        if self.original_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
            if file_path:
                if self.edit_mask_mode_enabled and self.mask_layer is not None:
                    # Save image with mask overlay
                    if self.scale_factor != 1.0:
                        # Scale mask to original resolution
                        orig_h, orig_w = self.original_image.shape[:2]
                        full_res_mask = cv2.resize(self.mask_layer, (orig_w, orig_h), 
                                               interpolation=cv2.INTER_NEAREST)
                    else:
                        full_res_mask = self.mask_layer
                        
                    # Blend mask with original image
                    result = blend_image_with_mask(self.original_image, full_res_mask)
                    
                    # Save the result
                    save_image(result, file_path)
                    print(f"Saved image with mask overlay to {file_path}")
                else:
                    # Normal save with contours
                    if self.original_image is not None and self.current_contours:
                        # Scale contours back to original image size if needed
                        if self.scale_factor != 1.0:
                            full_res_contours = self.scale_contours_to_original(self.current_contours, self.scale_factor)
                        else:
                            full_res_contours = self.current_contours
                            
                        # Draw walls on the original high-resolution image
                        high_res_result = draw_walls(self.original_image, full_res_contours)
                        
                        # Save the high-resolution result
                        save_image(high_res_result, file_path)
                        print(f"Saved high-resolution image ({self.original_image.shape[:2]}) to {file_path}")
                    else:
                        # Just save the current view if no contours
                        save_image(self.original_image, file_path)
                        print(f"Saved original image to {file_path}")

    # color
    def add_wall_color(self):
        """Open a color dialog to add a new wall color."""
        color = QColorDialog.getColor(QColor(0, 0, 0), self, "Select Wall Color")
        if color.isValid():
            # Use the global threshold value as default for new colors
            default_threshold = 0
            item = self.add_wall_color_to_list(color, default_threshold)
            
            # Select the new color
            self.wall_colors_list.setCurrentItem(item)
            self.select_color(item)
            
            # Update detection if image is loaded
            if self.current_image is not None:
                self.update_image()
    
    # color
    def select_color(self, item):
        """Handle selection of a color in the list."""
        self.selected_color_item = item
        
        # Get color data
        color_data = item.data(Qt.ItemDataRole.UserRole)
        threshold = color_data["threshold"]
        
        # Update the threshold slider to show the selected color's threshold
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(threshold * 10))
        self.threshold_slider.blockSignals(False)
        self.threshold_label.setText(f"Threshold: {threshold:.1f}")
        
        # Show the threshold container
        self.threshold_container.setVisible(True)
    
    # color
    def update_selected_threshold(self, value):
        """Update the threshold for the selected color."""
        if not self.selected_color_item:
            return
            
        # Calculate the actual threshold value
        threshold = value / 10.0
        self.threshold_label.setText(f"Threshold: {threshold:.1f}")
        
        # Get the current color data
        color_data = self.selected_color_item.data(Qt.ItemDataRole.UserRole)
        color = color_data["color"]
        
        # Update the color data with the new threshold
        self.update_color_list_item(self.selected_color_item, color, threshold)
        
        # Update detection immediately for visual feedback
        if self.current_image is not None and self.color_detection_radio.isChecked():
            self.update_image()
    
    # color
    def edit_wall_color(self, item):
        """Edit an existing color."""
        color_data = item.data(Qt.ItemDataRole.UserRole)
        current_color = color_data["color"]
        current_threshold = color_data["threshold"]
        
        new_color = QColorDialog.getColor(current_color, self, "Edit Wall Color")
        if new_color.isValid():
            # Keep the threshold and update the color
            self.update_color_list_item(item, new_color, current_threshold)
            # Update detection if image is loaded
            if self.current_image is not None:
                self.update_image()
    
    # color
    def update_color_list_item(self, item, color, threshold):
        """Update a color list item with new color and threshold."""
        # Store both color and threshold in the item data
        color_data = {"color": color, "threshold": threshold}
        item.setData(Qt.ItemDataRole.UserRole, color_data)
        
        # Update item text and appearance
        item.setText(f"RGB: {color.red()}, {color.green()}, {color.blue()} (T: {threshold:.1f})")
        item.setBackground(color)
        
        # Set text color based on background brightness
        if color.lightness() < 128:
            item.setForeground(QColor(255, 255, 255))
        else:
            item.setForeground(QColor(0, 0, 0))
    
    # color
    def add_wall_color_to_list(self, color, threshold=10.0):
        """Add a color with threshold to the wall colors list."""
        item = QListWidgetItem()
        self.update_color_list_item(item, color, threshold)
        self.wall_colors_list.addItem(item)
        return item
    
    # color
    def remove_wall_color(self):
        """Remove the selected color from the list."""
        selected_items = self.wall_colors_list.selectedItems()
        for item in selected_items:
            self.wall_colors_list.takeItem(self.wall_colors_list.row(item))
        
        # Hide threshold controls if no colors are selected or all are removed
        if not self.wall_colors_list.selectedItems() or self.wall_colors_list.count() == 0:
            self.threshold_container.setVisible(False)
            self.selected_color_item = None
        
        # Update detection if image is loaded and we still have colors
        if self.current_image is not None and self.wall_colors_list.count() > 0:
            self.update_image()

    # app
    def export_to_foundry_vtt(self):
        """Prepare walls for export to Foundry VTT and show a preview."""
        if self.current_image is None:
            return
            
        # Determine which walls to export (contours or mask)
        walls_to_export = None
        
        # Get original image dimensions for proper scaling
        if self.original_image is not None:
            image_shape = self.original_image.shape
        else:
            image_shape = self.current_image.shape
            
        # Always use the mask_layer if it exists, regardless of current mode
        if self.mask_layer is not None:
            # Extract contours from the mask - use alpha channel to determine walls
            alpha_mask = self.mask_layer[:, :, 3].copy()
            
            # If we're working with a scaled image, we need to scale the mask back to original size
            if self.scale_factor != 1.0:
                orig_h, orig_w = image_shape[:2]
                alpha_mask = cv2.resize(alpha_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
            walls_to_export = alpha_mask
        elif self.current_contours:
            # Use detected contours directly
            walls_to_export = self.current_contours
            
            # If using working image, scale contours back to original size
            if (self.scale_factor != 1.0):
                walls_to_export = self.scale_contours_to_original(walls_to_export, self.scale_factor)
        else:
            print("No walls to export.")
            return
            
        # Create a single dialog to gather all export parameters
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Parameters")
        layout = QVBoxLayout(dialog)

        # Add export preset dropdown at the top
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Export Preset:")
        preset_combo = QComboBox()
        preset_combo.addItem("-- Select Preset --")
        for name in sorted(self.export_presets.keys()):
            preset_combo.addItem(name)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(preset_combo, 1)
        
        # Add buttons for preset management
        preset_buttons_layout = QHBoxLayout()
        preset_buttons_layout.addStretch(1)
        
        save_preset_button = QPushButton("Save Preset")
        save_preset_button.setToolTip("Save current settings as a preset")
        preset_buttons_layout.addWidget(save_preset_button)
        
        manage_presets_button = QPushButton("Manage")
        manage_presets_button.setToolTip("Manage saved presets")
        preset_buttons_layout.addWidget(manage_presets_button)
        
        # Preset section container
        preset_section = QVBoxLayout()
        preset_section.addLayout(preset_layout)
        preset_section.addLayout(preset_buttons_layout)
        
        layout.addLayout(preset_section)
        
        # Add a separator after presets
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Simplification tolerance
        tolerance_label = QLabel("Simplification Tolerance (0 = maintain full detail):")
        tolerance_input = QDoubleSpinBox()
        tolerance_input.setRange(0.0, 1.0)
        tolerance_input.setDecimals(4)
        tolerance_input.setSingleStep(0.0005)
        tolerance_input.setValue(0.0005)
        tolerance_input.setToolTip("Controls how much wall detail is simplified:\n"
                                  "0 = No simplification, preserves all curves and detail\n"
                                  "0.0005 = Minor simplification, removes microscopic zigzags\n"
                                  "0.001-0.005 = Moderate simplification, smooths small details\n"
                                  "0.01+ = Heavy simplification, only keeps major shapes")
        layout.addWidget(tolerance_label)
        layout.addWidget(tolerance_input)

        # Maximum wall length
        max_length_label = QLabel("Maximum Wall Segment Length (pixels):")
        max_length_input = QSpinBox()
        max_length_input.setRange(5, 500)
        max_length_input.setSingleStep(5)
        max_length_input.setValue(50)
        max_length_input.setToolTip("Limits the maximum length of a single wall segment:\n"
                                   "Lower values (20-50): Creates more, shorter walls which are more adjustable in Foundry\n"
                                   "Higher values (100+): Creates fewer, longer wall segments for better performance\n"
                                   "This setting affects Foundry VTT performance - longer walls mean fewer total walls")
        layout.addWidget(max_length_label)
        layout.addWidget(max_length_input)

        # Maximum number of walls
        max_walls_label = QLabel("Maximum Number of Generation Points:")
        max_walls_input = QSpinBox()
        max_walls_input.setRange(100, 20000)
        max_walls_input.setSingleStep(100)
        max_walls_input.setValue(5000)
        max_walls_input.setToolTip("Caps the total number of wall points generated:\n"
                                  "Lower values (1000-3000): Better performance but may truncate complex maps\n"
                                  "Higher values (5000+): Handles more complex maps but may impact Foundry performance\n"
                                  "These are not the walls that will be exported, but the points used to generate them")
        layout.addWidget(max_walls_label)
        layout.addWidget(max_walls_input)

        # Point merge distance
        merge_distance_label = QLabel("Point Merge Distance (pixels):")
        merge_distance_input = QDoubleSpinBox()
        merge_distance_input.setRange(0.0, 500.0)
        merge_distance_input.setDecimals(1)
        merge_distance_input.setSingleStep(1.0)
        merge_distance_input.setValue(25.0)
        merge_distance_input.setToolTip("Controls how close wall endpoints must be to get merged together:\n"
                                       "0: No merging at all, keeps all endpoints separate\n"
                                       "1-5: Minimal merging, only combines endpoints that are very close\n"
                                       "10-25: Standard merging, fixes most small gaps between walls\n"
                                       "25+: Aggressive merging, can close larger gaps but may connect unrelated walls")
        layout.addWidget(merge_distance_label)
        layout.addWidget(merge_distance_input)

        # Angle tolerance
        angle_tolerance_label = QLabel("Angle Tolerance (degrees):")
        angle_tolerance_input = QDoubleSpinBox()
        angle_tolerance_input.setRange(0.0, 30.0)
        angle_tolerance_input.setDecimals(2)
        angle_tolerance_input.setSingleStep(0.5)
        angle_tolerance_input.setValue(1.0)
        angle_tolerance_input.setToolTip("When merging straight walls, this controls how closely aligned walls must be:\n"
                                        "0: Walls must be perfectly aligned to merge\n"
                                        "1-2: Only merge walls with very similar angles\n"
                                        "5+: Merge walls even if they're at slightly different angles\n"
                                        "Higher values create fewer wall segments but can distort corners")
        layout.addWidget(angle_tolerance_label)
        layout.addWidget(angle_tolerance_input)

        # Maximum gap
        max_gap_label = QLabel("Maximum Straight Gap to Connect (pixels):")
        max_gap_input = QDoubleSpinBox()
        max_gap_input.setRange(0.0, 100.0)
        max_gap_input.setDecimals(1)
        max_gap_input.setSingleStep(1.0)
        max_gap_input.setValue(10.0)
        max_gap_input.setToolTip("Maximum distance between straight walls that should be connected:\n"
                                "This should be smaller than the smallest opening between 2 straight walls\n"
                                "0: Don't connect straight walls\n"
                                "1-5: Connect only very close straight walls\n"
                                "10-20: Good for most cases\n"
                                "20+: Aggressively connect straight walls")
        layout.addWidget(max_gap_label)
        layout.addWidget(max_gap_input)

        # Grid snapping section
        grid_section_label = QLabel("Grid Snapping:")
        grid_section_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(grid_section_label)

        # Grid size
        grid_size_layout = QHBoxLayout()
        grid_size_label = QLabel("Grid Size (pixels, 0 to disable):")
        grid_size_input = QSpinBox()
        grid_size_input.setRange(0, 500)
        grid_size_input.setSingleStep(1)
        grid_size_input.setValue(0)
        grid_size_input.setToolTip("Aligns walls to a grid of this size (in pixels):\n"
                                  "0: No grid alignment, walls follow exact detected contours\n"
                                  "70-100: Common grid sizes for standard VTT maps\n"
                                  "Set this to match your Foundry scene's grid size for perfect alignment")
        grid_size_layout.addWidget(grid_size_label)
        grid_size_layout.addWidget(grid_size_input)
        layout.addLayout(grid_size_layout)

        # Allow half grid checkbox
        allow_half_grid = QCheckBox("Allow Half-Grid Positions")
        allow_half_grid.setChecked(False)
        allow_half_grid.setToolTip("When grid snapping is enabled:\n"
                                  "Checked: Walls can align to half-grid positions (more precise)\n"
                                  "Unchecked: Walls only align to full grid intersections (cleaner)")
        layout.addWidget(allow_half_grid)

        # Connect preset selector to update form
        def apply_selected_preset(index):
            if index <= 0:  # Skip the placeholder item
                return
                
            preset_name = preset_combo.itemText(index)
            if preset_name in self.export_presets:
                preset = self.export_presets[preset_name]
                tolerance_input.setValue(preset.get("simplify_tolerance", 0.0005))
                max_length_input.setValue(preset.get("max_wall_length", 50))
                max_walls_input.setValue(preset.get("max_walls", 5000))
                merge_distance_input.setValue(preset.get("merge_distance", 25.0))
                angle_tolerance_input.setValue(preset.get("angle_tolerance", 1.0))
                max_gap_input.setValue(preset.get("max_gap", 10.0))
                grid_size_input.setValue(preset.get("grid_size", 0))
                allow_half_grid.setChecked(preset.get("allow_half_grid", False))
                
        preset_combo.currentIndexChanged.connect(apply_selected_preset)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Connect the preset management buttons
        def save_preset_handler():
            # Call the save method and get the new preset name if successful
            new_preset_name = self.save_export_preset_from_dialog(
                tolerance_input.value(),
                max_length_input.value(),
                max_walls_input.value(),
                merge_distance_input.value(),
                angle_tolerance_input.value(),
                max_gap_input.value(),
                grid_size_input.value(),
                allow_half_grid.isChecked()
            )
            
            # If a new preset was saved, update the dropdown and select it
            if new_preset_name:
                # Update the preset dropdown with the new preset
                preset_combo.blockSignals(True)
                current_index = preset_combo.currentIndex()
                
                # Clear and rebuild the preset list to ensure it's sorted
                preset_combo.clear()
                preset_combo.addItem("-- Select Preset --")
                for name in sorted(self.export_presets.keys()):
                    preset_combo.addItem(name)
                
                # Select the new preset
                index = preset_combo.findText(new_preset_name)
                if index != -1:
                    preset_combo.setCurrentIndex(index)
                else:
                    preset_combo.setCurrentIndex(current_index)
                    
                preset_combo.blockSignals(False)
        
        save_preset_button.clicked.connect(save_preset_handler)
        manage_presets_button.clicked.connect(self.manage_export_presets)

        # Show dialog and get results
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Retrieve values
        tolerance = tolerance_input.value()
        max_length = max_length_input.value()
        max_walls = max_walls_input.value()
        merge_distance = merge_distance_input.value()
        angle_tolerance = angle_tolerance_input.value()
        max_gap = max_gap_input.value()
        grid_size = grid_size_input.value()
        half_grid_allowed = allow_half_grid.isChecked()
        
        # Store export parameters for later use when saving
        self.foundry_export_params = {
            'walls_to_export': walls_to_export,
            'image_shape': image_shape,
            'simplify_tolerance': tolerance,
            'max_wall_length': max_length,
            'max_walls': max_walls,
            'merge_distance': merge_distance,
            'angle_tolerance': angle_tolerance,
            'max_gap': max_gap,
            'grid_size': grid_size,
            'allow_half_grid': half_grid_allowed
        }
        
        # Store the current export settings (useful when creating new presets)
        self.current_export_settings = {
            'simplify_tolerance': tolerance,
            'max_wall_length': max_length,
            'max_walls': max_walls,
            'merge_distance': merge_distance,
            'angle_tolerance': angle_tolerance,
            'max_gap': max_gap,
            'grid_size': grid_size,
            'allow_half_grid': half_grid_allowed
        }

        # Switch to deletion mode for less interference with the preview
        self.color_selection_mode_radio.setChecked(True)
        
        # Generate walls for preview
        self.preview_foundry_walls()

    # app
    def preview_foundry_walls(self):
        """Generate and display a preview of the Foundry VTT walls."""
        if not self.foundry_export_params:
            return
            
        params = self.foundry_export_params
        
        # Generate walls without saving to file
        from src.wall_detection.mask_editor import contours_to_foundry_walls
        
        if isinstance(params['walls_to_export'], list):  # It's contours
            contours = params['walls_to_export']
            foundry_walls = contours_to_foundry_walls(
                contours,
                params['image_shape'],
                simplify_tolerance=params['simplify_tolerance'],
                max_wall_length=params['max_wall_length'],
                max_walls=params['max_walls'],
                merge_distance=params['merge_distance'],
                angle_tolerance=params['angle_tolerance'],
                max_gap=params['max_gap'],
                grid_size=params['grid_size'],
                allow_half_grid=params['allow_half_grid']
            )
        else:  # It's a mask
            # Extract contours from the mask - use RETR_CCOMP instead of RETR_EXTERNAL to get inner contours
            mask = params['walls_to_export']
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours to include both outer and inner contours
            from src.wall_detection.detector import process_contours_with_hierarchy
            processed_contours = process_contours_with_hierarchy(contours, hierarchy, 0, None)
            
            foundry_walls = contours_to_foundry_walls(
                processed_contours,
                params['image_shape'],
                simplify_tolerance=params['simplify_tolerance'],
                max_wall_length=params['max_wall_length'],
                max_walls=params['max_walls'],
                merge_distance=params['merge_distance'],
                angle_tolerance=params['angle_tolerance'],
                max_gap=params['max_gap'],
                grid_size=params['grid_size'],
                allow_half_grid=params['allow_half_grid']
            )
        
        # Store the generated walls for later use
        self.foundry_walls_preview = foundry_walls
        
        # Create a preview image showing the walls
        self.display_foundry_preview()
        
        # Enable save/cancel/copy buttons
        self.save_foundry_button.setEnabled(True)
        self.cancel_foundry_button.setEnabled(True)
        self.copy_foundry_button.setEnabled(True)
        
        # Set flag for preview mode
        self.foundry_preview_active = True
        
        # Disable detection controls while in preview mode
        self.set_controls_enabled(False)
        
        # Update status with more detailed information
        wall_count = len(foundry_walls)
        self.setStatusTip(f"Previewing {wall_count} walls for Foundry VTT. Click 'Save Foundry Walls' to export or 'Copy to Clipboard'.")

    # app
    def display_foundry_preview(self):
        """Display a preview of the Foundry VTT walls over the current image."""
        if not self.foundry_walls_preview or self.current_image is None:
            return
            
        # Make a copy of the current image for the preview
        preview_image = self.current_image.copy()
        
        # Convert back to RGB for better visibility
        if len(preview_image.shape) == 2:  # Grayscale
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2BGR)
            
        # Draw the walls on the preview image
        for wall in self.foundry_walls_preview:
            # Get wall coordinates
            start_x, start_y, end_x, end_y = wall["c"]
            
            # Scale coordinates to match current image if needed
            if self.scale_factor != 1.0:
                start_x *= self.scale_factor
                start_y *= self.scale_factor
                end_x *= self.scale_factor
                end_y *= self.scale_factor
            
            # Draw line for this wall segment
            cv2.line(
                preview_image,
                (int(start_x), int(start_y)),
                (int(end_x), int(end_y)),
                (0, 255, 255),  # Yellow color for preview
                2,  # Thickness
                cv2.LINE_AA  # Anti-aliased line
            )
            
            # Draw dots at wall endpoints (like Foundry VTT does)
            endpoint_color = (255, 128, 0)  # Orange dots for endpoints
            dot_radius = 4
            cv2.circle(preview_image, (int(start_x), int(start_y)), dot_radius, endpoint_color, -1)  # Start point
            cv2.circle(preview_image, (int(end_x), int(end_y)), dot_radius, endpoint_color, -1)  # End point
        
        # Add text showing the number of walls
        wall_count = len(self.foundry_walls_preview)
        # Position in top-left corner with padding
        x_pos, y_pos = 20, 40
        font_scale = 1.2
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Walls: {wall_count}"
        
        # Add a dark background for better visibility
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(
            preview_image, 
            (x_pos - 10, y_pos - text_height - 10), 
            (x_pos + text_width + 10, y_pos + 10), 
            (0, 0, 0), 
            -1
        )
        
        # Draw the text
        cv2.putText(
            preview_image,
            text,
            (x_pos, y_pos),
            font, 
            font_scale,
            (255, 255, 255),  # White text
            font_thickness
        )
        
        # Save a copy of the original processed image if not already saved
        if self.original_processed_image is None:
            self.original_processed_image = preview_image.copy()
        
        # Update the display with the preview
        self.processed_image = preview_image
        self.display_image(self.processed_image)

    # app
    def save_foundry_preview(self):
        """Save the previewed Foundry VTT walls to a JSON file."""
        if not self.foundry_walls_preview:
            return
            
        # Get file path for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Walls for Foundry VTT", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
            
        # Add .json extension if not present
        if not file_path.lower().endswith('.json'):
            file_path += '.json'
            
        # Save walls directly to JSON file
        try:
            with open(file_path, 'w') as f:
                import json
                json.dump(self.foundry_walls_preview, f, indent=2)
                
            print(f"Successfully exported {len(self.foundry_walls_preview)} walls to {file_path}")
            self.setStatusTip(f"Walls exported to {file_path}. Import in Foundry using Walls > Import")
            
            # After successful saving:
            # Disable save/cancel buttons
            self.save_foundry_button.setEnabled(False)
            self.cancel_foundry_button.setEnabled(False)
            self.copy_foundry_button.setEnabled(False)
            
            # Exit preview mode
            self.cancel_foundry_preview()
        except Exception as e:
            print(f"Failed to export walls: {e}")
            self.setStatusTip(f"Failed to export walls: {e}")

    # app
    def cancel_foundry_preview(self):
        """Cancel the Foundry VTT wall preview and return to normal view."""
        # Disable buttons
        self.save_foundry_button.setEnabled(False)
        self.cancel_foundry_button.setEnabled(False)
        self.copy_foundry_button.setEnabled(False)
        
        # Clear preview-related data
        self.foundry_walls_preview = None
        self.foundry_export_params = None
        self.foundry_preview_active = False
        
        # Re-enable detection controls
        self.set_controls_enabled(True)
        
        # Restore original display
        if self.original_processed_image is not None:
            self.processed_image = self.original_processed_image.copy()
            self.display_image(self.processed_image)
        
        # Update status
        self.setStatusTip("Foundry VTT preview canceled")

    # app
    def set_controls_enabled(self, enabled, color_detection_mode=False):
        """Enable or disable detection controls based on preview state."""
        # Disable/enable all sliders
        for slider_name, slider_info in self.sliders.items():
            if 'slider' in slider_info:
                slider_info['slider'].setEnabled(enabled)
        
        if not color_detection_mode:
            # Disable/enable detection mode radio buttons
            self.edge_detection_radio.setEnabled(enabled)
            self.color_detection_radio.setEnabled(enabled)
            
            # Disable/enable mode toggle radio buttons
            self.deletion_mode_radio.setEnabled(enabled)
            self.color_selection_mode_radio.setEnabled(enabled)
            self.edit_mask_mode_radio.setEnabled(enabled)
            self.thin_mode_radio.setEnabled(enabled)
        
        # Disable/enable high-res checkbox
        self.high_res_checkbox.setEnabled(enabled)
        
        # Disable/enable color management
        self.add_color_button.setEnabled(enabled)
        self.remove_color_button.setEnabled(enabled)
        self.wall_colors_list.setEnabled(enabled)
        
        # If re-enabling, respect color detection mode
        if enabled and self.color_detection_radio.isChecked():
            self.toggle_detection_mode_radio(True)

    # thinning
    def update_target_width(self, value):
        """Update the target width parameter for thinning."""
        self.target_width = value
        self.target_width_value.setText(str(value))
    
    # thinning
    def update_max_iterations(self, value):
        """Update the max iterations parameter for thinning."""
        self.max_iterations = value
        self.max_iterations_value.setText(str(value))

    # app
    def copy_foundry_to_clipboard(self):
        """Copy the Foundry VTT walls JSON to the clipboard."""
        if not self.foundry_walls_preview:
            QMessageBox.warning(self, "No Walls", "No walls available to copy.")
            return
            
        try:
            # Convert walls to JSON string
            walls_json = json.dumps(self.foundry_walls_preview, indent=2)
            
            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(walls_json)
            
            # Show confirmation
            self.setStatusTip(f"{len(self.foundry_walls_preview)} walls copied to clipboard. Paste in Foundry using Walls > Import.")
            QMessageBox.information(self, "Copied to Clipboard", 
                               f"{len(self.foundry_walls_preview)} walls copied to clipboard.\n"
                               f"Use Walls > Import JSON in Foundry VTT to import them.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to copy walls to clipboard: {str(e)}")

    # app
    def resizeEvent(self, event):
        """Handle window resize events to update the image display."""
        super().resizeEvent(event)
        
        # If we have a current image displayed, update it to fit the new window size
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            self.display_image(self.processed_image)
            
        # If we're in foundry preview mode, redraw the preview
        if hasattr(self, 'foundry_preview_active') and self.foundry_preview_active and self.foundry_walls_preview:
            self.display_foundry_preview()
        
        # Update the position of the update notification
        if hasattr(self, 'update_notification'):
            self.update_notification.setGeometry(
                self.width() - 250, 10, 240, 40
            )

    # app
    def save_state(self):
        """Save the current state to history for undo functionality."""
        if self.current_image is None:
            # Don't save state if there's no image loaded
            return
            
        # Save different data depending on the current mode
        if self.edit_mask_mode_enabled and self.mask_layer is not None:
            state = {
                'mode': 'mask',
                'mask': self.mask_layer.copy(),
                'original_image': None if self.original_processed_image is None else self.original_processed_image.copy()
            }
        else:
            state = {
                'mode': 'contour',
                'contours': copy.deepcopy(self.current_contours),
                'original_image': None if self.original_processed_image is None else self.original_processed_image.copy()
            }
        
        # Add state to history
        self.history.append(state)
        
        # Enable the undo button once we have history
        self.undo_button.setEnabled(True)
        
        print(f"State saved to history. History size: {len(self.history)}")

    # app
    def undo(self):
        """Restore the previous state from history."""
        if not self.history:
            print("No history available to undo")
            self.setStatusTip("Nothing to undo")
            return
            
        print(f"Undoing action. History size before: {len(self.history)}")
        
        # Pop the most recent state (we don't need it anymore)
        self.history.pop()
        
        # If no more history, disable undo button
        if not self.history:
            self.undo_button.setEnabled(False)
            self.setStatusTip("No more undo history available")
            return
        
        # Get the previous state (now the last item in the queue)
        prev_state = self.history[-1]
        
        # Restore based on the mode of the previous state
        if prev_state['mode'] == 'mask':
            self.mask_layer = prev_state['mask'].copy()
            
            if prev_state['original_image'] is not None:
                self.original_processed_image = prev_state['original_image'].copy()
                self.processed_image = self.original_processed_image.copy()
            
            # Make sure we're in edit mask mode
            if not self.edit_mask_mode_enabled:
                self.edit_mask_mode_radio.setChecked(True)
                # This is important - toggle_mode needs to be called explicitly
                self.toggle_mode()
            
            # Update the display
            self.update_display_with_mask()
            self.setStatusTip("Restored previous mask state")
            print("Restored previous mask state")
            
        else:  # contour mode
            self.current_contours = copy.deepcopy(prev_state['contours'])
            
            if prev_state['original_image'] is not None:
                self.original_processed_image = prev_state['original_image'].copy()
                self.processed_image = self.original_processed_image.copy()
            
            # Make sure we're not in mask edit mode
            if self.edit_mask_mode_enabled:
                self.deletion_mode_radio.setChecked(True)
                # This is important - toggle_mode needs to be called explicitly
                self.toggle_mode()
            
            # Update the display
            self.update_display_from_contours()
            self.setStatusTip("Restored previous contour state")
            print("Restored previous contour state")

    # app
    def keyPressEvent(self, event):
        """Handle key press events."""
        # Add debugging for Ctrl+Z
        if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            print("Ctrl+Z detected via keyPressEvent")
            self.undo()
        else:
            super().keyPressEvent(event)

    # app
    def apply_stylesheet(self):
        """Apply the application stylesheet from the CSS file."""
        try:
            # Get the path to the stylesheet
            style_path = os.path.join(os.path.dirname(__file__), 'style.qss')
            
            # Check if the file exists
            if not os.path.exists(style_path):
                print(f"Warning: Stylesheet not found at {style_path}")
                return
                
            # Read and apply the stylesheet
            with open(style_path, 'r') as f:
                stylesheet = f.read()
                self.setStyleSheet(stylesheet)
                print(f"Applied stylesheet from {style_path}")
        except Exception as e:
            print(f"Error applying stylesheet: {e}")

    # update
    def check_for_updates(self):
        """Check for updates and show notification if available."""
        try:
            is_update_available, latest_version, download_url = check_for_updates(
                self.app_version, self.github_repo
            )
            
            if is_update_available:
                self.update_available = True
                self.update_url = download_url
                self.update_text.setText(f"Update {latest_version} Available!")
                self.update_notification.show()
                print(f"Update available: version {latest_version}")
        except Exception as e:
            print(f"Error checking for updates: {e}")
    
    # update
    def open_update_url(self, event):
        """Open the update URL when the notification is clicked."""
        if self.update_url:
            QDesktopServices.openUrl(QUrl(self.update_url))

    # hatching removal
    def toggle_hatching_removal(self, enabled):
        """Toggle hatching removal options visibility."""
        self.hatching_options.setVisible(enabled)
        
        # Update the image if one is loaded
        if self.current_image is not None:
            self.update_image()
    
    # hatching removal
    def select_hatching_color(self):
        """Open a color dialog to select hatching color."""
        color = QColorDialog.getColor(self.hatching_color, self, "Select Hatching Color")
        if color.isValid():
            self.hatching_color = color
            # Update button color
            self.hatching_color_button.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")
            
            # Update the image if one is loaded and removal is enabled
            if self.current_image is not None and self.remove_hatching_checkbox.isChecked():
                self.update_image()
    
    # hatching removal
    def update_hatching_threshold(self, value):
        """Update the threshold for hatching color matching."""
        # Convert from slider value (0-300) to actual threshold (0-30.0)
        self.hatching_threshold = value / 10.0
        self.hatching_threshold_value.setText(f"{self.hatching_threshold:.1f}")
        
        # Update the image if one is loaded and removal is enabled
        if self.current_image is not None and self.remove_hatching_checkbox.isChecked():
            self.update_image()
    
    # hatching removal
    def update_hatching_width(self, value):
        """Update the maximum width of lines to remove."""
        self.hatching_width = value
        self.hatching_width_value.setText(str(value))
        
        # Update the image if one is loaded and removal is enabled
        if self.current_image is not None and self.remove_hatching_checkbox.isChecked():
            self.update_image()

    def save_export_preset_from_dialog(self, tolerance, max_length, max_walls, merge_distance, angle_tolerance, max_gap, grid_size, allow_half_grid):
        """Save current export settings from the dialog as a preset."""
        preset_name, ok = QInputDialog.getText(self, "Save Export Preset", "Enter preset name:")
        if ok and preset_name:
            # Check if overwriting a default preset
            if preset_name in self.get_default_export_presets():
                reply = QMessageBox.question(self, "Overwrite Default Preset?",
                                            f"'{preset_name}' is a default preset. Overwrite it?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return  # User cancelled overwrite

            # Check if overwriting an existing user preset
            elif preset_name in self.export_presets:
                reply = QMessageBox.question(self, "Overwrite Preset?",
                                            f"Preset '{preset_name}' already exists. Overwrite?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return  # User cancelled overwrite

            # Save the preset using the parameters provided directly from the dialog
            new_preset = {
                "simplify_tolerance": tolerance,
                "max_wall_length": max_length,
                "max_walls": max_walls,
                "merge_distance": merge_distance,
                "angle_tolerance": angle_tolerance,
                "max_gap": max_gap,
                "grid_size": grid_size,
                "allow_half_grid": allow_half_grid
            }
            
            # Store the preset and save to file
            self.export_presets[preset_name] = new_preset
            self.save_presets_to_file()
            
            # Also update current_export_settings for consistency
            self.current_export_settings = new_preset.copy()
            
            self.setStatusTip(f"Saved export preset '{preset_name}'")
            
            # Return the preset name so the dialog can update its dropdown
            return preset_name
        elif ok and not preset_name:
            QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")
            
        return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WallDetectionApp()
    window.show()
    sys.exit(app.exec())