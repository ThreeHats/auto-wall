import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QWidget, 
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup, QColorDialog, QListWidget, QListWidgetItem,
    QScrollArea, QSizePolicy, QDialog, QDialogButtonBox, QFrame, QSpinBox, QInputDialog, QDoubleSpinBox,
    QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt, QPoint, QRect, QBuffer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QMouseEvent, QCursor, QClipboard, QGuiApplication
import cv2
import numpy as np
import math
import json
import requests
import io
import urllib.parse
from sklearn.cluster import KMeans

from src.wall_detection.detector import detect_walls, draw_walls, merge_contours, split_edge_contours
from src.wall_detection.image_utils import load_image, save_image, convert_to_rgb
from src.wall_detection.mask_editor import create_mask_from_contours, blend_image_with_mask, draw_on_mask, export_mask_to_foundry_json, contours_to_foundry_walls, thin_contour


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
                    self.parent_app.start_drawing(x, y)
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
                    self.parent_app.continue_drawing(self.last_point.x(), self.last_point.y(), x, y)
                    self.last_point = current_point
                elif self.parent_app.thin_mode_enabled:
                    self.parent_app.update_selection(x, y)
            # Just hovering - this always runs for any mouse movement
            else:
                if self.parent_app.deletion_mode_enabled:
                    self.parent_app.handle_hover(pos.x(), pos.y())
                elif self.parent_app.edit_mask_mode_enabled:
                    # Always update brush preview when hovering in edit mask mode
                    self.parent_app.update_brush_preview(x, y)
                
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
                    self.parent_app.end_drawing()
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
            if self.parent_app.deletion_mode_enabled:
                self.parent_app.clear_hover()
            elif self.parent_app.edit_mask_mode_enabled:
                # Clear brush preview when mouse leaves the widget
                self.parent_app.clear_brush_preview()
        super().leaveEvent(event)


class WallDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto-Wall: Battle Map Wall Detection")
        
        # Get the screen size and set the window to maximize
        screen = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        self.showMaximized()

        # Main layout - use central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Use horizontal layout for main container (controls on left, image on right)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Left panel for controls (using scroll area for many controls)
        self.controls_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_panel)
        
        # Wrap controls in a scroll area to handle many options
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.controls_panel)
        self.scroll_area.setMinimumWidth(350)  # Set minimum width for controls panel
        self.scroll_area.setMaximumWidth(400)  # Set maximum width for controls panel
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

        # Add title for controls section
        self.controls_title = QLabel("Wall Detection Controls")
        self.controls_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.controls_layout.addWidget(self.controls_title)

        # Sliders
        self.sliders = {}
        self.add_slider("Min Area", 0, 10000, 6170)
        self.add_slider("Blur", 1, 21, 5, step=2)
        self.add_slider("Canny1", 0, 255, 255)
        self.add_slider("Canny2", 0, 255, 106)
        self.add_slider("Edge Margin", 0, 50, 0)
        
        # Use a scaling factor of 10 for float values (0 to 10.0 with 0.1 precision)
        self.add_slider("Min Merge Distance", 0, 100, 5, scale_factor=0.1)  # Default 0.5

        # Mode selection (Detection/Deletion/Color Selection/Edit Mask)
        self.mode_layout = QHBoxLayout()
        self.controls_layout.addLayout(self.mode_layout)
        
        self.mode_label = QLabel("Mode:")
        self.mode_layout.addWidget(self.mode_label)
        
        self.color_selection_mode_radio = QRadioButton("Color Pick")
        self.deletion_mode_radio = QRadioButton("Deletion")
        self.edit_mask_mode_radio = QRadioButton("Edit Mask")
        self.thin_mode_radio = QRadioButton("Thin")  # New radio button for thinning mode
        self.deletion_mode_radio.setChecked(True)

        self.mode_layout.addWidget(self.color_selection_mode_radio)
        self.mode_layout.addWidget(self.deletion_mode_radio)
        self.mode_layout.addWidget(self.edit_mask_mode_radio)
        self.mode_layout.addWidget(self.thin_mode_radio)  # Add the new radio button
        
        # Connect mode radio buttons
        self.color_selection_mode_radio.toggled.connect(self.toggle_mode)
        self.deletion_mode_radio.toggled.connect(self.toggle_mode)
        self.edit_mask_mode_radio.toggled.connect(self.toggle_mode)
        self.thin_mode_radio.toggled.connect(self.toggle_mode)  # Connect new radio button
        
        # Add color selection options
        self.color_selection_options = QWidget()
        self.color_selection_layout = QHBoxLayout(self.color_selection_options)
        self.color_selection_layout.setContentsMargins(0, 0, 0, 0)
        
        self.color_count_label = QLabel("Colors:")
        self.color_selection_layout.addWidget(self.color_count_label)
        
        self.color_count_spinner = QSpinBox()
        self.color_count_spinner.setMinimum(1)
        self.color_count_spinner.setMaximum(10)
        self.color_count_spinner.setValue(3)
        self.color_count_spinner.setToolTip("Number of colors to extract")
        self.color_selection_layout.addWidget(self.color_count_spinner)
        
        self.controls_layout.addWidget(self.color_selection_options)
        self.color_selection_options.setVisible(False)
        
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
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)
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
        
        # Add tools to the layout in a grid (3x2)
        self.draw_tool_layout.addWidget(self.brush_tool_radio, 0, 0)
        self.draw_tool_layout.addWidget(self.line_tool_radio, 0, 1)
        self.draw_tool_layout.addWidget(self.rectangle_tool_radio, 1, 0)
        self.draw_tool_layout.addWidget(self.circle_tool_radio, 1, 1)
        self.draw_tool_layout.addWidget(self.ellipse_tool_radio, 2, 0)
        self.draw_tool_layout.addWidget(self.fill_tool_radio, 2, 1)
        
        self.mask_edit_layout.addLayout(self.draw_tool_layout)

        self.controls_layout.addWidget(self.mask_edit_options)
        self.mask_edit_options.setVisible(False)
        
        # Deletion mode is initially disabled
        self.deletion_mode_enabled = True
        self.color_selection_mode_enabled = False
        self.edit_mask_mode_enabled = False
        self.thin_mode_enabled = False  # Add new state variable for thin mode

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
        
        # Add thinning options to main controls
        self.controls_layout.addWidget(self.thin_options)
        self.thin_options.setVisible(False)
        
        # Store thinning parameters
        self.target_width = 5
        self.max_iterations = 3

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

        # Checkboxes for merge options
        self.merge_options_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.merge_options_layout)

        self.merge_contours = QCheckBox("Merge Contours")
        self.merge_contours.setChecked(False)
        self.merge_options_layout.addWidget(self.merge_contours)

        # Add a checkbox for high-resolution processing
        self.high_res_checkbox = QCheckBox("Process at Full Resolution")
        self.high_res_checkbox.setChecked(False)
        self.high_res_checkbox.setToolTip("Process at full resolution (slower but more accurate)")
        self.high_res_checkbox.stateChanged.connect(self.reload_working_image)
        self.merge_options_layout.addWidget(self.high_res_checkbox)
        
        # Add color detection section
        self.color_section_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.color_section_layout)
        
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
        
        # Store the currently selected color item
        self.selected_color_item = None
        
        # Color detection options - Add the missing checkbox
        self.use_color_detection = QCheckBox("Enable Color Detection")
        self.use_color_detection.setChecked(False)
        self.use_color_detection.toggled.connect(self.toggle_detection_mode)
        self.color_section_layout.addWidget(self.use_color_detection)
        
        # Label for edge detection section
        self.edge_section_title = QLabel("Edge Detection Settings:")
        self.edge_section_title.setStyleSheet("font-weight: bold;")
        self.controls_layout.addWidget(self.edge_section_title)
        
        # Group edge detection settings
        self.edge_detection_widgets = []
        self.edge_detection_widgets.append(self.sliders["Canny1"])
        self.edge_detection_widgets.append(self.sliders["Canny2"])
        
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
        self.mask_layer = None  # Will hold the editable mask
        self.brush_size = 10
        self.drawing_mode = True  # True for draw, False for erase
        self.current_tool = "brush"  # Default tool
        self.drawing_start_pos = None  # Starting position for shape tools
        self.temp_drawing = None  # For temporary preview of shapes

        # Add these new variables to track the brush preview state
        self.brush_preview_active = False
        self.last_preview_image = None
        self.foundry_preview_active = False 

        # Create a new section for wall action buttons in the main sidebar
        self.wall_actions_title = QLabel("Wall Actions")
        self.wall_actions_title.setStyleSheet("font-weight: bold;")
        self.controls_layout.addWidget(self.wall_actions_title)
        
        # Create a layout for the wall action buttons
        self.wall_actions_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.wall_actions_layout)
        
        # Move the Bake button to the main sidebar
        self.bake_button = QPushButton("Bake Contours to Mask")
        self.bake_button.clicked.connect(self.bake_contours_to_mask)
        self.wall_actions_layout.addWidget(self.bake_button)
        
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
        
        # Add a separator between sections
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.controls_layout.addWidget(separator)

        # Connect drawing tool radio buttons to handler
        self.brush_tool_radio.toggled.connect(self.update_drawing_tool)
        self.line_tool_radio.toggled.connect(self.update_drawing_tool)
        self.rectangle_tool_radio.toggled.connect(self.update_drawing_tool)
        self.circle_tool_radio.toggled.connect(self.update_drawing_tool)
        self.ellipse_tool_radio.toggled.connect(self.update_drawing_tool)
        self.fill_tool_radio.toggled.connect(self.update_drawing_tool)

    # Then add this new method:
    def reload_working_image(self):
        """Reload the working image when resolution setting changes."""
        if self.original_image is None:
            return
        
        # Recreate the working image with the current checkbox state
        self.current_image, self.scale_factor = self.create_working_image(self.original_image)
        print(f"Resolution changed: Working size {self.current_image.shape}, Scale factor {self.scale_factor}")
        
        # Update the image with new resolution
        self.update_image()

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

    def toggle_mode(self):
        """Toggle between detection, deletion, color selection, edit mask, and thinning modes."""
        self.color_selection_mode_enabled = self.color_selection_mode_radio.isChecked()
        self.deletion_mode_enabled = self.deletion_mode_radio.isChecked()
        self.edit_mask_mode_enabled = self.edit_mask_mode_radio.isChecked()
        self.thin_mode_enabled = self.thin_mode_radio.isChecked()  # Add thin mode check
        
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
                    self.update_brush_preview(cursor_pos.x(), cursor_pos.y())
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

    def update_brush_size(self, value):
        """Update the brush size."""
        self.brush_size = value
        self.brush_size_value.setText(str(value))
        
        # Update brush preview if in edit mode
        if self.edit_mask_mode_enabled:
            # Get current cursor position relative to the image label
            cursor_pos = self.image_label.mapFromGlobal(QCursor.pos())
            if self.image_label.rect().contains(cursor_pos):
                self.update_brush_preview(cursor_pos.x(), cursor_pos.y())

    def update_brush_preview(self, x, y):
        """Show a preview of the brush outline at the current mouse position."""
        if not self.edit_mask_mode_enabled or self.current_image is None:
            return
        
        # Convert display coordinates to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        if img_x is None or img_y is None:
            return
        
        # Make a copy of the blended image with mask
        if self.mask_layer is not None:
            # Create the blended image (original + current mask)
            blended_image = blend_image_with_mask(self.current_image, self.mask_layer)
            
            # Draw brush outline with different colors for draw/erase mode
            color = (0, 255, 0) if self.draw_radio.isChecked() else (0, 0, 255)  # Green for draw, Red for erase
            cv2.circle(blended_image, (img_x, img_y), self.brush_size, color, 1)
            
            # Store this as our preview image
            self.last_preview_image = blended_image.copy()
            self.brush_preview_active = True
            
            # Display the image with brush preview
            self.display_image(blended_image)
        elif self.original_processed_image is not None:
            # If no mask exists yet, draw on the original image
            preview_image = self.original_processed_image.copy()
            
            # Draw brush outline
            color = (0, 255, 0) if self.draw_radio.isChecked() else (0, 0, 255)
            cv2.circle(preview_image, (img_x, img_y), self.brush_size, color, 1)
            
            # Store this as our preview image
            self.last_preview_image = preview_image.copy()
            self.brush_preview_active = True
            
            # Display the preview
            self.display_image(preview_image)

    def clear_brush_preview(self):
        """Clear the brush preview when mouse leaves the widget."""
        self.brush_preview_active = False
        
        # Restore the original display
        if self.edit_mask_mode_enabled and self.mask_layer is not None:
            # Redraw the blend without the brush preview
            self.update_display_with_mask()
        elif self.original_processed_image is not None:
            # Restore the original image
            self.display_image(self.original_processed_image)

    def create_empty_mask(self):
        """Create an empty transparent mask layer."""
        if self.current_image is None:
            return
            
        height, width = self.current_image.shape[:2]
        # Create a transparent mask (4th channel is alpha, all 0 = fully transparent)
        self.mask_layer = np.zeros((height, width, 4), dtype=np.uint8)

    def bake_contours_to_mask(self):
        """Bake the current contours to the mask layer."""
        if self.current_image is None or not self.current_contours:
            return
            
        # Create the mask from contours
        self.mask_layer = create_mask_from_contours(
            self.current_image.shape, 
            self.current_contours,
            color=(0, 255, 0, 255)  # Green
        )
        
        # Switch to mask editing mode
        self.edit_mask_mode_radio.setChecked(True)
        
        # Enable the Export to Foundry VTT button
        self.export_foundry_button.setEnabled(True)
        
        # Update display
        self.update_display_with_mask()

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

    def start_drawing(self, x, y):
        """Start drawing on the mask at the given point."""
        if self.mask_layer is None:
            self.create_empty_mask()
            
        # Convert display coordinates to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        if img_x is None or img_y is None:
            return
            
        # Store the drawing mode (draw or erase)
        self.drawing_mode = self.draw_radio.isChecked()
        
        # Handle based on the current tool
        if self.current_tool == "brush":
            # Same as original brush tool behavior
            self.last_drawing_position = (img_x, img_y)
            
            draw_on_mask(
                self.mask_layer,
                img_x, img_y,
                self.brush_size,
                color=(0, 255, 0, 255),  # Green
                erase=not self.drawing_mode
            )
            
            # Clear the brush preview while actively drawing
            self.brush_preview_active = False
            
            # Initialize drawing throttling variables
            self.drawing_update_counter = 0
            self.drawing_update_threshold = 2  # Update display every N points
            
            # Update display
            self.update_display_with_mask()
        elif self.current_tool == "fill":
            # For fill tool, perform flood fill immediately
            self.perform_fill(img_x, img_y)
        else:
            # For all other shape tools, just record the starting position
            self.drawing_start_pos = (img_x, img_y)
            self.temp_drawing = self.mask_layer.copy()

    def continue_drawing(self, x1, y1, x2, y2):
        """Continue drawing on the mask between two points (optimized)."""
        # Convert display coordinates to image coordinates
        img_x2, img_y2 = self.convert_to_image_coordinates(x2, y2)
        
        if img_x2 is None or img_y2 is None:
            return
            
        # Handle based on the current tool
        if self.current_tool == "brush":
            # Original brush behavior
            if not hasattr(self, 'last_drawing_position') or self.last_drawing_position is None:
                self.last_drawing_position = (img_x2, img_y2)
                return
                
            img_x1, img_y1 = self.last_drawing_position
                
            # Get drawing/erasing mode
            self.drawing_mode = self.draw_radio.isChecked()
            
            # Calculate the distance between points
            distance = np.sqrt((img_x2 - img_x1)**2 + (img_y2 - img_y1)**2)
            
            # Determine intermediate points for a smooth line
            if distance < (self.brush_size * 0.5):
                draw_on_mask(
                    self.mask_layer,
                    img_x2, img_y2,
                    self.brush_size,
                    color=(0, 255, 0, 255),
                    erase=not self.drawing_mode
                )
            else:
                # Calculate points for a continuous line
                step_size = self.brush_size / 3
                num_steps = max(int(distance / step_size), 1)
                
                for i in range(num_steps + 1):
                    t = i / num_steps
                    x = int(img_x1 + t * (img_x2 - img_x1))
                    y = int(img_y1 + t * (img_y2 - img_y1))
                    
                    draw_on_mask(
                        self.mask_layer,
                        x, y,
                        self.brush_size,
                        color=(0, 255, 0, 255),
                        erase=not self.drawing_mode
                    )
            
            # Store the current position for next segment
            self.last_drawing_position = (img_x2, img_y2)
            
            # Throttle display updates
            self.drawing_update_counter += 1
            if self.drawing_update_counter >= self.drawing_update_threshold:
                self.drawing_update_counter = 0
                self.update_display_with_mask()
        elif self.current_tool != "fill" and self.drawing_start_pos is not None:
            # For shape tools, continuously update the preview
            self.update_shape_preview(img_x2, img_y2)

    def end_drawing(self):
        """End drawing on the mask."""
        # For brush tool
        if self.current_tool == "brush":
            # Reset drawing variables
            self.last_drawing_position = None
            self.drawing_update_counter = 0
            
            # Always update display at the end of drawing
            self.update_display_with_mask()
        
        # For shape tools (except fill which completes immediately)
        elif self.current_tool != "fill" and self.drawing_start_pos is not None:
            # Finalize the shape
            self.finalize_shape()
            self.drawing_start_pos = None
            self.temp_drawing = None
            
            # Update display
            self.update_display_with_mask()
        
        # Restore brush preview after drawing ends
        cursor_pos = self.image_label.mapFromGlobal(QCursor.pos())
        if self.image_label.rect().contains(cursor_pos):
            self.update_brush_preview(cursor_pos.x(), cursor_pos.y())

    def update_shape_preview(self, img_x2, img_y2):
        """Update the preview for shape drawing tools."""
        if self.drawing_start_pos is None or self.temp_drawing is None:
            return
        
        # Start with a clean copy of the mask (without the current shape)
        preview_mask = self.temp_drawing.copy()
        
        # Get the start position
        img_x1, img_y1 = self.drawing_start_pos
        
        # Get drawing color based on draw/erase mode
        color = (0, 255, 0, 255) if self.draw_radio.isChecked() else (0, 0, 0, 0)
        
        # Draw the appropriate shape based on the current tool
        if self.current_tool == "line":
            # Draw a line from start to current position
            cv2.line(
                preview_mask,
                (img_x1, img_y1),
                (img_x2, img_y2),
                color,
                thickness=self.brush_size
            )
        elif self.current_tool == "rectangle":
            # Draw a rectangle from start to current position
            cv2.rectangle(
                preview_mask,
                (img_x1, img_y1),
                (img_x2, img_y2),
                color,
                thickness=self.brush_size if self.brush_size > 1 else -1  # Filled if size is 1
            )
        elif self.current_tool == "circle":
            # Calculate radius for circle based on distance
            radius = int(np.sqrt((img_x2 - img_x1)**2 + (img_y2 - img_y1)**2))
            cv2.circle(
                preview_mask,
                (img_x1, img_y1),  # Center at start position
                radius,
                color,
                thickness=self.brush_size if self.brush_size > 1 else -1  # Filled if size is 1
            )
        elif self.current_tool == "ellipse":
            # Calculate width/height for ellipse
            width = abs(img_x2 - img_x1)
            height = abs(img_y2 - img_y1)
            center_x = (img_x1 + img_x2) // 2
            center_y = (img_y1 + img_y2) // 2
            cv2.ellipse(
                preview_mask,
                (center_x, center_y),
                (width // 2, height // 2),
                0,  # Angle
                0,  # Start angle
                360,  # End angle
                color,
                thickness=self.brush_size if self.brush_size > 1 else -1  # Filled if size is 1
            )
        
        # Temporarily update the mask layer with the preview
        self.mask_layer = preview_mask
        
        # Update display
        self.update_display_with_mask()

    def finalize_shape(self):
        """Finalize the shape being drawn."""
        # The shape is already in the mask_layer from update_shape_preview
        # Just make sure it's properly stored
        if self.mask_layer is not None:
            # No need to do anything else, the mask is already updated
            pass

    def perform_fill(self, img_x, img_y):
        """Perform flood fill on the mask."""
        if self.mask_layer is None:
            return
        
        # Get alpha channel for fill (this is where we're drawing/erasing)
        alpha_channel = self.mask_layer[:, :, 3].copy()
        
        # Get current value at fill point
        current_value = alpha_channel[img_y, img_x]
        
        # Get target value based on draw/erase mode
        target_value = 255 if self.draw_radio.isChecked() else 0
        
        # If current value is already the target, no need to fill
        if current_value == target_value:
            return
        
        # Perform flood fill
        mask = np.zeros((alpha_channel.shape[0] + 2, alpha_channel.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(alpha_channel, mask, (img_x, img_y), target_value, 
                      loDiff=0, upDiff=0, flags=8)
        
        # Update the mask's alpha channel
        if self.draw_radio.isChecked():
            # For draw mode, set RGB to green where alpha is filled
            self.mask_layer[:, :, 3] = alpha_channel
            filled_area = (alpha_channel == 255)
            self.mask_layer[:, :, 0][filled_area] = 0    # B
            self.mask_layer[:, :, 1][filled_area] = 255  # G
            self.mask_layer[:, :, 2][filled_area] = 0    # R
        else:
            # For erase mode, just set alpha to 0
            self.mask_layer[:, :, 3] = alpha_channel
        
        # Update display
        self.update_display_with_mask()

    def update_drawing_tool(self, checked):
        """Update the current drawing tool based on radio button selection."""
        if not checked:
            return
            
        if self.brush_tool_radio.isChecked():
            self.current_tool = "brush"
        elif self.line_tool_radio.isChecked():
            self.current_tool = "line"
        elif self.rectangle_tool_radio.isChecked():
            self.current_tool = "rectangle"
        elif self.circle_tool_radio.isChecked():
            self.current_tool = "circle"
        elif self.ellipse_tool_radio.isChecked():
            self.current_tool = "ellipse"
        elif self.fill_tool_radio.isChecked():
            self.current_tool = "fill"
        
        self.setStatusTip(f"Using {self.current_tool} tool")

    def toggle_detection_mode(self, use_color):
        """Toggle between color-based and edge detection modes."""
        # Enable/disable edge detection settings based on color detection mode
        for widget in self.edge_detection_widgets:
            widget.setEnabled(not use_color)
            
        # Also disable blur, edge margin, and min merge distance in color detection mode
        self.sliders["Blur"].setEnabled(not use_color)
        self.sliders["Edge Margin"].setEnabled(not use_color)
            
        # Update labels to reflect active/inactive state
        if use_color:
            self.edge_section_title.setText("Edge Detection Settings (Inactive):")
            self.edge_section_title.setStyleSheet("font-weight: bold; color: gray;")
        else:
            self.edge_section_title.setText("Edge Detection Settings:")
            self.edge_section_title.setStyleSheet("font-weight: bold; color: black;")
            
        # Update the detection if an image is loaded
        if self.current_image is not None:
            self.update_image()

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

    def start_selection(self, x, y):
        """Start a selection rectangle at the given coordinates."""
        # Convert to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        
        if img_x is None or img_y is None:
            return
            
        if self.deletion_mode_enabled:
            # Check if click is inside or on a contour - if so, handle as single click
            for i, contour in enumerate(self.current_contours):
                if cv2.pointPolygonTest(contour, (img_x, img_y), False) >= 0:
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
            # Check if click is inside a contour - if so, handle as single click for thinning
            for i, contour in enumerate(self.current_contours):
                if cv2.pointPolygonTest(contour, (img_x, img_y), False) >= 0:
                    self.handle_thinning_click(x, y)
                    return
                    
            # Otherwise, start a selection for thinning multiple contours
            self.selecting = True
            self.selection_start_img = (img_x, img_y)
            self.selection_current_img = (img_x, img_y)
            self.selected_contour_indices = []

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
        
        # Find and highlight contours within the selection
        self.selected_contour_indices = []
        
        for i, contour in enumerate(self.current_contours):
            # Check if contour is at least partially within selection rectangle
            for point in contour:
                px, py = point[0]
                if x1 <= px <= x2 and y1 <= py <= y2:
                    self.selected_contour_indices.append(i)
                    # Highlight with different colors based on mode
                    highlight_color = (0, 0, 255) if self.deletion_mode_enabled else (255, 0, 255)  # Red for delete, Magenta for thin
                    cv2.drawContours(self.processed_image, [contour], 0, highlight_color, 2)
                    break
                    
        # Display the updated image
        self.display_image(self.processed_image)

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
        
        # Enable color detection mode
        self.use_color_detection.setChecked(True)
        
        # Update the image with the new colors
        self.update_image()
    
    def delete_selected_contours(self):
        """Delete the selected contours from the current image."""
        if not self.selected_contour_indices:
            return
        
        # Delete selected contours
        for index in sorted(self.selected_contour_indices, reverse=True):
            if 0 <= index < len(self.current_contours):
                print(f"Deleting contour {index} (area: {cv2.contourArea(self.current_contours[index])})")
                self.current_contours.pop(index)
        
        # Clear selection and update display
        self.clear_selection()
        self.update_display_from_contours()

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
            
        # Find the contour under the cursor
        found_index = -1
        
        # First check if cursor is inside any contour
        for i, contour in enumerate(self.current_contours):
            if cv2.pointPolygonTest(contour, (img_x, img_y), False) >= 0:
                found_index = i
                break
                
        # If not inside any contour, check if cursor is on a line
        if found_index == -1:
            for i, contour in enumerate(self.current_contours):
                contour_points = contour.reshape(-1, 2)
                
                for j in range(len(contour_points)):
                    p1 = contour_points[j]
                    p2 = contour_points[(j + 1) % len(contour_points)]
                    distance = self.point_to_line_distance(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                    
                    # If point is close enough to a line segment
                    if distance < 5:  # Threshold for line detection (pixels)
                        found_index = i
                        break
                        
                if found_index != -1:
                    break
        
        # Update highlight if needed
        if found_index != self.highlighted_contour_index:
            self.highlighted_contour_index = found_index
            self.update_highlight()

    def clear_hover(self):
        """Clear any contour highlighting."""
        if self.highlighted_contour_index != -1:
            self.highlighted_contour_index = -1
            self.update_highlight()

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
        
        # Use the highlighted contour if available
        if self.highlighted_contour_index != -1:
            print(f"Deleting highlighted contour {self.highlighted_contour_index}")
            self.current_contours.pop(self.highlighted_contour_index)
            self.highlighted_contour_index = -1  # Reset highlight
            self.update_display_from_contours()
            return
            
        # First, check if click is inside any contour
        for i, contour in enumerate(self.current_contours):
            if cv2.pointPolygonTest(contour, (img_x, img_y), False) >= 0:
                # Click is inside this contour - delete it
                print(f"Deleting contour {i} (area: {cv2.contourArea(contour)})")
                self.current_contours.pop(i)
                self.update_display_from_contours()
                return
        
        # If not inside any contour, check if click is on a line
        for i, contour in enumerate(self.current_contours):
            # Check distance to each line segment in contour
            contour_points = contour.reshape(-1, 2)
            min_distance = float('inf')
            
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                distance = self.point_to_line_distance(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                min_distance = min(min_distance, distance)
                
                # If point is close enough to a line segment
                if min_distance < 5:  # Threshold for line detection (pixels)
                    print(f"Deleting contour {i} (line clicked)")
                    self.current_contours.pop(i)
                    self.update_display_from_contours()
                    return

    # Thinning methods
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

    def thin_selected_contours(self):
        """Thin the selected contours."""
        if not self.selected_contour_indices:
            return
        
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
        
        # Use the highlighted contour if available
        if self.highlighted_contour_index != -1:
            print(f"Thinning highlighted contour {self.highlighted_contour_index}")
            contour = self.current_contours[self.highlighted_contour_index]
            thinned_contour = self.thin_selected_contour(contour)
            self.current_contours[self.highlighted_contour_index] = thinned_contour
            self.highlighted_contour_index = -1  # Reset highlight
            self.update_display_from_contours()
            return
            
        # First, check if click is inside any contour
        for i, contour in enumerate(self.current_contours):
            if cv2.pointPolygonTest(contour, (img_x, img_y), False) >= 0:
                # Click is inside this contour - thin it
                print(f"Thinning contour {i}")
                thinned_contour = self.thin_selected_contour(contour)
                self.current_contours[i] = thinned_contour
                self.update_display_from_contours()
                return

    def thin_contour(self, contour):
        """Thin a single contour using morphological thinning."""
        # Create a mask for the contour
        mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

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

    def open_image(self):
        """Open an image file and prepare scaled versions for processing."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # Load the original full-resolution image
            self.original_image = load_image(file_path)
            
            # Create a scaled down version for processing if needed
            self.current_image, self.scale_factor = self.create_working_image(self.original_image)
            
            print(f"Image loaded: Original size {self.original_image.shape}, Working size {self.current_image.shape}, Scale factor {self.scale_factor}")
            
            # Reset the mask layer when loading a new image to prevent dimension mismatch
            self.mask_layer = None
            
            # Reset button states when loading a new image
            self.export_foundry_button.setEnabled(False)
            self.save_foundry_button.setEnabled(False)
            self.cancel_foundry_button.setEnabled(False)
            self.copy_foundry_button.setEnabled(False)
            
            # Update the display
            self.update_image()

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

    def save_image(self):
        """Save the processed image at full resolution."""
        if self.original_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
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
                    cv2.imwrite(file_path, result)
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
        if self.current_image is not None and self.use_color_detection.isChecked():
            self.update_image()
    
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
    
    def add_wall_color_to_list(self, color, threshold=10.0):
        """Add a color with threshold to the wall colors list."""
        item = QListWidgetItem()
        self.update_color_list_item(item, color, threshold)
        self.wall_colors_list.addItem(item)
        return item
    
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

    def update_image(self):
        """Update the displayed image based on the current settings."""
        if self.current_image is None:
            return

        # Get slider values
        min_area = self.sliders["Min Area"].value()
        blur = self.sliders["Blur"].value()
        
        # Handle special case for blur=1 (no blur) and ensure odd values
        if blur > 1 and blur % 2 == 0:
            blur += 1  # Ensure kernel size is odd when > 1
        
        canny1 = self.sliders["Canny1"].value()
        canny2 = self.sliders["Canny2"].value()
        edge_margin = self.sliders["Edge Margin"].value()
        
        # Get min_merge_distance as a float value
        min_merge_distance = self.sliders["Min Merge Distance"].value() * 0.1
        
        # Adjust area thresholds based on scale factor (for downscaled processing)
        if self.scale_factor != 1.0:
            working_min_area = int(min_area * self.scale_factor * self.scale_factor)
        else:
            working_min_area = min_area
        
        # Set up color detection parameters with per-color thresholds
        wall_colors_with_thresholds = None
        default_threshold = 0
        
        if self.use_color_detection.isChecked() and self.wall_colors_list.count() > 0:
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
        print(f"Parameters: min_area={min_area} (working: {working_min_area}), "
              f"blur={blur}, canny1={canny1}, canny2={canny2}, edge_margin={edge_margin}")

        # Process the image directly with detect_walls
        contours = detect_walls(
            self.current_image,
            min_contour_area=working_min_area,
            max_contour_area=None,
            blur_kernel_size=blur,
            canny_threshold1=canny1,
            canny_threshold2=canny2,
            edge_margin=edge_margin,
            wall_colors=wall_colors_with_thresholds,  # Now includes per-color thresholds
            color_threshold=default_threshold  # Kept for backward compatibility
        )
        
        print(f"Detected {len(contours)} contours before merging")

        # Merge before Min Area if specified
        if self.merge_contours.isChecked():
            contours = merge_contours(
                self.current_image, 
                contours, 
                min_merge_distance=min_merge_distance
            )
            print(f"After merge before min area: {len(contours)} contours")
        
        # Filter contours by area BEFORE splitting edges
        contours = [c for c in contours if cv2.contourArea(c) >= working_min_area]
        print(f"After min area filter: {len(contours)} contours")

        # Split contours that touch image edges AFTER area filtering, but only if not in color detection mode
        if not self.use_color_detection.isChecked():
          split_contours = split_edge_contours(self.current_image, contours)

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
            self.processed_image = self.current_image.copy()
        else:
            # Draw merged contours
            self.processed_image = draw_walls(self.current_image, contours)

        # Save the original image for highlighting
        if self.processed_image is not None:
            self.original_processed_image = self.processed_image.copy()
            
        # Clear any existing selection when re-detecting
        self.clear_selection()
        
        # Reset highlighted contour when re-detecting
        self.highlighted_contour_index = -1

        # Convert to QPixmap and display
        self.display_image(self.processed_image)

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
            
        if self.mask_layer is not None and self.edit_mask_mode_enabled:
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
            if self.scale_factor != 1.0:
                walls_to_export = self.scale_contours_to_original(walls_to_export, self.scale_factor)
        else:
            print("No walls to export.")
            return
            
        # Create a single dialog to gather all export parameters
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Parameters")
        layout = QVBoxLayout(dialog)

        # Simplification tolerance
        tolerance_label = QLabel("Simplification Tolerance (0 = maintain details):")
        tolerance_input = QDoubleSpinBox()
        tolerance_input.setRange(0.0, 1.0)
        tolerance_input.setDecimals(4)
        tolerance_input.setSingleStep(0.0005)
        tolerance_input.setValue(0.0005)
        layout.addWidget(tolerance_label)
        layout.addWidget(tolerance_input)

        # Maximum wall length
        max_length_label = QLabel("Maximum Wall Segment Length (pixels):")
        max_length_input = QSpinBox()
        max_length_input.setRange(5, 500)
        max_length_input.setSingleStep(5)
        max_length_input.setValue(50)
        layout.addWidget(max_length_label)
        layout.addWidget(max_length_input)

        # Maximum number of walls
        max_walls_label = QLabel("Maximum Number of Walls:")
        max_walls_input = QSpinBox()
        max_walls_input.setRange(100, 20000)
        max_walls_input.setSingleStep(100)
        max_walls_input.setValue(5000)
        layout.addWidget(max_walls_label)
        layout.addWidget(max_walls_input)

        # Point merge distance
        merge_distance_label = QLabel("Point Merge Distance (pixels):")
        merge_distance_input = QDoubleSpinBox()
        merge_distance_input.setRange(0.0, 100.0)
        merge_distance_input.setDecimals(1)
        merge_distance_input.setSingleStep(1.0)
        merge_distance_input.setValue(25.0)
        layout.addWidget(merge_distance_label)
        layout.addWidget(merge_distance_input)

        # Angle tolerance
        angle_tolerance_label = QLabel("Angle Tolerance (degrees):")
        angle_tolerance_input = QDoubleSpinBox()
        angle_tolerance_input.setRange(0.0, 30.0)
        angle_tolerance_input.setDecimals(2)
        angle_tolerance_input.setSingleStep(0.5)
        angle_tolerance_input.setValue(1.0)
        layout.addWidget(angle_tolerance_label)
        layout.addWidget(angle_tolerance_input)

        # Maximum gap
        max_gap_label = QLabel("Maximum Gap (coords):")
        max_gap_input = QDoubleSpinBox()
        max_gap_input.setRange(0.0, 50.0)
        max_gap_input.setDecimals(1)
        max_gap_input.setSingleStep(1.0)
        max_gap_input.setValue(10.0)
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
        grid_size_layout.addWidget(grid_size_label)
        grid_size_layout.addWidget(grid_size_input)
        layout.addLayout(grid_size_layout)

        # Allow half grid checkbox
        allow_half_grid = QCheckBox("Allow Half-Grid Positions")
        allow_half_grid.setChecked(False)
        allow_half_grid.setToolTip("If checked, walls can snap to half-grid positions, otherwise only to full grid intersections")
        layout.addWidget(allow_half_grid)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

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

        # Switch to deletion mode for less interference with the preview
        self.color_selection_mode_radio.setChecked(True)
        
        # Generate walls for preview
        self.preview_foundry_walls()

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
            # Extract contours from the mask
            mask = params['walls_to_export']
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    def set_controls_enabled(self, enabled, color_detection_mode=False):
        """Enable or disable detection controls based on preview state."""
        # Disable/enable all sliders
        for slider_name, slider in self.sliders.items():
            slider.setEnabled(enabled)
            
        # Disable/enable color detection checkbox
        self.use_color_detection.setEnabled(enabled)
        
        if not color_detection_mode:
            # Disable/enable merge contours checkbox
            self.merge_contours.setEnabled(enabled)
        
        # Disable/enable high-res checkbox
        self.high_res_checkbox.setEnabled(enabled)
        
        # Disable/enable color management
        self.add_color_button.setEnabled(enabled)
        self.remove_color_button.setEnabled(enabled)
        self.wall_colors_list.setEnabled(enabled)
        
        # If re-enabling, respect color detection mode
        if enabled and self.use_color_detection.isChecked():
            # Re-apply color detection limitations
            self.toggle_detection_mode(True)

    def update_target_width(self, value):
        """Update the target width parameter for thinning."""
        self.target_width = value
        self.target_width_value.setText(str(value))
    
    def update_max_iterations(self, value):
        """Update the max iterations parameter for thinning."""
        self.max_iterations = value
        self.max_iterations_value.setText(str(value))

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

    def resizeEvent(self, event):
        """Handle window resize events to update the image display."""
        super().resizeEvent(event)
        
        # If we have a current image displayed, update it to fit the new window size
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            self.display_image(self.processed_image)
            
        # If we're in foundry preview mode, redraw the preview
        if hasattr(self, 'foundry_preview_active') and self.foundry_preview_active and self.foundry_walls_preview:
            self.display_foundry_preview()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WallDetectionApp()
    window.show()
    sys.exit(app.exec())