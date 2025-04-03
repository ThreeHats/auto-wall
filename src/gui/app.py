import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QWidget, 
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup, QColorDialog, QListWidget, QListWidgetItem,
    QScrollArea, QSizePolicy, QDialog, QDialogButtonBox, QFrame, QSpinBox
)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QMouseEvent
import cv2
import numpy as np
import math
from sklearn.cluster import KMeans

from src.wall_detection.detector import detect_walls, draw_walls, merge_contours, split_edge_contours
from src.wall_detection.image_utils import load_image, save_image, convert_to_rgb
from src.wall_detection.mask_editor import create_mask_from_contours, blend_image_with_mask, draw_on_mask


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
            # Just hovering in deletion mode
            elif self.parent_app.deletion_mode_enabled:
                self.parent_app.handle_hover(pos.x(), pos.y())
                
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
                
                # Clear selection points
                self.selection_start = None
                self.selection_current = None
                
        super().mouseReleaseEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leaving the widget."""
        if self.parent_app and self.parent_app.deletion_mode_enabled:
            self.parent_app.clear_hover()
        super().leaveEvent(event)


class WallDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto-Wall: Battle Map Wall Detection")
        self.setGeometry(100, 100, 1200, 800)  # Wider window to accommodate side-by-side layout

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
        self.add_slider("Min Area", 0, 10000, 100)
        self.add_slider("Blur", 1, 21, 5, step=2)
        self.add_slider("Canny1", 0, 255, 50)
        self.add_slider("Canny2", 0, 255, 150)
        self.add_slider("Edge Margin", 0, 50, 5)
        
        # Use a scaling factor of 10 for float values (0 to 10.0 with 0.1 precision)
        self.add_slider("Min Merge Distance", 0, 100, 30, scale_factor=0.1)  # Default 3.0

        # Mode selection (Detection/Deletion/Color Selection/Edit Mask)
        self.mode_layout = QHBoxLayout()
        self.controls_layout.addLayout(self.mode_layout)
        
        self.mode_label = QLabel("Mode:")
        self.mode_layout.addWidget(self.mode_label)
        
        self.detection_mode_radio = QRadioButton("Detection")
        self.deletion_mode_radio = QRadioButton("Deletion")
        self.color_selection_mode_radio = QRadioButton("Color Pick")
        self.edit_mask_mode_radio = QRadioButton("Edit Mask")
        self.detection_mode_radio.setChecked(True)
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.detection_mode_radio)
        self.mode_group.addButton(self.deletion_mode_radio)
        self.mode_group.addButton(self.color_selection_mode_radio)
        self.mode_group.addButton(self.edit_mask_mode_radio)
        
        # Add mode options vertically to save horizontal space
        self.mode_radios_layout = QVBoxLayout()
        self.mode_radios_layout.addWidget(self.detection_mode_radio)
        self.mode_radios_layout.addWidget(self.deletion_mode_radio)
        self.mode_radios_layout.addWidget(self.color_selection_mode_radio)
        self.mode_radios_layout.addWidget(self.edit_mask_mode_radio)
        self.mode_layout.addLayout(self.mode_radios_layout)
        
        # Connect mode radio buttons
        self.detection_mode_radio.toggled.connect(self.toggle_mode)
        self.deletion_mode_radio.toggled.connect(self.toggle_mode)
        self.color_selection_mode_radio.toggled.connect(self.toggle_mode)
        self.edit_mask_mode_radio.toggled.connect(self.toggle_mode)
        
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
        
        # Bake button
        self.bake_button = QPushButton("Bake Contours to Mask")
        self.bake_button.clicked.connect(self.bake_contours_to_mask)
        self.mask_edit_layout.addWidget(self.bake_button)
        
        self.controls_layout.addWidget(self.mask_edit_options)
        self.mask_edit_options.setVisible(False)
        
        # Deletion mode is initially disabled
        self.deletion_mode_enabled = False
        self.color_selection_mode_enabled = False
        self.edit_mask_mode_enabled = False

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
        """Toggle between detection, deletion, color selection, and edit mask modes."""
        self.deletion_mode_enabled = self.deletion_mode_radio.isChecked()
        self.color_selection_mode_enabled = self.color_selection_mode_radio.isChecked()
        self.edit_mask_mode_enabled = self.edit_mask_mode_radio.isChecked()
        
        # Show/hide color selection options
        self.color_selection_options.setVisible(self.color_selection_mode_enabled)
        
        # Show/hide mask edit options
        self.mask_edit_options.setVisible(self.edit_mask_mode_enabled)
        
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

    def start_drawing(self, x, y):
        """Start drawing on the mask at the given point."""
        if self.mask_layer is None:
            self.create_empty_mask()
            
        # Convert display coordinates to image coordinates
        img_x, img_y = self.convert_to_image_coordinates(x, y)
        if img_x is None or img_y is None:
            return
            
        # Get drawing/erasing mode
        self.drawing_mode = self.draw_radio.isChecked()
        
        # Store the last drawing position for path tracking
        self.last_drawing_position = (img_x, img_y)
        
        # Draw on the mask
        draw_on_mask(
            self.mask_layer,
            img_x,
            img_y,
            self.brush_size,
            color=(0, 255, 0, 255),  # Green
            erase=not self.drawing_mode
        )
        
        # Initialize drawing throttling variables
        self.drawing_update_counter = 0
        self.drawing_update_threshold = 2  # Update display every N points
        
        # Update display
        self.update_display_with_mask()

    def continue_drawing(self, x1, y1, x2, y2):
        """Continue drawing on the mask between two points (optimized)."""
        # Convert display coordinates to image coordinates
        img_x2, img_y2 = self.convert_to_image_coordinates(x2, y2)
        
        if img_x2 is None or img_y2 is None:
            return
            
        # Check if last_drawing_position exists
        if not hasattr(self, 'last_drawing_position') or self.last_drawing_position is None:
            # If it doesn't exist yet, initialize it with the current position
            self.last_drawing_position = (img_x2, img_y2)
            return
            
        img_x1, img_y1 = self.last_drawing_position
            
        # Get drawing/erasing mode
        self.drawing_mode = self.draw_radio.isChecked()
        
        # Calculate the distance between points
        distance = np.sqrt((img_x2 - img_x1)**2 + (img_y2 - img_y1)**2)
        
        # Determine number of intermediate points based on distance and brush size
        # For small distances relative to brush size, just draw at the end point
        if distance < (self.brush_size * 0.5):
            draw_on_mask(
                self.mask_layer,
                img_x2,
                img_y2,
                self.brush_size,
                color=(0, 255, 0, 255),
                erase=not self.drawing_mode
            )
        else:
            # Calculate number of points needed for a continuous line
            # Use brush_size/3 for smoother lines with less computation
            step_size = self.brush_size / 3
            num_steps = max(int(distance / step_size), 1)
            
            # Calculate and draw intermediate points along the line
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
        
        # Throttle display updates for smoother performance
        self.drawing_update_counter += 1
        if self.drawing_update_counter >= self.drawing_update_threshold:
            self.drawing_update_counter = 0
            self.update_display_with_mask()

    def end_drawing(self):
        """End drawing on the mask."""
        # Reset drawing variables
        self.last_drawing_position = None
        self.drawing_update_counter = 0
        
        # Always update display at the end of drawing
        self.update_display_with_mask()

    def toggle_detection_mode(self, use_color):
        """Toggle between color-based and edge detection modes."""
        # Enable/disable edge detection settings based on color detection mode
        for widget in self.edge_detection_widgets:
            widget.setEnabled(not use_color)
            
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
                    # Highlight this contour
                    cv2.drawContours(self.processed_image, [contour], 0, (0, 0, 255), 2)
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
            highlight_color = (0, 0, 255)  # Highlight in red
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
        rgb_image = convert_to_rgb(image)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
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
            
            self.update_image()

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WallDetectionApp()
    window.show()
    sys.exit(app.exec())