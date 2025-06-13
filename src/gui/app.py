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
from src.wall_detection.image_utils import load_image, convert_to_rgb
from src.wall_detection.mask_editor import create_mask_from_contours, blend_image_with_mask, draw_on_mask, export_mask_to_foundry_json, contours_to_foundry_walls, thin_contour
from src.utils.update_checker import check_for_updates
from src.gui.drawing_tools import DrawingTools
from src.gui.preset_manager import PresetManager
from src.core.image_processor import ImageProcessor
from src.core.selection import SelectionManager
from src.gui.image_viewer import InteractiveImageLabel
from src.core.contour_processor import ContourProcessor
from src.gui.detection_panel import DetectionPanel
from src.gui.export_panel import ExportPanel
from src.core.mask_processor import MaskProcessor



class WallDetectionApp(QMainWindow):
    def __init__(self, version="0.9.0", github_repo="ThreeHats/auto-wall"):
        super().__init__()
        self.app_version = version
        self.github_repo = github_repo
        self.update_available = False
        self.update_url = ""

        self.drawing_tools = DrawingTools(self)
        self.preset_manager = PresetManager(self)
        self.image_processor = ImageProcessor(self)
        self.selection_manager = SelectionManager(self)
        self.contour_processor = ContourProcessor(self)
        self.detection_panel = DetectionPanel(self)
        self.export_panel = ExportPanel(self)
        self.mask_processor = MaskProcessor(self)
        
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
        self.edge_detection_radio.toggled.connect(self.detection_panel.toggle_detection_mode_radio)
        self.color_detection_radio.toggled.connect(self.detection_panel.toggle_detection_mode_radio)
        
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
        self.deletion_mode_radio.toggled.connect(self.detection_panel.toggle_mode)
        self.thin_mode_radio.toggled.connect(self.detection_panel.toggle_mode)
        self.color_selection_mode_radio.toggled.connect(self.detection_panel.toggle_mode)
        self.edit_mask_mode_radio.toggled.connect(self.detection_panel.toggle_mode)

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
        self.target_width_slider.valueChanged.connect(self.detection_panel.update_target_width)
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
        self.max_iterations_slider.valueChanged.connect(self.detection_panel.update_max_iterations)
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
        self.min_area_percentage_radio.toggled.connect(self.detection_panel.toggle_min_area_mode)
        self.min_area_pixels_radio.toggled.connect(self.detection_panel.toggle_min_area_mode)
        
        self.controls_layout.addLayout(self.min_area_mode_layout)
        
        # Min Area is now a percentage (0.0001% to 1% of image area) or pixels (1 to 1000)
        self.detection_panel.add_slider("Min Area", 1, 25000, 100, scale_factor=0.001)  # Default 0.1%
        self.detection_panel.add_slider("Smoothing", 1, 21, 5, step=2)  # Changed from "Blur"
        self.detection_panel.add_slider("Edge Sensitivity", 0, 255, 255)  # Changed from "Canny1"
        self.detection_panel.add_slider("Edge Threshold", 0, 255, 106)  # Changed from "Canny2"
        self.detection_panel.add_slider("Edge Margin", 0, 50, 0)
        
        # Checkboxes for merge options
        self.merge_options_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.merge_options_layout)

        self.merge_contours = QCheckBox("Merge Contours")
        self.merge_contours.setChecked(False)
        self.merge_options_layout.addWidget(self.merge_contours)

        # Use a scaling factor of 10 for float values (0 to 10.0 with 0.1 precision)
        self.detection_panel.add_slider("Min Merge Distance", 0, 100, 5, scale_factor=0.1)  # Default 0.5
        
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
        self.remove_hatching_checkbox.toggled.connect(self.detection_panel.toggle_hatching_removal)
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
        self.hatching_color_button.clicked.connect(self.detection_panel.select_hatching_color)
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
        self.hatching_threshold_slider.valueChanged.connect(self.detection_panel.update_hatching_threshold)
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
        self.hatching_width_slider.valueChanged.connect(self.detection_panel.update_hatching_width)
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
        self.wall_colors_list.itemClicked.connect(self.detection_panel.select_color)
        self.wall_colors_list.itemDoubleClicked.connect(self.detection_panel.edit_wall_color)
        self.wall_colors_layout.addWidget(self.wall_colors_list)
        
        # Buttons for color management
        self.color_buttons_layout = QHBoxLayout()
        self.wall_colors_layout.addLayout(self.color_buttons_layout)
        
        self.add_color_button = QPushButton("Add Color")
        self.add_color_button.clicked.connect(self.detection_panel.add_wall_color)
        self.color_buttons_layout.addWidget(self.add_color_button)
        
        self.remove_color_button = QPushButton("Remove Color")
        self.remove_color_button.clicked.connect(self.detection_panel.remove_wall_color)
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
        self.threshold_slider.valueChanged.connect(self.detection_panel.update_selected_threshold)
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
        self.high_res_checkbox.stateChanged.connect(self.image_processor.reload_working_image)
        self.controls_layout.addWidget(self.high_res_checkbox)
        
        # Group edge detection settings
        self.edge_detection_widgets = []
        self.edge_detection_widgets.append(self.sliders["Edge Sensitivity"])
        self.edge_detection_widgets.append(self.sliders["Edge Threshold"])
        
        # State for color detection - now a list of colors
        self.wall_colors = []  # List to store QColor objects
        
        # Add a default black color with default threshold
        self.detection_panel.add_wall_color_to_list(QColor(0, 0, 0), 10.0)
        
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
        self.detection_preset_combo.currentIndexChanged.connect(self.preset_manager.load_detection_preset_selected)
        presets_line1_layout.addWidget(self.detection_preset_combo, 1) # Stretch 1 to take available space

        # Second line: Buttons
        presets_line2_layout = QHBoxLayout()
        presets_line2_layout.addStretch(1) # Add stretch to push buttons to the right

        save_preset_button = QPushButton("Save Preset")
        save_preset_button.setObjectName("save_preset_button")
        save_preset_button.setToolTip("Save current detection settings as a new preset")
        save_preset_button.clicked.connect(self.preset_manager.save_detection_preset)
        presets_line2_layout.addWidget(save_preset_button) # Stretch 0

        manage_presets_button = QPushButton("Manage")
        manage_presets_button.setObjectName("manage_presets_button")
        manage_presets_button.setToolTip("Manage saved detection presets")
        manage_presets_button.clicked.connect(self.preset_manager.manage_detection_presets)
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
        self.open_button.clicked.connect(self.image_processor.open_image)
        self.buttons_layout.addWidget(self.open_button)

        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.image_processor.save_image)
        self.buttons_layout.addWidget(self.save_button)

        # Add URL image loading button next to Open Image button
        self.url_button = QPushButton("Load from URL")
        self.url_button.clicked.connect(self.image_processor.load_image_from_url)
        self.url_button.setToolTip("Load image from URL in clipboard")
        self.buttons_layout.addWidget(self.url_button)
        
        # Create a layout for the wall action buttons
        self.wall_actions_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.wall_actions_layout)

        # Move the Bake button to the main sidebar
        self.bake_button = QPushButton("Bake Contours to Mask")
        self.bake_button.clicked.connect(self.mask_processor.bake_contours_to_mask)
        self.wall_actions_layout.addWidget(self.bake_button)
        
        # Add Undo button
        self.mask_processor.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.mask_processor.undo_button.clicked.connect(self.mask_processor.undo)
        self.mask_processor.undo_button.setEnabled(False)  # Initially disabled until actions are performed
        self.wall_actions_layout.addWidget(self.mask_processor.undo_button)
        
        # Move the Export to Foundry button
        self.export_foundry_button = QPushButton("Export to Foundry VTT")
        self.export_foundry_button.clicked.connect(self.export_panel.export_to_foundry_vtt)
        self.export_foundry_button.setToolTip("Export walls as JSON for Foundry VTT")
        self.export_foundry_button.setEnabled(False)  # Initially disabled
        self.wall_actions_layout.addWidget(self.export_foundry_button)
        
        # Move the Save Foundry Walls and Cancel Preview buttons
        self.save_foundry_button = QPushButton("Save Foundry Walls")
        self.save_foundry_button.clicked.connect(self.export_panel.save_foundry_preview)
        self.save_foundry_button.setToolTip("Save the previewed walls to Foundry VTT JSON file")
        self.save_foundry_button.setEnabled(False)
        self.wall_actions_layout.addWidget(self.save_foundry_button)
        
        self.cancel_foundry_button = QPushButton("Cancel Preview")
        self.cancel_foundry_button.clicked.connect(self.export_panel.cancel_foundry_preview)
        self.cancel_foundry_button.setToolTip("Return to normal view")
        self.cancel_foundry_button.setEnabled(False)
        self.wall_actions_layout.addWidget(self.cancel_foundry_button)
        
        # Add a copy to clipboard button next to Save Foundry Walls
        self.copy_foundry_button = QPushButton("Copy to Clipboard")
        self.copy_foundry_button.clicked.connect(self.export_panel.copy_foundry_to_clipboard)
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
        self.mask_processor.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.mask_processor.undo_shortcut.activated.connect(self.mask_processor.undo)

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
        self.preset_manager.load_presets_from_file() # Load presets on startup

        # Populate preset combo box AFTER loading presets
        self.preset_manager.update_detection_preset_combo()

        # Apply stylesheet at the end
        self.apply_stylesheet()

        # Check for updates
        self.check_for_updates()

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
        self.image_processor.display_image(self.processed_image)

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
        self.selection_manager.clear_selection()
        
        # Save state before deleting
        self.mask_processor.save_state()
        
        # Use the highlighted contour if available
        if self.highlighted_contour_index != -1:
            print(f"Deleting highlighted contour {self.highlighted_contour_index}")
            self.current_contours.pop(self.highlighted_contour_index)
            self.highlighted_contour_index = -1  # Reset highlight
            self.contour_processor.update_display_from_contours()
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
            self.contour_processor.update_display_from_contours()
            return


    # thinning


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
        self.selection_manager.clear_selection()
        
        # Save state before modifying
        self.mask_processor.save_state()
        
        # Use the highlighted contour if available
        if self.highlighted_contour_index != -1:
            print(f"Thinning highlighted contour {self.highlighted_contour_index}")
            contour = self.current_contours[self.highlighted_contour_index]
            thinned_contour = self.contour_processor.thin_selected_contour(contour)
            self.current_contours[self.highlighted_contour_index] = thinned_contour
            self.highlighted_contour_index = -1  # Reset highlight
            self.contour_processor.update_display_from_contours()
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
            thinned_contour = self.contour_processor.thin_selected_contour(contour)
            self.current_contours[closest_contour_index] = thinned_contour
            self.contour_processor.update_display_from_contours()
            return


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
    def resizeEvent(self, event):
        """Handle window resize events to update the image display."""
        super().resizeEvent(event)
        
        # If we have a current image displayed, update it to fit the new window size
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            self.image_processor.display_image(self.processed_image)
            
        # If we're in foundry preview mode, redraw the preview
        if hasattr(self, 'foundry_preview_active') and self.foundry_preview_active and self.foundry_walls_preview:
            self.export_panel.display_foundry_preview()
        
        # Update the position of the update notification
        if hasattr(self, 'update_notification'):
            self.update_notification.setGeometry(
                self.width() - 250, 10, 240, 40
            )

    # app
    def keyPressEvent(self, event):
        """Handle key press events."""
        # Add debugging for Ctrl+Z
        if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            print("Ctrl+Z detected via keyPressEvent")
            self.mask_processor.undo()
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WallDetectionApp()
    window.show()
    sys.exit(app.exec())