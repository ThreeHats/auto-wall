import sys
import os
import cv2

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QWidget, 
    QCheckBox, QRadioButton, QButtonGroup, QListWidget,
    QScrollArea, QSizePolicy, QDialog, QFrame, QSpinBox,
    QGridLayout, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QColor, QGuiApplication, QKeySequence, QShortcut
from collections import deque

from src.utils.update_checker import check_for_updates, open_update_url
from src.gui.drawing_tools import DrawingTools
from src.gui.preset_manager import PresetManager
from src.core.image_processor import ImageProcessor
from src.core.selection import SelectionManager
from src.gui.image_viewer import InteractiveImageLabel
from src.core.contour_processor import ContourProcessor
from src.gui.detection_panel import DetectionPanel
from src.gui.export_panel import ExportPanel
from src.core.mask_processor import MaskProcessor

from src.utils.ui_helpers import apply_stylesheet



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
        # Set up unified undo shortcut
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.unified_undo)
        
        # Zoom shortcuts
        self.zoom_in_shortcut = QShortcut(QKeySequence("Ctrl++"), self)
        self.zoom_in_shortcut.activated.connect(self.zoom_in)
        
        self.zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        self.zoom_out_shortcut.activated.connect(self.zoom_out)
        
        # Reset view shortcut
        self.reset_view_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        self.reset_view_shortcut.activated.connect(self.reset_view)
        
        # Fit to window shortcut
        self.fit_to_window_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        self.fit_to_window_shortcut.activated.connect(self.fit_to_window)

        # Screen setup
        self.setWindowTitle(f"Auto-Wall: Battle Map Wall Detection v{self.app_version}")
        screen = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        self.showMaximized()
        
        # Set up the main layout and widgets
        self.initialize_state()
        self.setup_ui()

        # Load presets on startup
        self.preset_manager.load_presets_from_file()
        # Populate preset combo box after loading presets
        self.preset_manager.update_detection_preset_combo()

        apply_stylesheet(self)
        check_for_updates(self)
        
    def initialize_state(self):
        self.original_image = None  # Original full-size image
        self.current_image = None   # Working image
        self.max_working_dimension = 1500  # Maximum dimension for processing
        self.scale_factor = 1.0     # Scale factor between original and working image
        self.processed_image = None
        self.current_contours = []
        self.display_scale_factor = 1.0
        self.display_offset = (0, 0)
        self.sliders = {}

        # Modes
        self.deletion_mode_enabled = True
        self.color_selection_mode_enabled = False
        self.edit_mask_mode_enabled = False
        self.thin_mode_enabled = False
        
        # Hover highlighting
        self.highlighted_contour_index = -1  # -1 means no contour is highlighted
        self.original_processed_image = None  # Store original image without highlight

        # Thinning
        self.target_width = 5
        self.max_iterations = 3
        
        # Hatching removal settings
        self.hatching_threshold = 10.0
        self.hatching_width = 3
        self.hatching_color = QColor(0, 0, 0)
        # Drag selection
        self.selecting = False
        self.selection_start_img = None
        self.selection_current_img = None
        self.selected_contour_indices = []

        # Color selection
        self.wall_colors = []  # List to store QColor objects
        self.selecting_colors = False
        self.color_selection_start = None
        self.color_selection_current = None
        self.selected_color_item = None

        # Preview state
        self.brush_preview_active = False
        self.last_preview_image = None
        self.uvtt_preview_active = False
        
        # UVTT wall editing states
        self.uvtt_draw_mode = False      # Drawing new walls
        self.uvtt_edit_mode = False      # Moving wall points
        self.uvtt_delete_mode = False    # Deleting walls
        self.uvtt_portal_mode = False    # Drawing new portals/doors
        self.selected_wall_index = -1    # Currently selected wall
        self.selected_point_index = -1   # Currently selected point (0=start, 1=end)
        self.drawing_new_wall = False    # Flag for currently drawing a new wall
        self.drawing_new_portal = False  # Flag for currently drawing a new portal
        self.new_wall_start = None       # Start point for new wall being drawn
        self.new_wall_end = None         # End point for new wall being drawn
        self.new_portal_start = None     # Start point for new portal being drawn
        self.new_portal_end = None       # End point for new portal being drawn
        self.selecting_walls = False     # Flag for wall selection box
        self.wall_selection_start = None # Start point of wall selection box
        self.wall_selection_current = None # Current point of wall selection box
        self.selected_wall_indices = []  # List of selected wall indices
        self.selected_points = []        # List of selected wall points (wall_idx, point_idx) tuples
        self.multi_wall_drag = False     # Flag for multi-wall drag operation
        self.multi_wall_drag_start = None # Start position for multi-wall drag
        self.dragging_from_line = False  # Flag for dragging from wall line rather than endpoints
        
        # Grid overlay
        self.grid_overlay_enabled = False
        self.pixels_per_grid = 72  # Default grid size
        
        # History tracking for undo feature
        self.history = deque(maxlen=5)  # Store up to 5 previous states

    def setup_ui(self):
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

        # Add a checkbox for high-resolution processing
        self.high_res_checkbox = QCheckBox("Process at Full Resolution")
        self.high_res_checkbox.setChecked(False)
        self.high_res_checkbox.setToolTip("Process at full resolution (slower but more accurate)")
        self.high_res_checkbox.stateChanged.connect(self.image_processor.reload_working_image)
        self.controls_layout.addWidget(self.high_res_checkbox)

        # Add grid overlay controls
        self.grid_section = QWidget()
        self.grid_section_layout = QVBoxLayout(self.grid_section)
        self.grid_section_layout.setContentsMargins(0, 0, 0, 0)
        
        # Grid overlay checkbox
        self.grid_overlay_checkbox = QCheckBox("Show Grid Overlay")
        self.grid_overlay_checkbox.setChecked(False)
        self.grid_overlay_checkbox.setToolTip("Display a grid overlay on the image")
        self.grid_overlay_checkbox.stateChanged.connect(self.toggle_grid_overlay)
        self.grid_section_layout.addWidget(self.grid_overlay_checkbox)
        
        # Grid size control
        self.grid_size_layout = QHBoxLayout()
        self.grid_size_label = QLabel("Grid Size (pixels):")
        self.grid_size_layout.addWidget(self.grid_size_label)
        
        self.grid_size_input = QSpinBox()
        self.grid_size_input.setRange(10, 500)
        self.grid_size_input.setSingleStep(5)
        self.grid_size_input.setValue(70)
        self.grid_size_input.setToolTip("Size of grid squares in pixels")
        self.grid_size_input.valueChanged.connect(self.update_grid_size)
        self.grid_size_layout.addWidget(self.grid_size_input)
        
        self.grid_section_layout.addLayout(self.grid_size_layout)
        self.controls_layout.addWidget(self.grid_section)

        # Group edge detection settings
        self.edge_detection_widgets = []
        self.edge_detection_widgets.append(self.sliders["Edge Sensitivity"])
        self.edge_detection_widgets.append(self.sliders["Edge Threshold"])

        # Add a default black color with default threshold
        self.detection_panel.add_wall_color_to_list(QColor(0, 0, 0), 10.0)

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
        
        # Add unified Undo button
        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(self.unified_undo)
        self.undo_button.setEnabled(False)  # Initially disabled until actions are performed
        self.wall_actions_layout.addWidget(self.undo_button)
        # Keep a reference in mask_processor for compatibility
        self.mask_processor.undo_button = self.undo_button
        
        # Move the Export button
        self.export_uvtt_button = QPushButton("Export")
        self.export_uvtt_button.clicked.connect(self.export_panel.export_to_uvtt)
        self.export_uvtt_button.setToolTip("Export walls as Universal VTT format")
        self.export_uvtt_button.setEnabled(False)  # Initially disabled
        self.wall_actions_layout.addWidget(self.export_uvtt_button)
        
        # Move the Save File and Cancel Preview buttons
        self.save_uvtt_button = QPushButton("Save File")
        self.save_uvtt_button.clicked.connect(self.export_panel.save_uvtt_preview)
        self.save_uvtt_button.setToolTip("Save the previewed walls to Universal VTT file")
        self.save_uvtt_button.setEnabled(False)
        self.wall_actions_layout.addWidget(self.save_uvtt_button)
        
        self.cancel_uvtt_button = QPushButton("Cancel Preview")
        self.cancel_uvtt_button.clicked.connect(self.export_panel.cancel_uvtt_preview)
        self.cancel_uvtt_button.setToolTip("Return to normal view")
        self.cancel_uvtt_button.setEnabled(False)
        self.wall_actions_layout.addWidget(self.cancel_uvtt_button)
        
        # Add a copy to clipboard button next to Save File
        self.copy_uvtt_button = QPushButton("Copy to Clipboard")
        self.copy_uvtt_button.clicked.connect(self.export_panel.copy_uvtt_to_clipboard)
        self.copy_uvtt_button.setToolTip("Copy the UVTT file JSON to clipboard")
        self.copy_uvtt_button.setEnabled(False)
        self.wall_actions_layout.addWidget(self.copy_uvtt_button)
        
        # Connect drawing tool radio buttons to handler
        self.brush_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.line_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.rectangle_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.circle_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.ellipse_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.fill_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)

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
        # Draw an upward arrow
        arrow_points = [
            QPoint(12, 6),   # Top point
            QPoint(8, 10),   # Left point
            QPoint(10, 10),  # Left inner
            QPoint(10, 18),  # Left bottom
            QPoint(14, 18),  # Right bottom
            QPoint(14, 10),  # Right inner
            QPoint(16, 10)   # Right point
        ]
        painter.drawPolygon(arrow_points)
        painter.end()
        update_icon_label.setPixmap(update_icon)
        update_layout.addWidget(update_icon_label)
        
        # Add text for the update notification
        self.update_text = QLabel("Update Available! Click to download")
        self.update_text.setCursor(Qt.CursorShape.PointingHandCursor)
        update_layout.addWidget(self.update_text)
        
        # Position the notification in the top right corner
        self.update_notification.setGeometry(
            self.width() - 210, 10, 200, 40
        )
        self.update_notification.hide()  # Initially hidden
        
        # Connect the click event to open the download page
        self.update_notification.mousePressEvent = lambda event: open_update_url(self, event)

        
    def zoom_in(self):
        """Zoom in on the image."""
        if hasattr(self.image_label, 'zoom_factor'):
            current_zoom = self.image_label.zoom_factor
            new_zoom = min(self.image_label.max_zoom, current_zoom * 1.2)
            self.image_label.zoom_factor = new_zoom
            self.image_label.update_display()
        
    def zoom_out(self):
        """Zoom out on the image."""
        if hasattr(self.image_label, 'zoom_factor'):
            current_zoom = self.image_label.zoom_factor
            new_zoom = max(self.image_label.min_zoom, current_zoom * 0.8)
            self.image_label.zoom_factor = new_zoom
            self.image_label.update_display()
        
    def reset_view(self):
        """Reset the zoom and pan to default values."""
        if hasattr(self.image_label, 'reset_view'):
            self.image_label.reset_view()
        
    def fit_to_window(self):
        """Fit the image to the current window size."""
        if hasattr(self.image_label, 'fit_to_window'):
            self.image_label.fit_to_window()

    def toggle_grid_overlay(self, state):
        """Toggle the grid overlay on the image."""
        self.grid_overlay_enabled = state == Qt.CheckState.Checked.value
        if self.processed_image is not None:
            self.refresh_display()
    
    def update_grid_size(self, value):
        """Update the grid size and refresh display if grid is enabled."""
        self.pixels_per_grid = value
        if self.grid_overlay_enabled and self.processed_image is not None:
            self.refresh_display()
    
    def refresh_display(self):
        """Refresh the image display with current overlays."""
        if self.processed_image is not None:
            display_image = self.processed_image.copy()
            
            # Add grid overlay if enabled
            if self.grid_overlay_enabled:
                display_image = self.add_grid_overlay(display_image)
            
            self.image_processor.display_image(display_image, preserve_view=True)
    
    def add_grid_overlay(self, image):
        """Add a grid overlay to the image."""
        if self.pixels_per_grid <= 0:
            return image
        
        overlay_image = image.copy()
        height, width = overlay_image.shape[:2]
        
        # Grid color (light gray, semi-transparent)
        grid_color = (128, 128, 128)  # Gray
        thickness = 1
        
        # Draw vertical lines
        for x in range(0, width, self.pixels_per_grid):
            cv2.line(overlay_image, (x, 0), (x, height), grid_color, thickness)
        
        # Draw horizontal lines
        for y in range(0, height, self.pixels_per_grid):
            cv2.line(overlay_image, (0, y), (width, y), grid_color, thickness)
        
        return overlay_image

    def unified_undo(self):
        """Unified undo function that calls the appropriate undo method based on current mode."""
        # Print debug info about current state
        print(f"Unified undo called. UVTT mode: {getattr(self, 'uvtt_preview_active', False)}")
        if hasattr(self, 'wall_edit_history'):
            print(f"Wall edit history size: {len(self.wall_edit_history)}")
        if hasattr(self, 'history'):
            print(f"General history size: {len(self.history)}")
            
        # Decide which undo to use based on current mode
        if hasattr(self, 'uvtt_preview_active') and self.uvtt_preview_active:
            # We're in wall edit mode, use wall undo if we have wall history
            if (hasattr(self, 'wall_edit_history') and 
                len(self.wall_edit_history) > 1 and 
                hasattr(self.export_panel, 'undo_wall_edit')):
                
                print("Using wall edit undo")
                self.export_panel.undo_wall_edit()
            else:
                print("No wall edit history to undo")
                QMessageBox.information(self, "Undo", "Nothing to undo in wall edit mode")
                
        elif hasattr(self, 'history') and self.history:
            # We're in mask/contour edit mode, use mask undo
            print("Using mask processor undo")
            self.mask_processor.undo()
            self.setStatusTip("Last edit undone")
        else:
            # No history to undo
            print("No history to undo")
            self.setStatusTip("Nothing to undo")
            QMessageBox.information(self, "Undo", "Nothing to undo")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WallDetectionApp()
    window.show()
    sys.exit(app.exec())