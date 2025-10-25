import sys
import os
import cv2

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QWidget, 
    QCheckBox, QRadioButton, QButtonGroup, QListWidget,
    QScrollArea, QSizePolicy, QDialog, QFrame, QSpinBox, QDoubleSpinBox,
    QGridLayout, QComboBox, QMessageBox, QGroupBox, QFileDialog, QInputDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QPixmap, QPainter, QColor, QGuiApplication, QKeySequence, QShortcut, QIcon
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
        
        # Zoom shortcuts handled by menu actions
        
        # Reset view and fit to window shortcuts handled by menu actions

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
        # Populate preset combo boxes after loading presets
        self.preset_manager.update_detection_preset_combo()
        self.preset_manager.update_export_preset_combo()

        apply_stylesheet(self)
        check_for_updates(self)
        
    def initialize_state(self):
        self.original_image = None  # Original full-size image
        self.current_image = None   # Working image
        self.max_working_dimension = 1500  # Maximum dimension for processing
        self.scale_factor = 1.0     # Scale factor between original and working image
        self.processed_image = None
        self.current_contours = []
        self.current_lights = []   # Detected light points
        self.display_scale_factor = 1.0
        self.display_offset = (0, 0)
        self.sliders = {}
        self.mask_layer = None  # Mask layer for paint mode

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

        # Light color selection
        self.light_colors = []  # List to store QColor objects for light detection
        self.selected_light_color_item = None

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
        
        # Portal editing states  
        self.selected_portal_index = -1  # Currently selected portal
        self.selected_portal_indices = [] # List of selected portal indices for multi-selection
        self.selected_portal_points = []  # List of selected portal points (portal_idx, point_idx) tuples
        self.multi_portal_drag = False    # Flag for multi-portal drag operation
        self.multi_portal_drag_start = None # Start position for multi-portal drag
        self.dragging_from_portal_line = False # Flag for dragging from portal line rather than endpoints
        
        # Light editing states
        self.selected_light_index = -1   # Currently selected light
        self.selected_light_indices = [] # List of selected light indices for multi-selection
        self.dragging_light = False      # Flag for light dragging operation
        self.multi_light_drag = False    # Flag for multi-light drag operation
        self.multi_light_drag_start = None # Start position for multi-light drag
        
        # Grid overlay

        
        # History tracking for undo feature
        self.history = deque(maxlen=5)  # Store up to 5 previous states
        
        # UVTT-related attributes
        self.uvtt_walls_preview = None
        self.uvtt_export_params = None
        self.wall_edit_history = []
        
        # Tool management
        self._last_tool_id = 0  # Track last tool for rollback on cancel

    def setup_ui(self):
        # Create menu bar
        self.setup_menu_bar()
        
        # Main layout - use central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main horizontal layout: left panel | central image | right panel
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(0)  # Remove spacing between panels
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the layout
        
        # Create the three main panels
        self.setup_left_tools_panel()
        self.setup_central_image_area()
        self.setup_right_properties_panel()
        
    def setup_menu_bar(self):
        """Create the menu bar with File, Edit, View, and Help menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Open action
        open_action = file_menu.addAction('&Open Image')
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.image_processor.open_image)
        
        # Load from URL action
        load_url_action = file_menu.addAction('Load from &URL')
        load_url_action.triggered.connect(self.image_processor.load_image_from_url)
        
        file_menu.addSeparator()
        
        # Save Image action
        save_image_action = file_menu.addAction('&Save Image')
        save_image_action.setShortcut('Ctrl+S')
        save_image_action.triggered.connect(self.image_processor.save_image)
        
        # Export SVG action
        export_svg_action = file_menu.addAction('Export S&VG')
        export_svg_action.triggered.connect(self.export_panel.export_to_svg)
        
        # Save File action (for UVTT files)
        save_file_action = file_menu.addAction('Save &File')
        save_file_action.triggered.connect(self.export_panel.save_uvtt_preview)
        
        # Copy to Clipboard action
        copy_clipboard_action = file_menu.addAction('&Copy to Clipboard')
        copy_clipboard_action.triggered.connect(self.export_panel.copy_uvtt_to_clipboard)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = file_menu.addAction('E&xit')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu('&Edit')
        
        # Undo action (shortcut handled by QShortcut in __init__)
        undo_action = edit_menu.addAction('&Undo')
        undo_action.triggered.connect(self.unified_undo)
        
        # Clear Walls action
        clear_walls_action = edit_menu.addAction('&Clear Walls')
        clear_walls_action.triggered.connect(self.export_panel.cancel_uvtt_preview)
        
        edit_menu.addSeparator()
        
        # Preset actions
        save_preset_action = edit_menu.addAction('Save Detection &Preset')
        save_preset_action.triggered.connect(self.preset_manager.save_detection_preset)
        
        manage_presets_action = edit_menu.addAction('&Manage Presets')
        manage_presets_action.triggered.connect(self.preset_manager.manage_detection_presets)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Zoom actions
        zoom_in_action = view_menu.addAction('Zoom &In')
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(self.zoom_in)
        
        zoom_out_action = view_menu.addAction('Zoom &Out')
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        
        fit_to_window_action = view_menu.addAction('&Fit to Window')
        fit_to_window_action.setShortcut('Ctrl+F')
        fit_to_window_action.triggered.connect(self.fit_to_window)
        
        reset_view_action = view_menu.addAction('&Reset View')
        reset_view_action.setShortcut('Ctrl+0')
        reset_view_action.triggered.connect(self.reset_view)
        
        view_menu.addSeparator()
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = help_menu.addAction('&About')
        about_action.triggered.connect(self.show_about)
        
    def setup_left_tools_panel(self):
        """Create the left panel with tool selection."""
        # Left panel container
        self.left_panel = QWidget()
        self.left_panel.setObjectName("leftToolPanel")
        self.left_panel.setFixedWidth(100)  # Fixed width for tool panel
        
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setSpacing(5)
        self.left_layout.setContentsMargins(5, 10, 5, 10)
        
        # Tool selection buttons
        self.tool_group = QButtonGroup()
        self.tool_group.setExclusive(True)
        
        # Get the resources directory path
        resources_dir = os.path.join(os.path.dirname(__file__), '../../resources')
        
        # Detection tool
        self.detect_tool_btn = QPushButton(" Detect")
        detect_icon = QIcon(os.path.join(resources_dir, 'quickview-icon.svg'))
        self.detect_tool_btn.setIcon(detect_icon)
        self.detect_tool_btn.setIconSize(QSize(24, 24))
        self.detect_tool_btn.setCheckable(True)
        self.detect_tool_btn.setChecked(True)
        self.detect_tool_btn.setToolTip("Wall Detection Tool")
        self.tool_group.addButton(self.detect_tool_btn, 0)
        self.left_layout.addWidget(self.detect_tool_btn)
        
        # Paint/Edit tool
        self.paint_tool_btn = QPushButton(" Draw")
        draw_icon = QIcon(os.path.join(resources_dir, 'pen-tool-vector-design-icon.svg'))
        self.paint_tool_btn.setIcon(draw_icon)
        self.paint_tool_btn.setIconSize(QSize(24, 24))
        self.paint_tool_btn.setCheckable(True)
        self.paint_tool_btn.setToolTip("Draw/Edit Mask Tool")
        self.tool_group.addButton(self.paint_tool_btn, 1)
        self.left_layout.addWidget(self.paint_tool_btn)
        
        # UVTT Editor tool
        self.uvtt_tool_btn = QPushButton(" Walls")
        walls_icon = QIcon(os.path.join(resources_dir, 'marquee-rectangle-tool-icon.svg'))
        self.uvtt_tool_btn.setIcon(walls_icon)
        self.uvtt_tool_btn.setIconSize(QSize(24, 24))
        self.uvtt_tool_btn.setCheckable(True)
        self.uvtt_tool_btn.setToolTip("UVTT Wall Editor")
        self.tool_group.addButton(self.uvtt_tool_btn, 2)
        self.left_layout.addWidget(self.uvtt_tool_btn)
        
        self.left_layout.addStretch()
        
        # Connect tool changes
        self.tool_group.buttonClicked.connect(self.on_tool_changed)
        
        self.main_layout.addWidget(self.left_panel)
        
    def setup_central_image_area(self):
        """Create the central image viewing area."""
        # Central image area
        self.image_container = QWidget()
        self.image_container.setObjectName("imageContainer")
        
        self.image_layout = QVBoxLayout(self.image_container)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image viewer
        self.image_label = InteractiveImageLabel(self)
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create scroll area for the image
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("imageScrollArea")
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.image_layout.addWidget(self.scroll_area)
        
        # Add update notification to image area (will be positioned in bottom left)
        self.setup_update_notification()
        
        self.main_layout.addWidget(self.image_container, 1)  # Give it stretch factor 1
        
    def setup_update_notification(self):
        """Create and position the update notification in the bottom left."""
        # Create the update notification widget (initially hidden)
        self.update_notification = QWidget(self.image_container)
        self.update_notification.setObjectName("updateNotification")
        
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
            QPoint(12, 6),
            QPoint(8, 10),
            QPoint(10, 10),
            QPoint(10, 18),
            QPoint(14, 18),
            QPoint(14, 10),
            QPoint(16, 10)
        ]
        painter.drawPolygon(arrow_points)
        painter.end()
        update_icon_label.setPixmap(update_icon)
        update_layout.addWidget(update_icon_label)
        
        # Add text for the update notification - make it shorter to leave room
        self.update_text = QLabel("Update Available!")
        self.update_text.setCursor(Qt.CursorShape.PointingHandCursor)
        update_layout.addWidget(self.update_text)
        
        # Add minimal close button
        self.dismiss_button = QPushButton("×")
        self.dismiss_button.setObjectName("dismissButton")
        self.dismiss_button.setFixedSize(20, 20)
        self.dismiss_button.setToolTip("Close")
        self.dismiss_button.clicked.connect(self.update_notification.hide)
        update_layout.addWidget(self.dismiss_button)
        
        # Set a reasonable width for the notification
        self.update_notification.setFixedSize(280, 40)
        
        self.update_notification.hide()  # Initially hidden
        
        # Position will be set when shown (after layout is complete)
        
        # Connect the click event to open the download page
        self.update_notification.mousePressEvent = lambda event: open_update_url(self, event)
        
    def position_update_notification(self):
        """Position the update notification in the bottom left of the image container."""
        if hasattr(self, 'update_notification') and self.update_notification.isVisible():
            # Ensure the widget is properly sized first
            self.update_notification.adjustSize()
            
            # Position in bottom left with some margin
            margin = 10
            x = margin
            y = self.image_container.height() - self.update_notification.height() - margin
            self.update_notification.move(x, y)
            
    def resizeEvent(self, event):
        """Handle window resize events to reposition the update notification."""
        super().resizeEvent(event)
        # Reposition the update notification when window is resized
        self.position_update_notification()
        
    def setup_right_properties_panel(self):
        """Create the right panel with tool properties and settings."""
        # # Right panel container with scroll area
        # self.right_panel_scroll = QScrollArea()
        # self.right_panel_scroll.setFixedWidth(300)  # Fixed width for properties panel
        # self.right_panel_scroll.setWidgetResizable(True)
        # self.right_panel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # self.right_panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.right_panel_scroll.setStyleSheet("""
        #     QScrollArea {
        #         background-color: #2b2b2b;
        #         border-left: 1px solid #555;
        #         border: none;
        #     }
        #     QScrollBar:vertical {
        #         background-color: #2b2b2b;
        #         border: none;
        #         width: 12px;
        #     }
        #     QScrollBar::handle:vertical {
        #         background-color: #555;
        #         border-radius: 6px;
        #         min-height: 20px;
        #     }
        #     QScrollBar::handle:hover {
        #         background-color: #777;
        #     }
        # """)
        
        self.right_panel = QWidget()
        self.right_panel.setObjectName("rightPropertiesPanel")
        
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setSpacing(10)
        self.right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create property sections that will be shown/hidden based on tool selection
        self.setup_detection_properties()
        self.setup_paint_properties()
        self.setup_uvtt_properties()
        
        # Add stretch at the bottom
        self.right_layout.addStretch()
        
        # self.right_panel_scroll.setWidget(self.right_panel)
        # self.main_layout.addWidget(self.right_panel_scroll)
        
        # # Detection mode selection at the top
        # self.detection_mode_title = QLabel("Detection Mode:")
        # self.detection_mode_title.setStyleSheet("font-weight: bold;")
        # self.right_layout.addWidget(self.detection_mode_title)
        
        # self.detection_mode_layout = QHBoxLayout()
        # self.right_layout.addLayout(self.detection_mode_layout)
        
        # self.detection_mode_group = QButtonGroup()
        # self.edge_detection_radio = QRadioButton("Edge Detection")
        # self.color_detection_radio = QRadioButton("Color Detection")
        # self.edge_detection_radio.setChecked(True)  # Default to edge detection
        # self.detection_mode_group.addButton(self.edge_detection_radio)
        # self.detection_mode_group.addButton(self.color_detection_radio)
        
        # self.detection_mode_layout.addWidget(self.edge_detection_radio)
        # self.detection_mode_layout.addWidget(self.color_detection_radio)
        
        # # Connect detection mode radio buttons
        # self.edge_detection_radio.toggled.connect(self.detection_panel.toggle_detection_mode_radio)
        # self.color_detection_radio.toggled.connect(self.detection_panel.toggle_detection_mode_radio)
        
        # Add a separator after detection mode
        self.tool_separator_top = QFrame()
        self.tool_separator_top.setFrameShape(QFrame.Shape.HLine)
        self.tool_separator_top.setFrameShadow(QFrame.Shadow.Sunken)
        self.right_layout.addWidget(self.tool_separator_top)

        # Mode selection (Detection/Deletion/Color Selection/Edit Mask)
        self.mode_layout = QHBoxLayout()
        self.right_layout.addLayout(self.mode_layout)
        
        self.mode_label = QLabel("Tool:")
        self.mode_layout.addWidget(self.mode_label)
        
        self.deletion_mode_radio = QRadioButton("Deletion")
        self.thin_mode_radio = QRadioButton("Thin")
        self.deletion_mode_radio.setChecked(True)
        self.color_selection_mode_radio = QRadioButton("Color Pick")
        self.color_selection_mode_radio.setVisible(False)  # Hide this radio button initially
        self.edit_mask_mode_radio = QRadioButton("Edit Mask")
        self.edit_mask_mode_radio.setVisible(False)  # Hide this radio button initially

        # Create button group for mode selection to ensure only one is selected
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.deletion_mode_radio)
        self.mode_button_group.addButton(self.thin_mode_radio)
        self.mode_button_group.addButton(self.color_selection_mode_radio)
        self.mode_button_group.addButton(self.edit_mask_mode_radio)
        self.mode_button_group.setExclusive(True)

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
        self.tool_separator_bottom = QFrame()
        self.tool_separator_bottom.setFrameShape(QFrame.Shape.HLine)
        self.tool_separator_bottom.setFrameShadow(QFrame.Shadow.Sunken)
        self.right_layout.addWidget(self.tool_separator_bottom)

        

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
        self.right_layout.addWidget(self.thin_options)
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
        
        self.right_layout.addWidget(self.color_selection_options)
        self.color_selection_options.setVisible(False)
        
        # Wrap controls in a scroll area to handle many options
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.right_panel)
        self.scroll_area.setMinimumWidth(350)  # Set minimum width for controls panel
        self.scroll_area.setMaximumWidth(400)  # Set maximum width for controls panel
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Prevent horizontal scrolling
        self.main_layout.addWidget(self.scroll_area)
        
        # # Right panel for image display
        # self.image_panel = QWidget()
        # self.image_layout = QVBoxLayout(self.image_panel)
        # self.main_layout.addWidget(self.image_panel, 1)  # Give image panel more space (stretch factor 1)
        
        # # Image display (using custom interactive label)
        # self.image_label = InteractiveImageLabel(self)
        # self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # self.image_layout.addWidget(self.image_label)

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
        
        self.right_layout.addLayout(self.min_area_mode_layout)
        
        # Min Area is now a percentage (0.0001% to 1% of image area) or pixels (1 to 1000)
        self.detection_panel.add_slider("Min Area", 1, 25000, 100, scale_factor=0.001)  # Default 0.1%
        self.detection_panel.add_slider("Smoothing", 1, 21, 5, step=2)  # Changed from "Blur"
        self.detection_panel.add_slider("Edge Sensitivity", 0, 255, 255)  # Changed from "Canny1"
        self.detection_panel.add_slider("Edge Threshold", 0, 255, 106)  # Changed from "Canny2"
        self.detection_panel.add_slider("Edge Margin", 0, 50, 0)
        
        # Checkboxes for merge options
        self.merge_options_layout = QVBoxLayout()
        self.right_layout.addLayout(self.merge_options_layout)

        self.merge_contours = QCheckBox("Merge Contours")
        self.merge_contours.setChecked(False)
        self.merge_options_layout.addWidget(self.merge_contours)

        # Use a scaling factor of 10 for float values (0 to 10.0 with 0.1 precision)
        self.detection_panel.add_slider("Min Merge Distance", 0, 100, 5, scale_factor=0.1)  # Default 0.5

        # Create layout for hatching controls
        self.hatching_layout = QVBoxLayout()
        self.right_layout.addLayout(self.hatching_layout)
        
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
        self.hatching_color_button.setObjectName("hatchingColorButton")
        self.hatching_color_button.setFixedSize(30, 20)
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
        self.color_section_title.setProperty("headerLabel", True)
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
        self.threshold_header.setProperty("headerLabel", True)
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
        self.right_layout.addWidget(self.color_section)
        
        # Initially hide the entire color section
        self.color_section.setVisible(False)

        # Add a checkbox for high-resolution processing
        self.high_res_checkbox = QCheckBox("Process at Full Resolution")
        self.high_res_checkbox.setChecked(False)
        self.high_res_checkbox.setToolTip("Process at full resolution (slower but more accurate)")
        self.high_res_checkbox.stateChanged.connect(self.image_processor.reload_working_image)
        self.right_layout.addWidget(self.high_res_checkbox)

        # Group edge detection settings
        self.edge_detection_widgets = []
        self.edge_detection_widgets.append(self.sliders["Edge Sensitivity"])
        self.edge_detection_widgets.append(self.sliders["Edge Threshold"])

        # Add a default black color with default threshold
        self.detection_panel.add_wall_color_to_list(QColor(0, 0, 0), 10.0)

        # --- Presets UI ---
        # Create a container widget for the presets so we can hide/show it
        self.presets_container = QWidget()
        self.presets_main_layout = QVBoxLayout(self.presets_container)
        self.presets_main_layout.setSpacing(2) # Reduce spacing between the two lines
        self.presets_main_layout.setContentsMargins(0, 0, 0, 0)

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
        self.presets_main_layout.addLayout(presets_line1_layout)
        self.presets_main_layout.addLayout(presets_line2_layout)

        # Add the presets container to the controls layout
        self.right_layout.addWidget(self.presets_container)

        # Add a divider with auto margin on top
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        
        # Add a stretch item above the separator to push it to the bottom
        self.right_layout.addStretch(1) # Stretch is now AFTER presets
        
        self.right_layout.addWidget(separator)
        
        # Light detection functionality moved to bottom section
        
        # Light detection functionality moved to bottom section
        
        # Connect drawing tool radio buttons to handler
        self.brush_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.line_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.rectangle_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.circle_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)
        self.ellipse_tool_radio.toggled.connect(self.drawing_tools.update_drawing_tool)

        
    def setup_detection_properties(self):
        """Setup properties panel for detection tool."""
        
        # Detection mode selection (moved to main layout without group box)
        mode_layout = QHBoxLayout()
        self.edge_detection_radio = QRadioButton("Edge Detection")
        self.color_detection_radio = QRadioButton("Color Detection")
        self.edge_detection_radio.setChecked(True)
        
        self.detection_mode_group = QButtonGroup()
        self.detection_mode_group.addButton(self.edge_detection_radio)
        self.detection_mode_group.addButton(self.color_detection_radio)
        
        mode_layout.addWidget(self.edge_detection_radio)
        mode_layout.addWidget(self.color_detection_radio)
        self.right_layout.addLayout(mode_layout)
        
        # Connect detection mode radio buttons
        self.edge_detection_radio.toggled.connect(self.detection_panel.toggle_detection_mode_radio)
        self.color_detection_radio.toggled.connect(self.detection_panel.toggle_detection_mode_radio)
        
        # Add detection sliders (will be populated by detection_panel.add_slider)
        self.sliders = {}  # Initialize sliders dict
        
        # Color detection section (initially hidden)
        self.color_section = QWidget()
        self.color_section_layout = QVBoxLayout(self.color_section)
        
        # Wall colors list and buttons
        self.wall_colors_list = QListWidget()
        self.wall_colors_list.setMaximumHeight(100)
        self.color_section_layout.addWidget(QLabel("Wall Colors:"))
        self.color_section_layout.addWidget(self.wall_colors_list)
        
        color_buttons_layout = QHBoxLayout()
        self.add_wall_color_button = QPushButton("Add Color")
        self.remove_wall_color_button = QPushButton("Remove")
        color_buttons_layout.addWidget(self.add_wall_color_button)
        color_buttons_layout.addWidget(self.remove_wall_color_button)
        self.color_section_layout.addLayout(color_buttons_layout)
        
        self.right_layout.addWidget(self.color_section)
        self.color_section.setVisible(False)
        
        # Light detection section (moved from the top)
        self.light_group = QGroupBox("Light Detection")
        light_layout = QVBoxLayout(self.light_group)
        
        self.enable_light_detection = QCheckBox("Enable Light Detection")
        self.enable_light_detection.setChecked(False)
        self.enable_light_detection.stateChanged.connect(self.detection_panel.toggle_light_detection)
        light_layout.addWidget(self.enable_light_detection)
        
        # Light options (will be populated by setup methods)
        self.light_options = QWidget()
        self.light_options_layout = QVBoxLayout(self.light_options)
        self.light_options_layout.setContentsMargins(0, 0, 0, 0)
        
        # Light brightness threshold
        self.light_brightness_layout = QHBoxLayout()
        self.light_brightness_label = QLabel("Brightness Threshold:")
        self.light_brightness_layout.addWidget(self.light_brightness_label)
        
        self.light_brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.light_brightness_slider.setMinimum(30)  # 0.3
        self.light_brightness_slider.setMaximum(100)  # 1.0
        self.light_brightness_slider.setValue(80)  # 0.8 default
        self.light_brightness_slider.valueChanged.connect(self.detection_panel.update_light_brightness)
        self.light_brightness_layout.addWidget(self.light_brightness_slider)
        
        self.light_brightness_value = QLabel("0.8")
        self.light_brightness_layout.addWidget(self.light_brightness_value)
        self.light_options_layout.addLayout(self.light_brightness_layout)
        
        # Light minimum size
        self.light_min_size_layout = QHBoxLayout()
        self.light_min_size_label = QLabel("Min Light Size:")
        self.light_min_size_layout.addWidget(self.light_min_size_label)
        
        self.light_min_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.light_min_size_slider.setMinimum(1)
        self.light_min_size_slider.setMaximum(50)
        self.light_min_size_slider.setValue(5)
        self.light_min_size_slider.valueChanged.connect(self.detection_panel.update_light_min_size)
        self.light_min_size_layout.addWidget(self.light_min_size_slider)
        
        self.light_min_size_value = QLabel("5")
        self.light_min_size_layout.addWidget(self.light_min_size_value)
        self.light_options_layout.addLayout(self.light_min_size_layout)
        
        # Light maximum size
        self.light_max_size_layout = QHBoxLayout()
        self.light_max_size_label = QLabel("Max Light Size:")
        self.light_max_size_layout.addWidget(self.light_max_size_label)
        
        self.light_max_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.light_max_size_slider.setMinimum(50)
        self.light_max_size_slider.setMaximum(2000)
        self.light_max_size_slider.setValue(500)
        self.light_max_size_slider.valueChanged.connect(self.detection_panel.update_light_max_size)
        self.light_max_size_layout.addWidget(self.light_max_size_slider)
        
        self.light_max_size_value = QLabel("500")
        self.light_max_size_layout.addWidget(self.light_max_size_value)
        self.light_options_layout.addLayout(self.light_max_size_layout)
        
        # Light merge distance
        self.light_merge_distance_layout = QHBoxLayout()
        self.light_merge_distance_label = QLabel("Merge Distance:")
        self.light_merge_distance_layout.addWidget(self.light_merge_distance_label)
        
        self.light_merge_distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.light_merge_distance_slider.setMinimum(0)   # 0 pixels
        self.light_merge_distance_slider.setMaximum(100) # 100 pixels
        self.light_merge_distance_slider.setValue(20)    # 20 pixels default
        self.light_merge_distance_slider.valueChanged.connect(self.detection_panel.update_light_merge_distance)
        self.light_merge_distance_layout.addWidget(self.light_merge_distance_slider)
        
        self.light_merge_distance_value = QLabel("20")
        self.light_merge_distance_layout.addWidget(self.light_merge_distance_value)
        self.light_options_layout.addLayout(self.light_merge_distance_layout)
        
        # Light color selection
        self.light_colors_layout = QVBoxLayout()
        self.light_options_layout.addLayout(self.light_colors_layout)
        
        self.light_colors_label = QLabel("Light Colors:")
        self.light_colors_layout.addWidget(self.light_colors_label)
        
        # List widget to display selected light colors
        self.light_colors_list = QListWidget()
        self.light_colors_list.setMaximumHeight(80)
        self.light_colors_list.itemClicked.connect(self.detection_panel.select_light_color)
        self.light_colors_list.itemDoubleClicked.connect(self.detection_panel.edit_light_color)
        self.light_colors_layout.addWidget(self.light_colors_list)
        
        # Buttons for light color management
        self.light_color_buttons_layout = QHBoxLayout()
        self.light_colors_layout.addLayout(self.light_color_buttons_layout)
        
        self.add_light_color_button = QPushButton("Add Light Color")
        self.add_light_color_button.clicked.connect(self.detection_panel.add_light_color)
        self.light_color_buttons_layout.addWidget(self.add_light_color_button)
        
        self.remove_light_color_button = QPushButton("Remove Color")
        self.remove_light_color_button.clicked.connect(self.detection_panel.remove_light_color)
        self.light_color_buttons_layout.addWidget(self.remove_light_color_button)
        
        # Light color threshold controls
        self.light_threshold_container = QWidget()
        self.light_threshold_layout = QVBoxLayout(self.light_threshold_container)
        self.light_threshold_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add a header for the threshold section
        self.light_threshold_header = QLabel("Selected Light Color Threshold:")
        self.light_threshold_header.setProperty("headerLabel", True)
        self.light_threshold_layout.addWidget(self.light_threshold_header)
        
        # Create the threshold slider
        self.light_current_threshold_layout = QHBoxLayout()
        self.light_threshold_layout.addLayout(self.light_current_threshold_layout)
        
        self.light_threshold_label = QLabel("Threshold: 15.0")
        self.light_current_threshold_layout.addWidget(self.light_threshold_label)
        
        self.light_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.light_threshold_slider.setMinimum(50)   # 5.0
        self.light_threshold_slider.setMaximum(500)  # 50.0
        self.light_threshold_slider.setValue(150)    # Default value 15.0
        self.light_threshold_slider.valueChanged.connect(self.detection_panel.update_selected_light_threshold)
        self.light_current_threshold_layout.addWidget(self.light_threshold_slider)
        
        self.light_colors_layout.addWidget(self.light_threshold_container)
        
        # Initially hide the threshold controls until a color is selected
        self.light_threshold_container.setVisible(False)
        
        # Add a separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.light_options_layout.addWidget(separator)
        
        light_layout.addWidget(self.light_options)
        
        # Initially hide the light options until enabled
        self.light_options.setVisible(False)
        
        self.right_layout.addWidget(self.light_group)
        self.light_group.setVisible(False)
        
        # Add a default bright white color for light detection
        self.detection_panel.add_light_color_to_list(QColor(255, 255, 200), 15.0)
        

        
    def setup_paint_properties(self):
        """Setup properties panel for paint/edit tool."""
        
        self.paint_group = QGroupBox("Paint Settings")
        paint_layout = QVBoxLayout(self.paint_group)
        
        # Brush size
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Brush Size:"))
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setMinimum(0)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.valueChanged.connect(self.drawing_tools.update_brush_size)
        brush_layout.addWidget(self.brush_size_slider)
        self.brush_size_value = QLabel("10")
        brush_layout.addWidget(self.brush_size_value)
        paint_layout.addLayout(brush_layout)
        
        # Draw/Erase mode
        mode_layout = QHBoxLayout()
        self.draw_radio = QRadioButton("Draw")
        self.erase_radio = QRadioButton("Erase")
        self.draw_radio.setChecked(True)
        
        self.draw_mode_group = QButtonGroup()
        self.draw_mode_group.addButton(self.draw_radio)
        self.draw_mode_group.addButton(self.erase_radio)
        
        mode_layout.addWidget(self.draw_radio)
        mode_layout.addWidget(self.erase_radio)
        paint_layout.addLayout(mode_layout)
        
        # Drawing tools
        tools_group = QGroupBox("Drawing Tools")
        tools_layout = QGridLayout(tools_group)
        
        self.brush_tool_radio = QRadioButton("Brush")
        self.line_tool_radio = QRadioButton("Line")
        self.rectangle_tool_radio = QRadioButton("Rectangle")
        self.circle_tool_radio = QRadioButton("Circle")
        self.ellipse_tool_radio = QRadioButton("Ellipse")
        
        self.brush_tool_radio.setChecked(True)
        
        self.drawing_tool_group = QButtonGroup()
        self.drawing_tool_group.addButton(self.brush_tool_radio)
        self.drawing_tool_group.addButton(self.line_tool_radio)
        self.drawing_tool_group.addButton(self.rectangle_tool_radio)
        self.drawing_tool_group.addButton(self.circle_tool_radio)
        self.drawing_tool_group.addButton(self.ellipse_tool_radio)
        
        tools_layout.addWidget(self.brush_tool_radio, 0, 0)
        tools_layout.addWidget(self.line_tool_radio, 0, 1)
        tools_layout.addWidget(self.rectangle_tool_radio, 0, 2)
        tools_layout.addWidget(self.circle_tool_radio, 1, 0)
        tools_layout.addWidget(self.ellipse_tool_radio, 1, 1)
        
        paint_layout.addWidget(tools_group)
        
        self.paint_group.setVisible(False)
        self.right_layout.addWidget(self.paint_group)
        

        
    def setup_uvtt_properties(self):
        """Setup properties panel for UVTT editor tool with detection and export settings."""
        
        # Create main UVTT container (use QWidget instead of QGroupBox for cleaner look)
        self.uvtt_group = QWidget()
        uvtt_layout = QVBoxLayout(self.uvtt_group)
        uvtt_layout.setContentsMargins(0, 0, 0, 0)
        uvtt_layout.setSpacing(10)
        
        # === UVTT Export Settings Section ===
        export_group = QGroupBox("UVTT Export Settings")
        export_layout = QVBoxLayout(export_group)
        
        # Simplification tolerance
        tolerance_layout = QHBoxLayout()
        tolerance_layout.addWidget(QLabel("Simplification:"))
        self.uvtt_tolerance_spinbox = QDoubleSpinBox()
        self.uvtt_tolerance_spinbox.setRange(0.0, 1.0)
        self.uvtt_tolerance_spinbox.setDecimals(4)
        self.uvtt_tolerance_spinbox.setSingleStep(0.0005)
        self.uvtt_tolerance_spinbox.setValue(0.0005)
        self.uvtt_tolerance_spinbox.setToolTip("Controls wall detail: 0=full detail, 0.0005=minor smoothing, 0.01+=heavy simplification")
        tolerance_layout.addWidget(self.uvtt_tolerance_spinbox)
        export_layout.addLayout(tolerance_layout)
        
        # Maximum wall length
        max_length_layout = QHBoxLayout()
        max_length_layout.addWidget(QLabel("Max Wall Length:"))
        self.uvtt_max_length_spinbox = QSpinBox()
        self.uvtt_max_length_spinbox.setRange(5, 500)
        self.uvtt_max_length_spinbox.setSingleStep(5)
        self.uvtt_max_length_spinbox.setValue(50)
        self.uvtt_max_length_spinbox.setToolTip("Maximum length of wall segments in pixels (lower=more walls, higher=better performance)")
        max_length_layout.addWidget(self.uvtt_max_length_spinbox)
        export_layout.addLayout(max_length_layout)
        
        # Maximum number of walls
        max_walls_layout = QHBoxLayout()
        max_walls_layout.addWidget(QLabel("Max Points:"))
        self.uvtt_max_walls_spinbox = QSpinBox()
        self.uvtt_max_walls_spinbox.setRange(100, 20000)
        self.uvtt_max_walls_spinbox.setSingleStep(100)
        self.uvtt_max_walls_spinbox.setValue(5000)
        self.uvtt_max_walls_spinbox.setToolTip("Maximum number of generation points (affects complexity vs performance)")
        max_walls_layout.addWidget(self.uvtt_max_walls_spinbox)
        export_layout.addLayout(max_walls_layout)
        
        # Point merge distance
        merge_distance_layout = QHBoxLayout()
        merge_distance_layout.addWidget(QLabel("Merge Distance:"))
        self.uvtt_merge_distance_spinbox = QDoubleSpinBox()
        self.uvtt_merge_distance_spinbox.setRange(0.0, 500.0)
        self.uvtt_merge_distance_spinbox.setDecimals(1)
        self.uvtt_merge_distance_spinbox.setSingleStep(1.0)
        self.uvtt_merge_distance_spinbox.setValue(25.0)
        self.uvtt_merge_distance_spinbox.setToolTip("How close wall endpoints must be to merge (0=no merge, 25=standard)")
        merge_distance_layout.addWidget(self.uvtt_merge_distance_spinbox)
        export_layout.addLayout(merge_distance_layout)
        
        # Angle tolerance
        angle_tolerance_layout = QHBoxLayout()
        angle_tolerance_layout.addWidget(QLabel("Angle Tolerance:"))
        self.uvtt_angle_tolerance_spinbox = QDoubleSpinBox()
        self.uvtt_angle_tolerance_spinbox.setRange(0.0, 30.0)
        self.uvtt_angle_tolerance_spinbox.setDecimals(2)
        self.uvtt_angle_tolerance_spinbox.setSingleStep(0.5)
        self.uvtt_angle_tolerance_spinbox.setValue(1.0)
        self.uvtt_angle_tolerance_spinbox.setToolTip("Angle alignment tolerance for merging straight walls (0=perfect, 5+=loose)")
        angle_tolerance_layout.addWidget(self.uvtt_angle_tolerance_spinbox)
        export_layout.addLayout(angle_tolerance_layout)
        
        # Maximum gap
        max_gap_layout = QHBoxLayout()
        max_gap_layout.addWidget(QLabel("Max Gap Connect:"))
        self.uvtt_max_gap_spinbox = QDoubleSpinBox()
        self.uvtt_max_gap_spinbox.setRange(0.0, 100.0)
        self.uvtt_max_gap_spinbox.setDecimals(1)
        self.uvtt_max_gap_spinbox.setSingleStep(1.0)
        self.uvtt_max_gap_spinbox.setValue(10.0)
        self.uvtt_max_gap_spinbox.setToolTip("Maximum distance between straight walls to connect (0=don't connect, 10-20=good)")
        max_gap_layout.addWidget(self.uvtt_max_gap_spinbox)
        export_layout.addLayout(max_gap_layout)
        
        # Grid snapping
        grid_label = QLabel("Grid Snapping:")
        grid_label.setProperty("headerLabel", True)
        export_layout.addWidget(grid_label)
        
        self.uvtt_enable_grid = QCheckBox("Enable Grid")
        self.uvtt_enable_grid.setChecked(False)
        self.uvtt_enable_grid.setToolTip("Enable grid snapping for wall alignment")
        self.uvtt_enable_grid.stateChanged.connect(self.on_grid_snapping_toggled)
        export_layout.addWidget(self.uvtt_enable_grid)
        
        self.uvtt_same_as_overlay = QCheckBox("Same as Overlay")
        self.uvtt_same_as_overlay.setChecked(True)
        self.uvtt_same_as_overlay.setToolTip("Use same grid size and offset as overlay (recommended)")
        self.uvtt_same_as_overlay.stateChanged.connect(self.on_same_as_overlay_toggled)
        export_layout.addWidget(self.uvtt_same_as_overlay)
        
        # Independent grid settings (shown when "Same as Overlay" is unchecked)
        self.uvtt_independent_grid_group = QWidget()
        independent_grid_layout = QVBoxLayout(self.uvtt_independent_grid_group)
        independent_grid_layout.setContentsMargins(20, 0, 0, 0)  # Indent to show dependency
        
        grid_size_layout = QHBoxLayout()
        grid_size_layout.addWidget(QLabel("Grid Size:"))
        self.uvtt_grid_size_spinbox = QSpinBox()
        self.uvtt_grid_size_spinbox.setRange(10, 500)
        self.uvtt_grid_size_spinbox.setSingleStep(1)
        self.uvtt_grid_size_spinbox.setValue(70)
        self.uvtt_grid_size_spinbox.setToolTip("Grid size for snapping (70-100=common VTT grid sizes)")
        grid_size_layout.addWidget(self.uvtt_grid_size_spinbox)
        independent_grid_layout.addLayout(grid_size_layout)
        
        self.uvtt_allow_half_grid = QCheckBox("Allow Half-Grid Positions")
        self.uvtt_allow_half_grid.setChecked(False)
        self.uvtt_allow_half_grid.setToolTip("Allow half-grid positions for more precision")
        independent_grid_layout.addWidget(self.uvtt_allow_half_grid)
        
        # Grid offset controls for independent settings
        grid_offset_layout = QHBoxLayout()
        grid_offset_layout.addWidget(QLabel("Grid Offset X:"))
        self.uvtt_grid_offset_x_spinbox = QDoubleSpinBox()
        self.uvtt_grid_offset_x_spinbox.setDecimals(1)
        self.uvtt_grid_offset_x_spinbox.setSingleStep(1.0)
        self.uvtt_grid_offset_x_spinbox.setValue(0.0)
        self.uvtt_grid_offset_x_spinbox.setToolTip("Horizontal grid offset in pixels")
        grid_offset_layout.addWidget(self.uvtt_grid_offset_x_spinbox)
        independent_grid_layout.addLayout(grid_offset_layout)
        
        grid_offset_y_layout = QHBoxLayout()
        grid_offset_y_layout.addWidget(QLabel("Grid Offset Y:"))
        self.uvtt_grid_offset_y_spinbox = QDoubleSpinBox()
        self.uvtt_grid_offset_y_spinbox.setDecimals(1)
        self.uvtt_grid_offset_y_spinbox.setSingleStep(1.0)
        self.uvtt_grid_offset_y_spinbox.setValue(0.0)
        self.uvtt_grid_offset_y_spinbox.setToolTip("Vertical grid offset in pixels")
        grid_offset_y_layout.addWidget(self.uvtt_grid_offset_y_spinbox)
        independent_grid_layout.addLayout(grid_offset_y_layout)
        
        export_layout.addWidget(self.uvtt_independent_grid_group)
        
        # Initially hide independent grid settings since "Same as Overlay" is checked by default
        self.uvtt_independent_grid_group.setVisible(False)
        
        # Grid overlay section
        overlay_label = QLabel("Grid Overlay:")
        overlay_label.setProperty("headerLabel", True)
        export_layout.addWidget(overlay_label)
        
        self.uvtt_show_grid_overlay = QCheckBox("Show Grid Overlay")
        self.uvtt_show_grid_overlay.setChecked(False)
        self.uvtt_show_grid_overlay.setToolTip("Show visual grid overlay during UVTT preview")
        self.uvtt_show_grid_overlay.stateChanged.connect(self.on_grid_overlay_toggled)
        export_layout.addWidget(self.uvtt_show_grid_overlay)
        
        overlay_size_layout = QHBoxLayout()
        overlay_size_layout.addWidget(QLabel("Overlay Grid Size:"))
        self.uvtt_overlay_grid_size_spinbox = QSpinBox()
        self.uvtt_overlay_grid_size_spinbox.setRange(10, 500)
        self.uvtt_overlay_grid_size_spinbox.setSingleStep(1)
        self.uvtt_overlay_grid_size_spinbox.setValue(70)
        self.uvtt_overlay_grid_size_spinbox.setToolTip("Size of visual grid overlay in pixels")
        self.uvtt_overlay_grid_size_spinbox.valueChanged.connect(self.on_grid_overlay_setting_changed)
        overlay_size_layout.addWidget(self.uvtt_overlay_grid_size_spinbox)
        export_layout.addLayout(overlay_size_layout)
        
        overlay_offset_x_layout = QHBoxLayout()
        overlay_offset_x_layout.addWidget(QLabel("Overlay Offset X:"))
        self.uvtt_overlay_offset_x_spinbox = QDoubleSpinBox()
        self.uvtt_overlay_offset_x_spinbox.setRange(-500.0, 500.0)
        self.uvtt_overlay_offset_x_spinbox.setDecimals(1)
        self.uvtt_overlay_offset_x_spinbox.setSingleStep(1.0)
        self.uvtt_overlay_offset_x_spinbox.setValue(0.0)
        self.uvtt_overlay_offset_x_spinbox.setToolTip("Horizontal offset for grid overlay in pixels")
        self.uvtt_overlay_offset_x_spinbox.valueChanged.connect(self.on_grid_overlay_setting_changed)
        overlay_offset_x_layout.addWidget(self.uvtt_overlay_offset_x_spinbox)
        export_layout.addLayout(overlay_offset_x_layout)
        
        overlay_offset_y_layout = QHBoxLayout()
        overlay_offset_y_layout.addWidget(QLabel("Overlay Offset Y:"))
        self.uvtt_overlay_offset_y_spinbox = QDoubleSpinBox()
        self.uvtt_overlay_offset_y_spinbox.setRange(-500.0, 500.0)
        self.uvtt_overlay_offset_y_spinbox.setDecimals(1)
        self.uvtt_overlay_offset_y_spinbox.setSingleStep(1.0)
        self.uvtt_overlay_offset_y_spinbox.setValue(0.0)
        self.uvtt_overlay_offset_y_spinbox.setToolTip("Vertical offset for grid overlay in pixels")
        self.uvtt_overlay_offset_y_spinbox.valueChanged.connect(self.on_grid_overlay_setting_changed)
        overlay_offset_y_layout.addWidget(self.uvtt_overlay_offset_y_spinbox)
        export_layout.addLayout(overlay_offset_y_layout)
        
        uvtt_layout.addWidget(export_group)
        
        # === Export Presets Section ===
        presets_group = QGroupBox("Export Presets")
        presets_layout = QVBoxLayout(presets_group)
        
        # Preset dropdown
        self.uvtt_export_preset_combo = QComboBox()
        self.uvtt_export_preset_combo.addItem("-- Select Preset --")
        self.uvtt_export_preset_combo.currentIndexChanged.connect(self.preset_manager.load_export_preset_selected)
        presets_layout.addWidget(self.uvtt_export_preset_combo)
        
        # Preset buttons
        preset_buttons_layout = QHBoxLayout()
        preset_buttons_layout.addStretch(1)
        
        self.uvtt_save_export_preset_button = QPushButton("Save")
        self.uvtt_save_export_preset_button.setToolTip("Save current export settings as preset")
        self.uvtt_save_export_preset_button.clicked.connect(self.preset_manager.save_export_preset_dialog)
        preset_buttons_layout.addWidget(self.uvtt_save_export_preset_button)
        
        self.uvtt_manage_export_preset_button = QPushButton("Manage")
        self.uvtt_manage_export_preset_button.setToolTip("Manage export presets")
        self.uvtt_manage_export_preset_button.clicked.connect(self.preset_manager.manage_export_presets)
        preset_buttons_layout.addWidget(self.uvtt_manage_export_preset_button)
        
        presets_layout.addLayout(preset_buttons_layout)
        uvtt_layout.addWidget(presets_group)
        
        # === Generate Button ===
        self.uvtt_generate_button = QPushButton("Generate Walls")
        self.uvtt_generate_button.setObjectName("generateWallsButton")
        self.uvtt_generate_button.setMinimumHeight(45)
        self.uvtt_generate_button.clicked.connect(self.generate_uvtt_walls)
        uvtt_layout.addWidget(self.uvtt_generate_button)
        
        # Add stretch at bottom to push everything to top
        uvtt_layout.addStretch()
        
        self.uvtt_group.setVisible(False)
        self.right_layout.addWidget(self.uvtt_group)
        
    def generate_uvtt_walls(self):
        """Generate UVTT walls using current detection and export settings."""
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        # Gather export parameters from UVTT panel
        # Determine grid snapping settings based on new controls
        grid_enabled = self.uvtt_enable_grid.isChecked()
        same_as_overlay = self.uvtt_same_as_overlay.isChecked()
        
        if not grid_enabled:
            # No grid snapping
            grid_size = 0
            grid_offset_x = 0.0
            grid_offset_y = 0.0
            allow_half_grid = True
        elif same_as_overlay:
            # Use overlay settings for snapping
            grid_size = int(self.uvtt_overlay_grid_size_spinbox.value())
            grid_offset_x = self.uvtt_overlay_offset_x_spinbox.value()
            grid_offset_y = self.uvtt_overlay_offset_y_spinbox.value()
            allow_half_grid = False  # Use full grid positions for overlay mode
        else:
            # Use independent grid settings
            grid_size = self.uvtt_grid_size_spinbox.value()
            grid_offset_x = self.uvtt_grid_offset_x_spinbox.value()
            grid_offset_y = self.uvtt_grid_offset_y_spinbox.value()
            allow_half_grid = self.uvtt_allow_half_grid.isChecked()
        
        export_params = {
            'simplify_tolerance': self.uvtt_tolerance_spinbox.value(),
            'max_wall_length': self.uvtt_max_length_spinbox.value(),
            'max_walls': self.uvtt_max_walls_spinbox.value(),
            'merge_distance': self.uvtt_merge_distance_spinbox.value(),
            'angle_tolerance': self.uvtt_angle_tolerance_spinbox.value(),
            'max_gap': self.uvtt_max_gap_spinbox.value(),
            'grid_size': grid_size,
            'allow_half_grid': allow_half_grid,
            'grid_offset_x': grid_offset_x,
            'grid_offset_y': grid_offset_y,
            'show_grid_overlay': self.uvtt_show_grid_overlay.isChecked(),
            'overlay_grid_size': self.uvtt_overlay_grid_size_spinbox.value(),
            'overlay_offset_x': self.uvtt_overlay_offset_x_spinbox.value(),
            'overlay_offset_y': self.uvtt_overlay_offset_y_spinbox.value()
        }
        
        # Store export parameters for later use
        self.current_export_settings = export_params
        
        # Determine which walls to export (contours or mask)
        walls_to_export = None
        
        # Get original image dimensions for proper scaling
        if self.original_image is not None:
            image_shape = self.original_image.shape
        else:
            image_shape = self.current_image.shape
            
        # Always use the mask_layer if it exists
        if self.mask_layer is not None:
            # Extract contours from the mask - use alpha channel to determine walls
            alpha_mask = self.mask_layer[:, :, 3].copy()
            
            # If we're working with a scaled image, we need to scale the mask back to original size
            if self.scale_factor != 1.0:
                alpha_mask = cv2.resize(alpha_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                
            walls_to_export = alpha_mask
        elif self.current_contours:
            # Use detected contours directly
            walls_to_export = self.current_contours
        else:
            QMessageBox.warning(self, "No Walls Detected", 
                              "Please detect walls first using the detection tool before generating UVTT walls.")
            return
        
        # Store parameters for the export panel
        self.uvtt_export_params = {
            'walls_to_export': walls_to_export,
            'image_shape': image_shape,
            **export_params
        }
        
        # Call the export panel to generate and preview the walls
        if hasattr(self, 'export_panel'):
            self.export_panel.preview_uvtt_walls()
        
    def on_tool_changed(self, button):
        """Handle tool selection changes with mask hierarchy management."""
        tool_id = self.tool_group.id(button)
        
        # === MASK HIERARCHY MANAGEMENT ===
        # Detection mode has lowest priority - switching to it clears all masks
        if tool_id == 0:  # Switching TO Detection tool
            # Warn if there are masks/drawings that will be lost
            if self.mask_layer is not None or (hasattr(self, 'history') and len(self.history) > 0):
                from PyQt6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self,
                    "Clear Drawings?",
                    "Switching to Detection mode will clear any current drawings.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    # Restore previous tool selection
                    if hasattr(self, '_last_tool_id'):
                        for btn_id in [0, 1, 2]:
                            btn = self.tool_group.button(btn_id)
                            if btn_id == self._last_tool_id:
                                btn.setChecked(True)
                                break
                    return
                
                # Clear the mask layer and history
                self.mask_layer = None
                self.history.clear()
                
        elif tool_id == 1:  # Switching TO Paint tool
            # Check if switching from Walls mode (UVTT preview active)
            if hasattr(self, '_last_tool_id') and self._last_tool_id == 2 and self.uvtt_preview_active:
                from PyQt6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self,
                    "Clear Wall Preview?",
                    "Switching to Draw mode will clear the current wall preview but keep your drawings.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    # Restore UVTT tool selection
                    self.uvtt_tool_btn.setChecked(True)
                    return
                
                # Clear UVTT preview but keep mask layer
                if hasattr(self, 'export_panel'):
                    self.export_panel.cancel_uvtt_preview()
                self.uvtt_preview_active = False
                
                # Refresh display to show the preserved mask layer
                if hasattr(self, 'image_processor'):
                    self.image_processor.update_image()
                print("Cleared UVTT preview while preserving mask layer")
            
            # Bake detection results into mask layer if we have contours
            if self.current_contours and self.mask_layer is None and self.current_image is not None:
                # Use mask_processor to properly handle scaling
                self.mask_processor.bake_contours_to_mask()
                print("Baked detection contours into mask layer with proper scaling")
        
        # Store current tool for potential rollback
        self._last_tool_id = tool_id
        
        # === UI UPDATES ===
        # Hide all property groups
        self.paint_group.setVisible(False)
        self.uvtt_group.setVisible(False)
        
        # Hide detection components
        self.edge_detection_radio.setVisible(False)
        self.color_detection_radio.setVisible(False)
        self.color_section.setVisible(False)
        
        # Hide all mode radio buttons initially
        self.deletion_mode_radio.setVisible(False)
        self.thin_mode_radio.setVisible(False)
        self.color_selection_mode_radio.setVisible(False)
        self.edit_mask_mode_radio.setVisible(False)
        
        # Hide Tool bar label and separators
        self.mode_label.setVisible(False)
        self.tool_separator_top.setVisible(False)
        self.tool_separator_bottom.setVisible(False)
        
        # Hide all detection-specific widgets
        for widget_name in ['min_area_mode_layout', 'merge_options_layout', 'hatching_layout', 
                            'high_res_checkbox', 'presets_container']:
            if hasattr(self, widget_name):
                widget = getattr(self, widget_name)
                if hasattr(widget, 'setVisible'):
                    widget.setVisible(False)
                elif isinstance(widget, QHBoxLayout) or isinstance(widget, QVBoxLayout):
                    # For layouts, hide their parent widget if it exists
                    for i in range(widget.count()):
                        item = widget.itemAt(i)
                        if item.widget():
                            item.widget().setVisible(False)
        
        # Hide all sliders
        for slider_name, slider_widgets in self.sliders.items():
            slider_widgets['container'].setVisible(False)
        
        # Hide light detection group (added in setup_detection_properties)
        if hasattr(self, 'light_group'):
            self.light_group.setVisible(False)
        
        # Reset modes
        self.deletion_mode_enabled = False
        self.thin_mode_enabled = False
        self.color_selection_mode_enabled = False
        self.edit_mask_mode_enabled = False
        self.uvtt_preview_active = False
        
        if tool_id == 0:  # Detection tool
            # Show detection mode controls and color section if needed
            self.edge_detection_radio.setVisible(True)
            self.color_detection_radio.setVisible(True)
            if self.color_detection_radio.isChecked():
                self.color_section.setVisible(True)
            
            # Show Tool bar and detection tool modes: deletion, thin, color pick
            self.mode_label.setVisible(True)
            self.tool_separator_top.setVisible(True)
            self.tool_separator_bottom.setVisible(True)
            
            self.deletion_mode_radio.setVisible(True)
            self.thin_mode_radio.setVisible(True)
            # Only show color selection mode if in color detection mode
            if self.color_detection_radio.isChecked():
                self.color_selection_mode_radio.setVisible(True)
            else:
                self.color_selection_mode_radio.setVisible(False)
            self.edit_mask_mode_radio.setVisible(False)
            
            # Default to deletion mode if no valid mode is selected
            if not (self.deletion_mode_radio.isChecked() or 
                    self.thin_mode_radio.isChecked() or 
                    (self.color_selection_mode_radio.isChecked() and self.color_detection_radio.isChecked())):
                self.deletion_mode_radio.setChecked(True)
            
            # Set appropriate mode based on current selection
            if self.deletion_mode_radio.isChecked():
                self.deletion_mode_enabled = True
            elif self.thin_mode_radio.isChecked():
                self.thin_mode_enabled = True
            elif self.color_selection_mode_radio.isChecked():
                self.color_selection_mode_enabled = True
            
            # Show all detection-specific widgets
            for widget_name in ['min_area_mode_layout', 'merge_options_layout', 'hatching_layout',
                                'high_res_checkbox', 'presets_container']:
                if hasattr(self, widget_name):
                    widget = getattr(self, widget_name)
                    if hasattr(widget, 'setVisible'):
                        widget.setVisible(True)
                    elif isinstance(widget, QHBoxLayout) or isinstance(widget, QVBoxLayout):
                        # For layouts, show their parent widgets
                        for i in range(widget.count()):
                            item = widget.itemAt(i)
                            if item.widget():
                                item.widget().setVisible(True)
            
            # Show all sliders
            for slider_name, slider_widgets in self.sliders.items():
                slider_widgets['container'].setVisible(True)
            
            # Show light detection group
            if hasattr(self, 'light_group'):
                self.light_group.setVisible(True)
            
            # Update detection if image is loaded
            if self.current_image is not None:
                self.image_processor.update_image()
                
        elif tool_id == 1:  # Paint tool
            self.paint_group.setVisible(True)
            
            # Hide all mode buttons for Paint tool - not needed since we have Paint Settings
            self.deletion_mode_radio.setVisible(False)
            self.thin_mode_radio.setVisible(False)
            self.color_selection_mode_radio.setVisible(False)
            self.edit_mask_mode_radio.setVisible(False)
            
            # Enable edit mask mode internally
            self.edit_mask_mode_enabled = True
            
            # Refresh display to ensure mask layer is shown (especially when switching from UVTT)
            if self.current_image is not None and hasattr(self, 'image_processor'):
                self.image_processor.update_image()
            
        elif tool_id == 2:  # UVTT editor tool
            self.uvtt_group.setVisible(True)
            
            # Hide all mode buttons for UVTT tool
            self.deletion_mode_radio.setVisible(False)
            self.thin_mode_radio.setVisible(False)
            self.color_selection_mode_radio.setVisible(False)
            self.edit_mask_mode_radio.setVisible(False)
            
            # Show presets container in UVTT mode too (for detection presets)
            if hasattr(self, 'presets_container'):
                self.presets_container.setVisible(True)
            
            self.uvtt_preview_active = True
            # Show UVTT preview if walls exist, otherwise show grid overlay on main display
            if hasattr(self, 'export_panel') and hasattr(self, 'uvtt_walls_preview') and self.uvtt_walls_preview:
                self.export_panel.display_uvtt_preview()
            else:
                # Show grid overlay on main image when no walls are generated yet
                self.refresh_uvtt_grid_display()
        
        # Update cursor and display
        self.update_cursor_for_mode()
        
    def update_cursor_for_mode(self):
        """Update cursor based on current mode."""
        if self.edit_mask_mode_enabled:
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        elif self.color_selection_mode_enabled:
            self.image_label.setCursor(Qt.CursorShape.PointingHandCursor)
        elif self.deletion_mode_enabled or self.thin_mode_enabled:
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            
    def on_grid_overlay_toggled(self, state=None):
        """Handle grid overlay toggle state changes.

        Accepts the optional `state` argument from QCheckBox.stateChanged signal.
        """
        # Update export parameters with current toggle state if they exist
        if hasattr(self, 'uvtt_export_params') and self.uvtt_export_params:
            self.uvtt_export_params['show_grid_overlay'] = self.uvtt_show_grid_overlay.isChecked()
            
        # Refresh display based on current state
        if hasattr(self, 'uvtt_preview_active') and self.uvtt_preview_active:
            # If we have walls generated and are in preview mode, use export panel preview
            if (hasattr(self, 'export_panel') and self.export_panel and 
                hasattr(self, 'uvtt_walls_preview') and self.uvtt_walls_preview):
                self.export_panel.display_uvtt_preview()
            else:
                # Otherwise show grid on main display
                self.refresh_uvtt_grid_display()
        else:
            # For all other modes (detection, drawing, etc.), refresh the main display
            # This will use the new refresh_display() logic that checks grid overlay setting
            self.refresh_display()
            
    def on_grid_overlay_setting_changed(self):
        """Handle changes to grid overlay size/offset settings."""
        # Only refresh if we're in UVTT mode
        if not (hasattr(self, 'uvtt_preview_active') and self.uvtt_preview_active):
            return
            
        # If we have walls generated and are in preview mode, use export panel preview
        if (hasattr(self, 'export_panel') and self.export_panel and 
            hasattr(self, 'uvtt_walls_preview') and self.uvtt_walls_preview):
            self.export_panel.display_uvtt_preview()
        else:
            # Otherwise show grid on main display
            self.refresh_uvtt_grid_display()
            
    def refresh_uvtt_grid_display(self):
        """Refresh the main image display with grid overlay when in UVTT mode but no walls generated."""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        # Ensure we have minimal export params for the grid overlay system to work
        if not hasattr(self, 'uvtt_export_params') or not self.uvtt_export_params:
            self.uvtt_export_params = {}
            
        # Update the grid overlay setting in export params
        self.uvtt_export_params['show_grid_overlay'] = (
            hasattr(self, 'uvtt_show_grid_overlay') and self.uvtt_show_grid_overlay.isChecked()
        )
        
        # Update overlay parameters in export params from side panel controls
        if hasattr(self, 'uvtt_overlay_grid_size_spinbox'):
            self.uvtt_export_params['overlay_grid_size'] = int(self.uvtt_overlay_grid_size_spinbox.value())
        if hasattr(self, 'uvtt_overlay_offset_x_spinbox'):
            self.uvtt_export_params['overlay_offset_x'] = self.uvtt_overlay_offset_x_spinbox.value()
        if hasattr(self, 'uvtt_overlay_offset_y_spinbox'):
            self.uvtt_export_params['overlay_offset_y'] = self.uvtt_overlay_offset_y_spinbox.value()
            
        # Use the existing refresh_display method which already handles grid overlays correctly
        self.refresh_display()
        
    def add_uvtt_grid_overlay(self, image):
        """Add UVTT grid overlay to an image using current side panel settings."""
        # Get grid parameters from side panel controls
        overlay_size = 70  # default
        overlay_offset_x = 0.0  # default
        overlay_offset_y = 0.0  # default
        
        if hasattr(self, 'uvtt_overlay_grid_size_spinbox'):
            overlay_size = int(self.uvtt_overlay_grid_size_spinbox.value())
            
        if hasattr(self, 'uvtt_overlay_offset_x_spinbox'):
            overlay_offset_x = self.uvtt_overlay_offset_x_spinbox.value()
            
        if hasattr(self, 'uvtt_overlay_offset_y_spinbox'):
            overlay_offset_y = self.uvtt_overlay_offset_y_spinbox.value()
        
        if overlay_size <= 0:
            return image
        
        overlay_image = image.copy()
        height, width = overlay_image.shape[:2]
        
        # Grid color (light gray, semi-transparent)
        grid_color = (128, 128, 128)  # Gray in RGB
        thickness = 1
        
        # Draw vertical lines with offset
        start_x = int(overlay_offset_x) % overlay_size
        for x in range(start_x, width, overlay_size):
            cv2.line(overlay_image, (x, 0), (x, height), grid_color, thickness)
        
        # Draw horizontal lines with offset
        start_y = int(overlay_offset_y) % overlay_size
        for y in range(start_y, height, overlay_size):
            cv2.line(overlay_image, (0, y), (width, y), grid_color, thickness)
        
        return overlay_image
        
    def on_grid_snapping_toggled(self, state=None):
        """Handle grid snapping enable/disable."""
        # Update grid snapping controls visibility based on enable state
        enabled = hasattr(self, 'uvtt_enable_grid') and self.uvtt_enable_grid.isChecked()
        
        # Enable/disable the "Same as Overlay" checkbox
        if hasattr(self, 'uvtt_same_as_overlay'):
            self.uvtt_same_as_overlay.setEnabled(enabled)
            
        # If grid is disabled, hide independent settings
        if not enabled and hasattr(self, 'uvtt_independent_grid_group'):
            self.uvtt_independent_grid_group.setVisible(False)
        elif enabled:
            # If enabled, check "Same as Overlay" setting to show/hide independent settings
            self.on_same_as_overlay_toggled()
    
    def on_same_as_overlay_toggled(self, state=None):
        """Handle 'Same as Overlay' checkbox toggle."""
        # Only show independent grid settings if grid is enabled and "Same as Overlay" is unchecked
        grid_enabled = hasattr(self, 'uvtt_enable_grid') and self.uvtt_enable_grid.isChecked()
        same_as_overlay = hasattr(self, 'uvtt_same_as_overlay') and self.uvtt_same_as_overlay.isChecked()
        
        show_independent = grid_enabled and not same_as_overlay
        
        if hasattr(self, 'uvtt_independent_grid_group'):
            self.uvtt_independent_grid_group.setVisible(show_independent)
            
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Auto-Wall",
                         f"Auto-Wall v{self.app_version}\n\n"
                         "Battle Map Wall Detection Tool\n\n"
                         "Extract walls and light sources from battle maps\n"
                         "for use in Virtual Tabletop applications.")

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
    

    


    def refresh_display(self):
        """Refresh the image display with current processed image and any overlays."""
        if not hasattr(self, 'processed_image') or self.processed_image is None:
            return
            
        display_image = self.processed_image.copy()
        
        # Check if grid overlay should be shown
        show_grid = False
        
        # Priority 1: Export panel mode (UVTT preview)
        if hasattr(self, 'uvtt_export_params') and self.uvtt_export_params:
            if (hasattr(self, 'export_panel') and 
                hasattr(self.export_panel, 'add_grid_overlay') and
                self.uvtt_export_params.get('show_grid_overlay', False)):
                display_image = self.export_panel.add_grid_overlay(display_image)
                show_grid = True
        
        # Priority 2: Side panel grid overlay setting (for all modes including detect/draw)
        if (not show_grid and 
            hasattr(self, 'uvtt_show_grid_overlay') and 
            self.uvtt_show_grid_overlay.isChecked()):
            from src.utils.debug_logger import log_debug
            display_image = self.add_uvtt_grid_overlay(display_image)
        elif not show_grid:
            from src.utils.debug_logger import log_debug
        
        # Display the final image
        self.image_processor.display_image(display_image, preserve_view=True)

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