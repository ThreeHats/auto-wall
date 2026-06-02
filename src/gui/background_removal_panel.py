import cv2
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QWidget, QProgressBar
)
from src.core.background_remover import (
    REMBG_MODELS, DEFAULT_MODEL,
    REMBG_RESOLUTIONS, DEFAULT_RESOLUTION,
    get_available_devices,
)


class BackgroundRemovalPanel:
    """UI logic for background removal controls in the detection panel."""

    def __init__(self, app):
        self.app = app

    def setup_ui(self):
        """Create and wire all background removal widgets into the right panel."""
        app = self.app

        # Main container layout
        app.bg_removal_layout = QVBoxLayout()
        app.right_layout.addLayout(app.bg_removal_layout)

        # Enable/disable checkbox
        app.bg_removal_checkbox = QCheckBox("Background Removal")
        app.bg_removal_checkbox.setChecked(False)
        app.bg_removal_checkbox.toggled.connect(self.toggle_bg_removal)
        app.bg_removal_layout.addWidget(app.bg_removal_checkbox)

        # Options container (shown/hidden with checkbox)
        app.bg_removal_options = QWidget()
        app.bg_removal_options_layout = QVBoxLayout(app.bg_removal_options)
        app.bg_removal_options_layout.setContentsMargins(0, 0, 0, 0)
        app.bg_removal_layout.addWidget(app.bg_removal_options)

        # Model selector
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        app.bg_removal_model_combo = QComboBox()
        for model_id, display_name in REMBG_MODELS:
            app.bg_removal_model_combo.addItem(display_name, model_id)
        default_idx = next(
            (i for i, (mid, _) in enumerate(REMBG_MODELS) if mid == DEFAULT_MODEL), 0
        )
        app.bg_removal_model_combo.setCurrentIndex(default_idx)
        app.bg_removal_model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_row.addWidget(app.bg_removal_model_combo)
        app.bg_removal_options_layout.addLayout(model_row)

        # Resolution selector
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        app.bg_removal_res_combo = QComboBox()
        for res_value, display_name in REMBG_RESOLUTIONS:
            app.bg_removal_res_combo.addItem(display_name, res_value)
        default_res_idx = next(
            (i for i, (v, _) in enumerate(REMBG_RESOLUTIONS) if v == DEFAULT_RESOLUTION), 0
        )
        app.bg_removal_res_combo.setCurrentIndex(default_res_idx)
        res_row.addWidget(app.bg_removal_res_combo)
        app.bg_removal_options_layout.addLayout(res_row)

        # Device selector (CPU / GPU 0 / GPU 1 / ...)
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        app.bg_removal_device_combo = QComboBox()
        devices = get_available_devices()
        for device_id, display_name in devices:
            app.bg_removal_device_combo.addItem(display_name, device_id)
        # Default to first GPU if available, otherwise CPU
        default_device_idx = 1 if len(devices) > 1 else 0
        app.bg_removal_device_combo.setCurrentIndex(default_device_idx)
        device_row.addWidget(app.bg_removal_device_combo)
        app.bg_removal_options_layout.addLayout(device_row)

        # Remove Background button
        app.bg_removal_run_button = QPushButton("Remove Background")
        app.bg_removal_run_button.clicked.connect(self.run_removal)
        app.bg_removal_options_layout.addWidget(app.bg_removal_run_button)

        # Preview toggle
        app.bg_removal_preview_checkbox = QCheckBox("Show Preview")
        app.bg_removal_preview_checkbox.setChecked(False)
        app.bg_removal_preview_checkbox.toggled.connect(self.toggle_preview)
        app.bg_removal_options_layout.addWidget(app.bg_removal_preview_checkbox)

        # Status label
        app.bg_removal_status = QLabel("Ready")
        app.bg_removal_options_layout.addWidget(app.bg_removal_status)

        # Progress bar (hidden until processing)
        app.bg_removal_progress = QProgressBar()
        app.bg_removal_progress.setRange(0, 0)  # Indeterminate
        app.bg_removal_progress.setVisible(False)
        app.bg_removal_options_layout.addWidget(app.bg_removal_progress)

        # Initially hide options
        app.bg_removal_options.setVisible(False)

    def toggle_bg_removal(self, checked):
        """Enable/disable background removal and re-run detection."""
        self.app.bg_removal_options.setVisible(checked)
        if self.app.current_image is not None:
            # Always re-detect: the source image for detection changes
            self.app.image_processor.update_image()

    def run_removal(self):
        """Start the background removal worker."""
        if self.app.current_image is None:
            return

        model_name = self.app.bg_removal_model_combo.currentData()
        max_dim = self.app.bg_removal_res_combo.currentData()
        device = self.app.bg_removal_device_combo.currentData()

        self.app.bg_removal_run_button.setEnabled(False)
        self.app.bg_removal_progress.setVisible(True)
        self.update_status("Starting...")

        self.app.background_remover.start_removal(
            self.app.current_image.copy(), model_name, max_dim, device
        )

    def on_model_changed(self):
        """Invalidate cached result when model changes."""
        self.app.bg_removed_image = None
        self.update_status("Ready (model changed)")

    def toggle_preview(self, checked):
        """Toggle bg-removed preview. Re-runs the display pipeline so contour
        highlighting and other overlays continue to work correctly."""
        if self.app.current_image is not None:
            self.app.contour_processor.update_display_from_contours()

    def update_status(self, msg):
        """Update status label."""
        self.app.bg_removal_status.setText(msg)

    def on_removal_finished(self):
        """Handle successful background removal."""
        self.app.bg_removal_run_button.setEnabled(True)
        self.app.bg_removal_progress.setVisible(False)
        self.update_status("Done")

    def on_removal_error(self, error_msg):
        """Handle background removal error."""
        self.app.bg_removal_run_button.setEnabled(True)
        self.app.bg_removal_progress.setVisible(False)
        self.update_status(f"Error: {error_msg}")
