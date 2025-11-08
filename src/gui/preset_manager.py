import os
import json
import copy

from PyQt6.QtWidgets import QInputDialog, QMessageBox
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from src.core.image_processor import ImageProcessor
from src.utils.debug_logger import log_debug, log_info, log_warning, log_error

# Define preset file paths
DETECTION_PRESETS_FILE = "detection_presets.json"
EXPORT_PRESETS_FILE = "export_presets.json"

class PresetManager:
    def __init__(self, app):
        """Initialize the PresetManager with the application instance.
        Args:
            app: The application instance that this manager will interact with.
        """
        self.app = app
        self.detection_presets = {}
        self.export_presets = {}
        self.last_export_settings = None
    
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
        self.app.detection_preset_combo.blockSignals(True) # Prevent triggering load while updating
        current_selection = self.app.detection_preset_combo.currentText()
        self.app.detection_preset_combo.clear()
        # Add a placeholder item first
        self.app.detection_preset_combo.addItem("-- Select Preset --")
        # Add sorted preset names
        sorted_names = sorted(self.detection_presets.keys())
        for name in sorted_names:
            self.app.detection_preset_combo.addItem(name)

        # Try to restore previous selection
        index = self.app.detection_preset_combo.findText(current_selection)
        if index != -1:
            self.app.detection_preset_combo.setCurrentIndex(index)
        else:
            self.app.detection_preset_combo.setCurrentIndex(0) # Select placeholder

        self.app.detection_preset_combo.blockSignals(False)

    def update_export_preset_combo(self):
        """Update the UVTT export preset combo box with current preset names."""
        self.app.uvtt_export_preset_combo.blockSignals(True)
        current_selection = self.app.uvtt_export_preset_combo.currentText()
        self.app.uvtt_export_preset_combo.clear()
        self.app.uvtt_export_preset_combo.addItem("-- Select Preset --")
        for name in sorted(self.export_presets.keys()):
            self.app.uvtt_export_preset_combo.addItem(name)
        
        # Try to restore previous selection
        index = self.app.uvtt_export_preset_combo.findText(current_selection)
        if index != -1:
            self.app.uvtt_export_preset_combo.setCurrentIndex(index)
        else:
            self.app.uvtt_export_preset_combo.setCurrentIndex(0)
            
        self.app.uvtt_export_preset_combo.blockSignals(False)

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
        for name, info in self.app.sliders.items():
            if 'slider' in info:
                settings["sliders"][name] = info['slider'].value()

        # Checkboxes
        settings["checkboxes"]["High Resolution"] = self.app.high_res_checkbox.isChecked()
        settings["checkboxes"]["Merge Contours"] = self.app.merge_contours.isChecked()
        settings["checkboxes"]["Remove Hatching"] = self.app.remove_hatching_checkbox.isChecked()

        # Radio Buttons
        settings["radios"]["Edge Detection"] = self.app.edge_detection_radio.isChecked()
        settings["radios"]["Color Detection"] = self.app.color_detection_radio.isChecked()
        settings["radios"]["Min Area Percentage"] = self.app.min_area_percentage_radio.isChecked()
        settings["radios"]["Min Area Pixels"] = self.app.min_area_pixels_radio.isChecked()

        # Colors
        for i in range(self.app.wall_colors_list.count()):
            item = self.app.wall_colors_list.item(i)
            color_data = item.data(Qt.ItemDataRole.UserRole)
            qcolor = color_data["color"]
            settings["colors"].append({
                "color": [qcolor.red(), qcolor.green(), qcolor.blue()],
                "threshold": color_data["threshold"]
            })

        # Hatching Settings
        settings["hatching"]["color"] = [self.app.hatching_color.red(), self.app.hatching_color.green(), self.app.hatching_color.blue()]
        settings["hatching"]["threshold"] = self.app.hatching_threshold
        settings["hatching"]["width"] = self.app.hatching_width

        # Light Detection Settings
        settings["light_detection"] = {
            "enabled": self.app.enable_light_detection.isChecked(),
            "brightness_threshold": self.app.light_brightness_slider.value(),
            "min_size": self.app.light_min_size_slider.value(),
            "max_size": self.app.light_max_size_slider.value(),
            "merge_distance": self.app.light_merge_distance_slider.value(),
            "colors": []
        }

        # Light Colors
        for i in range(self.app.light_colors_list.count()):
            item = self.app.light_colors_list.item(i)
            color_data = item.data(Qt.ItemDataRole.UserRole)
            qcolor = color_data["color"]
            settings["light_detection"]["colors"].append({
                "color": [qcolor.red(), qcolor.green(), qcolor.blue()],
                "threshold": color_data["threshold"]
            })

        # Grid Settings (for walls mode and overlay)
        settings["grid_settings"] = {
            "enable_grid": self.app.uvtt_enable_grid.isChecked(),
            "same_as_overlay": self.app.uvtt_same_as_overlay.isChecked(),
            "grid_size": self.app.uvtt_grid_size_spinbox.value(),
            "allow_half_grid": self.app.uvtt_allow_half_grid.isChecked(),
            "grid_offset_x": self.app.uvtt_grid_offset_x_spinbox.value(),
            "grid_offset_y": self.app.uvtt_grid_offset_y_spinbox.value(),
            "show_grid_overlay": self.app.uvtt_show_grid_overlay.isChecked(),
            "overlay_grid_size": self.app.uvtt_overlay_grid_size_spinbox.value(),
            "overlay_offset_x": self.app.uvtt_overlay_offset_x_spinbox.value(),
            "overlay_offset_y": self.app.uvtt_overlay_offset_y_spinbox.value()
        }

        return settings

    def get_current_export_settings(self):
        """Gather current export settings into a dictionary from UVTT panel."""
        settings = {
            "simplify_tolerance": self.app.uvtt_tolerance_spinbox.value(),
            "max_wall_length": self.app.uvtt_max_length_spinbox.value(),
            "max_walls": self.app.uvtt_max_walls_spinbox.value(),
            "merge_distance": self.app.uvtt_merge_distance_spinbox.value(),
            "angle_tolerance": self.app.uvtt_angle_tolerance_spinbox.value(),
            "max_gap": self.app.uvtt_max_gap_spinbox.value(),
            "enable_grid": self.app.uvtt_enable_grid.isChecked(),
            "same_as_overlay": self.app.uvtt_same_as_overlay.isChecked(),
            "grid_size": self.app.uvtt_grid_size_spinbox.value(),
            "allow_half_grid": self.app.uvtt_allow_half_grid.isChecked(),
            "grid_offset_x": self.app.uvtt_grid_offset_x_spinbox.value(),
            "grid_offset_y": self.app.uvtt_grid_offset_y_spinbox.value(),
            "show_grid_overlay": self.app.uvtt_show_grid_overlay.isChecked(),
            "overlay_grid_size": self.app.uvtt_overlay_grid_size_spinbox.value(),
            "overlay_offset_x": self.app.uvtt_overlay_offset_x_spinbox.value(),
            "overlay_offset_y": self.app.uvtt_overlay_offset_y_spinbox.value()
        }
        return settings

    def apply_detection_settings(self, settings):
        """Apply settings from a dictionary to the UI and internal state."""
        log_debug(f"apply_detection_settings called with {len(settings) if settings else 0} setting groups")
        if not settings:
            log_warning("Attempted to apply empty settings.")
            return

        # Block signals to prevent unwanted updates during application
        for info in self.app.sliders.values():
            if 'slider' in info: 
                info['slider'].blockSignals(True)
        self.app.high_res_checkbox.blockSignals(True)
        self.app.merge_contours.blockSignals(True)
        self.app.remove_hatching_checkbox.blockSignals(True)
        self.app.edge_detection_radio.blockSignals(True)
        self.app.color_detection_radio.blockSignals(True)
        self.app.min_area_percentage_radio.blockSignals(True)
        self.app.min_area_pixels_radio.blockSignals(True)
        self.app.wall_colors_list.blockSignals(True)
        self.app.hatching_color_button.blockSignals(True)
        self.app.hatching_threshold_slider.blockSignals(True)
        self.app.hatching_width_slider.blockSignals(True)
        
        # Light detection controls
        self.app.enable_light_detection.blockSignals(True)
        self.app.light_brightness_slider.blockSignals(True)
        self.app.light_min_size_slider.blockSignals(True)
        self.app.light_max_size_slider.blockSignals(True)
        self.app.light_merge_distance_slider.blockSignals(True)
        self.app.light_colors_list.blockSignals(True)
        
        # Grid settings controls
        self.app.uvtt_enable_grid.blockSignals(True)
        self.app.uvtt_same_as_overlay.blockSignals(True)
        self.app.uvtt_grid_size_spinbox.blockSignals(True)
        self.app.uvtt_allow_half_grid.blockSignals(True)
        self.app.uvtt_grid_offset_x_spinbox.blockSignals(True)
        self.app.uvtt_grid_offset_y_spinbox.blockSignals(True)
        self.app.uvtt_show_grid_overlay.blockSignals(True)
        self.app.uvtt_overlay_grid_size_spinbox.blockSignals(True)
        self.app.uvtt_overlay_offset_x_spinbox.blockSignals(True)
        self.app.uvtt_overlay_offset_y_spinbox.blockSignals(True)

        try:
            # Apply Radio Buttons FIRST (handle mutually exclusive groups)
            # This ensures min_area mode is set before we apply the slider values
            if "radios" in settings:
                # Set detection mode radio buttons
                if "Edge Detection" in settings["radios"]:
                    self.app.edge_detection_radio.setChecked(settings["radios"]["Edge Detection"])
                if "Color Detection" in settings["radios"]:
                    self.app.color_detection_radio.setChecked(settings["radios"]["Color Detection"])
                
                # Set min area mode radio buttons
                if "Min Area Percentage" in settings["radios"] and settings["radios"]["Min Area Percentage"]:
                    self.app.min_area_percentage_radio.setChecked(True)
                    self.app.using_pixels_mode = False
                elif "Min Area Pixels" in settings["radios"] and settings["radios"]["Min Area Pixels"]:
                    self.app.min_area_pixels_radio.setChecked(True)
                    self.app.using_pixels_mode = True

            # Apply Sliders
            if "sliders" in settings:
                for name, value in settings["sliders"].items():
                    if name in self.app.sliders:
                        # Update slider with the preset value
                        self.app.sliders[name]['slider'].setValue(value)
                        
                        # Update label with proper format based on mode
                        if name == "Min Area":
                            label = self.app.sliders[name]['label']
                            if hasattr(self, 'using_pixels_mode') and self.app.using_pixels_mode:
                                # In pixel mode, show raw value
                                label.setText(f"Min Area: {value} px")
                            elif 'scale' in self.app.sliders[name]:
                                # In percentage mode, apply scale factor
                                scale = self.app.sliders[name]['scale']
                                label.setText(f"Min Area: {value * scale:.1f}")

            # Apply Checkboxes
            if "checkboxes" in settings:
                if "High Resolution" in settings["checkboxes"]:
                    self.app.high_res_checkbox.setChecked(settings["checkboxes"]["High Resolution"])
                if "Merge Contours" in settings["checkboxes"]:
                    self.app.merge_contours.setChecked(settings["checkboxes"]["Merge Contours"])
                if "Remove Hatching" in settings["checkboxes"]:
                    self.app.remove_hatching_checkbox.setChecked(settings["checkboxes"]["Remove Hatching"])

            # Apply Colors
            if "colors" in settings:
                # Clear existing colors
                self.app.wall_colors_list.clear()
                
                # Add all colors from the preset
                for color_data in settings["colors"]:
                    rgb = color_data["color"]
                    qcolor = QColor(rgb[0], rgb[1], rgb[2])
                    threshold = color_data["threshold"]
                    self.app.detection_panel.add_wall_color_to_list(qcolor, threshold)

            # Apply Hatching Settings
            if "hatching" in settings:
                if "color" in settings["hatching"]:
                    rgb = settings["hatching"]["color"]
                    self.app.hatching_color = QColor(rgb[0], rgb[1], rgb[2])
                    self.app.hatching_color_button.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});")
                
                if "threshold" in settings["hatching"]:
                    self.app.hatching_threshold = settings["hatching"]["threshold"]
                    self.app.hatching_threshold_slider.setValue(int(self.app.hatching_threshold * 10))
                    self.app.hatching_threshold_value.setText(f"{self.app.hatching_threshold:.1f}")
                
                if "width" in settings["hatching"]:
                    self.app.hatching_width = settings["hatching"]["width"]
                    self.app.hatching_width_slider.setValue(self.app.hatching_width)
                    self.app.hatching_width_value.setText(str(self.app.hatching_width))

            # Apply Light Detection Settings (always apply defaults for missing settings)
            light_settings = settings.get("light_detection", {})
            
            # Enable/disable light detection with default
            enabled = light_settings.get("enabled", False)
            self.app.enable_light_detection.setChecked(enabled)
            self.app.light_options.setVisible(enabled)
            
            # Light detection parameters with defaults
            brightness_threshold = light_settings.get("brightness_threshold", 80)
            self.app.light_brightness_slider.setValue(brightness_threshold)
            brightness = brightness_threshold / 100.0
            self.app.light_brightness_value.setText(f"{brightness:.2f}")
            
            min_size = light_settings.get("min_size", 5)
            self.app.light_min_size_slider.setValue(min_size)
            self.app.light_min_size_value.setText(str(min_size))
            
            max_size = light_settings.get("max_size", 500)
            self.app.light_max_size_slider.setValue(max_size)
            self.app.light_max_size_value.setText(str(max_size))
            
            merge_distance = light_settings.get("merge_distance", 20)
            self.app.light_merge_distance_slider.setValue(merge_distance)
            self.app.light_merge_distance_value.setText(str(merge_distance))
            
            # Light colors - always reset and apply from preset
            self.app.light_colors_list.clear()
            if "colors" in light_settings:
                for color_data in light_settings["colors"]:
                    rgb = color_data["color"]
                    qcolor = QColor(rgb[0], rgb[1], rgb[2])
                    threshold = color_data["threshold"]
                    self.app.detection_panel.add_light_color_to_list(qcolor, threshold)
            else:
                # Add default light color if no colors in preset
                self.app.detection_panel.add_light_color_to_list(QColor(255, 255, 200), 15.0)

            # Apply Grid Settings (always apply defaults for missing settings)
            grid_settings = settings.get("grid_settings", {})
            log_debug(f"Applying grid settings: {grid_settings}")
            
            # Grid snapping settings with defaults
            enable_grid = grid_settings.get("enable_grid", False)
            log_debug(f"Setting enable_grid to: {enable_grid}")
            self.app.uvtt_enable_grid.setChecked(enable_grid)
            
            same_as_overlay = grid_settings.get("same_as_overlay", True)
            log_debug(f"Setting same_as_overlay to: {same_as_overlay}")
            self.app.uvtt_same_as_overlay.setChecked(same_as_overlay)
            
            self.app.uvtt_grid_size_spinbox.setValue(grid_settings.get("grid_size", 70))
            self.app.uvtt_allow_half_grid.setChecked(grid_settings.get("allow_half_grid", False))
            self.app.uvtt_grid_offset_x_spinbox.setValue(grid_settings.get("grid_offset_x", 0.0))
            self.app.uvtt_grid_offset_y_spinbox.setValue(grid_settings.get("grid_offset_y", 0.0))
            
            # Grid overlay settings with defaults
            show_overlay = grid_settings.get("show_grid_overlay", False)
            log_debug(f"Setting grid overlay to: {show_overlay}")
            self.app.uvtt_show_grid_overlay.setChecked(show_overlay)
            self.app.uvtt_overlay_grid_size_spinbox.setValue(grid_settings.get("overlay_grid_size", 70))
            self.app.uvtt_overlay_offset_x_spinbox.setValue(grid_settings.get("overlay_offset_x", 0.0))
            self.app.uvtt_overlay_offset_y_spinbox.setValue(grid_settings.get("overlay_offset_y", 0.0))
            
            # Update UI visibility based on grid settings
            self.app.on_grid_snapping_toggled()
            self.app.on_same_as_overlay_toggled()
            
            # Always trigger grid overlay refresh (whether enabled or disabled)
            # This ensures the display is updated properly in both cases
            self.app.on_grid_overlay_toggled()

        finally:
            # Unblock signals
            for info in self.app.sliders.values():
                if 'slider' in info:
                    info['slider'].blockSignals(False)
            self.app.high_res_checkbox.blockSignals(False)
            self.app.merge_contours.blockSignals(False)
            self.app.remove_hatching_checkbox.blockSignals(False)
            self.app.edge_detection_radio.blockSignals(False)
            self.app.color_detection_radio.blockSignals(False)
            self.app.min_area_percentage_radio.blockSignals(False)
            self.app.min_area_pixels_radio.blockSignals(False)
            self.app.wall_colors_list.blockSignals(False)
            self.app.hatching_color_button.blockSignals(False)
            self.app.hatching_threshold_slider.blockSignals(False)
            self.app.hatching_width_slider.blockSignals(False)
            
            # Unblock light detection controls
            self.app.enable_light_detection.blockSignals(False)
            self.app.light_brightness_slider.blockSignals(False)
            self.app.light_min_size_slider.blockSignals(False)
            self.app.light_max_size_slider.blockSignals(False)
            self.app.light_merge_distance_slider.blockSignals(False)
            self.app.light_colors_list.blockSignals(False)
            
            # Unblock grid settings controls
            self.app.uvtt_enable_grid.blockSignals(False)
            self.app.uvtt_same_as_overlay.blockSignals(False)
            self.app.uvtt_grid_size_spinbox.blockSignals(False)
            self.app.uvtt_allow_half_grid.blockSignals(False)
            self.app.uvtt_grid_offset_x_spinbox.blockSignals(False)
            self.app.uvtt_grid_offset_y_spinbox.blockSignals(False)
            self.app.uvtt_show_grid_overlay.blockSignals(False)
            self.app.uvtt_overlay_grid_size_spinbox.blockSignals(False)
            self.app.uvtt_overlay_offset_x_spinbox.blockSignals(False)
            self.app.uvtt_overlay_offset_y_spinbox.blockSignals(False)

        # Now that all settings are applied, explicitly call toggle_detection_mode_radio
        # to ensure the UI reflects the detection mode correctly
        if "radios" in settings:
            self.app.detection_panel.toggle_detection_mode_radio(self.app.color_detection_radio.isChecked())

        # Trigger image update after applying settings
        if self.app.current_image is not None:
            self.app.image_processor.update_image()
            # Force refresh display to ensure grid overlay state is correct
            log_debug(f"Grid overlay enabled after preset: {self.app.uvtt_show_grid_overlay.isChecked()}")
            self.app.refresh_display()
        self.app.setStatusTip("Applied detection preset.")

    def apply_export_settings(self, settings):
        """Apply export settings from a dictionary to the UVTT panel spinboxes."""
        if not settings:
            print("Warning: Attempted to apply empty export settings.")
            return
        
        # Block signals during update to prevent triggering changes
        self.app.uvtt_tolerance_spinbox.blockSignals(True)
        self.app.uvtt_max_length_spinbox.blockSignals(True)
        self.app.uvtt_max_walls_spinbox.blockSignals(True)
        self.app.uvtt_merge_distance_spinbox.blockSignals(True)
        self.app.uvtt_angle_tolerance_spinbox.blockSignals(True)
        self.app.uvtt_max_gap_spinbox.blockSignals(True)
        self.app.uvtt_enable_grid.blockSignals(True)
        self.app.uvtt_same_as_overlay.blockSignals(True)
        self.app.uvtt_grid_size_spinbox.blockSignals(True)
        self.app.uvtt_allow_half_grid.blockSignals(True)
        self.app.uvtt_grid_offset_x_spinbox.blockSignals(True)
        self.app.uvtt_grid_offset_y_spinbox.blockSignals(True)
        self.app.uvtt_show_grid_overlay.blockSignals(True)
        self.app.uvtt_overlay_grid_size_spinbox.blockSignals(True)
        self.app.uvtt_overlay_offset_x_spinbox.blockSignals(True)
        self.app.uvtt_overlay_offset_y_spinbox.blockSignals(True)
        
        try:
            # Apply all settings with defaults for missing values
            self.app.uvtt_tolerance_spinbox.setValue(settings.get("simplify_tolerance", 0.001))
            self.app.uvtt_max_length_spinbox.setValue(settings.get("max_wall_length", 50))
            self.app.uvtt_max_walls_spinbox.setValue(settings.get("max_walls", 20000))
            self.app.uvtt_merge_distance_spinbox.setValue(settings.get("merge_distance", 25.0))
            self.app.uvtt_angle_tolerance_spinbox.setValue(settings.get("angle_tolerance", 1.0))
            self.app.uvtt_max_gap_spinbox.setValue(settings.get("max_gap", 10.0))
            
            # Grid settings with defaults
            self.app.uvtt_enable_grid.setChecked(settings.get("enable_grid", False))
            self.app.uvtt_same_as_overlay.setChecked(settings.get("same_as_overlay", True))
            self.app.uvtt_grid_size_spinbox.setValue(settings.get("grid_size", 72))
            self.app.uvtt_allow_half_grid.setChecked(settings.get("allow_half_grid", False))
            self.app.uvtt_grid_offset_x_spinbox.setValue(settings.get("grid_offset_x", 0.0))
            self.app.uvtt_grid_offset_y_spinbox.setValue(settings.get("grid_offset_y", 0.0))
            
            # Grid overlay settings with defaults (these are newer settings)
            self.app.uvtt_show_grid_overlay.setChecked(settings.get("show_grid_overlay", False))
            self.app.uvtt_overlay_grid_size_spinbox.setValue(settings.get("overlay_grid_size", 70))
            self.app.uvtt_overlay_offset_x_spinbox.setValue(settings.get("overlay_offset_x", 0.0))
            self.app.uvtt_overlay_offset_y_spinbox.setValue(settings.get("overlay_offset_y", 0.0))
                
        finally:
            # Unblock signals
            self.app.uvtt_tolerance_spinbox.blockSignals(False)
            self.app.uvtt_max_length_spinbox.blockSignals(False)
            self.app.uvtt_max_walls_spinbox.blockSignals(False)
            self.app.uvtt_merge_distance_spinbox.blockSignals(False)
            self.app.uvtt_angle_tolerance_spinbox.blockSignals(False)
            self.app.uvtt_max_gap_spinbox.blockSignals(False)
            self.app.uvtt_enable_grid.blockSignals(False)
            self.app.uvtt_same_as_overlay.blockSignals(False)
            self.app.uvtt_grid_size_spinbox.blockSignals(False)
            self.app.uvtt_allow_half_grid.blockSignals(False)
            self.app.uvtt_grid_offset_x_spinbox.blockSignals(False)
            self.app.uvtt_grid_offset_y_spinbox.blockSignals(False)
            self.app.uvtt_show_grid_overlay.blockSignals(False)
            self.app.uvtt_overlay_grid_size_spinbox.blockSignals(False)
            self.app.uvtt_overlay_offset_x_spinbox.blockSignals(False)
            self.app.uvtt_overlay_offset_y_spinbox.blockSignals(False)
            
            # Update UI state after applying settings
            self.app.on_grid_snapping_toggled()
            self.app.on_same_as_overlay_toggled()

    def save_detection_preset(self):
        """Save the current detection settings as a new preset."""
        preset_name, ok = QInputDialog.getText(self.app, "Save Detection Preset", "Enter preset name:")
        if ok and preset_name:
            # Check if overwriting a default preset
            if preset_name in self.get_default_detection_presets():
                 reply = QMessageBox.question(self.app, "Overwrite Default Preset?",
                                             f"'{preset_name}' is a default preset. Overwrite it?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                 if reply == QMessageBox.StandardButton.No:
                     return # User cancelled overwrite

            # Check if overwriting an existing user preset
            elif preset_name in self.detection_presets:
                 reply = QMessageBox.question(self.app, "Overwrite Preset?",
                                             f"Preset '{preset_name}' already exists. Overwrite?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                 if reply == QMessageBox.StandardButton.No:
                     return # User cancelled overwrite

            current_settings = self.get_current_detection_settings()
            self.detection_presets[preset_name] = current_settings
            self.update_detection_preset_combo()
            # Select the newly saved preset in the combo box
            index = self.app.detection_preset_combo.findText(preset_name)
            if index != -1:
                self.app.detection_preset_combo.setCurrentIndex(index)
            self.save_presets_to_file() # Persist changes
            self.app.setStatusTip(f"Saved detection preset '{preset_name}'.")
        elif ok and not preset_name:
             QMessageBox.warning(self.app, "Invalid Name", "Preset name cannot be empty.")

    def load_detection_preset_selected(self, index):
        """Load the detection preset selected in the combo box."""
        preset_name = self.app.detection_preset_combo.itemText(index)
        log_debug(f"Preset selection changed: index={index}, name='{preset_name}'")
        if index > 0 and preset_name in self.detection_presets: # Index 0 is placeholder
            log_info(f"Loading detection preset: {preset_name}")
            self.apply_detection_settings(self.detection_presets[preset_name])
            self.app.setStatusTip(f"Loaded detection preset '{preset_name}'.")
        elif index == 0:
            log_debug("Placeholder selected, not loading preset")
            self.app.setStatusTip("Select a detection preset to load.")
        else:
            log_warning(f"Preset '{preset_name}' not found in detection presets")

    def manage_detection_presets(self):
        """Show a dialog or menu to manage (delete) user presets."""
        default_preset_names = set(self.get_default_detection_presets().keys())
        user_preset_names = sorted([name for name in self.detection_presets.keys() if name not in default_preset_names])

        if not user_preset_names:
            QMessageBox.information(self.app, "Manage Presets", "No user-defined presets to manage.")
            return

        preset_to_delete, ok = QInputDialog.getItem(self.app, "Delete User Preset",
                                                    "Select preset to delete:", user_preset_names, 0, False)

        if ok and preset_to_delete:
            reply = QMessageBox.question(self.app, "Confirm Deletion",
                                         f"Are you sure you want to delete the preset '{preset_to_delete}'?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if preset_to_delete in self.detection_presets:
                    del self.detection_presets[preset_to_delete]
                    self.update_detection_preset_combo()
                    self.save_presets_to_file() # Persist deletion
                    self.app.setStatusTip(f"Deleted preset '{preset_to_delete}'.")
                    print(f"Deleted preset: {preset_to_delete}")
                else:
                     QMessageBox.warning(self.app, "Error", f"Preset '{preset_to_delete}' not found.")

    def save_export_preset(self):
        if self.app.current_export_settings is None:
            QMessageBox.warning(self.app, "Save Export Preset", "No export settings available to save.")
            return

        preset_name, ok = QInputDialog.getText(self.app, "Save Export Preset", "Enter preset name:")
        if ok and preset_name:
            self.export_presets[preset_name] = copy.deepcopy(self.app.current_export_settings)
            self.save_presets_to_file()
            self.app.setStatusTip(f"Saved export preset '{preset_name}'")

    def load_export_preset_selected(self, index):
        """Load the export preset selected in the UVTT combo box."""
        preset_name = self.app.uvtt_export_preset_combo.itemText(index)
        if index > 0 and preset_name in self.export_presets:
            settings = self.export_presets[preset_name]
            self.apply_export_settings(settings)
            self.app.current_export_settings = copy.deepcopy(settings)
            self.app.setStatusTip(f"Loaded export preset '{preset_name}'")
            # Refresh display to update grid overlay
            self.app.refresh_display()
        elif index == 0:
            self.app.setStatusTip("Select an export preset to load.")

    def manage_export_presets(self):
        """Show a dialog to manage (delete) export presets."""
        default_preset_names = set(self.get_default_export_presets().keys())
        user_preset_names = sorted([name for name in self.export_presets.keys() if name not in default_preset_names])
        
        if not user_preset_names:
            QMessageBox.information(self.app, "Manage Export Presets", "No user-defined export presets to manage.")
            return

        preset_to_delete, ok = QInputDialog.getItem(self.app, "Delete Export Preset", 
                                                    "Select preset to delete:", user_preset_names, 0, False)
        if ok and preset_to_delete:
            reply = QMessageBox.question(self.app, "Confirm Deletion",
                                         f"Are you sure you want to delete the preset '{preset_to_delete}'?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if preset_to_delete in self.export_presets:
                    del self.export_presets[preset_to_delete]
                    self.update_export_preset_combo()
                    self.save_presets_to_file()
                    self.app.setStatusTip(f"Deleted export preset '{preset_to_delete}'")
                    print(f"Deleted export preset: {preset_to_delete}")
                else:
                    QMessageBox.warning(self.app, "Error", f"Preset '{preset_to_delete}' not found.")

    def save_export_preset_dialog(self):
        """Open a dialog to save current UVTT panel export settings as a preset."""
        current_settings = self.get_current_export_settings()

        preset_name, ok = QInputDialog.getText(self.app, "Save Export Preset", "Enter preset name:")
        if ok and preset_name:
            # Check if overwriting a default preset
            if preset_name in self.get_default_export_presets():
                reply = QMessageBox.question(self.app, "Overwrite Default Preset?",
                                            f"'{preset_name}' is a default preset. Overwrite it?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return  # User cancelled overwrite

            # Check if overwriting an existing user preset
            elif preset_name in self.export_presets:
                reply = QMessageBox.question(self.app, "Overwrite Preset?",
                                            f"Preset '{preset_name}' already exists. Overwrite?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return  # User cancelled overwrite

            # Save the preset
            self.export_presets[preset_name] = copy.deepcopy(current_settings)
            self.update_export_preset_combo()
            
            # Select the newly saved preset in the combo box
            index = self.app.uvtt_export_preset_combo.findText(preset_name)
            if index != -1:
                self.app.uvtt_export_preset_combo.setCurrentIndex(index)
                
            self.save_presets_to_file()
            self.app.current_export_settings = copy.deepcopy(current_settings)
            self.app.setStatusTip(f"Saved export preset '{preset_name}'")
        elif ok and not preset_name:
            QMessageBox.warning(self.app, "Invalid Name", "Preset name cannot be empty.")

    def save_export_preset_from_dialog(self, tolerance, max_length, max_walls, merge_distance, angle_tolerance, max_gap, grid_size, allow_half_grid, grid_offset_x=0.0, grid_offset_y=0.0, show_grid_overlay=False, overlay_grid_size=70, overlay_offset_x=0.0, overlay_offset_y=0.0):
        """Save current export settings from the dialog as a preset."""
        preset_name, ok = QInputDialog.getText(self.app, "Save Export Preset", "Enter preset name:")
        if ok and preset_name:
            # Check if overwriting a default preset
            if preset_name in self.get_default_export_presets():
                reply = QMessageBox.question(self.app, "Overwrite Default Preset?",
                                            f"'{preset_name}' is a default preset. Overwrite it?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return  # User cancelled overwrite

            # Check if overwriting an existing user preset
            elif preset_name in self.export_presets:
                reply = QMessageBox.question(self.app, "Overwrite Preset?",
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
                "allow_half_grid": allow_half_grid,
                "grid_offset_x": grid_offset_x,
                "grid_offset_y": grid_offset_y,
                "show_grid_overlay": show_grid_overlay,
                "overlay_grid_size": overlay_grid_size,
                "overlay_offset_x": overlay_offset_x,
                "overlay_offset_y": overlay_offset_y
            }
            
            # Store the preset and save to file
            self.export_presets[preset_name] = new_preset
            self.save_presets_to_file()
            
            # Also update current_export_settings for consistency
            self.app.current_export_settings = new_preset.copy()
            
            self.app.setStatusTip(f"Saved export preset '{preset_name}'")
            
            # Return the preset name so the dialog can update its dropdown
            return preset_name
        elif ok and not preset_name:
            QMessageBox.warning(self.app, "Invalid Name", "Preset name cannot be empty.")
            
        return None