from PyQt6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QCheckBox, QListWidget,
    QDialog, QDialogButtonBox, QFrame, QSpinBox, QDoubleSpinBox,
    QMessageBox, QComboBox
)
import cv2
import json

class ExportPanel:
    def __init__(self, app):
        self.app = app

    def export_to_foundry_vtt(self):
        """Prepare walls for export to Foundry VTT and show a preview."""
        if self.app.current_image is None:
            return
            
        # Determine which walls to export (contours or mask)
        walls_to_export = None
        
        # Get original image dimensions for proper scaling
        if self.app.original_image is not None:
            image_shape = self.app.original_image.shape
        else:
            image_shape = self.app.current_image.shape
            
        # Always use the mask_layer if it exists, regardless of current mode
        if self.app.mask_layer is not None:
            # Extract contours from the mask - use alpha channel to determine walls
            alpha_mask = self.app.mask_layer[:, :, 3].copy()
            
            # If we're working with a scaled image, we need to scale the mask back to original size
            if self.app.scale_factor != 1.0:
                orig_h, orig_w = image_shape[:2]
                alpha_mask = cv2.resize(alpha_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
            walls_to_export = alpha_mask
        elif self.app.current_contours:
            # Use detected contours directly (keep them at working resolution)
            walls_to_export = self.app.current_contours
        else:
            print("No walls to export.")
            return
            
        # Create a single dialog to gather all export parameters
        dialog = QDialog(self.app)
        dialog.setWindowTitle("Export Parameters")
        layout = QVBoxLayout(dialog)

        # Add export preset dropdown at the top
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Export Preset:")
        preset_combo = QComboBox()
        preset_combo.addItem("-- Select Preset --")
        for name in sorted(self.app.preset_manager.export_presets.keys()):
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
            if preset_name in self.app.preset_manager.export_presets:
                preset = self.app.preset_manager.export_presets[preset_name]
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
            new_preset_name = self.app.preset_manager.save_export_preset_from_dialog(
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
                for name in sorted(self.app.preset_manager.export_presets.keys()):
                    preset_combo.addItem(name)
                
                # Select the new preset
                index = preset_combo.findText(new_preset_name)
                if index != -1:
                    preset_combo.setCurrentIndex(index)
                else:
                    preset_combo.setCurrentIndex(current_index)
                    
                preset_combo.blockSignals(False)
        
        save_preset_button.clicked.connect(save_preset_handler)
        manage_presets_button.clicked.connect(self.app.preset_manager.manage_export_presets)

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
        self.app.foundry_export_params = {
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
        self.app.current_export_settings = {
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
        self.app.color_selection_mode_radio.setChecked(True)
        
        # Generate walls for preview
        self.preview_foundry_walls()

    def preview_foundry_walls(self):
        """Generate and display a preview of the Foundry VTT walls."""
        if not self.app.foundry_export_params:
            return
            
        params = self.app.foundry_export_params        
        # Generate walls without saving to file
        from src.wall_detection.mask_editor import contours_to_foundry_walls
        
        if isinstance(params['walls_to_export'], list):  # It's contours
            contours = params['walls_to_export']
            
            # Scale contours to match the image_shape if needed
            if self.app.scale_factor != 1.0 and self.app.original_image is not None:
                # Contours are at working resolution, but image_shape is full resolution
                # Scale contours up to match the full-resolution image_shape
                contours = self.app.contour_processor.scale_contours_to_original(contours, self.app.scale_factor)
            
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
        self.app.foundry_walls_preview = foundry_walls
        
        # Create a preview image showing the walls
        self.display_foundry_preview()
        
        # Enable save/cancel/copy buttons
        self.app.save_foundry_button.setEnabled(True)
        self.app.cancel_foundry_button.setEnabled(True)
        self.app.copy_foundry_button.setEnabled(True)
        
        # Set flag for preview mode
        self.app.foundry_preview_active = True
        
        # Disable detection controls while in preview mode
        self.set_controls_enabled(False)
        
        # Update status with more detailed information
        wall_count = len(foundry_walls)
        self.app.setStatusTip(f"Previewing {wall_count} walls for Foundry VTT. Click 'Save Foundry Walls' to export or 'Copy to Clipboard'.")

    def display_foundry_preview(self):
        """Display a preview of the Foundry VTT walls over the current image."""
        if not self.app.foundry_walls_preview or self.app.current_image is None:
            return
            
        # Use original image for display if available, otherwise use current image
        if self.app.original_image is not None:
            preview_image = self.app.original_image.copy()
        else:
            preview_image = self.app.current_image.copy()
        
        # Convert back to RGB for better visibility
        if len(preview_image.shape) == 2:  # Grayscale
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2BGR)
              # Draw the walls on the preview image
        for wall in self.app.foundry_walls_preview:
            # Get wall coordinates
            start_x, start_y, end_x, end_y = wall["c"]
            
            # Wall coordinates are now already at the correct resolution (full-res)
            # No scaling needed since we scaled the contours before generating walls
            
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
        wall_count = len(self.app.foundry_walls_preview)
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
        if self.app.original_processed_image is None:
            self.app.original_processed_image = preview_image.copy()
          # Update the display with the preview
        self.app.processed_image = preview_image
        self.app.image_processor.display_image(self.app.processed_image, preserve_view=True)

    def save_foundry_preview(self):
        """Save the previewed Foundry VTT walls to a JSON file."""
        if not self.app.foundry_walls_preview:
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
                json.dump(self.app.foundry_walls_preview, f, indent=2)
                
            print(f"Successfully exported {len(self.app.foundry_walls_preview)} walls to {file_path}")
            self.app.setStatusTip(f"Walls exported to {file_path}. Import in Foundry using Walls > Import")
            
            # After successful saving:
            # Disable save/cancel buttons
            self.app.save_foundry_button.setEnabled(False)
            self.app.cancel_foundry_button.setEnabled(False)
            self.app.copy_foundry_button.setEnabled(False)
            
            # Exit preview mode
            self.cancel_foundry_preview()
        except Exception as e:
            print(f"Failed to export walls: {e}")
            self.app.setStatusTip(f"Failed to export walls: {e}")

    def cancel_foundry_preview(self):
        """Cancel the Foundry VTT wall preview and return to normal view."""
        # Disable buttons
        self.app.save_foundry_button.setEnabled(False)
        self.app.cancel_foundry_button.setEnabled(False)
        self.app.copy_foundry_button.setEnabled(False)
        
        # Clear preview-related data
        self.app.foundry_walls_preview = None
        self.app.foundry_export_params = None
        self.app.foundry_preview_active = False
        
        # Re-enable detection controls
        self.set_controls_enabled(True)
        
        # Restore original display
        if self.app.original_processed_image is not None:
            self.app.processed_image = self.app.original_processed_image.copy()
            self.app.image_processor.display_image(self.app.processed_image)
        
        # Update status
        self.app.setStatusTip("Foundry VTT preview canceled")

    def copy_foundry_to_clipboard(self):
        """Copy the Foundry VTT walls JSON to the clipboard."""
        if not self.app.foundry_walls_preview:
            QMessageBox.warning(self.app, "No Walls", "No walls available to copy.")
            return
            
        try:
            # Convert walls to JSON string
            walls_json = json.dumps(self.app.foundry_walls_preview, indent=2)
            
            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(walls_json)
            
            # Show confirmation
            self.app.setStatusTip(f"{len(self.app.foundry_walls_preview)} walls copied to clipboard. Paste in Foundry using Walls > Import.")
            QMessageBox.information(self.app, "Copied to Clipboard", 
                               f"{len(self.app.foundry_walls_preview)} walls copied to clipboard.\n"
                               f"Use Walls > Import JSON in Foundry VTT to import them.")
        except Exception as e:
            QMessageBox.warning(self.app, "Error", f"Failed to copy walls to clipboard: {str(e)}")

    def set_controls_enabled(self, enabled, color_detection_mode=False):
        """Enable or disable detection controls based on preview state."""
        # Disable/enable all sliders
        for slider_name, slider_info in self.app.sliders.items():
            if 'slider' in slider_info:
                slider_info['slider'].setEnabled(enabled)
        
        if not color_detection_mode:
            # Disable/enable detection mode radio buttons
            self.app.edge_detection_radio.setEnabled(enabled)
            self.app.color_detection_radio.setEnabled(enabled)
            
            # Disable/enable mode toggle radio buttons
            self.app.deletion_mode_radio.setEnabled(enabled)
            self.app.color_selection_mode_radio.setEnabled(enabled)
            self.app.edit_mask_mode_radio.setEnabled(enabled)
            self.app.thin_mode_radio.setEnabled(enabled)
        
        # Disable/enable high-res checkbox
        self.app.high_res_checkbox.setEnabled(enabled)
        
        # Disable/enable color management
        self.app.add_color_button.setEnabled(enabled)
        self.app.remove_color_button.setEnabled(enabled)
        self.app.wall_colors_list.setEnabled(enabled)
        
        # If re-enabling, respect color detection mode
        if enabled and self.app.color_detection_radio.isChecked():
            self.app.detection_panel.toggle_detection_mode_radio(True)
