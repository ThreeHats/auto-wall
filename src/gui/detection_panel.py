from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QSlider, QWidget,
    QColorDialog, QListWidget, QListWidgetItem,
    QDialog, QSpinBox, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QCursor

class DetectionPanel:
    def __init__(self, app):
        self.app = app
    
    # Detection mode and controls
    def toggle_detection_mode_radio(self, checked): # 'checked' parameter is from the signal, might not reflect the final state if called manually
        """Toggle controls visibility based on the currently selected detection mode radio button."""
        # Always check the current state of the radio buttons, ignore the 'checked' parameter from the signal

        if self.app.color_detection_radio.isChecked():
            # Color Detection Mode is active
            self.app.edge_detection_radio.setChecked(False) # Ensure consistency

            # Hide edge detection controls and their labels
            sliders_to_hide = ["Smoothing", "Edge Sensitivity", "Edge Threshold", "Edge Margin"]
            for slider_name, slider_info in self.app.sliders.items():
                # Hide the entire container which includes both slider and label
                if 'container' in slider_info and slider_name in sliders_to_hide:
                    slider_info['container'].setVisible(False)

            # Show color detection controls
            self.app.color_section.setVisible(True)
            self.app.color_selection_mode_radio.setVisible(True) # Show color pick tool

            # Update labels to reflect active/inactive state (optional)
            # self.app.color_section_title.setText("Color Detection:")
            # self.app.color_section_title.setStyleSheet("font-weight: bold;")
        else: # Edge Detection Mode is active (self.app.edge_detection_radio should be checked)
            self.app.color_detection_radio.setChecked(False) # Ensure consistency
            if not self.app.edge_detection_radio.isChecked(): # Double check and force if needed
                 self.app.edge_detection_radio.setChecked(True)

            # Show edge detection controls and their labels
            sliders_to_show = ["Smoothing", "Edge Sensitivity", "Edge Threshold", "Edge Margin"]
            for slider_name, slider_info in self.app.sliders.items():
                # Show the entire container which includes both slider and label
                if 'container' in slider_info and slider_name in sliders_to_show:
                    slider_info['container'].setVisible(True)

            # Hide color detection controls
            self.app.color_section.setVisible(False)
            self.app.color_selection_mode_radio.setVisible(False) # Hide color pick tool


        # Update the detection if an image is loaded
        if self.app.current_image is not None:
            self.app.image_processor.update_image()

    def toggle_light_detection(self, checked):
        """Toggle light detection options visibility."""
        is_enabled = checked == 2  # Qt.CheckState.Checked
        self.app.light_options.setVisible(is_enabled)
        
        # Only show light detection in edge or color detection modes
        if self.app.current_image is not None:
            # Update lights only, don't re-detect contours
            self.app.image_processor.update_lights_only()

    def update_light_brightness(self, value):
        if self.app.current_image is not None:
            self.app.image_processor.update_lights_only()

    def update_light_min_size(self, value):
        if self.app.current_image is not None:
            self.app.image_processor.update_lights_only()

    def update_light_max_size(self, value):
        if self.app.current_image is not None:
            self.app.image_processor.update_lights_only()

    def update_light_merge_distance(self, value):
        if self.app.current_image is not None:
            self.app.image_processor.update_lights_only()

    def add_slider(self, label, min_val, max_val, initial_val, step=1, scale_factor=None):
        """Add a slider with a label and a synced input spinbox."""
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)

        slider_label = QLabel(f"{label}:")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(initial_val)
        slider.setSingleStep(step)

        if scale_factor:
            decimals = 3 if scale_factor < 0.01 else 1
            spinbox = QDoubleSpinBox()
            spinbox.setDecimals(decimals)
            spinbox.setMinimum(round(min_val * scale_factor, decimals))
            spinbox.setMaximum(round(max_val * scale_factor, decimals))
            spinbox.setSingleStep(round(step * scale_factor, decimals))
            spinbox.setValue(round(initial_val * scale_factor, decimals))

            def _on_slider(v, sb=spinbox, lbl=label):
                # Read the current scale at runtime so mode toggles (e.g. Min Area
                # percentage <-> pixels) convert correctly without re-binding closures.
                sf = self.app.sliders[lbl].get('scale', scale_factor)
                d = 3 if sf < 0.01 else 1
                sb.blockSignals(True)
                sb.setValue(round(v * sf, d))
                sb.blockSignals(False)

            def _on_spinbox(v, sl=slider, lbl=label):
                sf = self.app.sliders[lbl].get('scale', scale_factor)
                sl.setValue(round(v / sf))
        else:
            spinbox = QSpinBox()
            spinbox.setMinimum(min_val)
            spinbox.setMaximum(max_val)
            spinbox.setSingleStep(step)
            spinbox.setValue(initial_val)

            def _on_slider(v, sb=spinbox):
                sb.blockSignals(True)
                sb.setValue(v)
                sb.blockSignals(False)

            def _on_spinbox(v, sl=slider):
                sl.setValue(v)

        spinbox.setFixedWidth(70)
        slider.valueChanged.connect(_on_slider)
        slider.valueChanged.connect(self.app.image_processor.update_image)
        spinbox.valueChanged.connect(_on_spinbox)

        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(slider)
        slider_layout.addWidget(spinbox)

        self.app.right_layout.addWidget(slider_container)

        self.app.sliders[label] = {
            'slider': slider,
            'container': slider_container,
            'label': slider_label,
            'spinbox': spinbox,
        }
        if scale_factor:
            self.app.sliders[label]['scale'] = scale_factor

    def update_slider(self, label, label_text, value, scale_factor=None):
        """Update the spinbox for a slider (called on explicit mode changes)."""
        slider_info = self.app.sliders.get(label_text, {})
        spinbox = slider_info.get('spinbox')
        if spinbox is None:
            return
        spinbox.blockSignals(True)
        if scale_factor:
            decimals = 3 if scale_factor < 0.01 else 1
            spinbox.setValue(round(value * scale_factor, decimals))
        else:
            spinbox.setValue(value)
        spinbox.blockSignals(False)

    def toggle_mode(self):
        """Toggle between detection, deletion, color selection, edit mask, and thinning modes."""
        # Check if we need to save state before mode changes
        if self.app.processed_image is not None:
            previous_mode = None
            if hasattr(self, 'edit_mask_mode_enabled') and self.app.edit_mask_mode_enabled:
                previous_mode = 'mask'
            
            # Update mode flags based on radio button states
            self.app.color_selection_mode_enabled = self.app.color_selection_mode_radio.isChecked()
            self.app.deletion_mode_enabled = self.app.deletion_mode_radio.isChecked()
            self.app.edit_mask_mode_enabled = self.app.edit_mask_mode_radio.isChecked()
            self.app.thin_mode_enabled = self.app.thin_mode_radio.isChecked()
            self.app.thicken_mode_enabled = self.app.thicken_mode_radio.isChecked()
            
            # If switching to/from mask mode, save state
            current_mode = 'mask' if self.app.edit_mask_mode_enabled else 'contour'
            if previous_mode != current_mode:
                self.app.mask_processor.save_state()
        else:
            # Just update the mode flags as before
            self.app.color_selection_mode_enabled = self.app.color_selection_mode_radio.isChecked()
            self.app.deletion_mode_enabled = self.app.deletion_mode_radio.isChecked()
            self.app.edit_mask_mode_enabled = self.app.edit_mask_mode_radio.isChecked()
            self.app.thin_mode_enabled = self.app.thin_mode_radio.isChecked()
            self.app.thicken_mode_enabled = self.app.thicken_mode_radio.isChecked()
        
        # Show/hide color selection options
        self.app.color_selection_options.setVisible(self.app.color_selection_mode_enabled)
        
        # Show/hide mask edit options
        if hasattr(self.app, 'mask_edit_options'):
            self.app.mask_edit_options.setVisible(self.app.edit_mask_mode_enabled)
        
        # Show/hide thinning/thickening options
        self.app.thin_options.setVisible(self.app.thin_mode_enabled or self.app.thicken_mode_enabled)
        
        if self.app.deletion_mode_enabled:
            self.app.setStatusTip("Deletion Mode: Click inside contours or on lines to delete them")
            # Store original image for highlighting
            if self.app.processed_image is not None:
                self.app.original_processed_image = self.app.processed_image.copy()
        elif self.app.color_selection_mode_enabled:
            self.app.setStatusTip("Color Selection Mode: Drag to select an area for color extraction")
            # Store original image for selection rectangle
            if self.app.processed_image is not None:
                self.app.original_processed_image = self.app.processed_image.copy()
        elif self.app.edit_mask_mode_enabled:
            self.app.setStatusTip("Edit Mask Mode: Draw or erase on the mask layer")
            # Make sure we have a mask to edit
            if self.app.mask_layer is None and self.app.current_image is not None:
                # Create an empty mask if none exists
                self.app.mask_processor.create_empty_mask()
            if self.app.processed_image is not None:
                self.app.original_processed_image = self.app.processed_image.copy()
                # Display the mask with the image
                self.app.mask_processor.update_display_with_mask()
                
            # Reset brush preview state when entering edit mode
            self.app.brush_preview_active = False
            
            # Get current cursor position to start showing brush immediately
            if self.app.current_image is not None:
                cursor_pos = self.app.image_label.mapFromGlobal(QCursor.pos())
                if self.app.image_label.rect().contains(cursor_pos):
                    self.app.drawing_tools.update_brush_preview(cursor_pos.x(), cursor_pos.y())
        elif self.app.thin_mode_enabled:
            self.app.setStatusTip("Thinning Mode: Click on contours to thin them")
            if self.app.processed_image is not None:
                self.app.original_processed_image = self.app.processed_image.copy()
        elif self.app.thicken_mode_enabled:
            self.app.setStatusTip("Thickening Mode: Click on contours to thicken them")
            if self.app.processed_image is not None:
                self.app.original_processed_image = self.app.processed_image.copy()
        else:
            self.app.setStatusTip("")
            # Clear any highlighting
            self.app.image_label.clear_hover()            # Display normal image without mask
            if self.app.processed_image is not None:
                self.app.refresh_display()
        
        # Clear any selection when switching modes
        self.app.selection_manager.clear_selection()
        
        # Make sure to initialize the drawing position attribute
        if hasattr(self, 'last_drawing_position'):
            self.app.last_drawing_position = None
        else:
            setattr(self, 'last_drawing_position', None)

    def toggle_min_area_mode(self):
        """Toggle between percentage and pixel mode for Min Area."""
        min_area_slider = self.app.sliders["Min Area"]['slider']
        min_area_spinbox = self.app.sliders["Min Area"]['spinbox']

        if self.app.min_area_percentage_radio.isChecked():
            # Switch to percentage mode (0.001% to 25%)
            min_area_slider.setMinimum(1)
            min_area_slider.setMaximum(25000)

            if hasattr(self, 'using_pixels_mode') and self.app.using_pixels_mode and self.app.current_image is not None:
                current_pixel_value = min_area_slider.value()
                image_area = self.app.current_image.shape[0] * self.app.current_image.shape[1]
                if image_area > 0:
                    percentage = (current_pixel_value / image_area) * 100.0
                    slider_value = max(1, min(25000, int(percentage / 0.001)))
                    min_area_slider.setValue(slider_value)

            self.app.using_pixels_mode = False
            # Update the mutable scale so the slider/spinbox closures convert correctly.
            self.app.sliders["Min Area"]['scale'] = 0.001

            # Reconfigure spinbox for percentage display
            min_area_spinbox.blockSignals(True)
            min_area_spinbox.setSuffix("")
            min_area_spinbox.setDecimals(3)
            min_area_spinbox.setMinimum(0.001)
            min_area_spinbox.setMaximum(25.0)
            min_area_spinbox.setSingleStep(0.001)
            min_area_spinbox.setValue(round(min_area_slider.value() * 0.001, 3))
            min_area_spinbox.blockSignals(False)
            print("Switched Min Area mode to Percentage")

        else:  # Pixels mode
            min_area_slider.setMinimum(1)
            min_area_slider.setMaximum(1000)

            if (not hasattr(self, 'using_pixels_mode') or not self.app.using_pixels_mode) and self.app.current_image is not None:
                current_slider_value = min_area_slider.value()
                percentage = current_slider_value * 0.001
                image_area = self.app.current_image.shape[0] * self.app.current_image.shape[1]
                if image_area > 0:
                    pixel_value = max(1, min(1000, int((percentage / 100.0) * image_area)))
                    min_area_slider.setValue(pixel_value)
                else:
                    min_area_slider.setValue(min(1000, max(1, 50)))
            elif self.app.current_image is None:
                min_area_slider.setValue(min(1000, max(1, min_area_slider.value())))

            self.app.using_pixels_mode = True
            # Update the mutable scale so the slider/spinbox closures convert correctly.
            self.app.sliders["Min Area"]['scale'] = 1.0

            # Reconfigure spinbox for pixel display
            min_area_spinbox.blockSignals(True)
            min_area_spinbox.setDecimals(0)
            min_area_spinbox.setSuffix(" px")
            min_area_spinbox.setMinimum(1)
            min_area_spinbox.setMaximum(1000)
            min_area_spinbox.setSingleStep(1)
            min_area_spinbox.setValue(min_area_slider.value())
            min_area_spinbox.blockSignals(False)
            self.app.image_processor.update_image()

    # Color detection specific
    def add_wall_color(self):
        """Open a color dialog to add a new wall color."""
        color = QColorDialog.getColor(QColor(0, 0, 0), self.app, "Select Wall Color")
        if color.isValid():
            # Use the global threshold value as default for new colors
            default_threshold = 0
            item = self.add_wall_color_to_list(color, default_threshold)
            
            # Select the new color
            self.app.wall_colors_list.setCurrentItem(item)
            self.select_color(item)
            
            # Update detection if image is loaded
            if self.app.current_image is not None:
                self.app.image_processor.update_image()
    
    def select_color(self, item):
        """Handle selection of a color in the list."""
        self.app.selected_color_item = item
        
        # Get color data
        color_data = item.data(Qt.ItemDataRole.UserRole)
        threshold = color_data["threshold"]
        
        # Update the threshold slider to show the selected color's threshold
        self.app.threshold_slider.blockSignals(True)
        self.app.threshold_slider.setValue(int(threshold * 10))
        self.app.threshold_slider.blockSignals(False)
        self.app.threshold_spinbox.blockSignals(True)
        self.app.threshold_spinbox.setValue(threshold)
        self.app.threshold_spinbox.blockSignals(False)

        # Show the threshold container
        self.app.threshold_container.setVisible(True)

    def update_selected_threshold(self, value):
        """Update the threshold for the selected color."""
        if not self.app.selected_color_item:
            return

        threshold = value / 10.0

        # Get the current color data
        color_data = self.app.selected_color_item.data(Qt.ItemDataRole.UserRole)
        color = color_data["color"]
        
        # Update the color data with the new threshold
        self.update_color_list_item(self.app.selected_color_item, color, threshold)
        
        # Update detection immediately for visual feedback
        if self.app.current_image is not None and self.app.color_detection_radio.isChecked():
            self.app.image_processor.update_image()
    

    def edit_wall_color(self, item):
        """Edit an existing color."""
        color_data = item.data(Qt.ItemDataRole.UserRole)
        current_color = color_data["color"]
        current_threshold = color_data["threshold"]
        
        new_color = QColorDialog.getColor(current_color, self.app, "Edit Wall Color")
        if new_color.isValid():
            # Keep the threshold and update the color
            self.update_color_list_item(item, new_color, current_threshold)
            # Update detection if image is loaded
            if self.app.current_image is not None:
                self.app.image_processor.update_image()

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
        self.app.wall_colors_list.addItem(item)
        return item

    def remove_wall_color(self):
        """Remove the selected color from the list."""
        selected_items = self.app.wall_colors_list.selectedItems()
        for item in selected_items:
            self.app.wall_colors_list.takeItem(self.app.wall_colors_list.row(item))
        
        # Hide threshold controls if no colors are selected or all are removed
        if not self.app.wall_colors_list.selectedItems() or self.app.wall_colors_list.count() == 0:
            self.app.threshold_container.setVisible(False)
            self.app.selected_color_item = None
        
        # Update detection if image is loaded and we still have colors
        if self.app.current_image is not None and self.app.wall_colors_list.count() > 0:
            self.app.image_processor.update_image()

    # Light color detection methods
    def add_light_color(self):
        """Open a color dialog to add a new light color."""
        color = QColorDialog.getColor(QColor(255, 255, 255), self.app, "Select Light Color")
        if color.isValid():
            self.add_light_color_to_list(color, 15.0)  # Default threshold for lights
            # Update lights only, don't re-detect contours
            if self.app.current_image is not None:
                self.app.image_processor.update_lights_only()
    
    def select_light_color(self, item):
        """Handle selection of a light color in the list."""
        self.app.selected_light_color_item = item
        
        # Get color data
        color_data = item.data(Qt.ItemDataRole.UserRole)
        threshold = color_data["threshold"]
        
        # Update the threshold slider to show the selected color's threshold
        self.app.light_threshold_slider.blockSignals(True)
        self.app.light_threshold_slider.setValue(int(threshold * 10))
        self.app.light_threshold_slider.blockSignals(False)
        self.app.light_threshold_spinbox.blockSignals(True)
        self.app.light_threshold_spinbox.setValue(threshold)
        self.app.light_threshold_spinbox.blockSignals(False)

        # Show the threshold container
        self.app.light_threshold_container.setVisible(True)

    def update_selected_light_threshold(self, value):
        """Update the threshold for the selected light color."""
        if not self.app.selected_light_color_item:
            return

        threshold = value / 10.0

        # Get the current color data
        color_data = self.app.selected_light_color_item.data(Qt.ItemDataRole.UserRole)
        color = color_data["color"]
        
        # Update the color data with the new threshold
        self.update_light_color_list_item(self.app.selected_light_color_item, color, threshold)
        
        # Update detection immediately for visual feedback
        if self.app.current_image is not None:
            self.app.image_processor.update_lights_only()
    
    def edit_light_color(self, item):
        """Edit an existing light color."""
        color_data = item.data(Qt.ItemDataRole.UserRole)
        current_color = color_data["color"]
        current_threshold = color_data["threshold"]
        
        new_color = QColorDialog.getColor(current_color, self.app, "Edit Light Color")
        if new_color.isValid():
            self.update_light_color_list_item(item, new_color, current_threshold)
            # Update lights only, don't re-detect contours
            if self.app.current_image is not None:
                self.app.image_processor.update_lights_only()

    def update_light_color_list_item(self, item, color, threshold):
        """Update a light color list item with new color and threshold."""
        # Store both color and threshold in the item data
        color_data = {"color": color, "threshold": threshold}
        item.setData(Qt.ItemDataRole.UserRole, color_data)
        
        # Update item text and appearance
        item.setText(f"RGB: {color.red()}, {color.green()}, {color.blue()} (T: {threshold:.1f})")
        item.setBackground(color)
        
        # Set text color based on background brightness
        if color.lightness() < 128:
            item.setForeground(QColor(255, 255, 255))  # White text on dark background
        else:
            item.setForeground(QColor(0, 0, 0))  # Black text on light background

    def add_light_color_to_list(self, color, threshold=15.0):
        """Add a light color with threshold to the light colors list."""
        item = QListWidgetItem()
        self.update_light_color_list_item(item, color, threshold)
        self.app.light_colors_list.addItem(item)
        return item

    def remove_light_color(self):
        """Remove the selected light color from the list."""
        selected_items = self.app.light_colors_list.selectedItems()
        for item in selected_items:
            self.app.light_colors_list.takeItem(self.app.light_colors_list.row(item))
        
        # Hide threshold controls if no colors are selected or all are removed
        if not self.app.light_colors_list.selectedItems() or self.app.light_colors_list.count() == 0:
            self.app.light_threshold_container.setVisible(False)
        
        # Update lights only, don't re-detect contours
        if self.app.current_image is not None:
            self.app.image_processor.update_lights_only()

    # Hatching removal
    def toggle_hatching_removal(self, enabled):
        """Toggle hatching removal options visibility."""
        self.app.hatching_options.setVisible(enabled)
        
        # Update the image if one is loaded
        if self.app.current_image is not None:
            self.app.image_processor.update_image()

    def select_hatching_color(self):
        """Open a color dialog to select hatching color."""
        color = QColorDialog.getColor(self.app.hatching_color, self.app, "Select Hatching Color")
        if color.isValid():
            self.app.hatching_color = color
            # Update button color
            self.app.hatching_color_button.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")
            
            # Update the image if one is loaded and removal is enabled
            if self.app.current_image is not None and self.app.remove_hatching_checkbox.isChecked():
                self.app.image_processor.update_image()

    def update_hatching_threshold(self, value):
        self.app.hatching_threshold = value / 10.0
        if self.app.current_image is not None and self.app.remove_hatching_checkbox.isChecked():
            self.app.image_processor.update_image()

    def update_hatching_width(self, value):
        self.app.hatching_width = value
        if self.app.current_image is not None and self.app.remove_hatching_checkbox.isChecked():
            self.app.image_processor.update_image()

    # Thinning controls
    def update_target_width(self, value):
        self.app.target_width = value

    def update_max_iterations(self, value):
        self.app.max_iterations = value