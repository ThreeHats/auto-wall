from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QSlider, QWidget, 
    QColorDialog, QListWidget, QListWidgetItem,
    QDialog
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
        slider.valueChanged.connect(self.app.image_processor.update_image)

        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(slider)
        
        # Add the container to the controls layout
        self.app.controls_layout.addWidget(slider_container)

        # Store both the slider, its container, AND the label in the dictionary
        self.app.sliders[label] = {'slider': slider, 'container': slider_container, 'label': slider_label}
        
        # Store scale factor if provided (moved after self.app.sliders assignment)
        if scale_factor:
            self.app.sliders[label]['scale'] = scale_factor

    def update_slider(self, label, label_text, value, scale_factor=None):
        """Update the slider label."""
        # Special case for Min Area slider in pixel mode
        if label_text == "Min Area" and hasattr(self, 'using_pixels_mode') and self.app.using_pixels_mode:
            label.setText(f"{label_text}: {value} px")
        elif scale_factor:
            display_value = value * scale_factor
            label.setText(f"{label_text}: {display_value:.3f}%")
        else:
            label.setText(f"{label_text}: {value}")

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
        
        # Show/hide color selection options
        self.app.color_selection_options.setVisible(self.app.color_selection_mode_enabled)
        
        # Show/hide mask edit options
        self.app.mask_edit_options.setVisible(self.app.edit_mask_mode_enabled)
        
        # Show/hide thinning options
        self.app.thin_options.setVisible(self.app.thin_mode_enabled)
        
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
            # Store original image for highlighting
            if self.app.processed_image is not None:
                self.app.original_processed_image = self.app.processed_image.copy()
        else:
            self.app.setStatusTip("")
            # Clear any highlighting
            self.app.image_label.clear_hover()
            # Display normal image without mask
            if self.app.processed_image is not None:
                self.app.image_processor.display_image(self.app.processed_image)
        
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
        min_area_label = self.app.sliders["Min Area"]['label']
        label_text = "Min Area"

        if self.app.min_area_percentage_radio.isChecked():
            # Switch to percentage mode (0.001% to 25%)
            # Slider range 1 to 25000 represents this
            min_area_slider.setMinimum(1)
            min_area_slider.setMaximum(25000) # Represents 25% with scale 0.001

            # If coming from pixels mode, try to convert value
            if hasattr(self, 'using_pixels_mode') and self.app.using_pixels_mode and self.app.current_image is not None:
                current_pixel_value = min_area_slider.value()
                image_area = self.app.current_image.shape[0] * self.app.current_image.shape[1]
                if image_area > 0:
                    percentage = (current_pixel_value / image_area) * 100.0
                    slider_value = max(1, min(25000, int(percentage / 0.001))) # Convert back to slider scale
                    min_area_slider.setValue(slider_value)

            self.app.using_pixels_mode = False
            self.update_slider(min_area_label, label_text, min_area_slider.value(), 0.001)
            print("Switched Min Area mode to Percentage")

        else: # Pixels mode is checked
            # Switch to pixels mode (1 to 1000 pixels)
            min_area_slider.setMinimum(1)
            min_area_slider.setMaximum(1000)

            # If coming from percentage mode, try to convert value
            if (not hasattr(self, 'using_pixels_mode') or not self.app.using_pixels_mode) and self.app.current_image is not None:
                 current_slider_value = min_area_slider.value()
                 percentage = current_slider_value * 0.001
                 image_area = self.app.current_image.shape[0] * self.app.current_image.shape[1]
                 if image_area > 0:
                     pixel_value = max(1, min(1000, int((percentage / 100.0) * image_area)))
                     min_area_slider.setValue(pixel_value)
                 else:
                      # If no image, set to a reasonable default pixel value if converting
                      min_area_slider.setValue(min(1000, max(1, 50))) # e.g., 50 pixels
            elif self.app.current_image is None:
                 # If no image and already in pixels mode (or first time), ensure value is in range
                 min_area_slider.setValue(min(1000, max(1, min_area_slider.value())))


            self.app.using_pixels_mode = True
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
        self.app.threshold_label.setText(f"Threshold: {threshold:.1f}")
        
        # Show the threshold container
        self.app.threshold_container.setVisible(True)

    def update_selected_threshold(self, value):
        """Update the threshold for the selected color."""
        if not self.app.selected_color_item:
            return
            
        # Calculate the actual threshold value
        threshold = value / 10.0
        self.app.threshold_label.setText(f"Threshold: {threshold:.1f}")
        
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
        """Update the threshold for hatching color matching."""
        # Convert from slider value (0-300) to actual threshold (0-30.0)
        self.app.hatching_threshold = value / 10.0
        self.app.hatching_threshold_value.setText(f"{self.app.hatching_threshold:.1f}")
        
        # Update the image if one is loaded and removal is enabled
        if self.app.current_image is not None and self.app.remove_hatching_checkbox.isChecked():
            self.app.image_processor.update_image()

    def update_hatching_width(self, value):
        """Update the maximum width of lines to remove."""
        self.app.hatching_width = value
        self.app.hatching_width_value.setText(str(value))
        
        # Update the image if one is loaded and removal is enabled
        if self.app.current_image is not None and self.app.remove_hatching_checkbox.isChecked():
            self.app.image_processor.update_image()

    # Thinning controls
    def update_target_width(self, value):
        """Update the target width parameter for thinning."""
        self.app.target_width = value
        self.app.target_width_value.setText(str(value))

    def update_max_iterations(self, value):
        """Update the max iterations parameter for thinning."""
        self.app.max_iterations = value
        self.app.max_iterations_value.setText(str(value))