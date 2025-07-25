import numpy as np
import copy
from src.wall_detection.mask_editor import create_mask_from_contours, blend_image_with_mask



class MaskProcessor:
    def __init__(self, app):
        self.app = app

    def create_empty_mask(self):
        """Create an empty transparent mask layer."""
        if self.app.current_image is None:
            return
            
        # Use original image dimensions for mask if available
        if self.app.original_image is not None:
            height, width = self.app.original_image.shape[:2]
        else:
            height, width = self.app.current_image.shape[:2]
        # Create a transparent mask (4th channel is alpha, all 0 = fully transparent)
        self.app.mask_layer = np.zeros((height, width, 4), dtype=np.uint8)

    def bake_contours_to_mask(self):
        """Bake the current contours to the mask layer."""
        if self.app.current_image is None or not self.app.current_contours:
            return
        
        # Save state before modifying
        self.save_state()
        
        # Use original image dimensions and scale contours if needed
        if self.app.original_image is not None and self.app.scale_factor != 1.0:
            # Scale contours to original resolution
            full_res_contours = self.app.contour_processor.scale_contours_to_original(self.app.current_contours, self.app.scale_factor)
            image_shape = self.app.original_image.shape
        else:
            full_res_contours = self.app.current_contours
            image_shape = self.app.current_image.shape
            
        # Create the mask from contours
        self.app.mask_layer = create_mask_from_contours(
            image_shape, 
            full_res_contours,
            color=(0, 255, 0, 255)  # Green
        )
        
        # Switch to mask editing mode
        self.app.edit_mask_mode_radio.setVisible(True)
        self.app.edit_mask_mode_radio.setChecked(True)
        
        # Enable the Export to UVTT button
        self.app.export_uvtt_button.setEnabled(True)
          # Update display
        self.update_display_with_mask()
        
    def update_display_with_mask(self, region=None):
        """Update the display to show the image with the mask overlay.
        
        Args:
            region: Optional tuple (x, y, width, height) for updating just a specific region
        """
        if self.app.mask_layer is None:
            return
        
        # Use original image for display if available, otherwise use current image
        if self.app.original_image is not None:
            display_base_image = self.app.original_image
        else:
            display_base_image = self.app.current_image
            
        if display_base_image is None:
            return
        
        # Check if we should update just a region
        if region is not None and hasattr(self.app.image_label, 'update_region'):
            # Extract region coordinates
            x, y, width, height = region
            
            # Blend just the region
            region_image = blend_image_with_mask(display_base_image, self.app.mask_layer, region)
            if region_image is not None:
                # Update just the affected region
                self.app.image_label.update_region(region_image, x, y, width, height)
                
                # We still need to update the full image in the background to maintain proper state
                # But we don't need to redisplay it since we already updated the region
                full_display_image = blend_image_with_mask(display_base_image, self.app.mask_layer)
                self.app.last_preview_image = full_display_image.copy()
                self.app.processed_image = full_display_image.copy()
                return
        
        # If no region specified or region update not supported, update the full image
        display_image = blend_image_with_mask(display_base_image, self.app.mask_layer)
        # Store for refresh_display
        self.app.processed_image = display_image.copy()
        # Display the blended image
        self.app.refresh_display()
        
        # Store this as the baseline image for brush preview
        self.app.last_preview_image = display_image.copy()
        
        # Important: Also update the processed_image
        self.app.processed_image = display_image.copy()

    # State management
    def save_state(self):
        """Save the current state to history for undo functionality."""
        if self.app.current_image is None:
            # Don't save state if there's no image loaded
            return
            
        # Save different data depending on the current mode
        if self.app.edit_mask_mode_enabled and self.app.mask_layer is not None:
            state = {
                'mode': 'mask',
                'mask': self.app.mask_layer.copy(),
                'original_image': None if self.app.original_processed_image is None else self.app.original_processed_image.copy()
            }
        else:
            state = {
                'mode': 'contour',
                'contours': copy.deepcopy(self.app.current_contours),
                'original_image': None if self.app.original_processed_image is None else self.app.original_processed_image.copy()
            }
        
        # Add state to history
        self.app.history.append(state)
        
        # Enable the unified undo button once we have history
        if hasattr(self.app, 'undo_button'):
            self.app.undo_button.setEnabled(True)
        
        print(f"State saved to history. History size: {len(self.app.history)}")

    def undo(self):
        """Restore the previous state from history."""
        if not self.app.history:
            print("No history available to undo")
            self.app.setStatusTip("Nothing to undo")
            return
            
        print(f"Undoing action. History size before: {len(self.app.history)}")
        
        # Pop the most recent state (we don't need it anymore)
        self.app.history.pop()
        
        # If no more history, disable undo button
        if not self.app.history:
            # Update the unified undo button state
            if hasattr(self.app, 'undo_button'):
                self.app.undo_button.setEnabled(False)
            self.app.setStatusTip("No more undo history available")
            return
        
        # Get the previous state (now the last item in the queue)
        prev_state = self.app.history[-1]
        
        # Restore based on the mode of the previous state
        if prev_state['mode'] == 'mask':
            self.app.mask_layer = prev_state['mask'].copy()
            
            if prev_state['original_image'] is not None:
                self.app.original_processed_image = prev_state['original_image'].copy()
                self.app.processed_image = self.app.original_processed_image.copy()
            
            # Make sure we're in edit mask mode
            if not self.app.edit_mask_mode_enabled:
                self.app.edit_mask_mode_radio.setChecked(True)
                # This is important - toggle_mode needs to be called explicitly
                self.app.detection_panel.toggle_mode()
            
            # Update the display
            self.update_display_with_mask()
            self.app.setStatusTip("Restored previous mask state")
            print("Restored previous mask state")
            
        else:  # contour mode
            self.app.current_contours = copy.deepcopy(prev_state['contours'])
            
            if prev_state['original_image'] is not None:
                self.app.original_processed_image = prev_state['original_image'].copy()
                self.app.processed_image = self.app.original_processed_image.copy()
            
            # Make sure we're not in mask edit mode
            if self.app.edit_mask_mode_enabled:
                self.app.deletion_mode_radio.setChecked(True)
                # This is important - toggle_mode needs to be called explicitly
                self.app.detection_panel.toggle_mode()
            
            # Update the display
            self.app.contour_processor.update_display_from_contours()
            self.app.setStatusTip("Restored previous contour state")
            print("Restored previous contour state")
