from PyQt6.QtGui import QCursor
from PyQt6.QtCore import QTimer
import cv2
import numpy as np
import time
from src.wall_detection.mask_editor import create_mask_from_contours, blend_image_with_mask, draw_on_mask, export_mask_to_foundry_json, contours_to_foundry_walls, thin_contour
from src.utils.geometry import convert_to_image_coordinates, point_to_line_distance, line_segments_intersect

class DrawingTools:
    def __init__(self, app):
        """
        Initialize drawing tools with a reference to the main app.
        
        Args:
            app: The main application instance
        """
        self.app = app

        self.brush_size = 10
        self.drawing_mode = True  # True for draw, False for erase
        self.current_tool = "brush"  # Default tool
        self.drawing_start_pos = None  # Starting position for shape tools
        self.temp_drawing = None
        self.last_preview_update = 0
        self.preview_update_interval = 0.1  # 100 ms interval for preview updates
        self.last_display_coords = None
        self.last_converted_coords = None
        self.cached_base_blend = None
        self.cache_valid = False
        
        # Timer for detecting mouse idle
        self.mouse_idle_timer = QTimer()
        self.mouse_idle_timer.setSingleShot(True)
        self.mouse_idle_timer.timeout.connect(self.on_mouse_idle)
        self.idle_detection_interval = 200  # ms

    @property
    def mask_layer(self):
        return self.app.mask_layer
        
    @mask_layer.setter
    def mask_layer(self, value):
        self.app.mask_layer = value
        # Invalidate cache when mask changes
        self.invalidate_cache()
    
    def on_mouse_idle(self):
        """Called when the mouse has been idle (stopped moving) for the specified interval.
        Shows the brush preview at the last known position."""
        # Only show if in edit mode and we're not currently drawing
        if not self.app.edit_mask_mode_enabled:
            return
        
        if (hasattr(self.app, 'last_drawing_position') and 
            self.app.last_drawing_position is not None):
            return
            
        # Use the last known position to show the preview
        if self.last_display_coords is not None:
            x, y = self.last_display_coords
            # Force the brush preview to show, bypassing any throttling
            self.update_brush_preview(x, y, force=True)
            
    def update_brush_size(self, value):
        """Update the brush size."""
        self.brush_size = value
        self.app.brush_size_value.setText(str(value))
        
        # Update brush preview if in edit mode
        if self.app.edit_mask_mode_enabled:
            # Get current cursor position to the image label
            cursor_pos = self.app.image_label.mapFromGlobal(QCursor.pos())
            if self.app.image_label.rect().contains(cursor_pos):                self.update_brush_preview(cursor_pos.x(), cursor_pos.y())
                
    def update_brush_preview(self, x, y, force=False):
        """Ultra-fast brush preview using direct pixmap manipulation - no full image processing."""
        if not self.app.edit_mask_mode_enabled:
            return

        # Check if we have an image to work with
        if self.app.original_image is None and self.app.current_image is None:
            return
        
        # Don't show brush preview if we're currently drawing
        if (hasattr(self.app, 'last_drawing_position') and 
            self.app.last_drawing_position is not None):
            return

        # Reset the idle timer if this isn't a forced update
        if not force:
            self.mouse_idle_timer.start(self.idle_detection_interval)
            
        # Very light throttling - only skip if same position and not forced
        if not force and (self.last_display_coords is not None and 
            abs(x - self.last_display_coords[0]) < 2 and 
            abs(y - self.last_display_coords[1]) < 2):
            return
        
        # Convert display coordinates to image coordinates
        img_x, img_y = convert_to_image_coordinates(self.app, x, y)
        if img_x is None or img_y is None:
            # If conversion failed, clear any existing preview
            if hasattr(self.app, 'brush_preview_active') and self.app.brush_preview_active:
                self.clear_brush_preview()
            return
        
        # Cache the conversion
        self.last_display_coords = (x, y)
        self.last_converted_coords = (img_x, img_y)
        
        # First, clear any existing brush preview to ensure a clean state
        was_active = hasattr(self.app, 'brush_preview_active') and self.app.brush_preview_active
        if was_active:
            self.clear_brush_preview()
            
        # Use the optimized overlay drawing method if available (preferred approach)
        if hasattr(self.app.image_label, 'draw_brush_overlay_on_region'):
            is_erase_mode = not self.app.draw_radio.isChecked()
            self.app.image_label.draw_brush_overlay_on_region(img_x, img_y, self.brush_size, is_erase_mode)
            self.app.brush_preview_active = True
            return
            
        # Fall back to region-based approach if the optimized method isn't available
        # Calculate the region that will be affected
        brush_size = self.brush_size
        region_x = max(0, img_x - brush_size - 2)  # Add 2 pixels padding for the outline
        region_y = max(0, img_y - brush_size - 2)
        region_width = min(brush_size * 2 + 4, self.app.current_image.shape[1] - region_x)
        region_height = min(brush_size * 2 + 4, self.app.current_image.shape[0] - region_y)
        region = (region_x, region_y, region_width, region_height)
        
        if self.mask_layer is not None:
            # Use original image for display if available, otherwise use current image
            if self.app.original_image is not None:
                base_image = self.app.original_image
            else:
                base_image = self.app.current_image
                
            # Create the blended image just for the affected region
            blended_region = blend_image_with_mask(base_image, self.mask_layer, region)
            if blended_region is None:
                return
                
            # Calculate brush position relative to the region
            local_x = img_x - region_x
            local_y = img_y - region_y
            
            # Draw only the brush outline with different colors for draw/erase mode
            # Using a single-pixel outline to minimize any image impact
            if self.app.draw_radio.isChecked():
                color = (0, 255, 0)  # Green for draw
            else:
                color = (0, 0, 255)  # Red for erase
            
            # Draw ONLY the outline (thickness=1), not a filled circle or alpha-blended overlay
            cv2.circle(blended_region, (local_x, local_y), self.brush_size, color, 1)
            
            # Store information about this preview update
            self.app.brush_preview_active = True
            self.app.brush_preview_region = region
            
            # Update just the affected region of the display
            if hasattr(self.app.image_label, 'update_region'):
                self.app.image_label.update_region(blended_region, region_x, region_y, region_width, region_height)
            else:
                # Fall back to full image update if region update not available
                if hasattr(self.app, 'last_preview_image') and self.app.last_preview_image is not None:
                    blended_image = self.app.last_preview_image.copy()
                else:
                    blended_image = blend_image_with_mask(base_image, self.mask_layer)
                    
                cv2.circle(blended_image, (img_x, img_y), self.brush_size, color, 1)
                self.app.last_preview_image = blended_image.copy()
                
                # Update processed image and display with grid overlay
                self.app.processed_image = blended_image.copy()
                self.app.refresh_display()
        
        elif self.app.original_processed_image is not None:
            # If no mask exists yet, draw on the original image (just the region)
            if hasattr(self.app.image_label, 'update_region'):
                # Extract the region from the original image - make a deep copy to avoid modifying original
                region_image = self.app.original_processed_image[
                    region_y:region_y+region_height, 
                    region_x:region_x+region_width
                ].copy()
                
                # Calculate brush position relative to the region
                local_x = img_x - region_x
                local_y = img_y - region_y
                
                # Draw only the brush outline
                if self.app.draw_radio.isChecked():
                    color = (0, 255, 0)  # Green for draw
                else:
                    color = (0, 0, 255)  # Red for erase
                    
                # Draw ONLY the outline (thickness=1), not a filled circle
                cv2.circle(region_image, (local_x, local_y), self.brush_size, color, 1)
                
                # Update just the region
                self.app.brush_preview_active = True
                self.app.brush_preview_region = region
                self.app.image_label.update_region(region_image, region_x, region_y, region_width, region_height)
            else:
                # Fall back to full image update
                preview_image = self.app.original_processed_image.copy()
                color = (0, 255, 0) if self.app.draw_radio.isChecked() else (0, 0, 255)
                cv2.circle(preview_image, (img_x, img_y), self.brush_size, color, 1)
                self.app.last_preview_image = preview_image.copy()
                
                # Update processed image and display with grid overlay
                self.app.processed_image = preview_image.copy()
                self.app.refresh_display()
                
    def clear_brush_preview(self):
        """Clear the brush preview when mouse leaves the widget or drawing starts."""
        if not hasattr(self.app, 'brush_preview_active') or not self.app.brush_preview_active:
            return
            
        self.app.brush_preview_active = False
        
        # Always prefer using the reset_brush_overlay method if available
        # This will restore the original pixmap without any brush overlay
        if hasattr(self.app.image_label, 'reset_brush_overlay'):
            self.app.image_label.reset_brush_overlay()
            return
            
        # Second best option - use fast display refresh
        if hasattr(self.app.image_label, 'update_display'):
            # Just refresh the display from the base pixmap - much faster than full image processing
            self.app.image_label.update_display()
            return
            
        # If we have region information, update only that region
        if hasattr(self.app, 'brush_preview_region') and self.app.brush_preview_region and hasattr(self.app.image_label, 'update_region'):
            region = self.app.brush_preview_region
            x, y, width, height = region
            
            # Use original image for display if available, otherwise use current image
            if self.app.original_image is not None:
                base_image = self.app.original_image
            else:
                base_image = self.app.current_image
                
            if base_image is None:
                return
                
            # Create a clean blend of just the affected region without any brush preview
            if self.mask_layer is not None:
                region_image = blend_image_with_mask(base_image, self.mask_layer, region)
                if region_image is not None:
                    # Update just the affected region with the clean image
                    self.app.image_label.update_region(region_image, x, y, width, height)
                    return
        
        # Last resort - fall back to a full image update
        if self.app.edit_mask_mode_enabled and self.mask_layer is not None:
            # Use cached base blend if available for better performance
            base_blend = self.get_base_blend()
            if base_blend is not None:
                # Update processed image and display with grid overlay
                self.app.processed_image = base_blend.copy()
                self.app.refresh_display()
            else:
                # Fallback to full update
                self.app.mask_processor.update_display_with_mask()
        elif self.app.original_processed_image is not None:
            # Restore the original image
            self.app.processed_image = self.app.original_processed_image.copy()
            self.app.refresh_display()
                
    def invalidate_cache(self):
        """Invalidate the cached blend when mask or image changes."""
        self.cache_valid = False
        self.cached_base_blend = None

    def get_base_blend(self):
        """Get or create the base image + mask blend with caching."""
        if self.cache_valid and self.cached_base_blend is not None:
            return self.cached_base_blend
        
        # Use original image for display if available, otherwise use current image
        if self.app.original_image is not None:
            base_image = self.app.original_image
        else:
            base_image = self.app.current_image
            
        if base_image is None:
            return None
            
        # Create the blended image (original + current mask) and cache it
        if self.mask_layer is not None:
            self.cached_base_blend = blend_image_with_mask(base_image, self.mask_layer)
        else:
            self.cached_base_blend = base_image.copy()
            
        self.cache_valid = True
        return self.cached_base_blend

    def start_drawing(self, x, y):
        """Start drawing on the mask at the given point."""
        # Check if the app has a mask_layer, if not create it
        if not hasattr(self.app, 'mask_layer') or self.mask_layer is None:
            self.app.mask_processor.create_empty_mask()
        
        # Save state before modifying
        self.app.mask_processor.save_state()
        
        # Convert display coordinates to image coordinates
        img_x, img_y = convert_to_image_coordinates(self.app, x, y)
        if img_x is None or img_y is None:
            return
        
        # IMPORTANT: Clear brush preview immediately when drawing starts
        self.app.brush_preview_active = False
        self.clear_brush_preview()
        
        # Store the drawing mode (draw or erase)
        self.drawing_mode = self.app.draw_radio.isChecked()
        
        # Handle based on the current tool
        if self.current_tool == "brush":
            # Same as original brush tool behavior
            self.app.last_drawing_position = (img_x, img_y)
            
            self.mask_layer, affected_region = draw_on_mask(
                self.mask_layer,
                img_x, img_y,
                self.brush_size,
                color=(0, 255, 0, 255),  # Green
                erase=not self.drawing_mode
            )
            
            # Invalidate cache since we modified the mask
            self.invalidate_cache()
            
            # Initialize drawing throttling variables
            self.app.drawing_update_counter = 0
            self.app.drawing_update_threshold = 2  # Update display every N points
            
            # Calculate region for update
            if affected_region:
                x, y, w, h = affected_region
                update_region = (x, y, w, h)
                self.update_display_with_brush_region(img_x, img_y, update_region)
            else:
                # Fall back to full update if no region was returned
                self.update_display_with_brush(img_x, img_y)
                
        elif self.current_tool == "fill":
            # For fill tool, perform flood fill immediately
            self.perform_fill(img_x, img_y)
        else:            
            # For all other shape tools, just record the starting position
            self.drawing_start_pos = (img_x, img_y)
            self.temp_drawing = self.mask_layer.copy()
            
    def continue_drawing(self, x1, y1, x2, y2):
        """Continue drawing on the mask between two points (optimized)."""
        # Convert display coordinates to image coordinates
        img_x2, img_y2 = convert_to_image_coordinates(self.app, x2, y2)
        
        if img_x2 is None or img_y2 is None:
            return
            
        # Handle based on the current tool
        if self.current_tool == "brush":
            # Original brush behavior
            if not hasattr(self.app, 'last_drawing_position') or self.app.last_drawing_position is None:
                self.app.last_drawing_position = (img_x2, img_y2)
                return
                
            img_x1, img_y1 = self.app.last_drawing_position
                
            # Get drawing/erasing mode
            self.drawing_mode = self.app.draw_radio.isChecked()
            
            # Track the affected region for this drawing operation
            min_x = min(img_x1, img_x2) - self.brush_size - 1
            min_y = min(img_y1, img_y2) - self.brush_size - 1
            max_x = max(img_x1, img_x2) + self.brush_size + 1
            max_y = max(img_y1, img_y2) + self.brush_size + 1
            
            # Ensure region is within image bounds
            height, width = self.mask_layer.shape[:2]
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(width, max_x)
            max_y = min(height, max_y)
            
            # Calculate the distance between points
            distance = np.sqrt((img_x2 - img_x1)**2 + (img_y2 - img_y1)**2)
            
            # Determine intermediate points for a smooth line
            affected_regions = []
            
            if distance < (self.brush_size * 0.5):
                self.mask_layer, affected_region = draw_on_mask(
                    self.mask_layer,
                    img_x2, img_y2,
                    self.brush_size,
                    color=(0, 255, 0, 255),
                    erase=not self.drawing_mode
                )
                if affected_region:
                    affected_regions.append(affected_region)
            else:
                # Calculate points for a continuous line
                step_size = self.brush_size / 3
                num_steps = max(int(distance / step_size), 1)
                
                for i in range(num_steps + 1):
                    t = i / num_steps
                    x = int(img_x1 + t * (img_x2 - img_x1))
                    y = int(img_y1 + t * (img_y2 - img_y1))
                    
                    self.mask_layer, step_region = draw_on_mask(
                        self.mask_layer,
                        x, y,
                        self.brush_size,
                        color=(0, 255, 0, 255),
                        erase=not self.drawing_mode
                    )
                    if step_region:
                        affected_regions.append(step_region)
            
            # Invalidate cache since we modified the mask
            self.invalidate_cache()
            
            # Store the current position for next segment
            self.app.last_drawing_position = (img_x2, img_y2)
            
            # Define the region that we need to update - combine all affected regions
            if affected_regions:
                # Merge all affected regions into one bounding box
                all_x_mins = [r[0] for r in affected_regions]
                all_y_mins = [r[1] for r in affected_regions]
                all_x_maxs = [r[0] + r[2] for r in affected_regions]
                all_y_maxs = [r[1] + r[3] for r in affected_regions]
                
                update_x = max(0, min(all_x_mins))
                update_y = max(0, min(all_y_mins))
                update_width = min(width - update_x, max(all_x_maxs) - update_x)
                update_height = min(height - update_y, max(all_y_maxs) - update_y)                
                update_region = (update_x, update_y, update_width, update_height)
            else:
                # Fall back to the calculated region based on line endpoints
                update_region = (min_x, min_y, max_x - min_x, max_y - min_y)
              # Update only the affected region of the display
            self.update_display_with_brush_region(img_x2, img_y2, update_region)
            
        elif self.current_tool != "fill" and self.drawing_start_pos is not None:
            # For shape tools, continuously update the preview
            self.update_shape_preview(img_x2, img_y2)
            
    def end_drawing(self):
        """End drawing on the mask."""
        # For brush tool
        if self.current_tool == "brush":
            # Reset drawing variables
            self.app.last_drawing_position = None
            self.app.drawing_update_counter = 0
            
            # Always update display at the end of drawing
            self.app.mask_processor.update_display_with_mask()
          # For shape tools (except fill which completes immediately)
        elif self.current_tool != "fill" and self.drawing_start_pos is not None:
            # Finalize the shape
            self.finalize_shape()
            
            # Update display
            self.app.mask_processor.update_display_with_mask()
        
        # Restore brush preview after drawing ends
        if self.app.image_label:
            cursor_pos = self.app.image_label.mapFromGlobal(QCursor.pos())
            if self.app.image_label.rect().contains(cursor_pos):
                self.update_brush_preview(cursor_pos.x(), cursor_pos.y(), force=True)
                # Also start the idle timer to ensure the preview stays visible
                self.mouse_idle_timer.start(self.idle_detection_interval)
                
    def update_shape_preview(self, img_x2, img_y2):
        """Update the preview for shape drawing tools with region-based optimization."""
        if self.drawing_start_pos is None or self.temp_drawing is None:
            return
            
        # Track previous preview region to clear it properly
        previous_region = getattr(self, 'last_shape_preview_region', None)
        
        # Start with a clean copy of the mask (without the current shape)
        preview_mask = self.temp_drawing.copy()
        
        # Get the start position
        img_x1, img_y1 = self.drawing_start_pos
        
        # Get drawing color based on draw/erase mode
        color = (0, 255, 0, 255) if self.app.draw_radio.isChecked() else (0, 0, 255, 0)  # Green for draw, Red for erase
        
        # Calculate the region that will be affected by the shape
        # First, get the bounding box of the shape
        region_x = min(img_x1, img_x2) - self.brush_size - 1
        region_y = min(img_y1, img_y2) - self.brush_size - 1
        region_width = abs(img_x2 - img_x1) + self.brush_size * 2 + 2
        region_height = abs(img_y2 - img_y1) + self.brush_size * 2 + 2
        
        # For circle and ellipse, adjust the bounding box
        if self.current_tool == "circle":
            radius = int(np.sqrt((img_x2 - img_x1)**2 + (img_y2 - img_y1)**2))
            region_x = img_x1 - radius - self.brush_size - 1
            region_y = img_y1 - radius - self.brush_size - 1
            region_width = radius * 2 + self.brush_size * 2 + 2
            region_height = radius * 2 + self.brush_size * 2 + 2
        elif self.current_tool == "ellipse":
            width = abs(img_x2 - img_x1)
            height = abs(img_y2 - img_y1)
            center_x = (img_x1 + img_x2) // 2
            center_y = (img_y1 + img_y2) // 2
            region_x = center_x - (width // 2) - self.brush_size - 1
            region_y = center_y - (height // 2) - self.brush_size - 1
            region_width = width + self.brush_size * 2 + 2
            region_height = height + self.brush_size * 2 + 2
            
        # Ensure region is within image bounds
        height, width = preview_mask.shape[:2]
        region_x = max(0, region_x)
        region_y = max(0, region_y)
        region_width = min(width - region_x, region_width)
        region_height = min(height - region_y, region_height)
        
        # Store the region for potential updates
        region = (region_x, region_y, region_width, region_height)
        
        # Draw the appropriate shape based on the current tool
        if self.current_tool == "line":
            # Draw a line from start to current position
            cv2.line(
                preview_mask,
                (img_x1, img_y1),
                (img_x2, img_y2),
                color,
                thickness=self.brush_size * 2
            )
        elif self.current_tool == "rectangle":
            # Draw a rectangle from start to current position
            cv2.rectangle(
                preview_mask,
                (img_x1, img_y1),
                (img_x2, img_y2),
                color,
                thickness=self.brush_size * 2 if self.brush_size > 1 else -1  # Filled if size is 1
            )
        elif self.current_tool == "circle":
            # Calculate radius for circle based on distance
            radius = int(np.sqrt((img_x2 - img_x1)**2 + (img_y2 - img_y1)**2))
            cv2.circle(
                preview_mask,
                (img_x1, img_y1),  # Center at start position
                radius,
                color,
                thickness=self.brush_size * 2 if self.brush_size > 1 else -1  # Filled if size is 1
            )
        elif self.current_tool == "ellipse":
            # Calculate width/height for ellipse
            width = abs(img_x2 - img_x1)
            height = abs(img_y2 - img_y1)
            center_x = (img_x1 + img_x2) // 2
            center_y = (img_y1 + img_y2) // 2
            
            # Ensure width and height are not zero
            width = max(width, 1)
            height = max(height, 1)
            
            cv2.ellipse(
                preview_mask,
                (center_x, center_y),
                (width // 2, height // 2),
                0,  # Angle
                0,  # Start angle
                360,  # End angle
                color,
                thickness=self.brush_size * 2 if self.brush_size > 1 else -1  # Filled if size is 1
            )
              # Save the previous mask to restore it later
        prev_mask = self.mask_layer
        
        # Temporarily update the mask layer with the preview
        self.mask_layer = preview_mask
          # Instead of blending with mask which can cause tinting, use the image_label's
        # ability to draw a clean overlay on top of the original image
        if hasattr(self.app.image_label, 'draw_brush_overlay_on_region'):
            # Calculate the shape bounds
            shape_bounds = [img_x1, img_y1, img_x2, img_y2]
            
            # For circle, we need the center and radius
            if self.current_tool == "circle":
                center_x, center_y = img_x1, img_y1
                radius = int(np.sqrt((img_x2 - img_x1)**2 + (img_y2 - img_y1)**2))
                # Call the special circle preview method (which we'll add)
                self.app.image_label.draw_shape_overlay_circle(
                    center_x, center_y, radius, 
                    self.brush_size * 2, 
                    is_erase_mode=not self.app.draw_radio.isChecked()
                )
                self.last_shape_preview_region = region
                return
            # For ellipse, we need center and radii
            elif self.current_tool == "ellipse":
                width = abs(img_x2 - img_x1)
                height = abs(img_y2 - img_y1)
                center_x = (img_x1 + img_x2) // 2
                center_y = (img_y1 + img_y2) // 2
                # Call the special ellipse preview method (which we'll add)
                self.app.image_label.draw_shape_overlay_ellipse(
                    center_x, center_y, width//2, height//2, 
                    self.brush_size * 2,
                    is_erase_mode=not self.app.draw_radio.isChecked()
                )
                self.last_shape_preview_region = region
                return
            # For rectangle, we need top-left and bottom-right
            elif self.current_tool == "rectangle":
                # Call the special rectangle preview method (which we'll add)
                self.app.image_label.draw_shape_overlay_rectangle(
                    img_x1, img_y1, img_x2, img_y2,
                    self.brush_size * 2,
                    is_erase_mode=not self.app.draw_radio.isChecked()
                )
                self.last_shape_preview_region = region
                return
            # For line, we need start and end points
            elif self.current_tool == "line":
                # Call the special line preview method (which we'll add)
                self.app.image_label.draw_shape_overlay_line(
                    img_x1, img_y1, img_x2, img_y2,
                    self.brush_size * 2,
                    is_erase_mode=not self.app.draw_radio.isChecked()
                )
                self.last_shape_preview_region = region
                return
                
        # Fall back to the original approach if overlay drawing isn't possible
        if hasattr(self.app.image_label, 'update_region'):
            # Get original image
            if self.app.original_image is not None:
                base_image = self.app.original_image
            else:
                base_image = self.app.current_image
                
            if base_image is not None:
                # First, clear previous region if it exists and is different from current region
                if previous_region and previous_region != region:
                    prev_x, prev_y, prev_width, prev_height = previous_region
                    # Create a clean region from the original image without any overlay
                    clean_region = self.temp_drawing.copy()
                    clean_blended = blend_image_with_mask(base_image, clean_region, previous_region)
                    if clean_blended is not None:
                        # Update the previous region to clear it
                        self.app.image_label.update_region(
                            clean_blended,
                            prev_x,
                            prev_y,
                            prev_width,
                            prev_height
                        )
                
                # For the fallback approach, don't blend, just copy the original image for the region
                # and draw the shape directly on it
                region_img = base_image[region_y:region_y+region_height, region_x:region_x+region_width].copy()
                
                # Draw the shape on the region image
                if self.current_tool == "line":
                    cv2.line(
                        region_img,
                        (img_x1-region_x, img_y1-region_y),
                        (img_x2-region_x, img_y2-region_y),
                        (0, 255, 0) if self.app.draw_radio.isChecked() else (255, 0, 0),
                        thickness=self.brush_size * 2
                    )
                elif self.current_tool == "rectangle":
                    cv2.rectangle(
                        region_img,
                        (img_x1-region_x, img_y1-region_y),
                        (img_x2-region_x, img_y2-region_y),
                        (0, 255, 0) if self.app.draw_radio.isChecked() else (255, 0, 0),
                        thickness=self.brush_size * 2 if self.brush_size > 1 else 1
                    )
                # Add more shape handling here
                
                # Update just the affected region of the display
                self.app.image_label.update_region(
                    region_img, 
                    region_x, 
                    region_y, 
                    region_width, 
                    region_height
                )
                # Store the current region for next time
                self.last_shape_preview_region = region
                return
                    
        # Fall back to full update if region update is not possible
        self.app.mask_processor.update_display_with_mask()    # drawing
    def finalize_shape(self):
        """Finalize the shape being drawn."""
        if self.mask_layer is not None:
            # The shape is already drawn in the mask layer by update_shape_preview
            # We just need to invalidate the cache since the mask has changed
            self.invalidate_cache()
            
            # Reset drawing variables
            self.drawing_start_pos = None
            self.temp_drawing = None
            
            # Clear shape preview region tracking
            if hasattr(self, 'last_shape_preview_region'):
                self.last_shape_preview_region = None

    # drawing
    def perform_fill(self, img_x, img_y):
        """Perform flood fill on the mask."""
        if self.mask_layer is None:
            return
        
        # Get alpha channel for fill (this is where we're drawing/erasing)
        alpha_channel = self.mask_layer[:, :, 3].copy()
        
        # Get current value at fill point
        current_value = alpha_channel[img_y, img_x]
        
        # Get target value based on draw/erase mode
        target_value = 255 if self.app.draw_radio.isChecked() else 0
        
        # If current value is already the target, no need to fill
        if current_value == target_value:
            return
        
        # Save state before filling (if not already done)
        # Note: This might be redundant if start_drawing already saved state
        if hasattr(self, 'last_saved_state') and self.last_saved_state != id(self.mask_layer):
            self.save_state()
            self.last_saved_state = id(self.mask_layer)
        
        # Perform flood fill
        mask = np.zeros((alpha_channel.shape[0] + 2, alpha_channel.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(alpha_channel, mask, (img_x, img_y), target_value, 
                      loDiff=0, upDiff=0, flags=8)
        
        # Update the mask's alpha channel
        if self.app.draw_radio.isChecked():
            # For draw mode, set RGB to green where alpha is filled
            self.mask_layer[:, :, 3] = alpha_channel
            filled_area = (alpha_channel == 255)
            self.mask_layer[:, :, 0][filled_area] = 0    # B
            self.mask_layer[:, :, 1][filled_area] = 255  # G
            self.mask_layer[:, :, 2][filled_area] = 0    # R
        else:
            # For erase mode, just set alpha to 0
            self.mask_layer[:, :, 3] = alpha_channel
        
        # Update display
        self.app.mask_processor.update_display_with_mask()

    def update_display_with_brush(self, img_x, img_y):
        """Update the display with both the mask and brush outline at current position."""
        if self.mask_layer is None:
            return
        
        # Use original image for display if available, otherwise use current image
        if self.app.original_image is not None:
            base_image = self.app.original_image
        else:
            base_image = self.app.current_image
            
        if base_image is None:
            return
        
        # Create the blended image (original + current mask)
        blended_image = blend_image_with_mask(base_image, self.mask_layer)
        
        # Draw brush outline with different colors for draw/erase mode
        color = (0, 255, 0) if self.app.draw_radio.isChecked() else (0, 0, 255)  # Green for draw, Red for erase
        cv2.circle(blended_image, (img_x, img_y), color=color, radius=self.brush_size, thickness=1)
        
        # Store this as our preview image
        self.app.last_preview_image = blended_image.copy()
        
        # Display the image with brush preview and grid overlay
        self.app.refresh_display()

    def update_display_with_brush_region(self, img_x, img_y, region):
        """Update only the affected region of the display with both mask and brush outline.
        
        Args:
            img_x, img_y: Current brush position in image coordinates
            region: Tuple (x, y, width, height) defining the region to update
        """
        if self.mask_layer is None:
            return
        
        x, y, width, height = region
        
        # Use original image for display if available, otherwise use current image
        if self.app.original_image is not None:
            base_image = self.app.original_image
        else:
            base_image = self.app.current_image
            
        if base_image is None:
            return
        
        # Create the blended image just for the affected region
        blended_region = blend_image_with_mask(base_image, self.mask_layer, region)
        if blended_region is None:
            return
            
        # Draw brush outline with different colors for draw/erase mode
        color = (0, 255, 0) if self.app.draw_radio.isChecked() else (0, 0, 255)  # Green for draw, Red for erase
        # Calculate brush position relative to the region
        local_x = img_x - x
        local_y = img_y - y
        
        # Only draw the brush outline if it's within the region
        if 0 <= local_x < width and 0 <= local_y < height:
            cv2.circle(blended_region, (local_x, local_y), color=color, radius=self.brush_size, thickness=1)
        
        # Update just the affected region of the display
        if hasattr(self.app.image_label, 'update_region'):
            self.app.image_label.update_region(blended_region, x, y, width, height)
        else:
            # Fall back to full display update if region update not supported
            self.update_display_with_brush(img_x, img_y)

    # drawing
    def update_drawing_tool(self, checked):
        """Update the current drawing tool based on radio button selection."""
        if not checked:
            return
            
        if self.app.brush_tool_radio.isChecked():
            self.current_tool = "brush"
        elif self.app.line_tool_radio.isChecked():
            self.current_tool = "line"
        elif self.app.rectangle_tool_radio.isChecked():
            self.current_tool = "rectangle"
        elif self.app.circle_tool_radio.isChecked():
            self.current_tool = "circle"
        elif self.app.ellipse_tool_radio.isChecked():
            self.current_tool = "ellipse"
        elif self.app.fill_tool_radio.isChecked():
            self.current_tool = "fill"
        
        self.app.setStatusTip(f"Using {self.current_tool} tool")

