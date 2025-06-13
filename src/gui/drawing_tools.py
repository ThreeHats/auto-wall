from PyQt6.QtGui import QCursor
import cv2
import numpy as np
from src.wall_detection.mask_editor import create_mask_from_contours, blend_image_with_mask, draw_on_mask, export_mask_to_foundry_json, contours_to_foundry_walls, thin_contour

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
        self.temp_drawing = None  # For temporary preview of shapes

    @property
    def mask_layer(self):
        return self.app.mask_layer
    
    @mask_layer.setter
    def mask_layer(self, value):
        self.app.mask_layer = value

    def update_brush_size(self, value):
        """Update the brush size."""
        self.brush_size = value
        self.app.brush_size_value.setText(str(value))
        
        # Update brush preview if in edit mode
        if self.app.edit_mask_mode_enabled:
            # Get current cursor position relative to the image label
            cursor_pos = self.app.image_label.mapFromGlobal(QCursor.pos())
            if self.app.image_label.rect().contains(cursor_pos):
                self.app.update_brush_preview(cursor_pos.x(), cursor_pos.y())

    def update_brush_preview(self, x, y):
        """Show a preview of the brush outline at the current mouse position."""
        if not self.app.edit_mask_mode_enabled or self.app.current_image is None:
            return
        
        # Convert display coordinates to image coordinates
        img_x, img_y = self.app.convert_to_image_coordinates(x, y)
        if img_x is None or img_y is None:
            return
        
        # Make a copy of the blended image with mask
        if self.mask_layer is not None:
            # Create the blended image (original + current mask)
            blended_image = blend_image_with_mask(self.app.current_image, self.mask_layer)
            
            # Draw brush outline with different colors for draw/erase mode
            color = (0, 255, 0) if self.app.draw_radio.isChecked() else (0, 0, 255)  # Green for draw, Red for erase
            cv2.circle(blended_image, (img_x, img_y), self.brush_size, color, 1)
            
            # Store this as our preview image
            self.app.last_preview_image = blended_image.copy()
            self.app.brush_preview_active = True
            
            # Display the image with brush preview
            self.app.image_processor.display_image(blended_image)
        elif self.app.original_processed_image is not None:
            # If no mask exists yet, draw on the original image
            preview_image = self.app.original_processed_image.copy()
            
            # Draw brush outline
            color = (0, 255, 0) if self.app.draw_radio.isChecked() else (0, 0, 255)
            cv2.circle(preview_image, (img_x, img_y), self.brush_size, color, 1)
            
            # Store this as our preview image
            self.app.last_preview_image = preview_image.copy()
            self.app.brush_preview_active = True
            
            # Display the preview
            self.app.image_processor.display_image(preview_image)

    def clear_brush_preview(self):
        """Clear the brush preview when mouse leaves the widget."""
        self.app.brush_preview_active = False
        
        # Restore the original display
        if self.app.edit_mask_mode_enabled and self.mask_layer is not None:
            # Redraw the blend without the brush preview
            self.app.mask_processor.update_display_with_mask()
        elif self.app.original_processed_image is not None:
            # Restore the original image
            self.app.image_processor.display_image(self.app.original_processed_image)
    
    def start_drawing(self, x, y):
        """Start drawing on the mask at the given point."""
        # Check if the app has a mask_layer, if not create it
        if not hasattr(self.app, 'mask_layer') or self.mask_layer is None:
            self.app.mask_processor.create_empty_mask()
        
        # Save state before modifying
        self.app.mask_processor.save_state()
        
        # Convert display coordinates to image coordinates
        img_x, img_y = self.app.convert_to_image_coordinates(x, y)
        if img_x is None or img_y is None:
            return
        
        # Store the drawing mode (draw or erase)
        self.drawing_mode = self.app.draw_radio.isChecked()
        
        # Handle based on the current tool
        if self.current_tool == "brush":
            # Same as original brush tool behavior
            self.app.last_drawing_position = (img_x, img_y)
            
            draw_on_mask(
                self.mask_layer,
                img_x, img_y,
                self.brush_size,
                color=(0, 255, 0, 255),  # Green
                erase=not self.drawing_mode
            )
            
            # Keep the brush preview active while drawing
            self.app.brush_preview_active = True
            
            # Initialize drawing throttling variables
            self.app.drawing_update_counter = 0
            self.app.drawing_update_threshold = 2  # Update display every N points
            
            # Update display with brush preview
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
        img_x2, img_y2 = self.app.convert_to_image_coordinates(x2, y2)
        
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
            
            # Calculate the distance between points
            distance = np.sqrt((img_x2 - img_x1)**2 + (img_y2 - img_y1)**2)
            
            # Determine intermediate points for a smooth line
            if distance < (self.brush_size * 0.5):
                draw_on_mask(
                    self.mask_layer,
                    img_x2, img_y2,
                    self.brush_size,
                    color=(0, 255, 0, 255),
                    erase=not self.drawing_mode
                )
            else:
                # Calculate points for a continuous line
                step_size = self.brush_size / 3
                num_steps = max(int(distance / step_size), 1)
                
                for i in range(num_steps + 1):
                    t = i / num_steps
                    x = int(img_x1 + t * (img_x2 - img_x1))
                    y = int(img_y1 + t * (img_y2 - img_y1))
                    
                    draw_on_mask(
                        self.mask_layer,
                        x, y,
                        self.brush_size,
                        color=(0, 255, 0, 255),
                        erase=not self.drawing_mode
                    )
            
            # Store the current position for next segment
            self.app.last_drawing_position = (img_x2, img_y2)
            
            # Update display with brush preview at current position
            # This shows the brush outline during drawing
            self.update_display_with_brush(img_x2, img_y2)
            
            # Reset drawing throttling counter since we're updating on every move now
            self.app.drawing_update_counter = 0
            
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
                self.update_brush_preview(cursor_pos.x(), cursor_pos.y())

    def update_shape_preview(self, img_x2, img_y2):
        """Update the preview for shape drawing tools."""
        if self.drawing_start_pos is None or self.temp_drawing is None:
            return
        
        # Start with a clean copy of the mask (without the current shape)
        preview_mask = self.temp_drawing.copy()
        
        # Get the start position
        img_x1, img_y1 = self.drawing_start_pos
        
        # Get drawing color based on draw/erase mode
        color = (0, 255, 0, 255) if self.app.draw_radio.isChecked() else (0, 0, 255, 0)  # Green for draw, Red for erase
        
        # Draw the appropriate shape based on the current tool
        if self.current_tool == "line":
            # Draw a line from start to current position
            cv2.line(
                preview_mask,
                (img_x1, img_y1),
                (img_x2, img_y2),
                color,
                thickness=self.brush_size
            )
        elif self.current_tool == "rectangle":
            # Draw a rectangle from start to current position
            cv2.rectangle(
                preview_mask,
                (img_x1, img_y1),
                (img_x2, img_y2),
                color,
                thickness=self.brush_size if self.brush_size > 1 else -1  # Filled if size is 1
            )
        elif self.current_tool == "circle":
            # Calculate radius for circle based on distance
            radius = int(np.sqrt((img_x2 - img_x1)**2 + (img_y2 - img_y1)**2))
            cv2.circle(
                preview_mask,
                (img_x1, img_y1),  # Center at start position
                radius,
                color,
                thickness=self.brush_size if self.brush_size > 1 else -1  # Filled if size is 1
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
                thickness=self.brush_size if self.brush_size > 1 else -1  # Filled if size is 1
            )
        
        # Temporarily update the mask layer with the preview
        self.mask_layer = preview_mask
        
        # Update display
        self.app.mask_processor.update_display_with_mask()

    # drawing
    def finalize_shape(self):
        """Finalize the shape being drawn."""
        if self.mask_layer is not None:
            
            # Reset drawing variables
            self.drawing_start_pos = None
            self.temp_drawing = None

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
        if self.app.current_image is None or self.mask_layer is None:
            return
        
        # Create the blended image (original + current mask)
        blended_image = blend_image_with_mask(self.app.current_image, self.mask_layer)
        
        # Draw brush outline with different colors for draw/erase mode
        color = (0, 255, 0) if self.app.draw_radio.isChecked() else (0, 0, 255)  # Green for draw, Red for erase
        cv2.circle(blended_image, (img_x, img_y), self.brush_size, color, 1)
        
        # Store this as our preview image
        self.last_preview_image = blended_image.copy()
        
        # Display the image with brush preview
        self.app.image_processor.display_image(blended_image)

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

